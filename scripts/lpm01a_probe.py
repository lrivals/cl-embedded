"""
lpm01a_probe.py — Pilote headless X-NUCLEO-LPM01A (PowerShield) → CSV.

Remplace STM32CubeMonitor-Power (indisponible sur ce poste) : configure la sonde
via son interface série ASCII (UM2243), lance une acquisition, décode le flux et
écrit un CSV **directement consommable** par `scripts/energy_capture.py`
(colonnes ``time,current,voltage``, cf. `_load_csv`).

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ :
    Ce script n'écrit que des échantillons réellement renvoyés par la sonde. La
    colonne ``voltage`` reporte la tension d'alimentation **relue** sur la sonde
    (``volt get``), pas une constante supposée. Aucune donnée n'est générée si la
    sonde ne répond pas : le script échoue au lieu de fabriquer une trace.

Limite matérielle mesurée (S5001/S5002) :
    Le flux LPM01A ne transporte **que le courant** — il n'y a pas de voie
    numérique de synchronisation. La colonne ``sync`` attendue par
    `derive_phase_windows` (niveau PA8) **ne peut donc pas venir de la sonde**.
    Deux voies restent ouvertes, toutes deux sans fabrication de données :
      * ``--trigsrc d7`` : l'acquisition **démarre** sur un front de PA8 câblé sur
        le connecteur Arduino D7 (pont à souder, UM2243) → l'origine des temps est
        le début de phase, les fenêtres se déduisent du protocole et de N.
      * protocole **delta** (capture idle seule vs idle+inférence) : l'énergie par
        inférence sort d'une différence de captures réelles (cf. S5002).

Protocole série (relevé sur la sonde, FW 1.0.1) :
    ``htc`` → ``acqmode`` → ``funcmode`` → ``output current`` → ``format`` →
    ``freq`` → ``acqtime`` → ``volt`` → ``start`` … flux … ``summary`` → ``hrc``.

Usage :
    # Diagnostic du banc (autotest + température + version, aucune écriture) :
    python scripts/lpm01a_probe.py --selftest

    # Capture 10 s à 100 kSPS (format binaire obligatoire au-dessus de 10 kHz) :
    python scripts/lpm01a_probe.py --capture --freq 100k --duration 10 \\
        --out captures/ewc_int8.csv

    # Capture démarrée par le marqueur PA8 câblé sur D7 :
    python scripts/lpm01a_probe.py --capture --trigsrc d7 --freq 100k \\
        --duration 1 --out captures/ewc_int8.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import serial
import yaml

# Le PowerShield énumère en CDC « PowerShield (Virtual ComPort in FS Mode) »,
# VID:PID 0483:5740. Le numéro `/dev/ttyACM*` n'est PAS stable : il change au
# rebranchement, et devient ambigu dès que l'ST-LINK de la NUCLEO est connecté en
# parallèle (il expose lui aussi un port CDC). On résout donc le port par son
# identifiant USB stable — envoyer des commandes PowerShield sur l'UART de la
# carte cible n'aurait aucun sens.
PROBE_BY_ID_GLOB = "/dev/serial/by-id/*PowerShield*"
FALLBACK_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 921600

# Au-delà de cette fréquence, le format ASCII décimal ne suit plus (UM2243) :
# le firmware impose le format binaire hexadécimal.
ASCII_MAX_HZ = 10_000

# Durée d'acquisition maximale acceptée par `acqtime` (UM2243) ; au-delà il faut
# demander une durée infinie et arrêter explicitement.
ACQTIME_MAX_S = 10.0

# Marqueurs du flux binaire (relevés sur le banc, FW 1.0.1). Le flux n'est PAS
# un bloc continu d'échantillons : il est découpé en blocs précédés d'un en-tête
# de 9 octets `F0F3 <horodatage 32 bits, ms> <octet d'état> FFFF`, émis toutes les
# ~10 ms (1000 échantillons à 100 kSPS). Ignorer ces en-têtes intermédiaires
# désaligne le décodage et fabrique des valeurs aberrantes — ils sont donc
# reconnus et sautés bloc par bloc (cf. `_iter_bin_blocks`).
BIN_START = b"\xf0\xf3"
BIN_END = b"\xf0\xf4"
BIN_HEADER_LEN = 9

PROMPT = "PowerShield >"

#: Courant moyen en dessous duquel une acquisition n'est PAS une mesure de la carte.
#:
#: Constat de banc (2026-09-07, S5305) : trois exécutions ont profilé une carte **non
#: alimentée** — sortie coupée par un `--power-off` antérieur — et ont écrit des JSON
#: d'apparence parfaitement normale. Rien dans la chaîne ne distinguait alors « la carte
#: consomme peu » de « la carte n'est pas alimentée ».
#:
#: Le seuil est calé sur les mesures : le courant le plus bas jamais relevé sur cette carte
#: est 18,2 mA (repos à 45 MHz, `exp_S53_freq_sweep/45.json`), soit ×18 au-dessus de ce
#: seuil ; une sortie coupée lit ~0. Il ne discrimine donc aucun régime réel de la carte.
MIN_PLAUSIBLE_CURRENT_A = 1e-3


class LPM01AError(RuntimeError):
    """Erreur de dialogue avec la sonde (commande refusée, sonde muette)."""


def find_probe_port() -> str:
    """Résout le port série de la sonde par son identifiant USB stable.

    Returns
    -------
    str
        Chemin du port. Repli sur `FALLBACK_PORT` si aucun lien `by-id` ne
        correspond (montage inhabituel), la connexion échouera alors clairement.

    Raises
    ------
    LPM01AError
        Si plusieurs PowerShield sont connectés — le choix serait arbitraire.
    """
    matches = sorted(glob.glob(PROBE_BY_ID_GLOB))
    if not matches:
        return FALLBACK_PORT
    if len(matches) > 1:
        raise LPM01AError(
            f"Plusieurs PowerShield détectés ({matches}) : préciser --port explicitement."
        )
    return str(Path(matches[0]).resolve())


# --------------------------------------------------------------------------- #
# Décodage des échantillons                                                    #
# --------------------------------------------------------------------------- #
def decode_ascii_sample(token: str) -> float:
    """Décode un échantillon ASCII décimal ``MMMM-EE`` en ampères.

    Format UM2243 : ``6409-07`` ⇔ 6409 × 10⁻⁷ = 640,9 µA. L'exposant peut aussi
    être positif (``+EE``).

    Parameters
    ----------
    token : str
        Jeton brut d'une ligne du flux (sans espaces ni caractères de contrôle).

    Returns
    -------
    float
        Courant en ampères.

    Raises
    ------
    ValueError
        Si le jeton n'a pas la forme mantisse/exposant attendue.
    """
    tok = token.strip().strip("\x00")
    for sep, sign in (("-", -1), ("+", 1)):
        idx = tok.rfind(sep)
        if idx > 0:
            mantissa = int(tok[:idx])
            exponent = int(tok[idx + 1 :])
            return float(mantissa) * 10.0 ** (sign * exponent)
    raise ValueError(f"Échantillon ASCII LPM01A illisible : {token!r}")


def decode_bin_word(word: int) -> float:
    """Décode un mot binaire 16 bits du LPM01A en ampères.

    Format UM2243 : le quartet de poids fort est l'exposant, les 3 quartets bas
    la mantisse — ``0x52A0`` ⇔ (0x2A0) × 16⁻⁵ = 640,9 µA.

    Parameters
    ----------
    word : int
        Mot 16 bits big-endian du flux.

    Returns
    -------
    float
        Courant en ampères.
    """
    exponent = (word >> 12) & 0xF
    mantissa = word & 0x0FFF
    return float(mantissa) * 16.0 ** (-exponent)


def _is_block_header(buf: bytes, pos: int) -> bool:
    """Vrai si un en-tête de bloc valide commence à `pos`.

    Un en-tête fait 9 octets : ``F0F3`` + horodatage 32 bits + octet d'état +
    ``FFFF``. Exiger le motif complet (début ET séparateur final) évite de
    prendre pour un en-tête une paire d'octets ``F0F3`` issue de deux
    échantillons voisins.
    """
    if buf[pos : pos + 2] != BIN_START:
        return False
    return buf[pos + BIN_HEADER_LEN - 2 : pos + BIN_HEADER_LEN] == b"\xff\xff"


def _iter_bin_blocks(buf: bytes):
    """Parcourt les blocs de données du flux binaire, en-têtes retirés.

    Yields
    ------
    bytes
        Charge utile d'un bloc (suite de mots 16 bits big-endian).
    """
    pos = buf.find(BIN_START)
    if pos < 0 or not _is_block_header(buf, pos):
        raise LPM01AError("En-tête de bloc binaire (F0F3 … FFFF) absent du flux.")
    while pos < len(buf):
        pos += BIN_HEADER_LEN
        # Le bloc court jusqu'au prochain en-tête, ou jusqu'au marqueur de fin.
        end = len(buf)
        scan = pos
        while scan < len(buf):
            nxt = buf.find(BIN_START, scan)
            if nxt < 0:
                break
            if _is_block_header(buf, nxt):
                end = nxt
                break
            scan = nxt + 1
        stop = buf.find(BIN_END, pos, end)
        if stop >= 0:
            yield buf[pos:stop]
            return
        yield buf[pos:end]
        if end >= len(buf):
            return
        pos = end


def decode_bin_stream(buf: bytes) -> np.ndarray:
    """Décode un flux binaire complet (avec ses en-têtes de bloc) en courants (A).

    Parameters
    ----------
    buf : bytes
        Flux brut tel que renvoyé par la sonde entre ``start`` et le résumé.

    Returns
    -------
    np.ndarray
        Échantillons de courant (A), dans l'ordre d'acquisition.
    """
    chunks: list[np.ndarray] = []
    for block in _iter_bin_blocks(buf):
        n_words = len(block) // 2
        if n_words == 0:
            continue
        raw = np.frombuffer(block[: n_words * 2], dtype=">u2")
        mantissa = (raw & 0x0FFF).astype(np.float64)
        exponent = (raw >> 12).astype(np.float64)
        chunks.append(mantissa * np.power(16.0, -exponent))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def decode_ascii_stream(text: str) -> np.ndarray:
    """Décode un flux ASCII décimal en tableau de courants (A)."""
    out: list[float] = []
    for line in text.splitlines():
        tok = line.strip().strip("\x00")
        if not tok or tok.startswith(PROMPT) or tok in ("end",):
            continue
        try:
            out.append(decode_ascii_sample(tok))
        except ValueError:
            # Lignes de service (horodatage « TimeStamp: … », résumé) ignorées.
            continue
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Dialogue série                                                               #
# --------------------------------------------------------------------------- #
class PowerShield:
    """Session série avec le X-NUCLEO-LPM01A (mode « controlled by host »)."""

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD) -> None:
        self.port = port or find_probe_port()
        self.ser = serial.Serial(self.port, baud, timeout=1.0)
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def command(self, cmd: str, wait_s: float = 0.4, n_read: int = 1 << 16) -> str:
        """Envoie une commande et renvoie la réponse texte de la sonde.

        Raises
        ------
        LPM01AError
            Si la sonde répond ``error`` (commande inconnue ou argument invalide).
        """
        self.ser.write((cmd + "\n").encode())
        self.ser.flush()
        time.sleep(wait_s)
        resp = self.ser.read(n_read).decode("utf-8", "replace")
        if "error" in resp.lower() and "ack" not in resp.lower():
            raise LPM01AError(f"Commande refusée par la sonde : {cmd!r} → {resp.strip()}")
        return resp

    def _drain(self, quiet_s: float = 0.3, max_s: float = 2.0) -> None:
        """Vide le port jusqu'à `quiet_s` sans le moindre octet reçu.

        À l'ouverture, la sonde émet une bannière et un ou plusieurs prompts, qui
        n'arrivent pas forcément dans les 0,2 s du constructeur : le `reset_input_buffer`
        se fait alors trop tôt, le prompt résiduel est lu comme la réponse à la commande
        suivante, et `htc` est rejeté (« Command unknown »). Mesuré 2026-09-01 : le
        premier `htc` d'une session échoue régulièrement, le second passe toujours.
        """
        fin = time.time() + max_s
        dernier = time.time()
        while time.time() < fin and time.time() - dernier < quiet_s:
            if self.ser.read(1 << 12):
                dernier = time.time()
        self.ser.reset_input_buffer()

    def take_control(self) -> None:
        """Prend la main sur la sonde (`htc`), avec une reprise unique.

        La reprise n'est pas une commodité : sans elle, une campagne complète échoue au
        premier appel sur un aléa de tampon série (constaté sur `run_s53_phase_profile`).
        Un second échec, lui, est une vraie panne et se propage.
        """
        self._drain()
        try:
            self.command("htc")
        except LPM01AError:
            self._drain()
            self.command("htc")

    def release(self) -> None:
        try:
            self.command("hrc", wait_s=0.2)
        finally:
            self.ser.close()

    def voltage_v(self) -> float:
        """Relit la tension d'alimentation appliquée au target (V).

        Réponse nominale : « ack volt get 3300-03 » (mantisse/exposant, UM2243). Le
        dernier jeton n'est PAS fiable : la sonde intercale des lignes de service (fin
        d'auto-calibration déclenchée par `volt`, prompts) et le dernier mot peut alors
        être du texte. Mesuré 2026-09-07 pendant S5305 : le jeton lu valait « handled »,
        `decode_ascii_sample` levait, et la cellule entière sortait en N/A sur ce seul
        aléa série. On cherche donc le dernier jeton AU FORMAT attendu.
        """
        resp = self.command("volt get")
        jetons = re.findall(r"\b\d+[-+]\d+\b", resp)
        if not jetons:
            raise LPM01AError(
                f"Tension illisible dans la réponse de la sonde : {resp.strip()!r}"
            )
        return decode_ascii_sample(jetons[-1])

    def firmware_version(self) -> str:
        resp = self.command("version")
        return resp.strip().split(":")[-1].strip()


def parse_freq_hz(freq: str) -> float:
    """Convertit une consigne de fréquence LPM01A (``100k``, ``500``) en Hz."""
    f = freq.strip().lower()
    if f.endswith("k"):
        return float(f[:-1]) * 1e3
    return float(f)


def capture(
    probe: PowerShield,
    freq: str,
    duration_s: float,
    voltage_mv: int,
    acqmode: str = "dyn",
    funcmode: str = "optim",
    trigsrc: str = "sw",
    min_current_a: float | None = MIN_PLAUSIBLE_CURRENT_A,
) -> tuple[np.ndarray, float, str]:
    """Configure la sonde, lance une acquisition et renvoie les échantillons.

    Toutes les acquisitions du dépôt passent par cette fonction — `measure_current` des
    pilotes de banc, `capture_under_load` du profil par phase, `warmup`. C'est donc ici, et
    nulle part ailleurs, que se fait le contrôle de plausibilité du courant.

    Parameters
    ----------
    probe : PowerShield
        Session série ouverte, contrôle pris (`take_control`).
    freq : str
        Consigne de fréquence d'échantillonnage (``100k``, ``10k``, …).
    duration_s : float
        Durée d'acquisition (s), transmise à ``acqtime``.
    voltage_mv : int
        Tension d'alimentation du target (mV), transmise à ``volt``.
    acqmode, funcmode : str
        Mode d'acquisition (``dyn``/``stat``) et optimisation (``optim``/``high``).
    trigsrc : str
        Source de déclenchement : ``sw`` (immédiat) ou ``d7`` (front externe,
        pont à souder — permet de caler t=0 sur le marqueur PA8).
    min_current_a : float or None
        Courant moyen en dessous duquel l'acquisition est refusée (cf.
        `MIN_PLAUSIBLE_CURRENT_A`). ``None`` lève le contrôle — à réserver à une mesure
        DÉLIBÉRÉMENT sous le seuil (mise en veille profonde) : l'échappatoire doit rester
        explicite, jamais un défaut silencieux.

    Returns
    -------
    tuple[np.ndarray, float, str]
        Échantillons de courant (A), tension relue (V), résumé texte de la sonde.

    Raises
    ------
    LPM01AError
        Si aucun échantillon n'est décodé, ou si le courant moyen est sous
        `min_current_a` — auquel cas l'acquisition ne mesure pas la carte.
    """
    fs_hz = parse_freq_hz(freq)
    fmt = "ascii_dec" if fs_hz <= ASCII_MAX_HZ else "bin_hexa"

    probe.command(f"acqmode {acqmode}")
    probe.command(f"funcmode {funcmode}")
    # État explicite : `pwrend` est rémanent, et un `--power-off` antérieur laisserait
    # la cible coupée après chaque acquisition (donc redémarrée à la suivante).
    probe.command("pwrend on")
    probe.command("output current")
    probe.command(f"format {fmt}")
    probe.command(f"freq {freq}")
    probe.command(f"acqtime {duration_s:g}")
    probe.command(f"trigsrc {trigsrc}")
    # Le réglage de tension déclenche une auto-calibration de la sonde (UM2243) :
    # laisser le temps à la carte de la mener à son terme avant de démarrer.
    probe.command(f"volt {voltage_mv}m", wait_s=1.5)
    voltage_v = probe.voltage_v()

    probe.ser.reset_input_buffer()
    probe.ser.write(b"start\n")
    probe.ser.flush()

    # Lecture jusqu'au résumé de fin (« summary end ») émis par la sonde, avec
    # une marge sur la durée demandée pour couvrir la vidange du tampon USB.
    deadline = time.time() + duration_s + 10.0
    buf = b""
    while time.time() < deadline:
        buf += probe.ser.read(1 << 20)
        if b"summary end" in buf:
            break
    probe.command("stop", wait_s=0.3)

    summary = ""
    i_sum = buf.find(b"summary beg")
    if i_sum >= 0:
        summary = buf[i_sum:].decode("utf-8", "replace")

    if fmt == "bin_hexa":
        samples = decode_bin_stream(buf)
    else:
        samples = decode_ascii_stream(buf.decode("utf-8", "replace"))
    if samples.size == 0:
        raise LPM01AError(
            "Aucun échantillon décodé : vérifier l'alimentation du target et le "
            "câblage de la sonde (aucune trace n'est fabriquée)."
        )
    if min_current_a is not None:
        i_mean = float(np.mean(samples))
        if i_mean < min_current_a:
            raise LPM01AError(
                f"Courant moyen implausible : {i_mean * 1e3:.3f} mA < "
                f"{min_current_a * 1e3:.3f} mA — cette acquisition ne mesure pas la "
                f"carte. Causes usuelles : sortie coupée (« --power-off » antérieur, "
                f"« pwrend off » rémanent), cavalier JP5 hors position, câblage CN14 "
                f"absent. Aucune cellule n'est écrite depuis une acquisition à vide."
            )
    return samples, voltage_v, summary


#: Durée de l'acquisition de préchauffage jetée en début de session (s).
WARMUP_DURATION_S = 10.0


def warmup(probe: PowerShield, voltage_mv: int, duration_s: float = WARMUP_DURATION_S) -> float:
    """Jette une première acquisition : elle est systématiquement biaisée.

    **Constat de banc mesuré (2026-08-04), pas une précaution de principe.**
    Quatre acquisitions statiques enchaînées dans des conditions strictement
    identiques (carte au repos) donnent :

        capture 1 : 62,74 mA      ← biaisée, ~8 mA trop haut
        capture 2 : 54,94 mA
        capture 3 : 54,86 mA
        capture 4 : 54,82 mA      ← régime établi, dispersion ±0,06 mA

    La première acquisition d'une session lit donc ~8 mA de trop, puis la sonde
    se stabilise. Toute comparaison qui place sa référence en première position
    (c'était le cas du protocole delta : fenêtre au repos d'abord) en hérite un
    biais qui dépasse largement l'effet cherché — au point d'inverser le signe du
    résultat. Une session de mesure DOIT donc commencer par ce rebut.

    Parameters
    ----------
    probe : PowerShield
        Session série ouverte, contrôle pris.
    voltage_mv : int
        Tension d'alimentation de la cible (mV).
    duration_s : float
        Durée de l'acquisition jetée (s).

    Returns
    -------
    float
        Courant moyen (A) de l'acquisition jetée — conservé pour la traçabilité
        du biais, jamais à utiliser comme mesure.
    """
    samples, _voltage, _summary = capture(
        probe, freq="1k", duration_s=duration_s, voltage_mv=voltage_mv, acqmode="stat"
    )
    return float(np.mean(samples))


def write_csv(
    out_path: Path,
    samples_a: np.ndarray,
    voltage_v: float,
    fs_hz: float,
    metadata: dict[str, str],
) -> None:
    """Écrit la trace au format attendu par `energy_capture._load_csv`.

    Colonnes : ``time`` (s), ``current`` (A), ``voltage`` (V). La base de temps
    est reconstruite au pas nominal ``1/fs`` — la sonde n'horodate pas chaque
    échantillon ; l'écart éventuel avec la fréquence réelle est reporté dans
    l'en-tête de commentaire, jamais corrigé en silence.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(samples_a.size, dtype=np.float64) / fs_hz
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for key, value in metadata.items():
            fh.write(f"# {key}: {value}\n")
        writer = csv.writer(fh)
        writer.writerow(["time", "current", "voltage"])
        for ti, ii in zip(t, samples_a):
            writer.writerow([f"{ti:.9f}", f"{ii:.9e}", f"{voltage_v:.4f}"])


def _load_calibration(config_path: Path) -> dict:
    """Lit le bloc ``energy_calibration`` du profil matériel (source unique)."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("energy_calibration", {}) or {}


def power_off(probe: PowerShield) -> str:
    """Coupe l'alimentation de la sortie cible.

    Après une acquisition, le firmware laisse la sortie **sous tension** par défaut
    (`pwrend on`) : la sortie reste donc active tant qu'on ne la coupe pas
    explicitement. À utiliser avant toute manipulation de câblage.
    """
    probe.command("pwrend off")
    # `targrst 0` : coupe l'alimentation du target et l'y laisse (UM2243).
    return probe.command("targrst 0", wait_s=0.6).strip()


def power_on(probe: PowerShield, voltage_mv: int, hold_s: float, acqmode: str = "stat") -> str:
    """Maintient la sortie d'alimentation active, pour repérage au multimètre.

    Sert à la mise en service du banc : identifier physiquement quelle broche de
    `CN14` porte VOUT(+) et laquelle porte GND **avant** de câbler la cible, en
    mesurant la tension au multimètre. Une acquisition de durée `hold_s` est
    lancée (la sortie est alimentée pendant toute l'acquisition), puis arrêtée.

    Parameters
    ----------
    probe : PowerShield
        Session série ouverte, contrôle pris.
    voltage_mv : int
        Tension à appliquer sur la sortie (mV).
    hold_s : float
        Durée de maintien (s).

    Returns
    -------
    str
        Tension relue sur la sonde, formatée pour l'affichage.
    """
    # `stat` par défaut : le mode dynamique est plafonné à ~59 mA et une carte
    # F439ZI à 180 MHz le dépasse (« Overcurrent », cf. lpm01a_setup.md §4bis).
    probe.command(f"acqmode {acqmode}")
    probe.command("output current")
    probe.command("format ascii_dec")
    probe.command("freq 1")
    # `acqtime` est plafonné à 10 s par le firmware : au-delà, on passe en durée
    # infinie et c'est l'arrêt explicite qui borne le maintien.
    probe.command("acqtime inf" if hold_s > ACQTIME_MAX_S else f"acqtime {hold_s:g}")
    probe.command(f"volt {voltage_mv}m", wait_s=1.5)
    read_v = probe.voltage_v()
    probe.ser.reset_input_buffer()
    probe.ser.write(b"start\n")
    probe.ser.flush()
    time.sleep(hold_s)
    probe.command("stop", wait_s=0.3)
    return f"{read_v:.3f} V"


def hold_run(probe: PowerShield, voltage_mv: int, command: str, acqmode: str = "stat") -> int:
    """Exécute une commande hôte pendant que la sonde alimente la cible.

    **Nécessaire, pas un confort** (constat banc 2026-08-04) : hors acquisition,
    la sortie du LPM01A ne tient pas la NUCLEO-F439ZI — la carte s'effondre et
    redémarre en boucle (~100 redémarrages en 6 s, mesuré par la bannière de boot
    UART), ce qui rend le SWD inaccessible (`init mode failed`). Pendant une
    acquisition, la carte est stable (0 redémarrage) et le SWD répond. Tout
    `make flash` ou streaming UART doit donc s'exécuter à l'intérieur d'une
    acquisition maintenue.

    Parameters
    ----------
    probe : PowerShield
        Session série ouverte, contrôle pris.
    voltage_mv : int
        Tension d'alimentation de la cible (mV).
    command : str
        Commande shell à exécuter pendant le maintien.
    acqmode : str
        Mode d'acquisition du maintien (`stat` par défaut : le mode dynamique
        est plafonné à ~59 mA, dépassé par la carte).

    Returns
    -------
    int
        Code de retour de la commande.
    """
    probe.command(f"acqmode {acqmode}")
    probe.command("output current")
    probe.command("format ascii_dec")
    probe.command("freq 1")
    probe.command("pwrend on")
    probe.command("acqtime inf")
    probe.command(f"volt {voltage_mv}m", wait_s=1.5)
    probe.ser.reset_input_buffer()
    probe.ser.write(b"start\n")
    probe.ser.flush()
    # Laisse la cible démarrer proprement avant de lancer la commande hôte.
    time.sleep(2.0)
    try:
        completed = subprocess.run(command, shell=True)
        return completed.returncode
    finally:
        probe.command("stop", wait_s=0.5)


def selftest(probe: PowerShield) -> str:
    """Diagnostic du banc : version, identité, autotest, température."""
    lines = [
        probe.command("powershield").strip(),
        f"version: {probe.firmware_version()}",
        probe.command("temp degc").strip(),
        probe.command("autotest status", wait_s=2.0).strip(),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--port",
        default=None,
        help="port série de la sonde (par défaut : détection par identifiant USB stable)",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--selftest", action="store_true", help="diagnostic, aucune écriture")
    parser.add_argument("--calibrate", action="store_true", help="auto-calibration de la sonde")
    parser.add_argument("--capture", action="store_true", help="lance une acquisition")
    parser.add_argument(
        "--power-on",
        type=float,
        metavar="SECONDES",
        default=None,
        help="maintient la sortie sous tension (repérage de VOUT sur CN14 au multimètre)",
    )
    parser.add_argument(
        "--hold-run",
        metavar="COMMANDE",
        default=None,
        help="exécute COMMANDE pendant que la sonde alimente la cible "
        "(indispensable : hors acquisition la carte redémarre en boucle)",
    )
    parser.add_argument(
        "--power-off",
        action="store_true",
        help="coupe l'alimentation de la sortie (avant toute manipulation de câblage)",
    )
    parser.add_argument("--freq", default=None, help="fréquence (100k, 10k, …)")
    parser.add_argument("--duration", type=float, default=10.0, help="durée d'acquisition (s)")
    parser.add_argument("--voltage-mv", type=int, default=None, help="alim target (mV)")
    # Pas de défaut figé : le mode utile dépend de l'opération. `dyn` (profil temporel)
    # pour une capture ; `stat` pour tout MAINTIEN d'alimentation (`--hold-run`,
    # `--power-on`), car le mode dynamique est plafonné à ~59 mA et la NUCLEO-F439ZI à
    # 180 MHz le dépasse : l'acquisition s'arrête sur « Overcurrent », la sortie retombe
    # et la cible redémarre en boucle pendant toute la commande hôte (mesuré 2026-09-01 :
    # 46 bannières de boot en 3 s sous `dyn`, contre 6 au démarrage puis stable sous
    # `stat`). `None` laisse chaque opération choisir, tout en gardant l'option
    # explicite prioritaire.
    parser.add_argument("--acqmode", default=None, choices=("dyn", "stat"))
    parser.add_argument("--funcmode", default="optim", choices=("optim", "high"))
    parser.add_argument("--trigsrc", default="sw", choices=("sw", "d7"))
    parser.add_argument("--out", type=Path, default=None, help="CSV de sortie")
    parser.add_argument(
        "--hw-profile",
        type=Path,
        default=Path("configs/hw_profile_f439zi.yaml"),
        help="profil matériel fournissant fréquence et tension par défaut",
    )
    args = parser.parse_args(argv)

    # `dyn` pour une capture, `stat` pour un maintien — cf. `--acqmode`.
    acqmode_capture = args.acqmode or "dyn"
    acqmode_hold = args.acqmode or "stat"

    calib = _load_calibration(args.hw_profile)
    freq = args.freq or f"{int(calib.get('sampling_rate_hz', 100000)) // 1000}k"
    voltage_mv = args.voltage_mv or int(float(calib.get("supply_voltage_v", 3.3)) * 1000)

    probe = PowerShield(args.port, args.baud)
    try:
        probe.take_control()
        if args.selftest:
            print(selftest(probe))
        if args.calibrate:
            print(probe.command("calib", wait_s=5.0).strip())
        if args.hold_run:
            rc = hold_run(probe, voltage_mv, args.hold_run, acqmode_hold)
            if rc != 0:
                print(f"[lpm01a] commande terminée avec le code {rc}", file=sys.stderr)
                return rc
        if args.power_off:
            print(f"[lpm01a] sortie coupée — {power_off(probe)}")
        if args.power_on is not None:
            print(
                f"[lpm01a] sortie maintenue {args.power_on:g} s à "
                f"{power_on(probe, voltage_mv, args.power_on, acqmode_hold)} — "
                "cible alimentée pendant toute la durée (flash/stream possibles)."
            )
        if args.capture:
            if args.out is None:
                parser.error("--capture exige --out <fichier.csv>")
            samples, voltage_v, summary = capture(
                probe,
                freq=freq,
                duration_s=args.duration,
                voltage_mv=voltage_mv,
                acqmode=acqmode_capture,
                funcmode=args.funcmode,
                trigsrc=args.trigsrc,
            )
            fs_hz = parse_freq_hz(freq)
            # En mode statique la sonde ne renvoie qu'UNE valeur moyennée sur la
            # fenêtre : le nombre d'échantillons attendu n'est pas fs × durée.
            n_expected = 1 if acqmode_capture == "stat" else int(fs_hz * args.duration)
            write_csv(
                args.out,
                samples,
                voltage_v,
                fs_hz,
                metadata={
                    "source": "X-NUCLEO-LPM01A (lpm01a_probe.py)",
                    "firmware": probe.firmware_version(),
                    "sampling_rate_hz": f"{fs_hz:g}",
                    "acqmode": acqmode_capture,
                    "funcmode": args.funcmode,
                    "trigsrc": args.trigsrc,
                    "n_samples_decoded": str(samples.size),
                    "n_samples_expected": str(n_expected),
                    "duration_s": f"{args.duration:g}",
                },
            )
            print(summary.strip())
            print(
                f"[lpm01a] {samples.size} échantillons décodés "
                f"(attendus {n_expected}) · "
                f"I_max={samples.max() * 1e6:.3f} µA · V={voltage_v:.3f} V → {args.out}"
            )
    finally:
        probe.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
