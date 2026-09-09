"""
energy_capture.py — Pilote PowerShield X-NUCLEO-LPM01A / STM32CubeMonitor-Power (S3305).

Acquiert (ou importe) une trace courant/tension du LPM01A, la segmente selon les
marqueurs de phase GPIO du firmware (PA8, S3304 — `EnergyPhase` :
startup/acquisition/inference/idle), intègre l'énergie en **µJ par phase**
(E = Σ I·V·dt) et exporte un JSON normalisé dans `experiments/exp_S33_energy/`.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ :
    Tant que la board + le LPM01A n'ont pas réellement tourné (pas de CSV `--csv`
    fourni), les champs énergie du JSON portent la valeur littérale ``"à mesurer"``
    (constante `A_MESURER`). Ce script NE FABRIQUE jamais de courant/µJ.
    La fréquence d'échantillonnage et la calibration LPM01A restent `TODO(dorra)`
    tant qu'elles ne sont pas confirmées.

Patterns réutilisés :
    - CLI / pilotage série : scripts/sensor_stream.py
    - Structure d'enregistrement d'expérience : scripts/board_experiment_recorder.py

Usage :
    # Un couple modèle × encodage (placeholder tant que pas de CSV LPM01A) :
    python scripts/energy_capture.py --model ewc --encoding fp32 \\
        --duration 10 --output experiments/exp_S33_energy/ewc_fp32.json

    # Avec une trace LPM01A réelle exportée par STM32CubeMonitor-Power :
    python scripts/energy_capture.py --model ewc --encoding fp32 \\
        --csv captures/ewc_fp32.csv --output experiments/exp_S33_energy/ewc_fp32.json

    # Campagne complète (4 modèles × {fp32,int8} + summary.json) :
    python scripts/energy_capture.py --campaign \\
        --output-dir experiments/exp_S33_energy/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Valeur littérale écrite tant qu'aucune mesure réelle n'est disponible (règle
# « aucun chiffre inventé »). Tout consommateur (notebook, autonomy.py) doit la
# détecter avant de calculer.
A_MESURER = "à mesurer"

# Phases énergie — DOIT correspondre à l'enum firmware EnergyPhase (S3304,
# firmware/stm32f4_blink/inc/profiling.h).
PHASES = ("startup", "acquisition", "inference", "idle")

# Couples modèle × encodage de la campagne S3306.
CAMPAIGN_MODELS = ("ewc", "hdc", "tinyol", "maha")
CAMPAIGN_ENCODINGS = ("fp32", "int8")


def capture_session(
    duration_s: float,
    sampling_rate_hz: float,
    output_csv: Path,
    csv_in: Path | None = None,
) -> Path | None:
    """Acquiert une session LPM01A, ou importe un CSV existant.

    Le pilotage headless de STM32CubeMonitor-Power n'étant pas garanti
    disponible sur ce poste, l'acquisition « live » n'est PAS simulée : si aucun
    CSV n'est fourni, la fonction renvoie ``None`` (aucune donnée réelle), ce qui
    déclenche le mode placeholder en aval. Si `csv_in` est fourni, il est validé
    et son chemin renvoyé pour la segmentation.

    Parameters
    ----------
    duration_s : float
        Durée nominale de la capture (s) — métadonnée, reportée dans le JSON.
    sampling_rate_hz : float
        Fréquence d'échantillonnage LPM01A (Hz). `TODO(dorra)` si non calibrée.
    output_csv : Path
        Chemin où une capture live serait écrite (non utilisé en mode import).
    csv_in : Path | None
        Trace LPM01A déjà exportée (colonnes temps, courant, [tension]).

    Returns
    -------
    Path | None
        Le chemin du CSV exploitable, ou ``None`` si aucune donnée réelle.
    """
    if csv_in is not None:
        csv_in = Path(csv_in)
        if not csv_in.is_file():
            raise FileNotFoundError(f"CSV LPM01A introuvable : {csv_in}")
        return csv_in
    # Pas de CSV : aucune fabrication de mesure (règle « aucun chiffre inventé »).
    _ = (duration_s, sampling_rate_hz, output_csv)
    return None


def _load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Charge une trace LPM01A (colonnes : time_s, current_a, [voltage_v]).

    Le format CSV exact de STM32CubeMonitor-Power n'étant pas figé (`TODO(dorra)`),
    on lit de façon tolérante : on cherche des colonnes nommées et, à défaut,
    on retombe sur l'ordre [temps, courant, tension].
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"CSV vide : {csv_path}")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(not _is_float(c) for c in header)
    data_rows = rows[1:] if has_header else rows

    def _col(*names: str, default: int | None = None) -> int | None:
        for n in names:
            if n in header:
                return header.index(n)
        return default

    i_time = _col("time", "time_s", "timestamp", "t", default=0)
    i_cur = _col("current", "current_a", "i", "i_a", default=1)
    i_volt = _col("voltage", "voltage_v", "v", "u", default=None)
    # Colonne numérique de synchronisation = signal GPIO PA8 du firmware (S3304),
    # capté par le LPM01A en parallèle du courant. Ses fronts délimitent les phases
    # (cf. derive_phase_windows). Absente sur un export courant seul → None.
    i_sync = _col("sync", "pa8", "gpio", "digital", "marker", default=None)

    arr = np.array([[float(x) for x in r] for r in data_rows], dtype=np.float64)
    out = {
        "time_s": arr[:, i_time],
        "current_a": arr[:, i_cur],
    }
    out["voltage_v"] = arr[:, i_volt] if i_volt is not None else None
    out["sync"] = arr[:, i_sync] if i_sync is not None else None
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def integrate_energy_uj(
    courant_a: np.ndarray, tension_v: np.ndarray | float, dt_s: float | np.ndarray
) -> float:
    """Énergie en µJ : E = Σ(I × V × dt) × 1e6.

    Parameters
    ----------
    courant_a : np.ndarray
        Échantillons de courant (A).
    tension_v : np.ndarray | float
        Tension (V), scalaire ou par échantillon.
    dt_s : float | np.ndarray
        Pas de temps (s), scalaire (échantillonnage uniforme) ou par intervalle.

    Returns
    -------
    float
        Énergie en microjoules (µJ).
    """
    i = np.asarray(courant_a, dtype=np.float64)
    v = np.asarray(tension_v, dtype=np.float64)
    power_w = i * v  # W = A × V, échantillon par échantillon
    energy_j = float(np.sum(power_w * dt_s))
    return energy_j * 1e6


def energy_uj_per_inference_delta(
    i_idle_a: float,
    i_active_a: float,
    voltage_v: float,
    window_s: float,
    n_inference: int,
) -> float | None:
    """Énergie par inférence par **protocole delta** (µJ), sans profil temporel.

    Voie retenue quand la sonde ne peut pas fournir de profil temporel exploitable
    (cf. `docs/context/lpm01a_setup.md`) : deux mesures de **courant moyen** sur
    des fenêtres de même durée — l'une au repos, l'autre pendant N inférences —
    et l'énergie marginale d'une inférence en est la différence :

        µJ/inférence = (I_actif − I_repos) × V × T / N × 1e6

    Les deux courants sont des mesures réelles ; rien n'est extrapolé.

    Parameters
    ----------
    i_idle_a, i_active_a : float
        Courants moyens mesurés (A) sur la fenêtre au repos et sur la fenêtre
        active.
    voltage_v : float
        Tension d'alimentation relue sur la sonde (V).
    window_s : float
        Durée de chaque fenêtre (s) — identique pour les deux mesures.
    n_inference : int
        Nombre d'inférences exécutées dans la fenêtre active.

    Returns
    -------
    float | None
        µJ par inférence, ou ``None`` si la mesure n'est pas concluante — delta
        de courant nul ou négatif (charge active non distinguable du repos), ou
        paramètres invalides. Le ``None`` doit être reporté « à mesurer », jamais
        remplacé par 0 (règle « aucun chiffre inventé »).
    """
    if n_inference <= 0 or window_s <= 0 or voltage_v <= 0:
        return None
    delta_a = float(i_active_a) - float(i_idle_a)
    if delta_a <= 0.0:
        return None
    energy_j = delta_a * float(voltage_v) * float(window_s)
    return energy_j * 1e6 / float(n_inference)


def segment_by_phase(csv_path: Path, phase_timestamps: list[tuple[str, float, float]]) -> dict:
    """Découpe la trace I/V selon les fenêtres [t_start, t_end] par `EnergyPhase`.

    Les fenêtres proviennent des marqueurs GPIO PA8 (S3304) horodatés DWT,
    convertis en secondes et alignés sur l'axe temps de la trace LPM01A.

    Parameters
    ----------
    csv_path : Path
        Trace LPM01A (cf. `_load_csv`).
    phase_timestamps : list[tuple[str, float, float]]
        Liste de (nom_phase, t_start_s, t_end_s) — nom_phase ∈ `PHASES`.

    Returns
    -------
    dict
        ``{phase: uj}`` pour chaque phase de `PHASES` (0.0 si fenêtre vide).
    """
    trace = _load_csv(csv_path)
    t = trace["time_s"]
    i = trace["current_a"]
    v = trace["voltage_v"]
    if v is None:
        raise ValueError(
            "Trace sans colonne tension : fournir la tension d'alim (V) dans le CSV "
            "ou utiliser un profil HW. La fabrication d'une tension est interdite."
        )

    phases_uj: dict[str, float] = {p: 0.0 for p in PHASES}
    for name, t0, t1 in phase_timestamps:
        if name not in phases_uj:
            raise ValueError(f"Phase inconnue : {name!r} (attendu {PHASES}).")
        mask = (t >= t0) & (t < t1)
        if not np.any(mask):
            continue
        seg_t = t[mask]
        dt = np.gradient(seg_t) if seg_t.size > 1 else np.array([0.0])
        phases_uj[name] += integrate_energy_uj(i[mask], v[mask], dt)
    return phases_uj


def derive_phase_windows(
    trace: dict[str, np.ndarray], threshold: float = 0.5
) -> list[tuple[str, float, float]]:
    """Déduit les fenêtres de phase des fronts du signal de sync PA8 (S3304).

    Le firmware met PA8 au niveau HAUT pendant les phases actives
    (`startup`/`acquisition`/`inference`, `ENERGY_MARKER_SET`) et au niveau BAS en
    attente (`idle`, `ENERGY_MARKER_CLEAR`). Avec le schéma **1-bit** actuel ces
    trois phases actives partagent le même niveau haut : on ne peut donc pas les
    distinguer depuis la seule trace. On reporte honnêtement chaque plateau HAUT
    comme `"inference"` (phase active mesurée) et chaque plateau BAS comme
    `"idle"` — aucune phase fabriquée. Une granularité 4-phases exigerait un
    encodage multi-bit côté firmware (évolution future, hors périmètre S33).

    Parameters
    ----------
    trace : dict[str, np.ndarray]
        Sortie de `_load_csv` ; doit contenir une colonne `"sync"` non ``None``.
    threshold : float
        Seuil de binarisation du niveau logique (V ou unité brute du LPM01A).

    Returns
    -------
    list[tuple[str, float, float]]
        Fenêtres ``(nom_phase, t_start_s, t_end_s)`` consommables par
        `segment_by_phase` ; ``[]`` si le signal est constant (aucune transition).

    Raises
    ------
    ValueError
        Si la trace ne contient pas de colonne de synchronisation.
    """
    sync = trace.get("sync")
    if sync is None:
        raise ValueError(
            "Trace sans colonne de synchronisation (PA8) : impossible de déduire "
            "les fenêtres de phase. Fournir la colonne sync/pa8/gpio du LPM01A."
        )
    t = trace["time_s"]
    level = np.asarray(sync, dtype=np.float64) >= threshold
    windows: list[tuple[str, float, float]] = []
    # Parcours des plateaux de niveau constant : chaque segment [i, j) homogène
    # devient une fenêtre, bornée par le timestamp d'entrée et celui du front.
    start = 0
    for k in range(1, level.size):
        if level[k] != level[start]:
            name = "inference" if level[start] else "idle"
            windows.append((name, float(t[start]), float(t[k])))
            start = k
    if level.size:
        name = "inference" if level[start] else "idle"
        windows.append((name, float(t[start]), float(t[-1])))
    return windows


# --------------------------------------------------------------------------- #
# S5305 — Segmentation par phase SANS voie numérique (« voie A »)              #
# --------------------------------------------------------------------------- #
# `derive_phase_windows` ci-dessus reste la voie PA8 et n'est PAS modifiée. Ce qui
# suit est une seconde voie, ajoutée à côté : quand la sonde tourne en `acqmode dyn`
# à 100 kSPS (possible seulement sous le plafond de 59 mA, cf. S5303), les bursts
# d'inférence sont des créneaux nets dans la trace de courant elle-même. Le signal
# EST la synchro — aucune soudure PA8→D7 n'est alors nécessaire.

#: Nombre d'écarts absolus médians au-dessus de la médiane pour séparer l'actif du
#: repos. Le seuil n'est JAMAIS saisi à la main (critère d'acceptation S5305) : il se
#: déduit de la trace. 5 σ robustes — assez haut pour ignorer le bruit d'acquisition,
#: assez bas pour attraper un burst dont le surcoût est de quelques mA.
MAD_K = 5.0

#: Passage de l'écart absolu médian à un équivalent écart-type pour une loi normale.
#: Constante mathématique (1/Φ⁻¹(3/4)), pas un réglage de banc.
MAD_TO_SIGMA = 1.4826

#: Écart relatif toléré entre le nombre de créneaux DÉTECTÉS et le nombre ATTENDU
#: (`cadence × durée`) avant de refuser la segmentation (spec S5305). Au-delà, ce que
#: le seuil a découpé n'est pas la suite des inférences : publier une énergie par
#: inférence à partir d'un tel découpage reviendrait à diviser par un dénominateur faux.
BURST_TOLERANCE = 0.05

#: Libellé du troisième estimateur de `energy_uj_per_inference`. Il DOIT rester distinct
#: de celui du protocole delta (S5302) et de `rate_regression.METHOD` (« régression
#: I(rate) », S5304) : la comparaison des trois voies est elle-même un résultat de
#: méthodologie (S5309), et une fusion dans un même champ la détruirait.
METHOD_INTEGRATION = "intégration du profil temporel"

#: Raison de N/A des deux phases que le schéma de marquage ne sait pas isoler. Vraie
#: pour la voie PA8 (marqueur 1 bit, `profiling.c`) comme pour la voie A (un créneau de
#: courant sépare l'actif de l'inactif, pas les trois sous-phases actives).
NA_PHASES_1BIT = (
    "marqueur 1 bit : startup/acquisition/inference partagent le niveau actif — une "
    "granularité 4 phases exigerait un encodage multi-bit (PA8 + PA9), amorcé nulle "
    "part dans le firmware."
)


def robust_current_threshold(current_a: np.ndarray, k: float = MAD_K) -> float:
    """Seuil actif/inactif déduit de la trace : ``médiane + k · MAD · 1,4826``.

    Médiane et MAD plutôt que moyenne et écart-type : à cadence basse, les bursts sont
    minoritaires en nombre d'échantillons mais extrêmes en valeur ; une moyenne et un
    écart-type classiques seraient tirés par ce qu'on cherche justement à isoler.

    Cas dégénéré — MAD nulle : le repos est alors plat au pas de quantification près
    (moins de la moitié des échantillons s'écartent de la médiane), et `k · MAD` vaudrait
    zéro, c'est-à-dire un seuil AU niveau du repos qui déclarerait toute la trace active.
    On retombe alors sur le milieu entre le repos et le maximum relevé — toujours déduit
    de la trace, jamais saisi. Une trace strictement constante rend un seuil égal à son
    propre niveau, et la binarisation stricte n'y voit aucun créneau : c'est le bon
    résultat, il n'y a pas d'inférence dans une trace plate.

    Parameters
    ----------
    current_a : np.ndarray
        Échantillons de courant (A).
    k : float
        Nombre d'écarts robustes au-dessus de la médiane.

    Returns
    -------
    float
        Seuil en ampères.
    """
    i = np.asarray(current_a, dtype=np.float64)
    median = float(np.median(i))
    mad = float(np.median(np.abs(i - median)))
    if mad > 0.0:
        return median + k * mad * MAD_TO_SIGMA
    return median + 0.5 * (float(np.max(i)) - median)


def derive_phase_windows_from_current(
    current_a: np.ndarray, dt_s: float, threshold: float | None = None
) -> list[tuple[str, float, float]]:
    """Déduit les fenêtres de phase des créneaux de la trace de COURANT (voie A).

    Même reddition honnête que `derive_phase_windows` : chaque plateau au-dessus du
    seuil est reporté ``"inference"`` (phase active mesurée), chaque plateau en dessous
    ``"idle"``. La voie A a la même limite que le marqueur 1 bit, formulée autrement —
    elle sépare l'actif de l'inactif, **pas** les trois sous-phases actives.

    Parameters
    ----------
    current_a : np.ndarray
        Échantillons de courant (A), échantillonnage uniforme.
    dt_s : float
        Pas de temps entre deux échantillons (s), soit ``1 / fs``.
    threshold : float | None
        Seuil de binarisation (A). ``None`` → `robust_current_threshold` (recommandé :
        un seuil saisi à la main n'est pas traçable à la trace).

    Returns
    -------
    list[tuple[str, float, float]]
        Fenêtres ``(nom_phase, t_start_s, t_end_s)``, mêmes conventions que
        `derive_phase_windows` ; ``[]`` si la trace est vide.
    """
    i = np.asarray(current_a, dtype=np.float64)
    if i.size == 0:
        return []
    seuil = robust_current_threshold(i) if threshold is None else float(threshold)
    # Comparaison STRICTE : sur une trace plate, le seuil vaut le niveau de repos
    # lui-même (cf. `robust_current_threshold`), et un `>=` y verrait un créneau unique
    # couvrant toute l'acquisition — une inférence fabriquée.
    level = i > seuil
    t = np.arange(i.size, dtype=np.float64) * float(dt_s)

    windows: list[tuple[str, float, float]] = []
    start = 0
    for k in range(1, level.size):
        if level[k] != level[start]:
            windows.append(
                ("inference" if level[start] else "idle", float(t[start]), float(t[k]))
            )
            start = k
    windows.append(
        ("inference" if level[start] else "idle", float(t[start]), float(t[-1]))
    )
    return windows


def count_bursts(windows: list[tuple[str, float, float]]) -> int:
    """Nombre de créneaux actifs dans une liste de fenêtres."""
    return sum(1 for name, _, _ in windows if name == "inference")


def validate_burst_count(
    n_detected: int,
    rate_hz: float,
    duration_s: float,
    tolerance: float = BURST_TOLERANCE,
) -> tuple[bool, float | None, str | None]:
    """La segmentation décrit-elle bien la suite des inférences ?

    À cadence imposée, le nombre d'inférences dans la fenêtre vaut ``cadence × durée``
    **par construction** (c'est `sensor_stream.py --rate-hz` qui l'impose, cf. S5008).
    C'est donc un attendu mesuré, pas une estimation : si le découpage s'en écarte de
    plus de `tolerance`, ce qui a été découpé n'est pas la suite des inférences et la
    segmentation est refusée.

    Returns
    -------
    tuple[bool, float | None, str | None]
        ``(accepté, écart_relatif_pct, raison_du_refus)``. La raison est chiffrée et
        part telle quelle dans le champ ``phases_na_reason`` du JSON.
    """
    expected = float(rate_hz) * float(duration_s)
    if expected <= 0:
        return False, None, (
            f"nombre de créneaux attendu non calculable (cadence={rate_hz:g} Hz, "
            f"durée={duration_s:g} s) : sans cadence imposée, le dénominateur d'une "
            f"énergie par inférence n'existe pas."
        )
    ecart = (n_detected - expected) / expected
    if abs(ecart) > tolerance:
        return False, ecart * 100.0, (
            f"segmentation refusée : {n_detected} créneaux détectés pour "
            f"{expected:.0f} attendus ({ecart * 100:+.1f} %, tolérance "
            f"±{tolerance * 100:.0f} %) — le seuil ne découpe pas la suite des "
            f"inférences, diviser par ce dénominateur fabriquerait une énergie."
        )
    return True, ecart * 100.0, None


def marginal_uj_per_burst(
    inference_uj: float,
    idle_uj: float,
    active_duration_s: float,
    idle_duration_s: float,
    n_bursts: int,
) -> float:
    """Surcoût énergétique d'UN créneau, repos déduit (µJ).

    C'est la grandeur comparable au protocole delta (S5302) et à la régression de
    cadence (S5304) : toutes trois mesurent un **surcoût**, pas une consommation
    absolue. La puissance de repos est celle des plateaux bas de LA MÊME trace — jamais
    une référence importée d'une autre acquisition, dont le Sprint 50 a montré qu'elle
    n'est pas comparable (artefact d'ordre, S5301).

    Les deux voies (courant et PA8) passent par ici : sans quoi l'une publierait un
    surcoût et l'autre une consommation brute sous le même nom de champ.
    """
    if n_bursts <= 0:
        raise ValueError("aucun créneau : le surcoût par inférence n'a pas de dénominateur.")
    idle_power_w = (idle_uj / 1e6) / idle_duration_s if idle_duration_s > 0 else 0.0
    return (inference_uj - idle_power_w * active_duration_s * 1e6) / n_bursts


def profile_from_current(
    trace: dict[str, np.ndarray],
    rate_hz: float,
    threshold: float | None = None,
    tolerance: float = BURST_TOLERANCE,
) -> dict:
    """Profil énergétique par phase à partir de la seule trace de courant (voie A).

    Enchaîne seuil robuste → créneaux → contrôle du dénominateur → intégration
    (`integrate_energy_uj`). Rend le bloc de grandeurs du JSON S5305.

    Deux énergies par inférence sont rendues, et elles ne se confondent pas :

        - ``energy_uj_per_inference`` — **marginale** : l'énergie du créneau moins la
          puissance de repos intégrée sur sa durée. C'est la grandeur comparable au
          protocole delta (S5302) et à la régression de cadence (S5304), qui mesurent
          toutes deux un surcoût.
        - ``energy_uj_per_burst_gross`` — **brute** : tout ce qui est consommé pendant
          le créneau, repos compris. Utile pour l'autonomie, pas pour la comparaison.

    Si le contrôle du nombre de créneaux échoue, aucune énergie n'est publiée : tous
    les champs passent à `A_MESURER` et portent la raison chiffrée du refus.

    Parameters
    ----------
    trace : dict[str, np.ndarray]
        Sortie de `_load_csv` — exige ``current_a``, ``time_s`` et ``voltage_v``.
    rate_hz : float
        Cadence imposée au flux pendant l'acquisition (Hz).
    threshold : float | None
        Seuil de binarisation (A) ; ``None`` → déduit de la trace.
    tolerance : float
        Tolérance du contrôle de dénominateur.

    Returns
    -------
    dict
        Bloc consommable tel quel par le pilote S5305.
    """
    t = np.asarray(trace["time_s"], dtype=np.float64)
    i = np.asarray(trace["current_a"], dtype=np.float64)
    v = trace.get("voltage_v")
    duration_s = float(t[-1] - t[0]) if t.size > 1 else 0.0
    fs_hz = (t.size - 1) / duration_s if duration_s > 0 else 0.0
    dt_s = 1.0 / fs_hz if fs_hz > 0 else 0.0

    base: dict = {
        "segmentation": "courant (voie A)",
        "fs_hz": fs_hz,
        "duration_s": duration_s,
        "n_samples": int(i.size),
        "method": METHOD_INTEGRATION,
        "phases_na_reason": NA_PHASES_1BIT,
    }

    def _refus(raison: str, **extra) -> dict:
        """Sortie N/A honnête — jamais 0, jamais une estimation."""
        return {
            **base,
            **extra,
            "phases_uj": {p: A_MESURER for p in PHASES},
            "total_uj": A_MESURER,
            "energy_uj_per_inference": A_MESURER,
            "energy_uj_per_burst_gross": A_MESURER,
            "energy_na_reason": raison,
        }

    if v is None:
        return _refus(
            "trace sans colonne tension : l'énergie exige I ET V mesurés, fabriquer "
            "une tension d'alimentation est interdit."
        )
    if dt_s <= 0.0:
        return _refus(
            f"base de temps inexploitable ({i.size} échantillon(s), durée "
            f"{duration_s:g} s) : aucun profil temporel n'en sort."
        )

    seuil = robust_current_threshold(i) if threshold is None else float(threshold)
    windows = derive_phase_windows_from_current(i, dt_s, seuil)
    n_bursts = count_bursts(windows)
    n_expected = float(rate_hz) * duration_s
    ok, ecart_pct, raison = validate_burst_count(n_bursts, rate_hz, duration_s, tolerance)

    v_arr = np.asarray(v, dtype=np.float64)
    detection = {
        "threshold_a": seuil,
        "n_bursts_detected": n_bursts,
        "n_bursts_expected": n_expected,
        "detection_error_pct": ecart_pct,
        "tolerance_pct": tolerance * 100.0,
        "rate_hz": float(rate_hz),
        "samples_per_burst_median": _median_burst_samples(windows, dt_s),
    }
    if not ok:
        return _refus(raison or "segmentation refusée", **detection)

    # Intégration par phase — mêmes conventions que `segment_by_phase`, mais les
    # fenêtres viennent ici du courant et non d'une colonne de synchronisation.
    phases_uj = {p: 0.0 for p in PHASES}
    idle_duration_s, burst_duration_s = 0.0, 0.0
    for name, t0, t1 in windows:
        i0, i1 = int(round(t0 / dt_s)), int(round(t1 / dt_s))
        if i1 <= i0:
            continue
        phases_uj[name] += integrate_energy_uj(i[i0:i1], v_arr[i0:i1], dt_s)
        if name == "idle":
            idle_duration_s += (i1 - i0) * dt_s
        else:
            burst_duration_s += (i1 - i0) * dt_s

    gross_uj = phases_uj["inference"] / n_bursts
    marginal_uj = marginal_uj_per_burst(
        phases_uj["inference"], phases_uj["idle"],
        burst_duration_s, idle_duration_s, n_bursts)
    idle_power_w = ((phases_uj["idle"] / 1e6) / idle_duration_s
                    if idle_duration_s > 0 else 0.0)

    return {
        **base,
        **detection,
        "phases_uj": {
            # Le marqueur ne distingue pas ces deux phases du niveau actif : elles
            # restent « à mesurer », elles ne valent pas zéro.
            "startup": A_MESURER,
            "acquisition": A_MESURER,
            "inference": phases_uj["inference"],
            "idle": phases_uj["idle"],
        },
        "total_uj": phases_uj["inference"] + phases_uj["idle"],
        "idle_power_w": idle_power_w,
        "active_duration_s": burst_duration_s,
        "duty_cycle": burst_duration_s / duration_s if duration_s > 0 else None,
        "energy_uj_per_inference": marginal_uj,
        "energy_uj_per_burst_gross": gross_uj,
    }


def _median_burst_samples(
    windows: list[tuple[str, float, float]], dt_s: float
) -> float | None:
    """Nombre médian d'échantillons par créneau actif — la limite de la voie A.

    À 100 kSPS, HDC (2095 µs) couvre ~200 échantillons par burst, EWC (~50 µs) en
    couvre 5 : le second est à la limite du segmentable, et ce champ le rend visible
    dans le JSON au lieu de le laisser deviner.
    """
    tailles = [(t1 - t0) / dt_s for name, t0, t1 in windows if name == "inference"]
    return float(np.median(tailles)) if tailles else None


def export_energy_json(
    phases_uj: dict,
    model: str,
    encoding: str,
    output_path: Path,
    sampling_rate_hz: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Exporte un JSON énergie normalisé pour un couple modèle × encodage.

    Schéma :
        {"model", "encoding", "phases_uj": {startup, acquisition, inference, idle},
         "total_uj", "sampling_rate_hz", "duration_s", "timestamp", "source"}

    Si `phases_uj` est ``None`` (aucune mesure réelle), tous les champs énergie
    valent la constante littérale `A_MESURER` — aucun chiffre n'est inventé.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    measured = phases_uj is not None
    if measured:
        phases = {p: float(phases_uj.get(p, 0.0)) for p in PHASES}
        total = float(sum(phases.values()))
    else:
        phases = {p: A_MESURER for p in PHASES}
        total = A_MESURER

    payload = {
        "model": model,
        "encoding": encoding,
        "phases_uj": phases,
        "total_uj": total,
        # TODO(dorra) : calibrer la fréquence d'échantillonnage / plage de courant
        # LPM01A (capter la veille en µA comme l'inférence en mA).
        "sampling_rate_hz": sampling_rate_hz if sampling_rate_hz else A_MESURER,
        "duration_s": duration_s if duration_s else A_MESURER,
        "source": "lpm01a_csv" if measured else "placeholder",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    status = "mesuré" if measured else f"placeholder ({A_MESURER})"
    print(f"  ✔ {output_path}  [{model}/{encoding}] — {status}")


def _capture_one(
    model: str,
    encoding: str,
    output_path: Path,
    duration_s: float,
    sampling_rate_hz: float,
    csv_in: Path | None,
) -> None:
    """Pipeline complet pour un couple : capture → segmentation → export JSON."""
    csv_path = capture_session(
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        output_csv=output_path.with_suffix(".csv"),
        csv_in=csv_in,
    )
    if csv_path is None:
        export_energy_json(
            None,
            model,
            encoding,
            output_path,
            sampling_rate_hz=sampling_rate_hz or None,
            duration_s=duration_s,
        )
        return
    # CSV réel : les fenêtres de phase viennent du signal de sync PA8 (S3304),
    # capté par le LPM01A dans la même trace. Sans cette colonne, on refuse
    # plutôt que de fabriquer une segmentation.
    trace = _load_csv(csv_path)
    if trace.get("sync") is None:
        raise ValueError(
            f"CSV {csv_path} sans colonne de synchronisation (sync/pa8/gpio) : "
            "impossible de segmenter par phase. Exporter le signal PA8 du firmware "
            "(marqueurs S3304) en parallèle du courant depuis STM32CubeMonitor-Power."
        )
    windows = derive_phase_windows(trace)
    phases_uj = segment_by_phase(csv_path, windows)
    export_energy_json(
        phases_uj,
        model,
        encoding,
        output_path,
        sampling_rate_hz=sampling_rate_hz or None,
        duration_s=duration_s,
    )


def _write_summary(output_dir: Path) -> None:
    """Agrège les JSON modèle×encodage en `summary.json` (table FP32 vs INT8).

    Pour chaque modèle : `delta_uj`, `ratio` (int8/fp32). Tant que les énergies
    sont `A_MESURER`, ces deltas le restent aussi. Le lien Gap 3 est explicité
    en texte (réduction RAM INT8 sans accélération latence FPU, Sprint 29).
    """
    rows: dict[str, dict] = {}
    for model in CAMPAIGN_MODELS:
        entry: dict = {}
        for enc in CAMPAIGN_ENCODINGS:
            p = output_dir / f"{model}_{enc}.json"
            if p.is_file():
                entry[enc] = json.loads(p.read_text(encoding="utf-8")).get("total_uj")
        fp32, int8 = entry.get("fp32"), entry.get("int8")
        numeric = isinstance(fp32, (int, float)) and isinstance(int8, (int, float))
        entry["delta_uj"] = (int8 - fp32) if numeric else A_MESURER
        entry["ratio_int8_fp32"] = (int8 / fp32) if numeric and fp32 else A_MESURER
        rows[model] = entry

    summary = {
        "description": "Campagne énergie S3306 — µJ total par modèle × encodage (LPM01A).",
        "phases": list(PHASES),
        "per_model": rows,
        "gap3_note": (
            "Sprint 29 : l'INT8 réduit la RAM sans accélérer la latence (FPU "
            "Cortex-M4, pas de NPU INT8). Question énergie : l'INT8 réduit-il "
            "néanmoins les µJ (moins d'accès mémoire) ? — réponse via mesures "
            "LPM01A réelles, champs '" + A_MESURER + "' tant que non capturées."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✔ {out}  — agrégat FP32 vs INT8")


def run_campaign(output_dir: Path, duration_s: float, sampling_rate_hz: float) -> None:
    """Campagne S3306 : 4 modèles × {fp32, int8} + summary.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Campagne énergie S3306 → {output_dir}/")
    for model in CAMPAIGN_MODELS:
        for enc in CAMPAIGN_ENCODINGS:
            _capture_one(
                model,
                enc,
                output_dir / f"{model}_{enc}.json",
                duration_s,
                sampling_rate_hz,
                csv_in=None,
            )
    _write_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture énergie LPM01A → µJ par phase (S3305/S3306)."
    )
    parser.add_argument("--model", choices=CAMPAIGN_MODELS, help="Modèle ciblé.")
    parser.add_argument("--encoding", choices=CAMPAIGN_ENCODINGS, help="Encodage.")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Durée nominale de capture (s)."
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=0.0,
        help="Fréq. échantillonnage LPM01A (0 = TODO(dorra)).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Trace LPM01A exportée (STM32CubeMonitor-Power)."
    )
    parser.add_argument("--output", type=Path, help="Chemin JSON de sortie (mode couple unique).")
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="Campagne complète 4 modèles × {fp32,int8} + summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/exp_S33_energy"),
        help="Répertoire de sortie (mode --campaign).",
    )
    args = parser.parse_args()

    if args.campaign:
        run_campaign(args.output_dir, args.duration, args.sampling_rate_hz)
        return

    if not (args.model and args.encoding and args.output):
        print("Erreur : --model, --encoding et --output requis (ou --campaign).", file=sys.stderr)
        sys.exit(2)

    _capture_one(
        args.model,
        args.encoding,
        args.output,
        args.duration,
        args.sampling_rate_hz,
        args.csv,
    )


if __name__ == "__main__":
    main()
