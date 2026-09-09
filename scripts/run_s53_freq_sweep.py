"""
run_s53_freq_sweep.py — S5303 : balayage de fréquence SYSCLK 180 / 90 / 45 MHz.

DEUX QUESTIONS, UNE SEULE MODIFICATION FIRMWARE (`-DSYSCLK_MHZ`, cf. hw_info.c) :

    1. **Un argument système que le mémoire n'a pas.** Le Gap 2 dispose de trois ordres
       de grandeur de marge (pire latence mesurée 2095 µs contre 100 ms). Faut-il
       ralentir le MCU pour tenir l'autonomie ? La latence croît en 1/f et le courant
       décroît : l'énergie par inférence est-elle constante (le calcul domine) ou
       décroissante (les fuites et la scrutation dominent) ? La réponse peut
       parfaitement être « ralentir ne gagne rien » — c'est un résultat.
    2. **Le plafond de 59 mA du mode dynamique de la sonde.** `acqmode dyn` refuse
       au-delà ; la carte en tire 63–75 mA à 180 MHz. Baisser la fréquence est la
       dernière voie de contournement après l'échec mesuré de la veille du PHY. Si la
       carte passe dessous, le profil temporel par phase redevient accessible (S5305).

CE QUE LE PILOTE NE FAIT PAS : il ne flashe pas. Le flash se fait sur alimentation
normale (`JP5` en place), la mesure sur la sonde — c'est une manipulation manuelle de
cavalier, constatée nécessaire (le flash sous alimentation sonde est instable). Le
pilote exige donc `--sysclk-mhz`, qu'il recopie tel quel, et le CONTRÔLE contre la
fréquence que la carte rapporte elle-même (`read_reported_sysclk` : reset, puis lecture de
la bannière `hw_info_print`, dont le `SYSCLK` est RECALCULÉ à bord depuis `PLLCFGR`). En
cas d'écart, aucune cellule n'est écrite : un JSON qui ment sur sa fréquence serait pire
que pas de mesure — c'est la leçon du drapeau TinyOL du Sprint 52.

INTÉGRITÉ DU FLUX — réserve honnête : sous protocole V3, le compteur `crc_errors` de
`sensor_stream.py` est structurellement nul (le firmware resynchronise en silence sur
CRC invalide, `pipeline.c`, et la réponse V3 ne porte pas d'octet de statut). Une trame
perdue se manifeste par un échantillon MANQUANT. L'intégrité est donc contrôlée par
`n_samples` reçus vs attendus, et c'est cette grandeur qui est reportée — pas un
« 0 CRC » qui ne prouverait rien.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : un essai qui échoue est consigné avec la
valeur relevée, jamais avec une appréciation.

Usage (le build `-DSYSCLK_MHZ=<f>` doit être flashé AVANT) :
    python scripts/run_s53_freq_sweep.py --sysclk-mhz 90 --board-port /dev/serial/by-id/…
    python scripts/run_s53_freq_sweep.py --summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")
lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")

A_MESURER = ec.A_MESURER
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_freq_sweep"
MA = 1000.0
GAP2_BUDGET_US = 100_000.0   # critère Gap 2 : 100 ms par inférence + mise à jour

#: Nom de l'estimateur, distinct de `rate_regression.METHOD` : c'est le MÊME ajustement,
#: mais à fréquence SYSCLK fixée. Deux estimateurs différents ne portent pas le même nom.
METHOD = "régression I(cadence) à fréquence SYSCLK fixée"

#: Fraction des échantillons attendus en-deçà de laquelle une acquisition dynamique est
#: tenue pour INTERROMPUE (surintensité). La sonde ne lève rien dans ce cas : elle rend ce
#: qu'elle a eu le temps de décoder — cf. `try_dynamic_mode`.
DYN_COMPLETENESS = 0.95

#: Invariant de conception du balayage : PCLK1 reste à 45 MHz aux trois fréquences, sinon
#: `USART3->BRR = 0x0187` (figé en dur, hw_info.c) cesse d'être valide et tout le protocole
#: tombe. C'est PPRE1 qui absorbe la variation de PLLP.
PCLK1_HZ = 45_000_000

#: Fréquence de référence du projet : toutes les latences publiées jusqu'au Sprint 52 ont
#: été mesurées à 180 MHz. Sert de dénominateur au facteur 1/f attendu.
REF_MHZ = 180

#: Configuration attendue par fréquence — miroir EXACT de `hw_clock_init` (hw_info.c).
#: Sert à documenter le JSON, et à vérifier que la carte rapporte bien ce qui a été
#: compilé (le champ `sysclk_hz` est RECALCULÉ à bord depuis PLLCFGR).
PLL_BY_MHZ = {
    180: {"pllm": 8, "plln": 180, "pllp": 2, "ppre1": 4, "ppre2": 2,
          "flash_ws": 5, "overdrive": True, "pclk2_hz": 90_000_000},
    90:  {"pllm": 8, "plln": 180, "pllp": 4, "ppre1": 2, "ppre2": 1,
          "flash_ws": 2, "overdrive": False, "pclk2_hz": 90_000_000},
    45:  {"pllm": 8, "plln": 180, "pllp": 8, "ppre1": 1, "ppre2": 1,
          "flash_ws": 1, "overdrive": False, "pclk2_hz": 45_000_000},
}

#: Modèles couvrant le spectre de latence (Maha ~5 µs → HDC ~2 ms).
LATENCY_MODELS = ("mahalanobis", "ewc", "hdc")

#: Bannière émise par `hw_info_print` au démarrage :
#:     SYSCLK : 90 MHz  HCLK : 90 MHz  PCLK1 : 45 MHz  PCLK2 : 90 MHz
BANNER_RE = re.compile(
    r"SYSCLK\s*:\s*(\d+)\s*MHz.*?PCLK1\s*:\s*(\d+)\s*MHz", re.DOTALL)


def read_reported_sysclk(board_port: str, baud: int = 115200,
                         timeout_s: float = 8.0) -> tuple[int, int, str]:
    """Reset la carte et lit la fréquence QU'ELLE rapporte, en MHz.

    Pourquoi ce contrôle est indispensable : `--sysclk-mhz` est recopié tel quel dans le
    nom du fichier et dans le champ `sysclk_mhz`. Sans vérification, une cellule mesurée
    sur un binaire 180 MHz mais lancée avec `--sysclk-mhz 45` s'écrirait sous un nom qui
    ment sur son contenu — exactement le bug du drapeau TinyOL du Sprint 52.

    Ce que la bannière vaut : `hw_info_collect` RECALCULE `sysclk_hz` depuis `PLLCFGR`
    (hw_info.c) au lieu de réafficher la constante de compilation. La valeur lue est donc
    la configuration réellement chargée dans le silicium, pas un écho du `-D`.

    Retourne `(sysclk_mhz, pclk1_mhz, banniere_brute)`.
    """
    import serial   # dépendance déjà requise par sensor_stream.py

    # Le port est ouvert et VIDÉ *avant* le reset, et la bannière retenue est la DERNIÈRE
    # lue, pas la première. Sans ces deux précautions, le contrôle peut porter sur une
    # bannière PÉRIMÉE : la carte en émet une à chaque démarrage — donc au « Resetting
    # Target » de `make flash` et à chaque mise sous tension par la sonde — et ces octets
    # attendent dans le tampon du système. Mesuré 2026-09-08 : après un flash
    # `-DSYSCLK_MHZ=90` vérifié OK, le pilote a lu « 180 MHz » (la bannière du binaire
    # précédent, encore en tampon) et refusé d'écrire la cellule, alors que la carte
    # tournait bien à 90 MHz — vérifié par relecture directe et par la constante PLLCFGR
    # du binaire (PLLP=4). Le refus était la bonne réaction ; c'est la lecture qui était
    # fausse. Même classe de défaut que `lpm01a_probe.voltage_v` (S5305).
    with serial.Serial(board_port, baud, timeout=1.0) as ser:
        ser.reset_input_buffer()
        subprocess.run(
            ["openocd", "-f", "interface/stlink.cfg", "-f", "target/stm32f4x.cfg",
             "-c", "init; reset; exit"],
            capture_output=True, timeout=20,
        )
        deadline = time.time() + timeout_s
        banner = ""
        while time.time() < deadline:
            banner += ser.read(512).decode("ascii", errors="replace")
            trouvees = list(BANNER_RE.finditer(banner))
            # Une bannière complète est suivie de la ligne « RAM total » : attendre cette
            # suite évite de trancher sur un fragment coupé en plein vol.
            if trouvees and "RAM total" in banner[trouvees[-1].end():]:
                found = trouvees[-1]
                return int(found.group(1)), int(found.group(2)), banner.strip()[-400:]
    raise RuntimeError(
        "bannière `hw_info_print` illisible sur "
        f"{board_port} après {timeout_s:.0f} s : la fréquence réellement chargée n'est pas "
        "constatable. Relancer après reset, ou assumer explicitement --skip-hw-check "
        "(la cellule sortira alors avec sysclk_reported_mhz = null et sa raison)."
    )


def check_frequency(args) -> dict:
    """Bloc `hw_check` de la cellule — la carte confirme-t-elle le build flashé ?"""
    if args.skip_hw_check:
        return {
            "sysclk_reported_mhz": None,
            "pclk1_reported_hz": None,
            "sysclk_match": None,
            "na_reason": "contrôle de fréquence explicitement désactivé (--skip-hw-check) : "
                         "la fréquence de cette cellule est DÉCLARÉE par l'opérateur, pas "
                         "constatée à bord.",
        }
    reported, pclk1, banner = read_reported_sysclk(args.board_port, args.baud)
    if reported != args.sysclk_mhz:
        raise RuntimeError(
            f"la carte rapporte {reported} MHz alors que --sysclk-mhz vaut "
            f"{args.sysclk_mhz} : le binaire flashé n'est pas celui annoncé. Aucune "
            f"cellule n'est écrite — un JSON qui ment sur sa fréquence est pire que "
            f"pas de mesure."
        )
    if pclk1 * 1_000_000 != PCLK1_HZ:
        raise RuntimeError(
            f"PCLK1 rapporté = {pclk1} MHz au lieu de 45 MHz : le BRR UART figé à 0x0187 "
            f"n'est plus valide, le protocole est hors spécification."
        )
    print(f"[freq {args.sysclk_mhz}] carte confirmée : SYSCLK {reported} MHz, "
          f"PCLK1 {pclk1} MHz")
    return {
        "sysclk_reported_mhz": reported,
        "pclk1_reported_hz": pclk1 * 1_000_000,
        "sysclk_match": True,
        "hw_banner": banner,
    }


def stream_once(probe, voltage_mv: int, model: str, dataset: str, port: str,
                n_samples: int, rate_hz: float) -> dict:
    """Un flux complet exécuté DANS une acquisition maintenue (la sonde alimente).

    Hors acquisition, la carte s'effondre et redémarre en boucle : `hold_run` est une
    nécessité mesurée, pas une commodité.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    cmd = rc.stream_command(model, dataset, port, n_samples, rate_hz) + \
        ["--output", str(out_path)]
    shell_cmd = "cd " + str(ROOT) + " && " + \
        " ".join(f"'{c}'" if " " in c else c for c in cmd)
    rcode = lp.hold_run(probe, voltage_mv, shell_cmd)
    try:
        if rcode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"flux « {model} » échoué (code {rcode}) : le contrôle d'intégrité ne "
                f"peut pas être remplacé par une valeur par défaut."
            )
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        out_path.unlink(missing_ok=True)


def try_dynamic_mode(probe, voltage_mv: int, duration_s: float) -> dict:
    """Tente `acqmode dyn` et consigne le RÉSULTAT RELEVÉ, succès ou échec.

    Le mode dynamique est ce qui débloquerait le profil temporel par phase (S5305) ;
    son plafond de 59 mA est une limite matérielle mesurée, pas un réglage.
    """
    try:
        samples, voltage_v, summary = lp.capture(
            probe, freq="100k", duration_s=duration_s,
            voltage_mv=voltage_mv, acqmode="dyn")
        # Une acquisition dynamique peut ne PAS lever et avoir été INTERROMPUE par la
        # sonde (surintensité au-delà de ~59 mA) : elle rend alors les quelques dizaines
        # de ms décodées avant l'arrêt, et `lp.capture` ne lève que sur un flux VIDE.
        # Mesuré 2026-09-01 à 180 MHz : 71 ms et 7130 échantillons pour 1 s demandée,
        # « Measurement interrupted due to errors », I_max 75,7 mA — soit exactement
        # l'échec documenté du 2026-08-04. Sans ce contrôle la cellule publierait
        # « mode dynamique OK », l'inverse de la mesure, et débloquerait S5305 à tort.
        n_expected = int(lp.parse_freq_hz("100k") * duration_s)
        n_decoded = int(len(samples))
        interrompu = any(mot in summary.lower() for mot in ("interrupted", "overcurrent"))
        complet = n_decoded >= int(DYN_COMPLETENESS * n_expected)
        cell = {
            "attempted": True,
            "succeeded": bool(complet and not interrompu),
            "n_samples": n_decoded,
            "n_samples_expected": n_expected,
            "i_mean_ma": float(np.mean(samples)) * MA,
            "i_max_ma": float(np.max(samples)) * MA,
            "tension_v": float(voltage_v),
            "summary": summary.strip()[-400:],
        }
        if not cell["succeeded"]:
            cell["na_reason"] = (
                f"acquisition dynamique interrompue par la sonde : {n_decoded}/{n_expected} "
                f"échantillons décodés, I_max={cell['i_max_ma']:.1f} mA"
                + (" ; résumé : « Measurement interrupted »" if interrompu else "")
            )
        return cell
    except Exception as exc:   # LPM01AError ou erreur de flux : les deux se consignent
        message = str(exc)
        found = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*mA", message)
        return {
            "attempted": True,
            "succeeded": False,
            "i_max_ma": float(found.group(1).replace(",", ".")) if found else None,
            "na_reason": f"mode dynamique refusé par la sonde : {message}",
        }


def measure_cell(args) -> dict:
    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    expected = PLL_BY_MHZ[args.sysclk_mhz]

    # Tension de service : valeur de consigne du profil, écrasée par celle que la sonde
    # rapporte à la première acquisition. Initialisée ici pour que la cellule reste
    # écrivable même si la boucle de cadence est vide (`--rates` réduit à un point).
    voltage_v = float(calib.get("supply_voltage_v", voltage_mv / 1000.0))

    probe = lp.PowerShield(args.port)
    try:
        probe.take_control()
        warmups = [lp.warmup(probe, voltage_mv)
                   for _ in range(max(1, args.warmup_repeats))]

        # 0. La carte tourne-t-elle à la fréquence annoncée ? Contrôlé avant toute MESURE :
        #    si le binaire flashé n'est pas celui déclaré, rien n'est écrit.
        #
        #    Ce contrôle s'exécute DANS la session de sonde, après le préchauffage : `JP5`
        #    étant retiré, la cible n'est alimentée que par la sonde, et `pwrend on` la
        #    maintient une fois l'acquisition terminée. Placé avant `take_control()`, il
        #    interrogeait une carte hors tension et échouait sur « bannière illisible »
        #    (mesuré 2026-09-01 à 45 MHz) — les cellules précédentes n'étaient passées que
        #    parce que la sortie était restée énergisée d'une exécution antérieure.
        hw_check = check_frequency(args)

        # 1. Vérification protocole + latence DWT, un flux par modèle.
        protocol: dict[str, dict] = {}
        latency_p50: dict[str, float | None] = {}
        for model in LATENCY_MODELS:
            res = stream_once(probe, voltage_mv, model, args.dataset,
                              args.board_port, args.n_check, args.rate_hz)
            # Intégrité : `--n-samples` est le nombre de trames ENVOYÉES ; une trame perdue
            # se lit comme un échantillon manquant (cf. réserve en tête de module).
            n_expected = int(args.n_check)
            protocol[model] = {
                "n_samples_received": res.get("n_samples"),
                "n_samples_expected": n_expected,
                "samples_lost": n_expected - int(res.get("n_samples", 0)),
                "crc_errors": res.get("crc_errors"),
                "latency_p50_us": res.get("latency_p50_us"),
                "latency_p99_us": res.get("latency_p99_us"),
            }
            latency_p50[model] = res.get("latency_p50_us")
            print(f"[freq {args.sysclk_mhz}] {model:12s} : "
                  f"{res.get('n_samples')}/{n_expected} éch., "
                  f"P50={res.get('latency_p50_us')} µs")

        # 2. Courant moyen à plusieurs cadences → pente (méthode S5304).
        points = []
        n_samples = int(max(args.rates) * (args.window + args.settle + 4.0))
        for rate in args.rates:
            cmd = (None if rate <= 0 else
                   rc.stream_command(rc.STREAM_MODEL[(args.model, args.encoding)],
                                     args.dataset, args.board_port, n_samples, rate))
            runs = []
            for _ in range(args.repeats):
                i_run, voltage_v = rc.measure_current(
                    probe, args.window, voltage_mv, cmd, args.settle if cmd else 0.0)
                runs.append(i_run)
            point = {
                "rate_hz": float(rate),
                "i_mean_a": float(np.mean(runs)),
                "i_std_a": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
                "n_repeats": len(runs),
            }
            # A7 — `achieved_rate_hz` est un champ de MESURE : il ne se remplit pas avec la
            # consigne. Seule la borne haute est re-streamée (étape 2bis) ; les autres
            # cadences restent NON MESURÉES, ce que `saturation_rate_hz` traite déjà comme
            # « ne prouve rien » (elle ignore les points sans cadence atteinte).
            if rate > 0:
                point["achieved_rate_hz"] = None
                point["achieved_rate_source"] = "non mesuré"
            points.append(point)
            print(f"[freq {args.sysclk_mhz}] {rate:5.0f} Hz : "
                  f"{np.mean(runs) * MA:7.3f} mA")

        # 2bis. Cadence ATTEINTE à la borne haute (port de S5304). La saturation ne se lit
        #    ni dans les CRC ni dans les trames perdues : au-delà du plafond de transport,
        #    `sensor_stream.py` émet simplement moins vite que la consigne. Les points
        #    saturés se tassent alors sur l'axe des abscisses et SOUS-ESTIMENT la pente.
        #    Mesuré 2026-09-01 à 180 MHz sur hdc_int8 (plafond réel 126 Hz) : le point
        #    200 Hz tirait la pente de 62,9 à 39,5 µA/Hz, soit 205,6 → 129,1 µJ/inférence,
        #    contre 191,6 ± 16,0 µJ mesurés indépendamment par S5304. Le biais s'aggrave à
        #    mesure que l'horloge ralentit (la latence double puis quadruple), donc il
        #    fabriquerait une fausse décroissance de l'énergie avec la fréquence — soit
        #    l'inverse de la conclusion cherchée.
        #
        #    `--achieved-at-each-rate` (port de S5304) relève la cadence atteinte à CHAQUE
        #    point au lieu de la seule borne haute : l'inférence ci-dessous ne sert alors
        #    plus qu'aux cadences non re-streamées, au prix d'un flux supplémentaire par
        #    point. À utiliser quand la cellule est rejouée pour lever un doute sur sa
        #    linéarité (B1) — c'est précisément le cas où l'abscisse doit être mesurée.
        rate_max = max(args.rates)
        mesurees = ([r for r in args.rates if r > 0] if args.achieved_at_each_rate
                    else ([rate_max] if rate_max > 0 else []))
        atteints: dict[float, float | None] = {}
        for rate in mesurees:
            res_rate = stream_once(probe, voltage_mv,
                                   rc.STREAM_MODEL[(args.model, args.encoding)],
                                   args.dataset, args.board_port, args.n_check, rate)
            atteints[float(rate)] = res_rate.get("achieved_rate_hz")
            if args.achieved_at_each_rate:
                print(f"[freq {args.sysclk_mhz}] cadence atteinte : consigne {rate:.0f} Hz "
                      f"→ atteint {res_rate.get('achieved_rate_hz')} Hz")
        for p in points:
            valeur = atteints.get(p["rate_hz"])
            if p["rate_hz"] > 0 and valeur is not None:
                p["achieved_rate_hz"] = float(valeur)
                p["achieved_rate_source"] = "mesuré"
        if rate_max > 0:
            atteint = atteints.get(float(rate_max))
            # Le plafond est une propriété de la CELLULE (calcul + transport), pas du seul
            # point où on l'a mesuré : toute cadence commandée au-dessus est irréalisable.
            # On le propage donc aux cadences supérieures, sinon `saturation_rate_hz` — qui
            # cherche la plus petite consigne dont la cadence atteinte décroche — ne voit
            # que la borne haute et conserve des points saturés. Mesuré 2026-09-01 à
            # 45 MHz : plafond 71,4 Hz, mais seul 200 Hz écarté, le point 100 Hz retenu
            # tirait le r² à 0,888 et sortait la cellule en N/A.
            #
            # Réserve : pour les cadences intermédiaires c'est une INFÉRENCE (le plafond
            # n'y est pas mesuré point par point), physiquement fondée car la cadence
            # réalisable ne croît pas avec la demande au-delà du plafond. Elle ne sert qu'à
            # ÉCARTER des points, jamais à corriger une abscisse : aucune donnée fabriquée.
            if atteint is not None:
                for p in points:
                    # Les cadences RE-STREAMÉES portent déjà leur mesure (boucle ci-dessus)
                    # et ne sont jamais écrasées par l'inférence : celle-ci ne comble que
                    # les points laissés « non mesuré ».
                    if (p["rate_hz"] > 0
                            and p.get("achieved_rate_source") == "non mesuré"
                            and p["rate_hz"] > float(atteint)):
                        p["achieved_rate_hz"] = float(atteint)
                        p["achieved_rate_source"] = "inféré du plafond mesuré"
            print(f"[freq {args.sysclk_mhz}] plafond {args.model}_{args.encoding} : "
                  f"consigne {rate_max:.0f} Hz → atteint {atteint} Hz")

        # Ajustement sur les seuls points NON saturés — la règle vit dans le module commun
        # de S5304 (`usable_points`/`saturation_rate_hz`), jamais réimplémentée ici. Les
        # points écartés restent écrits dans le JSON : une mesure écartée se montre.
        saturation = rr.saturation_rate_hz(points)

        # 3. Mode dynamique : la question du plafond 59 mA.
        dyn = try_dynamic_mode(probe, voltage_mv, args.dyn_duration)
        print(f"[freq {args.sysclk_mhz}] acqmode dyn : "
              f"{'OK' if dyn['succeeded'] else 'refusé'}")
    finally:
        probe.release()

    worst_us = max((v for v in latency_p50.values() if v is not None), default=None)
    derived = derive_cell(points, float(voltage_v),
                          latency_p50.get(args.model) or latency_p50.get("ewc"))

    cell = {
        "sysclk_mhz": int(args.sysclk_mhz),
        "firmware_build": (f"-DSYSCLK_MHZ={args.sysclk_mhz}"
                           if args.sysclk_mhz != 180 else "défaut (180 MHz)"),
        "pll": {k: expected[k] for k in ("pllm", "plln", "pllp", "ppre1", "ppre2")},
        "pclk1_hz": PCLK1_HZ,   # invariant par construction — c'est ce qui sauve le BRR
        "pclk2_hz": expected["pclk2_hz"],
        "flash_ws": expected["flash_ws"],
        "overdrive": expected["overdrive"],
        "hw_check": hw_check,
        "protocol_check": protocol,
        "dwt_latency_us_p50": latency_p50,
        "gap2_budget_us": GAP2_BUDGET_US,
        "gap2_ok": bool(worst_us is not None and worst_us < GAP2_BUDGET_US),
        "gap2_worst_us": worst_us,
        "gap2_margin_x": (GAP2_BUDGET_US / worst_us) if worst_us else None,
        "gap2_verdict": gap2_verdict(args.sysclk_mhz, worst_us, latency_p50),
        "i_mean_ma_by_rate": [{"rate_hz": p["rate_hz"], "i_ma": p["i_mean_a"] * MA}
                              for p in points],
        "current_points": points,
        "saturation_rate_hz": saturation,
        "n_points_measured": len(points),
        "regression_model": f"{args.model}_{args.encoding}",
        **derived,
        "acqmode_dyn": dyn,
        "tension_v": float(voltage_v),
        "window_s": float(args.window),
        "warmup_discarded_a": [float(w) for w in warmups],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return cell


def derive_cell(points: list[dict], tension_v: float,
                latency_us: float | None) -> dict:
    """Grandeurs dérivées d'une cellule SYSCLK — déléguées à la régression de S5304.

    Le pilote ne fitte plus lui-même. Il passe les points DICTS, `i_std_a` compris : c'est
    ce que faisait la version précédente (`_linear_fit` recevait des couples `(x, y)` et
    jetait les écarts-types), et c'est ce qui faisait diverger les deux estimateurs du dépôt
    — le balayage de cadence pondérait par `1/σ²`, celui-ci non. Sur les trois cellules
    mesurées, les deux règles publiaient bien deux chiffres différents pour la même mesure,
    et la cellule 90 MHz passait de « publiable » (r²=0,919) à N/A (r²=0,783 pondéré).

    La décision « énergie chiffrée ou "à mesurer" » vient donc, elle aussi, de
    `rate_regression.energy_uj_per_inference` : r² suffisant ET pente séparable de zéro à
    2σ. `method` est réaffecté après coup — la fréquence FIXÉE est ce qui distingue cet
    estimateur de celui de S5304, et deux estimateurs différents ne doivent pas porter le
    même nom.
    """
    try:
        bloc = rr.fit_cell(points, tension_v, latency_us)
    except (ValueError, ZeroDivisionError) as exc:
        # Moins de trois cadences distinctes : la droite est exacte par construction et son
        # erreur-type n'existe pas. Aucune pente n'est publiée depuis un tel ajustement.
        return {
            "method": METHOD,
            "energy_uj_per_inference": A_MESURER,
            "energy_na_reason": f"ajustement impossible sur ce balayage : {exc}",
            "n_points_fitted": len(rr.usable_points(points)),
        }
    bloc["method"] = METHOD
    # COLLISION DE CLÉ : `fit_cell` renvoie la latence SCALAIRE du modèle de régression sous
    # `dwt_latency_us_p50`, alors que la cellule SYSCLK porte sous ce nom un DICTIONNAIRE
    # {modèle → latence} (c'est lui qui alimente le contrôle 1/f). L'écraser transformait le
    # dictionnaire en flottant et cassait `latency_scaling`. La latence du modèle régressé
    # est donc renommée : les deux grandeurs coexistent, aucune n'est perdue.
    bloc["regression_latency_us_p50"] = bloc.pop("dwt_latency_us_p50", None)
    return bloc


def refit(out_dir: Path) -> dict[str, dict]:
    """Recalcule les grandeurs dérivées des cellules déjà mesurées, SANS carte ni sonde.

    Même rôle que `run_s53_rate_sweep.refit` : reprendre une session après coup, ou faire
    passer les cellules existantes sous une règle d'ajustement corrigée. Les `current_points`
    — les seules valeurs MESURÉES — ne sont jamais touchés.
    """
    cells: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        if "current_points" not in cell or "sysclk_mhz" not in cell:
            continue   # artefact tiers : lecture seule, jamais réécrit
        latences = cell.get("dwt_latency_us_p50", {})
        modele = str(cell.get("regression_model", "")).rsplit("_", 1)[0]
        latency = latences.get(modele) or latences.get("ewc")
        # Les champs dérivés d'une règle antérieure sont retirés avant réécriture : une clé
        # orpheline (`intercept_ma` d'un fit non pondéré) mentirait sur sa provenance.
        for perimee in ("energy_na_reason", "slope_ua_per_hz", "slope_std_ua_per_hz",
                        "intercept_ma", "r2", "energy_uncertainty_uj"):
            cell.pop(perimee, None)
        rr.normalize_achieved_source(cell["current_points"])
        cell.update(derive_cell(cell["current_points"], cell["tension_v"], latency))
        cell["saturation_rate_hz"] = rr.saturation_rate_hz(cell["current_points"])
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
        cells[str(cell["sysclk_mhz"])] = cell
    return cells


def gap2_verdict(sysclk_mhz: int, worst_us: float | None,
                 latency_p50: dict[str, float | None]) -> str:
    """Verdict Gap 2 CHIFFRÉ à cette fréquence (critère d'acceptation, pas une opinion)."""
    if worst_us is None:
        return (f"aucune latence relevée à {sysclk_mhz} MHz : le verdict Gap 2 exige une "
                f"mesure, il n'est pas déductible de la fréquence.")
    pire = max((m for m, v in latency_p50.items() if v == worst_us), default="?")
    marge = GAP2_BUDGET_US / worst_us
    tenu = "TENU" if worst_us < GAP2_BUDGET_US else "DÉPASSÉ"
    return (f"Gap 2 {tenu} à {sysclk_mhz} MHz : pire latence mesurée {worst_us:.0f} µs "
            f"({pire}) contre un budget de {GAP2_BUDGET_US:.0f} µs, soit une marge de "
            f"×{marge:.0f}.")


def latency_scaling(cells: dict[str, dict]) -> dict:
    """Contrôle de sanité 1/f : la latence doit croître comme 180/f quand f baisse.

    C'est aussi un contrôle de la calibration DWT — `hw_dwt_calibrate` reçoit un
    `sysclk_hz` recalculé à bord, donc si les µs rapportées ne suivaient PAS 1/f, ce
    serait la conversion cycles→µs qui serait fausse, pas le silicium.
    """
    ref = cells.get(str(REF_MHZ))
    if ref is None:
        return {"na_reason": f"pas de cellule à {REF_MHZ} MHz : le rapport 1/f n'a pas "
                             f"de référence."}
    out: dict[str, dict] = {}
    for key, cell in cells.items():
        mhz = int(key)
        attendu = REF_MHZ / mhz
        par_modele = {}
        for model, value in cell.get("dwt_latency_us_p50", {}).items():
            base = ref.get("dwt_latency_us_p50", {}).get(model)
            par_modele[model] = (None if not base or value is None
                                 else round(float(value) / float(base), 3))
        out[key] = {"ratio_attendu": attendu, "ratio_mesure_par_modele": par_modele}
    return out


def build_summary(out_dir: Path) -> dict:
    """Conclusion CALCULÉE : l'énergie par inférence croît-elle, stagne-t-elle, ou
    décroît-elle avec la fréquence ? Lecture seule sur les cellules mesurées."""
    cells = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        cells[str(cell["sysclk_mhz"])] = cell

    chiffrees = {int(k): c["energy_uj_per_inference"] for k, c in cells.items()
                 if isinstance(c.get("energy_uj_per_inference"), (int, float))}
    if len(chiffrees) >= 2:
        freqs = sorted(chiffrees)
        e_hi, e_lo = chiffrees[freqs[-1]], chiffrees[freqs[0]]
        ecart_rel = (e_hi - e_lo) / e_lo if e_lo else None
        if ecart_rel is None:
            tendance, raison = A_MESURER, "énergie de référence nulle"
        elif abs(ecart_rel) < 0.1:
            tendance = "constante"
            raison = (f"l'énergie par inférence varie de {ecart_rel * 100:+.1f} % entre "
                      f"{freqs[0]} et {freqs[-1]} MHz : le calcul domine, ralentir le MCU "
                      f"ne change pas le coût énergétique d'une inférence.")
        elif ecart_rel > 0:
            tendance = "croissante_avec_f"
            raison = (f"l'énergie par inférence croît de {ecart_rel * 100:+.1f} % de "
                      f"{freqs[0]} à {freqs[-1]} MHz : ralentir le MCU réduit le coût "
                      f"par inférence — la marge du Gap 2 est convertible en autonomie.")
        else:
            tendance = "decroissante_avec_f"
            raison = (f"l'énergie par inférence décroît de {ecart_rel * 100:+.1f} % de "
                      f"{freqs[0]} à {freqs[-1]} MHz : la part indépendante de la "
                      f"fréquence domine, ralentir allonge l'inférence sans rien gagner.")
    else:
        tendance = A_MESURER
        raison = (f"{len(chiffrees)} fréquence(s) avec une énergie chiffrée : la "
                  f"tendance exige au moins deux points.")

    return {
        "frequencies_mhz": sorted(int(k) for k in cells),
        "energy_uj_per_inference_by_mhz": {
            k: c["energy_uj_per_inference"] for k, c in cells.items()},
        "slope_ua_per_hz_by_mhz": {k: c["slope_ua_per_hz"] for k, c in cells.items()},
        "i_idle_ma_by_mhz": {
            k: next((p["i_ma"] for p in c["i_mean_ma_by_rate"] if p["rate_hz"] == 0),
                    None) for k, c in cells.items()},
        "dwt_latency_us_p50_by_mhz": {
            k: c["dwt_latency_us_p50"] for k, c in cells.items()},
        "latency_scaling_vs_180": latency_scaling(cells),
        "gap2_ok_by_mhz": {k: c["gap2_ok"] for k, c in cells.items()},
        "gap2_worst_us_by_mhz": {k: c["gap2_worst_us"] for k, c in cells.items()},
        "gap2_verdict_by_mhz": {k: c.get("gap2_verdict") for k, c in cells.items()},
        "sysclk_reported_mhz_by_mhz": {
            k: c.get("hw_check", {}).get("sysclk_reported_mhz") for k, c in cells.items()},
        "acqmode_dyn_by_mhz": {
            k: {"succeeded": c["acqmode_dyn"]["succeeded"],
                "i_max_ma": c["acqmode_dyn"].get("i_max_ma")} for k, c in cells.items()},
        "tendance_energie": tendance,
        "tendance_rationale": raison,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    """Options du pilote, isolées de `main` pour être vérifiables sans banc."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sysclk-mhz", type=int, choices=sorted(PLL_BY_MHZ),
                        help="fréquence RÉELLEMENT flashée (build -DSYSCLK_MHZ)")
    parser.add_argument("--summary", action="store_true",
                        help="recalcule summary.json depuis les cellules existantes")
    parser.add_argument("--refit", action="store_true",
                        help="recalcule les grandeurs DÉRIVÉES des cellules déjà mesurées "
                             "(pente, r², énergie) sans carte ni sonde ; les points mesurés "
                             "ne sont pas touchés")
    parser.add_argument("--board-port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--skip-hw-check", action="store_true",
                        help="n'exige PAS que la carte confirme sa fréquence (la cellule "
                             "sort alors avec sysclk_reported_mhz = null et sa raison)")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto)")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--model", default="hdc",
                        help="modèle de la régression de cadence (celui dont le taux "
                             "d'occupation est mesurable : HDC par défaut)")
    parser.add_argument("--encoding", default="int8")
    parser.add_argument("--rates", type=float, nargs="+", default=[0, 50, 100, 200],
                        help="cadences de la régression (0 = repos)")
    parser.add_argument("--rate-hz", type=float, default=100.0,
                        help="cadence des flux de vérification protocole")
    parser.add_argument("--n-check", type=int, default=300,
                        help="échantillons du contrôle d'intégrité par modèle")
    parser.add_argument("--achieved-at-each-rate", action="store_true",
                        help="relever la cadence ATTEINTE à chaque point et non à la seule "
                             "borne haute (règle A7 : un flux de plus par cadence, "
                             "l'inférence du plafond ne comble alors que les trous)")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--dyn-duration", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.sysclk_mhz is not None:
        if not args.board_port:
            parser.error("--board-port est requis pour mesurer une fréquence")
        cell = measure_cell(args)
        path = args.out / f"{args.sysclk_mhz}.json"
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[freq] {args.sysclk_mhz} MHz → {path.name} "
              f"(Gap 2 {'OK' if cell['gap2_ok'] else 'KO'}, "
              f"pire {cell['gap2_worst_us']} µs)")

    if args.refit:
        for mhz, cell in sorted(refit(args.out).items(), key=lambda kv: int(kv[0])):
            energie = cell["energy_uj_per_inference"]
            rendu = (f"{energie:.1f} µJ" if isinstance(energie, (int, float))
                     else f"« {energie} »")
            print(f"[freq] {mhz:>3s} MHz réajusté : pente="
                  f"{cell.get('slope_ua_per_hz', float('nan')):+.3f} ± "
                  f"{cell.get('slope_std_ua_per_hz', float('nan')):.3f} µA/Hz, "
                  f"r²={cell.get('r2', float('nan')):.3f} → {rendu}")

    if args.summary or args.refit or args.sysclk_mhz is not None:
        summary = build_summary(args.out)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[freq] tendance : {summary['tendance_energie']} — "
              f"{summary['tendance_rationale']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
