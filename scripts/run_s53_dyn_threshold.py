"""
run_s53_dyn_threshold.py — S5303 / B6 : nature du plafond du mode dynamique de la sonde.

LA QUESTION

Le LPM01A annonce refuser l'acquisition dynamique « au-delà de 59 mA ». Les mesures du
Sprint 53 montrent que ce n'est pas la règle appliquée : à 45 MHz l'acquisition PASSE avec
un pic de 68,60 mA, à 90 MHz elle ÉCHOUE avec 69,58 mA — un milliampère d'écart, dix
milliampères au-dessus du seuil annoncé. Trois candidats peuvent déclencher l'arrêt : le
PIC instantané, la MOYENNE du courant, ou la DURÉE passée sous charge. Ce pilote produit
une matrice d'essais conçue pour les DÉCORRÉLER, et `src/evaluation/dyn_threshold.py` dit
ce qu'elle démontre — y compris « rien », si les candidats restent confondus.

TROIS AXES, ET CE QUE CHACUN SÉPARE

    * charge   — la cadence du flux fait varier la MOYENNE et le taux d'occupation ;
    * durée    — à charge identique, seule la durée demandée change : si un essai court
                 passe là où un long échoue, le déclencheur n'est pas instantané ;
    * échantillonnage — 10 kSPS contre 100 kSPS à charge et durée identiques : si le refus
                 suit la fréquence d'échantillonnage, ce n'est pas un plafond de courant
                 mais une limite de débit de décodage.

Pourquoi cela vaut la peine : le mode dynamique est la SEULE voie vers un profil temporel
par phase sans câblage (S5305, voie A), et la seule chose qu'on en sache aujourd'hui est
qu'il refuse « parfois ». Savoir ce qui le déclenche dit s'il est atteignable en ralentissant
l'horloge, en raccourcissant l'acquisition, ou pas du tout.

CE QUE CE PILOTE NE FAIT PAS : il ne flashe pas. Il mesure le binaire déjà en place et
CONTRÔLE la fréquence que la carte rapporte (`run_s53_freq_sweep.check_frequency`), qui
entre dans le nom du fichier de sortie.

Usage (après le flash d'un `-DSYSCLK_MHZ=<f>`, JP5 basculé sur la sonde) :
    python scripts/run_s53_dyn_threshold.py --sysclk-mhz 90 --board-port /dev/ttyACM0
    python scripts/run_s53_dyn_threshold.py --summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
fs = _load("run_s53_freq_sweep", ROOT / "scripts" / "run_s53_freq_sweep.py")
pp = _load("run_s53_phase_profile", ROOT / "scripts" / "run_s53_phase_profile.py")
from src.evaluation import dyn_threshold as dt   # noqa: E402

DEFAULT_OUT = ROOT / "experiments" / "exp_S53_dyn_threshold"
MA = 1000.0

#: Grille par défaut des trois axes. Elle est arrêtée AVANT la mesure : ajuster une grille
#: après avoir vu quels essais passent reviendrait à choisir sa conclusion.
DEFAULT_RATES = (0.0, 25.0, 50.0, 100.0, 200.0)
DEFAULT_DURATIONS = (0.05, 0.2, 1.0, 10.0)
DEFAULT_FREQS = ("10k", "100k")

#: Point de référence des axes « durée » et « échantillonnage » : la charge y est tenue.
REF_RATE_HZ = 100.0
REF_DURATION_S = 1.0
REF_FREQ = "100k"


def attempt(probe, voltage_mv: int, args, rate_hz: float, duration_s: float,
            freq: str) -> dict:
    """Un essai d'acquisition dynamique sous charge — succès ou échec, tel que relevé."""
    proc = (None if rate_hz <= 0 else
            pp.start_stream(args.model_flag, args.dataset, args.board_port,
                            rate_hz, duration_s, args.settle))
    essai = {"rate_hz": float(rate_hz), "duration_s": float(duration_s),
             "sampling_freq": freq, "fs_hz": lp.parse_freq_hz(freq)}
    try:
        if proc is not None:
            time.sleep(args.settle)
        samples, voltage_v, summary = lp.capture(
            probe, freq=freq, duration_s=duration_s, voltage_mv=voltage_mv,
            acqmode="dyn")
    except Exception as exc:      # LPM01AError ou flux échoué : les deux se consignent
        # `no_data` explicite : rien n'a été décodé, donc cet essai ne renseigne pas sur
        # le déclencheur et n'entre pas dans le verdict (la règle le déduit aussi du
        # compte, mais le drapeau rend l'enregistrement lisible sans elle).
        essai.update({"succeeded": False, "no_data": True, "n_samples": 0,
                      "n_samples_expected": int(lp.parse_freq_hz(freq) * duration_s),
                      "na_reason": f"acquisition refusée par la sonde : {exc}"})
        return essai
    finally:
        pp.stop_stream(proc)

    n_expected = int(lp.parse_freq_hz(freq) * duration_s)
    essai.update(dt.acquisition_outcome(int(samples.size), n_expected, summary))
    # Le COURANT d'un essai n'est retenu que si la trace se décode en courants plausibles :
    # après une réouverture de session, des octets résiduels produisent des valeurs
    # aberrantes (3 338 A relevés le 2026-09-08). L'essai reste compté — c'est l'arrêt qu'on
    # mesure — mais il n'entre plus dans les axes « pic » et « moyenne ».
    i_max_a = float(np.max(samples))
    fiable = dt.current_trustworthy(i_max_a)
    essai.update({
        "current_trustworthy": fiable,
        "i_mean_ma": (float(np.mean(samples)) * MA) if fiable else None,
        "i_max_ma": (i_max_a * MA) if fiable else None,
        "i_max_decoded_ma": i_max_a * MA,
        "voltage_v": float(voltage_v),
        # Temps effectivement acquis avant l'arrêt : à durée demandée constante, il dit
        # si la sonde tient plus ou moins longtemps selon la charge.
        "time_acquired_s": float(samples.size) / lp.parse_freq_hz(freq),
        "summary": summary.strip()[-200:],
    })
    return essai


def recover_probe(probe, port: str | None, voltage_mv: int):
    """Réouvre une session de sonde après un arrêt d'acquisition.

    Constat de banc MESURÉ 2026-09-08 : une fois l'acquisition dynamique interrompue en
    surintensité, la sonde refuse **toutes** les commandes suivantes de la session
    (« Commande refusée par la sonde : 'output current' »). Sans réouverture, seul le
    PREMIER essai d'une matrice est informatif et les huit autres rendent zéro échantillon
    — ce qui se lit comme huit échecs alors qu'aucun n'a été tenté.

    La réouverture n'est faite qu'APRÈS un échec : une session saine n'est pas perturbée,
    et surtout la règle du préchauffage est préservée. Une session neuve rend en effet sa
    première acquisition biaisée (~+8 mA, Sprint 50) : l'essai qui suit une réouverture est
    donc marqué, pour que son courant moyen ne soit pas comparé aux autres sans réserve.
    """
    try:
        probe.release()
    except Exception:
        pass
    neuve = lp.PowerShield(port)
    neuve.take_control()
    # `take_control` vide déjà le tampon, mais la sonde continue d'émettre après un arrêt :
    # sans une seconde purge plus patiente, les octets restants se décodent en échantillons
    # aberrants à l'acquisition suivante (mesuré le 2026-09-08).
    neuve._drain(quiet_s=dt.RECOVERY_QUIET_S, max_s=dt.RECOVERY_MAX_S)
    # Une acquisition de REBUT referme la session proprement. Mesuré 2026-09-08 : sans elle,
    # la première acquisition d'une session rouverte rend systématiquement ZÉRO échantillon
    # — les essais alternaient alors « abouti / aucune donnée » au rythme des réouvertures,
    # et non au rythme des conditions testées. C'est aussi ce qu'exige la règle du
    # préchauffage (Sprint 50) : la première acquisition d'une session n'est jamais retenue.
    try:
        lp.warmup(neuve, voltage_mv, duration_s=dt.RECOVERY_WARMUP_S)
    except Exception:
        # Un rebut qui échoue n'est pas une mesure perdue : l'essai suivant sera de toute
        # façon jugé sur ce qu'il rend.
        pass
    return neuve


def build_grid(args) -> list[tuple[float, float, str]]:
    """Matrice d'essais : un axe à la fois, les deux autres tenus au point de référence."""
    grille = [(r, args.ref_duration, args.ref_freq) for r in args.rates]
    grille += [(args.ref_rate, d, args.ref_freq) for d in args.durations
               if (args.ref_rate, d, args.ref_freq) not in grille]
    grille += [(args.ref_rate, args.ref_duration, f) for f in args.freqs
               if (args.ref_rate, args.ref_duration, f) not in grille]
    return grille


def measure(args) -> dict:
    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    probe = lp.PowerShield(args.port)
    essais: list[dict] = []
    try:
        probe.take_control()
        warmups = [lp.warmup(probe, voltage_mv)
                   for _ in range(max(1, args.warmup_repeats))]
        for k, w in enumerate(warmups):
            print(f"[dyn] préchauffage {k} écarté : {w * MA:.3f} mA (non retenu)")

        hw_check = fs.check_frequency(args)

        recuperation_precedente = False
        for rate_hz, duration_s, freq in build_grid(args):
            # Une acquisition de REBUT précède CHAQUE essai. Mesuré 2026-09-08 : deux
            # acquisitions dynamiques enchaînées sans intercalaire rendent zéro échantillon
            # à la seconde. Les essais « aboutis » des premières séries étaient précisément
            # ceux qu'un préchauffage ou une réouverture précédait — l'alternance
            # succès / aucune-donnée suivait donc l'HISTORIQUE de la session, pas les
            # conditions testées. La rendre uniforme est aussi la seule façon de comparer
            # les essais entre eux : ils partagent désormais le même passé immédiat.
            try:
                lp.warmup(probe, voltage_mv, duration_s=dt.RECOVERY_WARMUP_S)
            except Exception:
                probe = recover_probe(probe, args.port, voltage_mv)
                recuperation_precedente = True
            essai = attempt(probe, voltage_mv, args, rate_hz, duration_s, freq)
            essai["after_probe_recovery"] = recuperation_precedente
            essais.append(essai)
            recuperation_precedente = False
            if not essai["succeeded"]:
                probe = recover_probe(probe, args.port, voltage_mv)
                recuperation_precedente = True
                print("[dyn] sonde réouverte après l'arrêt (session refusant les "
                      "commandes suivantes)")
            print(f"[dyn] {rate_hz:6.1f} Hz  {duration_s:6.2f} s  {freq:>5s} : "
                  f"{'ABOUTI ' if essai['succeeded'] else 'ARRÊTÉ '}"
                  f"{essai.get('n_samples', 0)}/{essai.get('n_samples_expected', 0)} éch."
                  + (f"  I_max={essai['i_max_ma']:.1f} mA  "
                     f"I_moy={essai['i_mean_ma']:.1f} mA"
                     if essai.get("i_max_ma") is not None else
                     (f"  trace non décodable (I_max décodé "
                      f"{essai['i_max_decoded_ma']:.0f} mA)"
                      if "i_max_decoded_ma" in essai else "")))
    finally:
        probe.release()

    cellule = {
        "sysclk_mhz": args.sysclk_mhz,
        "firmware_build": (f"-DSYSCLK_MHZ={args.sysclk_mhz}" if args.sysclk_mhz is not None
                           else "défaut"),
        "hw_check": hw_check,
        "model_flag": args.model_flag,
        "dataset": args.dataset,
        "reference_point": {"rate_hz": args.ref_rate, "duration_s": args.ref_duration,
                            "sampling_freq": args.ref_freq},
        "attempts": essais,
        "classification": dt.classify(essais),
        "abort_time_stability": dt.abort_time_stability(essais),
        "settle_s": float(args.settle),
        "warmup_discarded_a": [float(w) for w in warmups],
        "source": "lpm01a_dyn_attempts",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return cellule


def build_summary(out_dir: Path) -> dict:
    """Classification par fréquence, plus celle de TOUS les essais réunis.

    Réunir les essais de plusieurs fréquences ajoute la variable la plus utile — le
    courant de base change de 18 à 40 mA d'une fréquence à l'autre — mais mélange aussi
    des séances : la réserve est portée dans le JSON, pas laissée au lecteur.
    """
    cellules = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        cellules[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    tous = [e for c in cellules.values() for e in c.get("attempts", [])]
    return {
        "question": ("Qu'est-ce qui déclenche l'arrêt de l'acquisition dynamique — le pic, "
                     "la moyenne, ou la durée ?"),
        "by_sysclk": {nom: c.get("classification") for nom, c in cellules.items()},
        "pooled": dt.classify(tous) if tous else None,
        "pooled_abort_time": dt.abort_time_stability(tous) if tous else None,
        "pooled_reserve": (
            "les essais réunis proviennent de séances distinctes (une par binaire) : le "
            "courant de base y diffère, ce qui est précisément ce qui décorrèle la "
            "moyenne du pic, mais interdit d'interpréter un écart de quelques dixièmes de "
            "milliampère entre essais de séances différentes."),
        "n_attempts": len(tous),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sysclk-mhz", type=int, choices=sorted(fs.PLL_BY_MHZ),
                        help="fréquence RÉELLEMENT flashée (contrôlée contre la carte)")
    parser.add_argument("--recompute", action="store_true",
                        help="recalcule les verdicts des cellules déjà écrites depuis "
                             "leurs essais, sans carte ni sonde (les essais mesurés ne "
                             "sont jamais touchés)")
    parser.add_argument("--summary", action="store_true",
                        help="recalcule summary.json depuis les cellules existantes")
    parser.add_argument("--board-port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--skip-hw-check", action="store_true")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto)")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--model-flag", default="hdc-int8",
                        help="drapeau --model du flux : le modèle le plus long tient la "
                             "charge la plus élevée à cadence donnée")
    parser.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--durations", type=float, nargs="+",
                        default=list(DEFAULT_DURATIONS))
    parser.add_argument("--freqs", nargs="+", default=list(DEFAULT_FREQS))
    parser.add_argument("--ref-rate", type=float, default=REF_RATE_HZ)
    parser.add_argument("--ref-duration", type=float, default=REF_DURATION_S)
    parser.add_argument("--ref-freq", default=REF_FREQ)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    args = parser.parse_args(argv)

    if args.sysclk_mhz is not None:
        if not args.board_port:
            parser.error("--board-port est requis pour mesurer")
        cellule = measure(args)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.sysclk_mhz}.json"
        path.write_text(json.dumps(cellule, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[dyn] {args.sysclk_mhz} MHz → {path.name} : "
              f"{cellule['classification']['verdict']}")
        print(f"[dyn] {cellule['classification']['rationale']}")

    if args.recompute:
        for path in sorted(args.out.glob("*.json")):
            if path.name == "summary.json":
                continue
            cell = json.loads(path.read_text(encoding="utf-8"))
            if "attempts" not in cell:
                continue
            cell["classification"] = dt.classify(cell["attempts"])
            cell["abort_time_stability"] = dt.abort_time_stability(cell["attempts"])
            path.write_text(json.dumps(cell, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            print(f"[dyn] {path.stem} recalculé : {cell['classification']['verdict']}")

    if args.summary or args.recompute or args.sysclk_mhz is not None:
        summary = build_summary(args.out)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if summary["pooled"]:
            print(f"[dyn] essais réunis ({summary['n_attempts']}) : "
                  f"{summary['pooled']['verdict']} — {summary['pooled']['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
