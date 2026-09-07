"""run_s53_rate_sweep.py — S5304 : µJ par inférence par régression de cadence.

CE QUE CE PILOTE MESURE, ET POURQUOI IL EXISTE :

Le courant moyen à cadence unique du Sprint 50 ne donne pas de µJ par inférence : il
faudrait soustraire un repos, et le repos mesuré est plus haut que la charge (anomalie de
−8 mA, S5301). Ce pilote balaye la CADENCE et régresse

    I_moy(rate) = I_base + pente · rate      →      E_par_inférence = pente · V

`I_base` absorbe intégralement le repos, la scrutation active de l'attente UART et le
trafic de l'hôte : rien n'est soustrait, donc rien ne dépend d'une référence douteuse. La
méthode reste valide quelles que soient les conclusions de S5301, S5302 et S5303 — c'est
pourquoi elle est placée tôt dans l'ordre de banc, avant tout reflash.

La règle de calcul et de décision vit dans `src/evaluation/rate_regression.py` (précédent
`counterbalance.py`, S5301) : ce fichier n'acquiert que des courants, il n'arbitre rien.

CE QUE LA PENTE CONTIENT — et comment on l'en sépare : le coût d'une trame UART (commun à
toutes les cellules) PLUS le coût de calcul. La régression de second niveau
`pente vs latence DWT`, écrite dans `slope_vs_latency.json`, sépare les deux (µJ par µs de
calcul et µJ par trame). Le pilote de banc du 2026-08-05 l'a validée : HDC INT8 233,9 µJ
contre un témoin Mahalanobis à 60,6 µJ, dont le calcul ne vaut que ~0,4 µJ.

DEUX LEÇONS DE BANC CÂBLÉES ICI :
    1. **Ordre randomisé** (S5301) : l'ordre des acquisitions ne doit jamais être corrélé
       à la condition. Le pilote tire l'ordre du produit (cellule × cadence × répétition)
       et consigne le rang `session_index` de chaque acquisition.
    2. **Saturation silencieuse** : au-delà de ~209 inf/s le flux tourne moins vite que la
       consigne sans perdre de trame ni lever de CRC. Chaque cellule mesure donc sa cadence
       ATTEINTE ; les points saturés sont écartés de l'ajustement mais restent écrits.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : une régression non concluante écrit
``"à mesurer"`` et sa raison chiffrée, jamais un zéro ni une pente négative.

Usage :
    # Balayage complet (une session, carte + sonde) :
    python scripts/run_s53_rate_sweep.py --board-port /dev/serial/by-id/…STLink… \\
        --rates 0 10 25 50 100 150 200 --repeats 3 --window 10

    # Cellule à firmware dédié, après flash -DMAHA_INT8 :
    python scripts/run_s53_rate_sweep.py --only maha_int8 --board-port /dev/…

    # Sans matériel : recalcule pentes, énergies et agrégats depuis les points écrits.
    python scripts/run_s53_rate_sweep.py --refit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
#: Pilote de référence du banc : `stream_command`, `measure_current`, `STREAM_MODEL` et
#: `BUILD_SPECIFIC` y sont déjà éprouvés — ils ne sont pas réécrits ici.
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")

A_MESURER = rr.A_MESURER
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_rate_sweep"
MA = rr.MA_PER_A

#: Cadences par défaut. La borne haute est à 200 Hz et NON à 400 : le plafond n'est pas le
#: calcul mais le transport (32 B de requête + 23 B de réponse V3 à 115200 bauds ≈ 209
#: inf/s, débit maximal relevé 140–218 inf/s selon les cellules). Au-delà, l'axe des
#: cadences saturerait en silence et la pente serait sous-estimée.
DEFAULT_RATES = (0, 10, 25, 50, 100, 150, 200)

#: Cadence des flux de contrôle (intégrité + latence DWT), en Hz. Loin du plafond de
#: transport, pour que le contrôle ne mesure pas la saturation au lieu de la latence.
CHECK_RATE_HZ = 100.0


def stream_once(probe, voltage_mv: int, model_flag: str, dataset: str, port: str,
                n_samples: int, rate_hz: float) -> dict:
    """Un flux de contrôle, exécuté dans une acquisition maintenue.

    Le lancement et le diagnostic des trois modes d'échec vivent dans le pilote de référence
    du banc (`run_s50_board_current.stream_in_acquisition`) : deux copies divergentes du
    même garde-fou finiraient par ne plus rapporter les mêmes échecs.
    """
    cmd = rc.stream_command(model_flag, dataset, port, n_samples, rate_hz)
    return rc.stream_in_acquisition(probe, voltage_mv, cmd, model_flag)


def parse_cells(only: str | None, parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """Cellules à mesurer — les cellules à firmware dédié restent explicites.

    `maha_int8` se choisit à la COMPILATION (`-DMAHA_INT8`) : la mesurer sur le build par
    défaut l'écrirait sous un nom qui ment sur son contenu, exactement le bug du drapeau
    TinyOL du Sprint 52. Elle n'entre donc dans le balayage que par `--only`.
    """
    if only:
        cells = []
        for token in only.split(","):
            model, _, encoding = token.strip().rpartition("_")
            if (model, encoding) not in rc.STREAM_MODEL:
                parser.error(f"cellule inconnue : {token.strip()}")
            cells.append((model, encoding))
        return cells
    cells = [c for c in rc.STREAM_MODEL if c not in rc.BUILD_SPECIFIC]
    for (m, e), flag in rc.BUILD_SPECIFIC.items():
        print(f"[cadence] {m}_{e} ignorée : exige un build {flag} "
              f"(la mesurer par « --only {m}_{e} » après flash)")
    return cells


def build_schedule(cells: list[tuple[str, str]], rates: list[float], repeats: int,
                   shuffle: bool, seed: int) -> list[tuple[tuple[str, str], float, int]]:
    """Plan d'acquisition (cellule, cadence, répétition), ordre randomisé par défaut.

    La leçon de S5301 : le Sprint 50 mesurait le repos en tête de session et les cellules
    ensuite, si bien que l'ordre était confondu avec la condition et qu'une dérive
    d'établissement se lisait comme un effet. Ici, une dérive lente se répartit sur toutes
    les conditions au lieu de s'aligner sur l'axe des cadences.
    """
    plan = [(cell, float(rate), k)
            for cell in cells for rate in rates for k in range(repeats)]
    if shuffle:
        random.Random(seed).shuffle(plan)
    return plan


def measure(args, cells: list[tuple[str, str]]) -> dict[str, dict]:
    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    voltage_v = float(calib.get("supply_voltage_v", voltage_mv / 1000))
    rates = [float(r) for r in args.rates]
    n_samples = int(max(rates) * (args.window + args.settle + 4.0)) or args.n_check

    probe = lp.PowerShield(args.port)
    runs: dict[tuple[tuple[str, str], float], list[float]] = {}
    indices: dict[tuple[tuple[str, str], float], list[int]] = {}
    checks: dict[str, dict] = {}
    sequence: list[dict] = []
    try:
        probe.take_control()

        # 1. Rebut de préchauffage — la 1re acquisition d'une session est biaisée (+8 mA
        #    relevés au Sprint 50). Sa valeur est consignée, jamais retenue.
        warmups = [lp.warmup(probe, voltage_mv)
                   for _ in range(max(1, args.warmup_repeats))]
        for k, w in enumerate(warmups):
            print(f"[cadence] préchauffage {k} écarté : {w * MA:.3f} mA (non retenu)")

        # 2. Contrôle par cellule : intégrité du flux ET latence DWT, mesurés EN SESSION.
        #    La latence est le second axe de la régression de niveau 2 : la recopier d'un
        #    sprint antérieur reviendrait à régresser sur un chiffre non mesuré ici.
        for cell in cells:
            flag = rc.STREAM_MODEL[cell]
            res = stream_once(probe, voltage_mv, flag, args.dataset,
                              args.board_port, args.n_check, CHECK_RATE_HZ)
            name = f"{cell[0]}_{cell[1]}"
            checks[name] = {
                "model_flag": flag,
                "n_samples_expected": int(args.n_check),
                "n_samples_received": res.get("n_samples"),
                "samples_lost": int(args.n_check) - int(res.get("n_samples", 0)),
                "crc_errors": res.get("crc_errors"),
                "latency_p50_us": res.get("latency_p50_us"),
                "latency_p99_us": res.get("latency_p99_us"),
                "achieved_rate_hz": res.get("achieved_rate_hz"),
                "check_rate_hz": CHECK_RATE_HZ,
            }
            print(f"[cadence] contrôle {name:12s} : {res.get('n_samples')}/{args.n_check} "
                  f"éch., P50={res.get('latency_p50_us')} µs, "
                  f"CRC={res.get('crc_errors')}")

        # 3. Balayage proprement dit, dans un ordre décorrélé de la condition.
        plan = build_schedule(cells, rates, args.repeats, args.shuffle_order, args.seed)
        for position, (cell, rate, k) in enumerate(plan):
            cmd = (None if rate <= 0 else
                   rc.stream_command(rc.STREAM_MODEL[cell], args.dataset,
                                     args.board_port, n_samples, rate))
            i_run, voltage_v = rc.measure_current(
                probe, args.window, voltage_mv, cmd, args.settle if cmd else 0.0)
            runs.setdefault((cell, rate), []).append(i_run)
            indices.setdefault((cell, rate), []).append(position)
            sequence.append({"session_index": position,
                             "cell": f"{cell[0]}_{cell[1]}",
                             "rate_hz": rate, "repeat": k, "i_a": float(i_run)})
            print(f"[cadence] #{position:03d} {cell[0]}_{cell[1]:5s} "
                  f"{rate:6.1f} Hz : {i_run * MA:7.3f} mA")

        # 4. Cadence ATTEINTE : c'est ce qui définit la saturation, et elle ne se lit ni
        #    dans les CRC ni dans les trames perdues. Par défaut, seule la borne haute est
        #    relevée (c'est là que le transport plafonne) ; `--achieved-at-each-rate` la
        #    relève à CHAQUE cadence, ce qui remplace l'inférence par une mesure au prix
        #    d'un flux supplémentaire par point.
        rate_max = max(rates)
        mesurees = ([r for r in rates if r > 0] if args.achieved_at_each_rate
                    else ([rate_max] if rate_max > 0 else []))
        achieved: dict[tuple[str, float], float | None] = {}
        for rate in mesurees:
            for cell in cells:
                res = stream_once(probe, voltage_mv, rc.STREAM_MODEL[cell], args.dataset,
                                  args.board_port, args.n_check, rate)
                achieved[(f"{cell[0]}_{cell[1]}", rate)] = res.get("achieved_rate_hz")
                print(f"[cadence] atteinte {cell[0]}_{cell[1]:5s} : consigne {rate:.0f} Hz "
                      f"→ atteint {res.get('achieved_rate_hz')} Hz")
    finally:
        probe.release()

    cells_json: dict[str, dict] = {}
    for cell in cells:
        name = f"{cell[0]}_{cell[1]}"
        points = []
        for rate in rates:
            values = runs.get((cell, rate), [])
            if not values:
                continue
            point = {
                "rate_hz": float(rate),
                "i_mean_a": float(np.mean(values)),
                "i_std_a": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_repeats": len(values),
                "session_index": indices[(cell, rate)],
            }
            # A7 — `achieved_rate_hz` est un champ de MESURE : il ne se remplit pas avec
            # la consigne. Une cadence non re-streamée reste donc `None`, ce que
            # `saturation_rate_hz` traite déjà comme « ne prouve rien » (elle ignore les
            # points sans cadence atteinte) : l'ajustement est inchangé, seule la
            # traçabilité l'est.
            valeur = achieved.get((name, rate))
            if rate > 0:
                point["achieved_rate_hz"] = (float(valeur) if valeur is not None else None)
                point["achieved_rate_source"] = ("mesuré" if valeur is not None
                                                 else "non mesuré")
            points.append(point)
        cells_json[name] = finalize_cell(cell[0], cell[1], points, voltage_v,
                                         checks.get(name, {}), args)
        cells_json[name]["session"] = {
            "sequence": [s for s in sequence if s["cell"] == name],
            "shuffled": bool(args.shuffle_order),
            "seed": int(args.seed),
            "warmup_discarded_a": [float(w) for w in warmups],
        }
    return cells_json


def finalize_cell(model: str, encoding: str, points: list[dict], tension_v: float,
                  check: dict, args) -> dict:
    """Assemble une cellule : points MESURÉS + grandeurs CALCULÉES par `rate_regression`."""
    latency = check.get("latency_p50_us")
    derived = rr.fit_cell(points, tension_v, latency)
    cell = {
        "model": model,
        "encoding": encoding,
        "dataset": args.dataset,
        "points": points,
        "protocol_check": check,
        "window_s": float(args.window),
        "settle_s": float(args.settle),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "lpm01a_current",
    }
    cell.update(derived)
    return cell


def refit(out_dir: Path) -> dict[str, dict]:
    """Recalcule les grandeurs dérivées depuis les `points` déjà écrits (hors banc).

    Deux usages : reprendre une session après coup sans remesurer, et vérifier la chaîne
    de calcul sans carte ni sonde. Les points mesurés ne sont jamais touchés.
    """
    cells: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"slope_vs_latency.json", "summary.json"}:
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        if "points" not in cell or "model" not in cell:
            continue   # pilote historique ou artefact tiers : lecture seule, jamais réécrit
        latency = cell.get("protocol_check", {}).get("latency_p50_us")
        # A7 — une cellule écrite avant le 2026-09-07 porte la CONSIGNE dans son champ de
        # cadence atteinte : la provenance est rétablie avant tout recalcul.
        rr.normalize_achieved_source(cell["points"])
        cell.update(rr.fit_cell(cell["points"], cell["tension_v"], latency))
        cells[f"{cell['model']}_{cell['encoding']}"] = cell
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
    return cells


def write_outputs(cells: dict[str, dict], out_dir: Path) -> dict:
    """Écrit les cellules, la régression de second niveau et le verdict Gap 3."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cell in cells.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")

    second = rr.slope_vs_latency(cells)
    second["timestamp"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "slope_vs_latency.json").write_text(
        json.dumps(second, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "method": rr.METHOD,
        "method_note": (
            "estimateur PAR RÉGRESSION — à ne jamais fusionner avec l'estimateur par "
            "delta (S5302) ni avec l'intégration du profil par phase (S5305) : leur "
            "comparaison est elle-même un résultat."
        ),
        "cells": {name: {
            "energy_uj_per_inference": cell.get("energy_uj_per_inference"),
            "energy_uncertainty_uj": cell.get("energy_uncertainty_uj"),
            "energy_na_reason": cell.get("energy_na_reason"),
            "slope_ua_per_hz": cell.get("slope_ua_per_hz"),
            "slope_std_ua_per_hz": cell.get("slope_std_ua_per_hz"),
            "r2": cell.get("r2"),
            "intercept_ma": cell.get("intercept_ma"),
            "saturation_rate_hz": cell.get("saturation_rate_hz"),
            "duty_cycle_at_max_rate": cell.get("duty_cycle_at_max_rate"),
            "dwt_latency_us_p50": cell.get("dwt_latency_us_p50"),
            "crc_errors": cell.get("protocol_check", {}).get("crc_errors"),
        } for name, cell in cells.items()},
        "slope_vs_latency": second,
        "gap3_energy_verdict": rr.gap3_energy_verdict(cells),
        "coherence_check": rr.coherence_check(cells, second),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--board-port", default=None,
                        help="port de la CARTE (ST-Link VCP) — requis pour mesurer")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto par défaut)")
    parser.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES),
                        help="cadences du balayage en Hz (0 = repos)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="répétitions par point → écart-type, donc barres d'erreur")
    parser.add_argument("--window", type=float, default=10.0, help="durée de fenêtre (s)")
    parser.add_argument("--settle", type=float, default=2.0,
                        help="délai d'établissement du flux avant la fenêtre (s)")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--only", default=None,
                        help="cellules à mesurer, séparées par des virgules "
                             "(ex. « maha_int8 » après flash -DMAHA_INT8)")
    parser.add_argument("--achieved-at-each-rate", action="store_true",
                        help="relève la cadence ATTEINTE à chaque cadence, et non à la "
                             "seule borne haute (un flux de contrôle par point : remplace "
                             "l'inférence de plafond par une mesure)")
    parser.add_argument("--n-check", type=int, default=300,
                        help="échantillons du contrôle d'intégrité par cellule")
    parser.add_argument("--warmup-repeats", type=int, default=1,
                        help="acquisitions de préchauffage JETÉES en début de session")
    parser.add_argument("--no-shuffle-order", dest="shuffle_order", action="store_false",
                        help="mesure dans l'ordre nominal (déconseillé : l'ordre "
                             "redevient corrélé à la condition, cf. S5301)")
    parser.add_argument("--seed", type=int, default=42,
                        help="graine de l'ordre randomisé (reproductibilité du plan)")
    parser.add_argument("--refit", action="store_true",
                        help="recalcule les grandeurs dérivées depuis les points écrits, "
                             "sans matériel")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    parser.set_defaults(shuffle_order=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.refit:
        cells = refit(args.out)
        if not cells:
            print(f"[cadence] aucune cellule à réajuster dans {args.out} — rien n'est "
                  f"écrit (un JSON sans points ne se recalcule pas).")
            return 0
    else:
        if not args.board_port:
            parser.error("--board-port est requis pour mesurer (ou utiliser --refit)")
        cells = measure(args, parse_cells(args.only, parser))

    summary = write_outputs(cells, args.out)
    for name, bloc in summary["cells"].items():
        energie = bloc["energy_uj_per_inference"]
        rendu = (f"{energie:.1f} ± {bloc['energy_uncertainty_uj']:.1f} µJ"
                 if isinstance(energie, (int, float)) else str(energie))
        print(f"[cadence] {name:12s} : {rendu}  (r²={bloc['r2']:.3f}, "
              f"latence {bloc['dwt_latency_us_p50']} µs)")
    second = summary["slope_vs_latency"]
    print(f"[cadence] second niveau : {second['uj_per_us_compute']} µJ/µs de calcul, "
          f"{second['uj_per_uart_frame']} µJ/trame UART")
    print(f"[cadence] Gap 3 énergie : {summary['gap3_energy_verdict']['verdict']} — "
          f"{summary['gap3_energy_verdict']['rationale']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
