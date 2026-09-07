#!/usr/bin/env python3
"""
Énergie des politiques de mise à jour P0–P3 du gate autonome (S5306).

Le Sprint 38 a mesuré sur carte réelle que le gate de nouveauté embarqué économise ~97 %
des mises à jour et 159–169 µs par échantillon, pour +300 B de RAM, à F1 préservé. Ce
résultat n'existe qu'en LATENCE et en RAM. Ce pilote le convertit en mA, en µJ et en heures
d'autonomie — et c'est la seule voie du projet vers `energy_uj_per_update`, N/A depuis le
Sprint 50 avec la raison « campagne d'inférence seule ».

MÉTHODE (celle de S5304, réutilisée sans être réécrite) : régression
`I_moy(rate) = I_base + pente · rate`. `I_base` absorbe le repos, la scrutation active de
l'attente UART et l'anomalie de S5301 ; seule la pente porte le coût marginal. P0
(`frozen`) et P1 (`always`) partagent le MÊME binaire, la MÊME trame UART et la MÊME
séance : leur différence de pente isole EXACTEMENT le coût d'une mise à jour CL.

CE QUE CE PILOTE NE FAIT PAS : il ne flashe pas pendant qu'il mesure. Le flash sous
alimentation sonde est instable (contrainte de banc mesurée, S5008) : il se fait sur
alimentation USB, JP5 en place, puis le cavalier est basculé à la main. D'où deux phases
séparées, `--prepare` (sonde débranchée) et `--measure` (sonde en contrôle).

GARDE ANTI-MISLABEL : le binaire déclaré au banc (`--build`) doit correspondre au
manifeste écrit par `--prepare`. Sans cela, une politique gated mesurée sur le binaire par
défaut serait écrite sous un nom qui ment sur son contenu — exactement le bug du drapeau
TinyOL du Sprint 52, et le motif du `hw_check` de `run_s53_freq_sweep.py`.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : ce pilote n'acquiert que des courants et
n'arbitre rien. Tous les seuils, différences et verdicts viennent de
`src/evaluation/policy_energy.py`, testés hors banc.

Usage :

    # 1. Sonde DÉBRANCHÉE, JP5 en place — construit et flashe le binaire d'une politique
    python scripts/run_s53_policy_energy.py --prepare --policy frozen --dataset monitoring

    # 2. Cavalier basculé, sonde en place — mesure les politiques portées par ce binaire
    python scripts/run_s53_policy_energy.py --measure --build default \\
           --policies frozen,always --dataset monitoring --board-port /dev/ttyACM0

    # 3. Sans matériel — recalcule pentes, énergies et agrégats depuis les points écrits
    python scripts/run_s53_policy_energy.py --refit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
#: Pilote de référence du banc : `measure_current`, `stream_command` et `STREAM_MODEL` y
#: sont déjà éprouvés — ils ne sont pas réécrits ici.
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")
pe = _load("policy_energy", ROOT / "src" / "evaluation" / "policy_energy.py")

A_MESURER = pe.A_MESURER
MA = rr.MA_PER_A
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_policy_energy"
MANIFEST_NAME = "firmware_state.json"

#: Cadences par défaut — identiques au balayage S5304, pour que les deux campagnes soient
#: superposables. La borne haute reste sous le plafond de transport UART (~209 inf/s).
DEFAULT_RATES = (0, 10, 25, 50, 100, 150, 200)

#: Cadences du témoin de session : trois points suffisent à une pente (le minimum exigé
#: par `weighted_linear_fit`), et le témoin n'a pas à être aussi précis que les cellules —
#: il doit seulement dire si la séance a dérivé.
ANCHOR_RATES = (0, 100, 200)

#: Cadence des flux de contrôle (intégrité, latence DWT, compteur du gate), en Hz.
CHECK_RATE_HZ = 100.0

#: Drapeau UART du modèle EWC — toutes les politiques passent par la tête EWC.
EWC_MODEL_FLAG = "ewc"

#: Modèle du témoin de session : son chemin d'exécution est inchangé par
#: `-DEWC_AUTO_UPDATE`, ce qui en fait une référence valide d'une séance à l'autre.
ANCHOR_MODEL_FLAG = "mahalanobis"


# ── Commande de flux ─────────────────────────────────────────────────────────

def policy_stream_command(policy: str, dataset: str, port: str,
                          n_samples: int, rate_hz: float) -> list[str]:
    """Commande `sensor_stream.py` d'une politique.

    `rc.stream_command` ne porte NI `--update` NI `--condition` : deux manques qui
    changeraient la mesure sans prévenir.
        - `--update` est ce qui distingue P1 de P0 sur le MÊME binaire (miroir de
          `run_sprint38_board.py:209`, `request_update = (policy == "always")`). Les
          politiques gated streament sans lui : le firmware décide seul.
        - `--condition 5feat` garantit que le flux consomme les MÊMES colonnes que le
          Sprint 38 (source unique `load_condition_arrays`, S3508) — sans quoi la
          comparaison croisée porterait sur deux entrées différentes.
    """
    cmd = rc.stream_command(EWC_MODEL_FLAG, dataset, port, n_samples, rate_hz)
    cmd += ["--condition", pe.CONDITION]
    if pe.UPDATE_FLAG_BY_POLICY[policy]:
        cmd += ["--update"]
    return cmd


def stream_once(probe, voltage_mv: int, cmd: list[str], label: str) -> dict:
    """Un flux complet (avec ses échantillons dumpés) dans une acquisition maintenue.

    Le lancement et le diagnostic des trois modes d'échec vivent dans le pilote de référence
    du banc (`run_s50_board_current.stream_in_acquisition`) : deux copies divergentes du
    même garde-fou finiraient par ne plus rapporter les mêmes échecs.
    """
    return rc.stream_in_acquisition(probe, voltage_mv, list(cmd) + ["--dump-samples"], label)


def gate_counters(stream_json: dict) -> dict:
    """Compteur de mises à jour et verdicts du gate, lus dans les échantillons dumpés.

    Sous `-DEWC_AUTO_UPDATE`, le firmware réinterprète deux champs du snapshot V3 : `auroc`
    porte le verdict (0 NORMAL / 1 FAULT / 2 DRIFT) et `forgetting` le compteur CUMULÉ de
    mises à jour (cf. `run_sprint38_board.py:313-337`). Le compteur se lit donc sur le
    DERNIER échantillon, jamais en sommant.

    Retourne des `None` honnêtes si le dump ne porte pas ces champs : sur le binaire par
    défaut, ils portent de vraies métriques et ne signifient rien pour le gate.
    """
    samples = stream_json.get("samples") or []
    forgetting = [s["forgetting"] for s in samples if "forgetting" in s]
    verdicts = [int(round(float(s["auroc"]))) for s in samples if "auroc" in s]
    if not forgetting:
        return {"n_updates": None, "update_rate": None, "verdict_counts": None,
                "na_reason": "compteur du gate absent du flux : sur le binaire par défaut, "
                             "les slots `auroc`/`forgetting` portent de vraies métriques."}
    n_updates = int(round(float(forgetting[-1])))
    monotone = all(b >= a for a, b in zip(forgetting, forgetting[1:]))
    counts: dict[str, int] = {}
    for code in verdicts:
        counts[str(code)] = counts.get(str(code), 0) + 1
    return {
        "n_updates": n_updates,
        "update_rate": (n_updates / len(samples)) if samples else None,
        "verdict_counts": counts,
        "counter_monotonic": monotone,
        "plausible_gate_build": monotone and set(counts).issubset({"0", "1", "2"}),
    }


# ── Phase 1 : préparation (build + flash), sonde DÉBRANCHÉE ─────────────────

def prepare(args) -> dict:
    """Entraîne/exporte/compile/flashe le binaire d'une politique, puis écrit le manifeste.

    Réutilise SANS LES RÉÉCRIRE `run_sprint38_board.build_and_flash` (P0/P1) et
    `build_and_flash_gated` (P2/P3, qui refait l'enrôlement Maha welford — miroir exact du
    PC — et exporte les seuils du gate). Les checkpoints PC du Sprint 38 sont réutilisés
    tels quels : ré-entraîner produirait un autre modèle, donc une autre énergie, et
    casserait la comparaison croisée.
    """
    os.chdir(ROOT)   # les pilotes S38 utilisent des chemins relatifs au dépôt
    s38 = _load("run_sprint38_board", ROOT / "scripts" / "run_sprint38_board.py")
    from src.evaluation.feature_conditions import load_condition_arrays

    policy, dataset = args.policy, args.dataset
    pc_dir = ROOT / "experiments" / f"exp_S38_PC_{policy}_{dataset}_{pe.INIT_MODE}"
    pc_ckpt = pc_dir / "checkpoints" / "ewc_head.pt"
    if not pc_ckpt.is_file():
        raise SystemExit(
            f"checkpoint PC absent : {pc_ckpt}\n"
            f"Le Sprint 38 doit avoir produit la référence PC de cette cellule "
            f"(scripts/run_sprint38_pc.py) — rien n'est ré-entraîné ici."
        )

    X, y, idx, names = load_condition_arrays(dataset, pe.CONDITION, "ewc", seed=args.seed)
    k = len(idx)
    exp_dir = args.out / f"build_{policy}_{dataset}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if policy in pe.GATED_POLICIES:
        drift_json = pc_dir / "drift_thresholds.json"
        if not drift_json.is_file():
            raise SystemExit(
                f"seuils du gate absents : {drift_json}\n"
                f"Sans eux, le gate embarqué ne déciderait pas comme le PC et la parité "
                f"du Sprint 38 serait rompue."
            )
        bss = s38.build_and_flash_gated(
            k, X, y, args.n_enrollment, pc_ckpt, drift_json, exp_dir,
            pseudo=(policy == "gated_pseudolabel"))
    else:
        bss = s38.build_and_flash(k, X, pc_ckpt, exp_dir)

    manifest = {
        "build": pe.BUILD_NAME_BY_POLICY[policy],
        "extra_cflags": pe.BUILD_BY_POLICY[policy],
        "policy_flashed": policy,
        "policies_supported": [p for p in pe.POLICIES
                               if pe.BUILD_NAME_BY_POLICY[p] == pe.BUILD_NAME_BY_POLICY[policy]],
        "dataset": dataset,
        "condition": pe.CONDITION,
        "init_mode": pe.INIT_MODE,
        "n_features": k,
        "feature_names": names,
        "bss_bytes": bss,
        "pc_checkpoint": str(pc_ckpt.relative_to(ROOT)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[politique] binaire « {manifest['build']} » flashé pour {dataset} "
          f"(k={k}, .bss={bss} B)")
    print(f"[politique] politiques mesurables sur ce binaire : "
          f"{', '.join(manifest['policies_supported'])}")
    print(f"[politique] manifeste → {args.out / MANIFEST_NAME}")
    print("\nBasculer le cavalier JP5 sur l'alimentation sonde, puis lancer --measure.")
    return manifest


# ── Garde anti-mislabel ──────────────────────────────────────────────────────

def check_manifest(args, policies: list[str]) -> dict:
    """Le binaire déclaré au banc est-il celui qui a été flashé ?

    Refuser d'acquérir est le seul comportement honnête : une cellule mesurée sur un
    binaire qui ne porte pas sa politique serait écrite sous un nom qui ment sur son
    contenu (leçon du Sprint 52).
    """
    path = args.out / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"manifeste de flash absent : {path}\n"
            f"Lancer d'abord « --prepare --policy <politique> --dataset {args.dataset} ». "
            f"Aucune mesure n'est écrite sans savoir quel binaire tourne."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["build"] != args.build:
        raise SystemExit(
            f"binaire déclaré « {args.build} » ≠ binaire flashé « {manifest['build']} » "
            f"({manifest['timestamp']}). Rien n'est mesuré."
        )
    if manifest["dataset"] != args.dataset:
        raise SystemExit(
            f"dataset déclaré « {args.dataset} » ≠ dataset flashé "
            f"« {manifest['dataset']} » : la tête EWC embarquée est celle d'un autre "
            f"dataset (dims et poids). Rien n'est mesuré."
        )
    incompatibles = [p for p in policies if pe.BUILD_NAME_BY_POLICY[p] != args.build]
    if incompatibles:
        raise SystemExit(
            f"politique(s) {', '.join(incompatibles)} non portée(s) par le binaire "
            f"« {args.build} » (exige « "
            f"{', '.join(pe.BUILD_BY_POLICY[p] or 'build par défaut' for p in incompatibles)} »). "
            f"Rien n'est mesuré."
        )
    return manifest


# ── Phase 2 : mesure ─────────────────────────────────────────────────────────

def build_schedule(policies: list[str], rates: list[float], repeats: int,
                   shuffle: bool, seed: int) -> list[tuple[str, float, int]]:
    """Plan d'acquisition (politique, cadence, répétition), ordre randomisé par défaut.

    La leçon de S5301 : ne jamais laisser l'ordre corrélé à la condition. Ici, P0 et P1
    sont mesurées dans la même séance et entrelacées — c'est ce qui rend leur différence
    robuste à une dérive lente de banc.
    """
    plan = [(policy, float(rate), k)
            for policy in policies for rate in rates for k in range(repeats)]
    if shuffle:
        random.Random(seed).shuffle(plan)
    return plan


def measure(args, policies: list[str], manifest: dict) -> dict[str, dict]:
    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    voltage_v = float(calib.get("supply_voltage_v", voltage_mv / 1000))
    rates = [float(r) for r in args.rates]
    n_samples = int(max(rates) * (args.window + args.settle + 4.0)) or args.n_check

    probe = lp.PowerShield(args.port)
    runs: dict[tuple[str, float], list[float]] = {}
    indices: dict[tuple[str, float], list[int]] = {}
    checks: dict[str, dict] = {}
    sequence: list[dict] = []
    anchor_runs: dict[float, list[float]] = {}
    try:
        probe.take_control()

        # 1. Rebut de préchauffage — la 1re acquisition d'une session est biaisée (+8 mA
        #    relevés au Sprint 50). Sa valeur est consignée, jamais retenue.
        warmups = [lp.warmup(probe, voltage_mv)
                   for _ in range(max(1, args.warmup_repeats))]
        for k, w in enumerate(warmups):
            print(f"[politique] préchauffage {k} écarté : {w * MA:.3f} mA (non retenu)")

        # 2. Contrôle par politique : intégrité du flux, latence DWT et compteur du gate,
        #    mesurés EN SÉANCE. Le taux de mise à jour est ainsi CONSTATÉ, puis confronté
        #    au Sprint 38 — pas simplement recopié.
        for policy in policies:
            cmd = policy_stream_command(policy, args.dataset, args.board_port,
                                        args.n_check, CHECK_RATE_HZ)
            res = stream_once(probe, voltage_mv, cmd, policy)
            counters = gate_counters(res)
            checks[policy] = {
                "model_flag": EWC_MODEL_FLAG,
                "update_flag_uart": pe.UPDATE_FLAG_BY_POLICY[policy],
                "n_samples_expected": int(args.n_check),
                "n_samples_received": res.get("n_samples"),
                "samples_lost": int(args.n_check) - int(res.get("n_samples", 0)),
                "crc_errors": res.get("crc_errors"),
                "latency_p50_us": res.get("latency_p50_us"),
                "latency_p99_us": res.get("latency_p99_us"),
                "achieved_rate_hz": res.get("achieved_rate_hz"),
                "check_rate_hz": CHECK_RATE_HZ,
                "f1_faulty": res.get("f1_faulty"),
                "gate": counters,
            }
            print(f"[politique] contrôle {policy:18s} : {res.get('n_samples')}/"
                  f"{args.n_check} éch., P50={res.get('latency_p50_us')} µs, "
                  f"CRC={res.get('crc_errors')}, MAJ={counters.get('n_updates')}")

        # 3. Balayage proprement dit, dans un ordre décorrélé de la politique.
        plan = build_schedule(policies, rates, args.repeats, args.shuffle_order, args.seed)
        for position, (policy, rate, k) in enumerate(plan):
            cmd = (None if rate <= 0 else
                   policy_stream_command(policy, args.dataset, args.board_port,
                                         n_samples, rate))
            i_run, voltage_v = rc.measure_current(
                probe, args.window, voltage_mv, cmd, args.settle if cmd else 0.0)
            runs.setdefault((policy, rate), []).append(i_run)
            indices.setdefault((policy, rate), []).append(position)
            sequence.append({"session_index": position, "policy": policy,
                             "rate_hz": rate, "repeat": k, "i_a": float(i_run)})
            print(f"[politique] #{position:03d} {policy:18s} {rate:6.1f} Hz : "
                  f"{i_run * MA:7.3f} mA")

        # 4. Cadence ATTEINTE à la borne haute : c'est ce qui définit la saturation, et
        #    elle ne se lit ni dans les CRC ni dans les trames perdues.
        rate_max = max(rates)
        achieved: dict[str, float | None] = {}
        if rate_max > 0:
            for policy in policies:
                cmd = policy_stream_command(policy, args.dataset, args.board_port,
                                            args.n_check, rate_max)
                res = stream_once(probe, voltage_mv, cmd, f"{policy}@{rate_max:.0f}Hz")
                achieved[policy] = res.get("achieved_rate_hz")
                print(f"[politique] plafond {policy:18s} : consigne {rate_max:.0f} Hz → "
                      f"atteint {res.get('achieved_rate_hz')} Hz")

        # 5. Témoin de séance — son chemin d'exécution est inchangé par le build gate.
        #    Si sa pente bouge d'une séance à l'autre, tout écart INTER-BUILD porte cette
        #    dérive, et le JSON le dira (constat, pas supposition).
        if args.anchor:
            for rate in ANCHOR_RATES:
                cmd = (None if rate <= 0 else
                       rc.stream_command(ANCHOR_MODEL_FLAG, args.dataset,
                                         args.board_port, n_samples, float(rate)))
                i_run, voltage_v = rc.measure_current(
                    probe, args.window, voltage_mv, cmd, args.settle if cmd else 0.0)
                anchor_runs.setdefault(float(rate), []).append(i_run)
                print(f"[politique] témoin {ANCHOR_MODEL_FLAG:12s} {rate:6.1f} Hz : "
                      f"{i_run * MA:7.3f} mA")
    finally:
        probe.release()

    anchor = _finalize_anchor(anchor_runs, voltage_v) if args.anchor else None

    cells: dict[str, dict] = {}
    for policy in policies:
        points = _points_for(runs, indices, policy, rates, achieved.get(policy))
        cell = finalize_cell(policy, points, voltage_v, checks.get(policy, {}),
                             manifest, args)
        cell["session"] = {
            "sequence": [s for s in sequence if s["policy"] == policy],
            "shuffled": bool(args.shuffle_order),
            "seed": int(args.seed),
            "warmup_discarded_a": [float(w) for w in warmups],
            "session_label": f"{args.build}_{args.dataset}",
        }
        cell["session_anchor"] = anchor
        cells[f"{policy}_{args.dataset}"] = cell
    return cells


def _points_for(runs, indices, policy: str, rates: list[float],
                achieved_max: float | None) -> list[dict]:
    """Points (cadence, courant moyen, σ) d'une politique — mesures brutes, rien de dérivé."""
    rate_max = max(rates)
    points = []
    for rate in rates:
        values = runs.get((policy, rate), [])
        if not values:
            continue
        point = {
            "rate_hz": float(rate),
            "i_mean_a": float(np.mean(values)),
            "i_std_a": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_repeats": len(values),
            "session_index": indices[(policy, rate)],
        }
        if rate == rate_max and achieved_max is not None:
            point["achieved_rate_hz"] = float(achieved_max)
        elif rate > 0:
            point["achieved_rate_hz"] = float(rate)
        points.append(point)
    return points


def _finalize_anchor(anchor_runs: dict, voltage_v: float) -> dict:
    """Régression du témoin de séance, ou N/A honnête si la séance n'en porte pas assez."""
    points = [{"rate_hz": rate,
               "i_mean_a": float(np.mean(vals)),
               "i_std_a": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
               "n_repeats": len(vals)}
              for rate, vals in sorted(anchor_runs.items())]
    anchor = {"model": ANCHOR_MODEL_FLAG, "points": points}
    try:
        anchor.update(rr.fit_cell(points, voltage_v))
    except ValueError as exc:
        anchor["na_reason"] = f"témoin non ajustable : {exc}"
    return anchor


def finalize_cell(policy: str, points: list[dict], tension_v: float, check: dict,
                  manifest: dict, args) -> dict:
    """Assemble une cellule : points MESURÉS + grandeurs CALCULÉES par `rate_regression`."""
    cell = {
        "policy": policy,
        "dataset": args.dataset,
        "condition": pe.CONDITION,
        "init_mode": pe.INIT_MODE,
        "firmware_build": pe.BUILD_BY_POLICY[policy] or "build par défaut",
        "build_name": pe.BUILD_NAME_BY_POLICY[policy],
        "bss_bytes": manifest.get("bss_bytes"),
        "n_features": manifest.get("n_features"),
        "update_flag_uart": pe.UPDATE_FLAG_BY_POLICY[policy],
        "points": points,
        "protocol_check": check,
        "window_s": float(args.window),
        "settle_s": float(args.settle),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "lpm01a_current",
    }
    cell.update(rr.fit_cell(points, tension_v, check.get("latency_p50_us")))
    return cell


# ── Grandeurs dérivées et sortie ─────────────────────────────────────────────

def refit(out_dir: Path) -> dict[str, dict]:
    """Recalcule les grandeurs dérivées depuis les `points` déjà écrits (hors banc).

    Deux usages : reprendre une séance après coup sans remesurer, et vérifier la chaîne de
    calcul sans carte ni sonde. Les points mesurés ne sont jamais touchés.
    """
    cells: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"economy_energy.json", MANIFEST_NAME}:
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        if "points" not in cell or "policy" not in cell:
            continue   # artefact tiers : lecture seule, jamais réécrit
        latency = cell.get("protocol_check", {}).get("latency_p50_us")
        cell.update(rr.fit_cell(cell["points"], cell["tension_v"], latency))
        cells[f"{cell['policy']}_{cell['dataset']}"] = cell
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
    return cells


def load_cells(out_dir: Path) -> dict[str, dict]:
    """Charge les cellules déjà écrites, sans rien recalculer ni réécrire."""
    cells: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"economy_energy.json", MANIFEST_NAME}:
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        if "points" in cell and "policy" in cell:
            cells[f"{cell['policy']}_{cell['dataset']}"] = cell
    return cells


def build_economy(cells: dict[str, dict], hw_profile: Path, s38_path: Path) -> dict:
    """Grandeurs dérivées de la campagne — toutes CALCULÉES, aucune saisie.

    C'est ici que la tâche produit ce que le mémoire n'a pas : le coût énergétique d'une
    mise à jour CL, le surcoût permanent du gate, l'économie qu'il procure, l'autonomie
    par politique, et le verdict des trois issues.
    """
    s38 = pe.load_s38_summary(s38_path)
    capacites = pe.load_capacities(hw_profile)

    per_cell: dict[str, dict] = {}
    per_dataset: dict[str, dict] = {}

    for dataset in pe.DATASETS:
        frozen = cells.get(f"{pe.BASELINE_POLICY}_{dataset}")
        always = cells.get(f"{pe.REFERENCE_POLICY}_{dataset}")
        update = pe.update_energy_uj(always, frozen)

        gates: dict[str, dict] = {}
        for policy in pe.GATED_POLICIES:
            cell = cells.get(f"{policy}_{dataset}")
            mesure = (cell or {}).get("protocol_check", {}).get("gate", {}).get("update_rate")
            depuis_s38 = pe.s38_update_rate(s38, dataset, policy)
            taux = mesure if pe.is_measured(mesure) else depuis_s38
            overhead = pe.gate_energy_overhead_uj(cell, frozen, taux, update)
            saved = pe.energy_saved_vs_always(cell, always, frozen)
            gates[policy] = {
                "gate_overhead_uj_per_sample": overhead,
                "energy_saved_vs_always": saved,
                "gate_verdict": pe.gate_verdict(overhead, saved),
                "update_rate_used": taux,
                "update_rate_measured": mesure,
                "update_rate_s38": depuis_s38,
                "update_rate_agreement": pe.update_rate_agreement(mesure, depuis_s38),
            }

        per_dataset[dataset] = {
            "energy_uj_per_update": update,
            "gated": gates,
            "ordering_check": pe.ordering_check(cells, dataset),
        }

        for policy in pe.POLICIES:
            key = f"{policy}_{dataset}"
            per_cell[key] = {
                "autonomy": pe.policy_autonomy(cells.get(key), capacites),
                "energy_saved_vs_always": pe.energy_saved_vs_always(
                    cells.get(key), always, frozen),
            }

    anchors = {}
    for key, cell in cells.items():
        label = cell.get("session", {}).get("session_label")
        anchor = cell.get("session_anchor")
        if label and anchor:
            anchors[label] = anchor

    return {
        "method": pe.METHOD,
        "method_note": (
            "estimateur PAR DIFFÉRENCE DE PENTES de régression (S5304) — à ne jamais "
            "fusionner avec l'estimateur par delta (S5302) ni avec l'intégration du profil "
            "par phase (S5305) : leur comparaison est elle-même un résultat."
        ),
        "by_dataset": per_dataset,
        "by_cell": per_cell,
        "anchor_drift": pe.anchor_drift(anchors),
        "cross_table": pe.cross_table(cells, s38, per_cell),
        "battery_capacities_mah": capacites,
        "autonomy_rate_hz": pe.AUTONOMY_RATE_HZ,
        "sources": {
            "s38_summary": str(s38_path),
            "s38_loaded": s38 is not None,
            "hw_profile": str(hw_profile),
            "cells": sorted(cells),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_outputs(cells: dict[str, dict], out_dir: Path, hw_profile: Path,
                  s38_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cell in cells.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
    economy = build_economy(cells, hw_profile, s38_path)
    (out_dir / "economy_energy.json").write_text(
        json.dumps(economy, indent=2, ensure_ascii=False), encoding="utf-8")
    return economy


def report(cells: dict[str, dict], economy: dict) -> None:
    """Restitution console — reprend ce qui est écrit, n'en déduit rien de neuf."""
    print("\n--- Cellules ---")
    for name, cell in sorted(cells.items()):
        energie = cell.get("energy_uj_per_inference")
        rendu = f"{energie:.2f} µJ/éch." if pe.is_measured(energie) else str(energie)
        print(f"  {name:32s} pente={cell.get('slope_ua_per_hz'):+.3f} µA/Hz  "
              f"r²={cell.get('r2'):.3f}  {rendu}")

    print("\n--- Énergie d'une mise à jour CL (always − frozen) ---")
    for dataset, bloc in economy["by_dataset"].items():
        update = bloc["energy_uj_per_update"]
        valeur = update.get("value_uj", A_MESURER)
        if pe.is_measured(valeur):
            print(f"  {dataset:12s} : {valeur:+.2f} ± "
                  f"{update.get('uncertainty_uj', 0.0):.2f} µJ")
        else:
            print(f"  {dataset:12s} : {valeur} — {update.get('na_reason')}")
        for policy, gate in bloc["gated"].items():
            print(f"      {policy:20s} {gate['gate_verdict']['verdict']}")

    drift = economy["anchor_drift"]
    if drift.get("comparable") is not None:
        print(f"\n--- Témoin de séance --- comparable={drift['comparable']} : "
              f"{drift['rationale']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_policies(value: str | None, parser: argparse.ArgumentParser) -> list[str]:
    if not value:
        return list(pe.POLICIES)
    policies = []
    for token in value.split(","):
        token = token.strip()
        if token not in pe.POLICIES:
            parser.error(f"politique inconnue : {token} (attendu : {', '.join(pe.POLICIES)})")
        policies.append(token)
    return policies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    mode = parser.add_argument_group("mode (exactement un)")
    mode.add_argument("--prepare", action="store_true",
                      help="construit et flashe le binaire d'une politique (sonde DÉBRANCHÉE)")
    mode.add_argument("--measure", action="store_true",
                      help="mesure les politiques portées par le binaire flashé")
    mode.add_argument("--refit", action="store_true",
                      help="recalcule tout depuis les points écrits, sans matériel")

    parser.add_argument("--policy", choices=list(pe.POLICIES),
                        help="politique à flasher (--prepare)")
    parser.add_argument("--policies", default=None,
                        help="politiques à mesurer, séparées par des virgules (--measure)")
    parser.add_argument("--build", choices=sorted(set(pe.BUILD_NAME_BY_POLICY.values())),
                        help="binaire RÉELLEMENT flashé, confronté au manifeste (--measure)")
    parser.add_argument("--dataset", default=pe.DATASETS[0], choices=list(pe.DATASETS))
    parser.add_argument("--board-port", default=None,
                        help="port UART de la CARTE (obligatoire avec --measure)")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto-détecté)")
    parser.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window", type=float, default=10.0,
                        help="durée d'une acquisition (s)")
    parser.add_argument("--settle", type=float, default=2.0,
                        help="établissement du flux avant acquisition (s)")
    parser.add_argument("--n-check", type=int, default=300,
                        help="échantillons du flux de contrôle (intégrité + gate)")
    parser.add_argument("--n-enrollment", type=int, default=500,
                        help="échantillons sains de l'enrôlement Maha du gate (miroir S38)")
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--no-shuffle-order", dest="shuffle_order", action="store_false",
                        help="conserve l'ordre nominal (déconseillé : leçon S5301)")
    parser.add_argument("--no-anchor", dest="anchor", action="store_false",
                        help="renonce au témoin de séance (la dérive inter-builds "
                             "devient alors non quantifiée)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    parser.add_argument("--s38-summary", type=Path,
                        default=ROOT / "experiments" / "exp_S38_summary.json")
    parser.set_defaults(shuffle_order=True, anchor=True)
    args = parser.parse_args(argv)

    if sum([args.prepare, args.measure, args.refit]) != 1:
        parser.error("choisir exactement un mode : --prepare, --measure ou --refit")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        if not args.policy:
            parser.error("--prepare exige --policy")
        prepare(args)
        return 0

    if args.refit:
        cells = refit(args.out)
        if not cells:
            print(f"[politique] aucune cellule à recalculer dans {args.out} — "
                  f"rien n'est écrit (le banc n'a pas encore tourné).")
            return 0
        economy = write_outputs(cells, args.out, args.hw_profile, args.s38_summary)
        report(cells, economy)
        return 0

    if not args.board_port:
        parser.error("--measure exige --board-port (port UART de la carte)")
    if not args.build:
        parser.error("--measure exige --build (binaire réellement flashé)")
    policies = parse_policies(args.policies, parser)
    manifest = check_manifest(args, policies)

    mesurees = measure(args, policies, manifest)
    # Les cellules des autres séances sont relues pour que les grandeurs croisées
    # (mise à jour, surcoût du gate, dérive du témoin) puissent être calculées dès que
    # les deux binaires ont été mesurés.
    cells = {**load_cells(args.out), **mesurees}
    economy = write_outputs(cells, args.out, args.hw_profile, args.s38_summary)
    report(mesurees, economy)
    print(f"\n[politique] {len(mesurees)} cellule(s) écrite(s) dans {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
