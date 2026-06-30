#!/usr/bin/env python3
"""board_pc_parity38.py — S3806 : parité board ↔ PC (prédictions + verdicts du gate).

Deux niveaux de parité, **lecture seule** (aucune métrique recalculée de zéro) :

  - **prédiction** : ``pred_board`` vs ``pred_pc`` (exacte en frozen ; approchée sinon —
    float32 board ≠ float64 PC, ou EWC mis à jour des deux côtés).
  - **verdict du gate** : ``verdict_board`` vs ``verdict_pc`` ∈ {NORMAL, DRIFT, FAULT} —
    spécifique au Sprint 38 ; atteste que la **décision d'update** est identique des deux côtés
    (mêmes seuils exportés). Présent uniquement pour les politiques ``gated_*``.

Sources (produites par run_sprint38_board.py, S3804/S3805) :
  - gated_* : ``board_samples.json`` contient déjà, par échantillon (dans l'ordre board),
    ``pred_board``/``pred_pc`` et ``verdict_board``/``verdict_pc`` (reconstruction PC du gate
    sur les features streamées) → table complète + matrice de confusion verdict.
  - frozen/always : pas de verdict (firmware par défaut) ; la parité prédiction agrégée est
    lue dans ``results.json`` (``parity_rate``) ; verdict → null.

Usage :
    python scripts/board_pc_parity38.py                       # toutes les cellules
    python scripts/board_pc_parity38.py --policy gated_pseudolabel --dataset pronostia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

EXPERIMENTS = Path("experiments")
DEFAULT_CONFIG = "configs/sprint38_autonomous_update.yaml"
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")
GATED = ("gated_truelabel", "gated_pseudolabel")
DATASETS = ("monitoring", "pronostia")
INIT_MODES = ("pretrained", "scratch")
VERDICTS = ("NORMAL", "DRIFT", "FAULT")


def _build_gated(policy: str, dataset: str, init: str) -> dict | None:
    """Table parité complète (prédiction + verdict) depuis board_samples.json gated."""
    board_dir = EXPERIMENTS / f"exp_S38_board_{policy}_{dataset}_{init}"
    bs_path = board_dir / "board_samples.json"
    if not bs_path.exists():
        print(f"  [skip {policy}/{dataset}/{init}] board_samples.json absent")
        return None
    bs = json.loads(bs_path.read_text())

    rows, pred_mismatches, verdict_mismatches = [], [], []
    # Matrice de confusion verdict_pc (lignes) × verdict_board (colonnes).
    confusion = {vp: {vb: 0 for vb in VERDICTS} for vp in VERDICTS}
    for s in bs:
        pp, pb = int(s["pred_pc"]), int(s["pred_board"])
        vp, vb = s["verdict_pc"], s["verdict_board"]
        pred_match, verdict_match = (pp == pb), (vp == vb)
        confusion[vp][vb] += 1
        row = {
            "idx": int(s["idx"]), "true": int(s["true"]),
            "pred_pc": pp, "pred_board": pb,
            "verdict_pc": vp, "verdict_board": vb,
            "pred_match": pred_match, "verdict_match": verdict_match,
        }
        rows.append(row)
        if not pred_match:
            pred_mismatches.append(row)
        if not verdict_match:
            verdict_mismatches.append(row)

    n = len(rows)
    return {
        "exp_id": f"exp_S38_parity_{policy}_{dataset}_{init}",
        "policy": policy, "dataset": dataset, "init_mode": init,
        "parity_class": "approx (EWC mis à jour) / verdict exact attendu",
        "n_compared": n,
        "prediction_parity_rate": float(np.mean([r["pred_match"] for r in rows])) if n else None,
        "verdict_parity_rate": float(np.mean([r["verdict_match"] for r in rows])) if n else None,
        "pred_mismatch_count": len(pred_mismatches),
        "verdict_mismatch_count": len(verdict_mismatches),
        "verdict_confusion_pc_x_board": confusion,
        "rows": rows,
        "pred_mismatches": pred_mismatches,
        "verdict_mismatches": verdict_mismatches,
    }


def _build_reference(policy: str, dataset: str, init: str) -> dict | None:
    """frozen/always : parité prédiction agrégée (lue de results.json) ; pas de verdict."""
    board_dir = EXPERIMENTS / f"exp_S38_board_{policy}_{dataset}_{init}"
    res_path = board_dir / "results.json"
    bs_path = board_dir / "board_samples.json"
    if not res_path.exists():
        print(f"  [skip {policy}/{dataset}/{init}] results.json absent")
        return None
    res = json.loads(res_path.read_text())
    rows = []
    if bs_path.exists():
        for s in json.loads(bs_path.read_text()):
            rows.append({"idx": int(s["idx"]), "true": int(s["true"]),
                         "pred_board": int(s["pred_board"]),
                         "pred_pc": None, "verdict_pc": None, "verdict_board": None,
                         "pred_match": None, "verdict_match": None})
    return {
        "exp_id": f"exp_S38_parity_{policy}_{dataset}_{init}",
        "policy": policy, "dataset": dataset, "init_mode": init,
        "parity_class": res.get("parity_class"),
        "n_compared": res.get("n_compared"),
        "prediction_parity_rate": res.get("parity_rate"),
        "verdict_parity_rate": None,   # firmware par défaut : pas de gate
        "pred_mismatch_count": res.get("parity_mismatch_count"),
        "verdict_mismatch_count": None,
        "verdict_confusion_pc_x_board": None,
        "rows": rows,
        "pred_mismatches": [],
        "verdict_mismatches": [],
    }


def build_cell(policy: str, dataset: str, init: str) -> dict | None:
    if policy in GATED:
        return _build_gated(policy, dataset, init)
    return _build_reference(policy, dataset, init)


def main() -> None:
    p = argparse.ArgumentParser(description="Parité board↔PC verdicts+prédictions (S3806)")
    p.add_argument("--policy", choices=POLICIES, default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None)
    p.add_argument("--init-mode", choices=INIT_MODES, default=None)
    args = p.parse_args()

    policies = [args.policy] if args.policy else list(POLICIES)
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    inits = [args.init_mode] if args.init_mode else list(INIT_MODES)

    written = 0
    for init in inits:
        for ds in datasets:
            for pol in policies:
                res = build_cell(pol, ds, init)
                if res is None:
                    continue
                out = EXPERIMENTS / f"exp_S38_parity_{pol}_{ds}_{init}.json"
                out.write_text(json.dumps(res, indent=2))
                pr = res["prediction_parity_rate"]
                vr = res["verdict_parity_rate"]
                print(f"  {out.name:54s} n={res['n_compared']} "
                      f"pred_parity={pr} verdict_parity={vr}")
                written += 1

    print(f"\n{written} fichiers parité écrits dans {EXPERIMENTS}/")


if __name__ == "__main__":
    main()
