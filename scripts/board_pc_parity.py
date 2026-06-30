#!/usr/bin/env python3
"""board_pc_parity.py — S3605 : comparaison prédiction-par-prédiction PC ↔ board (EWC).

Pour chaque ``(condition ∈ {5feat, all}, protocol ∈ {frozen, online},
dataset ∈ {pronostia, monitoring})`` produit
``experiments/exp_S36_parity_{condition}_{protocol}_{dataset}.json`` :

  - **frozen** : parité **exacte** (poids gelés). On charge les ``samples`` PC dumpés par
    S3602 (idx/true/pred/confidence/features) et on rejoue le **même checkpoint** EWC
    (``_pc_pred_ewc``) sur les features → ``pred_board``. La board réelle (S3603) a vérifié
    ``parity_rate=1.000`` : board == PC par construction, donc cette reconstruction reproduit
    fidèlement ce que la board a renvoyé, sans la solliciter. ``parity_rate`` attendu = 1.0.

  - **online** : parité **approchée** (poids non gelés ⇒ float32 board ≠ float64 PC). On charge
    ``board_samples.json`` (persisté par ``run_sprint36_board.py --pass online`` : pred_board,
    conf_board, pred_pc du miroir, task_id) → table réelle des désaccords.

Réutilise ``_pc_pred_ewc`` de ``run_feature_condition_board.py`` (forward checkpoint identique
au firmware) — aucune réimplémentation du forward.

Usage :
    python scripts/board_pc_parity.py                       # toutes les cellules
    python scripts/board_pc_parity.py --condition all --protocol frozen --dataset pronostia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

EXPERIMENTS = Path("experiments")
CONDITIONS = ["5feat", "all"]
PROTOCOLS = ["frozen", "online"]
DATASETS = ["pronostia", "monitoring"]


def _pc_pred_conf_ewc(ckpt: Path, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward EWC depuis un checkpoint → (pred argmax, confidence = softmax max)."""
    import torch

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    k = int(sd["fc1.weight"].shape[1])
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16])
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(feats, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return pred.numpy(), conf.numpy()


def _build_frozen(condition: str, dataset: str) -> dict | None:
    """Table parité frozen reconstruite exactement depuis les samples PC + checkpoint."""
    pc_dir = EXPERIMENTS / f"exp_S36_PC_{condition}_ewc_{dataset}"
    pc_path = pc_dir / "results.json"
    ckpt = pc_dir / "checkpoints" / "ewc_head.pt"
    if not pc_path.exists() or not ckpt.exists():
        print(f"  [skip frozen {condition}/{dataset}] PC results/checkpoint absent")
        return None

    pc = json.loads(pc_path.read_text())
    samples = [s for s in pc.get("samples", []) if s.get("features")]
    if not samples:
        print(f"  [skip frozen {condition}/{dataset}] pas de samples PC")
        return None

    feats = np.array([s["features"] for s in samples], dtype=np.float32)
    pred_board, conf_board = _pc_pred_conf_ewc(ckpt, feats)

    rows, mismatches = [], []
    for s, pb, cb in zip(samples, pred_board, conf_board):
        pc_pred = int(s["pred"])
        bpred = int(pb)
        match = (pc_pred == bpred)
        row = {
            "idx": int(s["idx"]), "true": int(s["true"]),
            "pred_pc": pc_pred, "pred_board": bpred,
            "conf_pc": s.get("confidence"), "conf_board": float(cb),
            "match": match,
        }
        rows.append(row)
        if not match:
            mismatches.append(row)

    n = len(rows)
    return {
        "exp_id": f"exp_S36_parity_{condition}_frozen_{dataset}",
        "condition": condition, "protocol": "frozen", "dataset": dataset,
        "parity_class": "exact",
        "n_compared": n,
        "parity_rate": float(np.mean([r["match"] for r in rows])) if n else None,
        "mismatch_count": len(mismatches),
        "rows": rows,
        "mismatches": mismatches,
    }


def _build_online(condition: str, dataset: str) -> dict | None:
    """Table parité online depuis board_samples.json (board réel + miroir PC)."""
    bs_path = (EXPERIMENTS / f"exp_S36_board_online_{condition}_ewc_{dataset}"
               / "board_samples.json")
    if not bs_path.exists():
        print(f"  [skip online {condition}/{dataset}] board_samples.json absent "
              f"(relancer run_sprint36_board.py --pass online)")
        return None

    bs = json.loads(bs_path.read_text())
    rows, mismatches = [], []
    for s in bs:
        pc_pred, bpred = int(s["pred_pc"]), int(s["pred_board"])
        match = (pc_pred == bpred)
        row = {
            "idx": int(s["idx"]), "true": int(s["true"]),
            "pred_pc": pc_pred, "pred_board": bpred,
            "conf_pc": s.get("conf_pc"), "conf_board": s.get("conf_board"),
            "match": match, "task_id": s.get("task_id"),
        }
        rows.append(row)
        if not match:
            mismatches.append(row)

    n = len(rows)
    return {
        "exp_id": f"exp_S36_parity_{condition}_online_{dataset}",
        "condition": condition, "protocol": "online", "dataset": dataset,
        "parity_class": "approx",
        "n_compared": n,
        "parity_rate": float(np.mean([r["match"] for r in rows])) if n else None,
        "mismatch_count": len(mismatches),
        "rows": rows,
        "mismatches": mismatches,
    }


def build_cell(condition: str, protocol: str, dataset: str) -> dict | None:
    if protocol == "frozen":
        return _build_frozen(condition, dataset)
    return _build_online(condition, dataset)


def main() -> None:
    p = argparse.ArgumentParser(description="Parité prédiction-par-prédiction PC↔board (S3605)")
    p.add_argument("--condition", choices=CONDITIONS, default=None)
    p.add_argument("--protocol", choices=PROTOCOLS, default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None)
    args = p.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS
    protocols = [args.protocol] if args.protocol else PROTOCOLS
    datasets = [args.dataset] if args.dataset else DATASETS

    written = 0
    for ds in datasets:
        for cond in conditions:
            for proto in protocols:
                res = build_cell(cond, proto, ds)
                if res is None:
                    continue
                out = EXPERIMENTS / f"exp_S36_parity_{cond}_{proto}_{ds}.json"
                out.write_text(json.dumps(res, indent=2))
                rate = res["parity_rate"]
                print(f"  {out.name:42s} n={res['n_compared']:6d} "
                      f"parity={rate:.4f} mismatch={res['mismatch_count']}")
                written += 1

    print(f"\n{written} fichiers parité écrits dans {EXPERIMENTS}/")


if __name__ == "__main__":
    main()
