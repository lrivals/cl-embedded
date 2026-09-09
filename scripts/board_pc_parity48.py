#!/usr/bin/env python3
"""board_pc_parity48.py — Parité par échantillon board↔PC des cellules sub-INT8 (S4805).

Le PC de référence est **l'émulateur subint8** (Sprint 47), rejoué sur la MÊME séquence que
la board (lue depuis ``board_samples.json`` produit par ``run_s48_board_depth.py``). Comme
l'export firmware et l'émulateur partagent les primitives de quantification (poids, scales,
calibration seed-48), la parité est **exacte par construction** (précédent S34/S45).

Pour chaque cellule mesurée, écrit ``experiments/exp_S48_board/exp_S48_parity_<cell>.json`` :
table ``[idx, true, pred_board, pred_pc, score_board, score_pc, match]`` + ``parity_pred``,
``mismatches``, ``max_score_err``.

Usage :
    python scripts/board_pc_parity48.py --all
    python scripts/board_pc_parity48.py --cell exp_S48_pronostia_ternary_per_channel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_s48_board_depth import OUT_DIR, emulator_predict  # noqa: E402

# mode → weight_mode émulateur (miroir de MODES dans le driver).
_MODE_WEIGHT = {"int4": "linear", "ternary": "ternary", "binary": "binary"}


def run_parity(cell_dir: Path) -> dict | None:
    """Rejoue l'émulateur sur la séquence board d'une cellule. None si non mesurée."""
    sp = cell_dir / "board_samples.json"
    if not sp.exists():
        return None
    payload = json.loads(sp.read_text())
    samples = payload.get("samples", [])
    if not samples:
        return None

    ckpt = Path(payload["checkpoint"])
    mode = payload["mode"]
    bits = int(payload["weight_bits"])
    weight_mode = _MODE_WEIGHT[mode]

    feats = np.array([s["features"] for s in samples], dtype=np.float32)
    trues = np.array([int(s["true"]) for s in samples])
    board_pred = np.array([int(s["pred"]) for s in samples])
    board_conf = np.array([float(s["confidence"]) for s in samples])

    pc_pred, pc_prob1, _ = emulator_predict(ckpt, weight_mode, bits, feats)

    match = (board_pred == pc_pred)
    parity = float(match.mean())
    mismatches = [int(i) for i in np.where(~match)[0]]
    max_score_err = float(np.max(np.abs(board_conf - pc_prob1)))

    table = [{"idx": int(i), "true": int(trues[i]),
              "pred_board": int(board_pred[i]), "pred_pc": int(pc_pred[i]),
              "score_board": float(board_conf[i]), "score_pc": float(pc_prob1[i]),
              "match": bool(match[i])} for i in range(len(samples))]

    out = {
        "cell": cell_dir.name, "dataset": payload["dataset"], "mode": mode,
        "weight_bits": bits, "packed": bool(payload["packed"]), "checkpoint": str(ckpt),
        "n_compared": len(samples), "parity_pred": parity,
        "parity_mismatch_count": len(mismatches),
        "mismatches": mismatches[:200], "max_score_err": max_score_err,
        "table": table,
    }
    out_path = OUT_DIR / f"exp_S48_parity_{cell_dir.name.replace('exp_S48_', '')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  {cell_dir.name}: parité={parity:.4f} mismatch={len(mismatches)}/{len(samples)} "
          f"max_score_err={max_score_err:.2e} → {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Parité board↔émulateur sub-INT8 (S4805)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cell", help="nom du répertoire cellule (exp_S48_<ds>_<mode>_...)")
    args = ap.parse_args()

    if args.cell:
        dirs = [OUT_DIR / args.cell]
    else:
        dirs = sorted(d for d in OUT_DIR.glob("exp_S48_*")
                      if d.is_dir() and (d / "board_samples.json").exists())
    if not dirs:
        print("Aucune cellule streamée (board_samples.json absent) — rien à faire.")
        return
    n = 0
    for d in dirs:
        if run_parity(d) is not None:
            n += 1
    print(f"[S4805] {n} cellule(s) de parité écrite(s).")


if __name__ == "__main__":
    main()
