#!/usr/bin/env python3
"""
scripts/board_pc_parity45.py — Sprint 45 (S4503) : parité board↔PC des détecteurs de drift.

Rejoue **la même séquence, même ordre** côté PC — en dérivant le signal du **modèle de référence
board exporté** (parité par construction, précédent S38 ``_pc_gate_replay``) — et compare au verdict
board enregistré par ``run_sprint45_board.py`` :

- supervisé (Page-Hinkley, DDM) : ``pred_pc = argmax EWCMlpMulticlass(features)`` (tête exportée),
  ``error = 1[pred_pc != true]`` → détecteur Python (mêmes paramètres calibrés) → ``verdict_pc`` ;
- non-supervisé (PSI) : ``score = Mahalanobis(features)`` (détecteur d'enrôlement exporté) → PSI Python
  (mêmes bornes/référence) → ``verdict_pc``.

Écrit ``experiments/exp_S45_parity_{detector}_{dataset}.json`` : table par échantillon
``[idx, signal, verdict_pc, verdict_board, match]`` + ``mismatches`` + ``verdict_parity``.
Attendu : **1.000** sur les détecteurs déterministes à paramètres identiques (comme drift_detector.c
S3803). Une divergence FP32(board)↔FP64(PC) résiduelle sur les détecteurs à accumulation est
reportée **honnêtement** (miroir online S38 0.963–0.989), jamais maquillée.

Usage
-----
    python scripts/board_pc_parity45.py --detector page_hinkley --dataset gas_sensor_drift
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.drift import DDM, PSI, PageHinkley  # noqa: E402

EXPERIMENTS = ROOT / "experiments"


def _pred_ewc(ewc_ckpt: Path, feats: np.ndarray) -> np.ndarray:
    """pred = argmax EWCMlpMulticlass(feats) — miroir bit-pour-bit de ewc_forward board."""
    import torch  # noqa: PLC0415
    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: PLC0415

    sd = torch.load(ewc_ckpt, map_location="cpu")["model_state_dict"]
    k = sd["fc1.weight"].shape[1]
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16])
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(feats, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def _maha_scores(maha_ckpt: Path, feats: np.ndarray) -> np.ndarray:
    with open(maha_ckpt, "rb") as f:
        maha = pickle.load(f)
    return np.asarray(maha.anomaly_score(feats), dtype=np.float64)


def _build_pc_detector(detector: str, params: dict) -> tuple:
    """Instancie le détecteur Python avec les MÊMES paramètres que le header exporté."""
    if detector == "page_hinkley":
        ph = params["page_hinkley"]
        return PageHinkley({"delta": ph["delta"], "lambda_": ph["lambda"],
                            "min_instances": ph["min_instances"]}), "error"
    if detector == "ddm":
        dd = params["ddm"]
        return DDM({"warning_level": dd["warning_level"], "drift_level": dd["drift_level"],
                    "min_instances": dd["min_instances"]}), "error"
    if detector == "psi":
        ps = params["psi"]
        det = PSI({"bins": ps["bins"], "block_size": ps["block_size"],
                   "metric": "psi", "psi_threshold": ps["threshold"]})
        det._edges = np.asarray(ps["edges"], dtype=np.float64)
        det._ref_probs = np.asarray(ps["ref_probs"], dtype=np.float64)
        det.reset()
        return det, "score"
    raise ValueError(detector)


def run_parity(detector: str, dataset: str) -> dict:
    board_dir = EXPERIMENTS / f"exp_S45_board_{detector}_{dataset}"
    samples = json.loads((board_dir / "board_samples.json").read_text())
    params = json.loads((board_dir / "drift_methods_params.json").read_text())

    feats = np.array([s["features"] for s in samples], dtype=np.float32)
    trues = np.array([int(s["true"]) for s in samples])
    verdict_board = [s["verdict"] for s in samples]
    board_pred = np.array([int(s["pred"]) for s in samples])

    det, signal_kind = _build_pc_detector(detector, params)

    # Signal PC dérivé du modèle de référence board exporté (parité par construction).
    pred_parity = None
    if signal_kind == "error":
        pred_pc = _pred_ewc(board_dir / "checkpoints" / "ewc_head.pt", feats)
        pred_parity = float((pred_pc == board_pred).mean())
        signal = (pred_pc != trues).astype(np.float64)
    else:
        signal = _maha_scores(board_dir / "checkpoints" / "mahalanobis_task0.pkl", feats)

    verdict_pc = [det.update(float(s)).name for s in signal]

    match = [vb == vp for vb, vp in zip(verdict_board, verdict_pc)]
    parity = float(np.mean(match)) if match else None
    mismatches = [{"idx": int(samples[i]["idx"]), "signal": float(signal[i]),
                   "verdict_pc": verdict_pc[i], "verdict_board": verdict_board[i]}
                  for i, m in enumerate(match) if not m]

    out = {
        "exp_id": f"exp_S45_parity_{detector}_{dataset}",
        "detector": detector, "dataset": dataset, "n_samples": len(samples),
        "signal_kind": signal_kind,
        "verdict_parity": parity, "verdict_mismatch_count": len(mismatches),
        "pred_parity": pred_parity,
        "table": [{"idx": int(samples[i]["idx"]), "signal": float(signal[i]),
                   "verdict_pc": verdict_pc[i], "verdict_board": verdict_board[i],
                   "match": bool(match[i])} for i in range(len(samples))],
        "mismatches": mismatches[:200],
    }
    out_path = EXPERIMENTS / f"exp_S45_parity_{detector}_{dataset}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[parity45] {out['exp_id']} : verdict_parity={parity} "
          f"mismatch={len(mismatches)} pred_parity={pred_parity} → {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Parité board↔PC des détecteurs de drift (S4503)")
    p.add_argument("--detector", required=True, choices=["page_hinkley", "ddm", "psi"])
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    run_parity(args.detector, args.dataset)


if __name__ == "__main__":
    main()
