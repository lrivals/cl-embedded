#!/usr/bin/env python3
"""run_s34_board_maha_q15.py — Expérience board Q15 Mahalanobis (Sprint 34, S3408).

Pour chaque dataset board-5feat :
  1. Entraîne une référence Mahalanobis 5-feat (train_board_reference.py) → checkpoint pkl.
  2. Exporte model_weights.h (μ/Σ⁻¹ FP32 + z-score identité + seuil) ET
     mahalanobis_q15_weights.h (μ INT8 + Σ⁻¹ int16 Q15) via export_weights_c.py.
  3. Recompile + flashe la NUCLEO-F439ZI (g_maha_q15 chargé depuis le header généré).
  4. Streame --model maha-q15 (flag 0xF0, SANS --update → poids figés) → réponses V3.
  5. Parité board↔PC : le firmware (maha_q15_score) doit reproduire le détecteur Q15 PC
     (anomaly_score_q15) sur les mêmes échantillons. Mesure latence DWT + .bss.

Aucun chiffre inventé : tout provient du flash réel (sauf --dry-run, qui saute board).

Usage :
    python scripts/run_s34_board_maha_q15.py --port /dev/ttyACM0 \
        --datasets cmapss pronostia --n-samples 300
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

FW_DIR = _ROOT / "firmware" / "stm32f4_blink"
OUT_DIR = _ROOT / "experiments" / "exp_S34_board_maha_q15"
DEFAULT_THRESHOLD = {"cmapss": 30.0, "pronostia": 0.5, "battery": 100.0}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _bss_bytes() -> int:
    out = subprocess.run(["arm-none-eabi-size", str(FW_DIR / "build/stm32f4_blink.elf")],
                         capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[-1].split()[2])


def _pc_q15_from_ckpt(ckpt: Path):
    """Reconstruit le détecteur Q15 PC depuis le checkpoint FP32 (mêmes poids quantifiés
    que mahalanobis_q15_weights.h → parité par construction)."""
    from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8

    with open(ckpt, "rb") as f:
        model = pickle.load(f)
    q15 = MahalanobisDetectorInt8({"quantization": "q15"})
    q15.mu_ = np.asarray(model.mu_, dtype=np.float32)
    q15.sigma_inv_ = np.asarray(model.sigma_inv_, dtype=np.float32)
    q15.threshold_ = float(model.threshold_)
    q15.n_features_ = int(model.mu_.shape[0])
    q15.calibrate_q15()
    return q15


def _parity(ckpt: Path, samples: list[dict]) -> dict:
    """Compare prédictions (et scores reconstruits) board ↔ PC Q15."""
    valid = [s for s in samples if s.get("features")]
    if not valid:
        return {"parity_ok": None, "n_compared": 0}
    feats = np.array([s["features"] for s in valid], dtype=np.float32)
    board_pred = np.array([int(s["pred"]) for s in valid])
    # score board reconstruit depuis conf = 1/(1+score) → score = 1/conf - 1
    board_conf = np.array([float(s.get("confidence", 0.0)) for s in valid])
    board_score = np.where(board_conf > 0, 1.0 / np.clip(board_conf, 1e-9, None) - 1.0, np.nan)

    q15 = _pc_q15_from_ckpt(ckpt)
    pc_pred = q15.predict_q15(feats).astype(int)
    pc_score = q15.anomaly_score_q15(feats)

    n_mismatch = int((board_pred != pc_pred).sum())
    score_mask = ~np.isnan(board_score)
    max_score_err = (
        float(np.max(np.abs(board_score[score_mask] - pc_score[score_mask])))
        if score_mask.any() else None
    )
    return {
        "parity_ok": bool(n_mismatch == 0),
        "n_compared": len(valid),
        "parity_mismatch_count": n_mismatch,
        "parity_rate": float((board_pred == pc_pred).mean()),
        "max_score_abs_error": max_score_err,
    }


def run_dataset(dataset: str, threshold: float, args) -> dict:
    exp_dir = OUT_DIR / dataset
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 64}\n  S3408 board Q15 — {dataset} (thr={threshold})\n{'=' * 64}")

    # 1) Référence board 5-feat
    r = _run([sys.executable, "scripts/train_board_reference.py",
              "--model", "mahalanobis", "--dataset", dataset,
              "--threshold", str(threshold), "--exp_dir", str(exp_dir)])
    if r.returncode != 0:
        print(f"  [FAIL train] {(r.stderr or r.stdout)[-400:]}")
        return {"dataset": dataset, "status": "train_failed"}
    ckpt = exp_dir / "checkpoints" / "mahalanobis_task0.pkl"

    # 2) Export model_weights.h (z-score identité) + mahalanobis_q15_weights.h
    r = _run([sys.executable, "scripts/export_weights_c.py",
              "--mahal", str(ckpt), "--maha-q15", str(ckpt)])
    if r.returncode != 0:
        print(f"  [FAIL export] {(r.stderr or r.stdout)[-400:]}")
        return {"dataset": dataset, "status": "export_failed"}

    if args.dry_run:
        print("  [dry-run] build/flash/stream sautés")
        return {"dataset": dataset, "status": "dry_run", "checkpoint": str(ckpt)}

    # 3) Build + flash
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    rb = _run(["make", "-C", str(FW_DIR), "all"])
    if rb.returncode != 0:
        print(f"  [FAIL build] {(rb.stderr or rb.stdout)[-400:]}")
        return {"dataset": dataset, "status": "build_failed"}
    bss = _bss_bytes()
    if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        return {"dataset": dataset, "status": "flash_failed", "bss_bytes": bss}

    # 4) Stream maha-q15 (sans --update)
    out_json = exp_dir / "stream.json"
    rs = _run([sys.executable, "scripts/sensor_stream.py",
               "--dataset", dataset, "--model", "maha-q15",
               "--n-samples", str(args.n_samples), "--rate-hz", str(args.rate_hz),
               "--protocol-version", "3", "--dump-samples",
               "--port", args.port, "--output", str(out_json)], timeout=600)
    if rs.returncode != 0 or not out_json.exists():
        print(f"  [FAIL stream] {(rs.stderr or rs.stdout)[-400:]}")
        return {"dataset": dataset, "status": "stream_failed", "bss_bytes": bss}
    stats = json.loads(out_json.read_text())
    samples = stats.get("samples", [])

    # 5) Parité + latence (latence DWT agrégée au niveau du JSON de stream)
    par = _parity(ckpt, samples)
    result = {
        "dataset": dataset,
        "threshold": threshold,
        "model": "maha_q15",
        "flag": "0xF0",
        "status": "ok",
        "bss_bytes": bss,
        "n_samples": len(samples),
        "latency_us_mean": stats.get("latency_mean_us"),
        "latency_us_p50": stats.get("latency_p50_us"),
        "latency_us_p99": stats.get("latency_p99_us"),
        "gap2_ok": bool((stats.get("latency_p99_us") or 0) < 100_000),
        **par,
    }
    (exp_dir / f"{dataset}.json").write_text(json.dumps(result, indent=2))
    print(f"  parity_ok={result['parity_ok']} rate={result.get('parity_rate')} "
          f"max_score_err={result.get('max_score_abs_error')} "
          f"lat_p50={result['latency_us_p50']}µs p99={result['latency_us_p99']}µs "
          f"gap2_ok={result['gap2_ok']} .bss={bss}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Expérience board Q15 Mahalanobis (S3408)")
    p.add_argument("--datasets", nargs="+", default=["cmapss", "pronostia"],
                   choices=["cmapss", "pronostia", "battery"])
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_dataset(ds, DEFAULT_THRESHOLD[ds], args) for ds in args.datasets]
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"sprint": "S34", "task": "S3408", "datasets": results}, indent=2))
    print(f"\n✅ summary → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
