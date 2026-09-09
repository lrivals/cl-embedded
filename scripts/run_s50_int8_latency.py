#!/usr/bin/env python3
"""run_s50_int8_latency.py — S5004 : breakdown latence INT8 (déquant/MAC/requant) board réelle.

Répond à la tâche 🟡 du CR (« détailler le coût de latence INT8 »). Le processeur étant FP32
(FPU), passer en INT8 ajoute des étapes de quantification/déquantification/requantification. Ce
driver **mesure au DWT** ces étapes, cycle par cycle, sur la NUCLEO-F439ZI réelle, pour le kernel
EWC INT8 v2 (``ewc_head_int8_v2.c``), et les compare à la latence FP32 sur les **mêmes
échantillons** — étayant le paradoxe latence FPU du Sprint 29 (INT8 = gain RAM, pas latence).

Mécanique (une seule image flashée par dataset) :
  build ``-DEWC_INT8_V2 -DINT8_SEGMENT_PROFILE`` → flash → stream 2 fois :
    - flag 0x40 (INT8) : ``ewc_int8_v2_forward`` remplit 3 accumulateurs DWT (cycles bruts)
      déquant/MAC/requant, remontés dans la réponse V3 via les slots [acc][auroc][forgetting]
      (wire format 23 B INCHANGÉ — réinterprétation build-gardée, cf. pipeline.c S5004) ;
    - flag 0x10 (FP32) : chemin ``ewc_head_forward`` (g_ewc_head), latence totale FP32.
  Le FP32 est byte-identique à un build non instrumenté (l'instrumentation est TOUTE sous
  ``#ifdef INT8_SEGMENT_PROFILE`` DANS ``ewc_int8_v2_forward``, jamais appelé sur le chemin 0x10).

Cycles → µs = cycles / 180 (SYSCLK 180 MHz). On reporte des **cycles bruts** par segment (la
conversion µs de profiling.c tronque à l'entier → un segment de dizaines de cycles = 0 µs).

Sortie :
  experiments/exp_S50_int8_latency/{monitoring,pronostia}.json  (cellules mesurées)
  experiments/exp_S50_int8_latency/ewc.json                     (agrégat, clé « segments »)

Règle « aucun chiffre inventé » : rien n'est écrit sans stream board réussi.

Usage :
    python scripts/run_s50_int8_latency.py                         # monitoring + pronostia
    python scripts/run_s50_int8_latency.py --dataset pronostia
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import _bss_bytes, train_maha_board  # noqa: E402
from scripts.run_sprint36_board import FW_DIR, GAP2_LATENCY_US  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402

EXPERIMENTS = Path("experiments")
OUT_DIR = EXPERIMENTS / "exp_S50_int8_latency"
CONDITION = "5feat"
CPU_HZ = 180_000_000
SEGMENTS = ("dequant", "mac", "requant")
# Slot de la réponse V3 (23 B) réinterprété en cycles bruts sous -DINT8_SEGMENT_PROFILE.
SEG_FIELD = {"dequant": "acc", "mac": "auroc", "requant": "forgetting"}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _pc_ckpt(dataset: str) -> Path:
    p = EXPERIMENTS / f"exp_S36_PC_{CONDITION}_ewc_{dataset}" / "checkpoints" / "ewc_head.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} absent — lancer scripts/run_sprint36_pc.py d'abord")
    return p


def build_and_flash(dataset: str, k: int, X: np.ndarray, pc_ckpt: Path,
                    exp_dir: Path, flash: bool = True) -> int:
    """Export → build INT8 v2 instrumenté (segment profile) → flash à la dim k. Retourne .bss."""
    maha_ckpt = train_maha_board(X, exp_dir)   # cohérence dims du build (non streamé)
    export_cmd = [sys.executable, "scripts/export_weights_c.py",
                  "--mahal", str(maha_ckpt), "--ewc-head", str(pc_ckpt),
                  "--int8-v2", str(pc_ckpt),
                  "--condition", CONDITION, "--dataset", dataset, "--model", "ewc"]
    if _run(export_cmd).returncode != 0:
        raise RuntimeError("export_weights_c échec")

    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    make_cmd = ["make", "-C", str(FW_DIR), *make_dims,
                "EXTRA_CFLAGS=-DEWC_INT8_V2 -DINT8_SEGMENT_PROFILE", "all"]
    if _run(make_cmd).returncode != 0:
        raise RuntimeError("make échec")
    bss = _bss_bytes()
    if flash and _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


def _stream(dataset: str, X: np.ndarray, y: np.ndarray, k: int, args,
            model_flags: int) -> list[dict]:
    return ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=1,
        rate_hz=args.rate_hz, request_update=False, verbose=args.verbose,
        protocol_version=3, model_flags=model_flags,
    )


def _pctile(vals: list[float], q: float) -> float | None:
    return round(float(np.percentile(vals, q)), 2) if vals else None


def _segment_stats(results: list[dict]) -> dict:
    """p50/p99/mean des cycles bruts par segment + conversion µs indicative."""
    out: dict = {}
    for seg in SEGMENTS:
        cyc = [float(r[SEG_FIELD[seg]]) for r in results if SEG_FIELD[seg] in r]
        out[seg] = {
            "cycles_p50": _pctile(cyc, 50),
            "cycles_p99": _pctile(cyc, 99),
            "cycles_mean": round(float(np.mean(cyc)), 2) if cyc else None,
            "us_p50": round(np.percentile(cyc, 50) / (CPU_HZ / 1e6), 3) if cyc else None,
        }
    return out


def run_cell(dataset: str, args) -> dict:
    print(f"\n{'='*70}\n=== S5004 latence INT8  dataset={dataset}  ===\n{'='*70}")
    X, y, idx, names = load_condition_arrays(dataset, CONDITION, "ewc", seed=42)
    k = len(idx)
    if args.max_samples and len(X) > args.max_samples:
        # Sous-échantillonnage borné : la latence est PAR inférence (frozen, déterministe),
        # les percentiles sur un échantillon varié sont représentatifs — aucun biais.
        rng = np.random.default_rng(42)
        sel = rng.choice(len(X), size=args.max_samples, replace=False)
        X, y = X[sel], y[sel]
    pc_ckpt = _pc_ckpt(dataset)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = OUT_DIR / f"cell_{dataset}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    bss = build_and_flash(dataset, k, X, pc_ckpt, exp_dir)

    # 1) INT8 (0x40) : segments DWT + latence totale INT8.
    res_int8 = _stream(dataset, X, y, k, args, ss.FRAME_FLAGS_INT8_MODE)
    if not res_int8:
        raise RuntimeError("stream INT8 vide (carte ?)")
    seg = _segment_stats(res_int8)
    lat_int8 = [r["latency_us"] for r in res_int8]
    crc_int8 = sum(1 for r in res_int8 if r["status"] & ss.STATUS_CRC_ERR)

    # 2) FP32 (0x10) : latence totale FP32, mêmes échantillons, même flash.
    res_fp32 = _stream(dataset, X, y, k, args, ss.FRAME_FLAGS_EWC_MODE)
    lat_fp32 = [r["latency_us"] for r in res_fp32] if res_fp32 else []

    int8_p50, fp32_p50 = _pctile(lat_int8, 50), _pctile(lat_fp32, 50)
    seg_sum_p50 = sum(seg[s]["cycles_p50"] for s in SEGMENTS
                      if seg[s]["cycles_p50"] is not None)

    cell = {
        "model": "ewc", "dataset": dataset, "platform": "nucleo_f439zi",
        "condition": CONDITION, "n_features": k, "feature_names": names,
        "cpu_hz": CPU_HZ, "kernel": "ewc_int8_v2", "date": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(res_int8), "bss_bytes": bss, "crc_errors": crc_int8,
        "segments": seg,
        "segments_sum_cycles_p50": round(seg_sum_p50, 2),
        "total_int8_us_p50": int8_p50, "total_int8_us_p99": _pctile(lat_int8, 99),
        "total_fp32_us_p50": fp32_p50, "total_fp32_us_p99": _pctile(lat_fp32, 99),
        "int8_minus_fp32_us_p50": (round(int8_p50 - fp32_p50, 2)
                                   if (int8_p50 is not None and fp32_p50 is not None) else None),
        "gap2_latency_compliant": (int8_p50 is not None and int8_p50 < GAP2_LATENCY_US),
        "note": ("cycles bruts DWT (180 MHz) ; INT8 ≥ FP32 sur Cortex-M4 FPU = paradoxe latence "
                 "S29 (déquant/requant ajoutés, MAC entier ≈ FPU). Gain INT8 = RAM ÷4, pas latence."),
    }
    (OUT_DIR / f"{dataset}.json").write_text(json.dumps(cell, indent=2))
    print(f"  k={k} .bss={bss} crc={crc_int8}  segments(cyc p50): "
          + " ".join(f"{s}={seg[s]['cycles_p50']}" for s in SEGMENTS)
          + f"\n  total INT8={int8_p50}µs  FP32={fp32_p50}µs  Δ={cell['int8_minus_fp32_us_p50']}µs")
    return cell


def _aggregate(cells: list[dict]) -> None:
    """ewc.json : agrégat mesuré (segments = moyenne des cycles p50 mesurés par dataset)."""
    seg_agg: dict = {}
    for s in SEGMENTS:
        p50s = [c["segments"][s]["cycles_p50"] for c in cells
                if c["segments"][s]["cycles_p50"] is not None]
        seg_agg[s] = {
            "cycles_p50_mean_over_datasets": round(float(np.mean(p50s)), 2) if p50s else None,
            "us_p50_mean_over_datasets": round(float(np.mean(p50s)) / (CPU_HZ / 1e6), 3) if p50s else None,
            "by_dataset_cycles_p50": {c["dataset"]: c["segments"][s]["cycles_p50"] for c in cells},
        }
    agg = {
        "model": "ewc", "platform": "nucleo_f439zi", "kernel": "ewc_int8_v2",
        "cpu_hz": CPU_HZ, "condition": CONDITION,
        "date": datetime.now().isoformat(timespec="seconds"),
        "datasets_measured": [c["dataset"] for c in cells],
        "segments": seg_agg,
        "total_int8_us_p50_by_dataset": {c["dataset"]: c["total_int8_us_p50"] for c in cells},
        "total_fp32_us_p50_by_dataset": {c["dataset"]: c["total_fp32_us_p50"] for c in cells},
        "note": ("Maha INT8 = N/A par construction (kernel déquant→distance, sans MAC/requant "
                 "par couche). Voir docs/context/int8_latency_breakdown.md."),
    }
    (OUT_DIR / "ewc.json").write_text(json.dumps(agg, indent=2))
    print(f"\nAgrégat → {OUT_DIR/'ewc.json'} (segments={list(seg_agg)})")


def main() -> None:
    p = argparse.ArgumentParser(description="S5004 — breakdown latence INT8 board (DWT)")
    p.add_argument("--dataset", choices=["monitoring", "pronostia"], action="append", default=None,
                   help="répétable ; défaut = monitoring + pronostia")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--max-samples", type=int, default=800,
                   help="borne le nombre d'inférences streamées par passe (0 = tout)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    datasets = args.dataset if args.dataset else ["monitoring", "pronostia"]
    cells = []
    for ds in datasets:
        try:
            cells.append(run_cell(ds, args))
        except Exception as exc:  # noqa: BLE001 — cellule robuste
            print(f"  [FAIL {ds}] {type(exc).__name__}: {exc}")
    if cells:
        _aggregate(cells)
    print(f"\n{'='*60}\nS5004 : {len(cells)} cellule(s) mesurée(s) → {OUT_DIR}/")


if __name__ == "__main__":
    main()
