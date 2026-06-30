#!/usr/bin/env python3
"""run_s29_board_int8.py — Orchestrateur d'expériences board INT8 (Sprint 29, S2904/S2905).

Pour un couple (modèle, dataset), mesure sur la NUCLEO-F439ZI la latence DWT,
la RAM (rapportée par le firmware) et la métrique de classification, en FP32 puis
en INT8, puis assemble un JSON au schéma S2904 (incluant les ratios et verdicts Gap 3).

Réutilise les internals de scripts/sensor_stream.py (chargement dataset + streaming UART).
Chaque appel à `_stream_uart` réinitialise la board via DTR → runs FP32/INT8 indépendants.

Métrique reportée :
  - ewc    → AUROC (classifieur binaire, AUROC firmware running)
  - hdc    → accuracy (multiclasse, AUROC firmware non significatif)
  - tinyol → AUROC (détection d'anomalie par reconstruction)

Usage :
  python scripts/run_s29_board_int8.py --model ewc --dataset cwru \\
      --n-samples 498 --output experiments/exp_S29_board_int8/results_ewc_int8_cwru.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

# ── Import dynamique de sensor_stream (script, pas un module installé) ─────────
_SS_PATH = Path(__file__).parent / "sensor_stream.py"
_spec = importlib.util.spec_from_file_location("sensor_stream", _SS_PATH)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

GAP2_LATENCY_LIMIT_US = 100_000.0  # 100 ms

# Flags FP32 / INT8 par modèle (cohérents pipeline.c). Le FP32 TinyOL utilise
# TINYOL_MODE (0x80) — non câblé dans sensor_stream.main, géré ici directement.
MODE_FLAGS = {
    "ewc":    (ss.FRAME_FLAGS_EWC_MODE,    ss.FRAME_FLAGS_INT8_MODE),
    "hdc":    (ss.FRAME_FLAGS_HDC_MODE,    ss.FRAME_FLAGS_HDC_INT8),
    "tinyol": (ss.FRAME_FLAGS_TINYOL_MODE, ss.FRAME_FLAGS_TINYOL_INT8),
    # Mahalanobis : chemin par défaut (flag 0x00) pour FP32 ET INT8 — la sélection se fait
    # à la COMPILATION (-DMAHA_INT8, nibble de flags protocole saturé, S2912). Le driver
    # d'extension (S2913) flashe 2 binaires distincts ; le flag de stream reste 0x00.
    "mahalanobis": (0x00, 0x00),
}

# Métrique reportée par modèle.
METRIC_NAME = {"ewc": "auroc", "hdc": "accuracy", "tinyol": "auroc", "mahalanobis": "auroc"}

# RAM « poids » analytique par modèle (octets), architecture-cohérente FP32 vs INT8.
# La RAM .bss totale rapportée par le firmware est constante (toutes les structs des
# modèles sont allouées simultanément) → non discriminante ; on compare ici l'empreinte
# des tableaux de poids quantifiables, indépendante du dataset (archi firmware fixe).
#   - EWC   : EWCHead vs EWCHeadInt8 (w/fisher/star + biais FP32), cf. ewc_head_int8.h / S23.
#   - HDC   : bv+am en FP32 hypothétique vs int8/int16, cf. hdc_int8.h (×3.06).
#   - TinyOL: encodeur w_enc1+w_enc2 FP32 vs INT8, cf. tinyol_int8.h (×4.0).
RAM_WEIGHTS = {
    "ewc":    {"fp32": 9728, "int8": 3600},     # S23 (mesuré/analytique), archi binaire fixe
    "hdc":    {"fp32": 106496, "int8": 34816},  # bv 9×2048 + am 4×2048 (FP32 vs int8/int16)
    "tinyol": {"fp32": 2688, "int8": 672},      # w_enc1 (32×5) + w_enc2 (16×32)
    # Mahalanobis (d=5) : mu (d×4) + sigma_inv (d²×4) FP32 = 120 B ; mu (d×1) + sigma (d²×1)
    # INT8 = 30 B → ×4.0 (S2912, empreinte poids analytique, indépendante du dataset).
    "mahalanobis": {"fp32": 120, "int8": 30},
}


def _run_one(port: str, baud: int, X, y, n_samples: int, n_tasks: int,
             rate_hz: float, model: str, model_flags: int) -> dict:
    """Un run streaming → dict {metric, latency_p50/p95/p99, ram_bytes, n, crc_errors}."""
    results = ss._stream_uart(
        port, baud, X, y, n_samples, n_tasks, rate_hz,
        request_update=True, verbose=False,
        protocol_version=3, model_flags=model_flags,
    )
    if not results:
        raise RuntimeError("Aucune réponse de la board (0 échantillon reçu)")

    lat = np.array([r["latency_us"] for r in results], dtype=float)
    preds = [r["pred"] for r in results]
    trues = [r["true"] for r in results]
    accuracy = sum(p == t for p, t in zip(preds, trues)) / len(results)
    # AUROC firmware : valeur courante cumulée → on prend celle du dernier échantillon.
    final_auroc = float(results[-1].get("auroc", float("nan")))
    crc_errors = sum(1 for r in results if r.get("status", 0) & ss.STATUS_CRC_ERR)

    metric_value = final_auroc if METRIC_NAME[model] == "auroc" else round(accuracy, 4)

    return {
        "n_samples": len(results),
        "metric_value": round(float(metric_value), 4),
        "accuracy": round(accuracy, 4),
        "auroc": round(final_auroc, 4),
        "latency_p50_us": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_us": round(float(np.percentile(lat, 95)), 2),
        "latency_p99_us": round(float(np.percentile(lat, 99)), 2),
        "ram_bytes": int(np.mean([r["ram_bytes"] for r in results])),
        "crc_errors": crc_errors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["ewc", "hdc", "tinyol", "mahalanobis"], required=True)
    p.add_argument("--dataset",
                   choices=["cwru", "monitoring", "pronostia", "cmapss", "paderborn"],
                   required=True)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--n-tasks", type=int, default=3)
    p.add_argument("--rate-hz", type=float, default=0.0)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-fp32", action="store_true",
                   help="Ne mesure que l'INT8 (FP32 = N/A dans le JSON)")
    args = p.parse_args()

    fp32_flag, int8_flag = MODE_FLAGS[args.model]

    print(f"Chargement dataset '{args.dataset}'...")
    X, y = ss._load_dataset(args.dataset)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    fp32 = None
    if not args.skip_fp32:
        print(f"\n=== Run FP32 ({args.model}) ===")
        fp32 = _run_one(args.port, args.baud, X, y, args.n_samples, args.n_tasks,
                        args.rate_hz, args.model, fp32_flag)
        print(f"  {fp32}")

    print(f"\n=== Run INT8 ({args.model}) ===")
    int8 = _run_one(args.port, args.baud, X, y, args.n_samples, args.n_tasks,
                    args.rate_hz, args.model, int8_flag)
    print(f"  {int8}")

    out = assemble_result(args.model, args.dataset, fp32, int8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSauvegardé : {out_path}")
    print(json.dumps(out, indent=2))


def assemble_result(model: str, dataset: str, fp32: dict | None, int8: dict,
                    na_reason: str | None = None) -> dict:
    """Assemble le JSON schéma S2904 depuis les runs FP32/INT8 (source unique, S2913).

    Si ``na_reason`` est fourni, ``metric_value`` est mis à ``None`` (combo dégénéré, ex.
    Paderborn mono-classe) — règle CLAUDE.md : ne jamais forcer un chiffre non significatif.
    La latence et la RAM restent valides et écrites dans tous les cas.
    """
    lat_ratio = round(int8["latency_p50_us"] / fp32["latency_p50_us"], 3) if fp32 else None
    gap3_latency_ok = (int8["latency_p50_us"] < fp32["latency_p50_us"]) if fp32 else None

    # RAM « poids » analytique (architecture-cohérente, indépendante du dataset).
    ram_fp32 = RAM_WEIGHTS[model]["fp32"]
    ram_int8 = RAM_WEIGHTS[model]["int8"]
    ram_ratio = round(ram_fp32 / ram_int8, 3)
    gap3_ram_ok = bool(ram_int8 < ram_fp32)

    metric_value = None if na_reason else int8["metric_value"]

    out = {
        "model": f"{model}_int8",
        "dataset": dataset,
        "board": "NUCLEO-F439ZI",
        "precision": "INT8",
        "n_samples": int8["n_samples"],
        "metric_name": METRIC_NAME[model],
        "metric_value": metric_value,
        "latency_dwt_us": {
            "p50": int8["latency_p50_us"],
            "p95": int8["latency_p95_us"],
            "p99": int8["latency_p99_us"],
        },
        "ram_weights_int8_bytes": ram_int8,
        "ram_weights_fp32_bytes": ram_fp32,
        "ram_bss_total_bytes": int8["ram_bytes"],  # .bss totale rapportée firmware (constante)
        "gap2_compliant": bool(int8["latency_p99_us"] < GAP2_LATENCY_LIMIT_US),
        "gap3_latency_ok": gap3_latency_ok,
        "gap3_ram_ok": gap3_ram_ok,
        "fp32_reference": fp32,
        "int8_detail": int8,
        "latency_ratio_int8_over_fp32": lat_ratio,
        "ram_ratio_fp32_over_int8": ram_ratio,
        "crc_errors": int8["crc_errors"] + (fp32["crc_errors"] if fp32 else 0),
    }
    if na_reason:
        out["na_reason"] = na_reason
    return out


if __name__ == "__main__":
    main()
