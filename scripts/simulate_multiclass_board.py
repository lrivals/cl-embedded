#!/usr/bin/env python3
"""
simulate_multiclass_board.py — Envoie échantillons CWRU (9 features) via UART,
collecte classes prédites, calcule F1-macro on-board.

Usage :
    python scripts/simulate_multiclass_board.py \\
        --port /dev/ttyACM0 \\
        --csv-path "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv" \\
        --n-samples-per-task 100 \\
        --output experiments/exp_S26_02/board_mc_results.json \\
        [--n-classes 10] [--baud 115200]

Protocole : v3 binaire, flag MULTICLASS_MODE=0x30 (EWC_MODE|HDC_MODE), réponse 21 B.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import serial
from sklearn.metrics import f1_score

from src.data.cwru_dataset import get_cl_splits

# ── Constantes protocole (miroir de pipeline.h) ────────────────────────────
PROTO_MAGIC = b"\xCD\xAB"
PROTO_VERSION_V3 = 0x03
PROTO_FLAG_UPDATE = 0x01
PROTO_FLAG_CONSOLIDATE = 0x04
PROTO_FLAG_EWC_MODE = 0x10
PROTO_FLAG_HDC_MODE = 0x20
# MULTICLASS_MODE = EWC_MODE | HDC_MODE (0x30) — pipeline.h Sprint 26
# NB: la spec doc écrivait 0x90 (EWC_MODE|TINYOL_MODE), valeur incorrecte.
PROTO_FLAG_MULTICLASS_MODE = PROTO_FLAG_EWC_MODE | PROTO_FLAG_HDC_MODE  # 0x30

# Réponse v3 = 23 B (miroir pipeline.c::uart_send_response_v3) :
# [pred:u8][conf:f32][lat_us:u32][ram_b:u16][acc:f32][auroc:f32][forgetting:f32]
RESPONSE_SIZE = 23


def crc8(data: bytes) -> int:
    """Polynomial 0x07 — identique à pipeline.c::proto_crc8."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_frame(
    features: np.ndarray,
    label: int,
    task_id: int,
    flags: int,
    timestamp_ms: int,
) -> bytes:
    """Construit une trame UART v3 binaire pour classification multiclasse."""
    n = len(features)
    payload = PROTO_MAGIC
    payload += struct.pack("<B", PROTO_VERSION_V3)
    payload += struct.pack("<B", task_id)
    payload += struct.pack("<I", timestamp_ms & 0xFFFFFFFF)
    payload += struct.pack("<B", n)
    for f in features:
        payload += struct.pack("<f", float(f))
    payload += struct.pack("<B", label & 0xFF)
    payload += struct.pack("<B", flags)
    payload += struct.pack("<B", crc8(payload))
    return payload


def parse_response(data: bytes) -> dict:
    """Décode une réponse v3 de 23 B (offsets miroir de uart_send_response_v3)."""
    (pred,) = struct.unpack_from("<B", data, 0)
    (conf,) = struct.unpack_from("<f", data, 1)
    (lat,) = struct.unpack_from("<I", data, 5)
    (ram,) = struct.unpack_from("<H", data, 9)
    (acc,) = struct.unpack_from("<f", data, 11)
    (aur,) = struct.unpack_from("<f", data, 15)
    (fgt,) = struct.unpack_from("<f", data, 19)
    return {
        "pred": pred,
        "conf": conf,
        "lat_us": lat,
        "ram_b": ram,
        "acc": acc,
        "auroc": aur,
        "forgetting": fgt,
    }


def run_simulation(
    port: str,
    csv_path: str,
    n_samples_per_task: int,
    output_path: Path,
    n_classes: int = 10,
    baud: int = 115200,
    no_update: bool = False,
) -> None:
    tasks = get_cl_splits(
        csv_path=csv_path,
        scenario="by_fault_type",
        mode="multiclass",
    )

    ser = serial.Serial(port, baud, timeout=2.0)
    time.sleep(1.0)
    ser.reset_input_buffer()   # vider les octets stale post-reset MCU

    all_preds: list[int] = []
    all_trues: list[int] = []
    latencies: list[int] = []

    # --no-update : inférence pure sur les poids entraînés figés (FLAG_UPDATE off)
    # → isole la qualité du modèle chargé du régime d'apprentissage online single-pass.
    mode_label = "inférence pure (no-update)" if no_update else "online single-pass (FLAG_UPDATE)"
    print(f"Mode : {mode_label}")

    for task_id, task in enumerate(tasks):
        print(f"\n── Tâche {task_id} ──")
        flags_base = PROTO_FLAG_MULTICLASS_MODE
        if not no_update:
            flags_base |= PROTO_FLAG_UPDATE
        X_task = task["X_train"]
        y_task = task["y_train"]
        n_avail = min(n_samples_per_task, len(X_task))
        last_idx = n_avail - 1

        for i in range(n_avail):
            x = X_task[i].astype(np.float32)
            y = int(y_task[i])
            # Pas de consolidation en mode inférence pure (rien à consolider)
            consolidate = (not no_update) and (i == last_idx)
            flags = flags_base | (PROTO_FLAG_CONSOLIDATE if consolidate else 0)

            frame = build_frame(
                x,
                y,
                task_id=task_id,
                flags=flags,
                timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
            )
            ser.write(frame)
            resp = ser.read(RESPONSE_SIZE)
            if len(resp) < RESPONSE_SIZE:
                print(f"[WARN] Sample {i}: réponse tronquée ({len(resp)} B)")
                continue

            r = parse_response(resp)
            all_preds.append(r["pred"])
            all_trues.append(y)
            latencies.append(r["lat_us"])

        f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)
        print(f"  F1-macro après task{task_id} : {f1:.3f} ({len(all_preds)} samples)")

    ser.close()

    if not all_preds:
        print("[ERROR] Aucun échantillon reçu.")
        return

    f1_final = f1_score(all_trues, all_preds, average="macro", zero_division=0)
    lat_s = sorted(latencies)
    n = len(lat_s)

    results = {
        "n_classes": n_classes,
        "n_tasks": len(tasks),
        "n_samples": len(all_preds),
        "mode": "inference_only" if no_update else "online_update",
        "f1_macro_board": f1_final,
        "latency_p50_us": lat_s[n // 2],
        "latency_p99_us": lat_s[int(n * 0.99)],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ F1-macro board = {f1_final:.3f} (critère ≥ 0.60)")
    print(f"Résultats → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation multiclasse board CWRU via UART protocole v3"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--csv-path",
        default="data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv",
        help="Chemin vers le CSV CWRU features",
    )
    parser.add_argument("--n-samples-per-task", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/exp_S26_02/board_mc_results.json"),
    )
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Inférence pure : n'envoie pas FLAG_UPDATE (poids entraînés figés)",
    )
    args = parser.parse_args()

    run_simulation(
        port=args.port,
        csv_path=args.csv_path,
        n_samples_per_task=args.n_samples_per_task,
        output_path=args.output,
        n_classes=args.n_classes,
        baud=args.baud,
        no_update=args.no_update,
    )


if __name__ == "__main__":
    main()
