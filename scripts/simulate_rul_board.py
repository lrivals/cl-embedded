#!/usr/bin/env python3
"""
simulate_rul_board.py — Envoie séquences CMAPSS FD001 via UART à la NUCLEO-F439ZI,
collecte les prédictions RUL, calcule RMSE vs labels réels et log latence DWT.

Usage :
    python scripts/simulate_rul_board.py \\
        --port /dev/ttyACM0 \\
        --config configs/cmapss_feature_subset.yaml \\
        --n-samples 200 \\
        --output experiments/exp_S26_01/board_rul_results.json \\
        [--data-dir data/raw/cmapss] [--baud 115200] [--task-id 0]
        [--update] [--consolidate-at 100]

Protocole : v3 binaire, flag RUL_MODE=0x50 (EWC_MODE|INT8_MODE), réponse 21 B.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import serial

from src.data.cmapss_loader import CMAPSS_RUL_CAP, get_cl_dataloaders

# ── Constantes protocole (miroir de pipeline.h) ────────────────────────────
PROTO_MAGIC = b"\xCD\xAB"
PROTO_VERSION_V3 = 0x03
PROTO_FLAG_UPDATE = 0x01
PROTO_FLAG_CONSOLIDATE = 0x04
PROTO_FLAG_EWC_MODE = 0x10
PROTO_FLAG_INT8_MODE = 0x40
# RUL_MODE = EWC_MODE | INT8_MODE (0x50) — pipeline.h Sprint 26
# NB: la spec doc écrivait 0x80 (TINYOL_MODE), valeur incorrecte.
PROTO_FLAG_RUL_MODE = PROTO_FLAG_EWC_MODE | PROTO_FLAG_INT8_MODE  # 0x50

# Réponse v3 = 23 B (miroir pipeline.c::uart_send_response_v3) :
# [pred:u8][conf:f32][lat_us:u32][ram_b:u16][acc:f32][auroc:f32][forgetting:f32]
# En mode RUL : conf transporte le RUL prédit, auroc transporte l'OnlineRMSE board.
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
    label_rul: float,
    task_id: int,
    flags: int,
    timestamp_ms: int,
) -> bytes:
    """
    Construit une trame UART v3 binaire.

    RUL est normalisé sur [0, 255] via CMAPSS_RUL_CAP (125) avant encodage uint8.
    Résolution : ~0.49 cycles/bit. Documenté dans le manuscrit (Question ouverte gap2).
    """
    n = len(features)
    # Normalise RUL ∈ [0, RUL_CAP] → uint8 [0, 255]
    label_u8 = int(round(min(label_rul, CMAPSS_RUL_CAP) / CMAPSS_RUL_CAP * 255))

    payload = PROTO_MAGIC
    payload += struct.pack("<B", PROTO_VERSION_V3)
    payload += struct.pack("<B", task_id)
    payload += struct.pack("<I", timestamp_ms & 0xFFFFFFFF)
    payload += struct.pack("<B", n)

    for f in features:
        payload += struct.pack("<f", float(f))

    payload += struct.pack("<B", label_u8)
    payload += struct.pack("<B", flags)
    payload += struct.pack("<B", crc8(payload))
    return payload


def parse_response(data: bytes) -> dict:
    """Décode une réponse v3 de 23 B (offsets miroir de uart_send_response_v3)."""
    (pred,) = struct.unpack_from("<B", data, 0)
    (conf,) = struct.unpack_from("<f", data, 1)  # RUL prédit en mode RUL_MODE
    (lat,) = struct.unpack_from("<I", data, 5)
    (ram,) = struct.unpack_from("<H", data, 9)
    (acc,) = struct.unpack_from("<f", data, 11)
    (aur,) = struct.unpack_from("<f", data, 15)
    (fgt,) = struct.unpack_from("<f", data, 19)
    return {
        "pred": pred,
        "rul_pred": conf,
        "lat_us": lat,
        "ram_b": ram,
        "acc": acc,
        "auroc": aur,
        "forgetting": fgt,
    }


def run_simulation(
    port: str,
    data_dir: Path,
    config_path: Path,
    n_samples: int,
    output_path: Path,
    baud: int = 115200,
    task_id: int = 0,
    do_update: bool = True,
    consolidate_at: int | None = None,
) -> None:
    tasks = get_cl_dataloaders(data_dir=data_dir, config_path=config_path, mode="rul")
    task = tasks[task_id]

    ser = serial.Serial(port, baud, timeout=2.0)
    time.sleep(1.0)
    ser.reset_input_buffer()   # vider les octets stale post-reset MCU

    rul_preds: list[float] = []
    rul_trues: list[float] = []
    latencies: list[int] = []

    flags_base = PROTO_FLAG_RUL_MODE
    if do_update:
        flags_base |= PROTO_FLAG_UPDATE

    for i, (x_batch, y_batch) in enumerate(task["train_loader"]):
        if i >= n_samples:
            break

        x = x_batch[0].numpy()
        y = float(y_batch[0].item())

        flags = flags_base
        if consolidate_at is not None and i == consolidate_at:
            flags |= PROTO_FLAG_CONSOLIDATE

        frame = build_frame(
            x,
            y,
            task_id=task_id,
            flags=flags,
            timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
        )
        ser.write(frame)

        resp_bytes = ser.read(RESPONSE_SIZE)
        if len(resp_bytes) < RESPONSE_SIZE:
            print(f"[WARN] Sample {i}: réponse tronquée ({len(resp_bytes)} B)")
            continue

        r = parse_response(resp_bytes)
        # La board retourne RUL en [0,1] (normalisé par RUL_CAP) → rescaler en cycles
        rul_preds.append(r["rul_pred"] * CMAPSS_RUL_CAP)
        rul_trues.append(y)
        latencies.append(r["lat_us"])

        if i % 20 == 0:
            rmse = math.sqrt(
                sum((p - t) ** 2 for p, t in zip(rul_preds, rul_trues)) / len(rul_preds)
            )
            print(
                f"[{i:4d}] RUL_pred={r['rul_pred']:.1f} true={y:.1f} "
                f"RMSE={rmse:.2f} lat={r['lat_us']}µs"
            )

    ser.close()

    if not rul_preds:
        print("[ERROR] Aucun échantillon reçu.")
        return

    rmse_final = math.sqrt(
        sum((p - t) ** 2 for p, t in zip(rul_preds, rul_trues)) / len(rul_preds)
    )
    lat_arr = sorted(latencies)
    n = len(lat_arr)

    results = {
        "n_samples": len(rul_preds),
        "rmse_board": rmse_final,
        "latency_p50_us": lat_arr[n // 2],
        "latency_p99_us": lat_arr[int(n * 0.99)],
        "latency_max_us": lat_arr[-1],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ RMSE board = {rmse_final:.2f} | lat P50={results['latency_p50_us']}µs")
    print(f"Résultats → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation RUL board CMAPSS via UART protocole v3"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/cmapss"),
        help="Répertoire racine des données CMAPSS",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cmapss_feature_subset.yaml"),
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/exp_S26_01/board_rul_results.json"),
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--update", action="store_true", default=True)
    parser.add_argument("--consolidate-at", type=int, default=None)
    args = parser.parse_args()

    run_simulation(
        port=args.port,
        data_dir=args.data_dir,
        config_path=args.config,
        n_samples=args.n_samples,
        output_path=args.output,
        baud=args.baud,
        task_id=args.task_id,
        do_update=args.update,
        consolidate_at=args.consolidate_at,
    )


if __name__ == "__main__":
    main()
