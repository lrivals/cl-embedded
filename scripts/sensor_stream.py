"""
sensor_stream.py — Streaming continu PC → carte STM32 (protocole UART v2).

Extension de sensor_sim.py : supporte multi-tâches, timestamps, rate-limiting,
et le protocole v2 étendu (task_id, FLAGS, réponse 14 B).

Protocole v2 :
    Trame envoyée :
        [MAGIC 0xABCD:2B] [VERSION:1B=0x02] [TASK_ID:1B] [TIMESTAMP_MS:4B]
        [N:1B] [features:f32×N] [label:1B] [FLAGS:1B] [CRC8:1B]
    Réponse firmware (14 B) :
        [pred:u8] [conf:f32] [latency_us:u32] [ram_b:u16] [throughput:u16] [status:u8]

Usage :
    # Dry-run (pas de board)
    python scripts/sensor_stream.py --dataset cwru --dry-run --n-samples 200

    # Board connectée, multi-tâches
    python scripts/sensor_stream.py --dataset monitoring --port /dev/ttyACM0 \\
        --n-samples 100 --rate-hz 10 --update --output experiments/exp_S18_01/stream.json
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Iterator

import numpy as np

PROTO_VERSION = 0x02
MAGIC = 0xABCD
UART_TIMEOUT_S = 2.0

FRAME_FMT_HDR = "<HBBIB"    # magic(u16), version(u8), task_id(u8), ts_ms(u32), n(u8)
FRAME_FLAGS_UPDATE    = 0x01
FRAME_FLAGS_PROFILING = 0x02

RESPONSE_V2_FMT  = "<BfIHHB"   # pred(u8), conf(f32), lat_us(u32), ram(u16), thr(u16), status(u8)
RESPONSE_V2_SIZE = struct.calcsize(RESPONSE_V2_FMT)  # 14 B

STATUS_OK         = 0x00
STATUS_CRC_ERR    = 0x01
STATUS_OOB        = 0x02
STATUS_UPDATE_DONE = 0x04


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc


def build_frame_v2(features: np.ndarray, label: int, task_id: int,
                   ts_ms: int, flags: int = 0) -> bytes:
    n = len(features)
    header = struct.pack("<HBBIB", MAGIC, PROTO_VERSION, task_id & 0xFF,
                         ts_ms & 0xFFFFFFFF, n)
    feat_bytes = features.astype(np.float32).tobytes()
    tail = struct.pack("<BB", label & 0xFF, flags & 0xFF)
    payload = header + feat_bytes + tail
    return payload + struct.pack("<B", crc8(payload))


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Charge un dataset Phase 1 (réutilise sensor_sim.py)."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "sensor_sim", Path(__file__).parent / "sensor_sim.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_dataset(name)


def _make_task_splits(
    X: np.ndarray, y: np.ndarray, n_tasks: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Découpe X/y en n_tasks tranches temporelles égales."""
    size = len(X) // n_tasks
    return [(X[i * size:(i + 1) * size], y[i * size:(i + 1) * size])
            for i in range(n_tasks)]


def _stream_dry_run(
    X: np.ndarray, y: np.ndarray, n_samples: int,
    n_tasks: int, request_update: bool, verbose: bool
) -> list[dict]:
    tasks = _make_task_splits(X, y, n_tasks)
    results = []
    t0_ms = int(time.time() * 1000)

    for task_id, (Xt, yt) in enumerate(tasks):
        per_task = n_samples // n_tasks
        indices = np.random.choice(len(Xt), size=min(per_task, len(Xt)), replace=False)
        for idx in indices:
            features, label = Xt[idx], int(yt[idx])
            flags = FRAME_FLAGS_UPDATE | FRAME_FLAGS_PROFILING if request_update else FRAME_FLAGS_PROFILING
            ts_ms = int(time.time() * 1000) - t0_ms
            frame = build_frame_v2(features, label, task_id, ts_ms, flags)
            payload, recv_crc = frame[:-1], frame[-1]
            ok = crc8(payload) == recv_crc
            results.append({
                "task_id": task_id,
                "ts_ms": ts_ms,
                "true": label,
                "pred": label,
                "confidence": 1.0,
                "latency_us": 3,
                "ram_bytes": 200,
                "throughput_ips": 333333,
                "status": STATUS_OK if ok else STATUS_CRC_ERR,
            })
            if verbose:
                print(f"[task={task_id} ts={ts_ms}ms] label={label} → OK (dry-run)")
    return results


def _stream_uart(
    port: str, baud: int,
    X: np.ndarray, y: np.ndarray,
    n_samples: int, n_tasks: int,
    rate_hz: float, request_update: bool, verbose: bool,
) -> list[dict]:
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial requis : pip install pyserial")

    results = []
    tasks = _make_task_splits(X, y, n_tasks)
    t0_ms = int(time.time() * 1000)
    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0

    with serial.Serial(port, baud, timeout=UART_TIMEOUT_S,
                       dsrdtr=False, rtscts=False) as ser:
        ser.dtr = True
        time.sleep(0.05)
        ser.dtr = False
        time.sleep(0.5)
        ser.reset_input_buffer()

        for task_id, (Xt, yt) in enumerate(tasks):
            per_task = n_samples // n_tasks
            indices = np.random.choice(len(Xt), size=min(per_task, len(Xt)), replace=False)
            for idx in indices:
                t_send = time.monotonic()
                features, label = Xt[idx], int(yt[idx])
                flags = FRAME_FLAGS_UPDATE | FRAME_FLAGS_PROFILING if request_update else FRAME_FLAGS_PROFILING
                ts_ms = int(time.time() * 1000) - t0_ms
                frame = build_frame_v2(features, label, task_id, ts_ms, flags)

                ser.write(frame)
                raw = ser.read(RESPONSE_V2_SIZE)
                if len(raw) != RESPONSE_V2_SIZE:
                    if verbose:
                        print(f"[WARN] Timeout task={task_id} ({len(raw)}/{RESPONSE_V2_SIZE} B)")
                    continue

                pred, conf, lat_us, ram_b, thr, status = struct.unpack(RESPONSE_V2_FMT, raw)
                results.append({
                    "task_id": task_id,
                    "ts_ms": ts_ms,
                    "true": label,
                    "pred": pred,
                    "confidence": float(conf),
                    "latency_us": lat_us,
                    "ram_bytes": ram_b,
                    "throughput_ips": thr,
                    "status": status,
                })
                if verbose:
                    print(f"[task={task_id} ts={ts_ms}ms] true={label} pred={pred} "
                          f"conf={conf:.3f} lat={lat_us}µs ram={ram_b}B thr={thr}/s")

                elapsed = time.monotonic() - t_send
                if interval_s > elapsed:
                    time.sleep(interval_s - elapsed)

    return results


def _compute_stats(results: list[dict]) -> dict:
    if not results:
        return {"n_samples": 0, "accuracy": 0.0}

    latencies = [r["latency_us"] for r in results]
    preds = [r["pred"] for r in results]
    trues = [r["true"] for r in results]
    acc = sum(p == t for p, t in zip(preds, trues)) / len(results)
    n_tasks = len(set(r["task_id"] for r in results))

    return {
        "n_samples": len(results),
        "n_tasks": n_tasks,
        "accuracy": round(acc, 4),
        "latency_mean_us": round(float(np.mean(latencies)), 2),
        "latency_p50_us":  round(float(np.percentile(latencies, 50)), 2),
        "latency_p99_us":  round(float(np.percentile(latencies, 99)), 2),
        "ram_mean_bytes":  int(np.mean([r["ram_bytes"] for r in results])),
        "throughput_mean_ips": int(np.mean([r["throughput_ips"] for r in results])),
        "crc_errors": sum(1 for r in results if r["status"] & STATUS_CRC_ERR),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming continu de données vers firmware STM32 (protocole v2)")
    parser.add_argument("--dataset", choices=["cwru", "monitoring"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-tasks", type=int, default=3, help="Nombre de tâches CL simulées")
    parser.add_argument("--rate-hz", type=float, default=0.0, help="Rate-limit (0=max speed)")
    parser.add_argument("--update", action="store_true", help="Demande mise à jour incrémentale au firmware")
    parser.add_argument("--output", type=str, help="Chemin JSON pour les statistiques")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Chargement dataset '{args.dataset}'...")
    X, y = _load_dataset(args.dataset)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    if args.dry_run:
        raw_results = _stream_dry_run(X, y, args.n_samples, args.n_tasks,
                                       args.update, args.verbose)
        stats = _compute_stats(raw_results)
        stats["mode"] = "dry-run"
    else:
        raw_results = _stream_uart(args.port, args.baud, X, y, args.n_samples,
                                    args.n_tasks, args.rate_hz, args.update, args.verbose)
        stats = _compute_stats(raw_results)
        stats["mode"] = "uart"
        stats["port"] = args.port

    print("\n--- Résultats streaming ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2))
        print(f"\nSauvegardé : {out}")


if __name__ == "__main__":
    main()
