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
FRAME_FLAGS_UPDATE      = 0x01
FRAME_FLAGS_PROFILING   = 0x02
FRAME_FLAGS_CONSOLIDATE = 0x04   # frontière de tâche → ewc_consolidate() firmware
FRAME_FLAGS_RESET       = 0x08   # réinitialise poids EWC → ewc_init() + reset métriques
FRAME_FLAGS_EWC_MODE    = 0x10   # utilise EWC head pour inférence (au lieu de Mahalanobis)

RESPONSE_V2_FMT  = "<BfIHHB"   # pred(u8), conf(f32), lat_us(u32), ram(u16), thr(u16), status(u8)
RESPONSE_V2_SIZE = struct.calcsize(RESPONSE_V2_FMT)  # 14 B

RESPONSE_V3_FMT  = "<BfIfff"   # pred(u8), conf(f32), lat_us(u32), acc(f32), auroc(f32), forgetting(f32)
RESPONSE_V3_SIZE = struct.calcsize(RESPONSE_V3_FMT)  # 21 B

STATUS_OK         = 0x00
STATUS_CRC_ERR    = 0x01
STATUS_OOB        = 0x02
STATUS_UPDATE_DONE = 0x04


def parse_response(data: bytes) -> dict:
    """Parse une réponse firmware UART v2 (14 B) ou v3 (21 B)."""
    if len(data) == RESPONSE_V2_SIZE:
        pred, conf, lat_us, ram_b, thr, status = struct.unpack(RESPONSE_V2_FMT, data)
        return {
            "pred": pred, "confidence": float(conf), "latency_us": lat_us,
            "ram_bytes": ram_b, "throughput_ips": thr, "status": status,
        }
    elif len(data) == RESPONSE_V3_SIZE:
        pred, conf, lat_us, acc, auroc, forgetting = struct.unpack(RESPONSE_V3_FMT, data)
        return {
            "pred": pred, "confidence": float(conf), "latency_us": lat_us,
            "ram_bytes": 0, "throughput_ips": 0, "status": STATUS_OK,
            "acc": float(acc), "auroc": float(auroc), "forgetting": float(forgetting),
        }
    raise ValueError(f"Unknown response length: {len(data)}")


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc


def send_reset_frame(ser: "serial.Serial", lambda_ewc: float = 0.0) -> None:
    """Envoie FRAME_FLAGS_RESET (0x08) pour réinitialiser les poids EWC sur board.

    raw[0] encode lambda_ewc (> 0 pour modifier la valeur courante firmware).
    Consomme la réponse V3 (21 B) du firmware.
    """
    features = np.array([lambda_ewc], dtype=np.float32)
    frame = build_frame_v2(features, 0, task_id=0xFF, ts_ms=0, flags=FRAME_FLAGS_RESET)
    ser.write(frame)
    _ = ser.read(RESPONSE_V3_SIZE)   # consomme la réponse reset


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
    n_tasks: int, request_update: bool, verbose: bool,
    protocol_version: int = 2,
    model_flags: int = 0,
) -> list[dict]:
    tasks = _make_task_splits(X, y, n_tasks)
    results = []
    t0_ms = int(time.time() * 1000)

    for task_id, (Xt, yt) in enumerate(tasks):
        per_task = n_samples // n_tasks
        indices = np.random.choice(len(Xt), size=min(per_task, len(Xt)), replace=False)
        for idx in indices:
            features, label = Xt[idx], int(yt[idx])
            flags = (FRAME_FLAGS_UPDATE | FRAME_FLAGS_PROFILING if request_update else FRAME_FLAGS_PROFILING) | model_flags
            ts_ms = int(time.time() * 1000) - t0_ms
            frame = build_frame_v2(features, label, task_id, ts_ms, flags)
            payload, recv_crc = frame[:-1], frame[-1]
            ok = crc8(payload) == recv_crc
            entry: dict = {
                "task_id": task_id,
                "ts_ms": ts_ms,
                "true": label,
                "pred": label,
                "confidence": 1.0,
                "latency_us": 3,
                "ram_bytes": 200,
                "throughput_ips": 333333,
                "status": STATUS_OK if ok else STATUS_CRC_ERR,
            }
            if protocol_version >= 3:
                entry["acc"] = 0.9
                entry["auroc"] = 0.75
                entry["forgetting"] = 0.01
            results.append(entry)
            if verbose:
                print(f"[task={task_id} ts={ts_ms}ms] label={label} → OK (dry-run)")
    return results


def _stream_uart(
    port: str, baud: int,
    X: np.ndarray, y: np.ndarray,
    n_samples: int, n_tasks: int,
    rate_hz: float, request_update: bool, verbose: bool,
    protocol_version: int = 2,
    model_flags: int = 0,
    reset_lambda: float | None = None,
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

        if reset_lambda is not None:
            send_reset_frame(ser, reset_lambda)

        for task_id, (Xt, yt) in enumerate(tasks):
            per_task = n_samples // n_tasks
            indices = np.random.choice(len(Xt), size=min(per_task, len(Xt)), replace=False)
            for idx in indices:
                t_send = time.monotonic()
                features, label = Xt[idx], int(yt[idx])
                flags = ((FRAME_FLAGS_UPDATE | FRAME_FLAGS_PROFILING if request_update else FRAME_FLAGS_PROFILING)
                         | model_flags)
                ts_ms = int(time.time() * 1000) - t0_ms
                frame = build_frame_v2(features, label, task_id, ts_ms, flags)

                ser.write(frame)
                if protocol_version >= 3:
                    resp_fmt, resp_size = RESPONSE_V3_FMT, RESPONSE_V3_SIZE
                else:
                    resp_fmt, resp_size = RESPONSE_V2_FMT, RESPONSE_V2_SIZE

                raw = ser.read(resp_size)
                if len(raw) != resp_size:
                    if verbose:
                        print(f"[WARN] Timeout task={task_id} ({len(raw)}/{resp_size} B)")
                    continue

                entry: dict = {"task_id": task_id, "ts_ms": ts_ms, "true": label}
                entry.update(parse_response(raw))
                results.append(entry)
                if verbose:
                    pred = entry["pred"]
                    conf = entry["confidence"]
                    lat_us = entry["latency_us"]
                    if "acc" in entry:
                        print(f"[task={task_id} ts={ts_ms}ms] true={label} pred={pred} "
                              f"conf={conf:.3f} lat={lat_us}µs "
                              f"acc={entry['acc']:.3f} auroc={entry['auroc']:.3f} fgt={entry['forgetting']:.3f}")
                    else:
                        print(f"[task={task_id} ts={ts_ms}ms] true={label} pred={pred} "
                              f"conf={conf:.3f} lat={lat_us}µs "
                              f"ram={entry['ram_bytes']}B thr={entry['throughput_ips']}/s")

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


def parse_cl_sequence(s: str) -> list[tuple[str, int]]:
    """Parse 'pump:167,turbine:167,compressor:166' → [(name, n_samples), ...]."""
    segments = []
    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"--cl-sequence: format attendu name:n_samples, reçu {part!r}")
        name, n_str = part.split(":", 1)
        segments.append((name.strip(), int(n_str.strip())))
    return segments


def _stream_cl_sequence(
    X: np.ndarray, y: np.ndarray,
    segments: list[tuple[str, int]],
    request_update: bool,
    consolidate: bool,
    verbose: bool,
    dry_run: bool,
    port: str = "/dev/ttyACM0",
    baud: int = 115200,
    rate_hz: float = 0.0,
    protocol_version: int = 2,
    output_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Stream une séquence CL domain-incremental tâche par tâche.

    Retourne (all_results, per_task_metrics) où per_task_metrics[i] contient
    les stats de la tâche i (accuracy, forgetting, etc.).
    """
    all_results: list[dict] = []
    per_task_metrics: list[dict] = []
    t0_ms = int(time.time() * 1000)

    n_tasks = len(segments)
    ser = None
    if not dry_run:
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial requis : pip install pyserial")
        ser = __import__("serial").Serial(
            port, baud, timeout=UART_TIMEOUT_S, dsrdtr=False, rtscts=False
        )
        ser.dtr = True
        time.sleep(0.05)
        ser.dtr = False
        time.sleep(0.5)
        ser.reset_input_buffer()

    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    total_samples = sum(n for _, n in segments)
    offset = 0   # indice global dans X/y

    try:
        for task_id, (task_name, n_samples) in enumerate(segments):
            is_last_task = (task_id == n_tasks - 1)
            task_results: list[dict] = []

            # Sélection des échantillons pour cette tâche (tranche temporelle)
            start = offset
            end = min(offset + n_samples, len(X))
            indices = np.arange(start, end)
            if len(indices) == 0:
                indices = np.random.choice(len(X), size=n_samples, replace=False)
            offset = end

            for local_i, idx in enumerate(indices):
                is_last_sample = (local_i == len(indices) - 1)
                features, label = X[idx], int(y[idx])
                ts_ms = int(time.time() * 1000) - t0_ms

                flags = FRAME_FLAGS_PROFILING
                if request_update:
                    flags |= FRAME_FLAGS_UPDATE
                # Dernier sample de la tâche (sauf dernière tâche) → signal consolidation
                if consolidate and is_last_sample and not is_last_task:
                    flags |= FRAME_FLAGS_CONSOLIDATE

                frame = build_frame_v2(features, label, task_id, ts_ms, flags)

                if dry_run:
                    payload, recv_crc = frame[:-1], frame[-1]
                    ok = crc8(payload) == recv_crc
                    entry: dict = {
                        "task_id": task_id, "task_name": task_name,
                        "ts_ms": ts_ms, "true": label,
                        "pred": label, "confidence": 1.0,
                        "latency_us": 3, "ram_bytes": 200,
                        "throughput_ips": 333333,
                        "status": STATUS_OK if ok else STATUS_CRC_ERR,
                        "consolidate": bool(flags & FRAME_FLAGS_CONSOLIDATE),
                    }
                    if protocol_version >= 3:
                        entry["acc"]        = 0.9
                        entry["auroc"]      = 0.75
                        entry["forgetting"] = 0.01
                else:
                    assert ser is not None
                    t_send = time.monotonic()
                    ser.write(frame)
                    resp_fmt = RESPONSE_V3_FMT if protocol_version >= 3 else RESPONSE_V2_FMT
                    resp_size = RESPONSE_V3_SIZE if protocol_version >= 3 else RESPONSE_V2_SIZE
                    raw = ser.read(resp_size)
                    if len(raw) != resp_size:
                        if verbose:
                            print(f"[WARN] Timeout task={task_id} ({len(raw)}/{resp_size} B)")
                        elapsed = time.monotonic() - t_send
                        if interval_s > elapsed:
                            time.sleep(interval_s - elapsed)
                        continue
                    entry = {"task_id": task_id, "task_name": task_name,
                             "ts_ms": ts_ms, "true": label,
                             "consolidate": bool(flags & FRAME_FLAGS_CONSOLIDATE)}
                    entry.update(parse_response(raw))
                    elapsed = time.monotonic() - t_send
                    if interval_s > elapsed:
                        time.sleep(interval_s - elapsed)

                task_results.append(entry)
                all_results.append(entry)

                if verbose:
                    consolidate_marker = " [CONSOLIDATE→]" if entry["consolidate"] else ""
                    if "acc" in entry:
                        print(f"[task={task_id}({task_name}) ts={ts_ms}ms] "
                              f"true={label} pred={entry['pred']} "
                              f"acc={entry['acc']:.3f} fgt={entry['forgetting']:.3f}"
                              f"{consolidate_marker}")
                    else:
                        print(f"[task={task_id}({task_name}) ts={ts_ms}ms] "
                              f"true={label} pred={entry['pred']}"
                              f"{consolidate_marker}")

            # Métriques intermédiaires pour cette tâche
            task_stats = _compute_stats(task_results)
            task_stats["task_id"]   = task_id
            task_stats["task_name"] = task_name
            task_stats["mode"]      = "dry-run" if dry_run else "uart"
            per_task_metrics.append(task_stats)

            if verbose:
                print(f"\n--- Fin tâche {task_id} ({task_name}) : acc={task_stats['accuracy']:.4f} ---\n")

            # Sauvegarde intermédiaire
            if output_dir:
                out_path = Path(output_dir) / f"task_{task_id}_metrics.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(task_stats, indent=2))
                if verbose:
                    print(f"  → Sauvegardé : {out_path}")

    finally:
        if ser is not None:
            ser.close()

    return all_results, per_task_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming continu de données vers firmware STM32 (protocole v2)")
    parser.add_argument("--dataset", choices=["cwru", "monitoring", "pronostia"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-tasks", type=int, default=3, help="Nombre de tâches CL simulées")
    parser.add_argument("--rate-hz", type=float, default=0.0, help="Rate-limit (0=max speed)")
    parser.add_argument("--update", action="store_true", help="Demande mise à jour incrémentale au firmware")
    parser.add_argument("--output", type=str, help="Chemin JSON pour les statistiques")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--protocol-version", type=int, default=2, choices=[2, 3],
                        help="Version du protocole firmware (2=14B, 3=21B avec métriques CL)")
    parser.add_argument("--cl-sequence", type=str, default=None,
                        metavar="NAME:N[,NAME:N...]",
                        help="Séquence CL domain-incremental, ex: pump:167,turbine:167,compressor:166")
    parser.add_argument("--consolidate-on-task-change", action="store_true",
                        help="Envoie FLAGS=0x04 sur le dernier sample de chaque tâche (→ ewc_consolidate firmware)")
    args = parser.parse_args()

    print(f"Chargement dataset '{args.dataset}'...")
    X, y = _load_dataset(args.dataset)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    if args.cl_sequence:
        segments = parse_cl_sequence(args.cl_sequence)
        print(f"Mode CL séquence : {len(segments)} tâches — "
              + ", ".join(f"{n}({k})" for n, k in segments))
        all_results, per_task_metrics = _stream_cl_sequence(
            X, y,
            segments=segments,
            request_update=args.update,
            consolidate=args.consolidate_on_task_change,
            verbose=args.verbose,
            dry_run=args.dry_run,
            port=args.port,
            baud=args.baud,
            rate_hz=args.rate_hz,
            protocol_version=args.protocol_version,
            output_dir=args.output,
        )
        stats = _compute_stats(all_results)
        stats["mode"] = "dry-run" if args.dry_run else "uart"
        stats["cl_sequence"] = args.cl_sequence
        stats["per_task"] = per_task_metrics

        print("\n--- Résultats séquence CL ---")
        for k, v in stats.items():
            if k != "per_task":
                print(f"  {k}: {v}")
        for tm in per_task_metrics:
            print(f"  [tâche {tm['task_id']} {tm['task_name']}] acc={tm['accuracy']:.4f}")

        if args.output:
            out = Path(args.output) / "stream_summary.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats, indent=2))
            print(f"\nSauvegardé : {out}")

    elif args.dry_run:
        raw_results = _stream_dry_run(X, y, args.n_samples, args.n_tasks,
                                       args.update, args.verbose,
                                       protocol_version=args.protocol_version)
        stats = _compute_stats(raw_results)
        stats["mode"] = "dry-run"

        print("\n--- Résultats streaming ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats, indent=2))
            print(f"\nSauvegardé : {out}")
    else:
        raw_results = _stream_uart(args.port, args.baud, X, y, args.n_samples,
                                    args.n_tasks, args.rate_hz, args.update, args.verbose,
                                    protocol_version=args.protocol_version)
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
