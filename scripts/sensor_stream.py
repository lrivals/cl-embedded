"""
sensor_stream.py — Streaming continu PC → carte STM32 (protocole UART v2/v3).

Extension de sensor_sim.py : supporte multi-tâches, timestamps, rate-limiting,
et les protocoles v2 (14 B) et v3 (23 B).

Protocole v2 :
    Trame envoyée :
        [MAGIC 0xABCD:2B] [VERSION:1B=0x02] [TASK_ID:1B] [TIMESTAMP_MS:4B]
        [N:1B] [features:f32×N] [label:1B] [FLAGS:1B] [CRC8:1B]
    Réponse firmware (14 B) :
        [pred:u8] [conf:f32] [latency_us:u32] [ram_b:u16] [throughput:u16] [status:u8]

Protocole v3 :
    Trame envoyée : identique v2 (VERSION=0x03)
    Réponse firmware (23 B) :
        [pred:u8] [conf:f32] [latency_us:u32] [ram_b:u16] [acc:f32] [auroc:f32] [forgetting:f32]

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
FRAME_FLAGS_HDC_MODE    = 0x20   # utilise HDCClassifier (bit 5, cohérence pipeline.h PROTO_FLAG_HDC_MODE)
FRAME_FLAGS_INT8_MODE   = 0x40   # utilise EWCHeadInt8 (bit 6, cohérence pipeline.h PROTO_FLAG_INT8_MODE)
FRAME_FLAGS_TINYOL_MODE = 0x80   # utilise TinyOL autoencoder (bit 7, cohérence pipeline.h PROTO_FLAG_TINYOL_MODE)
# Sprint 29 — modes INT8 firmware (combinaisons de bits, cohérence pipeline.h)
FRAME_FLAGS_HDC_INT8    = 0x60   # HDC_MODE|INT8_MODE → HDCInt8 (S2901)
FRAME_FLAGS_TINYOL_INT8 = 0xC0   # TINYOL_MODE|INT8_MODE → TinyOLEncoderInt8 + OtOHeadInt8 (S2902)
# Sprint 27 — DUAL_MODE : EWC_REG (RUL) + EWC_MC (faute) simultané
FRAME_FLAGS_DUAL_MODE   = (FRAME_FLAGS_EWC_MODE | FRAME_FLAGS_HDC_MODE | FRAME_FLAGS_INT8_MODE)  # 0x70
# Sprint 30 — PAIR_MODE : Mahalanobis + supervisé (valeurs de nibble libres, cohérence pipeline.h)
FRAME_FLAGS_PAIR_MAHA_EWC    = 0x90   # Mahalanobis + EWC binaire
FRAME_FLAGS_PAIR_MAHA_HDC    = 0xA0   # Mahalanobis + HDC
FRAME_FLAGS_PAIR_MAHA_TINYOL = 0xB0   # Mahalanobis + TinyOL recon
# Sprint 31 — TRIPLE_MODE : PAIR + méta-modèle de stacking (verdict final)
FRAME_FLAGS_TRIPLE_MAHA_EWC  = 0xD0   # Mahalanobis + EWC + méta
FRAME_FLAGS_TRIPLE_MAHA_HDC  = 0xE0   # Mahalanobis + HDC + méta
# Sprint 34 — MAHA_Q15_MODE : Mahalanobis seul, sigma_inv int16 Q15 (réponse V3, S3407).
# 0xF0 est le SEUL nibble libre (0x10–0xE0 pris) ; cohérence pipeline.h PROTO_FLAG_MAHA_Q15.
FRAME_FLAGS_MAHA_Q15         = 0xF0

RESPONSE_V2_FMT  = "<BfIHHB"   # pred(u8), conf(f32), lat_us(u32), ram(u16), thr(u16), status(u8)
RESPONSE_V2_SIZE = struct.calcsize(RESPONSE_V2_FMT)  # 14 B

RESPONSE_V3_FMT  = "<BfIHfff"  # pred(u8), conf(f32), lat_us(u32), ram_b(u16), acc(f32), auroc(f32), forgetting(f32)
RESPONSE_V3_SIZE = struct.calcsize(RESPONSE_V3_FMT)  # 23 B

# Sprint 27 — Réponse DUAL_MODE 25 B
RESPONSE_DUAL_FMT  = "<BffIfff"  # pred_fault(u8), conf_fault(f32), rul_pred(f32), lat_us(u32), f1_macro(f32), rmse_rul(f32), forgetting(f32)
RESPONSE_DUAL_SIZE = struct.calcsize(RESPONSE_DUAL_FMT)  # 25 B

# Sprint 30 — Réponse PAIR_MODE 22 B (Mahalanobis + supervisé)
RESPONSE_PAIR_FMT  = "<BfBfIff"  # pred_maha(u8), score_maha(f32), pred_sup(u8), conf_sup(f32), lat_us(u32), auroc_maha(f32), f1_sup(f32)
RESPONSE_PAIR_SIZE = struct.calcsize(RESPONSE_PAIR_FMT)  # 22 B

# Sprint 31 — Réponse TRIPLE_MODE 27 B (PAIR 22 B + verdict méta)
RESPONSE_TRIPLE_FMT  = "<BfBfIffBf"  # PAIR + pred_meta(u8), prob_meta(f32) ; conf_sup porte p_sup
RESPONSE_TRIPLE_SIZE = struct.calcsize(RESPONSE_TRIPLE_FMT)  # 27 B

STATUS_OK         = 0x00
STATUS_CRC_ERR    = 0x01
STATUS_OOB        = 0x02
STATUS_UPDATE_DONE = 0x04


def parse_response(data: bytes) -> dict:
    """Parse une réponse firmware UART : triple 27 B (Sprint 31), pair 22 B (Sprint 30), dual 25 B (Sprint 27), v3 23 B, v2 14 B."""
    if len(data) == RESPONSE_TRIPLE_SIZE:
        pred_maha, score_maha, pred_sup, p_sup, lat_us, auroc_maha, f1_sup, pred_meta, prob_meta = \
            struct.unpack(RESPONSE_TRIPLE_FMT, data)
        return {
            "mode":        "triple",
            "pred_maha":   int(pred_maha),
            "score_maha":  float(score_maha),
            "pred_sup":    int(pred_sup),
            "p_sup":       float(p_sup),       # le firmware envoie p_sup dans le slot conf_sup
            "conf_sup":    float(p_sup),
            "latency_us":  int(lat_us),
            "auroc_maha":  float(auroc_maha),
            "f1_sup":      float(f1_sup),
            "pred_meta":   int(pred_meta),
            "prob_meta":   float(prob_meta),
            # alias pour compat _compute_stats : la métrique d'intérêt est le verdict méta
            "pred":        int(pred_meta),
            "confidence":  float(prob_meta),
            "ram_bytes":   0,
            "throughput_ips": 0,
            "status":      STATUS_OK,
        }
    if len(data) == RESPONSE_PAIR_SIZE:
        pred_maha, score_maha, pred_sup, conf_sup, lat_us, auroc_maha, f1_sup = \
            struct.unpack(RESPONSE_PAIR_FMT, data)
        return {
            "mode":        "pair",
            "pred_maha":   int(pred_maha),
            "score_maha":  float(score_maha),
            "pred_sup":    int(pred_sup),
            "conf_sup":    float(conf_sup),
            "latency_us":  int(lat_us),
            "auroc_maha":  float(auroc_maha),
            "f1_sup":      float(f1_sup),
            # alias pour compat _compute_stats (accuracy/confidence du modèle supervisé)
            "pred":        int(pred_sup),
            "confidence":  float(conf_sup),
            "ram_bytes":   0,
            "throughput_ips": 0,
            "status":      STATUS_OK,
        }
    if len(data) == RESPONSE_DUAL_SIZE:
        pred_fault, conf_fault, rul_pred, lat_us, f1_macro, rmse_rul, forgetting = \
            struct.unpack(RESPONSE_DUAL_FMT, data)
        return {
            "mode":           "dual",
            "pred":           int(pred_fault),
            "confidence":     float(conf_fault),
            "rul_pred":       float(rul_pred),
            "latency_us":     int(lat_us),
            "f1_macro":       float(f1_macro),
            "rmse_rul":       float(rmse_rul),
            "forgetting":     float(forgetting),
            "ram_bytes":      0,
            "throughput_ips": 0,
            "status":         STATUS_OK,
        }
    elif len(data) == RESPONSE_V3_SIZE:
        pred, conf, lat_us, ram_b, acc, auroc, forgetting = struct.unpack(RESPONSE_V3_FMT, data)
        return {
            "pred": pred, "confidence": float(conf), "latency_us": lat_us,
            "ram_bytes": ram_b, "throughput_ips": 0, "status": STATUS_OK,
            "acc": float(acc), "auroc": float(auroc), "forgetting": float(forgetting),
        }
    elif len(data) == RESPONSE_V2_SIZE:
        pred, conf, lat_us, ram_b, thr, status = struct.unpack(RESPONSE_V2_FMT, data)
        return {
            "pred": pred, "confidence": float(conf), "latency_us": lat_us,
            "ram_bytes": ram_b, "throughput_ips": thr, "status": status,
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


def _load_paderborn() -> tuple[np.ndarray, np.ndarray]:
    """Charge Paderborn (3 tâches : K001→KA04→KI04) avec top-5 features FFT."""
    import yaml as _yaml
    from src.data.paderborn_loader import get_cl_dataloaders

    subset = _yaml.safe_load(Path("configs/paderborn_feature_subset.yaml").read_text())
    feature_names = subset["selected_features"]

    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/"),
        config_path=Path("configs/board_paderborn.yaml"),
        feature_names=feature_names,
    )
    Xs, ys = [], []
    for t in tasks:  # 3 tâches : healthy → OR → IR
        for xb, yb in t["train_loader"]:
            Xs.append(xb.numpy())
            ys.append(yb.numpy().flatten())
    return np.concatenate(Xs), np.concatenate(ys).astype(int)


def _load_cmapss() -> tuple[np.ndarray, np.ndarray]:
    """Charge CMAPSS FD001+FD002 avec les 5 features sélectionnées."""
    import yaml as _yaml
    from src.data.cmapss_loader import get_cl_dataloaders

    subset = _yaml.safe_load(Path("configs/cmapss_feature_subset.yaml").read_text())
    feature_names = subset.get("selected_features") or subset.get("features")

    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/CMAPSS Jet Engine Simulated Data/"),
        config_path=Path("configs/board_cmapss.yaml"),
        feature_names=feature_names,
    )
    Xs, ys = [], []
    for t in tasks[:2]:   # FD001 + FD002 (n_tasks_board=2)
        for xb, yb in t["train_loader"]:
            Xs.append(xb.numpy())
            ys.append(yb.numpy().flatten())
    return np.concatenate(Xs), np.concatenate(ys).astype(int)


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Charge un dataset (CMAPSS/Paderborn natifs, autres via sensor_sim.py)."""
    if name == "cmapss":
        return _load_cmapss()
    if name == "paderborn":
        return _load_paderborn()
    import importlib.util
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
                if model_flags in (FRAME_FLAGS_TRIPLE_MAHA_EWC,
                                   FRAME_FLAGS_TRIPLE_MAHA_HDC):
                    resp_size = RESPONSE_TRIPLE_SIZE  # Sprint 31 — réponse triple 27 B
                elif model_flags in (FRAME_FLAGS_PAIR_MAHA_EWC,
                                   FRAME_FLAGS_PAIR_MAHA_HDC,
                                   FRAME_FLAGS_PAIR_MAHA_TINYOL):
                    resp_size = RESPONSE_PAIR_SIZE   # Sprint 30 — réponse paire 22 B
                elif model_flags == FRAME_FLAGS_DUAL_MODE:
                    resp_size = RESPONSE_DUAL_SIZE   # Sprint 27 — réponse duale 25 B
                elif model_flags == FRAME_FLAGS_MAHA_Q15:
                    resp_size = RESPONSE_V3_SIZE     # Sprint 34 — Q15 réutilise la réponse V3 (23 B)
                elif protocol_version >= 3:
                    resp_size = RESPONSE_V3_SIZE
                else:
                    resp_size = RESPONSE_V2_SIZE

                raw = ser.read(resp_size)
                if len(raw) != resp_size:
                    if verbose:
                        print(f"[WARN] Timeout task={task_id} ({len(raw)}/{resp_size} B)")
                    continue

                entry: dict = {"task_id": task_id, "ts_ms": ts_ms, "true": label}
                entry.update(parse_response(raw))
                entry["features"] = [float(v) for v in features]  # parité board↔PC (S3205)
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

    # F1 détection de panne (classe faulty=1) — même définition que le PC (S3504).
    # Dérivé côté hôte depuis les prédictions/labels du flux board : aucun changement
    # du protocole UART (F1 n'est pas calculé par le firmware).
    from src.evaluation.metrics import compute_fault_f1

    f1 = compute_fault_f1(np.asarray(trues), np.asarray(preds))

    return {
        "n_samples": len(results),
        "n_tasks": n_tasks,
        "accuracy": round(acc, 4),
        "f1_faulty": round(f1["f1_faulty"], 4),
        "f1_macro": round(f1["f1_macro"], 4),
        "precision_faulty": round(f1["precision_faulty"], 4),
        "recall_faulty": round(f1["recall_faulty"], 4),
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
    model_flags: int = 0,
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

                flags = FRAME_FLAGS_PROFILING | model_flags
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


def _resp_size_for(model_flags: int, protocol_version: int) -> int:
    """Taille de réponse UART attendue selon le mode/protocole (cf. _stream_uart)."""
    if model_flags in (FRAME_FLAGS_TRIPLE_MAHA_EWC, FRAME_FLAGS_TRIPLE_MAHA_HDC):
        return RESPONSE_TRIPLE_SIZE
    if model_flags in (FRAME_FLAGS_PAIR_MAHA_EWC, FRAME_FLAGS_PAIR_MAHA_HDC,
                       FRAME_FLAGS_PAIR_MAHA_TINYOL):
        return RESPONSE_PAIR_SIZE
    if model_flags == FRAME_FLAGS_DUAL_MODE:
        return RESPONSE_DUAL_SIZE
    if model_flags == FRAME_FLAGS_MAHA_Q15:
        return RESPONSE_V3_SIZE   # Sprint 34 — Q15 réutilise la réponse V3
    if protocol_version >= 3:
        return RESPONSE_V3_SIZE
    return RESPONSE_V2_SIZE


def _measure_bss_bytes(elf_path: str | None) -> int | str:
    """Lit la taille .bss d'un ELF via arm-none-eabi-size. 'à mesurer' si indisponible."""
    if not elf_path or not Path(elf_path).exists():
        return "à mesurer"
    import subprocess
    try:
        out = subprocess.run(["arm-none-eabi-size", elf_path],
                             capture_output=True, text=True, check=True).stdout
        # En-tête : text data bss dec hex filename ; on lit la 3e colonne de la 2e ligne.
        cols = out.strip().splitlines()[1].split()
        return int(cols[2])
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return "à mesurer"


def _stream_sweep(
    port: str, baud: int,
    X: np.ndarray, y: np.ndarray,
    n_samples: int,
    profile_path: str,
    model: str | None,
    model_flags: int,
    protocol_version: int,
    window: int,
    elf_path: str | None,
    output_dir: str,
    verbose: bool,
) -> dict:
    """Balayage board (rate_hz × stride) → point de saturation débit/buffer (S3403).

    Pour chaque config, envoie `n_samples` trames à `rate_hz`, compte drops/timeouts/CRC,
    mesure la latence DWT (réponse UART), compare au modèle analytique S3401 et écrit un
    JSON par config + un summary.json agrégé. Aucun chiffre inventé : tout vient de la board.
    """
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial requis : pip install pyserial")

    from src.evaluation.streaming_model import (
        debit_max,
        debit_streaming,
        marge_temps_reel,
    )

    prof = _load_streaming_profile(profile_path)
    sweep = prof.get("sweep", {})
    rates = sweep.get("rates_hz", [prof.get("f_acq_hz", 100)])
    strides = sweep.get("strides", [prof.get("stride_s", 1)])
    latences = prof.get("latences_inf_us", {})
    lat_inf_us = latences.get(model) if model else None

    resp_size = _resp_size_for(model_flags, protocol_version)
    bss_bytes = _measure_bss_bytes(elf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_out: list[dict] = []
    saturation_first: dict | None = None

    with serial.Serial(port, baud, timeout=UART_TIMEOUT_S,
                       dsrdtr=False, rtscts=False) as ser:
        ser.dtr = True
        time.sleep(0.05)
        ser.dtr = False
        time.sleep(0.5)

        t0_ms = int(time.time() * 1000)
        for rate_hz in rates:
            for stride in strides:
                interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
                ser.reset_input_buffer()
                latencies: list[int] = []
                drops = 0
                crc_errors = 0
                sent = 0
                idx_pool = np.random.choice(len(X), size=min(n_samples, len(X)),
                                            replace=False)
                for k, idx in enumerate(idx_pool):
                    t_send = time.monotonic()
                    features = X[idx]
                    label = int(y[idx])
                    flags = FRAME_FLAGS_PROFILING | model_flags
                    ts_ms = int(time.time() * 1000) - t0_ms
                    frame = build_frame_v2(features, label, k % 256, ts_ms, flags)
                    ser.write(frame)
                    sent += 1
                    raw = ser.read(resp_size)
                    if len(raw) != resp_size:
                        drops += 1
                    else:
                        entry = parse_response(raw)
                        latencies.append(int(entry["latency_us"]))
                        if entry.get("status", 0) & STATUS_CRC_ERR:
                            crc_errors += 1
                    elapsed = time.monotonic() - t_send
                    if interval_s > elapsed:
                        time.sleep(interval_s - elapsed)

                # Modèle analytique S3401
                ds = debit_streaming(float(rate_hz), int(stride), int(window))
                dm = debit_max(lat_inf_us * 1e-6) if lat_inf_us else None
                marge = marge_temps_reel(ds, dm) if dm else None
                predicted_sat = (marge is not None) and (not marge["ok"])
                observed_sat = (drops > 0) or (crc_errors > 0)
                saturation_atteinte = bool(predicted_sat and observed_sat)

                cfg = {
                    "model": model,
                    "rate_hz": rate_hz,
                    "stride": stride,
                    "window": window,
                    "n_sent": sent,
                    "n_received": len(latencies),
                    "latence_dwt_us": (round(float(np.median(latencies)), 1)
                                       if latencies else "à mesurer"),
                    "latence_dwt_p99_us": (round(float(np.percentile(latencies, 99)), 1)
                                           if latencies else "à mesurer"),
                    "drops": drops,
                    "timeouts": drops,   # drop = réponse incomplète/absente dans le timeout
                    "crc_errors": crc_errors,
                    "bss_bytes": bss_bytes,
                    "debit_streaming_hz": round(ds, 3),
                    "debit_max_hz": (round(dm, 1) if dm else "à mesurer"),
                    "marge_pct": (round(marge["marge_pct"], 4) if marge else "à mesurer"),
                    "saturation_predite": predicted_sat,
                    "saturation_observee": observed_sat,
                    "saturation_atteinte": saturation_atteinte,
                }
                configs_out.append(cfg)
                fname = f"{model or 'model'}_rate{int(rate_hz)}_stride{int(stride)}_w{int(window)}.json"
                (out_dir / fname).write_text(json.dumps(cfg, indent=2))
                if saturation_atteinte and saturation_first is None:
                    saturation_first = cfg
                if verbose:
                    print(f"  rate={rate_hz}Hz stride={stride} W={window}: "
                          f"recv={len(latencies)}/{sent} drops={drops} crc={crc_errors} "
                          f"lat={cfg['latence_dwt_us']}µs sat={saturation_atteinte}")

    summary = {
        "model": model,
        "window": window,
        "bss_bytes": bss_bytes,
        "latence_inf_us": lat_inf_us if lat_inf_us else "à mesurer",
        "n_configs": len(configs_out),
        "saturation_first": saturation_first,
        "configs": configs_out,
    }
    (out_dir / f"summary_{model or 'model'}_w{int(window)}.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def _load_streaming_profile(path: str) -> dict:
    """Charge configs/streaming_profile.yaml et renvoie la section 'streaming'."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("streaming", cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming continu de données vers firmware STM32 (protocole v2)")
    parser.add_argument("--dataset", choices=["cwru", "monitoring", "pronostia", "cmapss", "paderborn", "battery"], required=True)
    parser.add_argument("--model",
                        choices=["ewc", "ewc-int8", "tinyol", "mahalanobis", "hdc", "dual",
                                 "hdc-int8", "tinyol-int8",
                                 "pair-maha-ewc", "pair-maha-hdc", "pair-maha-tinyol",
                                 "triple-maha-ewc", "triple-maha-hdc", "maha-q15"],
                        default=None, help="Modèle MCU cible (définit les FLAGS protocole)")
    parser.add_argument("--condition", choices=["5feat", "all", "best"], default=None,
                        help="Condition de features Sprint 35 (S3508) : sélectionne côté hôte "
                             "les colonnes natives envoyées à la board (5feat/all/best-par-modèle), "
                             "via resolve_feature_indices. Sans valeur : pipeline 5-feat historique.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-tasks", type=int, default=3, help="Nombre de tâches CL simulées")
    parser.add_argument("--rate-hz", type=float, default=0.0, help="Rate-limit (0=max speed)")
    parser.add_argument("--update", action="store_true", help="Demande mise à jour incrémentale au firmware")
    parser.add_argument("--output", type=str, help="Chemin JSON pour les statistiques")
    parser.add_argument("--dump-samples", action="store_true",
                        help="Inclut la liste par-échantillon (pred/true/confidence) dans le JSON "
                             "de sortie — requis pour la parité board↔PC (S3205)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--protocol-version", type=int, default=2, choices=[2, 3],
                        help="Version du protocole firmware (2=14B, 3=21B avec métriques CL)")
    parser.add_argument("--cl-sequence", type=str, default=None,
                        metavar="NAME:N[,NAME:N...]",
                        help="Séquence CL domain-incremental, ex: pump:167,turbine:167,compressor:166")
    parser.add_argument("--consolidate-on-task-change", action="store_true",
                        help="Envoie FLAGS=0x04 sur le dernier sample de chaque tâche (→ ewc_consolidate firmware)")
    parser.add_argument("--sweep", type=str, default=None, metavar="PROFILE.yaml",
                        help="Mode balayage débit/stride S3403 : lit configs/streaming_profile.yaml "
                             "(rate_hz × stride), mesure latence DWT/drops/CRC et point de saturation")
    parser.add_argument("--window", type=int, default=5,
                        help="Taille de fenêtre W de la build firmware (= STREAM_BUF_W) pour le sweep S3403")
    parser.add_argument("--elf", type=str, default="firmware/stm32f4_blink/build/stm32f4_blink.elf",
                        help="Chemin de l'ELF flashé (pour mesurer .bss par config W via arm-none-eabi-size)")
    args = parser.parse_args()

    model_flags = 0
    if args.model == "ewc":
        model_flags = FRAME_FLAGS_EWC_MODE
    elif args.model == "ewc-int8":
        model_flags = FRAME_FLAGS_INT8_MODE
    elif args.model == "hdc":
        model_flags = FRAME_FLAGS_HDC_MODE
    elif args.model == "dual":
        model_flags = FRAME_FLAGS_DUAL_MODE
    elif args.model == "hdc-int8":
        model_flags = FRAME_FLAGS_HDC_INT8
    elif args.model == "tinyol-int8":
        model_flags = FRAME_FLAGS_TINYOL_INT8
    elif args.model == "pair-maha-ewc":
        model_flags = FRAME_FLAGS_PAIR_MAHA_EWC
    elif args.model == "pair-maha-hdc":
        model_flags = FRAME_FLAGS_PAIR_MAHA_HDC
    elif args.model == "pair-maha-tinyol":
        model_flags = FRAME_FLAGS_PAIR_MAHA_TINYOL
    elif args.model == "triple-maha-ewc":
        model_flags = FRAME_FLAGS_TRIPLE_MAHA_EWC
    elif args.model == "triple-maha-hdc":
        model_flags = FRAME_FLAGS_TRIPLE_MAHA_HDC
    elif args.model == "maha-q15":
        model_flags = FRAME_FLAGS_MAHA_Q15
    # tinyol et mahalanobis n'ont pas de flag dédié (pipeline sélectionne via config firmware)

    print(f"Chargement dataset '{args.dataset}'...")
    if args.condition:
        # Sélection de features par condition (S3508) : mêmes colonnes natives que
        # le modèle de référence board entraîné → parité board↔PC par construction.
        from src.evaluation.feature_conditions import load_condition_arrays

        base_model = (args.model or "ewc").split("-")[0]  # ewc/mahalanobis/hdc/tinyol
        if base_model not in ("ewc", "mahalanobis", "hdc", "tinyol"):
            base_model = "ewc"  # défaut sûr pour all/5feat (indices indépendants du modèle)
        X, y, idx, names = load_condition_arrays(args.dataset, args.condition, base_model)
        print(f"  condition={args.condition} model={base_model} → {len(idx)} feats {names}")
    else:
        X, y = _load_dataset(args.dataset)
    print(f"  {len(X)} samples, {X.shape[1]} features")

    if args.sweep:
        if args.dry_run:
            raise SystemExit("--sweep requiert une board réelle (pas compatible --dry-run)")
        if not args.output:
            raise SystemExit("--sweep requiert --output <dir>")
        print(f"Mode balayage débit/stride (S3403) — profil {args.sweep}, W={args.window}")
        summary = _stream_sweep(
            args.port, args.baud, X, y, args.n_samples,
            profile_path=args.sweep,
            model=args.model,
            model_flags=model_flags,
            protocol_version=args.protocol_version,
            window=args.window,
            elf_path=args.elf,
            output_dir=args.output,
            verbose=args.verbose,
        )
        print("\n--- Balayage terminé ---")
        print(f"  configs testées : {summary['n_configs']}")
        print(f"  .bss : {summary['bss_bytes']} B")
        if summary["saturation_first"]:
            sf = summary["saturation_first"]
            print(f"  saturation : rate={sf['rate_hz']}Hz stride={sf['stride']} "
                  f"(drops={sf['drops']}, crc={sf['crc_errors']})")
        else:
            print("  saturation : non atteinte sur la plage testée")
        print(f"  Sauvegardé dans : {args.output}")
        return

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
            model_flags=model_flags,
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
                                       protocol_version=args.protocol_version,
                                       model_flags=model_flags)
        stats = _compute_stats(raw_results)
        stats["mode"] = "dry-run"
        if args.dump_samples:
            stats["samples"] = [
                {"pred": int(r["pred"]), "true": int(r["true"]),
                 "confidence": float(r.get("confidence", 0.0)),
                 "features": r.get("features")}
                for r in raw_results
            ]

        print("\n--- Résultats streaming ---")
        for k, v in stats.items():
            if k != "samples":
                print(f"  {k}: {v}")

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats, indent=2))
            print(f"\nSauvegardé : {out}")
    else:
        raw_results = _stream_uart(args.port, args.baud, X, y, args.n_samples,
                                    args.n_tasks, args.rate_hz, args.update, args.verbose,
                                    protocol_version=args.protocol_version,
                                    model_flags=model_flags)
        stats = _compute_stats(raw_results)
        stats["mode"] = "uart"
        stats["port"] = args.port
        if args.dump_samples:
            stats["samples"] = [
                {"pred": int(r["pred"]), "true": int(r["true"]),
                 "confidence": float(r.get("confidence", 0.0)),
                 "features": r.get("features")}
                for r in raw_results
            ]

        print("\n--- Résultats streaming ---")
        for k, v in stats.items():
            if k != "samples":
                print(f"  {k}: {v}")

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats, indent=2))
            print(f"\nSauvegardé : {out}")


if __name__ == "__main__":
    main()
