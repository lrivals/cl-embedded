"""
compare_mahalanobis_pc_vs_board.py — Validation end-to-end PC vs firmware C.

Compare les scores/logits calculés par la référence Python et ceux retournés
par le firmware C embarqué sur la même séquence de samples.

Modèles supportés :
  --model mahalanobis  : distance Mahalanobis FP64 vs FP32, tolerance 1e-4
  --model ewc          : sigmoid EWC float64 vs float32, tolerance 1e-3

Usage :
    # Dry-run Mahalanobis → experiments/exp_S19_01/comparison_results.json
    python scripts/compare_mahalanobis_pc_vs_board.py \\
        --dry-run --model mahalanobis --output experiments/exp_S19_01

    # Dry-run EWC → experiments/exp_S19_02/comparison_results.json
    python scripts/compare_mahalanobis_pc_vs_board.py \\
        --dry-run --model ewc --output experiments/exp_S19_02

    # Board connectée
    python scripts/compare_mahalanobis_pc_vs_board.py \\
        --port /dev/ttyACM0 --model mahalanobis --n-samples 500 \\
        --output experiments/exp_S19_01

Référence : S2007_pc_vs_board.md, S1901_mahalanobis_validation.md
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import tracemalloc
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Constantes protocole UART v2/v3 ──────────────────────────────────────
MAGIC              = 0xABCD
PROTO_VERSION      = 0x02
FRAME_FLAGS_UPDATE = 0x01

RESPONSE_V2_FMT  = "<BfIHHB"   # pred(u8), conf(f32), lat_us(u32), ram(u16), thr(u16), status(u8)
RESPONSE_V2_SIZE = struct.calcsize(RESPONSE_V2_FMT)   # 14 B

RESPONSE_V3_FMT  = "<BfIfff"   # pred(u8), conf(f32), lat_us(u32), acc(f32), auroc(f32), forgetting(f32)
RESPONSE_V3_SIZE = struct.calcsize(RESPONSE_V3_FMT)   # 21 B

# ── Paramètres Mahalanobis (identiques à model_weights.h placeholder) ────
MAHA_DIM       = 5
MAHA_MEAN      = np.zeros(MAHA_DIM, dtype=np.float32)
MAHA_PRECISION = np.eye(MAHA_DIM, dtype=np.float32)   # Σ⁻¹ = I (placeholder)
MAHA_THRESHOLD = 1.0
MAHA_EMA_ALPHA = 0.1
ZSCORE_MEAN    = np.zeros(MAHA_DIM, dtype=np.float32)
ZSCORE_STD     = np.ones(MAHA_DIM, dtype=np.float32)

TOLERANCE_MAHA = 1e-4   # FP32 vs FP64 sur distance Mahalanobis
TOLERANCE_EWC  = 1e-3   # FP32 vs FP64 sur sigmoid EWC (backprop introduce davantage d'erreur)

# ── EWC : dimensions réseau (identiques à firmware/inc/ewc_head.h) ────────
EWC_INPUT_DIM   = 6
EWC_HIDDEN_DIMS = [32, 16]


# ── CRC8 polynomial 0x07 ─────────────────────────────────────────────────

def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07 if crc & 0x80 else crc << 1) & 0xFF
    return crc


def build_frame_v2(features: np.ndarray, label: int,
                   task_id: int, ts_ms: int, flags: int = 0) -> bytes:
    n = len(features)
    header = struct.pack("<HBBIB", MAGIC, PROTO_VERSION,
                         task_id & 0xFF, ts_ms & 0xFFFFFFFF, n)
    feat_bytes = features.astype(np.float32).tobytes()
    tail       = struct.pack("<BB", label & 0xFF, flags & 0xFF)
    payload    = header + feat_bytes + tail
    return payload + struct.pack("<B", crc8(payload))


# ── Données synthétiques ─────────────────────────────────────────────────

def generate_synthetic_cwru(n_samples: int, n_tasks: int,
                             anomaly_ratio: float = 0.10,
                             seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mahalanobis : n_samples CWRU-like à MAHA_DIM=5 features."""
    rng = np.random.default_rng(seed)
    per_task = n_samples // n_tasks
    features_list, labels_list, tasks_list = [], [], []

    for task_id in range(n_tasks):
        n = per_task if task_id < n_tasks - 1 else n_samples - per_task * (n_tasks - 1)
        n_anom = max(1, int(n * anomaly_ratio))
        n_norm = n - n_anom

        normal  = rng.normal(0.5 * task_id, 0.1, (n_norm, MAHA_DIM)).astype(np.float32)
        anomaly = rng.normal(3.0, 0.5, (n_anom, MAHA_DIM)).astype(np.float32)

        feats = np.vstack([normal, anomaly])
        lbls  = np.concatenate([np.zeros(n_norm, dtype=np.int32),
                                 np.ones(n_anom, dtype=np.int32)])
        tasks = np.full(n, task_id, dtype=np.int32)

        idx = rng.permutation(n)
        features_list.append(feats[idx])
        labels_list.append(lbls[idx])
        tasks_list.append(tasks)

    return (np.vstack(features_list),
            np.concatenate(labels_list),
            np.concatenate(tasks_list))


def generate_synthetic_monitoring(n_samples: int, n_tasks: int,
                                  faulty_ratio: float = 0.20,
                                  seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EWC : n_samples monitoring-like à EWC_INPUT_DIM=6 features (temp, pression, vib, hum, RPM, charge)."""
    rng = np.random.default_rng(seed)
    per_task = n_samples // n_tasks
    features_list, labels_list, tasks_list = [], [], []

    for task_id in range(n_tasks):
        n = per_task if task_id < n_tasks - 1 else n_samples - per_task * (n_tasks - 1)
        n_faulty = max(1, int(n * faulty_ratio))
        n_normal = n - n_faulty

        # Domain shift : offset par tâche (pump → turbine → compressor)
        offset = np.array([0.2 * task_id] * EWC_INPUT_DIM, dtype=np.float64)
        normal  = rng.normal(offset, 0.15, (n_normal, EWC_INPUT_DIM))
        faulty  = rng.normal(offset + 2.0, 0.5, (n_faulty, EWC_INPUT_DIM))

        feats = np.vstack([normal, faulty]).astype(np.float32)
        lbls  = np.concatenate([np.zeros(n_normal, dtype=np.int32),
                                 np.ones(n_faulty, dtype=np.int32)])
        tasks = np.full(n, task_id, dtype=np.int32)

        idx = rng.permutation(n)
        features_list.append(feats[idx])
        labels_list.append(lbls[idx])
        tasks_list.append(tasks)

    return (np.vstack(features_list),
            np.concatenate(labels_list),
            np.concatenate(tasks_list))


# ── Référence Python Mahalanobis ─────────────────────────────────────────

def maha_score_python(x: np.ndarray, mean: np.ndarray,
                      precision: np.ndarray) -> float:
    """Même algo que maha_score() en C (FP32 ou FP64 selon dtype de x)."""
    diff  = x.astype(precision.dtype) - mean
    left  = precision @ diff
    dist2 = float(np.dot(left, diff))
    return float(np.sqrt(max(dist2, 0.0)))


# ── Métriques CL ─────────────────────────────────────────────────────────

class TaskMetrics(NamedTuple):
    task_id: int
    n_correct: int
    n_total: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.0


def compute_cl_metrics(per_task_acc: list[float]) -> dict:
    acc_final = float(np.mean(per_task_acc))
    return {
        "acc_final":         round(acc_final, 4),
        "avg_forgetting":    0.0,   # online → N/A
        "backward_transfer": 0.0,
        "per_task_acc":      {i: round(a, 4) for i, a in enumerate(per_task_acc)},
    }


def _per_task_accuracy(preds: np.ndarray, labels: np.ndarray,
                       task_ids: np.ndarray) -> list[float]:
    per_task_acc = []
    for tid in np.unique(task_ids):
        mask      = task_ids == tid
        n_correct = int(np.sum(preds[mask] == labels[mask]))
        per_task_acc.append(n_correct / int(np.sum(mask)))
    return per_task_acc


def format_results(model_name: str,
                   scores_py: list[float], scores_c: list[float],
                   preds: list[int], labels: np.ndarray, task_ids: np.ndarray,
                   latencies: list[float], tolerance: float, platform: str,
                   n_params: int, static_ram_bytes: int = 0) -> dict:
    """Produit le format de sortie S2007."""
    sp = np.array(scores_py)
    sc = np.array(scores_c)
    deltas = np.abs(sp - sc)

    per_task_acc = _per_task_accuracy(
        np.array(preds, dtype=np.int32), labels[:len(preds)], task_ids[:len(preds)]
    )
    cl = compute_cl_metrics(per_task_acc)

    return {
        "model":               model_name,
        "n_samples":           len(scores_py),
        "max_abs_delta":       float(deltas.max()),
        "mean_abs_delta":      float(deltas.mean()),
        "tolerance":           tolerance,
        "compliant":           bool(deltas.max() <= tolerance),
        "platform":            platform,
        **cl,
        "ram_peak_bytes":      static_ram_bytes,
        "inference_latency_ms": round(float(np.mean(latencies)), 4),
        "n_params":            n_params,
    }


# ── Mahalanobis — dry-run ────────────────────────────────────────────────

def run_mahalanobis_dry_run(features: np.ndarray, labels: np.ndarray,
                             task_ids: np.ndarray) -> dict:
    """Python FP64 (référence) vs Python FP32 (simulation C)."""
    mean_f64  = MAHA_MEAN.astype(np.float64)
    prec_f64  = MAHA_PRECISION.astype(np.float64)
    mean_f32  = MAHA_MEAN.copy()
    prec_f32  = MAHA_PRECISION.copy()

    scores_py: list[float] = []
    scores_c:  list[float] = []
    preds:     list[int]   = []
    latencies: list[float] = []

    for x in features:
        x_norm_f64 = ((x - ZSCORE_MEAN) / ZSCORE_STD).astype(np.float64)
        x_norm_f32 = ((x - ZSCORE_MEAN) / ZSCORE_STD).astype(np.float32)

        t0   = time.perf_counter()
        s_py = maha_score_python(x_norm_f64, mean_f64, prec_f64)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        s_c = maha_score_python(x_norm_f32, mean_f32, prec_f32)

        scores_py.append(s_py)
        scores_c.append(s_c)
        preds.append(1 if s_py > MAHA_THRESHOLD else 0)

        if s_py <= MAHA_THRESHOLD:
            mean_f64 = (1.0 - MAHA_EMA_ALPHA) * mean_f64 + MAHA_EMA_ALPHA * x_norm_f64
            mean_f32 = ((1.0 - MAHA_EMA_ALPHA) * mean_f32
                        + MAHA_EMA_ALPHA * x_norm_f32.astype(np.float32)).astype(np.float32)

    n_params = MAHA_DIM * (MAHA_DIM + 1) + 2   # mean + precision + threshold + alpha
    return format_results("mahalanobis", scores_py, scores_c, preds,
                          labels, task_ids, latencies,
                          tolerance=TOLERANCE_MAHA, platform="dry_run",
                          n_params=n_params, static_ram_bytes=220)


# ── Mahalanobis — board ──────────────────────────────────────────────────

def run_mahalanobis_board(port: str, features: np.ndarray, labels: np.ndarray,
                           task_ids: np.ndarray) -> dict:
    try:
        import serial
    except ImportError:
        raise RuntimeError("pyserial requis : pip install pyserial")

    ser    = serial.Serial(port, baudrate=115200, timeout=2.0)
    time.sleep(0.1)

    mean_f64 = MAHA_MEAN.astype(np.float64)
    prec_f64 = MAHA_PRECISION.astype(np.float64)

    scores_py: list[float] = []
    scores_c:  list[float] = []
    preds:     list[int]   = []
    latencies: list[float] = []
    ts_ms = 0

    for idx, (x, label, task_id) in enumerate(zip(features, labels, task_ids)):
        x_norm_f64 = ((x - ZSCORE_MEAN) / ZSCORE_STD).astype(np.float64)
        x_norm_f32 = x_norm_f64.astype(np.float32)

        frame = build_frame_v2(x_norm_f32, int(label), int(task_id), ts_ms,
                               FRAME_FLAGS_UPDATE)
        ser.write(frame)
        ts_ms += 100

        resp = ser.read(RESPONSE_V2_SIZE)
        if len(resp) != RESPONSE_V2_SIZE:
            print(f"[WARN] sample {idx}: réponse tronquée ({len(resp)} B)")
            continue

        _pred_c, conf_c, lat_us, _ram, _thr, _status = struct.unpack(RESPONSE_V2_FMT, resp)
        # conf = 1/(1+score) → score = (1/conf) - 1
        score_c = (1.0 / conf_c - 1.0) if conf_c > 0 else float("inf")

        s_py = maha_score_python(x_norm_f64, mean_f64, prec_f64)
        if s_py <= MAHA_THRESHOLD:
            mean_f64 = (1.0 - MAHA_EMA_ALPHA) * mean_f64 + MAHA_EMA_ALPHA * x_norm_f64

        scores_py.append(s_py)
        scores_c.append(score_c)
        preds.append(1 if s_py > MAHA_THRESHOLD else 0)
        latencies.append(lat_us / 1000.0)

    ser.close()
    n_params = MAHA_DIM * (MAHA_DIM + 1) + 2
    return format_results("mahalanobis", scores_py, scores_c, preds,
                          labels, task_ids, latencies,
                          tolerance=TOLERANCE_MAHA, platform=f"board:{port}",
                          n_params=n_params, static_ram_bytes=220)


# ── EWC — dry-run ────────────────────────────────────────────────────────

def run_ewc_dry_run(features: np.ndarray, labels: np.ndarray,
                    task_ids: np.ndarray) -> dict:
    """EWC sigmoid float64 (Python ref) vs float32 (simulation C)."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch requis : pip install torch")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.ewc.ewc_mlp import EWCMlpClassifier  # type: ignore

    model_f32 = EWCMlpClassifier(input_dim=EWC_INPUT_DIM, hidden_dims=EWC_HIDDEN_DIMS)
    model_f32.eval()

    # Cast vers float64 pour la référence Python
    model_f64 = EWCMlpClassifier(input_dim=EWC_INPUT_DIM, hidden_dims=EWC_HIDDEN_DIMS)
    model_f64.load_state_dict(model_f32.state_dict())
    model_f64.double().eval()

    scores_py: list[float] = []
    scores_c:  list[float] = []
    preds:     list[int]   = []
    latencies: list[float] = []

    with torch.no_grad():
        for x in features:
            x_f64 = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
            x_f32 = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            t0    = time.perf_counter()
            s_py  = float(model_f64(x_f64).item())
            latencies.append((time.perf_counter() - t0) * 1000.0)

            s_c = float(model_f32(x_f32).item())

            scores_py.append(s_py)
            scores_c.append(s_c)
            preds.append(1 if s_py > 0.5 else 0)

    n_params = sum(p.numel() for p in model_f32.parameters())
    # MEM: EWC tête (weights + Fisher + snapshot) ~ 9.5 KiB @ FP32
    static_ram = n_params * 3 * 4   # weights + Fisher diag + theta_star
    return format_results("ewc", scores_py, scores_c, preds,
                          labels, task_ids, latencies,
                          tolerance=TOLERANCE_EWC, platform="dry_run",
                          n_params=n_params, static_ram_bytes=static_ram)


# ── EWC — board ──────────────────────────────────────────────────────────

def run_ewc_board(port: str, features: np.ndarray, labels: np.ndarray,
                  task_ids: np.ndarray) -> dict:
    """Compare EWC Python float64 vs conf field du firmware (protocole v3)."""
    try:
        import serial
        import torch
    except ImportError:
        raise RuntimeError("pyserial + torch requis")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.ewc.ewc_mlp import EWCMlpClassifier  # type: ignore

    model_f64 = EWCMlpClassifier(input_dim=EWC_INPUT_DIM, hidden_dims=EWC_HIDDEN_DIMS)
    model_f64.double().eval()

    ser = serial.Serial(port, baudrate=115200, timeout=2.0)
    time.sleep(0.1)

    scores_py: list[float] = []
    scores_c:  list[float] = []
    preds:     list[int]   = []
    latencies: list[float] = []
    ts_ms = 0

    with torch.no_grad():
        for idx, (x, label, task_id) in enumerate(zip(features, labels, task_ids)):
            x_f64 = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
            x_f32 = x.astype(np.float32)

            frame = build_frame_v2(x_f32, int(label), int(task_id), ts_ms,
                                   FRAME_FLAGS_UPDATE)
            ser.write(frame)
            ts_ms += 100

            resp = ser.read(RESPONSE_V3_SIZE)
            if len(resp) != RESPONSE_V3_SIZE:
                print(f"[WARN] sample {idx}: réponse v3 tronquée ({len(resp)} B)")
                continue

            _pred_c, conf_c, lat_us, _acc, _auroc, _forg = struct.unpack(RESPONSE_V3_FMT, resp)

            s_py = float(model_f64(x_f64).item())
            scores_py.append(s_py)
            scores_c.append(float(conf_c))
            preds.append(1 if s_py > 0.5 else 0)
            latencies.append(lat_us / 1000.0)

    ser.close()
    n_params    = sum(p.numel() for p in model_f64.parameters())
    static_ram  = n_params * 3 * 4
    return format_results("ewc", scores_py, scores_c, preds,
                          labels, task_ids, latencies,
                          tolerance=TOLERANCE_EWC, platform=f"board:{port}",
                          n_params=n_params, static_ram_bytes=static_ram)


# ── Point d'entrée ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation end-to-end PC vs firmware C (Mahalanobis + EWC)"
    )
    parser.add_argument("--model",     choices=["mahalanobis", "ewc"], default="mahalanobis",
                        help="Modèle à comparer")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Mode sans board (simulation Python)")
    parser.add_argument("--port",      default=None,
                        help="Port série (ex. /dev/ttyACM0)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-tasks",   type=int, default=3)
    parser.add_argument("--output",    default=None,
                        help="Répertoire de sortie (défaut selon le modèle)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if not args.dry_run and args.port is None:
        parser.error("Spécifier --dry-run ou --port <tty>")

    default_dirs = {"mahalanobis": "experiments/exp_S19_01",
                    "ewc":         "experiments/exp_S19_02"}
    out_dir = Path(args.output or default_dirs[args.model])

    print(f"[S2007] Modèle : {args.model} — {args.n_samples} samples / {args.n_tasks} tâches")

    if args.model == "mahalanobis":
        features, labels, task_ids = generate_synthetic_cwru(
            args.n_samples, args.n_tasks, seed=args.seed
        )
    else:
        features, labels, task_ids = generate_synthetic_monitoring(
            args.n_samples, args.n_tasks, seed=args.seed
        )

    tracemalloc.start()
    if args.dry_run:
        print(f"[S2007] Mode dry-run")
        if args.model == "mahalanobis":
            results = run_mahalanobis_dry_run(features, labels, task_ids)
        else:
            results = run_ewc_dry_run(features, labels, task_ids)
    else:
        print(f"[S2007] Mode board — port {args.port}")
        if args.model == "mahalanobis":
            results = run_mahalanobis_board(args.port, features, labels, task_ids)
        else:
            results = run_ewc_board(args.port, features, labels, task_ids)

    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["ram_peak_bytes"] = max(results["ram_peak_bytes"], ram_peak)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    tol     = results["tolerance"]
    compliant_str = "✓ PASS" if results["compliant"] else "✗ FAIL"
    print(f"[S2007] Résultats écrits dans {out_path}")
    print(f"  model              = {results['model']}")
    print(f"  n_samples          = {results['n_samples']}")
    print(f"  max_abs_delta      = {results['max_abs_delta']:.2e}  (tol={tol:.0e})  {compliant_str}")
    print(f"  mean_abs_delta     = {results['mean_abs_delta']:.2e}")
    print(f"  acc_final          = {results['acc_final']:.4f}")
    print(f"  avg_forgetting     = {results['avg_forgetting']:.4f}")
    print(f"  ram_peak_bytes     = {results['ram_peak_bytes']}")
    print(f"  inference_latency  = {results['inference_latency_ms']:.4f} ms")
    print(f"  n_params           = {results['n_params']}")
    print(f"  platform           = {results['platform']}")

    if not results["compliant"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
