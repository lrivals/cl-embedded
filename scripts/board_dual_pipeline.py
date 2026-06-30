#!/usr/bin/env python3
"""board_dual_pipeline.py — Driver hôte DUAL_MODE : RUL + Faute simultanés.

Usage:
    python scripts/board_dual_pipeline.py --dry-run --n-samples 200 --output experiments/exp_S27_01
    python scripts/board_dual_pipeline.py --port /dev/ttyACM0 --n-samples 200 --update \\
        --output experiments/exp_S27_01 --verbose

Encodage dual :
    - features[0:5]  = top-5 CMAPSS FD001 (normalisées z-score)
    - features[5:9]  = 4 features CWRU supplémentaires (slots [5:8] du loader CWRU)
    - TASK_ID        = fault_label ∈ [0, N_CLASSES-1]  (TASK_ID réutilisé en DUAL_MODE)
    - label          = rul_u8 = round(RUL / RUL_CAP × 255)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sensor_stream import (  # noqa: E402
    FRAME_FLAGS_CONSOLIDATE,
    FRAME_FLAGS_DUAL_MODE,
    FRAME_FLAGS_PROFILING,
    FRAME_FLAGS_UPDATE,
    RESPONSE_DUAL_SIZE,
    build_frame_v2,
    parse_response,
)

RUL_CAP       = 125    # = CMAPSS_RUL_CAP : cap utilisé à l'entraînement/export du modèle
                       # board (ewc_reg). DOIT matcher sinon labels SGD + décodage rul_pred
                       # sont mal échelonnés (cf. simulate_rul_board.py).
N_CMAPSS_FEAT = 5      # top-5 features CMAPSS (cmapss_feature_subset.yaml)
N_FEATURES    = 9      # features totales envoyées dans la trame DUAL_MODE


CWRU_CSV_PATH = "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv"
CMAPSS_DIR    = "data/raw/CMAPSS Jet Engine Simulated Data"


def _load_datasets(n_samples: int):
    """Charge CMAPSS FD001 (5 feat, RUL) et CWRU task 0 (9 feat, faute)."""
    from src.data.cmapss_loader import get_cl_dataloaders
    from src.data.cwru_dataset import get_cl_splits

    # CMAPSS FD001 — mode rul, top-5 features via cmapss_feature_subset.yaml
    tasks_rul = get_cl_dataloaders(
        data_dir=Path(CMAPSS_DIR),
        config_path=Path("configs/cmapss_feature_subset.yaml"),
        mode="rul",
    )
    X_rul_list, y_rul_list = [], []
    for batch_x, batch_y in tasks_rul[0]["train_loader"]:
        X_rul_list.append(batch_x.numpy())
        y_rul_list.append(batch_y.numpy())
        if sum(len(a) for a in X_rul_list) >= n_samples:
            break
    X_rul = np.concatenate(X_rul_list)[:n_samples]
    y_rul = np.clip(np.concatenate(y_rul_list).reshape(-1)[:n_samples], 0, RUL_CAP)

    # CWRU — by_fault_type, task 0
    tasks_cwru = get_cl_splits(
        csv_path=CWRU_CSV_PATH,
        scenario="by_fault_type",
        mode="multiclass",
    )
    X_cwru = np.array(tasks_cwru[0]["X_train"])[:n_samples]
    y_fault = np.array(tasks_cwru[0]["y_train"])[:n_samples]

    n = min(n_samples, len(X_rul), len(X_cwru))
    return X_rul[:n], y_rul[:n], X_cwru[:n], y_fault[:n]


def _build_dual_features(x_rul: np.ndarray, x_cwru: np.ndarray) -> np.ndarray:
    features = np.zeros(N_FEATURES, dtype=np.float32)
    features[:N_CMAPSS_FEAT] = x_rul                    # slots [0:4] — CMAPSS
    features[N_CMAPSS_FEAT:] = x_cwru[5:9]              # slots [5:8] — CWRU extra
    return features


def _dry_run_response(rul_true: float, fault_label: int) -> dict:
    """Simule une réponse board plausible pour dry-run."""
    rul_pred   = float(np.clip(rul_true + np.random.normal(0, 22), 0, RUL_CAP))
    fault_pred = fault_label if np.random.rand() > 0.4 else (fault_label + 1) % 10
    return {
        "pred":        fault_pred,
        "confidence":  float(np.random.uniform(0.4, 0.9)),
        "rul_pred":    rul_pred / RUL_CAP,              # normalisé [0,1]
        "latency_us":  int(np.random.randint(550, 720)),
        "f1_macro":    float(np.random.uniform(0.50, 0.70)),
        "rmse_rul":    float(np.random.uniform(20, 26) / RUL_CAP),
        "forgetting":  float(np.random.uniform(0.0, 0.05)),
        "mode":        "dual",
    }


def run_dual_experiment(
    ser,
    X_rul: np.ndarray,
    y_rul: np.ndarray,
    X_cwru: np.ndarray,
    y_fault: np.ndarray,
    update: bool,
    consolidate_at: int,
    verbose: bool,
    dry_run: bool,
) -> list[dict]:
    n = len(X_rul)
    results = []

    for i in range(n):
        features = _build_dual_features(X_rul[i], X_cwru[i])
        rul_u8   = int(round(float(min(y_rul[i], RUL_CAP)) / RUL_CAP * 255))
        fault_lb = int(y_fault[i])

        flags = FRAME_FLAGS_DUAL_MODE | FRAME_FLAGS_PROFILING
        if update:
            flags |= FRAME_FLAGS_UPDATE
        if consolidate_at > 0 and i > 0 and i % consolidate_at == 0:
            flags |= FRAME_FLAGS_CONSOLIDATE

        if dry_run:
            resp = _dry_run_response(float(y_rul[i]), fault_lb)
        else:
            frame = build_frame_v2(
                features=features,
                label=rul_u8,
                task_id=fault_lb,
                ts_ms=int(time.time() * 1000) & 0xFFFFFFFF,
                flags=flags,
            )
            ser.write(frame)
            raw = ser.read(RESPONSE_DUAL_SIZE)
            if len(raw) != RESPONSE_DUAL_SIZE:
                print(f"[WARN] sample {i}: réponse {len(raw)} B (attendu {RESPONSE_DUAL_SIZE})")
                continue
            resp = parse_response(raw)

        rul_pred_cycles = resp["rul_pred"] * RUL_CAP
        results.append({
            "i":          i,
            "rul_true":   float(y_rul[i]),
            "rul_pred":   rul_pred_cycles,
            "fault_true": fault_lb,
            "fault_pred": resp["pred"],
            "conf_fault": resp["confidence"],
            "latency_us": resp["latency_us"],
            "f1_board":   resp["f1_macro"],
            "rmse_board": resp["rmse_rul"],
            "forgetting": resp["forgetting"],
        })

        if verbose and i % 50 == 0:
            print(f"  [{i:3d}] RUL true={y_rul[i]:.0f} pred={rul_pred_cycles:.1f} | "
                  f"fault true={fault_lb} pred={resp['pred']} | lat={resp['latency_us']} µs")

    return results


def compute_offline_metrics(results: list[dict]) -> dict:
    if not results:
        return {"rmse_rul_offline": None, "f1_fault_offline": None}
    rul_t = np.array([r["rul_true"]   for r in results])
    rul_p = np.array([r["rul_pred"]   for r in results])
    f_t   = np.array([r["fault_true"] for r in results])
    f_p   = np.array([r["fault_pred"] for r in results])

    rmse = float(np.sqrt(np.mean((rul_t - rul_p) ** 2)))

    classes = np.unique(f_t)
    f1s = []
    for c in classes:
        tp = int(np.sum((f_t == c) & (f_p == c)))
        fp = int(np.sum((f_t != c) & (f_p == c)))
        fn = int(np.sum((f_t == c) & (f_p != c)))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)

    return {"rmse_rul_offline": rmse, "f1_fault_offline": float(np.mean(f1s))}


def main() -> None:
    parser = argparse.ArgumentParser(description="DUAL_MODE board pipeline : RUL + Faute simultanés")
    parser.add_argument("--port",            default="/dev/ttyACM0")
    parser.add_argument("--baud",            type=int, default=115200)
    parser.add_argument("--n-samples",       type=int, default=200)
    parser.add_argument("--update",          action="store_true")
    parser.add_argument("--consolidate-at",  type=int, default=0)
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--output",          default="experiments/exp_S27_01")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement datasets CMAPSS FD001 + CWRU (n={args.n_samples})...")
    X_rul, y_rul, X_cwru, y_fault = _load_datasets(args.n_samples)
    n = len(X_rul)
    print(f"  CMAPSS: {n} samples | CWRU: {n} samples")

    if args.dry_run:
        print("Mode dry-run (pas de board)")
        ser = None
    else:
        import serial  # type: ignore[import]
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
        time.sleep(0.5)
        # Purge des octets résiduels post-reset (TODO(dorra)) — évite tout
        # désalignement de trame au démarrage de l'expérience.
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Board connectée sur {args.port} @ {args.baud} baud")

    print(f"Lancement expérience DUAL_MODE ({n} samples, update={args.update})...")
    results = run_dual_experiment(
        ser, X_rul, y_rul, X_cwru, y_fault,
        update=args.update,
        consolidate_at=args.consolidate_at,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    if ser:
        ser.close()

    if not results:
        print("[ERROR] Aucun résultat — vérifier la connexion board")
        sys.exit(1)

    lat_values = [r["latency_us"] for r in results]
    board_metrics = {
        "rmse_rul":    float(results[-1]["rmse_board"] * RUL_CAP),
        "f1_fault":    float(results[-1]["f1_board"]),
        "lat_mean_us": float(np.mean(lat_values)),
        "lat_p99_us":  float(np.percentile(lat_values, 99)),
        "forgetting":  float(results[-1]["forgetting"]),
    }

    offline = compute_offline_metrics(results)

    output = {
        "experiment":      out_dir.name,
        "mode":            "dual",
        "dataset_rul":     "CMAPSS_FD001",
        "dataset_fault":   "CWRU_task0",
        "n_samples":       n,
        "update":          args.update,
        "metrics_board":   board_metrics,
        "metrics_offline": offline,
        "bss_bytes":       66748,
        "samples":         results,
    }

    out_path = out_dir / "dual_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nRésultats sauvegardés : {out_path}")
    print(f"  RMSE_RUL  : {board_metrics['rmse_rul']:.2f} cycles")
    print(f"  F1_fault  : {board_metrics['f1_fault']:.3f}")
    print(f"  Lat mean  : {board_metrics['lat_mean_us']:.0f} µs")
    print(f"  Lat P99   : {board_metrics['lat_p99_us']:.0f} µs")
    print(f"  RMSE_off  : {offline['rmse_rul_offline']:.2f} (offline)")
    print(f"  F1_off    : {offline['f1_fault_offline']:.3f} (offline)")


if __name__ == "__main__":
    main()
