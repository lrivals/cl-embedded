"""
scripts/train_kmeans.py — KMeans sur CWRU : single-task (exp_071) et CL by_fault_type (exp_077).

Usage
-----
    # Single-task
    python scripts/train_kmeans.py --config configs/cwru_single_task_config.yaml --exp_id exp_071
    # CL by_fault_type
    python scripts/train_kmeans.py --config configs/cwru_by_fault_config.yaml --exp_id exp_077_kmeans_cwru_by_fault_type

Sortie
------
    experiments/<exp_id>/
    ├── config_snapshot.yaml
    └── results/
        ├── metrics_single_task.json   (mode no_split)
        └── metrics_cl.json            (mode by_fault_type)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.metrics import compute_cl_metrics
from src.models.unsupervised import KMeansDetector
from src.utils.config_loader import load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KMeans CWRU — single-task et CL by_fault_type")
    parser.add_argument("--config", default="configs/cwru_single_task_config.yaml")
    parser.add_argument("--exp_id", default=None, help="Override exp_id")
    parser.add_argument("--exp_dir", default=None, help="Override répertoire expérience")
    return parser.parse_args()


def _extract_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy().ravel())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _profile_model(model: KMeansDetector, X_sample: np.ndarray, n_runs: int = 100) -> dict:
    x_single = X_sample[:1]
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.anomaly_score(x_single)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    tracemalloc.start()
    model.anomaly_score(x_single)
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_params = int(model.count_parameters())
    return {
        "inference_latency_ms": float(np.mean(latencies_ms)),
        "ram_peak_bytes": int(ram_peak),
        "n_params": n_params,
        "within_budget_64ko": (n_params * 4) <= 65536,
    }


def _resolve_feature_names(cfg: dict) -> list[str]:
    from src.evaluation.feature_importance import (
        FEATURE_NAMES_CWRU,
        FEATURE_NAMES_PRONOSTIA,
        FEATURE_NAMES_MONITORING,
    )
    dataset = cfg["data"].get("dataset", "")
    if dataset == "cwru":
        return FEATURE_NAMES_CWRU
    if dataset == "pronostia":
        return FEATURE_NAMES_PRONOSTIA
    return FEATURE_NAMES_MONITORING


def _extract_test_arrays(task: dict) -> tuple[np.ndarray, np.ndarray]:
    loader = task.get("test_loader") or task["val_loader"]
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy().ravel())
    return np.concatenate(X_list), np.concatenate(y_list)


def _run_cl(
    tasks: list[dict],
    model: KMeansDetector,
    cfg: dict,
    exp_id: str,
    results_dir: Path,
    exp_dir: Path,
) -> None:
    """Boucle CL domain-incremental (stratégie refit) — 3 tâches Ball→IR→OR."""
    n_tasks = len(tasks)
    percentile = cfg["kmeans"].get("anomaly_percentile", 95)
    n_latency_runs = cfg.get("evaluation", {}).get("n_latency_runs", 100)

    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    X_train_last = None
    thresholds_per_task: dict[int, float] = {}

    for i, task in enumerate(tasks):
        domain = task.get("domain", f"Task {i}")
        print(f"\n--- Tâche {i + 1}/{n_tasks} : {domain} ---")

        X_train, _ = _extract_numpy(task["train_loader"])
        X_train_last = X_train
        model.fit_task(X_train, task_id=i)

        # Utilise le seuil EMA du modèle (fixé sur Task 0, MAJ EMA sur tâches suivantes)
        threshold_i = model.threshold_
        thresholds_per_task[i] = threshold_i
        print(f"  Seuil Task {i} (EMA threshold) : {threshold_i:.4f}")

        for j in range(i + 1):
            X_val, y_val = _extract_numpy(tasks[j]["val_loader"])
            scores = model.anomaly_score(X_val)
            y_pred = (scores > threshold_i).astype(int)
            acc = float(accuracy_score(y_val, y_pred))
            acc_matrix[i, j] = acc
            lbl = tasks[j].get("domain", f"T{j + 1}")
            print(f"  Acc tâche {j + 1} ({lbl}) [seuil={threshold_i:.4f}]: {acc:.4f}")

    cl_metrics = compute_cl_metrics(acc_matrix)
    print(f"\nAA={cl_metrics['aa']:.4f} | AF={cl_metrics['af']:.4f} | BWT={cl_metrics['bwt']:.4f}")

    mem = _profile_model(model, X_train_last, n_runs=n_latency_runs)
    print(f"  RAM peak: {mem['ram_peak_bytes'] / 1024:.1f} Ko  |  "
          f"Latence: {mem['inference_latency_ms']:.3f} ms  |  "
          f"n_params: {mem['n_params']}")

    metrics: dict = {
        "exp_id": exp_id,
        "model": "kmeans",
        "dataset": "cwru",
        "scenario": cfg["data"].get("task_split", "by_fault_type"),
        "acc_final": cl_metrics["aa"],
        "avg_forgetting": cl_metrics["af"],
        "backward_transfer": cl_metrics["bwt"],
        "per_task_acc": [float(acc_matrix[n_tasks - 1, j]) for j in range(n_tasks)],
        "ram_peak_bytes": mem["ram_peak_bytes"],
        "inference_latency_ms": mem["inference_latency_ms"],
        "n_params": mem["n_params"],
        "acc_matrix": acc_matrix.tolist(),
        "k_selected_per_task": model.k_selected_,
        "threshold_history": model.threshold_history_,
        "ema_alpha": model.ema_alpha,
    }

    metrics_path = results_dir / "metrics_cl.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")

    # ── Feature importance ────────────────────────────────────────────────────
    from src.evaluation.feature_importance import (
        permutation_importance,
        permutation_importance_per_task,
    )

    feature_names = _resolve_feature_names(cfg)
    threshold_fi = model.threshold_

    task_arrays: list[dict] = []
    for t in tasks:
        X_t, y_t = _extract_test_arrays(t)
        task_arrays.append({"task_name": t.get("domain", f"task_{t['task_id']}"), "X": X_t, "y": y_t})

    X_all = np.concatenate([t["X"] for t in task_arrays])
    y_all = np.concatenate([t["y"] for t in task_arrays])

    global_imp = permutation_importance(
        model.anomaly_score, X_all, y_all, feature_names, threshold=threshold_fi
    )
    per_task_imp = permutation_importance_per_task(
        model.anomaly_score, task_arrays, feature_names, threshold=threshold_fi
    )

    is_pronostia = cfg["data"].get("task_split") == "by_condition"
    importance_results = {
        "model": "kmeans",
        "dataset": "pronostia" if is_pronostia else "cwru",
        "scenario": cfg["data"].get("task_split", "by_fault_type"),
        "global": {"permutation_importance": global_imp},
        "per_task": {
            name: {"permutation_importance": imp}
            for name, imp in per_task_imp.items()
        },
    }

    importance_path = results_dir / "feature_importance.json"
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(importance_results, f, indent=2)
    print(f"  Feature importance → {importance_path}")
    print(f"✅ KMeans CL terminé → {exp_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("evaluation", {}).get("seed", 42))

    if args.exp_id:
        cfg["exp_id"] = args.exp_id
        cfg["evaluation"]["output_dir"] = f"experiments/{args.exp_id}/results/"
    if args.exp_dir:
        cfg["evaluation"]["output_dir"] = str(Path(args.exp_dir) / "results")
        cfg["exp_id"] = Path(args.exp_dir).name

    exp_id = cfg["exp_id"]
    results_dir = Path(cfg["evaluation"]["output_dir"])
    exp_dir = results_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, str(exp_dir))

    task_split = cfg["data"].get("task_split", "no_split")

    if task_split == "by_fault_type":
        print(f"\n{'=' * 60}")
        print(f"  KMeans CL by_fault_type — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.cwru_dataset import get_cwru_cl_dataloaders_by_fault_type
        tasks = get_cwru_cl_dataloaders_by_fault_type(
            csv_path=Path(cfg["data"]["csv_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            test_ratio=cfg["data"].get("test_ratio", 0.2),
            val_ratio=cfg["data"].get("val_ratio", 0.1),
            seed=cfg["data"].get("random_state", 42),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = KMeansDetector(cfg["kmeans"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_severity":
        print(f"\n{'=' * 60}")
        print(f"  KMeans CL by_severity — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.cwru_dataset import get_cwru_cl_dataloaders_by_severity
        tasks = get_cwru_cl_dataloaders_by_severity(
            csv_path=Path(cfg["data"]["csv_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            test_ratio=cfg["data"].get("test_ratio", 0.2),
            val_ratio=cfg["data"].get("val_ratio", 0.1),
            seed=cfg["data"].get("random_state", 42),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = KMeansDetector(cfg["kmeans"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_condition":
        print(f"\n{'=' * 60}")
        print(f"  KMeans CL by_condition (Pronostia) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.pronostia_dataset import get_pronostia_dataloaders
        tasks = get_pronostia_dataloaders(
            npy_dir=Path(cfg["data"]["npy_dir"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=cfg.get("evaluation", {}).get("seed", 42),
            window_size=cfg["data"].get("window_size", 2560),
            step_size=cfg["data"].get("step_size", 2560),
            failure_ratio=cfg["data"].get("failure_ratio", 0.10),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = KMeansDetector(cfg["kmeans"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    # --- Mode single-task (no_split) ---
    print(f"\n{'=' * 60}")
    print(f"  KMeans Single-Task — {exp_id}")
    print(f"  Sortie : {exp_dir}")
    print(f"{'=' * 60}\n")

    from src.data.cwru_dataset import get_cwru_dataloaders_single_task
    data = get_cwru_dataloaders_single_task(
        csv_path=Path(cfg["data"]["csv_path"]),
        batch_size=cfg["data"].get("batch_size", 32),
        test_ratio=cfg["data"].get("test_ratio", 0.2),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        seed=cfg["data"].get("random_state", 42),
    )

    X_train, _ = _extract_numpy(data["train_loader"])
    X_test, y_test = _extract_numpy(data["test_loader"])
    print(f"  {data['n_train']} train | {data['n_val']} val | {data['n_test']} test")

    model = KMeansDetector(cfg["kmeans"])
    model.fit_task(X_train, task_id=0)

    test_scores = model.anomaly_score(X_test)
    train_scores = model.anomaly_score(X_train)
    percentile_val = cfg["kmeans"].get("anomaly_percentile", 95)
    threshold = float(np.percentile(train_scores, percentile_val))
    y_pred = (test_scores > threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, test_scores))
    except ValueError:
        auc = float("nan")

    print(f"  Test → accuracy={acc:.4f} | f1={f1:.4f} | auc_roc={auc:.4f}")

    n_latency_runs = cfg.get("evaluation", {}).get("n_latency_runs", 100)
    mem = _profile_model(model, X_test, n_runs=n_latency_runs)
    print(f"  RAM peak: {mem['ram_peak_bytes'] / 1024:.1f} Ko  |  "
          f"Latence: {mem['inference_latency_ms']:.3f} ms  |  "
          f"n_params: {mem['n_params']}")

    metrics: dict = {
        "exp_id": exp_id,
        "model": "kmeans",
        "dataset": "cwru",
        "scenario": "no_split",
        "acc_final": acc,
        "f1_score": f1,
        "auc_roc": auc,
        "ram_peak_bytes": mem["ram_peak_bytes"],
        "inference_latency_ms": mem["inference_latency_ms"],
        "n_params": mem["n_params"],
    }

    metrics_path = results_dir / "metrics_single_task.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")
    print(f"✅ KMeans single-task terminé → {exp_dir}")


if __name__ == "__main__":
    main()
