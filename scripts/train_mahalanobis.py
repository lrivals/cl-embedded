"""
scripts/train_mahalanobis.py — Mahalanobis sur CWRU : single-task (exp_072) et CL by_fault_type (exp_078).

Usage
-----
    # Single-task
    python scripts/train_mahalanobis.py --config configs/cwru_single_task_config.yaml --exp_id exp_072
    # CL by_fault_type
    python scripts/train_mahalanobis.py --config configs/cwru_by_fault_config.yaml --exp_id exp_078_mahalanobis_cwru_by_fault_type

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
from src.models.unsupervised import MahalanobisDetector
from src.utils.config_loader import load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mahalanobis — single-task et CL (CWRU / Pronostia / Monitoring)")
    parser.add_argument("--config", default="configs/cwru_single_task_config.yaml")
    parser.add_argument("--data_config", default=None, help="Config data override (ex. configs/monitoring_by_location_config.yaml)")
    parser.add_argument("--exp_id", default=None, help="Override exp_id")
    parser.add_argument("--exp_dir", default=None, help="Override répertoire expérience")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset override : cwru | pronostia | pump | monitoring | cmapss | paderborn",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scénario CL : by_fault_type | by_condition | temporal | by_equipment | by_severity",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Active le profiling RAM par tracemalloc sur toute la durée du training",
    )
    parser.add_argument(
        "--output_dir",
        dest="exp_dir",
        default=None,
        help="Alias de --exp_dir (compatibilité commandes S2405)",
    )
    return parser.parse_args()


def _dump_checkpoint(model: MahalanobisDetector, exp_dir: Path, task_id: int) -> None:
    """Sérialise le détecteur fitté (pickle) pour l'export poids board.

    S3205 : ``export_weights_c.py --mahal`` lit ce ``.pkl`` (attrs ``mu_`` /
    ``sigma_inv_`` / ``threshold_``) pour générer ``model_weights.h`` et garantir
    la parité numérique board↔PC. Sans checkpoint, aucun poids exportable.
    """
    import pickle  # noqa: PLC0415

    if model.mu_ is None or model.sigma_inv_ is None:
        print("  [checkpoint] modèle non fitté — pickle ignoré")
        return
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"mahalanobis_task{task_id}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [checkpoint] détecteur Mahalanobis → {ckpt_path}")


def _extract_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy().ravel())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _profile_model(model: MahalanobisDetector, X_sample: np.ndarray, n_runs: int = 100) -> dict:
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
    if dataset == "paderborn":
        import yaml as _yaml
        subset_path = cfg["data"].get("feature_subset_path", "configs/paderborn_feature_subset.yaml")
        with open(subset_path) as _f:
            _subset = _yaml.safe_load(_f)
        return _subset.get("selected_features", [])
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
    model: MahalanobisDetector,
    cfg: dict,
    exp_id: str,
    results_dir: Path,
    exp_dir: Path,
) -> None:
    """Boucle CL domain-incremental (stratégie refit) — 3 tâches Ball→IR→OR."""
    n_tasks = len(tasks)
    percentile = cfg["mahalanobis"].get("anomaly_percentile", 95)
    n_latency_runs = cfg.get("evaluation", {}).get("n_latency_runs", 100)

    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    X_train_last = None

    _training_ram_peak: int | None = None
    if cfg.get("_profile_memory"):
        import tracemalloc as _tracemalloc
        _tracemalloc.start()

    for i, task in enumerate(tasks):
        domain = task.get("domain", f"Task {i}")
        print(f"\n--- Tâche {i + 1}/{n_tasks} : {domain} ---")

        X_train, _ = _extract_numpy(task["train_loader"])
        X_train_last = X_train
        model.fit_task(X_train, task_id=i)

        # En mode welford, threshold_ est figé sur Task 0 et jamais modifié
        threshold = model.threshold_
        print(f"  Seuil (model.threshold_) : {threshold:.4f}")

        for j in range(i + 1):
            X_val, y_val = _extract_numpy(tasks[j]["val_loader"])
            scores = model.anomaly_score(X_val)
            y_pred = (scores > threshold).astype(int)
            acc = float(accuracy_score(y_val, y_pred))
            acc_matrix[i, j] = acc
            lbl = tasks[j].get("domain", f"T{j + 1}")
            print(f"  Acc tâche {j + 1} ({lbl}): {acc:.4f}")

    cl_metrics = compute_cl_metrics(acc_matrix)
    print(f"\nAA={cl_metrics['aa']:.4f} | AF={cl_metrics['af']:.4f} | BWT={cl_metrics['bwt']:.4f}")

    if cfg.get("_profile_memory"):
        _, _training_ram_peak = _tracemalloc.get_traced_memory()
        _tracemalloc.stop()

    mem = _profile_model(model, X_train_last, n_runs=n_latency_runs)
    print(f"  RAM peak: {mem['ram_peak_bytes'] / 1024:.1f} Ko  |  "
          f"Latence: {mem['inference_latency_ms']:.3f} ms  |  "
          f"n_params: {mem['n_params']}")

    metrics: dict = {
        "exp_id": exp_id,
        "model": "mahalanobis",
        "dataset": cfg["data"].get("dataset", "cwru"),
        "scenario": cfg["data"].get("task_split", "by_fault_type"),
        "acc_final": cl_metrics["aa"],
        "avg_forgetting": cl_metrics["af"],
        "backward_transfer": cl_metrics["bwt"],
        "per_task_acc": [float(acc_matrix[n_tasks - 1, j]) for j in range(n_tasks)],
        "ram_peak_bytes": mem["ram_peak_bytes"],
        "inference_latency_ms": mem["inference_latency_ms"],
        "n_params": mem["n_params"],
        "acc_matrix": acc_matrix.tolist(),
        "welford_updates_per_task": model.welford_updates_per_task_,
        "cl_strategy": model.cl_strategy,
    }

    if _training_ram_peak is not None:
        metrics["ram_training_peak_bytes"] = _training_ram_peak
    metrics_path = results_dir / "metrics_cl.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")

    # Checkpoint du détecteur fitté (S3205 : requis pour l'export poids board → parité).
    _dump_checkpoint(model, exp_dir, task_id=n_tasks - 1)
    # results.json à la racine de exp_dir (format flat Sprint 24)
    _results_flat = {k: v for k, v in metrics.items() if k != "acc_matrix"}
    _results_flat["sprint"] = 24
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_results_flat, f, indent=2)

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
    _dataset = "pronostia" if is_pronostia else cfg["data"].get("dataset", "cwru")
    importance_results = {
        "model": "mahalanobis",
        "dataset": _dataset,
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
    print(f"✅ Mahalanobis CL terminé → {exp_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_config:
        data_cfg = load_config(args.data_config)
        cfg["data"].update(data_cfg.get("data", {}))

    # Résolution --dataset/--scenario → data config (S2405)
    _DATASET_CONFIG_MAP: dict[tuple[str | None, str | None], str] = {
        ("cwru", "by_fault_type"):     "configs/cwru_by_fault_config.yaml",
        ("cwru", "by_severity"):       "configs/cwru_by_severity_config.yaml",
        ("pronostia", "by_condition"): "configs/pronostia_config.yaml",
        ("pump", "temporal"):          "configs/pump_by_temporal_window_config.yaml",
        ("pump", "by_id"):             "configs/pump_by_id_config.yaml",
        ("cmapss", None):              "configs/cmapss_config.yaml",
        ("paderborn", None):           "configs/paderborn_config.yaml",
    }
    if args.dataset:
        key = (args.dataset, args.scenario)
        fallback_key = (args.dataset, None)
        data_cfg_path = _DATASET_CONFIG_MAP.get(key) or _DATASET_CONFIG_MAP.get(fallback_key)
        if data_cfg_path:
            data_cfg = load_config(data_cfg_path)
            cfg["data"].update(data_cfg.get("data", {}))
        if args.scenario:
            cfg["data"]["scenario"] = args.scenario

    cfg["_profile_memory"] = args.profile_memory

    set_seed(cfg.get("evaluation", {}).get("seed", 42))

    if args.exp_id:
        cfg["exp_id"] = args.exp_id
        cfg.setdefault("evaluation", {})["output_dir"] = f"experiments/{args.exp_id}/results/"
    if args.exp_dir:
        cfg.setdefault("evaluation", {})["output_dir"] = str(Path(args.exp_dir) / "results")
        cfg["exp_id"] = Path(args.exp_dir).name

    exp_id = cfg["exp_id"]
    results_dir = Path(cfg["evaluation"]["output_dir"])
    exp_dir = results_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, str(exp_dir))

    dataset = cfg["data"].get("dataset", "")
    if dataset == "paderborn":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL domain-incremental (Paderborn) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")
        from src.data.paderborn_loader import get_cl_dataloaders as get_paderborn_cl_dataloaders
        tasks = get_paderborn_cl_dataloaders(
            data_dir=Path(cfg["data"]["data_dir"]),
            config_path=Path(args.config),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")
        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    task_split = cfg["data"].get("task_split", "no_split")

    if task_split == "by_fault_type":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_fault_type — {exp_id}")
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

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_severity":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_severity — {exp_id}")
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

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_condition":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_condition (Pronostia) — {exp_id}")
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
            label_mode=cfg["data"].get("label_mode", "failure_ratio"),
            faulty_threshold=cfg["data"].get("faulty_threshold"),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_temporal_window":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_temporal_window (Battery) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.battery_dataset import (
            RUL_FAILURE_THRESHOLD,
            get_battery_dataloaders,
        )
        tasks = get_battery_dataloaders(
            csv_path=Path(cfg["data"]["csv_path"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=cfg.get("evaluation", {}).get("seed", 42),
            n_tasks=cfg["data"].get("n_tasks", 3),
            rul_failure_threshold=cfg["data"].get(
                "rul_failure_threshold", RUL_FAILURE_THRESHOLD
            ),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_equipment":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_equipment (Monitoring) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.monitoring_dataset import get_cl_dataloaders
        tasks = get_cl_dataloaders(
            csv_path=Path(cfg["data"]["csv_path"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=cfg.get("evaluation", {}).get("seed", 42),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_location":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_location (Monitoring) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.monitoring_dataset import get_cl_dataloaders_by_location
        tasks = get_cl_dataloaders_by_location(
            csv_path=Path(cfg["data"]["csv_path"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=cfg.get("evaluation", {}).get("seed", 42),
            location_order=cfg["data"].get("location_order"),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    elif task_split == "by_domain":
        print(f"\n{'=' * 60}")
        print(f"  Mahalanobis CL by_domain (CMAPSS) — {exp_id}")
        print(f"  Sortie : {exp_dir}")
        print(f"{'=' * 60}\n")

        from src.data.cmapss_loader import get_cl_dataloaders
        tasks = get_cl_dataloaders(
            data_dir=Path(cfg["data"]["data_dir"]),
            config_path=Path(args.config),
        )
        for t in tasks:
            print(f"  Task {t['task_id']} ({t['domain']}): {t['n_train']} train | {t['n_val']} val")

        model = MahalanobisDetector(cfg["mahalanobis"])
        _run_cl(tasks, model, cfg, exp_id, results_dir, exp_dir)
        return

    # --- Mode single-task (no_split) ---
    print(f"\n{'=' * 60}")
    print(f"  Mahalanobis Single-Task — {exp_id}")
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

    model = MahalanobisDetector(cfg["mahalanobis"])
    model.fit_task(X_train, task_id=0)

    test_scores = model.anomaly_score(X_test)
    train_scores = model.anomaly_score(X_train)
    percentile_val = cfg["mahalanobis"].get("anomaly_percentile", 95)
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
        "model": "mahalanobis",
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
    _dump_checkpoint(model, exp_dir, task_id=0)
    print(f"✅ Mahalanobis single-task terminé → {exp_dir}")


if __name__ == "__main__":
    main()
