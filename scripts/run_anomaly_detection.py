"""
scripts/run_anomaly_detection.py — Scénario anomaly detection CL (DBSCAN ou EWCOneClass).

Usage
-----
    python scripts/run_anomaly_detection.py \
        --model dbscan \
        --config configs/unsupervised_config.yaml \
        --exp_id exp_123 \
        --strategy refit

    python scripts/run_anomaly_detection.py \
        --model ewc_oneclass \
        --config configs/ewc_oneclass_config.yaml \
        --exp_id exp_125 \
        --strategy refit

    # Pronostia by_condition
    python scripts/run_anomaly_detection.py \
        --model hdc \
        --config configs/hdc_config.yaml \
        --exp_id exp_137 \
        --strategy refit \
        --dataset pronostia \
        --scenario by_condition

Sortie
------
    experiments/<exp_id>/
    ├── config_snapshot.yaml
    └── results/
        ├── metrics_anomaly.json
        └── auroc_matrix.npy
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.cwru_dataset import (
    FEATURE_COLS as CWRU_FEATURE_NAMES,
    get_cwru_dataloaders_anomaly_detection,
)
from src.data.monitoring_dataset import get_cl_dataloaders_anomaly_detection
from src.data.pronostia_dataset import (
    FEATURE_NAMES,
    get_pronostia_dataloaders_anomaly_detection,
)
from src.evaluation.anomaly_metrics import compute_cl_anomaly_metrics, save_anomaly_metrics
from src.training.scenarios import run_anomaly_detection_scenario
from src.utils.config_loader import load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly detection CL — DBSCAN ou EWCOneClass")
    parser.add_argument(
        "--model",
        required=True,
        choices=["dbscan", "ewc_oneclass", "hdc", "tinyol_ae", "kmeans", "mahalanobis"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--strategy", required=True, choices=["refit", "accumulate"])
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--scenario", default=None)
    return parser.parse_args()


def _build_hdc(cfg: dict, strategy: str):
    from src.models.hdc.hdc_classifier import HDCClassifier

    cfg["one_class_mode"] = True
    cfg["cl_strategy"] = strategy
    return HDCClassifier(cfg)


def _build_tinyol_ae(cfg: dict, strategy: str):
    from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector

    return TinyOLAnomalyDetector(cfg)


def _build_kmeans(cfg: dict, strategy: str):
    from src.models.unsupervised import KMeansDetector

    cfg["kmeans"]["cl_strategy"] = strategy
    return KMeansDetector(cfg["kmeans"])


def _build_mahalanobis(cfg: dict, strategy: str):
    from src.models.unsupervised import MahalanobisDetector

    cfg["mahalanobis"]["cl_strategy"] = strategy
    return MahalanobisDetector(cfg["mahalanobis"])


def _build_dbscan(cfg: dict, strategy: str):
    from src.models.unsupervised import DBSCANDetector

    cfg_dbscan = dict(cfg["dbscan"])
    cfg_dbscan["cl_strategy"] = strategy
    return DBSCANDetector(cfg_dbscan)


def _build_ewc_oneclass(cfg: dict, strategy: str):
    from src.models.ewc.ewc_oneclass import EWCOneClassDetector

    model_cfg = cfg.get("MODEL", {})
    train_cfg = cfg.get("TRAINING", {})

    input_dim = int(cfg.get("input_dim", model_cfg.get("INPUT_DIM", 4)))
    hidden_dim = int(model_cfg.get("HIDDEN_DIM", 32))
    latent_dim = int(model_cfg.get("LATENT_DIM", 8))
    n_epochs = int(train_cfg.get("N_EPOCHS", 20))
    lr = float(train_cfg.get("LR", 1e-3))

    lambda_ewc = 400.0 if strategy == "accumulate" else 0.0

    return EWCOneClassDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lambda_ewc=lambda_ewc,
        n_epochs=n_epochs,
        lr=lr,
    )


def _profile_model(model, X_sample: np.ndarray, n_runs: int = 100) -> dict:
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
    }


def _load_cwru_tasks(cfg: dict, seed: int, scenario: str | None = None) -> list[dict]:
    """Charge les tâches CWRU pour l'anomaly detection one-class."""
    ds_cfg = cfg.get("DATASETS", {}).get("cwru", {})
    csv_path = Path(ds_cfg.get(
        "csv_path",
        "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv",
    ))
    effective_scenario = scenario or ds_cfg.get("SPLIT_STRATEGY", "by_severity")
    batch_size = int(ds_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 32)))
    return get_cwru_dataloaders_anomaly_detection(
        data_path=csv_path,
        scenario=effective_scenario,
        batch_size=batch_size,
        seed=seed,
    )


def _apply_cwru_overrides(cfg: dict, model_name: str, tasks: list[dict]) -> None:
    """
    Applique les overrides dataset-specific CWRU sur cfg avant construction du modèle.

    Patche cfg in-place — les builders lisent directement les clés modifiées.
    """
    ds_cfg = cfg.get("DATASETS", {}).get("cwru", {})

    if model_name == "ewc_oneclass":
        model_cfg = cfg.setdefault("MODEL", {})
        if "INPUT_DIM" in ds_cfg:
            model_cfg["INPUT_DIM"] = ds_cfg["INPUT_DIM"]

    elif model_name == "hdc":
        cfg.setdefault("data", {})["n_features"] = ds_cfg.get("INPUT_DIM", 9)
        bv_path = ds_cfg.get("base_vectors_path")
        if bv_path and "hdc" in cfg:
            cfg["hdc"]["base_vectors_path"] = bv_path
            # Synchroniser D avec la dimension réelle du fichier de base vectors
            d_override = ds_cfg.get("D")
            if d_override is not None:
                cfg["hdc"]["D"] = int(d_override)
        X_t0 = np.concatenate([b[0].numpy() for b in tasks[0]["train_loader"]])
        feature_bounds = {
            CWRU_FEATURE_NAMES[i]: [float(X_t0[:, i].min()), float(X_t0[:, i].max())]
            for i in range(X_t0.shape[1])
        }
        cfg["feature_bounds"] = feature_bounds

    elif model_name == "tinyol_ae":
        backbone_overrides = ds_cfg.get("backbone", {})
        if backbone_overrides:
            cfg.setdefault("backbone", {}).update(backbone_overrides)

    elif model_name == "kmeans":
        kmeans_overrides = ds_cfg.get("kmeans", {})
        if kmeans_overrides:
            n_clusters = kmeans_overrides.get("N_CLUSTERS")
            if n_clusters is not None:
                cfg["kmeans"]["k_fixed"] = int(n_clusters)
                cfg["kmeans"]["k_method"] = "fixed"

    elif model_name == "mahalanobis":
        maha_overrides = ds_cfg.get("mahalanobis", {})
        if maha_overrides:
            reg_covar = maha_overrides.get("REG_COVAR")
            if reg_covar is not None:
                cfg["mahalanobis"]["reg_covar"] = float(reg_covar)

    elif model_name == "dbscan":
        dbscan_overrides = ds_cfg.get("dbscan", {})
        if dbscan_overrides:
            epsilon = dbscan_overrides.get("EPS")
            if epsilon is not None:
                cfg["dbscan"]["EPSILON"] = float(epsilon)
            min_samples = dbscan_overrides.get("MIN_SAMPLES")
            if min_samples is not None:
                cfg["dbscan"]["MIN_SAMPLES"] = int(min_samples)

    if model_name == "ewc_oneclass":
        ewc_overrides = ds_cfg.get("ewc_oneclass", {})
        pct = ewc_overrides.get("THRESHOLD_PERCENTILE")
        if pct is not None:
            cfg.setdefault("TRAINING", {})["THRESHOLD_PERCENTILE"] = int(pct)


def _load_pronostia_tasks(cfg: dict, seed: int, failure_ratio: float) -> list[dict]:
    """Charge les tâches Pronostia by_condition pour l'anomaly detection."""
    ds_cfg = cfg.get("DATASETS", {}).get("pronostia", {})
    npy_dir = Path(ds_cfg.get("npy_dir", "data/raw/Pronostia dataset/binaries"))
    normalizer_path = Path(ds_cfg.get("normalizer_path", "configs/pronostia_normalizer.yaml"))
    batch_size = int(ds_cfg.get("batch_size", cfg.get("data", {}).get("batch_size", 32)))
    return get_pronostia_dataloaders_anomaly_detection(
        npy_dir=npy_dir,
        normalizer_path=normalizer_path,
        failure_ratio=failure_ratio,
        batch_size=batch_size,
        seed=seed,
    )


def _apply_pronostia_overrides(cfg: dict, model_name: str, tasks: list[dict]) -> None:
    """
    Applique les overrides dataset-specific Pronostia sur cfg avant construction du modèle.

    Patche cfg in-place — les builders lisent directement les clés modifiées.
    """
    ds_cfg = cfg.get("DATASETS", {}).get("pronostia", {})

    if model_name == "ewc_oneclass":
        model_cfg = cfg.setdefault("MODEL", {})
        if "INPUT_DIM" in ds_cfg:
            model_cfg["INPUT_DIM"] = ds_cfg["INPUT_DIM"]
        if "HIDDEN_DIM" in ds_cfg:
            model_cfg["HIDDEN_DIM"] = ds_cfg["HIDDEN_DIM"]

    elif model_name == "hdc":
        cfg.setdefault("data", {})["n_features"] = ds_cfg.get("n_features", 13)
        if "base_vectors_path" in ds_cfg:
            cfg["hdc"]["base_vectors_path"] = ds_cfg["base_vectors_path"]
        # Calculer feature_bounds depuis Task 1 train data
        X_t1 = np.concatenate([b[0].numpy() for b in tasks[0]["train_loader"]])
        feature_bounds = {
            FEATURE_NAMES[i]: [float(X_t1[:, i].min()), float(X_t1[:, i].max())]
            for i in range(X_t1.shape[1])
        }
        cfg["feature_bounds"] = feature_bounds

    elif model_name == "tinyol_ae":
        backbone_overrides = ds_cfg.get("backbone", {})
        if backbone_overrides:
            cfg.setdefault("backbone", {}).update(backbone_overrides)

    elif model_name == "kmeans":
        kmeans_overrides = ds_cfg.get("kmeans", {})
        if kmeans_overrides:
            n_clusters = kmeans_overrides.get("N_CLUSTERS")
            if n_clusters is not None:
                cfg["kmeans"]["k_fixed"] = int(n_clusters)
                cfg["kmeans"]["k_method"] = "fixed"

    elif model_name == "mahalanobis":
        maha_overrides = ds_cfg.get("mahalanobis", {})
        if maha_overrides:
            reg_covar = maha_overrides.get("REG_COVAR")
            if reg_covar is not None:
                cfg["mahalanobis"]["reg_covar"] = float(reg_covar)

    elif model_name == "dbscan":
        dbscan_overrides = ds_cfg.get("dbscan", {})
        if dbscan_overrides:
            epsilon = dbscan_overrides.get("EPSILON")
            if epsilon is not None:
                cfg["dbscan"]["EPSILON"] = float(epsilon)
            min_samples = dbscan_overrides.get("MIN_SAMPLES")
            if min_samples is not None:
                cfg["dbscan"]["MIN_SAMPLES"] = int(min_samples)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", cfg.get("REPRODUCIBILITY", {}).get("SEED", 42))
    set_seed(seed)

    exp_id = args.exp_id
    exp_dir = Path("experiments") / exp_id
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(cfg, str(exp_dir))

    print(f"\n{'=' * 60}")
    print(f"  {args.model.upper()} Anomaly Detection — {exp_id}")
    print(f"  strategy={args.strategy}  dataset={args.dataset}  scenario={args.scenario}")
    print(f"  Sortie : {exp_dir}")
    print(f"{'=' * 60}\n")

    # --- Chargement des données ---
    failure_ratio = (
        cfg.get("DATASETS", {}).get("pronostia", {}).get("FAILURE_RATIO", 0.10)
        if args.dataset == "pronostia"
        else None
    )

    # Scénario effectif (--scenario overrides config pour CWRU, default "by_equipment" pour monitoring)
    if args.dataset == "cwru":
        effective_scenario = args.scenario or cfg.get("DATASETS", {}).get("cwru", {}).get("SPLIT_STRATEGY", "by_severity")
    elif args.dataset == "pronostia":
        effective_scenario = args.scenario or "by_condition"
    else:
        effective_scenario = args.scenario or "by_equipment"

    if args.dataset == "pronostia":
        tasks = _load_pronostia_tasks(cfg, seed, failure_ratio)
        _apply_pronostia_overrides(cfg, args.model, tasks)
        for t in tasks:
            print(
                f"  Task {t['task_id']} ({t['domain']}): "
                f"{t['n_train']} train (normal) | "
                f"{t['n_test']} test ({t['n_test_normal']} normal + {t['n_test_faulty']} faulty)"
            )
    elif args.dataset == "cwru":
        tasks = _load_cwru_tasks(cfg, seed, scenario=args.scenario)
        _apply_cwru_overrides(cfg, args.model, tasks)
        for t in tasks:
            print(
                f"  Task {t['task_id']} ({t['domain']}): "
                f"{t['n_train']} train (normal) | "
                f"{t['n_test']} test ({t['n_test_normal']} normal + {t['n_test_faulty']} faulty)"
            )
    else:
        csv_path = Path(cfg.get("data", {}).get(
            "csv_path",
            "data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset/equipment_anomaly_data.csv",
        ))
        normalizer_path = Path(
            "configs/monitoring_normalizer_anomaly.yaml"
        )
        batch_size = int(cfg.get("data", {}).get("batch_size", 32))

        tasks = get_cl_dataloaders_anomaly_detection(
            csv_path=csv_path,
            normalizer_path=normalizer_path,
            scenario=args.scenario or "by_equipment",
            batch_size=batch_size,
            seed=seed,
        )
        for t in tasks:
            print(
                f"  Task {t['task_id']} ({t['domain']}): "
                f"{t['n_train']} train (normal) | "
                f"{t['n_test']} test ({t['n_test_normal']} normal + {t['n_test_faulty']} faulty)"
            )

    builders = {
        "dbscan": _build_dbscan,
        "ewc_oneclass": _build_ewc_oneclass,
        "hdc": _build_hdc,
        "tinyol_ae": _build_tinyol_ae,
        "kmeans": _build_kmeans,
        "mahalanobis": _build_mahalanobis,
    }
    model = builders[args.model](cfg, args.strategy)

    summary_str = (
        model.summary()
        if hasattr(model, "summary")
        else f"{type(model).__name__}(n_params={model.count_parameters()})"
    )
    print(f"\n  Modèle : {summary_str}\n")

    auroc_matrix, _ = run_anomaly_detection_scenario(model, tasks, cfg)

    np.save(results_dir / "auroc_matrix.npy", auroc_matrix)
    print(f"\n  AUROC matrix sauvegardée → {results_dir / 'auroc_matrix.npy'}")

    cl_metrics = compute_cl_anomaly_metrics(auroc_matrix)
    print(
        f"\n  avg_auroc={cl_metrics['avg_auroc']:.4f}  "
        f"auroc_forgetting={cl_metrics['auroc_forgetting']:.4f}  "
        f"auroc_bwt={cl_metrics['auroc_bwt']:.4f}"
    )

    X_last = np.concatenate([
        b[0].numpy() for b in tasks[-1]["test_loader_mixed"]
    ])
    prof = _profile_model(model, X_last)
    print(
        f"  RAM peak: {prof['ram_peak_bytes'] / 1024:.1f} Ko  |  "
        f"Latence: {prof['inference_latency_ms']:.3f} ms  |  "
        f"n_params: {prof['n_params']}"
    )

    per_task_n_train = [t["n_train"] for t in tasks]
    n_train_report = (
        [sum(per_task_n_train[: i + 1]) for i in range(len(per_task_n_train))]
        if args.strategy == "accumulate"
        else per_task_n_train
    )

    extra = {
        "exp_id": exp_id,
        "model": args.model,
        "dataset": args.dataset,
        "scenario": effective_scenario,
        "strategy": args.strategy,
        "failure_ratio": failure_ratio,
        "n_train_normal_per_task": n_train_report,
        "auroc_per_task_final": [
            float(auroc_matrix[len(tasks) - 1, j]) for j in range(len(tasks))
        ],
        "ram_peak_bytes": prof["ram_peak_bytes"],
        "inference_latency_ms": prof["inference_latency_ms"],
        "n_params": prof["n_params"],
    }

    save_anomaly_metrics(
        cl_metrics,
        results_dir / "metrics_anomaly.json",
        extra_info=extra,
    )
    print(f"  Métriques → {results_dir / 'metrics_anomaly.json'}")
    print(f"\n✅ {args.model} {args.strategy} terminé → {exp_dir}")


if __name__ == "__main__":
    main()
