"""
scripts/train_hdc.py — Entraînement HDC Online sur Dataset 2 (Equipment Monitoring).

Usage
-----
    python scripts/train_hdc.py --config configs/hdc_config.yaml
    python scripts/train_hdc.py --config configs/hdc_config.yaml --skip-baselines

Sorties
-------
    experiments/exp_002_hdc_dataset2/
    ├── config_snapshot.yaml
    ├── checkpoints/
    │   └── hdc_task3_final.npz
    └── results/
        ├── metrics.json
        ├── acc_matrix_hdc.npy
        └── memory_report.json

Références
----------
    Benatti2019HDC, docs/models/hdc_spec.md, docs/sprints/sprint_2/S203_exp002.md
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
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.metrics import compute_cl_metrics, format_metrics_report, save_metrics
from src.models.hdc import HDCClassifier
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed

def _get_feature_names(cfg: dict) -> list[str]:
    """Retourne la liste ordonnée des noms de features selon le dataset."""
    dataset = cfg["data"].get("dataset", "equipment_monitoring")
    if dataset == "pump_maintenance":
        cols = cfg["data"]["feature_columns"]
        stats = cfg["data"]["features_per_channel"]
        return [f"{s}_{c}" for c in cols for s in stats] + ["temporal_position"]
    if dataset == "pronostia":
        from src.data.pronostia_dataset import FEATURE_NAMES
        return list(FEATURE_NAMES)
    if dataset == "cwru":
        from src.data.cwru_dataset import FEATURE_COLS
        return list(FEATURE_COLS)
    return cfg["data"]["feature_columns"]


def _get_tasks(cfg: dict) -> list[dict]:
    """Charge les tâches CL selon le dataset spécifié dans config."""
    dataset = cfg["data"].get("dataset", "equipment_monitoring")

    if dataset == "pronostia":
        from src.data.pronostia_dataset import (
            get_pronostia_dataloaders,
            get_pronostia_dataloaders_single_task,
        )
        npy_dir = Path(cfg["data"]["npy_dir"])
        batch_size = cfg["training"]["batch_size"]
        seed = cfg["training"]["seed"]
        task_split = cfg["data"].get("task_split", "by_condition")
        if task_split == "no_split":
            st = get_pronostia_dataloaders_single_task(
                npy_dir=npy_dir,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
                window_size=cfg["data"].get("window_size", 2560),
                step_size=cfg["data"].get("step_size", 2560),
                failure_ratio=cfg["data"].get("failure_ratio", 0.10),
            )
            st["_single_task_mode"] = True
            return [st]
        return get_pronostia_dataloaders(
            npy_dir=npy_dir,
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=batch_size,
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=seed,
            window_size=cfg["data"].get("window_size", 2560),
            step_size=cfg["data"].get("step_size", 2560),
            failure_ratio=cfg["data"].get("failure_ratio", 0.10),
        )

    if dataset == "cwru":
        task_split = cfg["data"].get("task_split", "no_split")
        if task_split == "by_fault_type":
            from src.data.cwru_dataset import get_cwru_cl_dataloaders_by_fault_type
            # Forcer recalcul des feature_bounds depuis Task 1
            cfg["feature_bounds"] = None
            return get_cwru_cl_dataloaders_by_fault_type(
                csv_path=Path(cfg["data"]["csv_path"]),
                batch_size=cfg["data"].get("batch_size", 1),
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=cfg["training"]["seed"],
            )
        elif task_split == "by_severity":
            from src.data.cwru_dataset import get_cwru_cl_dataloaders_by_severity
            cfg["feature_bounds"] = None
            return get_cwru_cl_dataloaders_by_severity(
                csv_path=Path(cfg["data"]["csv_path"]),
                batch_size=cfg["data"].get("batch_size", 1),
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=cfg["training"]["seed"],
            )
        from src.data.cwru_dataset import get_cwru_dataloaders_single_task
        st = get_cwru_dataloaders_single_task(
            csv_path=Path(cfg["data"]["csv_path"]),
            batch_size=cfg["data"].get("batch_size", 1),
            test_ratio=cfg["data"].get("test_ratio", 0.2),
            val_ratio=cfg["data"].get("val_ratio", 0.1),
            seed=cfg["training"]["seed"],
        )
        st["_single_task_mode"] = True
        return [st]

    csv_path = Path(cfg["data"]["csv_path"])
    normalizer_path = Path(cfg["data"]["normalizer_path"])
    batch_size = cfg["training"]["batch_size"]
    val_ratio = cfg["data"]["test_split"]
    seed = cfg["training"]["seed"]

    if dataset == "pump_maintenance":
        task_split = cfg["data"].get("task_split", "chronological")
        if task_split == "by_pump_id":
            from src.data.pump_dataset import get_pump_dataloaders_by_id
            return get_pump_dataloaders_by_id(
                csv_path=str(csv_path),
                normalizer_path=str(normalizer_path),
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
        elif task_split == "by_temporal_window":
            from src.data.pump_dataset import get_pump_dataloaders_by_temporal_window
            return get_pump_dataloaders_by_temporal_window(
                csv_path=str(csv_path),
                normalizer_path=str(normalizer_path),
                n_tasks=cfg["data"].get("n_tasks", 4),
                entries_per_task=cfg["data"].get("entries_per_task", 5000),
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
        elif task_split == "no_split":
            from src.data.pump_dataset import get_pump_dataloaders_single_task
            st = get_pump_dataloaders_single_task(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
            st["_single_task_mode"] = True
            return [st]
        from src.data.pump_dataset import get_pump_dataloaders
        return get_pump_dataloaders(
            csv_path=csv_path,
            normalizer_path=normalizer_path,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
            window_size=cfg["data"]["window_size"],
            step_size=cfg["data"]["step_size"],
        )
    else:
        task_split = cfg["data"].get("task_split", "by_equipment")
        if task_split == "no_split":
            from src.data.monitoring_dataset import get_monitoring_dataloaders_single_task
            st = get_monitoring_dataloaders_single_task(
                csv_path=csv_path,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
            )
            st["_single_task_mode"] = True
            return [st]
        if task_split == "by_location":
            from src.data.monitoring_dataset import get_cl_dataloaders_by_location
            return get_cl_dataloaders_by_location(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                location_order=cfg["data"].get("location_order"),
            )
        from src.data.monitoring_dataset import get_cl_dataloaders
        return get_cl_dataloaders(
            csv_path=csv_path,
            normalizer_path=normalizer_path,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HDC Online Training — Dataset 2")
    parser.add_argument("--config", default="configs/hdc_config.yaml")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ignoré pour HDC (NumPy pur) — présent pour compatibilité CLI",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Sauter les baselines (non applicable pour HDC — option conservée pour cohérence CLI)",
    )
    parser.add_argument(
        "--data_config",
        default=None,
        help="Config data override (ex. configs/pump_by_id_config.yaml)",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Répertoire expérience override (ex. experiments/exp_014_hdc_pump_by_id)",
    )
    parser.add_argument(
        "--exp_id",
        default=None,
        help="Override exp_id (ex. exp_069_hdc_cwru_single_task)",
    )
    return parser.parse_args()


def compute_feature_bounds_task1(
    train_loader,
    feature_names: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Calcule min/max de chaque feature sur le train set de Task 1.

    À appeler UNE SEULE FOIS (Task 1) et à persister dans le config YAML.
    Appliqué à toutes les tâches suivantes sans recalcul (pas de data leakage).

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader de Task 1 uniquement.
    feature_names : list[str]
        Noms des features dans l'ordre des colonnes du tenseur X.

    Returns
    -------
    dict[str, tuple[float, float]]
        {nom_feature: (min, max)} dans l'ordre de feature_names.
    """
    all_x = []
    for x_batch, _ in train_loader:
        all_x.append(x_batch.numpy())
    x_all = np.concatenate(all_x, axis=0)  # [N_task1, n_features]  # noqa: N806

    bounds: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(feature_names):
        bounds[name] = (float(x_all[:, i].min()), float(x_all[:, i].max()))
    return bounds


def _ensure_feature_bounds(config: dict, task1_loader, config_path: str) -> None:
    """
    S'assure que config["feature_bounds"] (racine) est calculé et persisté.

    HDCClassifier lit les bornes à la racine du dict config. Cette fonction
    les calcule depuis Task 1 si elles sont absentes ou contiennent des None,
    puis met à jour le fichier YAML pour les runs suivants.

    Parameters
    ----------
    config : dict
        Config complète (modifiée in-place).
    task1_loader : DataLoader
        DataLoader de Task 1 — source des bornes.
    config_path : str
        Chemin du fichier YAML à réécrire avec les bornes calculées.
    """
    bounds_cfg = config.get("feature_bounds", {})
    needs_update = not bounds_cfg or any(
        v is None or (isinstance(v, (list, tuple)) and None in v) for v in bounds_cfg.values()
    )
    if not needs_update:
        return

    feature_names = _get_feature_names(config)
    print(f"[HDC] Calcul des feature_bounds sur Task 1 ({len(feature_names)} features)...")
    bounds = compute_feature_bounds_task1(task1_loader, feature_names)
    config["feature_bounds"] = {k: list(v) for k, v in bounds.items()}

    # Persister dans le YAML pour les runs suivants
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"  feature_bounds écrits dans {config_path}")
    for name, (lo, hi) in bounds.items():
        print(f"    {name}: [{lo:.4f}, {hi:.4f}]")


def evaluate_task_hdc(model: HDCClassifier, val_loader) -> float:
    """
    Évalue l'accuracy binaire sur un val_loader.

    Parameters
    ----------
    model : HDCClassifier
    val_loader : DataLoader (PyTorch)

    Returns
    -------
    float : accuracy ∈ [0, 1]
    """
    correct, total = 0, 0
    for x_batch, y_batch in val_loader:
        preds = model.predict(x_batch.numpy())
        correct += int((preds == y_batch.numpy().astype(np.int64).ravel()).sum())
        total += len(y_batch)
    return correct / total if total > 0 else 0.0


def train_hdc(
    model: HDCClassifier,
    tasks: list[dict],
    config: dict,
) -> np.ndarray:
    """
    Entraînement HDC Online sur N tâches séquentielles.

    Pas d'optimizer, pas de gradient : accumulation additive de prototypes.
    Pas d'oubli catastrophique par construction (mémoire additive INT32).

    Parameters
    ----------
    model : HDCClassifier
    tasks : list[dict]
        Sortie de get_cl_dataloaders() — chaque dict contient train_loader + val_loader.
    config : dict
        Configuration hdc_config.yaml.

    Returns
    -------
    acc_matrix : np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur tâche j après entraînement sur tâche i.
        NaN si j > i (tâche future non encore vue).
    """
    n_tasks = len(tasks)
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)

    for task_idx, task in enumerate(tasks):
        domain = task.get("domain", f"Tâche {task['task_id']}")
        print(f"\n{'=' * 60}")
        print(f"Tâche {task_idx + 1}/{n_tasks} : {domain}")
        print(f"  Train: {task['n_train']} samples | Val: {task['n_val']} samples")

        # Entraînement : parcours unique du train_loader (online, 1 epoch)
        batch_errors = []
        for x_batch, y_batch in task["train_loader"]:
            err_rate = model.update(x_batch.numpy(), y_batch.numpy().ravel())
            batch_errors.append(err_rate)
        print(f"  Taux d'erreur moyen (train) : {np.mean(batch_errors):.4f}")

        # Callback fin de tâche (re-binarisation des prototypes)
        model.on_task_end(task_id=task_idx, dataloader=task["train_loader"])

        # Évaluation sur toutes les tâches vues (backward transfer)
        for eval_idx in range(task_idx + 1):
            acc = evaluate_task_hdc(model, tasks[eval_idx]["val_loader"])
            acc_matrix[task_idx, eval_idx] = acc
            lbl = tasks[eval_idx].get("domain", f"T{eval_idx + 1}")
            print(f"  Acc tâche {eval_idx + 1} ({lbl}): {acc:.4f}")

    return acc_matrix


def _profile_hdc_memory(model: HDCClassifier, n_features: int, n_runs: int = 100) -> dict:
    """
    Profile la mémoire et la latence du modèle HDC (inférence uniquement).

    HDCClassifier étant NumPy-based (pas de nn.Module), on utilise tracemalloc
    et time.perf_counter directement plutôt que profile_forward_pass().

    Parameters
    ----------
    model : HDCClassifier
    n_features : int
        Dimension des features d'entrée.
    n_runs : int
        Nombre de runs pour la mesure de latence.

    Returns
    -------
    dict
        Rapport mémoire compatible avec le format JSON de exp_001.
    """
    dummy = np.zeros((1, n_features), dtype=np.float32)

    # Mesure mémoire (tracemalloc)
    tracemalloc.start()
    _ = model.predict(dummy)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Mesure latence (100 runs)
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    budget = 65_536
    ram_fp32 = model.estimate_ram_bytes("fp32")
    ram_int8 = model.estimate_ram_bytes("int8")

    report = {
        "model": "HDCClassifier",
        "input_shape": [1, n_features],
        "forward": {
            "ram_peak_bytes": peak_mem,
            "ram_current_bytes": current_mem,
            "inference_latency_ms": float(np.mean(latencies_ms)),
            "inference_latency_std_ms": float(np.std(latencies_ms)),
            "n_params": model.count_parameters(),
            "params_fp32_bytes": ram_fp32,
            "params_int8_bytes": ram_int8,
            "within_budget_64ko": peak_mem < budget,
        },
        "estimated_ram_fp32_bytes": ram_fp32,
        "estimated_ram_int8_bytes": ram_int8,
        "within_budget": ram_fp32 < budget,
    }

    # Affichage
    pct = peak_mem / budget * 100
    print("\n   ┌─ Résultats ─────────────────────────────────────┐")
    print(f"   │  Paramètres    : {model.count_parameters():>10,} éléments")
    print(f"   │  RAM FP32 est. : {ram_fp32:>10,} B ({ram_fp32 / 1024:.1f} Ko)")
    print(f"   │  RAM INT8 est. : {ram_int8:>10,} B ({ram_int8 / 1024:.1f} Ko)")
    print(f"   │  RAM peak fwd  : {peak_mem:>10,} B ({peak_mem / 1024:.1f} Ko) — {pct:.1f}% budget")
    print(f"   │  Latence fwd   : {np.mean(latencies_ms):>10.3f} ms (± {np.std(latencies_ms):.3f})")
    status = "✅ DANS LE BUDGET" if peak_mem < budget else "❌ DÉPASSE LE BUDGET"
    print(f"   │  STM32N6 64Ko  : {status}")
    print("   └────────────────────────────────────────────────┘")

    return report


def _run_single_task_hdc(
    data: dict,
    cfg: dict,
    results_dir: Path,
    exp_dir: Path,
) -> None:
    """
    Entraîne et évalue HDC sur une seule tâche (tout le dataset monitoring fusionné).

    Sortie : ``results/metrics_single_task.json`` avec les 6 métriques obligatoires.

    Parameters
    ----------
    data : dict
        Sortie de get_monitoring_dataloaders_single_task().
    cfg : dict
        Config complète.
    results_dir : Path
    exp_dir : Path
    """
    print(f"\n{'=' * 40}")
    print(f"  HDC Single-Task — {cfg['exp_id']}")
    print(f"  {data['n_train']} train | {data['n_val']} val | {data['n_test']} test")
    print(f"{'=' * 40}")

    # Extraire numpy pour les bornes et le fit
    all_x_train = []
    for x_batch, _ in data["train_loader"]:
        all_x_train.append(x_batch.numpy())
    X_train = np.concatenate(all_x_train, axis=0)

    # Calculer feature_bounds depuis X_train (pas de Task 1 en mode single-task)
    # Noms dérivés dynamiquement selon le dataset (pump : 25 features, monitoring : 4 features)
    dataset_name = cfg.get("data", {}).get("dataset", "equipment_monitoring")
    if dataset_name == "pump_maintenance":
        from src.data.pump_dataset import FEATURE_NAMES as _PUMP_FEATURE_NAMES
        feature_names = _PUMP_FEATURE_NAMES
    elif dataset_name == "cwru":
        from src.data.cwru_dataset import FEATURE_COLS
        feature_names = list(FEATURE_COLS)
    else:
        feature_names = ["temperature", "pressure", "vibration", "humidity"]
    bounds: dict = {}
    for i, name in enumerate(feature_names):
        bounds[name] = [float(X_train[:, i].min()), float(X_train[:, i].max())]

    cfg_single = dict(cfg)
    cfg_single["feature_bounds"] = bounds

    model = HDCClassifier(cfg_single)
    n_features: int = cfg["data"].get("n_features", 4)
    print(f"\nHDCClassifier : D={model.D}, n_features={n_features}")

    # Entraînement (online, 1 pass)
    batch_errors = []
    for x_batch, y_batch in data["train_loader"]:
        err_rate = model.update(x_batch.numpy(), y_batch.numpy().ravel())
        batch_errors.append(err_rate)
    model.on_task_end(task_id=0, dataloader=data["train_loader"])
    print(f"  Taux d'erreur moyen (train) : {np.mean(batch_errors):.4f}")

    # Évaluation sur test
    test_preds, test_true = [], []
    for x_batch, y_batch in data["test_loader"]:
        preds = model.predict(x_batch.numpy())
        test_preds.extend(preds.tolist())
        test_true.extend(y_batch.numpy().ravel().astype(int).tolist())

    y_true = np.array(test_true)
    y_pred = np.array(test_preds)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    # HDC : pas de score de probabilité natif — AUC-ROC sur prédictions binaires
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auc = float("nan")

    print(f"\n  Test → accuracy={acc:.4f} | f1={f1:.4f} | auc_roc={auc:.4f}")

    # Profiling mémoire
    print("\n  Profiling mémoire...")
    memory_report = _profile_hdc_memory(model, n_features=n_features)
    fwd = memory_report["forward"]

    metrics: dict = {
        "exp_id": cfg["exp_id"],
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "ram_peak_bytes": fwd["ram_peak_bytes"],
        "inference_latency_ms": fwd["inference_latency_ms"],
        "n_params": model.count_parameters(),
    }

    metrics_path = results_dir / "metrics_single_task.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")
    print(f"✅ HDC single-task terminé → {exp_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Override section data depuis --data_config (ex. pump_by_id_config.yaml)
    if args.data_config:
        data_cfg = load_config(args.data_config)
        cfg["data"].update(data_cfg.get("data", {}))
        # Forcer recalcul des feature_bounds si scénario avec nouvelles données
        if cfg["data"].get("task_split") in ("by_pump_id", "by_temporal_window"):
            cfg["feature_bounds"] = None

    # Override exp_id depuis --exp_id
    if args.exp_id:
        cfg["exp_id"] = args.exp_id
        cfg["evaluation"]["output_dir"] = f"experiments/{args.exp_id}/results/"

    # Override répertoire expérience depuis --exp_dir
    if args.exp_dir:
        cfg["evaluation"]["output_dir"] = str(Path(args.exp_dir) / "results")
        cfg["exp_id"] = Path(args.exp_dir).name

    set_seed(cfg["training"]["seed"])

    exp_dir = get_exp_dir(cfg)
    results_dir = exp_dir / "results"
    checkpoints_dir = exp_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  HDC Online — {cfg['exp_id']}")
    print(
        f"  D={cfg['hdc']['D']} | n_levels={cfg['hdc']['n_levels']} | seed={cfg['training']['seed']}"
    )
    print(f"  Sortie : {exp_dir}")
    print(f"{'=' * 60}\n")

    save_config_snapshot(cfg, str(exp_dir))
    print("Config snapshot sauvegardé.")

    # --- Chargement données ---
    dataset_name = cfg.get("data", {}).get("dataset", "equipment_monitoring")
    if dataset_name != "pronostia":
        csv_path = Path(cfg["data"]["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")
        print(f"Dataset : {csv_path}")
    else:
        print(f"Dataset : {cfg['data']['npy_dir']} (FEMTO PRONOSTIA .npy)")

    tasks = _get_tasks(cfg)
    n_tasks = len(tasks)

    # Mode single-task (task_split: no_split) — baseline hors-CL
    if tasks[0].get("_single_task_mode"):
        _run_single_task_hdc(
            data=tasks[0],
            cfg=cfg,
            results_dir=results_dir,
            exp_dir=exp_dir,
        )
        return

    labels = [t.get("domain", f"T{t['task_id']}") for t in tasks]
    print(f"Tâches chargées : {labels} ({n_tasks} tâches)\n")

    # --- Feature bounds Task 1 (calculées si absentes) ---
    # Si --data_config fourni, écrire les bounds dans ce fichier (pas le config principal)
    bounds_config_path = args.data_config if args.data_config else args.config
    _ensure_feature_bounds(cfg, tasks[0]["train_loader"], bounds_config_path)

    # --- Modèle HDC ---
    model = HDCClassifier(cfg)
    n_features: int = cfg["data"]["n_features"]
    print(
        f"\nHDCClassifier initialisé : D={model.D}, n_features={n_features}, n_classes={model.n_classes}"
    )

    # --- Entraînement HDC ---
    print("\n" + "=" * 40)
    print("  Entraînement HDC Online")
    print("=" * 40)
    acc_matrix_hdc = train_hdc(model, tasks, cfg)
    np.save(results_dir / "acc_matrix_hdc.npy", acc_matrix_hdc)
    model.save(str(checkpoints_dir / "hdc_task3_final.npz"))
    print(f"\nCheckpoint sauvegardé → {checkpoints_dir / 'hdc_task3_final.npz'}")

    # --- Profiling mémoire ---
    print("\n" + "=" * 40)
    print("  Profiling mémoire HDC")
    print("=" * 40)
    memory_report = _profile_hdc_memory(model, n_features=n_features)
    with open(results_dir / "memory_report.json", "w") as f:
        json.dump(memory_report, f, indent=2)

    # --- Métriques CL ---
    metrics_hdc = compute_cl_metrics(acc_matrix_hdc)
    fwd = memory_report["forward"]
    full_metrics = {
        **metrics_hdc,
        "ram_peak_bytes": fwd["ram_peak_bytes"],
        "inference_latency_ms": fwd["inference_latency_ms"],
        "n_params": model.count_parameters(),
        "estimated_ram_fp32_bytes": memory_report["estimated_ram_fp32_bytes"],
        "estimated_ram_int8_bytes": memory_report["estimated_ram_int8_bytes"],
    }
    save_metrics(
        full_metrics, str(results_dir / "metrics.json"), extra_info={"exp_id": cfg["exp_id"]}
    )

    # metrics_cl.json — format unifié S12-05
    import json as _json
    metrics_cl = {
        "exp_id": cfg["exp_id"],
        "model": "hdc",
        "dataset": cfg["data"].get("dataset", "cwru"),
        "scenario": cfg["data"].get("task_split", "by_fault_type"),
        "acc_final": metrics_hdc.get("aa", float("nan")),
        "avg_forgetting": metrics_hdc.get("af", float("nan")),
        "backward_transfer": metrics_hdc.get("bwt", float("nan")),
        "per_task_acc": [float(acc_matrix_hdc[n_tasks - 1, j]) for j in range(n_tasks)],
        "ram_peak_bytes": fwd["ram_peak_bytes"],
        "inference_latency_ms": fwd["inference_latency_ms"],
        "n_params": model.count_parameters(),
        "acc_matrix": acc_matrix_hdc.tolist(),
    }
    with open(results_dir / "metrics_cl.json", "w", encoding="utf-8") as _f:
        _json.dump(metrics_cl, _f, indent=2)

    # --- Rapport stdout ---
    print("\n" + "=" * 60)
    report = format_metrics_report(metrics_hdc, model_name="HDCClassifier")
    print(report)

    ram_fp32 = memory_report["estimated_ram_fp32_bytes"]
    ram_int8 = memory_report["estimated_ram_int8_bytes"]
    within = memory_report["within_budget"]
    print(f"\nRAM estimée (FP32) : {ram_fp32 / 1024:.1f} Ko")
    print(f"RAM estimée (INT8) : {ram_int8 / 1024:.1f} Ko")
    print(f"Budget 64 Ko : {'✅ OK' if within else '❌ DÉPASSÉ'}")
    print(f"\n✅ Expérience terminée → {exp_dir}")


if __name__ == "__main__":
    main()
