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

from src.data.monitoring_dataset import get_cl_dataloaders
from src.evaluation.metrics import compute_cl_metrics, format_metrics_report, save_metrics
from src.models.hdc import HDCClassifier
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed

_FEATURE_NAMES = ["temperature", "pressure", "vibration", "humidity"]


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
    return parser.parse_args()


def compute_feature_bounds_task1(
    train_loader,
) -> dict[str, tuple[float, float]]:
    """
    Calcule min/max de chaque feature sur le train set de Task 1 (Pump).

    À appeler UNE SEULE FOIS (Task 1) et à persister dans hdc_config.yaml.
    Appliqué à toutes les tâches suivantes sans recalcul (pas de data leakage).

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader de Task 1 (Pump) uniquement.

    Returns
    -------
    dict[str, tuple[float, float]]
        {"temperature": (min, max), "pressure": (min, max), ...}
    """
    all_x = []
    for x_batch, _ in train_loader:
        all_x.append(x_batch.numpy())
    x_all = np.concatenate(all_x, axis=0)  # [N_task1, n_features]  # noqa: N806

    bounds: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(_FEATURE_NAMES):
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

    print("[HDC] Calcul des feature_bounds sur Task 1 (Pump)...")
    bounds = compute_feature_bounds_task1(task1_loader)
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
        domain = task["domain"]
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
            print(f"  Acc tâche {eval_idx + 1} ({tasks[eval_idx]['domain']}): {acc:.4f}")

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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

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
    data_path = Path(cfg["data"]["path"])
    normalizer_path = Path(cfg["data"]["normalizer_path"])

    csv_candidates = list(data_path.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans '{data_path}'.\n"
            "Télécharger le Dataset 2 (Equipment Monitoring) depuis Kaggle "
            "et le placer dans ce dossier."
        )
    csv_path = csv_candidates[0]
    print(f"Dataset : {csv_path}")

    tasks = get_cl_dataloaders(
        csv_path=csv_path,
        normalizer_path=normalizer_path,
        batch_size=cfg["training"]["batch_size"],
        val_ratio=cfg["data"]["test_split"],
        seed=cfg["training"]["seed"],
    )
    n_tasks = len(tasks)
    print(f"Tâches chargées : {[t['domain'] for t in tasks]} ({n_tasks} tâches)\n")

    # --- Feature bounds Task 1 (calculées si absentes) ---
    _ensure_feature_bounds(cfg, tasks[0]["train_loader"], args.config)

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
