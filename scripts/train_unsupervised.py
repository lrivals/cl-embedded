"""
scripts/train_unsupervised.py — S5-05: Entraînement des baselines non supervisées.

Domain-incremental CL sur Dataset 2 (Equipment Monitoring) :
    Pump → Turbine → Compressor

Usage
-----
    python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model all
    python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model kmeans

Sorties
-------
    experiments/exp_005_unsupervised_dataset2/
    ├── config_snapshot.yaml
    └── results/
        ├── metrics_kmeans.json / metrics_knn.json / metrics_pca.json
        ├── metrics_all.json
        ├── acc_matrix_{name}.npy     — shape [3, 3], NaN pour tâches futures
        ├── auroc_matrix_{name}.npy
        └── model_{name}.pkl

Références
----------
    docs/sprints/sprint_5/S505_train_unsupervised.md
    src/models/unsupervised/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.monitoring_dataset import get_cl_dataloaders
from src.evaluation.metrics import compute_cl_metrics, format_metrics_report, save_metrics
from src.models.unsupervised import DBSCANDetector, KMeansDetector, KNNDetector, MahalanobisDetector, PCABaseline
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised CL baselines — Dataset 2 (Equipment Monitoring)"
    )
    parser.add_argument(
        "--config",
        default="configs/unsupervised_config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--model",
        choices=["kmeans", "knn", "pca", "mahalanobis", "dbscan", "all"],
        default="all",
        help="Modèle à entraîner (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="monitoring",
        choices=["monitoring", "pump"],
        help="Dataset : monitoring (Dataset 2) | pump (Dataset 1)",
    )
    parser.add_argument(
        "--exp_id",
        default=None,
        help="Override exp_id depuis la config (ex. exp_007_mahalanobis)",
    )
    parser.add_argument(
        "--data_config",
        default=None,
        help="Config data override (ex. configs/pump_by_id_config.yaml)",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Répertoire expérience override (ex. experiments/exp_015_mahalanobis_pump_by_id)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Factory de modèles
# ---------------------------------------------------------------------------


def build_model(
    model_name: str, config: dict
) -> KMeansDetector | KNNDetector | PCABaseline | MahalanobisDetector | DBSCANDetector:
    """
    Instancie un détecteur non supervisé depuis la sous-section YAML correspondante.

    Parameters
    ----------
    model_name : str
        "kmeans", "knn", "pca", "mahalanobis" ou "dbscan".
    config : dict
        Config complète (unsupervised_config.yaml). La sous-section model_name est extraite.

    Returns
    -------
    Détecteur instancié, non encore entraîné.
    """
    if model_name == "kmeans":
        return KMeansDetector(config["kmeans"])
    elif model_name == "knn":
        return KNNDetector(config["knn"])
    elif model_name == "pca":
        return PCABaseline(config["pca"])
    elif model_name == "mahalanobis":
        return MahalanobisDetector(config["mahalanobis"])
    elif model_name == "dbscan":
        return DBSCANDetector(config["dbscan"])
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r}. Choix : kmeans, knn, pca, mahalanobis, dbscan")


# ---------------------------------------------------------------------------
# 3. Loader Dataset 1 (Pump)
# ---------------------------------------------------------------------------


def load_pump_tasks(config: dict) -> list[dict]:
    """
    Charge Dataset 1 (Pump Maintenance) en scénario domain-incremental.

    Supporte deux modes selon ``data_pump["task_split"]`` :
    - ``"chronological"`` (défaut) : 3 tâches temporelles (pump_healthy → wearing → prefailure)
    - ``"by_pump_id"`` : 5 tâches par identifiant de pompe (Pump_ID 1 → 2 → 3 → 4 → 5)

    Délègue à pump_dataset.py et injecte la clé "domain" pour compatibilité
    avec train_unsupervised().

    Parameters
    ----------
    config : dict
        Config complète (section "data_pump" requise).

    Returns
    -------
    list[dict]
        Liste de tâches (3 ou 5) avec : train_loader, val_loader, domain, n_train, n_val.
    """
    data_cfg = config["data_pump"]
    task_split = data_cfg.get("task_split", "chronological")

    if task_split == "by_pump_id":
        from src.data.pump_dataset import get_pump_dataloaders_by_id
        tasks = get_pump_dataloaders_by_id(
            csv_path=str(data_cfg["csv_path"]),
            normalizer_path=str(data_cfg["normalizer_path"]),
            batch_size=data_cfg.get("batch_size", 32),
            val_ratio=data_cfg.get("val_ratio", 0.2),
            seed=config.get("seed", 42),
            window_size=data_cfg.get("window_size", 32),
            step_size=data_cfg.get("step_size", 16),
        )
        for task in tasks:
            task["domain"] = f"pump{task['pump_id']}"
    else:
        from src.data.pump_dataset import get_pump_dataloaders
        domain_names = ["pump_healthy", "pump_wearing", "pump_prefailure"]
        tasks = get_pump_dataloaders(
            csv_path=Path(data_cfg["csv_path"]),
            normalizer_path=Path(data_cfg["normalizer_path"]),
            batch_size=data_cfg.get("batch_size", 32),
            val_ratio=data_cfg.get("val_ratio", 0.2),
            seed=config.get("seed", 42),
            window_size=data_cfg.get("window_size", 32),
            step_size=data_cfg.get("step_size", 16),
        )
        for i, task in enumerate(tasks):
            task["domain"] = domain_names[i]

    return tasks


# ---------------------------------------------------------------------------
# 4. Utilitaire DataLoader → NumPy
# ---------------------------------------------------------------------------


def extract_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    """
    Convertit un DataLoader PyTorch en tableaux NumPy (X, y).

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader retournant des batches (X_batch, y_batch).

    Returns
    -------
    X : np.ndarray, shape [N, n_features]
    y : np.ndarray, shape [N]
    """
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy().ravel())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# 4. Boucle CL
# ---------------------------------------------------------------------------


def train_unsupervised(
    model: KMeansDetector | KNNDetector | PCABaseline,
    tasks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle domain-incremental : entraîne le modèle sur T tâches successives
    et évalue sur toutes les tâches vues.

    Labels utilisés UNIQUEMENT en évaluation (score, AUROC), jamais pendant fit_task.

    Parameters
    ----------
    model : détecteur non supervisé (KMeans / KNN / PCA).
    tasks : list[dict]
        Sortie de get_cl_dataloaders() — clés : task_id, domain, train_loader, val_loader.

    Returns
    -------
    acc_matrix : np.ndarray, shape [T, T]
        acc_matrix[i, j] = accuracy sur tâche j après entraînement sur tâches 0..i.
        NaN pour j > i (tâche future non encore vue).
    auroc_matrix : np.ndarray, shape [T, T]
        Même structure, avec AUROC à la place de l'accuracy.
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)
    auroc_matrix = np.full((T, T), np.nan)

    # Pré-extraction des données de validation (évite de répéter l'itération)
    val_data = [extract_numpy(task["val_loader"]) for task in tasks]

    for i, task in enumerate(tasks):
        domain = task.get("domain", f"Task {i + 1}")
        print(f"\n  → Tâche {i + 1}/{T} : {domain}")

        # Entraînement non supervisé (pas de labels)
        X_train, _ = extract_numpy(task["train_loader"])
        model.fit_task(X_train, task_id=i)

        # Évaluation sur toutes les tâches vues jusqu'à i
        for j in range(i + 1):
            X_val, y_val = val_data[j]
            acc_matrix[i, j] = model.score(X_val, y_val)

            scores = model.anomaly_score(X_val)
            try:
                auroc_matrix[i, j] = roc_auc_score(y_val, scores)
            except ValueError:
                # Un seul label présent dans y_val → AUROC indéfini
                auroc_matrix[i, j] = np.nan

        # Affichage ligne courante
        acc_row = [f"{acc_matrix[i, j]:.4f}" if not np.isnan(acc_matrix[i, j]) else "   NaN" for j in range(T)]
        print(f"    acc  [{', '.join(acc_row)}]")

    return acc_matrix, auroc_matrix


# ---------------------------------------------------------------------------
# 5. Profil mémoire (NumPy pur — tracemalloc)
# ---------------------------------------------------------------------------


def profile_model(
    model: KMeansDetector | KNNDetector | PCABaseline,
    X_sample: np.ndarray,
    n_runs: int = 100,
) -> dict:
    """
    Mesure la latence d'inférence et le pic RAM du modèle.

    Même approche que _profile_hdc_memory() dans train_hdc.py :
    utilise tracemalloc (pas torch.cuda) car les modèles sont NumPy purs.

    Parameters
    ----------
    model : détecteur entraîné.
    X_sample : np.ndarray
        Données de référence pour le profiling (1 échantillon suffisant).
    n_runs : int
        Nombre de runs pour estimer la latence (default: 100).

    Returns
    -------
    dict avec : ram_peak_bytes, inference_latency_ms, n_params,
                ram_fp32_bytes, ram_int8_bytes, within_budget_64ko.
    """
    x_single = X_sample[:1]

    # Latence — moyenne sur n_runs
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.anomaly_score(x_single)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # RAM peak via tracemalloc
    tracemalloc.start()
    model.anomaly_score(x_single)
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_params = int(model.count_parameters())
    ram_fp32 = n_params * 4   # 4 octets par float32
    ram_int8 = n_params * 1   # 1 octet par int8

    return {
        "inference_latency_ms": float(np.mean(latencies_ms)),
        "inference_latency_std_ms": float(np.std(latencies_ms)),
        "ram_peak_bytes": int(ram_peak),
        "n_params": n_params,
        "ram_fp32_bytes": ram_fp32,
        "ram_int8_bytes": ram_int8,
        "within_budget_64ko": ram_fp32 <= 64 * 1024,
    }


# ---------------------------------------------------------------------------
# 6. Orchestration par modèle
# ---------------------------------------------------------------------------


def run_model(
    model_name: str,
    config: dict,
    tasks: list[dict],
    results_dir: Path,
    dataset_tag: str = "dataset2",
) -> dict:
    """
    Entraîne, évalue et profile un modèle non supervisé.

    Parameters
    ----------
    model_name : str
        "kmeans", "knn", "pca" ou "mahalanobis".
    config : dict
        Config complète.
    tasks : list[dict]
        Tâches CL (sortie de get_cl_dataloaders).
    results_dir : Path
        Répertoire de sortie pour les artefacts.
    dataset_tag : str
        Suffixe dataset pour les noms de fichiers ("dataset2" ou "dataset1").

    Returns
    -------
    dict
        Métriques CL (aa, af, bwt, fwt), auroc_avg, profil mémoire.
    """
    T = len(tasks)
    print(f"\n{'=' * 60}")
    print(f"  Modèle : {model_name.upper()}")
    print(f"{'=' * 60}")

    model = build_model(model_name, config)

    # --- Entraînement CL ---
    acc_matrix, auroc_matrix = train_unsupervised(model, tasks)

    # --- Métriques CL ---
    cl_metrics = compute_cl_metrics(acc_matrix)

    # AUROC final = ligne finale de la matrice (après toutes les tâches)
    auroc_final_row = auroc_matrix[T - 1, :T]
    auroc_avg = float(np.nanmean(auroc_final_row))

    print(f"\n  AA={cl_metrics['aa']:.4f}  AF={cl_metrics['af']:.4f}  "
          f"BWT={cl_metrics['bwt']:.4f}  AUROC={auroc_avg:.4f}")

    # --- Profil mémoire ---
    X_sample, _ = extract_numpy(tasks[0]["val_loader"])
    n_latency_runs = config["evaluation"].get("n_latency_runs", 100)
    mem = profile_model(model, X_sample, n_runs=n_latency_runs)
    print(f"  RAM peak: {mem['ram_peak_bytes'] / 1024:.1f} Ko  |  "
          f"Latence: {mem['inference_latency_ms']:.3f} ms  |  "
          f"n_params: {mem['n_params']}")

    # --- Sauvegarde artefacts ---
    np.save(results_dir / f"acc_matrix_{model_name}_{dataset_tag}.npy", acc_matrix)
    np.save(results_dir / f"auroc_matrix_{model_name}_{dataset_tag}.npy", auroc_matrix)

    checkpoints_dir = results_dir.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoints_dir / f"{model_name}_{dataset_tag}_task{T - 1}_final.pkl")

    # Clés plates requises par S511 + clés détaillées existantes
    result = {
        "model": model_name,
        # Clés requises à la racine (S511 REQUIRED_KEYS)
        "acc_final": float(cl_metrics["aa"]),
        "avg_forgetting": float(cl_metrics["af"]),
        "backward_transfer": float(cl_metrics["bwt"]),
        "auroc_final": auroc_avg,
        "ram_peak_bytes": mem["ram_peak_bytes"],
        "inference_latency_ms": mem["inference_latency_ms"],
        "n_params": mem["n_params"],
        # Clés détaillées
        **cl_metrics,
        "auroc_avg": auroc_avg,
        "auroc_per_task": [
            None if math.isnan(v) else v for v in auroc_final_row.tolist()
        ],
        "memory": mem,
    }

    metrics_path = results_dir / f"metrics_{model_name}_{dataset_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Résultats → {metrics_path}")

    return result


# ---------------------------------------------------------------------------
# 7. Point d'entrée principal
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Chargement config
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Override section data depuis --data_config selon le dataset ciblé
    if args.data_config:
        data_cfg = load_config(args.data_config)
        if args.dataset == "monitoring":
            cfg["data"].update(data_cfg.get("data", {}))
        else:
            if "data_pump" not in cfg:
                cfg["data_pump"] = {}
            cfg["data_pump"].update(data_cfg.get("data", {}))

    # --- Override exp_id depuis CLI ---
    dataset_tag = "dataset1" if args.dataset == "pump" else "dataset2"
    if args.exp_id:
        if args.dataset == "pump":
            cfg["exp_id_pump"] = args.exp_id
        else:
            cfg["exp_id"] = args.exp_id

    # --- Sélection dataset et répertoires ---
    if args.dataset == "pump":
        exp_id = cfg.get("exp_id_pump", "exp_006_unsupervised_dataset1")
        results_dir = Path(f"experiments/{exp_id}/results")
        cfg["_dataset"] = "pump"
    else:
        exp_id = cfg.get("exp_id", "exp_005_unsupervised_dataset2")
        results_dir = Path(f"experiments/{exp_id}/results")
        cfg["_dataset"] = "monitoring"

    # Override répertoire expérience depuis --exp_dir
    if args.exp_dir:
        results_dir = Path(args.exp_dir) / "results"
        exp_id = Path(args.exp_dir).name

    exp_dir = results_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Baselines non supervisées — {exp_id}")
    print(f"  Output : {results_dir}")
    print(f"{'=' * 60}")

    save_config_snapshot(cfg, str(exp_dir))

    # --- Chargement des données ---
    if args.dataset == "pump":
        print(f"\nChargement Dataset 1 (Pump Maintenance) ...")
        tasks = load_pump_tasks(cfg)
    else:
        csv_path = Path(cfg["data"]["csv_path"])
        if not csv_path.exists():
            # Recherche récursive comme fallback (cohérent avec train_hdc.py)
            data_dir = csv_path.parent
            candidates = list(data_dir.rglob("*.csv")) if data_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(
                    f"CSV introuvable : {csv_path}\n"
                    "Télécharger le Dataset 2 (Equipment Monitoring) depuis Kaggle "
                    "et le placer dans data/raw/equipment_monitoring/."
                )
            csv_path = candidates[0]
            print(f"CSV trouvé (fallback) : {csv_path}")

        normalizer_path = Path(cfg["data"]["normalizer_path"])
        print(f"\nChargement des données depuis {csv_path} ...")
        task_split = cfg["data"].get("task_split", "by_equipment")
        if task_split == "by_location":
            from src.data.monitoring_dataset import get_cl_dataloaders_by_location
            tasks = get_cl_dataloaders_by_location(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=cfg["data"]["batch_size"],
                val_ratio=cfg["data"]["val_ratio"],
                seed=cfg.get("seed", 42),
                location_order=cfg["data"].get("location_order"),
            )
        else:
            tasks = get_cl_dataloaders(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=cfg["data"]["batch_size"],
                val_ratio=cfg["data"]["val_ratio"],
                seed=cfg.get("seed", 42),
            )

    print(f"Tâches chargées : {[t['domain'] for t in tasks]} ({len(tasks)} tâches)")

    # --- Sélection des modèles ---
    model_names = ["kmeans", "knn", "pca", "mahalanobis", "dbscan"] if args.model == "all" else [args.model]

    # --- Entraînement ---
    all_results: dict[str, dict] = {}
    for name in model_names:
        all_results[name] = run_model(name, cfg, tasks, results_dir, dataset_tag=dataset_tag)

    # --- Tableau comparatif ---
    print(f"\n{'=' * 72}")
    print(f"  RÉSULTATS COMPARATIFS — {exp_id}")
    print(f"{'=' * 72}")
    print(f"  {'Modèle':<12} {'AA':>8} {'AF':>8} {'BWT':>8} {'AUROC':>8} {'RAM':>10} {'Latence':>10}")
    print(f"  {'-' * 70}")
    for name, r in all_results.items():
        ram_ko = r["memory"]["ram_peak_bytes"] / 1024
        lat_ms = r["memory"]["inference_latency_ms"]
        print(
            f"  {name.upper():<12} "
            f"{r['aa']:>8.4f} {r['af']:>8.4f} {r['bwt']:>8.4f} "
            f"{r['auroc_avg']:>8.4f} {ram_ko:>8.1f} Ko {lat_ms:>8.3f} ms"
        )

    # --- Rapport mémoire agrégé ---
    memory_report = {name: r["memory"] for name, r in all_results.items()}
    memory_report_path = results_dir / "memory_report.json"
    with open(memory_report_path, "w", encoding="utf-8") as f:
        json.dump(memory_report, f, indent=2)
    print(f"Rapport mémoire → {memory_report_path}")

    # --- Sauvegarde globale ---
    all_metrics_path = results_dir / "metrics_all.json"
    with open(all_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRésultats complets → {all_metrics_path}")

    # --- Comparatif vs exp_005 (Dataset 2 uniquement) ---
    if dataset_tag == "dataset2":
        ref_path = Path("experiments/exp_005_unsupervised_dataset2/results/metrics_all.json")
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                ref_results = json.load(f)
            comparison = {**ref_results, **all_results}
            comparison_path = results_dir / "metrics_comparison.json"
            with open(comparison_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            print(f"Comparatif vs exp_005 → {comparison_path}")


if __name__ == "__main__":
    main()
