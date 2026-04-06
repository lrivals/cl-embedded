# S5-05 — Script d'entraînement + évaluation CL non supervisé

| Champ | Valeur |
|-------|--------|
| **ID** | S5-05 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S5-02 (`KMeansDetector`), S5-03 (`KNNDetector`), S5-04 (`PCABaseline`), S1-03 (`get_cl_dataloaders`), S1-07 (`metrics.py`), S1-08 (`memory_profiler.py`) |
| **Fichiers cibles** | `scripts/train_unsupervised.py` |
| **Complété le** | — |

---

## Objectif

Créer le script `scripts/train_unsupervised.py` qui exécute la boucle d'entraînement CL domain-incremental pour les 3 modèles non supervisés (K-Means, KNN, PCA) sur Dataset 2. Le script suit le même patron que `scripts/train_ewc.py` et `scripts/train_hdc.py`.

**Points clés** :
- Argument `--model [kmeans|knn|pca|all]` pour entraîner un modèle ou tous
- Boucle CL : 3 tâches séquentielles (Pump → Turbine → Compressor)
- Métriques AA, AF, BWT (via `evaluation/metrics.py`) + AUROC (sklearn)
- Profiling mémoire via `evaluation/memory_profiler.py`
- Sortie dans `experiments/exp_005_unsupervised_dataset2/`

**Critère de succès** : `python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model all` s'exécute sans erreur et produit `experiments/exp_005_unsupervised_dataset2/results/metrics_all.json` avec AA, AF, BWT, AUROC pour les 3 modèles.

---

## Patron de référence

Ce script suit la même structure que `scripts/train_hdc.py` (S2-03) :
```
parse_args → load_config → set_seed → make_exp_dir → snapshot_config →
load_data → init_model → train_loop → profile_memory → compute_metrics → save_results
```

---

## Sous-tâches

### 1. Structure du script et `parse_args`

```python
#!/usr/bin/env python3
"""
train_unsupervised.py — Entraînement baselines non supervisées sur Dataset 2.

Usage
-----
    python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml
    python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model kmeans
    python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model all

Sorties
-------
    experiments/exp_005_unsupervised_dataset2/
    ├── config_snapshot.yaml
    ├── checkpoints/
    │   ├── kmeans_task2_final.pkl
    │   ├── knn_task2_final.pkl
    │   └── pca_task2_final.pkl
    └── results/
        ├── metrics_kmeans.json
        ├── metrics_knn.json
        ├── metrics_pca.json
        ├── metrics_all.json          # tableau comparatif 3 modèles
        ├── acc_matrix_kmeans.npy     # [T, T]
        ├── acc_matrix_knn.npy
        ├── acc_matrix_pca.npy
        └── memory_report.json

Références
----------
    docs/sprints/sprint_5/S502_kmeans_detector.md
    docs/sprints/sprint_5/S503_knn_detector.md
    docs/sprints/sprint_5/S504_pca_baseline.md
    scripts/train_hdc.py (patron de référence)
"""

import argparse
import json
import tracemalloc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src.data.monitoring_dataset import get_cl_dataloaders
from src.evaluation.metrics import compute_cl_metrics, save_metrics, format_metrics_report
from src.models.unsupervised import KMeansDetector, KNNDetector, PCABaseline
from src.utils.reproducibility import set_seed

SUPPORTED_MODELS = ["kmeans", "knn", "pca"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unsupervised CL baselines on Dataset 2 (Equipment Monitoring)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unsupervised_config.yaml",
        help="Chemin vers unsupervised_config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=SUPPORTED_MODELS + ["all"],
        help="Modèle à entraîner : kmeans | knn | pca | all",
    )
    return parser.parse_args()
```

### 2. Factory des modèles

```python
def build_model(model_name: str, config: dict) -> KMeansDetector | KNNDetector | PCABaseline:
    """
    Instancie le modèle non supervisé demandé à partir de la config YAML.

    Parameters
    ----------
    model_name : str
        "kmeans" | "knn" | "pca"
    config : dict
        Config complète (unsupervised_config.yaml).

    Returns
    -------
    KMeansDetector | KNNDetector | PCABaseline
    """
    if model_name == "kmeans":
        return KMeansDetector(config["kmeans"])
    elif model_name == "knn":
        return KNNDetector(config["knn"])
    elif model_name == "pca":
        return PCABaseline(config["pca"])
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r}. Valeurs valides : {SUPPORTED_MODELS}")
```

### 3. Boucle d'entraînement CL (`train_unsupervised`)

```python
def train_unsupervised(
    model: KMeansDetector | KNNDetector | PCABaseline,
    tasks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boucle d'entraînement domain-incremental sur N tâches séquentielles.

    Labels exclus de l'entraînement — utilisés uniquement pour l'évaluation.

    Parameters
    ----------
    model
        Modèle non supervisé instancié.
    tasks : list[dict]
        Sortie de get_cl_dataloaders() — chaque dict contient train_loader + val_loader.

    Returns
    -------
    acc_matrix : np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur tâche j après entraînement sur tâche i.
        NaN si j > i.
    auroc_matrix : np.ndarray [T, T]
        Même structure avec AUROC à la place de l'accuracy.
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)
    auroc_matrix = np.full((T, T), np.nan)

    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Tâche {task_idx + 1}/{T} : {task['domain']}")
        print(f"  Train : {task['n_train']} samples | Val : {task['n_val']} samples")

        # Collecte des données d'entraînement (sans labels)
        X_train_list = []
        for x_batch, _ in task["train_loader"]:
            X_train_list.append(x_batch.numpy())
        X_train = np.concatenate(X_train_list, axis=0)  # [N_train, n_features]

        # Entraînement non supervisé (labels exclus)
        model.fit_task(X_train, task_id=task_idx)
        print(f"  {model.summary()}")

        # Évaluation sur toutes les tâches vues (backward transfer)
        for eval_idx in range(task_idx + 1):
            X_val_list, y_val_list = [], []
            for x_batch, y_batch in tasks[eval_idx]["val_loader"]:
                X_val_list.append(x_batch.numpy())
                y_val_list.append(y_batch.numpy())
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)

            # Accuracy (labels utilisés ici pour évaluation)
            acc = model.score(X_val, y_val)
            acc_matrix[task_idx, eval_idx] = acc

            # AUROC (labels utilisés pour évaluation)
            scores = model.anomaly_score(X_val)
            try:
                auroc = float(roc_auc_score(y_val, scores))
            except ValueError:
                # roc_auc_score lève ValueError si une seule classe présente
                auroc = float("nan")
            auroc_matrix[task_idx, eval_idx] = auroc

            domain = tasks[eval_idx]["domain"]
            print(
                f"  Acc tâche {eval_idx + 1} ({domain}) : {acc:.4f} | "
                f"AUROC : {auroc:.4f}"
            )

    return acc_matrix, auroc_matrix
```

### 4. Profiling mémoire

```python
def profile_model(
    model: KMeansDetector | KNNDetector | PCABaseline,
    X_sample: np.ndarray,
    n_runs: int = 100,
) -> dict:
    """
    Profil mémoire et latence du modèle non supervisé.

    Utilise tracemalloc (comme memory_profiler.py) pour mesurer la RAM peak
    pendant l'inférence (predict sur 1 échantillon).

    Parameters
    ----------
    model
        Modèle entraîné.
    X_sample : np.ndarray [N, n_features]
        Données de référence pour le profilage.
    n_runs : int
        Nombre de runs pour la mesure de latence.

    Returns
    -------
    dict
        ram_peak_bytes, inference_latency_ms, n_params.
    """
    x_single = X_sample[:1]  # 1 seul échantillon pour l'inférence

    # RAM peak (tracemalloc)
    tracemalloc.start()
    model.predict(x_single)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Latence (moyenne sur n_runs)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(x_single)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "ram_peak_bytes": peak,
        "inference_latency_ms": float(np.mean(latencies)),
        "inference_latency_std_ms": float(np.std(latencies)),
        "n_params": model.count_parameters(),
    }
```

### 5. Fonction `run_model` (orchestration d'un modèle)

```python
def run_model(
    model_name: str,
    config: dict,
    tasks: list[dict],
    exp_dir: Path,
) -> dict:
    """
    Entraîne, évalue et profile un modèle non supervisé. Sauvegarde les résultats.

    Parameters
    ----------
    model_name : str
    config : dict
    tasks : list[dict]
    exp_dir : Path

    Returns
    -------
    dict
        Métriques complètes (aa, af, bwt, auroc, ram_peak_bytes, latency, n_params).
    """
    print(f"\n{'#'*60}")
    print(f"# Modèle : {model_name.upper()}")
    print(f"{'#'*60}")

    model = build_model(model_name, config)

    # Entraînement CL
    acc_matrix, auroc_matrix = train_unsupervised(model, tasks)

    # Sauvegarde matrices
    np.save(exp_dir / "results" / f"acc_matrix_{model_name}.npy", acc_matrix)
    np.save(exp_dir / "results" / f"auroc_matrix_{model_name}.npy", auroc_matrix)

    # Sauvegarde checkpoint
    model.save(exp_dir / "checkpoints" / f"{model_name}_task2_final.pkl")

    # Métriques CL (AA, AF, BWT)
    cl_metrics = compute_cl_metrics(acc_matrix)

    # AUROC final (moyenne sur la diagonale de la dernière ligne — après entraînement complet)
    T = len(tasks)
    auroc_final = float(np.nanmean(auroc_matrix[T - 1, :T]))

    # Profiling mémoire (sur Task 0 val data)
    X_val_0 = np.concatenate(
        [x.numpy() for x, _ in tasks[0]["val_loader"]], axis=0
    )
    memory_report = profile_model(model, X_val_0, n_runs=config["evaluation"]["n_latency_runs"])

    full_metrics = {
        **cl_metrics,
        "auroc_final": auroc_final,
        **memory_report,
    }

    save_metrics(full_metrics, exp_dir / "results" / f"metrics_{model_name}.json")

    print(f"\n{format_metrics_report(cl_metrics, model_name=model_name.upper())}")
    print(f"AUROC final   : {auroc_final:.4f}")
    print(f"RAM peak      : {memory_report['ram_peak_bytes']} B ({memory_report['ram_peak_bytes']/1024:.1f} Ko)")
    print(f"Latence moy.  : {memory_report['inference_latency_ms']:.3f} ms")
    print(f"Paramètres    : {memory_report['n_params']}")

    return full_metrics
```

### 6. Fonction `main`

```python
def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    # Répertoire expérience
    exp_id = config.get("exp_id", "exp_005_unsupervised_dataset2")
    exp_dir = Path("experiments") / exp_id
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)

    # Snapshot config
    snapshot = {**config, "_timestamp": datetime.now().isoformat(), "_model": args.model}
    with open(exp_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)

    # Données
    data_cfg = config["data"]
    tasks = get_cl_dataloaders(
        csv_path=Path(data_cfg["csv_path"]),
        normalizer_path=Path(data_cfg["normalizer_path"]),
        batch_size=data_cfg.get("batch_size", 256),
        val_ratio=data_cfg.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
    )
    print(f"[Data] {len(tasks)} tâches : {[t['domain'] for t in tasks]}")

    # Sélection des modèles à entraîner
    models_to_run = SUPPORTED_MODELS if args.model == "all" else [args.model]

    all_metrics: dict[str, dict] = {}
    memory_reports: dict[str, dict] = {}

    for model_name in models_to_run:
        metrics = run_model(model_name, config, tasks, exp_dir)
        all_metrics[model_name] = metrics
        memory_reports[model_name] = {
            k: v for k, v in metrics.items()
            if k in ("ram_peak_bytes", "inference_latency_ms", "n_params")
        }

    # Tableau comparatif global
    if args.model == "all":
        with open(exp_dir / "results" / "metrics_all.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        with open(exp_dir / "results" / "memory_report.json", "w") as f:
            json.dump(memory_reports, f, indent=2)

        print("\n" + "="*60)
        print("COMPARATIF — 3 MODÈLES NON SUPERVISÉS")
        print("="*60)
        header = f"{'Modèle':<12} {'AA':>8} {'AF':>8} {'AUROC':>8} {'RAM(Ko)':>10} {'Latence(ms)':>13}"
        print(header)
        print("-" * len(header))
        for name, m in all_metrics.items():
            ram_ko = m.get("ram_peak_bytes", -1) / 1024
            print(
                f"{name.upper():<12} "
                f"{m.get('acc_final', float('nan')):>8.4f} "
                f"{m.get('avg_forgetting', float('nan')):>8.4f} "
                f"{m.get('auroc_final', float('nan')):>8.4f} "
                f"{ram_ko:>10.1f} "
                f"{m.get('inference_latency_ms', float('nan')):>13.3f}"
            )
        print(f"\nRésultats → {exp_dir}/results/")


if __name__ == "__main__":
    main()
```

---

## Structure de sortie attendue

```
experiments/exp_005_unsupervised_dataset2/
├── config_snapshot.yaml
├── checkpoints/
│   ├── kmeans_task2_final.pkl
│   ├── knn_task2_final.pkl
│   └── pca_task2_final.pkl
└── results/
    ├── metrics_kmeans.json
    ├── metrics_knn.json
    ├── metrics_pca.json
    ├── metrics_all.json           # comparatif 3 modèles
    ├── acc_matrix_kmeans.npy      # [3, 3]
    ├── acc_matrix_knn.npy
    ├── acc_matrix_pca.npy
    ├── auroc_matrix_kmeans.npy    # [3, 3]
    ├── auroc_matrix_knn.npy
    ├── auroc_matrix_pca.npy
    └── memory_report.json
```

---

## Métriques attendues (estimations)

| Métrique | KMeans | KNN | PCA | Justification |
|---------|:------:|:---:|:---:|---------------|
| `aa` | > 0.70 | > 0.70 | > 0.65 | Pas de supervision — précision inférieure aux modèles supervisés |
| `af` | ≈ 0.0 | ≈ 0.0 (accumulate) | variable | Dépend de la stratégie CL |
| `auroc_final` | > 0.65 | > 0.65 | > 0.60 | Dataset 2 a peu de domain shift |
| `inference_latency_ms` | < 1 ms | < 5 ms | < 1 ms | Opérations sklearn sur 4 features |

> Si `aa < 0.50` pour tous les modèles : vérifier que le seuil (`anomaly_percentile=95`) est adapté au ratio `faulty/normal` du Dataset 2. Un déséquilibre fort peut nécessiter un ajustement.
> Si `auroc_final ≈ 0.50` : les modèles ne discriminent pas mieux qu'un aléatoire — le domain shift est insuffisant pour les baselines non supervisées sur ce dataset.

---

## Critères d'acceptation

- [ ] `python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model kmeans` s'exécute sans erreur
- [ ] `python scripts/train_unsupervised.py --config configs/unsupervised_config.yaml --model all` produit `metrics_all.json`
- [ ] `experiments/exp_005_unsupervised_dataset2/results/metrics_all.json` contient `aa`, `avg_forgetting`, `auroc_final`, `ram_peak_bytes`, `inference_latency_ms` pour les 3 modèles
- [ ] `acc_matrix_*.npy` shape `[3, 3]` avec NaN pour les cases futures
- [ ] `config_snapshot.yaml` contient le timestamp d'exécution
- [ ] `ruff check scripts/train_unsupervised.py` + `black --check` passent
- [ ] Résultats reproductibles : deux runs avec `seed=42` produisent des métriques identiques

---

## Questions ouvertes

- `TODO(arnaud)` : AUROC ou accuracy avec seuil fixé comme métrique principale pour les non supervisés ? AUROC est plus robuste au déséquilibre de classes mais nécessite les labels en éval — ce qui est acceptable selon le protocole du sprint.
- `TODO(arnaud)` : faut-il ajouter une baseline supervisée (ex. MLP EWC) dans le même script pour comparaison directe, ou garder la comparaison dans le notebook S5-09 ?
- `TODO(fred)` : dans le contexte industriel Edge Spectrum, les labels sont-ils disponibles en temps réel pour calibrer le seuil de décision ? Sinon, le seuil doit être fixé a priori (ex. percentile sur données normales du commissioning).
- `FIXME(gap1)` : les 3 domaines du Dataset 2 ont des distributions similaires (voir S1-09) — les résultats non supervisés risquent d'être médiocres. Prévoir une note dans le rapport sur cette limitation.
- `FIXME(gap2)` : mesurer la RAM peak via tracemalloc et la comparer à une estimation analytique (taille des centroides, X_ref_, composantes PCA) pour valider la cohérence.
