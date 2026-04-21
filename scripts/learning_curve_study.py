"""
scripts/learning_curve_study.py — S9-08: Courbe AUROC vs N d'enrôlement.

Pour chaque N ∈ N_VALUES, sous-échantillonne X_normal[:N] depuis le domaine
d'enrôlement, fit le modèle, calcule l'AUROC sur un jeu de test fixe.
Répète n_repeats fois avec des seeds différents → mean ± std.

Usage
-----
    python scripts/learning_curve_study.py \\
      --config configs/unsupervised_config.yaml \\
      --models mahalanobis,kmeans,pca,knn \\
      --n_values 10,25,50,100,250,500,1000,2500 \\
      --n_repeats 5 \\
      --enrollment_domain Pump \\
      --exp_id exp_042_learning_curve

Sorties
-------
    experiments/exp_042_learning_curve/
    ├── config_snapshot.yaml
    └── results/
        ├── learning_curve_mahalanobis.json   # {N: [auroc_run1, ..., auroc_run5]}
        ├── learning_curve_kmeans.json
        ├── learning_curve_pca.json
        ├── learning_curve_knn.json
        └── learning_curve_all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.unsupervised import KMeansDetector, KNNDetector, MahalanobisDetector, PCABaseline
from src.utils.config_loader import load_config, save_config_snapshot
from src.utils.reproducibility import set_seed

# Valeurs de N à tester — passer par --n_values pour override
N_VALUES_DEFAULT: list[int] = [10, 25, 50, 100, 250, 500, 1000, 2500]
N_REPEATS_DEFAULT: int = 5
MAHALANOBIS_MIN_N: int = 5  # skip Mahalanobis si N < 5 (Σ non inversible)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Courbe AUROC vs N d'enrôlement — exp_042"
    )
    parser.add_argument("--config", default="configs/unsupervised_config.yaml")
    parser.add_argument(
        "--models",
        default="mahalanobis,kmeans,pca,knn",
        help="Modèles séparés par virgule (ex. mahalanobis,kmeans)",
    )
    parser.add_argument(
        "--n_values",
        default=",".join(str(n) for n in N_VALUES_DEFAULT),
        help="Valeurs de N séparées par virgule",
    )
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS_DEFAULT)
    parser.add_argument(
        "--enrollment_domain",
        default="Pump",
        help="Domaine d'enrôlement (équipement) : Pump | Turbine | Compressor",
    )
    parser.add_argument("--exp_id", default="exp_042_learning_curve")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Chargement données
# ---------------------------------------------------------------------------


def load_enrollment_data(cfg: dict, enrollment_domain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les données d'enrôlement (normaux uniquement) et le jeu de test fixe.

    Parameters
    ----------
    cfg : dict
        Config complète.
    enrollment_domain : str
        Domaine pour l'enrôlement (ex. "Pump").

    Returns
    -------
    X_normal : np.ndarray [N_normal, d]  — données normales du domaine d'enrôlement
    X_test   : np.ndarray [N_test, d]   — jeu de test fixe (tous domaines)
    y_test   : np.ndarray [N_test]      — labels test
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    csv_path = Path(cfg["data"]["csv_path"])
    feature_cols: list[str] = cfg["data"]["feature_columns"]
    label_col: str = cfg["data"]["label_column"]
    domain_col: str = cfg["data"]["domain_column"]

    df = pd.read_csv(csv_path)

    # Normalisation via normalizer YAML (cohérent avec le reste du projet)
    normalizer_path = Path(cfg["data"]["normalizer_path"])
    if normalizer_path.exists():
        import yaml
        with open(normalizer_path) as f:
            norm_params = yaml.safe_load(f)
        means = np.array([norm_params["mean"][c] for c in feature_cols])
        stds = np.array([norm_params["std"][c] for c in feature_cols])
        X_all = (df[feature_cols].values - means) / (stds + 1e-8)
    else:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[feature_cols].values)

    y_all = df[label_col].values.astype(np.int64)
    domains = df[domain_col].values

    # Enrôlement : normaux du domaine sélectionné
    mask_enroll = (domains == enrollment_domain) & (y_all == 0)
    X_normal = X_all[mask_enroll]

    # Test fixe : tous domaines mélangés (normal + anomalie)
    rng = np.random.default_rng(42)
    n_test = min(2000, len(X_all))
    idx_test = rng.choice(len(X_all), size=n_test, replace=False)
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]

    print(f"  Enrôlement : {X_normal.shape[0]} normaux ({enrollment_domain})")
    print(f"  Test fixe  : {X_test.shape[0]} samples ({y_test.sum()} anomalies)")
    return X_normal, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Factory modèles
# ---------------------------------------------------------------------------


def build_model(
    model_name: str, cfg: dict, n: int
) -> KMeansDetector | KNNDetector | PCABaseline | MahalanobisDetector:
    """
    Instancie un détecteur avec adaptation des hyperparamètres si N petit.

    Parameters
    ----------
    model_name : str
    cfg : dict
    n : int
        Nombre d'échantillons d'enrôlement (pour guard KMeans k_max).

    Returns
    -------
    Détecteur instancié.
    """
    if model_name == "mahalanobis":
        return MahalanobisDetector(cfg["mahalanobis"])
    elif model_name == "pca":
        return PCABaseline(cfg["pca"])
    elif model_name == "knn":
        return KNNDetector(cfg["knn"])
    elif model_name == "kmeans":
        kmeans_cfg = dict(cfg["kmeans"])
        kmeans_cfg["k_max"] = min(kmeans_cfg.get("k_max", 10), max(2, n // 2))
        return KMeansDetector(kmeans_cfg)
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r}")


# ---------------------------------------------------------------------------
# 4. Boucle learning curve
# ---------------------------------------------------------------------------


def run_learning_curve(
    model_name: str,
    cfg: dict,
    X_normal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_values: list[int],
    n_repeats: int,
) -> dict[int, list[float]]:
    """
    Calcule AUROC vs N pour un modèle donné.

    Parameters
    ----------
    model_name : str
    cfg : dict
    X_normal : np.ndarray [N_normal, d]
        Pool d'enrôlement (normaux uniquement).
    X_test : np.ndarray [N_test, d]
    y_test : np.ndarray [N_test]
    n_values : list[int]
    n_repeats : int

    Returns
    -------
    dict[int, list[float]]
        {N: [auroc_run1, ..., auroc_run_n_repeats]}
    """
    results: dict[int, list[float]] = {}

    for n in n_values:
        # Guard Mahalanobis : skip si N < MAHALANOBIS_MIN_N
        if model_name == "mahalanobis" and n < MAHALANOBIS_MIN_N:
            print(f"    [Mahalanobis] N={n} < {MAHALANOBIS_MIN_N} → skip")
            results[n] = []
            continue

        # Guard : skip si N > pool disponible
        if n > len(X_normal):
            print(f"    N={n} > pool ({len(X_normal)}) → skip")
            results[n] = []
            continue

        aurocs: list[float] = []
        for run in range(n_repeats):
            rng = np.random.default_rng(seed=run * 1000 + n)
            idx = rng.choice(len(X_normal), size=n, replace=False)
            X_enroll = X_normal[idx]

            model = build_model(model_name, cfg, n)
            model.fit_task(X_enroll, task_id=0)

            scores = model.anomaly_score(X_test)
            try:
                auroc = float(roc_auc_score(y_test, scores))
            except ValueError:
                auroc = float("nan")
            aurocs.append(auroc)

        valid = [v for v in aurocs if not (v != v)]  # filter NaN
        mean_auroc = float(np.mean(valid)) if valid else float("nan")
        print(f"    N={n:5d} | AUROC mean={mean_auroc:.4f} (runs={aurocs})")
        results[n] = aurocs

    return results


# ---------------------------------------------------------------------------
# 5. Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    n_values = [int(x) for x in args.n_values.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]

    exp_dir = Path(f"experiments/{args.exp_id}")
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Learning Curve Study — {args.exp_id}")
    print(f"  Domaine enrôlement : {args.enrollment_domain}")
    print(f"  N values : {n_values}")
    print(f"  Modèles  : {model_names}")
    print(f"  Répétitions : {args.n_repeats}")
    print(f"{'=' * 60}")

    save_config_snapshot(cfg, str(exp_dir))

    # Chargement données
    X_normal, X_test, y_test = load_enrollment_data(cfg, args.enrollment_domain)

    all_results: dict[str, dict[int, list[float]]] = {}

    for model_name in model_names:
        print(f"\n--- {model_name.upper()} ---")
        curve = run_learning_curve(
            model_name, cfg, X_normal, X_test, y_test, n_values, args.n_repeats
        )
        all_results[model_name] = curve

        # Sauvegarde individuelle — clés int sérialisées en str pour JSON
        out = {str(k): v for k, v in curve.items()}
        path = results_dir / f"learning_curve_{model_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  → {path}")

    # Sauvegarde globale
    all_out = {m: {str(k): v for k, v in curve.items()} for m, curve in all_results.items()}
    all_path = results_dir / "learning_curve_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nRésultats complets → {all_path}")

    # Résumé
    print(f"\n{'=' * 60}")
    print("  RÉSUMÉ — N minimal pour AUROC ≥ 0.85")
    print(f"{'=' * 60}")
    for model_name, curve in all_results.items():
        n_min = None
        for n in sorted(curve.keys()):
            vals = [v for v in curve[n] if v == v]  # filter NaN
            if vals and np.mean(vals) >= 0.85:
                n_min = n
                break
        print(f"  {model_name:<14} N_min={n_min if n_min else '> max'}")


if __name__ == "__main__":
    main()
