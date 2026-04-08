"""
visualize_feature_space.py — Génère les figures de visualisation espace features (S5-13).

Usage :
    python scripts/visualize_feature_space.py --config configs/unsupervised_config.yaml

Sorties (notebooks/figures/) :
    05_feature_space_scatter.png       — Scatter PCA 2D, coloré par domaine + label
    05_feature_space_kmeans.png        — Régions de Voronoï K-Means + scatter
    05_feature_space_mahalanobis.png   — Ellipses Mahalanobis 1σ/2σ/3σ par domaine
    05_feature_space_pca_recon.png     — Heatmap erreur reconstruction PCA
    05_feature_space_cl_evolution.png  — Évolution K-Means + Mahalanobis après T1/T2/T3
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ajout du répertoire racine au path (permet d'importer src.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.monitoring_dataset import get_cl_dataloaders
from src.evaluation.feature_space_plots import (
    COLORS_DOMAIN,
    DOMAIN_NAMES,
    FIGURE_FONT_SIZE,
    fit_pca2d,
    plot_cl_evolution,
    plot_feature_space_2d,
    plot_kmeans_voronoi,
    plot_mahalanobis_ellipse,
    plot_pca_reconstruction_heatmap,
)
from src.evaluation.plots import save_figure
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector
from src.utils.config_loader import load_config

OUTPUT_DIR = Path("notebooks/figures")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def extract_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    """Convertit un DataLoader PyTorch en tableaux NumPy (X, y)."""
    Xs, ys = [], []
    for X_batch, y_batch in loader:
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy().ravel())
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------


def load_dataset(cfg: dict) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    list[tuple[np.ndarray, np.ndarray]]
]:
    """Charge Dataset 2 et retourne les tableaux nécessaires aux figures.

    Parameters
    ----------
    cfg : dict
        Configuration issue de unsupervised_config.yaml.

    Returns
    -------
    X_all : np.ndarray, shape (N, 4)
        Toutes les features (train + val, 3 domaines).
    y_all : np.ndarray, shape (N,)
        Labels binaires correspondants.
    domain_ids_all : np.ndarray, shape (N,)
        Indice de domaine 0/1/2 (Pump/Turbine/Compressor).
    task_arrays : list[tuple[np.ndarray, np.ndarray]]
        [(X_train_pump, y_train_pump), (X_train_turbine, ...), (X_train_compressor, ...)].
        Train uniquement — utilisé pour l'entraînement dans plot_cl_evolution.
    """
    csv_path = Path(cfg["data"]["csv_path"])
    normalizer_path = Path(cfg["data"]["normalizer_path"])

    tasks = get_cl_dataloaders(csv_path=csv_path, normalizer_path=normalizer_path)

    X_parts, y_parts, domain_parts = [], [], []
    task_arrays = []

    for i, task in enumerate(tasks):
        X_train, y_train = extract_numpy(task["train_loader"])
        X_val, y_val = extract_numpy(task["val_loader"])

        # Pour l'évolution CL : train uniquement
        task_arrays.append((X_train, y_train))

        # Pour le scatter global : train + val
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        X_parts.append(X_full)
        y_parts.append(y_full)
        domain_parts.append(np.full(len(X_full), i, dtype=np.int64))

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    domain_ids_all = np.concatenate(domain_parts, axis=0)

    print(
        f"[data] X_all={X_all.shape}, y_all={y_all.shape}, "
        f"domain_ids_all={domain_ids_all.shape} "
        f"(faulty={y_all.sum()} / {len(y_all)})"
    )
    return X_all, y_all, domain_ids_all, task_arrays


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------


def load_models(cfg: dict) -> dict:
    """Charge les modèles pickles entraînés depuis exp_005 et exp_007.

    Returns
    -------
    dict avec clés "kmeans", "knn", "pca", "mahalanobis".
    """
    base_exp005 = Path("experiments/exp_005_unsupervised_dataset2/results")
    base_exp007 = Path("experiments/exp_007_mahalanobis/checkpoints")

    models = {}
    for name, path in [
        ("kmeans", base_exp005 / "model_kmeans.pkl"),
        ("knn", base_exp005 / "model_knn.pkl"),
        ("pca", base_exp005 / "model_pca.pkl"),
        ("mahalanobis", base_exp007 / "mahalanobis_dataset2_task2_final.pkl"),
    ]:
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"[models] {name} chargé depuis {path}")

    return models


# ---------------------------------------------------------------------------
# Génération des figures
# ---------------------------------------------------------------------------


def make_scatter(X_proj: np.ndarray, y: np.ndarray, domain_ids: np.ndarray,
                 pca2d, output_path: Path) -> None:
    """Figure 1 — Scatter PCA 2D coloré par domaine + label."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_feature_space_2d(
        X_proj, y,
        title="Espace des features — 3 domaines (PCA 2D globale)",
        ax=ax,
        domain_ids=domain_ids,
        alpha=0.4,
        s=8,
    )
    ax.set_xlabel(
        f"PC1 ({pca2d.explained_variance_ratio_[0] * 100:.1f}% var.)",
        fontsize=FIGURE_FONT_SIZE,
    )
    ax.set_ylabel(
        f"PC2 ({pca2d.explained_variance_ratio_[1] * 100:.1f}% var.)",
        fontsize=FIGURE_FONT_SIZE,
    )
    save_figure(fig, output_path)


def make_kmeans(X_proj: np.ndarray, y: np.ndarray, domain_ids: np.ndarray,
                pca2d, kmeans_model, output_path: Path) -> None:
    """Figure 2 — Régions de Voronoï K-Means + scatter."""
    centroids_proj = pca2d.transform(kmeans_model.kmeans_.cluster_centers_)
    k = len(centroids_proj)

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_kmeans_voronoi(X_proj, centroids_proj, ax=ax, grid_res=200)
    plot_feature_space_2d(
        X_proj, y,
        title=f"K-Means — Régions de Voronoï (k={k}, modèle après T3=Compressor)",
        ax=ax,
        domain_ids=domain_ids,
        alpha=0.3,
        s=6,
    )
    # Re-légender pour inclure les centroids
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=2, loc="best")
    save_figure(fig, output_path)


def make_mahalanobis(X_proj: np.ndarray, y: np.ndarray, domain_ids: np.ndarray,
                     pca2d, task_arrays: list, cfg: dict, output_path: Path) -> None:
    """Figure 3 — Ellipses Mahalanobis 1σ/2σ/3σ par domaine (refit par domaine).

    Le pickle mahalanobis_dataset2_task2_final.pkl ne contient que l'état après T3.
    On refitte un MahalanobisDetector par domaine pour obtenir les 3 ellipses.
    """
    V = pca2d.components_  # (2, 4)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter en arrière-plan
    plot_feature_space_2d(
        X_proj, y,
        title="Mahalanobis — Ellipses de confiance 1σ/2σ/3σ par domaine",
        ax=ax,
        domain_ids=domain_ids,
        alpha=0.2,
        s=5,
    )

    for i, (X_task, _) in enumerate(task_arrays):
        mah = MahalanobisDetector(cfg["mahalanobis"])
        mah.fit_task(X_task, task_id=i)

        mu_proj = pca2d.transform(mah.mu_.reshape(1, -1))[0]  # (2,)
        Sigma = np.linalg.inv(mah.sigma_inv_)                  # (4, 4)
        cov_proj = V @ Sigma @ V.T                              # (2, 2)

        plot_mahalanobis_ellipse(
            mu_proj,
            cov_proj,
            ax=ax,
            sigmas=[1, 2, 3],
            color=COLORS_DOMAIN[i],
            label=f"{DOMAIN_NAMES[i]} (T{i + 1})",
        )

    # Ajustement des limites pour que les ellipses soient visibles
    x_pad = (X_proj[:, 0].max() - X_proj[:, 0].min()) * 0.05
    y_pad = (X_proj[:, 1].max() - X_proj[:, 1].min()) * 0.05
    ax.set_xlim(X_proj[:, 0].min() - x_pad, X_proj[:, 0].max() + x_pad)
    ax.set_ylim(X_proj[:, 1].min() - y_pad, X_proj[:, 1].max() + y_pad)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=2, loc="best")
    save_figure(fig, output_path)


def make_pca_recon(X_proj: np.ndarray, y: np.ndarray, domain_ids: np.ndarray,
                   pca2d, pca_model, output_path: Path) -> None:
    """Figure 4 — Heatmap de l'erreur de reconstruction PCA."""
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_pca_reconstruction_heatmap(pca_model.pca_, pca2d, X_proj, ax=ax, grid_res=100)
    plot_feature_space_2d(
        X_proj, y,
        title="PCA — Heatmap erreur de reconstruction (modèle fitté sur T3=Compressor)",
        ax=ax,
        domain_ids=domain_ids,
        alpha=0.3,
        s=6,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=2, loc="best")
    save_figure(fig, output_path)


def make_cl_evolution(task_arrays: list, pca2d, output_path: Path) -> None:
    """Figure 5 — Évolution K-Means (haut) et Mahalanobis (bas) après T1/T2/T3."""
    fig = plot_cl_evolution(task_arrays, pca2d, figsize=(18, 10))
    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S5-13 — Génère les visualisations de l'espace des features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unsupervised_config.yaml",
        help="Chemin vers unsupervised_config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    print("=" * 60)
    print("S5-13 — Visualisation espace des features")
    print("=" * 60)

    # 1. Chargement des données
    X_all, y_all, domain_ids_all, task_arrays = load_dataset(cfg)

    # 2. PCA 2D globale (ajustée sur tous les domaines)
    pca2d, X_proj_all = fit_pca2d(X_all)
    print(
        f"[pca2d] Variance expliquée : "
        f"PC1={pca2d.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2={pca2d.explained_variance_ratio_[1] * 100:.1f}%"
    )

    # 3. Chargement des modèles entraînés
    models = load_models(cfg)

    # 4. Génération des figures
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[fig 1/5] Scatter PCA 2D...")
    make_scatter(X_proj_all, y_all, domain_ids_all, pca2d,
                 OUTPUT_DIR / "05_feature_space_scatter.png")

    print("[fig 2/5] K-Means Voronoï...")
    make_kmeans(X_proj_all, y_all, domain_ids_all, pca2d, models["kmeans"],
                OUTPUT_DIR / "05_feature_space_kmeans.png")

    print("[fig 3/5] Ellipses Mahalanobis...")
    make_mahalanobis(X_proj_all, y_all, domain_ids_all, pca2d, task_arrays, cfg,
                     OUTPUT_DIR / "05_feature_space_mahalanobis.png")

    print("[fig 4/5] Heatmap PCA reconstruction...")
    make_pca_recon(X_proj_all, y_all, domain_ids_all, pca2d, models["pca"],
                   OUTPUT_DIR / "05_feature_space_pca_recon.png")

    print("[fig 5/5] Évolution CL...")
    make_cl_evolution(task_arrays, pca2d, OUTPUT_DIR / "05_feature_space_cl_evolution.png")

    print("\n" + "=" * 60)
    print(f"[done] 5 figures sauvegardées dans {OUTPUT_DIR}/")
    for name in ["scatter", "kmeans", "mahalanobis", "pca_recon", "cl_evolution"]:
        path = OUTPUT_DIR / f"05_feature_space_{name}.png"
        size_kb = path.stat().st_size // 1024
        print(f"  {path.name}  ({size_kb} Ko)")
    print("=" * 60)


if __name__ == "__main__":
    main()
