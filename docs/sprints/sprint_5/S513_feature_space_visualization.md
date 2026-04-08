# S5-13 — Visualisation espace des features + clusters

| Champ | Valeur |
|-------|--------|
| **ID** | S5-13 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S5-06 (`exp_005` exécuté), S5-10 (`mahalanobis_detector.py`), S5-04 (`pca_baseline.py`), S5-02 (`kmeans_detector.py`) |
| **Fichiers cibles** | `src/evaluation/feature_space_plots.py`, `notebooks/figures/05_feature_space_*.png` |
| **Complété le** | 2026-04-08 |

---

## Objectif

Produire des visualisations 2D de l'espace des features pour les 4 modèles non supervisés (K-Means, Mahalanobis, PCA, KNN) sur Dataset 2, permettant une interprétation qualitative de la séparation normal/faulty et de l'évolution des représentations au fil des domaines CL.

Ces figures sont destinées au **manuscrit** (Section résultats non supervisés) et au diagnostic des limitations Gap 1.

---

## Visualisations à produire

| Figure | Description | Fichier |
|--------|-------------|---------|
| Scatter PCA 2D | Points colorés par label (normal/faulty) et par domaine | `05_feature_space_scatter.png` |
| K-Means Voronoï | Centroids + régions de Voronoï approximées en espace PCA 2D | `05_feature_space_kmeans.png` |
| Ellipses Mahalanobis | Contours isodistance 1σ / 2σ / 3σ par domaine | `05_feature_space_mahalanobis.png` |
| PCA gradient reconstruction | Heatmap de fond = erreur de reconstruction sur une grille 2D | `05_feature_space_pca_recon.png` |
| Évolution domaines | 3 subplots (après T1, T2, T3) pour K-Means et Mahalanobis | `05_feature_space_cl_evolution.png` |

---

## Implémentation

### Module `src/evaluation/feature_space_plots.py`

```python
"""
feature_space_plots.py — Visualisations 2D de l'espace des features.

Toutes les fonctions prennent un `ax` matplotlib et retournent None.
Projection : PCA 2D sklearn (pas de UMAP/t-SNE — dépendances non portables).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from typing import Optional


COLORS_LABEL = {0: "#4CAF50", 1: "#F44336"}   # vert = normal, rouge = faulty
COLORS_DOMAIN = ["#2196F3", "#FF9800", "#9C27B0"]  # bleu, orange, violet


def fit_pca2d(X: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Ajuste une PCA 2D et retourne (pca, X_proj).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)

    Returns
    -------
    pca : PCA fitted
    X_proj : np.ndarray, shape (n, 2)
    """
    pca = PCA(n_components=2, random_state=42)
    X_proj = pca.fit_transform(X)
    return pca, X_proj


def plot_feature_space_2d(
    X_proj: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: plt.Axes,
    domain_ids: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    s: int = 15,
) -> None:
    """Scatter plot PCA 2D des points colorés par label (et optionnellement par domaine).

    Parameters
    ----------
    X_proj : np.ndarray, shape (n, 2) — projection PCA 2D
    y : np.ndarray, shape (n,) — labels 0/1
    title : str
    ax : plt.Axes
    domain_ids : np.ndarray optionnel, shape (n,) — indice de domaine (0,1,2)
    alpha : transparence des points
    s : taille des marqueurs
    """
    if domain_ids is not None:
        for d_id in np.unique(domain_ids):
            mask = domain_ids == d_id
            for label in [0, 1]:
                lmask = mask & (y == label)
                ax.scatter(
                    X_proj[lmask, 0], X_proj[lmask, 1],
                    c=COLORS_LABEL[label],
                    marker=["o", "^"][d_id % 3] if d_id < 3 else "s",
                    alpha=alpha, s=s, linewidths=0,
                    label=f"D{d_id+1} {'faulty' if label else 'normal'}",
                )
    else:
        for label in [0, 1]:
            mask = y == label
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                c=COLORS_LABEL[label],
                alpha=alpha, s=s, linewidths=0,
                label="faulty" if label else "normal",
            )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=1.5)


def plot_kmeans_voronoi(
    X_proj: np.ndarray,
    centroids_proj: np.ndarray,
    ax: plt.Axes,
    grid_res: int = 200,
) -> None:
    """Affiche les régions de Voronoï K-Means (grille colorée) + centroids.

    Parameters
    ----------
    X_proj : np.ndarray, shape (n, 2)
    centroids_proj : np.ndarray, shape (k, 2) — centroids projetés en PCA 2D
    ax : plt.Axes
    grid_res : résolution de la grille
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d  # type: ignore

    x_min, x_max = X_proj[:, 0].min() - 0.5, X_proj[:, 0].max() + 0.5
    y_min, y_max = X_proj[:, 1].min() - 0.5, X_proj[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Attribuer chaque point de grille au centroid le plus proche
    dists = np.linalg.norm(grid[:, None] - centroids_proj[None], axis=2)  # (G, k)
    labels_grid = dists.argmin(axis=1).reshape(xx.shape)

    cmap = plt.cm.get_cmap("Pastel1", len(centroids_proj))
    ax.contourf(xx, yy, labels_grid, levels=len(centroids_proj) - 1, cmap=cmap, alpha=0.35)
    ax.scatter(
        centroids_proj[:, 0], centroids_proj[:, 1],
        c="black", marker="X", s=120, zorder=5, label="Centroids",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_mahalanobis_ellipse(
    mu_proj: np.ndarray,
    cov_proj: np.ndarray,
    ax: plt.Axes,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
    color: str = "#2196F3",
    label: str = "",
) -> None:
    """Trace les ellipses de confiance Mahalanobis dans l'espace PCA 2D.

    Parameters
    ----------
    mu_proj : np.ndarray, shape (2,) — moyenne projetée
    cov_proj : np.ndarray, shape (2, 2) — covariance projetée (PCA @ Sigma @ PCA.T)
    ax : plt.Axes
    sigmas : rayons (en nombre de σ) des ellipses à tracer
    color : couleur des ellipses
    label : étiquette pour la légende (affiché sur σ=2 seulement)
    """
    vals, vecs = np.linalg.eigh(cov_proj)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    for i, sigma in enumerate(sigmas):
        width, height = 2 * sigma * np.sqrt(vals)
        ellipse = Ellipse(
            xy=mu_proj,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            fc="none",
            lw=1.5 - i * 0.3,
            linestyle=["solid", "dashed", "dotted"][i],
            label=f"{label} {int(sigma)}σ" if label and i == 0 else f"{int(sigma)}σ",
        )
        ax.add_patch(ellipse)
    ax.scatter(*mu_proj, c=color, marker="+", s=80, zorder=5)


def plot_pca_reconstruction_heatmap(
    pca: PCA,
    ax: plt.Axes,
    x_range: tuple[float, float] = (-4.0, 4.0),
    y_range: tuple[float, float] = (-4.0, 4.0),
    grid_res: int = 100,
) -> None:
    """Heatmap de fond = erreur de reconstruction PCA sur une grille 2D.

    Parameters
    ----------
    pca : PCA sklearn ajusté (n_components=2)
    ax : plt.Axes
    x_range, y_range : étendue de la grille dans l'espace PCA 2D
    grid_res : résolution de la grille
    """
    xx, yy = np.meshgrid(
        np.linspace(*x_range, grid_res),
        np.linspace(*y_range, grid_res),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Reconstruction : projeter en espace original puis re-projeter
    grid_orig = pca.inverse_transform(grid_2d)
    grid_recon = pca.transform(grid_orig)
    errors = np.linalg.norm(grid_2d - grid_recon, axis=1).reshape(xx.shape)

    im = ax.pcolormesh(xx, yy, errors, cmap="YlOrRd", alpha=0.6, shading="auto")
    plt.colorbar(im, ax=ax, label="Erreur de reconstruction PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
```

### Script `scripts/visualize_feature_space.py`

```python
"""
visualize_feature_space.py — Génère les figures de visualisation espace features.

Usage :
    python scripts/visualize_feature_space.py --config configs/unsupervised_config.yaml

Sorties :
    notebooks/figures/05_feature_space_scatter.png
    notebooks/figures/05_feature_space_kmeans.png
    notebooks/figures/05_feature_space_mahalanobis.png
    notebooks/figures/05_feature_space_pca_recon.png
    notebooks/figures/05_feature_space_cl_evolution.png
"""
```

### Intégration notebook

Ajouter une **Section 7** dans `notebooks/05_supervised_vs_unsupervised.ipynb` :

```python
# Section 7 — Visualisation espace des features (PCA 2D)
from src.evaluation.feature_space_plots import (
    fit_pca2d,
    plot_feature_space_2d,
    plot_kmeans_voronoi,
    plot_mahalanobis_ellipse,
    plot_pca_reconstruction_heatmap,
)
```

---

## Contraintes techniques

- **Pas de UMAP/t-SNE** : dépendances lourdes, non installées dans l'environnement embarqué → PCA 2D sklearn uniquement
- **matplotlib uniquement dans `src/evaluation/`** : jamais importé dans les modules modèle
- **Pas d'annotations `# MEM:`** : code d'évaluation PC-only, hors contrainte 64 Ko
- **Grille de résolution paramétrable** : `grid_res=100` par défaut (< 1 s sur CPU)

---

## Critères d'acceptation

- [x] `from src.evaluation.feature_space_plots import plot_feature_space_2d` s'importe sans erreur
- [x] Les 5 figures PNG sont générées dans `notebooks/figures/`
- [x] Les ellipses Mahalanobis sont cohérentes avec les μ/Σ rapportés dans exp_007 (d=4, 3 domaines)
- [x] Les régions K-Means Voronoï correspondent aux K clusters identifiés en exp_005 (K=2, silhouette)
- [x] Chaque figure inclut titre, légende, labels d'axes

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il inclure les projections par domaine séparément (3 subplots par modèle) ou une vue globale colorée par domaine suffit-elle pour le manuscrit ?
- `FIXME(gap1)` : si la séparation normal/faulty en PCA 2D est quasi-parfaite sur Dataset 2 (peu de domain shift), documenter explicitement cette limitation et référencer `docs/triple_gap.md` dans la figure caption.
