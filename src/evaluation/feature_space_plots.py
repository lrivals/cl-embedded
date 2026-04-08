"""feature_space_plots.py — Visualisations 2D de l'espace des features.

Toutes les fonctions prennent un `ax` matplotlib et retournent None,
sauf plot_cl_evolution() qui crée et retourne sa propre Figure.

Projections disponibles (toutes sklearn, sans dépendance externe) :
  - PCA linéaire            : fit_pca2d()
  - Kernel PCA RBF          : fit_kpca2d(kernel='rbf')   — capture la structure radiale
  - Kernel PCA polynomial   : fit_kpca2d(kernel='poly')
  - t-SNE                   : fit_tsne2d()                — pas de transform() sur nouveaux points
  - Toutes via factory      : fit_projection(X, method)

Contexte données Dataset 2 : les points normaux forment une masse centrale compacte,
les points faulty se distribuent en anneau autour — structure radiale non-linéaire
que la PCA linéaire ne capture pas toujours.

Contrainte : pas d'annotations # MEM: — code d'évaluation PC-only, hors contrainte 64 Ko.
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

matplotlib.use("Agg")  # backend non-interactif (cohérent avec plots.py)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

COLORS_LABEL: dict[int, str] = {0: "#4CAF50", 1: "#F44336"}  # vert = normal, rouge = faulty
COLORS_DOMAIN: list[str] = ["#2196F3", "#FF9800", "#9C27B0"]  # bleu, orange, violet
DOMAIN_NAMES: list[str] = ["Pump", "Turbine", "Compressor"]
FIGURE_FONT_SIZE: int = 11
MARKERS_LABEL: dict[int, str] = {0: "o", 1: "x"}  # cercle = normal, croix = faulty


def fit_pca2d(X: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Ajuste une PCA 2D sur X et retourne (pca, X_proj).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Données d'entrée (ex. D=4 pour Dataset 2).

    Returns
    -------
    pca : PCA
        Modèle PCA sklearn ajusté (n_components=2).
    X_proj : np.ndarray, shape (N, 2)
        Projection 2D de X.
    """
    pca = PCA(n_components=2, random_state=42)
    X_proj = pca.fit_transform(X)
    return pca, X_proj


def fit_kpca2d(
    X: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
) -> tuple[KernelPCA, np.ndarray]:
    """Ajuste un Kernel PCA 2D et retourne (kpca, X_proj).

    Adapté à la structure radiale des données Dataset 2 (normaux au centre,
    faulty en anneau) que la PCA linéaire aplatit souvent.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    kernel : str
        'rbf' (défaut) ou 'poly'. RBF capture les structures radiales/sphériques ;
        poly capture les non-linéarités polynomiales.
    gamma : float | None
        Paramètre RBF (1 / (2 * sigma²)). None = 1/d (sklearn défaut).
    degree : int
        Degré pour kernel='poly' (ignoré si kernel='rbf').

    Returns
    -------
    kpca : KernelPCA
    X_proj : np.ndarray, shape (N, 2)
    """
    kpca = KernelPCA(
        n_components=2,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        random_state=42,
        fit_inverse_transform=False,
    )
    X_proj = kpca.fit_transform(X)
    return kpca, X_proj


def fit_tsne2d(
    X: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> tuple[TSNE, np.ndarray]:
    """Ajuste un t-SNE 2D et retourne (tsne, X_proj).

    t-SNE révèle la structure locale des clusters et l'anneau faulty autour
    de la masse normale mieux que la PCA linéaire.

    Limitations :
    - Pas de transform() sur de nouveaux points (embedding non-paramétrique).
    - Non déterministe même avec random_state=42 si le backend numpy varie.
    - Lent sur N > 10 000 (BH approximation, O(N log N)).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    perplexity : float
        Perplexité (compromis structure locale/globale). Valeur typique : 20–50.
        Réduire à ~10 si N < 500.
    n_iter : int
        Nombre d'itérations d'optimisation (défaut : 1000).

    Returns
    -------
    tsne : TSNE
        Instance sklearn fitted (embedding_ disponible, pas de transform()).
    X_proj : np.ndarray, shape (N, 2)
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        init="pca",          # initialisation PCA pour reproductibilité
        learning_rate="auto",
    )
    X_proj = tsne.fit_transform(X)
    return tsne, X_proj


_PROJECTION_LABELS: dict[str, tuple[str, str]] = {
    "pca":      ("PC1", "PC2"),
    "kpca_rbf": ("KPCA-RBF 1", "KPCA-RBF 2"),
    "kpca_poly": ("KPCA-Poly 1", "KPCA-Poly 2"),
    "tsne":     ("t-SNE 1", "t-SNE 2"),
}


def fit_projection(
    X: np.ndarray,
    method: str = "pca",
    **kwargs: object,
) -> tuple[object, np.ndarray, str, str]:
    """Factory de projections 2D — dispatche vers fit_pca2d / fit_kpca2d / fit_tsne2d.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    method : str
        'pca' | 'kpca_rbf' | 'kpca_poly' | 'tsne'
    **kwargs
        Forwarded vers la fonction correspondante (ex. gamma=0.5, perplexity=20).

    Returns
    -------
    model : PCA | KernelPCA | TSNE
        Modèle ajusté.
    X_proj : np.ndarray, shape (N, 2)
    xlabel : str
        Label axe X suggéré pour plot_feature_space_2d.
    ylabel : str
        Label axe Y suggéré pour plot_feature_space_2d.

    Raises
    ------
    ValueError
        Si method n'est pas dans {'pca', 'kpca_rbf', 'kpca_poly', 'tsne'}.
    """
    if method == "pca":
        model, X_proj = fit_pca2d(X)
    elif method == "kpca_rbf":
        model, X_proj = fit_kpca2d(X, kernel="rbf", **kwargs)  # type: ignore[arg-type]
    elif method == "kpca_poly":
        model, X_proj = fit_kpca2d(X, kernel="poly", **kwargs)  # type: ignore[arg-type]
    elif method == "tsne":
        model, X_proj = fit_tsne2d(X, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"method={method!r} invalide. Choisir parmi {list(_PROJECTION_LABELS)}")

    xlabel, ylabel = _PROJECTION_LABELS[method]
    return model, X_proj, xlabel, ylabel


def plot_feature_space_2d(
    X_proj: np.ndarray,
    y: np.ndarray,
    title: str,
    ax: plt.Axes,
    domain_ids: np.ndarray | None = None,
    alpha: float = 0.4,
    s: float = 8,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
) -> None:
    """Scatter 2D coloré par label (et optionnellement par domaine).

    Compatible avec toutes les projections retournées par fit_projection()
    (PCA, Kernel PCA, t-SNE).

    Parameters
    ----------
    X_proj : np.ndarray, shape (N, 2)
        Projection 2D (PCA, KPCA, t-SNE…).
    y : np.ndarray, shape (N,)
        Labels binaires 0=normal, 1=faulty.
    title : str
        Titre de la figure.
    ax : plt.Axes
        Axes matplotlib cibles.
    domain_ids : np.ndarray | None, shape (N,)
        Indice de domaine (0=Pump, 1=Turbine, 2=Compressor).
        Si None, coloration par label uniquement.
    alpha : float
        Transparence des points.
    s : float
        Taille des marqueurs.
    xlabel : str
        Label axe X (ex. 'PC1', 'KPCA-RBF 1', 't-SNE 1').
    ylabel : str
        Label axe Y.
    """
    if domain_ids is not None:
        for d_id in sorted(np.unique(domain_ids).astype(int)):
            for label in [0, 1]:
                mask = (domain_ids == d_id) & (y == label)
                if mask.sum() == 0:
                    continue
                color = COLORS_DOMAIN[d_id % len(COLORS_DOMAIN)]
                marker = MARKERS_LABEL[label]
                lbl = f"{DOMAIN_NAMES[d_id % len(DOMAIN_NAMES)]} ({'faulty' if label else 'normal'})"
                ax.scatter(
                    X_proj[mask, 0],
                    X_proj[mask, 1],
                    c=color,
                    marker=marker,
                    alpha=alpha,
                    s=s,
                    linewidths=0.5,
                    label=lbl,
                )
    else:
        for label in [0, 1]:
            mask = y == label
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                c=COLORS_LABEL[label],
                alpha=alpha,
                s=s,
                linewidths=0,
                label="faulty" if label else "normal",
            )

    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FIGURE_FONT_SIZE)
    ax.legend(fontsize=FIGURE_FONT_SIZE - 2, ncol=2, loc="best", markerscale=1.5)
    ax.grid(True, alpha=0.3)


def plot_kmeans_voronoi(
    X_proj: np.ndarray,
    centroids_proj: np.ndarray,
    ax: plt.Axes,
    grid_res: int = 200,
) -> None:
    """Affiche les régions de Voronoï K-Means (grille colorée) + centroids.

    Pas de scipy requis — les régions sont calculées par argmin numpy sur une grille.

    Parameters
    ----------
    X_proj : np.ndarray, shape (N, 2)
        Données projetées (utilisées pour l'étendue de la grille).
    centroids_proj : np.ndarray, shape (k, 2)
        Centroids K-Means projetés en PCA 2D.
    ax : plt.Axes
    grid_res : int
        Résolution de la grille (défaut=200, < 0.5 s sur CPU).
    """
    k = len(centroids_proj)
    x_min, x_max = X_proj[:, 0].min() - 0.5, X_proj[:, 0].max() + 0.5
    y_min, y_max = X_proj[:, 1].min() - 0.5, X_proj[:, 1].max() + 0.5

    # Grille de points pour l'approximation des régions de Voronoï
    xx_centers = np.linspace(x_min, x_max, grid_res)
    yy_centers = np.linspace(y_min, y_max, grid_res)
    xx, yy = np.meshgrid(xx_centers, yy_centers)
    grid_pts = np.c_[xx.ravel(), yy.ravel()]  # (grid_res^2, 2)

    # Distance euclidienne de chaque point de grille à chaque centroid
    dists = np.sqrt(
        ((grid_pts[:, None, :] - centroids_proj[None, :, :]) ** 2).sum(axis=-1)
    )  # (grid_res^2, k)
    region_labels = dists.argmin(axis=1).reshape(xx.shape)  # (grid_res, grid_res)

    # Colormap pastel à partir de COLORS_DOMAIN
    pastel = [mcolors.to_rgba(COLORS_DOMAIN[i % len(COLORS_DOMAIN)], alpha=0.25) for i in range(k)]
    cmap_v = mcolors.ListedColormap(pastel)

    # pcolormesh avec shading='flat' requiert grid_res+1 bords
    x_edges = np.linspace(x_min, x_max, grid_res + 1)
    y_edges = np.linspace(y_min, y_max, grid_res + 1)
    ax.pcolormesh(x_edges, y_edges, region_labels, cmap=cmap_v, vmin=-0.5, vmax=k - 0.5, shading="flat")

    # Centroids en étoiles noires
    ax.scatter(
        centroids_proj[:, 0],
        centroids_proj[:, 1],
        c="black",
        marker="*",
        s=200,
        zorder=5,
        label="Centroids",
    )
    ax.set_xlabel("PC1", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=FIGURE_FONT_SIZE)


def plot_mahalanobis_ellipse(
    mu_proj: np.ndarray,
    cov_proj: np.ndarray,
    ax: plt.Axes,
    sigmas: list[int] | None = None,
    color: str = "#2196F3",
    label: str = "",
) -> None:
    """Trace les ellipses de confiance Mahalanobis dans l'espace PCA 2D.

    Parameters
    ----------
    mu_proj : np.ndarray, shape (2,)
        Moyenne projetée dans l'espace PCA 2D.
    cov_proj : np.ndarray, shape (2, 2)
        Covariance projetée : V @ Sigma @ V.T où V = pca.components_ (2, d).
    ax : plt.Axes
    sigmas : list[int] | None
        Rayons en nombre de σ. Défaut : [1, 2, 3].
    color : str
        Couleur des ellipses.
    label : str
        Étiquette légende (affiché sur σ=1 seulement).
    """
    if sigmas is None:
        sigmas = [1, 2, 3]

    # Décomposition propre de la covariance (2x2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_proj)  # eigenvalues ordre croissant
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # garde numérique

    # Angle de l'axe majeur (eigenvecteur du plus grand eigenvalue)
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))

    linestyles = ["-", "--", ":"]
    for i, sigma in enumerate(sigmas):
        width = 2.0 * sigma * np.sqrt(eigenvalues[-1])  # axe majeur
        height = 2.0 * sigma * np.sqrt(eigenvalues[0])   # axe mineur
        ls = linestyles[i % len(linestyles)]
        ellipse = mpatches.Ellipse(
            xy=mu_proj,
            width=width,
            height=height,
            angle=angle_deg,
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
            linestyle=ls,
            alpha=0.85,
            label=f"{label} {sigma}σ" if label and i == 0 else f"{sigma}σ",
        )
        ax.add_patch(ellipse)

    ax.scatter(*mu_proj, c=color, marker="+", s=150, zorder=5, linewidths=2)


def plot_pca_reconstruction_heatmap(
    pca_model: PCA,
    pca2d: PCA,
    X_proj: np.ndarray,
    ax: plt.Axes,
    grid_res: int = 100,
) -> None:
    """Heatmap de l'erreur de reconstruction PCA sur une grille dans l'espace 2D.

    Permet de visualiser les zones où le modèle PCA commet des erreurs élevées
    (régions éloignées de la distribution apprise pendant l'entraînement).

    Parameters
    ----------
    pca_model : PCA
        Modèle PCA interne du PCABaseline (pca_baseline.pca_), ajusté sur les données.
        La reconstruction 4D→2D→4D est calculée sur une grille dans l'espace pca2d.
    pca2d : PCA
        PCA 2D global (ajusté sur tout le dataset). Utilisé pour mapper la grille → 4D.
    X_proj : np.ndarray, shape (N, 2)
        Projection globale (utilisée pour l'étendue de la grille).
    ax : plt.Axes
    grid_res : int
        Résolution de la grille (défaut=100, < 0.5 s sur CPU).
    """
    x_min, x_max = X_proj[:, 0].min() - 0.5, X_proj[:, 0].max() + 0.5
    y_min, y_max = X_proj[:, 1].min() - 0.5, X_proj[:, 1].max() + 0.5

    xx_c = np.linspace(x_min, x_max, grid_res)
    yy_c = np.linspace(y_min, y_max, grid_res)
    xx, yy = np.meshgrid(xx_c, yy_c)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]  # (grid_res^2, 2)

    # Retour en espace 4D via pca2d, puis reconstruction par pca_model
    grid_4d = pca2d.inverse_transform(grid_2d)  # (grid_res^2, 4)
    grid_proj = pca_model.transform(grid_4d)
    grid_recon = pca_model.inverse_transform(grid_proj)  # (grid_res^2, 4)

    errors = np.mean((grid_4d - grid_recon) ** 2, axis=1)  # MSE par point
    errors_2d = errors.reshape(grid_res, grid_res)

    im = ax.pcolormesh(xx, yy, errors_2d, cmap="hot_r", shading="auto", alpha=0.75)
    plt.colorbar(im, ax=ax, label="Reconstruction MSE")
    ax.set_xlabel("PC1", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("PC2", fontsize=FIGURE_FONT_SIZE)


def plot_cl_evolution(
    task_arrays: list[tuple[np.ndarray, np.ndarray]],
    pca2d: PCA,
    figsize: tuple[float, float] = (18, 10),
) -> plt.Figure:
    """Figure 2×3 montrant l'évolution des modèles K-Means et Mahalanobis après chaque tâche CL.

    Chaque colonne correspond à l'état du modèle après entraînement sur la tâche T_col+1.
    Les données vues jusqu'à la tâche courante sont affichées en arrière-plan (cumulatif).

    Parameters
    ----------
    task_arrays : list[tuple[np.ndarray, np.ndarray]]
        Liste de 3 tuples (X_train, y_train) pour Pump, Turbine, Compressor.
    pca2d : PCA
        PCA 2D global (ajusté sur tout le dataset).
    figsize : tuple[float, float]
        Taille de la figure (défaut : 18×10 pour 2×3 subplots).

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # k=2 fixé — toutes les 3 tâches ont sélectionné k=2 par critère silhouette (vérifié sur exp_005)
    K_CL = 2

    V = pca2d.components_  # (2, 4) — vecteurs propres PCA globale

    for col in range(3):
        # Données cumulatives jusqu'à la tâche col (incluse)
        X_cumul = np.concatenate([task_arrays[c][0] for c in range(col + 1)], axis=0)
        y_cumul = np.concatenate([task_arrays[c][1] for c in range(col + 1)], axis=0)
        domain_cumul = np.concatenate(
            [np.full(len(task_arrays[c][0]), c) for c in range(col + 1)], axis=0
        )
        X_proj_cumul = pca2d.transform(X_cumul)

        # Données de la tâche courante (pour l'entraînement du modèle)
        X_fit = task_arrays[col][0]

        # --- Ligne 0 : K-Means ---
        ax_km = axes[0, col]
        km = KMeans(n_clusters=K_CL, n_init=10, max_iter=300, random_state=42)
        km.fit(X_fit)
        centroids_proj = pca2d.transform(km.cluster_centers_)  # (2, 2)

        plot_kmeans_voronoi(X_proj_cumul, centroids_proj, ax=ax_km, grid_res=100)
        # Scatter léger des données cumulatives
        for d_id in range(col + 1):
            mask = domain_cumul == d_id
            ax_km.scatter(
                X_proj_cumul[mask, 0],
                X_proj_cumul[mask, 1],
                c=COLORS_DOMAIN[d_id],
                alpha=0.25,
                s=5,
                linewidths=0,
                label=DOMAIN_NAMES[d_id],
            )
        ax_km.set_title(
            f"K-Means après T{col + 1} ({DOMAIN_NAMES[col]})",
            fontsize=FIGURE_FONT_SIZE,
            fontweight="bold",
        )
        ax_km.legend(fontsize=7, ncol=1, loc="upper right")

        # --- Ligne 1 : Mahalanobis ---
        ax_mah = axes[1, col]
        mu = X_fit.mean(axis=0)  # (4,)
        cov = np.cov(X_fit, rowvar=False)  # (4, 4)
        cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
        sigma_inv = np.linalg.inv(cov_reg)
        Sigma = np.linalg.inv(sigma_inv)
        cov_proj = V @ Sigma @ V.T  # (2, 2)
        mu_proj = pca2d.transform(mu.reshape(1, -1))[0]  # (2,)

        # Scatter léger des données cumulatives
        for d_id in range(col + 1):
            mask = domain_cumul == d_id
            ax_mah.scatter(
                X_proj_cumul[mask, 0],
                X_proj_cumul[mask, 1],
                c=COLORS_DOMAIN[d_id],
                alpha=0.25,
                s=5,
                linewidths=0,
                label=DOMAIN_NAMES[d_id],
            )
        plot_mahalanobis_ellipse(
            mu_proj,
            cov_proj,
            ax=ax_mah,
            sigmas=[1, 2, 3],
            color=COLORS_DOMAIN[col],
            label=DOMAIN_NAMES[col],
        )
        ax_mah.set_title(
            f"Mahalanobis après T{col + 1} ({DOMAIN_NAMES[col]})",
            fontsize=FIGURE_FONT_SIZE,
            fontweight="bold",
        )
        ax_mah.set_xlabel("PC1", fontsize=FIGURE_FONT_SIZE - 1)
        ax_mah.set_ylabel("PC2", fontsize=FIGURE_FONT_SIZE - 1)
        # Ajustement des limites pour que les ellipses soient visibles
        x_pad = (X_proj_cumul[:, 0].max() - X_proj_cumul[:, 0].min()) * 0.1
        y_pad = (X_proj_cumul[:, 1].max() - X_proj_cumul[:, 1].min()) * 0.1
        ax_mah.set_xlim(X_proj_cumul[:, 0].min() - x_pad, X_proj_cumul[:, 0].max() + x_pad)
        ax_mah.set_ylim(X_proj_cumul[:, 1].min() - y_pad, X_proj_cumul[:, 1].max() + y_pad)
        ax_mah.legend(fontsize=7, ncol=1, loc="upper right")

    fig.suptitle(
        "Évolution CL — K-Means (haut) et Mahalanobis (bas) après chaque domaine",
        fontsize=FIGURE_FONT_SIZE + 2,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
