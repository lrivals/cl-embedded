"""Visualisation des résultats CL : accuracy matrix, forgetting curves, comparaison.

Contrainte : ce module ne doit être importé que depuis src/evaluation/ et notebooks/.
Ne jamais importer matplotlib/seaborn dans les modules modèles ou d'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (serveurs, CI)
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Style cohérent avec le projet
FIGURE_DPI: int = 150
FIGURE_FONT_SIZE: int = 11
CMAP_ACCURACY: str = "YlOrRd_r"  # vert=bonne accuracy, rouge=oubli


def plot_accuracy_matrix(
    acc_matrix: np.ndarray,
    task_names: list[str] | None = None,
    title: str = "Accuracy Matrix",
    ax: plt.Axes | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """
    Visualise la matrice d'accuracy CL sous forme de heatmap.

    acc_matrix[i, j] = accuracy sur la tâche j après entraînement sur la tâche i.
    Les cases NaN (tâches non encore vues) apparaissent en gris.

    Parameters
    ----------
    acc_matrix : np.ndarray [T, T]
        Matrice produite par run_cl_scenario() ou train_naive_sequential().
    task_names : list[str] | None
        Noms des tâches (ex. ["Pump", "Turbine", "Compressor"]).
        Si None, utilise ["T1", "T2", ...].
    title : str
        Titre de la figure.
    ax : plt.Axes | None
        Axes existants à utiliser. Si None, une nouvelle figure est créée.
    vmin, vmax : float
        Plage de couleur (défaut : [0, 1]).

    Returns
    -------
    plt.Figure
        Figure matplotlib (à sauvegarder via save_figure()).

    Notes
    -----
    Référence visuelle : DeLange2021Survey Fig. 3 — accuracy matrix display.
    """
    n_tasks = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"T{i + 1}" for i in range(n_tasks)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, n_tasks * 1.5), max(3, n_tasks * 1.2)))
    else:
        fig = ax.get_figure()

    # Masquer les NaN (triangle supérieur)
    mask = np.isnan(acc_matrix)
    data = np.where(mask, -1.0, acc_matrix)  # sentinelle pour la colormap

    if _HAS_SEABORN:
        cmap = sns.color_palette(CMAP_ACCURACY, as_cmap=True)
        cmap.set_under("lightgrey")  # NaN = gris
        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=task_names,
            yticklabels=[f"After T{i + 1}" for i in range(n_tasks)],
            linewidths=0.5,
            linecolor="white",
            mask=mask,
        )
    else:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=CMAP_ACCURACY, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(task_names)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([f"After T{i + 1}" for i in range(n_tasks)])
        for i in range(n_tasks):
            for j in range(n_tasks):
                if not mask[i, j]:
                    ax.text(j, i, f"{acc_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.set_xlabel("Evaluated on Task", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Trained up to Task", fontsize=FIGURE_FONT_SIZE)
    fig.tight_layout()
    return fig


def plot_forgetting_curve(
    acc_matrix: np.ndarray,
    task_names: list[str] | None = None,
    title: str = "Accuracy per Task over Training",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Trace l'évolution de l'accuracy de chaque tâche au fil des étapes d'entraînement.

    Pour chaque tâche j, trace acc_matrix[j:, j] — l'accuracy sur j après chaque
    entraînement suivant. Permet de visualiser l'oubli progressif (ou son absence).

    Parameters
    ----------
    acc_matrix : np.ndarray [T, T]
        Matrice CL (NaN au-dessus de la diagonale).
    task_names : list[str] | None
    title : str
    ax : plt.Axes | None

    Returns
    -------
    plt.Figure

    Notes
    -----
    Une courbe plate = pas d'oubli (idéal CL).
    Une courbe décroissante = oubli catastrophique.
    """
    n_tasks = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"T{i + 1}" for i in range(n_tasks)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, n_tasks * 1.5), 4))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for j in range(n_tasks):
        # Accuracy sur tâche j après chaque step i ≥ j
        steps = list(range(j, n_tasks))
        values = [acc_matrix[i, j] for i in steps]
        ax.plot(
            steps,
            values,
            marker="o",
            label=task_names[j],
            color=colors[j % len(colors)],
        )

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f"After T{i + 1}" for i in range(n_tasks)], fontsize=FIGURE_FONT_SIZE - 1)
    ax.set_xlabel("Training Step", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.legend(title="Task", fontsize=FIGURE_FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metrics_comparison(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "CL Metrics Comparison",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Barplot comparatif AA/AF/BWT pour plusieurs modèles.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Clés = noms de modèles (ex. "EWC", "HDC", "Fine-tuning").
        Valeurs = dicts de métriques issus de compute_cl_metrics().
        Ex. : {"EWC": {"aa": 0.98, "af": 0.001}, "HDC": {"aa": 0.95, "af": 0.0}}
    metrics : list[str] | None
        Métriques à afficher. Défaut : ["aa", "af", "bwt"].
    title : str
    ax : plt.Axes | None

    Returns
    -------
    plt.Figure

    Notes
    -----
    Utilisé dans notebooks/02_baseline_comparison.ipynb (S2-04).
    """
    if metrics is None:
        metrics = ["aa", "af", "bwt"]

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_metrics * n_models * 0.8), 4))
    else:
        fig = ax.get_figure()

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for k, model_name in enumerate(model_names):
        values = [results[model_name].get(m, float("nan")) for m in metrics]
        offset = (k - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, values, width, label=model_name, color=colors[k % len(colors)], alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Value", fontsize=FIGURE_FONT_SIZE)
    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.legend(fontsize=FIGURE_FONT_SIZE - 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = FIGURE_DPI,
) -> None:
    """
    Sauvegarde une figure matplotlib dans le dossier d'expérience.

    Crée le dossier parent si nécessaire. Format inféré depuis l'extension.

    Parameters
    ----------
    fig : plt.Figure
    path : str | Path
        Ex. "experiments/exp_002_hdc_dataset2/figures/accuracy_matrix.png"
    dpi : int
        Résolution. Défaut : 150 (suffisant pour rapport PDF).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[plots] Figure saved → {path}")
    plt.close(fig)
