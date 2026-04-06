# S2-06 — Visualisation accuracy matrix (heatmap forgetting)

| Champ | Valeur |
|-------|--------|
| **ID** | S2-06 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S1-07 (`metrics.py`), S2-03 (`exp_002` — pour données de démo), S2-05 (`scenarios.py` — fournit les `acc_matrix`) |
| **Fichiers cibles** | `src/evaluation/plots.py` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Fournir des fonctions de visualisation pour l'analyse des résultats CL. Le module `plots.py` consomme les `acc_matrix [T, T]` produites par `scenarios.py` et `baselines.py`, et génère :

1. **Heatmap** de la matrice d'accuracy (visualise l'oubli par case)
2. **Courbes de forgetting** (accuracy par tâche au fil des étapes d'entraînement)
3. **Barplot comparatif** (AA/AF/BWT pour plusieurs modèles côte à côte)

**Contrainte importante** : matplotlib et seaborn sont des dépendances de visualisation exclusivement tolérées dans `src/evaluation/` et `notebooks/`. Ils ne doivent jamais être importés dans les modules modèles ou d'entraînement.

**Critère de succès** : `python -c "from src.evaluation.plots import plot_accuracy_matrix"` passe, et chaque fonction génère un fichier PNG valide dans un `tmp_path` de test.

---

## Sous-tâches

### 1. Imports et configuration

```python
# src/evaluation/plots.py

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
```

### 2. Heatmap de la matrice d'accuracy

```python
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
    T = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"T{i + 1}" for i in range(T)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, T * 1.5), max(3, T * 1.2)))
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
            yticklabels=[f"After T{i + 1}" for i in range(T)],
            linewidths=0.5,
            linecolor="white",
            mask=mask,
        )
    else:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=CMAP_ACCURACY, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(T))
        ax.set_xticklabels(task_names)
        ax.set_yticks(range(T))
        ax.set_yticklabels([f"After T{i + 1}" for i in range(T)])
        for i in range(T):
            for j in range(T):
                if not mask[i, j]:
                    ax.text(j, i, f"{acc_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.set_xlabel("Evaluated on Task", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Trained up to Task", fontsize=FIGURE_FONT_SIZE)
    fig.tight_layout()
    return fig
```

### 3. Courbes de forgetting par tâche

```python
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
    T = acc_matrix.shape[0]
    if task_names is None:
        task_names = [f"T{i + 1}" for i in range(T)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, T * 1.5), 4))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for j in range(T):
        # Accuracy sur tâche j après chaque step i ≥ j
        steps = list(range(j, T))
        values = [acc_matrix[i, j] for i in steps]
        ax.plot(
            steps,
            values,
            marker="o",
            label=task_names[j],
            color=colors[j % len(colors)],
        )

    ax.set_xticks(range(T))
    ax.set_xticklabels([f"After T{i + 1}" for i in range(T)], fontsize=FIGURE_FONT_SIZE - 1)
    ax.set_xlabel("Training Step", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.legend(title="Task", fontsize=FIGURE_FONT_SIZE - 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
```

### 4. Barplot comparatif multi-modèles

```python
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
        ax.bar(x + offset, values, width, label=model_name, color=colors[k % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=FIGURE_FONT_SIZE)
    ax.set_ylabel("Value", fontsize=FIGURE_FONT_SIZE)
    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1)
    ax.legend(fontsize=FIGURE_FONT_SIZE - 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
```

### 5. Sauvegarde cohérente

```python
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
```

### 6. Écrire les tests

```python
# tests/test_plots.py
import numpy as np
import pytest
from pathlib import Path
from src.evaluation.plots import (
    plot_accuracy_matrix,
    plot_forgetting_curve,
    plot_metrics_comparison,
    save_figure,
)

@pytest.fixture
def sample_acc_matrix() -> np.ndarray:
    return np.array([
        [0.91, np.nan, np.nan],
        [0.88, 0.85,  np.nan],
        [0.86, 0.83,  0.89 ],
    ])

def test_plot_accuracy_matrix_returns_figure(sample_acc_matrix):
    import matplotlib.pyplot as plt
    fig = plot_accuracy_matrix(sample_acc_matrix, task_names=["Pump", "Turbine", "Compressor"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_forgetting_curve_returns_figure(sample_acc_matrix):
    import matplotlib.pyplot as plt
    fig = plot_forgetting_curve(sample_acc_matrix)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_metrics_comparison_returns_figure():
    import matplotlib.pyplot as plt
    results = {
        "EWC": {"aa": 0.98, "af": 0.001, "bwt": 0.0},
        "HDC": {"aa": 0.95, "af": 0.0, "bwt": 0.002},
    }
    fig = plot_metrics_comparison(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_save_figure_creates_file(sample_acc_matrix, tmp_path):
    fig = plot_accuracy_matrix(sample_acc_matrix)
    out = tmp_path / "figures" / "test_matrix.png"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0
```

---

## Critères d'acceptation

- [x] `from src.evaluation.plots import plot_accuracy_matrix, plot_forgetting_curve, plot_metrics_comparison, save_figure` — aucune erreur d'import
- [x] `plot_accuracy_matrix()` génère une figure avec NaN affichés en gris
- [x] `plot_forgetting_curve()` trace une courbe par tâche, accuracy ∈ [0, 1]
- [x] `plot_metrics_comparison()` fonctionne avec 2+ modèles
- [x] `save_figure()` crée le dossier parent si nécessaire et écrit un PNG non vide
- [x] `pytest tests/test_plots.py -v` — tous les tests passent (4/4)
- [x] Fonctionne sans seaborn (fallback matplotlib pur)
- [x] `ruff check src/evaluation/plots.py` + `black --check` passent

---

## Interface attendue par `notebooks/02_baseline_comparison.ipynb` (S2-04)

```python
from src.evaluation.plots import (
    plot_accuracy_matrix,
    plot_forgetting_curve,
    plot_metrics_comparison,
    save_figure,
)
from src.evaluation.metrics import compute_cl_metrics

# Générer les figures
fig_hm = plot_accuracy_matrix(acc_hdc, task_names=["Pump", "Turbine", "Compressor"],
                               title="HDC — Accuracy Matrix")
save_figure(fig_hm, "experiments/exp_002_hdc_dataset2/figures/accuracy_matrix_hdc.png")

fig_cmp = plot_metrics_comparison(
    results={
        "EWC": compute_cl_metrics(acc_ewc),
        "HDC": compute_cl_metrics(acc_hdc),
        "Fine-tuning": compute_cl_metrics(acc_naive),
    }
)
save_figure(fig_cmp, "experiments/exp_002_hdc_dataset2/figures/metrics_comparison.png")
```

---

## Questions ouvertes

- `TODO(arnaud)` : faut-il ajouter une visualisation de la RAM peak par modèle (barplot `ram_peak_bytes`) pour le rapport final ?
- `TODO(arnaud)` : stocker les figures en SVG (vectoriel) en plus du PNG pour le manuscrit ?
