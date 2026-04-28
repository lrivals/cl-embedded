"""
feature_importance.py — Contribution individuelle des variables pour les modèles CL embarqués.

Trois méthodes implémentées (sans dépendances lourdes, compatibles MCU-workflow) :
  1. permutation_importance  — modèle-agnostique (EWC, HDC, TinyOL)
  2. gradient_saliency       — spécifique PyTorch (EWC uniquement)
  3. feature_masking_importance — alternative à permutation pour HDC (zéro-masquage)

Références :
  - Breiman (2001) Random Forests : permutation importance originale
  - Simonyan et al. (2013) Deep Inside CNNs : gradient saliency
"""

from __future__ import annotations

from typing import Callable

import numpy as np

FEATURE_NAMES_MONITORING: list[str] = ["temperature", "pressure", "vibration", "humidity"]

from src.data.cwru_dataset import FEATURE_COLS as FEATURE_NAMES_CWRU
from src.data.pronostia_dataset import FEATURE_NAMES as FEATURE_NAMES_PRONOSTIA

CHANNEL_GROUPS_PRONOSTIA: dict[str, list[str]] = {
    "acc_horiz": [
        "mean_acc_horiz", "std_acc_horiz", "rms_acc_horiz",
        "kurtosis_acc_horiz", "peak_acc_horiz", "crest_factor_acc_horiz",
    ],
    "acc_vert": [
        "mean_acc_vert", "std_acc_vert", "rms_acc_vert",
        "kurtosis_acc_vert", "peak_acc_vert", "crest_factor_acc_vert",
    ],
    "temporal": ["temporal_position"],
}

_FIGURE_DPI: int = 150
_FONT_SIZE: int = 11


def resolve_feature_names(dataset: str) -> list[str]:
    """Return canonical feature names for a given dataset identifier.

    Parameters
    ----------
    dataset : str
        One of "monitoring", "cwru", "pronostia".

    Returns
    -------
    list[str]
        Ordered feature names for the requested dataset.

    Raises
    ------
    ValueError
        If `dataset` is not a known identifier.
    """
    _map: dict[str, list[str]] = {
        "monitoring": FEATURE_NAMES_MONITORING,
        "cwru": FEATURE_NAMES_CWRU,
        "pronostia": FEATURE_NAMES_PRONOSTIA,
    }
    if dataset not in _map:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(_map)}")
    return _map[dataset]


# ── 1. Permutation Importance ────────────────────────────────────────────────


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Importance de chaque feature par permutation aléatoire de sa colonne.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Fonction (X: [N, F]) -> scores [N]. Peut retourner des probabilités ou
        des labels binaires — la binarisation est appliquée avec `threshold`.
    X : np.ndarray [N, n_features]
        Données de test normalisées (float32).
    y : np.ndarray [N]
        Labels binaires 0/1.
    feature_names : list[str]
        Noms des features dans l'ordre des colonnes de X.
    n_repeats : int
        Nombre de permutations par feature. Défaut : 5.
    random_state : int
        Seed de base (seed + repeat pour chaque tirage). Défaut : 42.
    threshold : float
        Seuil de binarisation si predict_fn retourne des probabilités.

    Returns
    -------
    dict[str, float]
        {feature_name: importance_score} trié par importance décroissante.
        Score positif = la feature contribue (la perturber dégrade les perf).
        Score ≈ 0 = feature peu utile. Score négatif = artefact rare.
    """
    rng = np.random.default_rng(random_state)

    preds_base = predict_fn(X)
    acc_base = float(np.mean((preds_base >= threshold).astype(int) == y.astype(int)))

    importances: dict[str, float] = {}
    for j, feat in enumerate(feature_names):
        runs: list[float] = []
        for r in range(n_repeats):
            X_perm = X.copy()
            idx = rng.permutation(len(X_perm))
            X_perm[:, j] = X_perm[idx, j]
            preds_perm = predict_fn(X_perm)
            acc_perm = float(np.mean((preds_perm >= threshold).astype(int) == y.astype(int)))
            runs.append(acc_perm)
        importances[feat] = acc_base - float(np.mean(runs))

    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


def permutation_importance_per_task(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    tasks: list[dict],
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Lance permutation_importance sur chaque tâche séparément.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Même interface que permutation_importance.
    tasks : list[dict]
        [{"task_name": str, "X": np.ndarray, "y": np.ndarray}, ...]
    feature_names : list[str]
    n_repeats : int
    random_state : int
    threshold : float

    Returns
    -------
    dict[str, dict[str, float]]
        {task_name: {feature_name: score}}
    """
    results: dict[str, dict[str, float]] = {}
    for i, task in enumerate(tasks):
        name = task.get("task_name", f"task_{i}")
        results[name] = permutation_importance(
            predict_fn,
            task["X"],
            task["y"],
            feature_names,
            n_repeats=n_repeats,
            random_state=random_state,
            threshold=threshold,
        )
    return results


# ── 2. Gradient Saliency (PyTorch / EWC uniquement) ─────────────────────────


def gradient_saliency(
    model: "torch.nn.Module",  # noqa: F821
    X: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Saliency par gradient : importance[j] = mean_i |∂ŷ_i / ∂x_ij|.

    Applicable uniquement aux modèles PyTorch (EWCMlpClassifier).
    Ne nécessite pas de permutation aléatoire → reproductible et rapide.

    Parameters
    ----------
    model : torch.nn.Module
        Modèle PyTorch en mode eval(). Doit retourner un scalaire par échantillon.
    X : np.ndarray [N, n_features], dtype float32
        Données de test normalisées.
    feature_names : list[str]
        Noms des features (même ordre que colonnes de X).

    Returns
    -------
    dict[str, float]
        {feature_name: mean_abs_gradient} — plus grand = plus influent localement.

    Notes
    -----
    Mesure la sensibilité locale du modèle, pas l'importance causale.
    Peut sous-estimer des features importantes dont le gradient sature (ReLU morte).
    """
    import torch

    model.eval()
    all_grads: list[np.ndarray] = []

    for xi in X:
        x_t = torch.from_numpy(xi.astype(np.float32)).unsqueeze(0)
        x_t.requires_grad_(True)
        output = model(x_t)
        output.squeeze().backward()
        all_grads.append(x_t.grad.detach().numpy().flatten().__abs__())

    mean_grads = np.mean(all_grads, axis=0)
    return {feat: float(mean_grads[j]) for j, feat in enumerate(feature_names)}


# ── 3. Feature Masking (HDC / agnostique) ───────────────────────────────────


def feature_masking_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    mask_value: float = 0.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Importance par masquage : remplace la colonne j par `mask_value` (ex. 0 = moyenne Z-score).

    Utile pour HDC (pas de gradient) comme alternative déterministe à la permutation.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Même interface que permutation_importance.
    X : np.ndarray [N, n_features]
    y : np.ndarray [N]
    feature_names : list[str]
    mask_value : float
        Valeur de remplacement (0.0 = moyenne si données Z-score normalisées).
    threshold : float

    Returns
    -------
    dict[str, float]
        {feature_name: importance_score} trié par importance décroissante.
    """
    preds_base = predict_fn(X)
    acc_base = float(np.mean((preds_base >= threshold).astype(int) == y.astype(int)))

    importances: dict[str, float] = {}
    for j, feat in enumerate(feature_names):
        X_masked = X.copy()
        X_masked[:, j] = mask_value
        preds_masked = predict_fn(X_masked)
        acc_masked = float(np.mean((preds_masked >= threshold).astype(int) == y.astype(int)))
        importances[feat] = acc_base - acc_masked

    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


# ── 4. Visualisation ─────────────────────────────────────────────────────────


def plot_feature_importance(
    importances: dict[str, float],
    title: str = "Importance des variables",
    color: str = "#2196F3",
    ax: "plt.Axes | None" = None,
    show_values: bool = True,
) -> "plt.Figure":
    """
    Barplot horizontal de l'importance des features.

    Parameters
    ----------
    importances : dict[str, float]
        Sortie de permutation_importance(), gradient_saliency() ou feature_masking_importance().
    title : str
    color : str
        Couleur par défaut (ignorée si score négatif → rouge).
    ax : plt.Axes | None
        Axes existants, ou None pour créer une nouvelle figure.
    show_values : bool
        Afficher la valeur numérique à côté de chaque barre.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt

    features = list(importances.keys())
    scores = list(importances.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.9)), dpi=_FIGURE_DPI)
    else:
        fig = ax.get_figure()

    bar_colors = ["#4CAF50" if s >= 0 else "#F44336" for s in scores]
    bars = ax.barh(features[::-1], scores[::-1], color=bar_colors[::-1], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    if show_values and scores:
        max_abs = max(abs(s) for s in scores) if scores else 1.0
        offset = max_abs * 0.03
        for bar, val in zip(bars, scores[::-1]):
            ax.text(
                bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center",
                fontsize=_FONT_SIZE - 2,
            )

    ax.set_xlabel("Importance (chute d'accuracy après perturbation)", fontsize=_FONT_SIZE)
    ax.set_title(title, fontsize=_FONT_SIZE + 1, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance_comparison(
    importances_dict: dict[str, dict[str, float]],
    feature_names: list[str],
    title: str = "Comparaison des importances par méthode",
) -> "plt.Figure":
    """
    Barplot groupé comparant plusieurs méthodes d'importance sur les mêmes features.

    Parameters
    ----------
    importances_dict : dict[str, dict[str, float]]
        {method_name: {feature_name: score}}
        Ex. : {"Permutation": {...}, "Gradient saliency": {...}}
    feature_names : list[str]
        Ordre des features sur l'axe x.
    title : str

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(importances_dict.keys())
    n_methods = len(methods)
    n_features = len(feature_names)
    x = np.arange(n_features)
    width = 0.8 / n_methods
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(max(7, n_features * 2.0), 4), dpi=_FIGURE_DPI)

    for k, method in enumerate(methods):
        scores = [importances_dict[method].get(f, 0.0) for f in feature_names]
        offset = (k - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            scores,
            width,
            label=method,
            color=colors[k % len(colors)],
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=_FONT_SIZE)
    ax.set_ylabel("Importance", fontsize=_FONT_SIZE)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=_FONT_SIZE + 1, fontweight="bold")
    ax.legend(fontsize=_FONT_SIZE - 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
