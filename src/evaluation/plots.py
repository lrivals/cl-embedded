"""Visualisation des résultats CL : accuracy matrix, forgetting curves, comparaison.

Contrainte : ce module ne doit être importé que depuis src/evaluation/ et notebooks/.
Ne jamais importer matplotlib/seaborn dans les modules modèles ou d'entraînement.
"""

from __future__ import annotations

import math
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


def plot_confusion_matrix_grid(
    preds_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    task_names: list[str] | None = None,
    model_name: str = "Modèle",
    figsize: tuple[float, float] | None = None,
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Grille de matrices de confusion — une colonne par tâche évaluée, une ligne par step d'entraînement.

    Visualise comment la performance de classification évolue tâche par tâche.
    Cases (i, j) avec j > i sont affichées grisées ("N/A" — tâche pas encore vue).
    Chaque matrice est normalisée par ligne (recall par classe).

    Parameters
    ----------
    preds_dict : dict[tuple[int,int], tuple[np.ndarray, np.ndarray]]
        Sortie de run_cl_scenario_full() : preds_dict[(i,j)] = (y_true, y_pred).
        y_pred peut être des probabilités [0,1] ou des labels binaires {0,1}.
    task_names : list[str] | None
        Noms des tâches. Défaut : ["T1", "T2", ...].
    model_name : str
        Nom du modèle pour le titre.
    figsize : tuple[float, float] | None
        Taille de la figure. Défaut : (4*T, 3.5*T).
    threshold : float
        Seuil de binarisation si y_pred est en probabilités. Défaut : 0.5.

    Returns
    -------
    plt.Figure
    """
    try:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score
    except ImportError as e:
        raise ImportError("sklearn requis pour plot_confusion_matrix_grid()") from e

    # Déterminer T depuis les clés disponibles
    all_i = [k[0] for k in preds_dict]
    all_j = [k[1] for k in preds_dict]
    T = max(max(all_i), max(all_j)) + 1

    if task_names is None:
        task_names = [f"T{t + 1}" for t in range(T)]

    if figsize is None:
        figsize = (4.0 * T, 3.5 * T)

    fig, axes = plt.subplots(T, T, figsize=figsize)
    if T == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(T, T)

    class_labels = ["Normal", "Faulty"]

    for i in range(T):
        for j in range(T):
            ax = axes[i, j]
            if j > i or (i, j) not in preds_dict:
                # Tâche pas encore vue — cellule grisée
                ax.set_facecolor("#EEEEEE")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=14, color="#AAAAAA", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            y_true, y_pred_raw = preds_dict[(i, j)]
            y_pred = (y_pred_raw >= threshold).astype(int)

            cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
            # Normalisation par ligne (recall par classe)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_norm = cm / row_sums

            if _HAS_SEABORN:
                sns.heatmap(
                    cm_norm,
                    ax=ax,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    vmin=0.0,
                    vmax=1.0,
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    cbar=False,
                    linewidths=0.5,
                )
            else:
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
                for ri in range(2):
                    for ci in range(2):
                        ax.text(ci, ri, f"{cm_norm[ri, ci]:.2f}", ha="center", va="center", fontsize=9)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(class_labels, fontsize=8)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(class_labels, fontsize=8)

            prec = precision_score(y_true.astype(int), y_pred, zero_division=0)
            rec = recall_score(y_true.astype(int), y_pred, zero_division=0)
            ax.set_title(
                f"Train→T{i + 1} | Eval T{j + 1}\n"
                f"P={prec:.2f}  R={rec:.2f}",
                fontsize=FIGURE_FONT_SIZE - 1,
            )
            ax.set_xlabel("Prédit", fontsize=FIGURE_FONT_SIZE - 2)
            ax.set_ylabel("Réel", fontsize=FIGURE_FONT_SIZE - 2)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Matrices de confusion — {model_name}\n(normalisées par ligne | seuil={threshold})",
        fontsize=FIGURE_FONT_SIZE + 1,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_roc_curves_per_task(
    preds_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    scores_dict: dict[tuple[int, int], np.ndarray] | None = None,
    task_names: list[str] | None = None,
    model_name: str = "Modèle",
) -> plt.Figure:
    """
    Courbes ROC par tâche évaluée, avec une courbe par step d'entraînement.

    Pour chaque tâche j, affiche les courbes ROC des évaluations à chaque step i ≥ j.
    Particulièrement utile pour les méthodes non-supervisées (anomaly score continu).

    Parameters
    ----------
    preds_dict : dict[tuple[int,int], tuple[np.ndarray, np.ndarray]]
        Sortie de run_cl_scenario_full() : preds_dict[(i,j)] = (y_true, y_pred).
    scores_dict : dict | None
        Scores continus (probabilités ou anomaly scores) par (i,j). Si None,
        y_pred de preds_dict est utilisé directement.
    task_names : list[str] | None
    model_name : str

    Returns
    -------
    plt.Figure
    """
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as e:
        raise ImportError("sklearn requis pour plot_roc_curves_per_task()") from e

    all_j_keys = sorted({k[1] for k in preds_dict})
    T = max({k[0] for k in preds_dict}) + 1

    if task_names is None:
        task_names = [f"T{t + 1}" for t in range(T)]

    n_tasks_eval = len(all_j_keys)
    fig, axes = plt.subplots(1, n_tasks_eval, figsize=(5.0 * n_tasks_eval, 4.5), squeeze=False)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for col_idx, j in enumerate(all_j_keys):
        ax = axes[0, col_idx]
        # Diagonale : chance
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance (AUC=0.50)")

        for i in range(j, T):
            if (i, j) not in preds_dict:
                continue
            y_true, y_pred_raw = preds_dict[(i, j)]
            scores = scores_dict[(i, j)] if (scores_dict and (i, j) in scores_dict) else y_pred_raw

            # Vérifier que les deux classes sont présentes (sinon ROC indéfinie)
            if len(np.unique(y_true.astype(int))) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true.astype(int), scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linewidth=1.8,
                label=f"Après T{i + 1} (AUC={roc_auc:.3f})",
            )

        ax.set_xlabel("Taux faux positifs (FPR)", fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("Taux vrais positifs (TPR)", fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_title(f"ROC — Eval {task_names[j]}", fontsize=FIGURE_FONT_SIZE)
        ax.legend(fontsize=FIGURE_FONT_SIZE - 2, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        f"Courbes ROC par tâche — {model_name}",
        fontsize=FIGURE_FONT_SIZE + 1,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_model_radar(
    results: dict[str, dict],
    ram_budget_bytes: float = 65536.0,
    latency_budget_ms: float = 100.0,
) -> plt.Figure:
    """
    Radar/spider chart comparant les modèles sur 5 axes normalisés.

    Axes :
    - AA       : Average Accuracy (plus grand = meilleur)
    - Stabilité : 1 - AF (Average Forgetting)
    - BWT+     : 1 - |BWT| (transfert bidirectionnel neutre = bon)
    - RAM      : 1 - ram_peak / ram_budget
    - Vitesse  : 1 - latency / latency_budget

    Parameters
    ----------
    results : dict[str, dict]
        Clés = noms de modèles.
        Valeurs = dicts avec clés :
            aa (float), af (float), bwt (float),
            ram_peak_bytes (int | float), inference_latency_ms (float).
    ram_budget_bytes : float
        Budget RAM de référence (défaut : 65 536 = 64 Ko STM32N6).
    latency_budget_ms : float
        Budget latence de référence (défaut : 100 ms).

    Returns
    -------
    plt.Figure
    """
    AXES_LABELS = ["AA", "Stabilité\n(1−AF)", "BWT\n(1−|BWT|)", "RAM\n(1−peak/budget)", "Vitesse\n(1−lat/budget)"]
    N_AXES = len(AXES_LABELS)
    angles = np.linspace(0, 2 * np.pi, N_AXES, endpoint=False).tolist()
    angles += angles[:1]  # fermeture du polygone

    COLORS_RADAR = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for k, (model_name, m) in enumerate(results.items()):
        aa = float(np.clip(m.get("aa", 0.0), 0, 1))
        af = float(m.get("af", 0.0))
        bwt = float(m.get("bwt", 0.0))
        ram = float(m.get("ram_peak_bytes", 0.0))
        lat = float(m.get("inference_latency_ms", 0.0))

        values = [
            aa,
            float(np.clip(1.0 - af, 0, 1)),
            float(np.clip(1.0 - abs(bwt), 0, 1)),
            float(np.clip(1.0 - ram / ram_budget_bytes, 0, 1)),
            float(np.clip(1.0 - lat / latency_budget_ms, 0, 1)),
        ]
        values += values[:1]

        color = COLORS_RADAR[k % len(COLORS_RADAR)]
        ax.plot(angles, values, color=color, linewidth=2, label=model_name)
        ax.fill(angles, values, color=color, alpha=0.12)

    # Grille et labels
    ax.set_thetagrids(np.degrees(angles[:-1]), AXES_LABELS, fontsize=FIGURE_FONT_SIZE - 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
    ax.grid(True, alpha=0.4)
    ax.set_title(
        "Comparaison multi-critères des modèles CL\n(normalisé — plus grand = meilleur)",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        pad=18,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=FIGURE_FONT_SIZE - 1)
    fig.tight_layout()
    return fig


def plot_anomaly_score_distributions(
    scores_by_task: dict[str, list[np.ndarray]],
    labels_by_task: list[np.ndarray],
    task_names: list[str] | None = None,
) -> plt.Figure:
    """
    Distribution des scores d'anomalie par classe (normal/faulty) et par tâche.

    Visualise la séparabilité : un bon détecteur produit des distributions bien séparées.
    Utilise des KDE + violin plots superposés pour chaque modèle et chaque tâche.

    Parameters
    ----------
    scores_by_task : dict[str, list[np.ndarray]]
        Clés = noms de modèles (ex. "K-Means", "Mahalanobis").
        Valeurs = liste de T np.ndarray [N_t] — scores par tâche.
    labels_by_task : list[np.ndarray]
        Liste de T np.ndarray [N_t] — vraies étiquettes {0, 1} par tâche.
    task_names : list[str] | None
        Noms des tâches. Défaut : ["T1", "T2", ...].

    Returns
    -------
    plt.Figure

    Notes
    -----
    Les scores sont normalisés min-max par modèle+tâche pour permettre la comparaison visuelle.
    Un score élevé = plus anormal (inversion appliquée si nécessaire pour les modèles dont
    score faible = anomalie, ex. PCA reconstruction error inversée).
    """
    model_names = list(scores_by_task.keys())
    n_models = len(model_names)
    T = max(len(v) for v in scores_by_task.values())

    if task_names is None:
        task_names = [f"T{t + 1}" for t in range(T)]

    fig, axes = plt.subplots(n_models, T, figsize=(4.5 * T, 3.5 * n_models), squeeze=False)

    COLORS_CLASS = {0: "#4CAF50", 1: "#F44336"}
    CLASS_NAMES = {0: "Normal", 1: "Faulty"}

    for row, model_name in enumerate(model_names):
        task_scores = scores_by_task[model_name]
        for col in range(T):
            ax = axes[row, col]
            if col >= len(task_scores):
                ax.axis("off")
                continue

            scores = np.asarray(task_scores[col]).flatten()
            y_true = np.asarray(labels_by_task[col]).flatten().astype(int)

            # Normalisation min-max pour l'affichage comparatif
            s_min, s_max = scores.min(), scores.max()
            s_range = s_max - s_min if s_max > s_min else 1.0
            scores_norm = (scores - s_min) / s_range

            for label_val in [0, 1]:
                mask = y_true == label_val
                if mask.sum() < 2:
                    continue
                s_cls = scores_norm[mask]
                color = COLORS_CLASS[label_val]
                name = CLASS_NAMES[label_val]

                # KDE
                try:
                    from scipy.stats import gaussian_kde
                    x_kde = np.linspace(0, 1, 200)
                    kde = gaussian_kde(s_cls, bw_method="silverman")
                    ax.fill_between(x_kde, kde(x_kde), alpha=0.35, color=color, label=name)
                    ax.plot(x_kde, kde(x_kde), color=color, linewidth=1.5)
                except ImportError:
                    # Fallback histogramme si scipy absent
                    ax.hist(s_cls, bins=30, alpha=0.5, color=color, label=name, density=True)

            ax.set_title(f"{task_names[col]}", fontsize=FIGURE_FONT_SIZE)
            ax.set_xlabel("Score normalisé", fontsize=FIGURE_FONT_SIZE - 2)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=FIGURE_FONT_SIZE - 1)
            ax.legend(fontsize=FIGURE_FONT_SIZE - 3, loc="upper right")
            ax.set_xlim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Distribution des scores d'anomalie par classe et par tâche\n"
        "(séparation normale/faulty — score normalisé [0,1])",
        fontsize=FIGURE_FONT_SIZE + 1,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_performance_by_pump_id_bar(
    results: dict[str, dict],
    pump_ids: list[int],
    title: str = "Accuracy finale par Pump_ID",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Barplot groupé : accuracy finale par Pump_ID, une barre par modèle.

    Révèle quels Pump_ID sont les plus difficiles à retenir après entraînement
    complet sur les 5 tâches du scénario domain-incremental par pompe.

    Parameters
    ----------
    results : dict[str, dict]
        Clés = noms de modèles, valeurs = dict ``{pump_id (int): accuracy (float)}``.
        Ex. ``{"EWC": {1: 0.7, 2: 0.6, ...}, "HDC": {1: 0.6, ...}}``.
    pump_ids : list[int]
        Liste ordonnée des Pump_IDs à afficher (ex. ``[1, 2, 3, 4, 5]``).
    title : str
        Titre de la figure.
    ax : plt.Axes | None
        Axes existants (si None, une nouvelle figure est créée).

    Returns
    -------
    plt.Figure
        Figure matplotlib prête pour ``save_figure()``.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_pumps = len(pump_ids)

    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(n_pumps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_pumps * 1.4), 4), dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 1)))

    for i, (model_name, pump_accs) in enumerate(results.items()):
        offsets = x + (i - n_models / 2 + 0.5) * bar_width
        heights = [pump_accs.get(pid, 0.0) for pid in pump_ids]
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=model_name,
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Pump {pid}" for pid in pump_ids], fontsize=FIGURE_FONT_SIZE - 1)
    ax.set_ylabel("Accuracy finale", fontsize=FIGURE_FONT_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=FIGURE_FONT_SIZE + 1, fontweight="bold")
    ax.legend(fontsize=FIGURE_FONT_SIZE - 1, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_performance_heatmap_equipment_location(
    results_by_cell: dict[str, dict[tuple[str, str], float]],
    equipment_types: list[str],
    locations: list[str],
    title: str = "Accuracy par équipement × location",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Heatmap ``n_equipment × n_locations`` de l'accuracy finale par sous-groupe.

    Révèle quels croisements (type d'équipement, site géographique) sont
    systématiquement difficiles à détecter après entraînement CL complet.
    Une grille 2×2 est produite pour 4 modèles (ou 1×n_models sinon).

    Parameters
    ----------
    results_by_cell : dict[str, dict[tuple[str, str], float]]
        Clés = noms de modèles (ex. ``"EWC"``, ``"HDC"``).
        Valeurs = dicts ``{(equipment, location): accuracy}`` — les paires
        manquantes sont affichées comme NaN (cellule grisée).
        Ex. : ``{"EWC": {("Pump", "Atlanta"): 0.95, ("Turbine", "Chicago"): 0.88}}``.
    equipment_types : list[str]
        Axes des lignes. Ex. ``["Pump", "Turbine", "Compressor"]``.
    locations : list[str]
        Axes des colonnes. Ex. ``["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]``.
    title : str
        Titre global de la figure.
    figsize : tuple[int, int]
        Taille de la figure totale en pouces.

    Returns
    -------
    plt.Figure
        Figure avec une heatmap par modèle (disposition 2×2 pour 4 modèles,
        1×n_models pour un nombre différent).

    Notes
    -----
    Référence : scénario S5-19, docs/sprints/sprint_5/S519_monitoring_granular_experiments.md
    """
    model_names = list(results_by_cell.keys())
    n_models = len(model_names)
    n_eq = len(equipment_types)
    n_loc = len(locations)

    # Disposition des sous-figures : 2 colonnes pour ≥ 2 modèles
    ncols = min(2, n_models)
    nrows = math.ceil(n_models / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # Normaliser axes en tableau 2D
    if n_models == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes]) if ncols > 1 else np.array([[axes]])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        cell_data = results_by_cell[model_name]

        # Construire la matrice equipment × location (NaN si cellule absente)
        matrix = np.full((n_eq, n_loc), np.nan)
        for ei, eq in enumerate(equipment_types):
            for li, loc in enumerate(locations):
                val = cell_data.get((eq, loc))
                if val is not None:
                    matrix[ei, li] = float(val)

        if _HAS_SEABORN:
            cmap = sns.color_palette(CMAP_ACCURACY, as_cmap=True)
            cmap.set_bad("lightgrey")
            sns.heatmap(
                matrix,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                xticklabels=locations,
                yticklabels=equipment_types,
                linewidths=0.5,
                linecolor="white",
                mask=np.isnan(matrix),
            )
        else:
            masked = np.ma.masked_invalid(matrix)
            cmap_obj = plt.get_cmap(CMAP_ACCURACY)
            cmap_obj.set_bad("lightgrey")
            im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap_obj, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n_loc))
            ax.set_xticklabels(locations, fontsize=FIGURE_FONT_SIZE - 2, rotation=30, ha="right")
            ax.set_yticks(range(n_eq))
            ax.set_yticklabels(equipment_types, fontsize=FIGURE_FONT_SIZE - 1)
            for ei in range(n_eq):
                for li in range(n_loc):
                    if not np.isnan(matrix[ei, li]):
                        ax.text(li, ei, f"{matrix[ei, li]:.2f}", ha="center", va="center", fontsize=9)

        ax.set_title(model_name, fontsize=FIGURE_FONT_SIZE + 1, fontweight="bold")
        ax.set_xlabel("Location", fontsize=FIGURE_FONT_SIZE - 1)
        ax.set_ylabel("Equipment", fontsize=FIGURE_FONT_SIZE - 1)
        ax.tick_params(axis="x", labelsize=FIGURE_FONT_SIZE - 2, rotation=30)
        ax.tick_params(axis="y", labelsize=FIGURE_FONT_SIZE - 1)

    # Masquer les axes vides si n_models < nrows * ncols
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=FIGURE_FONT_SIZE + 2, fontweight="bold")
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
