"""
multiclass_metrics.py — Métriques d'évaluation pour la classification multi-classe CL.

Métriques implémentées :
    - F1-macro (moyenne non pondérée du F1 par classe)
    - Matrice de confusion
    - Précision par classe
    - Average Forgetting F1-macro

Références :
    DeLange2021Survey — Average Forgetting pour CL classification
    scikit-learn : f1_score(average='macro')
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """
    F1-macro (moyenne non pondérée sur toutes les classes).

    Parameters
    ----------
    y_true : np.ndarray shape (N,), int
    y_pred : np.ndarray shape (N,), int
    labels : liste des classes attendues (utile pour CWRU avec classes absentes)

    Returns
    -------
    float : F1-macro ∈ [0, 1]
    """
    return float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    """
    Matrice de confusion (shape n_classes × n_classes).

    Returns
    -------
    np.ndarray dtype int, shape (n_classes, n_classes)
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """
    Précision par classe (accuracy individuelle).

    Returns
    -------
    dict : {"class_0": float, "class_1": float, ...}
    """
    result = {}
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            result[f"class_{cls}"] = float("nan")
        else:
            result[f"class_{cls}"] = float((y_pred[mask] == cls).mean())
    return result


def compute_multiclass_metrics_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    labels: list[int] | None = None,
) -> dict[str, float | list]:
    """
    Calcule toutes les métriques multi-classe pour une tâche.

    Returns
    -------
    dict avec clés : f1_macro, confusion_matrix (liste 2D), per_class_accuracy (dict)
    """
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "f1_macro": compute_f1_macro(y_true, y_pred, labels=labels),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": compute_per_class_accuracy(y_true, y_pred, n_classes),
    }


def compute_avg_forgetting_f1(
    task_f1_matrix: list[list[float]],
) -> float:
    """
    Average Forgetting F1-macro.

    AF_f1 = (1 / T-1) · Σ_{i=1}^{T-1} (F1_i_peak - F1_i_final)

    AF > 0 signifie oubli, AF < 0 signifie backward transfer positif.

    Parameters
    ----------
    task_f1_matrix : list[list[float]]
        task_f1_matrix[epoch][task] = F1-macro sur la tâche `task` après l'époque `epoch`.

    Returns
    -------
    float : Average Forgetting F1-macro
    """
    if len(task_f1_matrix) < 2:
        return 0.0

    f1_matrix = np.array(task_f1_matrix)  # shape (n_epochs, n_tasks)
    n_tasks = f1_matrix.shape[1]

    forgettings = []
    for task_idx in range(n_tasks - 1):
        col = f1_matrix[:, task_idx]
        peak_f1 = col.max()
        final_f1 = col[-1]
        forgettings.append(peak_f1 - final_f1)

    return float(np.mean(forgettings))
