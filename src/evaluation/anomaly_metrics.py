"""
anomaly_metrics.py — Métriques pour le scénario anomaly detection (one-class CL).

Complète metrics.py (qui calcule AA/AF/BWT sur acc_matrix) avec des métriques
adaptées à la détection d'anomalies : AUROC, AUPRC, F1, et leurs équivalents CL.

Usage :
    from src.evaluation.anomaly_metrics import compute_anomaly_metrics, compute_cl_anomaly_metrics

    metrics = compute_anomaly_metrics(y_true, scores)
    cl_metrics = compute_cl_anomaly_metrics(auroc_matrix)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Calcule les métriques de détection d'anomalies depuis les scores bruts.

    Parameters
    ----------
    y_true : np.ndarray [N], int
        Labels vrais : 0=normal, 1=anomalie/faulty.
    scores : np.ndarray [N], float
        Scores d'anomalie continus — plus élevé = plus anormal.
    threshold : float | None
        Seuil de décision binaire. Si None, calculé via la statistique J de Youden
        (maximise TPR - FPR sur la courbe ROC).

    Returns
    -------
    dict avec clés :
        auroc         : float — Area Under ROC Curve ∈ [0, 1]
        auprc         : float — Area Under Precision-Recall Curve ∈ [0, 1]
        f1            : float — F1-score binaire au seuil optimal
        precision     : float — Précision au seuil optimal
        recall        : float — Recall au seuil optimal
        threshold_used : float — Seuil appliqué pour les métriques binaires

    Notes
    -----
    Si y_true est constant (toutes les labels identiques), AUROC/AUPRC ne peuvent
    pas être calculés — retourne 0.5/nan et logue un avertissement.
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    scores = np.asarray(scores, dtype=np.float32).flatten()

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        import warnings
        warnings.warn(
            f"compute_anomaly_metrics : y_true constant (pos={n_pos}, neg={n_neg}). "
            "AUROC et AUPRC non définis.",
            stacklevel=2,
        )
        return {
            "auroc": 0.5,
            "auprc": float(n_pos / len(y_true)) if len(y_true) > 0 else 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold_used": float(threshold) if threshold is not None else 0.5,
        }

    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        youden_j = tpr - fpr
        best_idx = int(np.argmax(youden_j))
        threshold = float(thresholds[best_idx])

    y_pred = (scores >= threshold).astype(np.int64)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold_used": float(threshold),
    }


def compute_cl_anomaly_metrics(
    auroc_matrix: np.ndarray,
) -> dict:
    """
    Calcule les métriques CL adaptées pour une matrice AUROC (analogue de compute_cl_metrics).

    Parameters
    ----------
    auroc_matrix : np.ndarray [T, T]
        auroc_matrix[i, j] = AUROC sur la tâche j après entraînement sur les tâches 0..i.
        Convention : NaN pour j > i (tâche pas encore vue).

    Returns
    -------
    dict avec clés :
        avg_auroc         : float — AUROC moyen sur la ligne finale (après toutes les tâches)
        auroc_forgetting  : float ≥ 0 — chute moyenne d'AUROC entre le pic et la fin
        auroc_bwt         : float — transfert rétroactif (< 0 = oubli)
        forgetting_per_task : list[float]
        bwt_per_task        : list[float]
        auroc_matrix        : list[list] — sérialisable JSON (NaN → null)
        n_tasks             : int

    References
    ----------
    DeLange2021Survey §3 — taxonomie CL.
    """
    T = auroc_matrix.shape[0]
    final_row = auroc_matrix[T - 1, :T]
    avg_auroc = float(np.nanmean(final_row))

    forgetting_per_task: list[float] = []
    bwt_per_task: list[float] = []

    for j in range(T - 1):
        col = auroc_matrix[:j + 1, j]  # valeurs disponibles pour la tâche j
        peak = float(np.nanmax(col))
        final = float(auroc_matrix[T - 1, j])
        forgetting_per_task.append(peak - final)
        bwt_per_task.append(final - float(auroc_matrix[j, j]))

    auroc_forgetting = float(np.mean(forgetting_per_task)) if forgetting_per_task else 0.0
    auroc_bwt = float(np.mean(bwt_per_task)) if bwt_per_task else 0.0

    # Sérialisation JSON (NaN → null)
    matrix_serializable = [
        [None if np.isnan(v) else float(v) for v in row]
        for row in auroc_matrix
    ]

    return {
        "avg_auroc": avg_auroc,
        "auroc_forgetting": auroc_forgetting,
        "auroc_bwt": auroc_bwt,
        "forgetting_per_task": forgetting_per_task,
        "bwt_per_task": bwt_per_task,
        "auroc_matrix": matrix_serializable,
        "n_tasks": T,
    }


def save_anomaly_metrics(
    metrics: dict,
    output_path: str | Path,
    extra_info: dict | None = None,
) -> None:
    """
    Sauvegarde les métriques anomaly detection au format JSON.

    Parameters
    ----------
    metrics : dict
        Retour de compute_cl_anomaly_metrics() ou compute_anomaly_metrics().
    output_path : str | Path
        Chemin de destination.
    extra_info : dict | None
        Informations additionnelles (model_name, exp_id, etc.) fusionnées dans le JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metrics)
    if extra_info:
        payload.update(extra_info)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
