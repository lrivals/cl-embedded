"""
rul_metrics.py — Métriques d'évaluation pour la régression RUL.

Métriques implémentées :
    - RMSE (Root Mean Square Error) — métrique principale
    - MAE (Mean Absolute Error)
    - Horizon Score — pénalité asymétrique PHM 2008 (sur-estimation plus pénalisée)
    - Average Forgetting RMSE — dégradation RMSE entre pic et fin par tâche CL

Toutes les fonctions retournent des float ou dict[str, float] compatibles results.json.

Références :
    PHM 2008 Challenge — Horizon Score asymétrique
    DeLange2021Survey — définition Average Forgetting adaptée à la régression
"""

from __future__ import annotations

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Parameters
    ----------
    y_true : np.ndarray shape (N,)
        RUL réel.
    y_pred : np.ndarray shape (N,)
        RUL prédit.

    Returns
    -------
    float : RMSE ∈ [0, +∞)
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Returns
    -------
    float : MAE ∈ [0, +∞)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_horizon_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """
    Horizon Score (score PHM 2008) — pénalité asymétrique.

    La sur-estimation (ŷ > y_true, prédiction trop optimiste) est plus pénalisée
    que la sous-estimation (ŷ < y_true, prédiction conservative).

    Score = Σ exp(d / a_i) - 1   où d = ŷ - y_true,
            a1=13 si d < 0 (sous-estimation),
            a2=10 si d ≥ 0 (sur-estimation).

    Un score bas (proche de 0) est meilleur.

    Parameters
    ----------
    y_true, y_pred : np.ndarray shape (N,)
    a1 : float
        Facteur pour la sous-estimation. Default : 13 (PHM 2008).
    a2 : float
        Facteur pour la sur-estimation. Default : 10 (PHM 2008).

    Returns
    -------
    float : Horizon Score ∈ [0, +∞)
    """
    d = y_pred - y_true
    a = np.where(d < 0, a1, a2)
    return float(np.sum(np.exp(d / a) - 1))


def compute_rul_metrics_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcule toutes les métriques RUL pour une tâche.

    Returns
    -------
    dict avec clés : rmse, mae, horizon_score
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "horizon_score": compute_horizon_score(y_true, y_pred),
    }


def compute_avg_forgetting_rmse(
    task_rmse_matrix: list[list[float]],
) -> float:
    """
    Average Forgetting adapté à la régression (RMSE).

    AF_rmse = (1 / T-1) · Σ_{i=1}^{T-1} (RMSE_i_final - RMSE_i_best)

    Attention : pour RMSE, l'oubli = RMSE_final > RMSE_best (la valeur monte).
    AF > 0 signifie dégradation (oubli), AF < 0 signifie amélioration (transfert positif).

    Parameters
    ----------
    task_rmse_matrix : list[list[float]]
        task_rmse_matrix[epoch][task] = RMSE sur la tâche `task` après l'époque `epoch`.
        Shape : (n_epochs, n_tasks).

    Returns
    -------
    float : Average Forgetting RMSE
    """
    if len(task_rmse_matrix) < 2:
        return 0.0

    n_tasks = len(task_rmse_matrix[0])
    rmse_matrix = np.array(task_rmse_matrix)  # shape (n_epochs, n_tasks)

    forgettings = []
    for task_idx in range(n_tasks - 1):
        col = rmse_matrix[:, task_idx]
        best_rmse = col.min()  # meilleur RMSE atteint sur cette tâche
        final_rmse = col[-1]   # RMSE final (après toutes les tâches)
        forgettings.append(final_rmse - best_rmse)

    return float(np.mean(forgettings))
