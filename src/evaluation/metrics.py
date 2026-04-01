"""
metrics.py — Métriques standard pour l'évaluation du Continual Learning.

Métriques implémentées :
    - AA  : Average Accuracy
    - AF  : Average Forgetting
    - BWT : Backward Transfer
    - FWT : Forward Transfer (optionnel)

Références :
    Lopez-Paz & Ranzato (2017). Gradient Episodic Memory for CL. NeurIPS.
    De Lange et al. (2021). A CL Survey. TPAMI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def compute_cl_metrics(
    acc_matrix: np.ndarray,
    random_baseline: Optional[np.ndarray] = None,
) -> dict:
    """
    Calcule les métriques CL standard depuis la matrice d'accuracy.

    Parameters
    ----------
    acc_matrix : np.ndarray [T, T]
        acc_matrix[i, j] = accuracy du modèle après l'entraînement sur
        la tâche i, évaluée sur la tâche j.
        Convention : acc_matrix[i, j] = NaN pour j > i (tâche pas encore vue).
    random_baseline : np.ndarray [T], optional
        Accuracy d'un classifieur aléatoire par tâche (pour FWT).
        Default : 1/n_classes pour chaque tâche (supposé = 0.5 si non fourni).

    Returns
    -------
    dict avec clés :
        aa   : Average Accuracy (float)
        af   : Average Forgetting (float, ≥ 0 = oubli, < 0 = amélioration rétroactive)
        bwt  : Backward Transfer (float, < 0 = oubli)
        fwt  : Forward Transfer (float, optionnel)
        acc_matrix : liste 2D sérialisable JSON

    Examples
    --------
    >>> M = np.array([
    ...     [0.91, np.nan, np.nan],
    ...     [0.88, 0.85,  np.nan],
    ...     [0.86, 0.83,  0.89 ],
    ... ])
    >>> metrics = compute_cl_metrics(M)
    >>> print(f"AA={metrics['aa']:.3f}, AF={metrics['af']:.3f}")
    AA=0.860, AF=0.045
    """
    T = acc_matrix.shape[0]
    assert acc_matrix.shape == (T, T), f"acc_matrix doit être carrée [T, T], got {acc_matrix.shape}"

    # --- Average Accuracy ---
    # Moyenne de la dernière ligne (performances finales sur toutes les tâches)
    final_row = acc_matrix[T - 1, :T]
    aa = float(np.nanmean(final_row))

    # --- Average Forgetting ---
    # Pour chaque tâche j < T : chute entre le pic de performance et la performance finale
    forgetting_per_task = []
    for j in range(T - 1):
        max_acc_j = float(np.nanmax(acc_matrix[:, j]))      # meilleure performance sur tâche j
        final_acc_j = float(acc_matrix[T - 1, j])           # performance finale sur tâche j
        forgetting_per_task.append(max_acc_j - final_acc_j)

    af = float(np.mean(forgetting_per_task)) if forgetting_per_task else 0.0

    # --- Backward Transfer ---
    # Impact de l'apprentissage futur sur les tâches passées
    # BWT_j = acc_final(j) - acc_just_after_training(j)
    bwt_per_task = []
    for j in range(T - 1):
        acc_just_after = float(acc_matrix[j, j])    # diagonale
        acc_final = float(acc_matrix[T - 1, j])     # fin de tout l'entraînement
        bwt_per_task.append(acc_final - acc_just_after)

    bwt = float(np.mean(bwt_per_task)) if bwt_per_task else 0.0

    # --- Forward Transfer ---
    # Impact de l'apprentissage passé sur l'apprentissage d'une nouvelle tâche
    if random_baseline is None:
        random_baseline = np.full(T, 0.5)   # classifieur aléatoire binaire

    fwt_per_task = []
    for j in range(1, T):
        acc_before_j = float(acc_matrix[j - 1, j]) if not np.isnan(acc_matrix[j - 1, j]) else random_baseline[j]
        fwt_per_task.append(acc_before_j - float(random_baseline[j]))

    fwt = float(np.mean(fwt_per_task)) if fwt_per_task else 0.0

    return {
        "aa": aa,
        "af": af,
        "bwt": bwt,
        "fwt": fwt,
        "forgetting_per_task": forgetting_per_task,
        "bwt_per_task": bwt_per_task,
        "acc_matrix": np.where(np.isnan(acc_matrix), None, acc_matrix).tolist(),
        "n_tasks": T,
    }


def format_metrics_report(
    metrics: dict,
    model_name: str,
    baseline_finetune: Optional[dict] = None,
    baseline_joint: Optional[dict] = None,
) -> str:
    """
    Formate un rapport lisible des métriques CL.

    Parameters
    ----------
    metrics : dict
        Sortie de compute_cl_metrics().
    model_name : str
    baseline_finetune : dict, optional
        Métriques du baseline fine-tuning naïf (pour comparaison).
    baseline_joint : dict, optional
        Métriques du baseline joint training (borne supérieure).

    Returns
    -------
    str : rapport formaté multi-ligne.
    """
    lines = [
        f"{'=' * 60}",
        f"  RÉSULTATS CL — {model_name}",
        f"{'=' * 60}",
        f"",
        f"PRÉCISION :",
        f"  AA  = {metrics['aa']:.4f}",
    ]

    if baseline_finetune:
        lines.append(f"        (Fine-tuning naïf : {baseline_finetune['aa']:.4f})")
    if baseline_joint:
        lines.append(f"        (Joint training  : {baseline_joint['aa']:.4f}  ← borne sup.)")

    lines += [
        f"",
        f"OUBLI (Average Forgetting) :",
        f"  AF  = {metrics['af']:.4f}  (0 = aucun oubli | + = oubli important)",
    ]

    if metrics["forgetting_per_task"]:
        for i, f in enumerate(metrics["forgetting_per_task"]):
            lines.append(f"        Tâche {i + 1} : {f:+.4f}")

    lines += [
        f"",
        f"TRANSFERT :",
        f"  BWT = {metrics['bwt']:.4f}  (< 0 = oubli | > 0 = transfert positif)",
        f"  FWT = {metrics['fwt']:.4f}  (> 0 = facilitation apprentissage futur)",
        f"",
        f"TRIPLE GAP :",
        f"  Gap 2 (RAM) : voir memory_profiler.py pour les chiffres mesurés",
        f"{'=' * 60}",
    ]

    return "\n".join(lines)


def save_metrics(
    metrics: dict,
    output_path: str,
    extra_info: Optional[dict] = None,
) -> None:
    """
    Sauvegarde les métriques dans un fichier JSON reproductible.

    Parameters
    ----------
    metrics : dict
        Sortie de compute_cl_metrics().
    output_path : str
        Chemin de sortie (ex. "experiments/exp_001/results/metrics.json").
    extra_info : dict, optional
        Informations supplémentaires à inclure (config, RAM, etc.).
    """
    output = {**(extra_info or {}), "cl_metrics": metrics}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Métriques sauvegardées → {output_path}")


def accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Accuracy binaire avec seuil.

    Parameters
    ----------
    y_true : np.ndarray [N]
    y_pred : np.ndarray [N]  (probabilités ∈ [0, 1])
    threshold : float

    Returns
    -------
    float : accuracy ∈ [0, 1]
    """
    predictions = (y_pred >= threshold).astype(int)
    return float(np.mean(predictions == y_true.astype(int)))
