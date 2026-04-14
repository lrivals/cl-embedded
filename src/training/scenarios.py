# ruff: noqa: N806  — T est une convention mathématique standard en CL (nombre de tâches)
"""
scenarios.py — Boucle d'entraînement CL générique (agnostique au modèle).

Contrairement à baselines.py (spécifique à EWCMlpClassifier + PyTorch),
ce module s'appuie sur l'interface BaseCLModel (predict/update/on_task_end)
pour évaluer n'importe quel modèle du projet (HDC, EWC, TinyOL).

Usage typique :
    from src.training.scenarios import run_cl_scenario
    acc_matrix = run_cl_scenario(model, tasks, config)

Références :
    DeLange2021Survey §3 — taxonomie des scénarios CL.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.evaluation.metrics import accuracy_binary
from src.models.base_cl_model import BaseCLModel

# Valeurs par défaut — surchargées par le fichier YAML de chaque modèle
DEFAULT_EVAL_BATCH_SIZE: int = 128


def evaluate_task_generic(
    model: BaseCLModel,
    val_loader: DataLoader,
) -> float:
    """
    Évalue un BaseCLModel sur un val_loader via model.predict().

    Ne suppose aucun framework sous-jacent — l'encodage est géré par le modèle.

    Parameters
    ----------
    model : BaseCLModel
        Modèle évalué (HDCClassifier, EWCMlpClassifier, etc.).
    val_loader : DataLoader
        DataLoader de validation. Retourne (x, y) avec x : Tensor [B, F].

    Returns
    -------
    float
        Accuracy binaire ∈ [0, 1].

    Notes
    -----
    x est converti en np.ndarray avant d'être passé à model.predict().
    Cela permet à HDCClassifier (NumPy) et EWCMlpClassifier (PyTorch) d'être
    appelés via la même interface.
    """
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for x, y in val_loader:
        x_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        y_np = y.numpy().flatten()
        preds = model.predict(x_np)  # [B] ou [B, 1] → doit retourner [B]
        all_preds.append(preds.flatten())
        all_labels.append(y_np)

    return accuracy_binary(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


def run_cl_scenario(
    model: BaseCLModel,
    tasks: list[dict],
    config: dict,
) -> np.ndarray:
    """
    Exécute un scénario domain-incremental sur une séquence de tâches.

    Boucle :
        pour chaque tâche i :
            1. update online sur chaque batch de train_loader
            2. appel on_task_end(task_id, train_loader) après la tâche
            3. évaluation sur toutes les tâches vues (0..i)

    Compatible avec tout modèle héritant de BaseCLModel.
    Pour HDC : update() accumule les prototypes (sans gradient).
    Pour EWC : update() effectue un pas de SGD + calcul Fisher online.

    Parameters
    ----------
    model : BaseCLModel
        Modèle initialisé (non entraîné) avant l'appel.
    tasks : list[dict]
        Liste de dicts avec clés :
            - 'task_id' : int
            - 'domain' : str (ex. "pump", "turbine", "compressor")
            - 'train_loader' : DataLoader
            - 'val_loader' : DataLoader
        Ordre : Pump → Turbine → Compressor (DOMAIN_ORDER du dataset).
    config : dict
        Configuration YAML chargée (hdc_config.yaml ou ewc_config.yaml).
        Transmis à on_task_end() si nécessaire.

    Returns
    -------
    np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur la tâche j après entraînement sur la tâche i.
        NaN pour j > i (tâche pas encore vue).

    Notes
    -----
    FIXME(gap1) : scénario domain-incremental uniquement pour l'instant.
        Étendre à class-incremental si nécessaire pour Dataset 1 (TinyOL).

    References
    ----------
    DeLange2021Survey §3 — taxonomie des scénarios CL.
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)

    for i, task in enumerate(tasks):
        # --- Mise à jour online sur la tâche i ---
        for x_batch, y_batch in task["train_loader"]:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy().flatten()

            # model.update() accepte un batch — compatible avec HDC et EWC
            model.update(x_np, y_np)

        # --- Fin de tâche : consolidation (Fisher pour EWC, normalisation pour HDC) ---
        model.on_task_end(task["task_id"], task["train_loader"])

        # --- Évaluation sur toutes les tâches vues jusqu'ici ---
        for j in range(i + 1):
            acc_matrix[i, j] = evaluate_task_generic(model, tasks[j]["val_loader"])

    return acc_matrix


def run_cl_scenario_full(
    model: BaseCLModel,
    tasks: list[dict],
    config: dict,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """
    Variante de run_cl_scenario() qui collecte aussi les prédictions brutes par (tâche_train, tâche_eval).

    Retourne acc_matrix ET preds_dict pour la construction de matrices de confusion et courbes ROC.
    L'interface run_cl_scenario() reste inchangée (compatibilité).

    Parameters
    ----------
    model : BaseCLModel
        Modèle initialisé (non entraîné) avant l'appel.
    tasks : list[dict]
        Même format que run_cl_scenario() — clés 'task_id', 'domain', 'train_loader', 'val_loader'.
    config : dict
        Configuration YAML chargée.

    Returns
    -------
    acc_matrix : np.ndarray [T, T]
        Matrice d'accuracy CL (NaN pour j > i).
    preds_dict : dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
        preds_dict[(i, j)] = (y_true, y_pred) — après entraînement sur tâche i, évaluation sur tâche j.
        Uniquement les couples (i, j) avec j ≤ i (tâches déjà vues).
        y_pred contient les scores bruts du modèle (probabilités ou scores continus).

    Notes
    -----
    Utilise evaluate_task_with_preds() (metrics.py) pour collecter y_true/y_pred sans dupliquer
    la logique de predict().
    """
    from src.evaluation.metrics import evaluate_task_with_preds

    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)
    preds_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for i, task in enumerate(tasks):
        # --- Mise à jour online sur la tâche i ---
        for x_batch, y_batch in task["train_loader"]:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy().flatten()
            model.update(x_np, y_np)

        # --- Fin de tâche : consolidation ---
        model.on_task_end(task["task_id"], task["train_loader"])

        # --- Évaluation sur toutes les tâches vues jusqu'ici ---
        for j in range(i + 1):
            acc, y_true, y_pred = evaluate_task_with_preds(model, tasks[j]["val_loader"])
            acc_matrix[i, j] = acc
            preds_dict[(i, j)] = (y_true, y_pred)

    return acc_matrix, preds_dict
