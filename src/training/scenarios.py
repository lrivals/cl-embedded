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

from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from src.evaluation.anomaly_metrics import compute_anomaly_metrics
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


# ---------------------------------------------------------------------------
# Scénario anomaly detection (one-class CL)
# ---------------------------------------------------------------------------


def run_anomaly_detection_scenario(
    model: Any,
    tasks: list[dict],
    config: dict,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """
    Scénario anomaly detection : entraîner sur données normales, évaluer sur données mixtes.

    Boucle :
        pour chaque tâche i :
            1. Entraîner sur task["train_loader"] (faulty==0 uniquement)
            2. Évaluer sur task["test_loader_mixed"] pour toutes les tâches j ≤ i
            3. Stocker l'AUROC dans auroc_matrix[i, j]

    Compatible avec :
        - HDCClassifier (one_class_mode=True)        → interface update()+on_task_end()
        - TinyOLAnomalyDetector                      → interface update()+on_task_end()
        - MahalanobisDetector / KMeansDetector       → interface fit_task(X, task_id)

    Parameters
    ----------
    model : object
        Modèle avec anomaly_score(X) → np.ndarray.
        Interface d'entraînement détectée automatiquement (fit_task ou update/on_task_end).
    tasks : list[dict]
        Depuis get_cl_dataloaders_anomaly_detection().
        Clés requises : task_id, train_loader, test_loader_mixed.
    config : dict
        Configuration YAML du modèle (transmis pour compatibilité).

    Returns
    -------
    auroc_matrix : np.ndarray [T, T]
        auroc_matrix[i, j] = AUROC sur la tâche j après entraînement sur les tâches 0..i.
        NaN pour j > i.
    scores_dict : dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
        scores_dict[(i, j)] = (y_true, scores) — pour courbes ROC post-hoc.
    """
    T = len(tasks)
    auroc_matrix = np.full((T, T), np.nan)
    scores_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # Pré-extraction des données de test (une seule fois, hors boucle d'entraînement)
    X_tests: list[np.ndarray] = []
    y_tests: list[np.ndarray] = []
    for task in tasks:
        X_t = np.concatenate([b[0].numpy() for b in task["test_loader_mixed"]])
        y_t = np.concatenate([b[1].numpy().flatten() for b in task["test_loader_mixed"]])
        X_tests.append(X_t)
        y_tests.append(y_t)

    for i, task in enumerate(tasks):
        _train_anomaly_model(model, task["train_loader"], task["task_id"])

        for j in range(i + 1):
            scores = model.anomaly_score(X_tests[j])
            metrics = compute_anomaly_metrics(y_tests[j], scores)
            auroc_matrix[i, j] = metrics["auroc"]
            scores_dict[(i, j)] = (y_tests[j], scores)
            print(
                f"  [AD] après tâche {i+1}/{T}, eval tâche {j+1} "
                f"({tasks[j]['domain']:12s}) — AUROC={metrics['auroc']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

    return auroc_matrix, scores_dict


def _train_anomaly_model(
    model: Any,
    train_loader: DataLoader,
    task_id: int,
) -> None:
    """
    Dispatch d'entraînement selon l'interface du modèle.

    - fit_task(X, task_id) : MahalanobisDetector, KMeansDetector (offline batch)
    - update(x, y) + on_task_end() : HDCClassifier, TinyOLAnomalyDetector (online)
    """
    if hasattr(model, "fit_task"):
        X_train = np.concatenate([b[0].numpy() for b in train_loader])
        model.fit_task(X_train, task_id=task_id - 1)  # task_id 1-based → 0-based
    else:
        y_zeros = np.zeros(1, dtype=np.float32)  # placeholder (ignoré en one-class)
        for x_batch, y_batch in train_loader:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy().flatten()
            model.update(x_np, y_np)
        model.on_task_end(task_id, train_loader)
