"""
baselines.py — Baselines CL pour comparaison avec EWC.

Deux baselines :
    1. Fine-tuning naïf  (train_naive_sequential) — borne inférieure : oubli catastrophique maximal
    2. Joint training    (train_joint)             — borne supérieure : toutes les tâches simultanément

Les deux fonctions retournent une acc_matrix [T, T] compatible avec metrics.py.

Références :
    De Lange et al. (2021). A CL Survey. TPAMI. — taxonomie baselines §3.1
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.evaluation.metrics import accuracy_binary
from src.models.ewc import EWCMlpClassifier

# Valeurs par défaut — toujours surchargées par ewc_config.yaml
DEFAULT_EPOCHS: int = 10
DEFAULT_LR: float = 0.01
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_BATCH_SIZE: int = 32


def evaluate_task(
    model: EWCMlpClassifier,
    val_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Évalue le modèle sur un val_loader.

    Parameters
    ----------
    model : EWCMlpClassifier
    val_loader : DataLoader
    device : str

    Returns
    -------
    float : accuracy ∈ [0, 1]
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).cpu().numpy().flatten()
            all_preds.extend(pred)
            all_labels.extend(y.cpu().numpy().flatten())
    return accuracy_binary(np.array(all_labels), np.array(all_preds))


def train_naive_sequential(
    model: EWCMlpClassifier,
    tasks: list[dict],
    config: dict,
    device: str = "cpu",
) -> np.ndarray:
    """
    Entraîne le modèle séquentiellement sur chaque tâche sans régularisation.

    La perte est purement BCE — aucune protection contre l'oubli.
    Modélise le pire cas : oubli catastrophique complet des tâches précédentes.

    Parameters
    ----------
    model : EWCMlpClassifier
        Modèle réinitialisé avant l'appel.
    tasks : list[dict]
        Liste de dicts avec clés 'train_loader', 'val_loader', 'task_id'.
        Ordre : Pump → Turbine → Compressor (conforme à DOMAIN_ORDER).
    config : dict
        Config chargée depuis ewc_config.yaml.
    device : str

    Returns
    -------
    np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur la tâche j après entraînement sur la tâche i.
        NaN pour j > i (tâche pas encore vue).

    Notes
    -----
    FIXME(gap1) : borne inférieure attendue — documenter les chiffres d'oubli obtenus.
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)

    epochs = config["training"]["epochs_per_task"]
    lr = config["training"]["learning_rate"]
    momentum = config["training"]["momentum"]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    model.to(device)

    for i, task in enumerate(tasks):
        # Entraînement sur la tâche i
        model.train()
        for _ in range(epochs):
            for x, y in task["train_loader"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

        # Évaluation sur toutes les tâches vues jusqu'ici
        for j in range(i + 1):
            acc_matrix[i, j] = evaluate_task(model, tasks[j]["val_loader"], device)

    return acc_matrix


def train_joint(
    model: EWCMlpClassifier,
    tasks: list[dict],
    config: dict,
    device: str = "cpu",
) -> np.ndarray:
    """
    Entraîne le modèle sur la concaténation de toutes les tâches.

    Pas de CL réel : le modèle voit toutes les données simultanément.
    Produit la borne supérieure d'accuracy — impossible à reproduire en CL pur.

    Parameters
    ----------
    model : EWCMlpClassifier
        Modèle réinitialisé avant l'appel.
    tasks : list[dict]
        Même format que train_naive_sequential.
    config : dict
    device : str

    Returns
    -------
    np.ndarray [T, T]
        acc_matrix[T-1, :] = performances finales sur chaque tâche.
        Les lignes 0..T-2 sont remplies de NaN (entraînement non-incrémental).

    Notes
    -----
    La matrice retournée est non-standard (une seule ligne non-NaN) mais reste
    compatible avec compute_cl_metrics() — AA = moyenne de la dernière ligne.
    TODO(arnaud) : budget d'epochs = epochs × T (même budget total) ou epochs seuls ?
    TODO(arnaud) : ajouter un troisième baseline cumulative (replay complet) ?
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)

    # Concaténer tous les train_loaders
    all_datasets = [task["train_loader"].dataset for task in tasks]
    joint_dataset = ConcatDataset(all_datasets)
    joint_loader = DataLoader(
        joint_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )

    epochs = config["training"]["epochs_per_task"] * T  # même budget total d'epochs
    lr = config["training"]["learning_rate"]
    momentum = config["training"]["momentum"]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    model.to(device)
    model.train()

    for _ in range(epochs):
        for x, y in joint_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # Évaluation finale sur toutes les tâches
    for j in range(T):
        acc_matrix[T - 1, j] = evaluate_task(model, tasks[j]["val_loader"], device)

    return acc_matrix
