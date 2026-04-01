# S1-06 — Implémenter `baselines.py` (fine-tuning naïf + joint training)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-06 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S1-03 (loader), S1-04 (`EWCMlpClassifier`), S1-07 (`metrics.py`) |
| **Fichier cible** | `src/training/baselines.py` |

---

## Objectif

Implémenter les deux baselines CL qui servent de bornes de référence pour comparer les méthodes incrémentales :

1. **Fine-tuning naïf** (`train_naive_sequential`) — entraînement séquentiel pur (BCE seule, sans régularisation EWC). Produit l'**oubli catastrophique maximal** → borne inférieure.
2. **Joint training** (`train_joint`) — entraînement sur la concaténation de toutes les tâches simultanément. Pas de CL réel, mais fournit la **borne supérieure** d'accuracy.

Les deux fonctions retournent une `acc_matrix [T, T]` compatible avec `metrics.py` (S1-07).

**Critère de succès** : `pytest tests/test_baselines.py -v` passe, et les deux baselines produisent une `acc_matrix` bien formée avec `aa_joint > aa_naive` (borne supérieure > borne inférieure).

---

## Sous-tâches

### 1. Constantes de configuration

```python
# src/training/baselines.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from src.models.ewc import EWCMlpClassifier
from src.evaluation.metrics import accuracy_binary

# Valeurs par défaut — toujours surchargées par ewc_config.yaml
DEFAULT_EPOCHS: int = 10
DEFAULT_LR: float = 0.01
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_BATCH_SIZE: int = 32
```

### 2. Évaluation sur une tâche

Fonction utilitaire partagée par les deux baselines :

```python
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
```

### 3. Fine-tuning naïf (borne inférieure)

```python
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
```

### 4. Joint training (borne supérieure)

```python
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
```

### 5. Écrire les tests

Créer `tests/test_baselines.py` :

```python
import numpy as np
import torch
from src.models.ewc import EWCMlpClassifier

def test_naive_acc_matrix_shape(cl_tasks, ewc_config):
    """acc_matrix doit être [T, T] avec NaN au-dessus de la diagonale."""
    from src.training.baselines import train_naive_sequential
    model = EWCMlpClassifier()
    M = train_naive_sequential(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    assert M.shape == (T, T)
    assert np.isnan(M[0, 1])  # tâche 2 pas encore vue après task 1

def test_joint_acc_matrix_shape(cl_tasks, ewc_config):
    """Joint training : seule la dernière ligne est non-NaN."""
    from src.training.baselines import train_joint
    model = EWCMlpClassifier()
    M = train_joint(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    assert not np.isnan(M[T - 1, :]).any()

def test_joint_aa_geq_naive_aa(cl_tasks, ewc_config):
    """Joint training doit avoir AA ≥ fine-tuning naïf (borne supérieure)."""
    from src.training.baselines import train_naive_sequential, train_joint
    from src.evaluation.metrics import compute_cl_metrics
    M_naive = train_naive_sequential(EWCMlpClassifier(), cl_tasks, ewc_config)
    M_joint = train_joint(EWCMlpClassifier(), cl_tasks, ewc_config)
    aa_naive = compute_cl_metrics(M_naive)["aa"]
    aa_joint = compute_cl_metrics(M_joint)["aa"]
    assert aa_joint >= aa_naive, f"Joint AA ({aa_joint:.3f}) < Naive AA ({aa_naive:.3f})"
```

> **Note** : les fixtures `cl_tasks` et `ewc_config` seront définies dans `tests/conftest.py` (S1-10).

---

## Critères d'acceptation

- [ ] `from src.training.baselines import train_naive_sequential, train_joint` — aucune erreur d'import
- [ ] `train_naive_sequential()` retourne `acc_matrix [T, T]` avec NaN au-dessus de la diagonale
- [ ] `train_joint()` retourne `acc_matrix [T, T]` avec la dernière ligne non-NaN
- [ ] `aa_joint >= aa_naive` (borne supérieure ≥ borne inférieure)
- [ ] Compatible avec `compute_cl_metrics()` de `src/evaluation/metrics.py`
- [ ] `pytest tests/test_baselines.py -v` — tous les tests passent
- [ ] `ruff check src/training/baselines.py` et `black --check` passent

---

## Interface attendue par `scripts/train_ewc.py` (S1-09)

```python
from src.training.baselines import train_naive_sequential, train_joint
from src.models.ewc import EWCMlpClassifier

# Borne inférieure
model_naive = EWCMlpClassifier(**model_kwargs)
acc_naive = train_naive_sequential(model_naive, tasks, config)

# Borne supérieure
model_joint = EWCMlpClassifier(**model_kwargs)
acc_joint = train_joint(model_joint, tasks, config)

# Résultats à inclure dans le rapport final (S1-09)
metrics_naive = compute_cl_metrics(acc_naive)
metrics_joint = compute_cl_metrics(acc_joint)
```

---

## Questions ouvertes

- `TODO(arnaud)` : le budget d'epochs du joint training doit-il être `epochs × T` (même budget total) ou `epochs` seuls (avantage pour le joint) ?
- `TODO(arnaud)` : faut-il inclure un troisième baseline **cumulative** (replay complet de toutes les tâches passées) pour mieux isoler l'effet de l'EWC ?
