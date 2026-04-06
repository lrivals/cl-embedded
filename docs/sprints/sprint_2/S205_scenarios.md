# S2-05 — Implémenter `scenarios.py` (gestion générique des streams CL)

| Champ | Valeur |
|-------|--------|
| **ID** | S2-05 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S1-03 (`monitoring_dataset.py`), S1-06 (`baselines.py`), S2-01 (`base_vectors.py`), S2-02 (`hdc_classifier.py`) |
| **Fichiers cibles** | `src/training/scenarios.py`, `tests/test_scenarios.py`, `tests/conftest.py` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Implémenter une boucle d'entraînement CL **générique** compatible avec tout modèle héritant de `BaseCLModel` (`src/models/base_cl_model.py`). Contrairement à `baselines.py` (spécifique à `EWCMlpClassifier` + PyTorch), `scenarios.py` s'appuie sur l'interface `predict/update/on_task_end` pour être **agnostique au modèle**.

Cette abstraction permet d'évaluer `EWCMlpClassifier`, `HDCClassifier` et futurs modèles avec la même boucle — un seul fichier de résultats, des métriques comparables.

**Critère de succès** : `run_cl_scenario(hdc_model, tasks, hdc_config)` et `run_cl_scenario(ewc_model, tasks, ewc_config)` retournent toutes deux une `acc_matrix [T, T]` valide, compatible avec `compute_cl_metrics()`.

---

## Sous-tâches

### 1. Imports et constantes

```python
# src/training/scenarios.py

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from src.models.base_cl_model import BaseCLModel
from src.evaluation.metrics import accuracy_binary

# Valeurs par défaut — surchargées par le fichier YAML de chaque modèle
DEFAULT_EVAL_BATCH_SIZE: int = 128
```

### 2. Évaluation générique sur une tâche

```python
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
        x_np = x.numpy()
        y_np = y.numpy().flatten()
        preds = model.predict(x_np)  # [B] ou [B, 1] → doit retourner [B]
        all_preds.append(preds.flatten())
        all_labels.append(y_np)

    return accuracy_binary(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )
```

### 3. Boucle CL générique

```python
def run_cl_scenario(
    model: BaseCLModel,
    tasks: list[dict],
    config: dict,
) -> np.ndarray:
    """
    Exécute un scénario domain-incremental sur une séquence de tâches.

    Boucle :
        pour chaque tâche i :
            1. update online sur chaque échantillon de train_loader
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
        Utilisée pour passer les métadonnées à on_task_end() si nécessaire.

    Returns
    -------
    np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur la tâche j après entraînement sur la tâche i.
        NaN pour j > i (tâche pas encore vue).

    Notes
    -----
    FIXME(gap1) : scénario domain-incremental uniquement pour l'instant.
        Étendre à class-incremental si nécessaire pour Dataset 1 (TinyOL).

    Références
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

            # Update échantillon par échantillon (online learning)
            # model.update() accepte un batch — compatible avec les deux modèles
            model.update(x_np, y_np)

        # --- Fin de tâche : consolidation (Fisher pour EWC, normalisation pour HDC) ---
        model.on_task_end(task["task_id"], task["train_loader"])

        # --- Évaluation sur toutes les tâches vues jusqu'ici ---
        for j in range(i + 1):
            acc_matrix[i, j] = evaluate_task_generic(model, tasks[j]["val_loader"])

    return acc_matrix
```

### 4. Écrire les tests dans `tests/test_scenarios.py`

```python
import numpy as np
import pytest
from src.training.scenarios import run_cl_scenario, evaluate_task_generic


def test_acc_matrix_shape(hdc_model_mock, cl_tasks):
    """acc_matrix doit être [T, T] avec NaN au-dessus de la diagonale."""
    acc = run_cl_scenario(hdc_model_mock, cl_tasks, {})
    T = len(cl_tasks)
    assert acc.shape == (T, T)
    assert np.isnan(acc[0, 1])  # tâche 2 pas encore vue après task 1
    assert not np.isnan(acc[T - 1, 0])  # tâche 1 vue depuis la fin


def test_evaluate_task_generic_range(hdc_model_mock, cl_tasks):
    """accuracy doit être dans [0, 1]."""
    acc = evaluate_task_generic(hdc_model_mock, cl_tasks[0]["val_loader"])
    assert 0.0 <= acc <= 1.0


def test_compatible_with_compute_cl_metrics(hdc_model_mock, cl_tasks):
    """La matrice doit être compatible avec compute_cl_metrics()."""
    from src.evaluation.metrics import compute_cl_metrics
    acc = run_cl_scenario(hdc_model_mock, cl_tasks, {})
    metrics = compute_cl_metrics(acc)
    assert "aa" in metrics
    assert "af" in metrics
    assert "bwt" in metrics
```

> **Note** : `hdc_model_mock` sera ajouté dans `tests/conftest.py` — un stub minimal
> qui implémente `predict/update/on_task_end` de `BaseCLModel` avec des valeurs aléatoires.

---

## Critères d'acceptation

- [x] `from src.training.scenarios import run_cl_scenario` — aucune erreur d'import
- [x] `run_cl_scenario(hdc_model, tasks, hdc_config)` retourne `acc_matrix [T, T]` valide
- [x] `run_cl_scenario(ewc_model, tasks, ewc_config)` retourne `acc_matrix [T, T]` valide
- [x] NaN au-dessus de la diagonale (tâches non vues)
- [x] Compatible avec `compute_cl_metrics()` de `src/evaluation/metrics.py`
- [x] `pytest tests/test_scenarios.py -v` — tous les tests passent (3/3)
- [x] `ruff check src/training/scenarios.py` + `black --check` passent

---

## Interface attendue par `scripts/train_hdc.py` (S2-03)

```python
from src.training.scenarios import run_cl_scenario
from src.models.hdc import HDCClassifier

model = HDCClassifier(config)
acc_matrix = run_cl_scenario(model, tasks, config)

from src.evaluation.metrics import compute_cl_metrics
metrics = compute_cl_metrics(acc_matrix)
# metrics = {"aa": ..., "af": ..., "bwt": ...}
```

---

## Questions ouvertes

- `TODO(arnaud)` : `update()` doit-il traiter le batch entier en une passe ou bien sample-by-sample (strictement online) ? HDC supporte les deux mais le comportement des métriques diffère.
- `TODO(arnaud)` : faut-il ajouter un paramètre `n_epochs_per_task` pour les modèles neuraux (EWC fait plusieurs passes sur chaque tâche) ? HDC est naturellement one-pass.
- `FIXME(gap1)` : scénario class-incremental pour Dataset 1 (TinyOL) non géré — à étendre en Sprint 3.
