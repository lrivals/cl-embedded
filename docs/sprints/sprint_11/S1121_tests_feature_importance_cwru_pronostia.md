# S11-21 — Tests `test_feature_importance_cwru_pronostia.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S11-21 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S11-11 (`permutation_importance_per_task()`, constantes CWRU/Pronostia dans `feature_importance.py`) |
| **Fichier cible** | `tests/test_feature_importance_cwru_pronostia.py` |

---

## Objectif

Valider que les extensions de `feature_importance.py` spécifiques à CWRU et Pronostia
fonctionnent correctement : constantes de noms de features, `permutation_importance_per_task()`,
et compatibilité avec les 9 features CWRU et 13 features Pronostia.

---

## Tests à implémenter

### 1. Constantes CWRU et Pronostia

```python
def test_feature_names_cwru_length():
    assert len(FEATURE_NAMES_CWRU) == 9

def test_feature_names_pronostia_length():
    assert len(FEATURE_NAMES_PRONOSTIA) == 13

def test_channel_groups_pronostia_coverage():
    all_in_groups = (
        CHANNEL_GROUPS_PRONOSTIA["acc_horiz"]
        + CHANNEL_GROUPS_PRONOSTIA["acc_vert"]
        + CHANNEL_GROUPS_PRONOSTIA["temporal"]
    )
    assert set(all_in_groups) == set(FEATURE_NAMES_PRONOSTIA)
```

### 2. `permutation_importance_per_task()` — comportement de base

```python
def test_permutation_importance_per_task_keys():
    """Vérifie que les clés du dict retourné = noms de tâches."""
    tasks = [
        {"task_name": "ball",       "X": np.random.randn(50, 9), "y": np.random.randint(0, 2, 50)},
        {"task_name": "inner_race", "X": np.random.randn(50, 9), "y": np.random.randint(0, 2, 50)},
    ]
    result = permutation_importance_per_task(
        predict_fn=lambda X: (X[:, 0] > 0).astype(float),
        tasks=tasks,
        feature_names=FEATURE_NAMES_CWRU,
        n_repeats=2,
    )
    assert set(result.keys()) == {"ball", "inner_race"}

def test_permutation_importance_per_task_feature_keys():
    """Vérifie que chaque sous-dict contient les 9 features CWRU."""
    tasks = [
        {"task_name": "outer_race", "X": np.random.randn(30, 9), "y": np.zeros(30)},
    ]
    result = permutation_importance_per_task(
        predict_fn=lambda X: np.zeros(len(X)),
        tasks=tasks,
        feature_names=FEATURE_NAMES_CWRU,
        n_repeats=2,
    )
    assert set(result["outer_race"].keys()) == set(FEATURE_NAMES_CWRU)
```

### 3. `permutation_importance_per_task()` — Pronostia (13 features)

```python
def test_permutation_importance_per_task_pronostia():
    tasks = [
        {"task_name": "cond_1", "X": np.random.randn(40, 13), "y": np.random.randint(0, 2, 40)},
        {"task_name": "cond_2", "X": np.random.randn(40, 13), "y": np.random.randint(0, 2, 40)},
    ]
    result = permutation_importance_per_task(
        predict_fn=lambda X: (X[:, 0] > 0).astype(float),
        tasks=tasks,
        feature_names=FEATURE_NAMES_PRONOSTIA,
        n_repeats=2,
    )
    assert len(result) == 2
    for task_name, imp in result.items():
        assert len(imp) == 13
```

### 4. Cohérence avec `permutation_importance()` globale

```python
def test_per_task_vs_global_score_range():
    """Les scores per-task doivent être dans [0, 1] (chute d'accuracy)."""
    X = np.random.randn(60, 9)
    y = (X[:, 0] > 0).astype(int)
    tasks = [{"task_name": "t1", "X": X, "y": y}]
    result = permutation_importance_per_task(
        predict_fn=lambda X: (X[:, 0] > 0).astype(float),
        tasks=tasks,
        feature_names=FEATURE_NAMES_CWRU,
        n_repeats=3,
    )
    for score in result["t1"].values():
        assert -0.1 <= score <= 1.0  # tolérance faible valeur négative (variance)
```

### 5. `_resolve_feature_names()` — sélection selon le dataset

```python
def test_resolve_feature_names_cwru():
    cfg = {"dataset": {"name": "cwru"}}
    assert _resolve_feature_names(cfg) == FEATURE_NAMES_CWRU

def test_resolve_feature_names_pronostia():
    cfg = {"dataset": {"name": "pronostia"}}
    assert _resolve_feature_names(cfg) == FEATURE_NAMES_PRONOSTIA

def test_resolve_feature_names_monitoring_default():
    cfg = {"dataset": {"name": "monitoring"}}
    names = _resolve_feature_names(cfg)
    assert len(names) == 4  # temperature, pressure, vibration, humidity
```

> **Note** : `_resolve_feature_names` est une fonction privée des scripts `train_*.py`.
> Si elle est extraite dans `feature_importance.py`, l'importer directement.
> Sinon, dupliquer la logique dans le test ou tester via les scripts.

---

## Fixtures à utiliser

```python
import numpy as np
import pytest
from src.evaluation.feature_importance import (
    FEATURE_NAMES_CWRU,
    FEATURE_NAMES_PRONOSTIA,
    CHANNEL_GROUPS_PRONOSTIA,
    permutation_importance_per_task,
)
```

Pas de fixture pytest complexe nécessaire — les données synthétiques `np.random.randn` suffisent.
Utiliser `np.random.default_rng(42)` pour la reproductibilité.

---

## Statut

✅ `tests/test_feature_importance_cwru_pronostia.py` — créé
