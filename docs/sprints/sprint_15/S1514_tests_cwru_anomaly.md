# S15-19 — Tests `get_cwru_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S15-19 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1.5h |
| **Dépendances** | S15-13 |
| **Fichier cible** | `tests/test_cwru_anomaly.py` |

---

## Objectif

Valider `get_cwru_dataloaders_anomaly_detection()` via des tests unitaires sur données synthétiques (fixture `tmp_path`), sans accès aux données réelles CWRU (`data/raw/`).

---

## Différence avec les tests Pronostia

| Aspect | Pronostia (S15-07) | CWRU (S15-19) |
|--------|-------------------|---------------|
| Format données | NPY (features préextractées) | CSV (`feature_time_48k_2048_load_1.csv`) |
| Features | 13 | 9 |
| Label normal/faulty | Construit par seuil temporel (FAILURE_RATIO) | Issu du CSV (colonne `fault_label`) |
| Paramètre spécifique | `failure_ratio` | `scenario` ("by_severity" / "by_fault_type") |
| Nombre de normaux | ~90% | ~10% |

---

## Tests à implémenter

```python
# tests/test_cwru_anomaly.py

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from src.data.cwru_dataset import get_cwru_dataloaders_anomaly_detection


@pytest.fixture
def synthetic_cwru_csv(tmp_path):
    """
    Crée un CSV CWRU synthétique avec 9 features statistiques.
    Structure : Normal_1 (~10%), Ball_007/014/021, IR_007/014/021, OR_007/014/021.
    """
    n_per_class = 30
    classes = [
        "Normal_1",
        "Ball_007", "Ball_014", "Ball_021",
        "IR_007", "IR_014", "IR_021",
        "OR_007", "OR_014", "OR_021",
    ]
    feature_cols = [f"feat_{i}" for i in range(9)]

    rows = []
    for cls in classes:
        X = np.random.randn(n_per_class, 9).astype(np.float32)
        for x in X:
            row = dict(zip(feature_cols, x))
            row["fault_label"] = cls
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "feature_time_48k_2048_load_1.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_returns_three_tasks_by_severity(synthetic_cwru_csv):
    tasks = get_cwru_dataloaders_anomaly_detection(
        data_path=synthetic_cwru_csv,
        scenario="by_severity",
        seed=42,
    )
    assert len(tasks) == 3


def test_returns_three_tasks_by_fault_type(synthetic_cwru_csv):
    tasks = get_cwru_dataloaders_anomaly_detection(
        data_path=synthetic_cwru_csv,
        scenario="by_fault_type",
        seed=42,
    )
    assert len(tasks) == 3


def test_train_loader_only_normal(synthetic_cwru_csv):
    tasks = get_cwru_dataloaders_anomaly_detection(
        synthetic_cwru_csv, scenario="by_severity"
    )
    for task in tasks:
        for X_batch, y_batch in task["train_loader"]:
            assert torch.all(y_batch == 0), "train_loader contient des données faulty"
            assert X_batch.shape[1] == 9


def test_test_loader_has_both_classes(synthetic_cwru_csv):
    tasks = get_cwru_dataloaders_anomaly_detection(
        synthetic_cwru_csv, scenario="by_severity"
    )
    for task in tasks:
        all_labels = []
        for _, y_batch in task["test_loader_mixed"]:
            all_labels.extend(y_batch.tolist())
        unique = set(all_labels)
        assert 0.0 in unique, "test_loader_mixed ne contient pas de données normales"
        assert 1.0 in unique, "test_loader_mixed ne contient pas de données faulty"


def test_input_dim_is_9(synthetic_cwru_csv):
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_cwru_csv)
    for task in tasks:
        for X_batch, _ in task["train_loader"]:
            assert X_batch.shape[1] == 9
            break
        for X_batch, _ in task["test_loader_mixed"]:
            assert X_batch.shape[1] == 9
            break
```

---

## Commande d'exécution

```bash
pytest tests/test_cwru_anomaly.py -v
```

---

## Critères d'acceptation

- [x] 5 tests, 100% pass
- [x] Aucun accès aux fichiers `data/raw/` ou `data/processed/` dans les tests
- [x] `tmp_path` utilisé pour le CSV synthétique (fixture pytest)
- [x] Les deux scénarios (`by_severity`, `by_fault_type`) sont couverts par les tests

## Statut

✅ Terminé

## Bilan

`tests/test_cwru_anomaly.py` implémenté et validé. Résultat `pytest` : **5 passed**. Tous les tests utilisent la fixture `synthetic_cwru_csv(tmp_path)` qui génère un CSV synthétique de 300 lignes (30 × 10 classes) en mémoire — aucun accès aux données brutes. Les 5 cas couvrent : retour de 3 tâches (by_severity), retour de 3 tâches (by_fault_type), train-loader uniquement normal, test-loader avec deux classes, et `input_dim=9`.
