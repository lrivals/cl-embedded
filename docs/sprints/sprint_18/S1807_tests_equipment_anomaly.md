# S18-07 — Tests `get_equipment_monitoring_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S18-07 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S18-02 |
| **Fichier cible** | `tests/test_equipment_anomaly.py` |

---

## Objectif

Valider `get_equipment_monitoring_dataloaders_anomaly_detection()` via des tests unitaires sur données synthétiques, avec attention au scénario by_equipment_type et au ratio ~50% normal.

---

## Tests à implémenter

```python
# tests/test_equipment_anomaly.py

import numpy as np
import pytest
import torch
from pathlib import Path
from src.data.equipment_monitoring_dataset import get_equipment_monitoring_dataloaders_anomaly_detection

@pytest.fixture
def synthetic_equipment_csv(tmp_path):
    """Crée un CSV Equipment Monitoring synthétique — 4 features, ~50% normal, 3 types."""
    import pandas as pd

    equipment_types = ["Pump", "Turbine", "Compressor"]
    rows = []
    for eq_type in equipment_types:
        n_normal = 30
        n_faulty = 30
        for _ in range(n_normal):
            rows.append({
                "temperature": np.random.randn(),
                "pressure": np.random.randn(),
                "vibration": np.random.randn(),
                "humidity": np.random.randn(),
                "equipment_type": eq_type,
                "faulty": 0,
            })
        for _ in range(n_faulty):
            rows.append({
                "temperature": np.random.randn() + 3.0,
                "pressure": np.random.randn() + 3.0,
                "vibration": np.random.randn() + 3.0,
                "humidity": np.random.randn() + 3.0,
                "equipment_type": eq_type,
                "faulty": 1,
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "equipment_synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_returns_three_tasks(synthetic_equipment_csv):
    loaders = get_equipment_monitoring_dataloaders_anomaly_detection(
        data_path=synthetic_equipment_csv,
    )
    assert len(loaders) == 3

def test_task_names_are_equipment_types(synthetic_equipment_csv):
    loaders = get_equipment_monitoring_dataloaders_anomaly_detection(
        synthetic_equipment_csv
    )
    task_names = [name for _, name, _, _ in loaders]
    assert set(task_names) == {"pump", "turbine", "compressor"}

def test_train_loader_only_normal(synthetic_equipment_csv):
    loaders = get_equipment_monitoring_dataloaders_anomaly_detection(
        synthetic_equipment_csv
    )
    for _, _, train_loader, _ in loaders:
        for X_batch, y_batch in train_loader:
            assert (y_batch == 0).all(), "train_loader doit ne contenir que des normaux (label=0)"

def test_test_loader_has_both_classes(synthetic_equipment_csv):
    loaders = get_equipment_monitoring_dataloaders_anomaly_detection(
        synthetic_equipment_csv
    )
    for _, _, _, test_loader in loaders:
        all_labels = []
        for _, y_batch in test_loader:
            all_labels.extend(y_batch.tolist())
        assert 0 in set(all_labels)
        assert 1 in set(all_labels)

def test_input_dim_is_4(synthetic_equipment_csv):
    loaders = get_equipment_monitoring_dataloaders_anomaly_detection(
        synthetic_equipment_csv
    )
    for _, _, train_loader, test_loader in loaders:
        for X, *_ in train_loader:
            assert X.shape[1] == 4, f"Expected input_dim=4, got {X.shape[1]}"
            break
```

---

## Commande d'exécution

```bash
pytest tests/test_equipment_anomaly.py -v
```

---

## Critères d'acceptation

- [ ] 5 tests, 100% pass
- [ ] Aucun accès aux données réelles Equipment Monitoring dans les tests
- [ ] Test `test_train_loader_only_normal` vérifie explicitement `y == 0`

## Statut

⬜ À faire
