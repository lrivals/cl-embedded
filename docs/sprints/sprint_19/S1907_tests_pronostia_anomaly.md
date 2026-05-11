# S19-07 — Tests `get_pronostia_dataloaders_anomaly_detection()` (mode anomaly detection)

| Champ | Valeur |
|-------|--------|
| **ID** | S19-07 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S19-02 |
| **Fichier cible** | `tests/test_pronostia_anomaly.py` |

---

## Objectif

Valider `get_pronostia_dataloaders_anomaly_detection()` en mode anomaly detection via des tests unitaires sur données synthétiques. Si Sprint 15 couvre déjà ces cas, compléter uniquement les tests manquants.

---

## Tests à implémenter

```python
# tests/test_pronostia_anomaly.py

import numpy as np
import pytest
import torch
from pathlib import Path
from src.data.pronostia_dataset import get_pronostia_dataloaders_anomaly_detection

@pytest.fixture
def synthetic_pronostia_data(tmp_path):
    """Crée des données Pronostia synthétiques — 13 features, ~90% normal, 3 conditions."""
    import pandas as pd

    conditions = ["early_life", "mid_life", "end_of_life"]
    rows = []
    ratios = {"early_life": (45, 5), "mid_life": (35, 15), "end_of_life": (20, 30)}

    for condition, (n_normal, n_faulty) in ratios.items():
        for _ in range(n_normal):
            row = {f"feat_{i}": np.random.randn() for i in range(13)}
            row["condition"] = condition
            row["label"] = 0
            rows.append(row)
        for _ in range(n_faulty):
            row = {f"feat_{i}": np.random.randn() + 4.0 for i in range(13)}
            row["condition"] = condition
            row["label"] = 1
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "pronostia_synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_returns_three_tasks(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        data_path=synthetic_pronostia_data,
        scenario="by_bearing_condition",
    )
    assert len(loaders) == 3

def test_train_loader_only_normal(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, scenario="by_bearing_condition"
    )
    for _, _, train_loader, _ in loaders:
        for X_batch, y_batch in train_loader:
            assert (y_batch == 0).all(), "train_loader doit ne contenir que des normaux (label=0)"

def test_test_loader_has_both_classes(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, scenario="by_bearing_condition"
    )
    for _, _, _, test_loader in loaders:
        all_labels = []
        for _, y_batch in test_loader:
            all_labels.extend(y_batch.tolist())
        assert 0 in set(all_labels)
        assert 1 in set(all_labels)

def test_input_dim_is_13(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, scenario="by_bearing_condition"
    )
    for _, _, train_loader, _ in loaders:
        for X, *_ in train_loader:
            assert X.shape[1] == 13, f"Expected input_dim=13, got {X.shape[1]}"
            break

def test_reproducibility(synthetic_pronostia_data):
    """Deux appels avec le même seed donnent les mêmes données."""
    loaders1 = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, scenario="by_bearing_condition", seed=42
    )
    loaders2 = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, scenario="by_bearing_condition", seed=42
    )
    for (_, _, tl1, _), (_, _, tl2, _) in zip(loaders1, loaders2):
        X1 = next(iter(tl1))[0]
        X2 = next(iter(tl2))[0]
        assert torch.allclose(X1, X2), "Les loaders doivent être reproductibles avec seed=42"
```

---

## Commande d'exécution

```bash
pytest tests/test_pronostia_anomaly.py -v
```

---

## Critères d'acceptation

- [ ] 5 tests, 100% pass
- [ ] Test `test_input_dim_is_13` confirme la dimensionnalité Pronostia
- [ ] Aucun accès aux données réelles Pronostia dans les tests

## Statut

⬜ À faire (vérifier d'abord si Sprint 15 couvre déjà ces cas)
