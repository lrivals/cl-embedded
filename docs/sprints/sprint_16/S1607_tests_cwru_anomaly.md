# S16-07 — Tests `get_cwru_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S16-07 |
| **Sprint** | Sprint 16 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S16-02 |
| **Fichier cible** | `tests/test_cwru_anomaly.py` |

---

## Objectif

Valider `get_cwru_dataloaders_anomaly_detection()` via des tests unitaires sur données synthétiques, avec attention particulière au faible ratio normal (~10%) et au split temporel/aléatoire selon le scénario.

---

## Tests à implémenter

```python
# tests/test_cwru_anomaly.py

import numpy as np
import pytest
import torch
from pathlib import Path
from src.data.cwru_dataset import get_cwru_dataloaders_anomaly_detection

@pytest.fixture
def synthetic_cwru_csv(tmp_path):
    """Crée un CSV CWRU synthétique avec 9 features, ~10% normal."""
    import pandas as pd
    n_normal = 23   # ~10% de 230
    n_faulty = 207  # ~90% de 230 (3 types × 3 sévérités × ~23)

    data_normal = np.random.randn(n_normal, 9).astype(np.float32)
    data_faulty = np.random.randn(n_faulty, 9).astype(np.float32) + 3.0

    labels_normal = ["Normal"] * n_normal
    labels_faulty = (
        ["Ball_007"] * 8 + ["Ball_014"] * 7 + ["Ball_021"] * 8 +
        ["IR_007"] * 8 + ["IR_014"] * 7 + ["IR_021"] * 8 +
        ["OR_007"] * 8 + ["OR_014"] * 7 + ["OR_021"] * 7 +
        ["Ball_007"] * 10 + ["IR_007"] * 10 + ["OR_007"] * 11  # padding
    )[:n_faulty]

    X = np.vstack([data_normal, data_faulty])
    labels = labels_normal + labels_faulty

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(9)])
    df["fault_type"] = ["Normal"] * n_normal + ["Faulty"] * n_faulty
    df["severity"] = ["0"] * n_normal + ["007"] * (n_faulty // 3) + ["014"] * (n_faulty // 3) + ["021"] * (n_faulty - 2 * (n_faulty // 3))
    df["label"] = [0] * n_normal + [1] * n_faulty

    csv_path = tmp_path / "cwru_synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_returns_three_tasks_by_severity(synthetic_cwru_csv):
    loaders = get_cwru_dataloaders_anomaly_detection(
        data_path=synthetic_cwru_csv,
        scenario="by_severity",
    )
    assert len(loaders) == 3

def test_returns_three_tasks_by_fault_type(synthetic_cwru_csv):
    loaders = get_cwru_dataloaders_anomaly_detection(
        data_path=synthetic_cwru_csv,
        scenario="by_fault_type",
    )
    assert len(loaders) == 3

def test_train_loader_only_normal(synthetic_cwru_csv):
    loaders = get_cwru_dataloaders_anomaly_detection(
        synthetic_cwru_csv, scenario="by_severity"
    )
    for _, _, train_loader, _ in loaders:
        for batch in train_loader:
            X_batch = batch[0]
            assert X_batch.shape[1] == 9

def test_test_loader_has_both_classes(synthetic_cwru_csv):
    loaders = get_cwru_dataloaders_anomaly_detection(
        synthetic_cwru_csv, scenario="by_severity"
    )
    for _, _, _, test_loader in loaders:
        all_labels = []
        for _, y_batch in test_loader:
            all_labels.extend(y_batch.tolist())
        assert 0 in set(all_labels)
        assert 1 in set(all_labels)

def test_input_dim_is_9(synthetic_cwru_csv):
    loaders = get_cwru_dataloaders_anomaly_detection(
        synthetic_cwru_csv, scenario="by_severity"
    )
    for _, _, train_loader, test_loader in loaders:
        for X, *_ in train_loader:
            assert X.shape[1] == 9
            break

def test_warning_emitted_on_low_normal_count(synthetic_cwru_csv):
    """Vérifie que UserWarning est émis si < 100 normaux par tâche."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_cwru_dataloaders_anomaly_detection(
            synthetic_cwru_csv, scenario="by_severity"
        )
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) > 0
```

---

## Commande d'exécution

```bash
pytest tests/test_cwru_anomaly.py -v
```

---

## Critères d'acceptation

- [ ] 6 tests, 100% pass
- [ ] Test `test_warning_emitted_on_low_normal_count` passe (warning bien émis)
- [ ] Aucun accès aux données réelles CWRU dans les tests

## Statut

⬜ À faire
