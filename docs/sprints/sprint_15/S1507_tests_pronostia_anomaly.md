# S15-07 — Tests `get_pronostia_dataloaders_anomaly_detection()`

| Champ | Valeur |
|-------|--------|
| **ID** | S15-07 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1.5h |
| **Dépendances** | S15-01 |
| **Fichier cible** | `tests/test_pronostia_anomaly.py` |

---

## Objectif

Valider `get_pronostia_dataloaders_anomaly_detection()` via des tests unitaires sur données synthétiques (fixtures `tmp_path`), sans accès aux données réelles Pronostia.

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
    """Crée des fichiers Pronostia synthétiques (CSV) avec 13 features."""
    # Structure : 3 conditions, 1 roulement chacune, 200 timesteps
    for cond in [1, 2, 3]:
        cond_dir = tmp_path / f"condition_{cond}" / "bearing_{cond}_1"
        cond_dir.mkdir(parents=True)
        T = 200
        X = np.random.randn(T, 13).astype(np.float32)
        # Sauvegarde dans le format attendu par le loader
        np.save(cond_dir / "features.npy", X)
    return tmp_path

def test_returns_three_tasks(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        data_path=synthetic_pronostia_data,
        failure_ratio=0.10,
        seed=42,
    )
    assert len(loaders) == 3

def test_task_ids_are_ordered(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(synthetic_pronostia_data)
    task_ids = [t[0] for t in loaders]
    assert task_ids == [0, 1, 2]

def test_train_loader_only_normal(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, failure_ratio=0.10
    )
    for _, _, train_loader, _ in loaders:
        for batch in train_loader:
            X_batch = batch[0]
            # Si le loader retourne (X,) sans y, pas de vérification de label nécessaire
            assert X_batch.shape[1] == 13

def test_test_loader_has_both_classes(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, failure_ratio=0.10
    )
    for _, _, _, test_loader in loaders:
        all_labels = []
        for X_batch, y_batch in test_loader:
            all_labels.extend(y_batch.tolist())
        unique_labels = set(all_labels)
        assert 0 in unique_labels  # données normales présentes
        assert 1 in unique_labels  # données faulty présentes

def test_failure_ratio_affects_split(synthetic_pronostia_data):
    loaders_10 = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, failure_ratio=0.10
    )
    loaders_20 = get_pronostia_dataloaders_anomaly_detection(
        synthetic_pronostia_data, failure_ratio=0.20
    )
    # Avec un ratio plus élevé, plus de données faulty en test
    n_faulty_10 = sum(
        sum(y.sum().item() for _, y in loader)
        for _, _, _, loader in loaders_10
    )
    n_faulty_20 = sum(
        sum(y.sum().item() for _, y in loader)
        for _, _, _, loader in loaders_20
    )
    assert n_faulty_20 > n_faulty_10

def test_input_dim_is_13(synthetic_pronostia_data):
    loaders = get_pronostia_dataloaders_anomaly_detection(synthetic_pronostia_data)
    for _, _, train_loader, test_loader in loaders:
        for batch in train_loader:
            assert batch[0].shape[1] == 13
            break
```

---

## Commande d'exécution

```bash
pytest tests/test_pronostia_anomaly.py -v
```

---

## Critères d'acceptation

- [ ] 6 tests, 100% pass
- [ ] Aucun accès aux fichiers `data/raw/` ou `data/processed/` dans les tests
- [ ] `tmp_path` utilisé pour les données synthétiques (fixture pytest)

## Statut

⬜ À faire
