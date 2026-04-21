# S10-08 — Tests unitaires `pronostia_dataset.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S10-08 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S10-01 (`PronostiaDataset` implémenté) |
| **Fichiers cibles** | `tests/test_pronostia_dataset.py` |

---

## Objectif

Écrire la suite de tests unitaires pour `src/data/pronostia_dataset.py`. Les tests valident le comportement du loader **sans dépendance aux données réelles** (utilisation de données synthétiques générées à la volée via `pytest` fixtures).

---

## Architecture des tests

```python
# tests/test_pronostia_dataset.py
import numpy as np
import pytest
from pathlib import Path
from src.data.pronostia_dataset import PronostiaDataset, PronostiaConditionStream

WINDOW_SIZE = 2560
N_EPOCHS_SMALL = 50   # Petit dataset synthétique pour les tests
N_CHANNELS = 2
```

---

## Fixture : données synthétiques

```python
@pytest.fixture
def synthetic_npy_dir(tmp_path: Path) -> Path:
    """Crée un répertoire temporaire avec 6 fichiers .npy synthétiques."""
    sizes = {
        "Bearing1_1": 200, "Bearing1_2": 180,  # Condition 1
        "Bearing2_1": 100, "Bearing2_2": 90,   # Condition 2
        "Bearing3_1": 120, "Bearing3_2": 110,  # Condition 3
    }
    for name, n in sizes.items():
        data = np.random.randn(n, N_CHANNELS, WINDOW_SIZE).astype(np.float32)
        np.save(tmp_path / f"{name}.npy", data)
    return tmp_path
```

---

## Tests à implémenter

### 1. Shape et type des features

```python
def test_dataset_feature_shape(synthetic_npy_dir):
    """Chaque sample doit retourner un vecteur de 13 features."""
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1"],
    )
    features, label = dataset[0]
    assert features.shape == (13,), f"Attendu (13,), obtenu {features.shape}"
    assert features.dtype == np.float32
```

### 2. Ratio de labels pré-défaillance

```python
def test_label_ratio(synthetic_npy_dir):
    """La proportion de labels=1 doit être ≈ failure_ratio ± tolérance."""
    failure_ratio = 0.10
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1"],
        failure_ratio=failure_ratio,
    )
    labels = [dataset[i][1] for i in range(len(dataset))]
    actual_ratio = sum(labels) / len(labels)
    assert abs(actual_ratio - failure_ratio) < 0.05, \
        f"Ratio attendu ≈ {failure_ratio}, obtenu {actual_ratio:.3f}"
```

### 3. Position temporelle

```python
def test_temporal_position_bounds(synthetic_npy_dir):
    """La feature temporal_position (index 12) doit être dans [0, 1]."""
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1"],
    )
    for i in range(len(dataset)):
        features, _ = dataset[i]
        assert 0.0 <= features[-1] <= 1.0, \
            f"temporal_position hors bornes à l'index {i} : {features[-1]}"
```

### 4. Longueur du dataset multi-roulements

```python
def test_dataset_length_multi_bearings(synthetic_npy_dir):
    """len() doit être la somme des epochs de tous les roulements chargés."""
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1", "Bearing1_2"],
    )
    # Bearing1_1 : 200 epochs, Bearing1_2 : 180 epochs
    assert len(dataset) == 200 + 180
```

### 5. Nombre de conditions dans le stream

```python
def test_condition_stream_yields_three(synthetic_npy_dir):
    """PronostiaConditionStream doit yielder exactement 3 conditions."""
    condition_map = {
        1: ["Bearing1_1", "Bearing1_2"],
        2: ["Bearing2_1", "Bearing2_2"],
        3: ["Bearing3_1", "Bearing3_2"],
    }
    stream = PronostiaConditionStream(
        npy_dir=str(synthetic_npy_dir),
        condition_map=condition_map,
    )
    conditions = list(stream)
    assert len(conditions) == 3
    condition_ids = [c_id for c_id, _ in conditions]
    assert condition_ids == [1, 2, 3]
```

### 6. Absence de data leakage dans la normalisation

```python
def test_normalize_no_leakage(synthetic_npy_dir):
    """La normalisation doit être calculée sur le train set uniquement."""
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1"],
        normalize=True,
    )
    # Les paramètres de normalisation doivent être attributs du dataset
    assert hasattr(dataset, "feature_mean"), "feature_mean manquant"
    assert hasattr(dataset, "feature_std"),  "feature_std manquant"
    assert dataset.feature_mean.shape == (12,), \
        "Normalisation sur les 12 features hors temporal_position"
```

### 7. Erreur si répertoire absent

```python
def test_missing_npy_dir_raises():
    """FileNotFoundError si le répertoire .npy n'existe pas."""
    with pytest.raises(FileNotFoundError):
        PronostiaDataset(
            npy_dir="/chemin/inexistant/",
            bearing_ids=["Bearing1_1"],
        )
```

### 8. Erreur si roulement absent

```python
def test_missing_bearing_raises(synthetic_npy_dir):
    """FileNotFoundError si un bearing_id demandé est absent."""
    with pytest.raises(FileNotFoundError):
        PronostiaDataset(
            npy_dir=str(synthetic_npy_dir),
            bearing_ids=["BearingXXX"],
        )
```

### 9. Contrainte RAM (taille features)

```python
def test_feature_ram_footprint(synthetic_npy_dir):
    """Un vecteur de 13 features FP32 doit peser exactement 52 octets."""
    dataset = PronostiaDataset(
        npy_dir=str(synthetic_npy_dir),
        bearing_ids=["Bearing1_1"],
    )
    features, _ = dataset[0]
    assert features.nbytes == 13 * 4, \
        f"RAM features: attendu 52 B, obtenu {features.nbytes} B"
```

---

## Commande d'exécution

```bash
pytest tests/test_pronostia_dataset.py -v
# ou avec couverture
pytest tests/test_pronostia_dataset.py -v --cov=src.data.pronostia_dataset
```

---

## Critères d'acceptation

- [ ] 9 tests implémentés dans `tests/test_pronostia_dataset.py`
- [ ] `pytest tests/test_pronostia_dataset.py -v` → 100% pass
- [ ] Aucun test ne dépend des données réelles dans `data/raw/` (fixtures synthétiques uniquement)
- [ ] Couverture de `src/data/pronostia_dataset.py` ≥ 80%

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il ajouter un test d'intégration qui charge un vrai fichier `.npy` depuis `data/raw/Pronostia dataset/binaries/` (marqué `@pytest.mark.integration`) pour valider le format réel, en plus des tests unitaires sur données synthétiques ?

---

**Complété le** : _(à renseigner)_
