# S2-08 — Tests unitaires HDC

| Champ | Valeur |
|-------|--------|
| **ID** | S2-08 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S2-01 (`base_vectors.py`), S2-02 (`hdc_classifier.py`), S2-07 (`hdc_config.yaml`) |
| **Fichiers cibles** | `tests/test_hdc.py`, `tests/conftest.py` (ajout fixture `hdc_config`) |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Écrire la suite de tests unitaires pour le module HDC (`src/models/hdc/`). Les tests couvrent :
- L'initialisation du classificateur depuis la config
- L'encodage et l'inférence (`predict`)
- La mise à jour incrémentale (`update`)
- La consolidation de fin de tâche (`on_task_end`)
- Les contraintes embarquées (RAM ≤ 64 Ko, nombre de paramètres)

Ces tests sont la porte d'entrée qualité avant l'expérience `exp_002` (S2-03). Ils doivent être suffisamment rapides pour tourner en CI (<5 s).

**Critère de succès** : `pytest tests/test_hdc.py -v` passe intégralement (≥ 10 tests, 0 skip).

---

## Sous-tâches

### 1. Ajouter la fixture `hdc_config` dans `tests/conftest.py`

Ajouter après la fixture `ewc_config` existante :

```python
# tests/conftest.py — ajout après les fixtures existantes

@pytest.fixture
def hdc_config() -> dict:
    """Config minimale compatible avec la structure de hdc_config.yaml."""
    return {
        "hdc": {
            "D": 64,            # Réduit pour la rapidité des tests (vs 1024 en prod)
            "n_levels": 5,      # Réduit (vs 10 en prod)
            "seed": 42,
            "base_vectors_path": None,  # Généré à la volée dans les tests
        },
        "data": {
            "n_features": 4,
            "n_classes": 2,
            "domain_order": ["pump", "turbine", "compressor"],
            "feature_bounds": {
                "temperature": [-3.0, 3.0],
                "pressure":    [-3.0, 3.0],
                "vibration":   [-3.0, 3.0],
                "humidity":    [-3.0, 3.0],
            },
        },
        "training": {
            "epochs_per_task": 1,
            "batch_size": 1,
            "seed": 42,
        },
        "memory": {
            "target_ram_bytes": 65536,
            "warn_if_above_bytes": 52000,
        },
    }


@pytest.fixture
def hdc_synthetic_loader():
    """Factory DataLoader synthétique pour HDC — input_dim=4."""
    def make_loader(n: int = 32, seed: int = 42) -> DataLoader:
        torch.manual_seed(seed)
        X = torch.randn(n, 4)
        y = torch.randint(0, 2, (n, 1)).float()
        return DataLoader(TensorDataset(X, y), batch_size=8)
    return make_loader


@pytest.fixture
def hdc_cl_tasks(hdc_synthetic_loader) -> list[dict]:
    """3 tâches CL synthétiques pour HDC (input_dim=4)."""
    return [
        {
            "task_id": i + 1,
            "domain": d,
            "train_loader": hdc_synthetic_loader(seed=i),
            "val_loader": hdc_synthetic_loader(n=16, seed=i + 10),
        }
        for i, d in enumerate(["pump", "turbine", "compressor"])
    ]
```

> **Note** : `D=64` et `n_levels=5` réduisent le temps de test à quelques ms.
> La RAM estimée avec ces paramètres = `(64×5 + 64×4) × 1 B = 576 B` (loin du budget 64 Ko).

### 2. Tests `tests/test_hdc.py`

```python
# tests/test_hdc.py
"""
Tests unitaires pour src/models/hdc/hdc_classifier.py et base_vectors.py.

Organisation :
    TestBaseVectors   — génération et I/O des hypervecteurs de base
    TestHDCClassifier — initialisation, predict, update, on_task_end
    TestHDCConstraints — contraintes embarquées (RAM, n_params)
    TestHDCIncremental — comportement CL (pas d'oubli par construction)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from src.models.hdc.base_vectors import generate_base_hvectors, save_base_vectors, load_base_vectors
from src.models.hdc.hdc_classifier import HDCClassifier


# =============================================================================
# TestBaseVectors
# =============================================================================

class TestBaseVectors:
    """Tests pour src/models/hdc/base_vectors.py."""

    def test_shapes(self):
        """H_level et H_pos ont les dimensions attendues."""
        H_level, H_pos = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        assert H_level.shape == (5, 64)
        assert H_pos.shape == (4, 64)

    def test_dtype_int8(self):
        """Les hypervecteurs sont en INT8 (contrainte MCU Flash)."""
        H_level, H_pos = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        assert H_level.dtype == np.int8
        assert H_pos.dtype == np.int8
        # MEM: 5 × 64 × 1 B = 320 B @ INT8 (H_level)
        # MEM: 4 × 64 × 1 B = 256 B @ INT8 (H_pos)

    def test_binary_values(self):
        """Valeurs ∈ {-1, +1} uniquement."""
        H_level, H_pos = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        assert set(np.unique(H_level)).issubset({-1, 1})
        assert set(np.unique(H_pos)).issubset({-1, 1})

    def test_reproducibility(self):
        """Deux appels avec le même seed donnent les mêmes vecteurs."""
        H1, P1 = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        H2, P2 = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(P1, P2)

    def test_different_seeds_differ(self):
        """Des seeds différents produisent des vecteurs différents."""
        H1, _ = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        H2, _ = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=99)
        assert not np.array_equal(H1, H2)

    def test_save_load_roundtrip(self, tmp_path):
        """save + load retournent des tableaux identiques."""
        H_level, H_pos = generate_base_hvectors(D=64, n_levels=5, n_features=4, seed=42)
        path = tmp_path / "test_vectors.npz"
        save_base_vectors(H_level, H_pos, path)
        H_loaded, P_loaded = load_base_vectors(path)
        np.testing.assert_array_equal(H_level, H_loaded)
        np.testing.assert_array_equal(H_pos, P_loaded)

    def test_load_missing_raises(self):
        """FileNotFoundError si le fichier n'existe pas."""
        with pytest.raises(FileNotFoundError):
            load_base_vectors(Path("/tmp/nonexistent_hdc_vectors.npz"))


# =============================================================================
# TestHDCClassifier
# =============================================================================

class TestHDCClassifier:
    """Tests pour src/models/hdc/hdc_classifier.py."""

    def test_init_from_config(self, hdc_config):
        """Le modèle s'initialise sans erreur depuis un dict de config."""
        model = HDCClassifier(hdc_config)
        assert model is not None

    def test_predict_shape(self, hdc_config):
        """predict() retourne un tableau [N] de prédictions binaires."""
        model = HDCClassifier(hdc_config)
        x = np.random.randn(16, 4).astype(np.float32)
        preds = model.predict(x)
        assert preds.shape == (16,), f"Expected (16,), got {preds.shape}"
        assert set(np.unique(preds)).issubset({0, 1}), "Prédictions doivent être binaires ∈ {0, 1}"

    def test_update_returns_float(self, hdc_config):
        """update() retourne un float (proxy de perte, ex. taux d'erreur sur le batch)."""
        model = HDCClassifier(hdc_config)
        x = np.random.randn(8, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float32)
        loss = model.update(x, y)
        assert isinstance(loss, float), f"Expected float, got {type(loss)}"
        assert 0.0 <= loss <= 1.0, f"Loss proxy attendu dans [0, 1], got {loss}"

    def test_on_task_end_no_error(self, hdc_config, hdc_synthetic_loader):
        """on_task_end() s'exécute sans erreur et met à jour les prototypes normalisés."""
        model = HDCClassifier(hdc_config)
        loader = hdc_synthetic_loader()
        # Accumuler quelques updates avant on_task_end
        for x_batch, y_batch in loader:
            model.update(x_batch.numpy(), y_batch.numpy().flatten())
        # Ne doit pas lever d'exception
        model.on_task_end(task_id=1, dataloader=loader)

    def test_predict_after_update_not_all_same(self, hdc_config):
        """Après plusieurs updates, le modèle ne prédit pas toujours la même classe."""
        model = HDCClassifier(hdc_config)
        rng = np.random.default_rng(42)
        # Entraîner avec des exemples étiquetés
        for _ in range(20):
            x = rng.standard_normal((4, 4)).astype(np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            model.update(x, y)
        x_test = rng.standard_normal((20, 4)).astype(np.float32)
        preds = model.predict(x_test)
        # Après entraînement, le modèle ne doit pas prédire toujours la même classe
        assert len(set(preds.tolist())) > 1, "Le modèle prédit toujours la même classe"


# =============================================================================
# TestHDCConstraints
# =============================================================================

class TestHDCConstraints:
    """Tests des contraintes embarquées STM32N6."""

    def test_count_parameters(self, hdc_config):
        """
        count_parameters() = N_LEVELS × D + N_FEATURES × D.

        Pour hdc_config de test (D=64, n_levels=5, n_features=4) :
            H_level : 5 × 64 = 320 entiers
            H_pos   : 4 × 64 = 256 entiers
            Total   : 576 éléments
        """
        model = HDCClassifier(hdc_config)
        D = hdc_config["hdc"]["D"]           # 64
        n_levels = hdc_config["hdc"]["n_levels"]  # 5
        n_features = hdc_config["data"]["n_features"]  # 4
        expected = n_levels * D + n_features * D
        assert model.count_parameters() == expected, (
            f"Expected {expected} params, got {model.count_parameters()}"
        )

    def test_estimate_ram_bytes_within_budget(self, hdc_config):
        """RAM estimée doit respecter le budget 64 Ko (65 536 B)."""
        model = HDCClassifier(hdc_config)
        ram = model.estimate_ram_bytes(dtype="fp32")
        assert ram <= 65536, (
            f"RAM estimée {ram} B > 65 536 B (64 Ko) — contrainte STM32N6 violée"
        )
        # MEM: 2 classes × 64 dims × 4 B = 512 B @ FP32 (prototypes)
        #      5 × 64 × 1 B + 4 × 64 × 1 B = 576 B @ INT8 (H_level + H_pos)

    def test_estimate_ram_bytes_int8_less_than_fp32(self, hdc_config):
        """Empreinte INT8 < FP32 pour les vecteurs de base."""
        model = HDCClassifier(hdc_config)
        ram_fp32 = model.estimate_ram_bytes(dtype="fp32")
        ram_int8 = model.estimate_ram_bytes(dtype="int8")
        assert ram_int8 < ram_fp32, "INT8 doit être plus compact que FP32"


# =============================================================================
# TestHDCIncremental
# =============================================================================

class TestHDCIncremental:
    """Tests du comportement CL — pas d'oubli par construction HDC."""

    def test_prototypes_accumulate(self, hdc_config):
        """Les prototypes s'accumulent sans remise à zéro entre les tâches."""
        model = HDCClassifier(hdc_config)

        rng = np.random.default_rng(0)
        x1 = rng.standard_normal((8, 4)).astype(np.float32)
        y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        model.update(x1, y1)

        # Récupérer l'état des prototypes après Task 1
        proto_after_task1 = model.prototypes.copy()  # [N_CLASSES, D]

        model.on_task_end(task_id=1, dataloader=None)

        # Entraîner sur Task 2
        x2 = rng.standard_normal((8, 4)).astype(np.float32)
        y2 = np.array([1, 1, 0, 0, 0, 1, 0, 1], dtype=np.float32)
        model.update(x2, y2)

        # Les prototypes doivent avoir changé (accumulation)
        assert not np.allclose(model.prototypes, proto_after_task1), (
            "Les prototypes n'ont pas évolué après Task 2 — accumulation non fonctionnelle"
        )

    def test_scenario_produces_valid_acc_matrix(self, hdc_config, hdc_cl_tasks):
        """run_cl_scenario() avec HDC produit une acc_matrix [T, T] valide."""
        from src.training.scenarios import run_cl_scenario
        model = HDCClassifier(hdc_config)
        acc = run_cl_scenario(model, hdc_cl_tasks, hdc_config)
        T = len(hdc_cl_tasks)
        assert acc.shape == (T, T)
        # Diagonale non-NaN (tâche évaluée juste après son entraînement)
        for i in range(T):
            assert not np.isnan(acc[i, i]), f"acc[{i},{i}] est NaN"
        # Triangle supérieur = NaN (tâches futures non vues)
        assert np.isnan(acc[0, 1])
        assert np.isnan(acc[0, 2])
        assert np.isnan(acc[1, 2])
```

---

## Critères d'acceptation

- [ ] `pytest tests/test_hdc.py -v` — ≥ 10 tests passent, 0 skip, 0 erreur
- [ ] `pytest tests/test_hdc.py -v --tb=short` tourne en < 10 secondes
- [ ] Fixture `hdc_config` ajoutée dans `tests/conftest.py`
- [ ] Fixture `hdc_cl_tasks` ajoutée dans `tests/conftest.py`
- [ ] Les tests de contraintes (`TestHDCConstraints`) passent avec les paramètres de prod (`D=1024, n_levels=10, n_features=4`)
- [ ] `ruff check tests/test_hdc.py` + `black --check` passent

---

## Interface attendue par `scripts/train_hdc.py` (S2-03)

```python
# Après S2-08, le train_hdc.py peut s'appuyer sur HDCClassifier validé :
from src.models.hdc import HDCClassifier
from src.training.scenarios import run_cl_scenario

model = HDCClassifier(config)
acc_matrix = run_cl_scenario(model, tasks, config)

# Vérification RAM avant l'expérience
ram_check = model.check_ram_budget()
if not ram_check["within_budget"]:
    print(f"WARN: RAM {ram_check['estimated_bytes']} B > budget {ram_check['budget_bytes']} B")
```

---

## Questions ouvertes

- `TODO(arnaud)` : `model.prototypes` doit-il être exposé comme attribut public (pour les tests) ou accédé via une méthode `get_prototypes()` ?
- `TODO(dorra)` : sur MCU, les prototypes sont-ils stockés en FP32 (pour l'accumulation) ou INT32 (pour éviter l'overflow lors de la somme de D valeurs ∈ {-1, +1}) ?
- `FIXME(gap2)` : ajouter un test qui mesure la RAM réelle avec `tracemalloc` (module `evaluation/memory_profiler.py`) plutôt que l'estimation statique.
