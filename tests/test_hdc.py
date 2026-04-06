# ruff: noqa: N806  — H_level, H_pos sont des conventions mathématiques pour matrices HDC
"""
Tests unitaires pour src/models/hdc/hdc_classifier.py et base_vectors.py.

Organisation :
    TestBaseVectors    — génération et I/O des hypervecteurs de base
    TestHDCClassifier  — initialisation, predict, update, on_task_end
    TestHDCConstraints — contraintes embarquées (RAM, n_params)
    TestHDCIncremental — comportement CL (pas d'oubli par construction)

Critère d'acceptation (S2-08) :
    pytest tests/test_hdc.py -v  → ≥ 10 tests, 0 skip, 0 erreur, < 10 s
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.models.hdc.base_vectors import generate_base_hvectors, load_base_vectors, save_base_vectors
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
            load_base_vectors(Path("/tmp/nonexistent_hdc_vectors_s208.npz"))


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
        rng = np.random.default_rng(0)
        x_train = rng.standard_normal((8, 4)).astype(np.float32)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        model.update(x_train, y_train)

        x = rng.standard_normal((16, 4)).astype(np.float32)
        preds = model.predict(x)
        assert preds.shape == (16,), f"Expected (16,), got {preds.shape}"
        assert set(np.unique(preds)).issubset({0, 1}), "Prédictions doivent être binaires ∈ {0, 1}"

    def test_update_returns_float(self, hdc_config):
        """update() retourne un float (taux d'erreur sur le batch, proxy de perte)."""
        model = HDCClassifier(hdc_config)
        x = np.random.randn(8, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float32)
        loss = model.update(x, y)
        assert isinstance(loss, float), f"Expected float, got {type(loss)}"
        assert 0.0 <= loss <= 1.0, f"Loss proxy attendu dans [0, 1], got {loss}"

    def test_on_task_end_no_error(self, hdc_config, hdc_synthetic_loader):
        """on_task_end() s'exécute sans erreur et re-binarise les prototypes."""
        model = HDCClassifier(hdc_config)
        loader = hdc_synthetic_loader()
        for x_batch, y_batch in loader:
            model.update(x_batch.numpy(), y_batch.numpy().flatten())
        model.on_task_end(task_id=1, dataloader=loader)

    def test_predict_after_update_not_all_same(self, hdc_config):
        """Après plusieurs updates, le modèle ne prédit pas toujours la même classe."""
        model = HDCClassifier(hdc_config)
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal((4, 4)).astype(np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            model.update(x, y)
        x_test = rng.standard_normal((20, 4)).astype(np.float32)
        preds = model.predict(x_test)
        assert len(set(preds.tolist())) > 1, "Le modèle prédit toujours la même classe"


# =============================================================================
# TestHDCConstraints
# =============================================================================


class TestHDCConstraints:
    """Tests des contraintes embarquées STM32N6."""

    def test_count_parameters(self, hdc_config):
        """
        count_parameters() = N_CLASSES × D (prototypes entraînables).

        Note : les base vectors H_level / H_pos sont l'architecture fixe du modèle
        (stockés en Flash sur MCU), non comptés comme paramètres entraînables.
        Pour D=64, n_classes=2 → attendu = 128.
        """
        model = HDCClassifier(hdc_config)
        D = hdc_config["hdc"]["D"]             # 64
        n_classes = hdc_config["data"]["n_classes"]  # 2
        expected = n_classes * D               # 128
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
        # MEM: 2×64×4 B + 2×64×1 B + 4×64×4 B + 2×4 B = 1 560 B @ FP32

    def test_estimate_ram_bytes_int8_less_than_fp32(self, hdc_config):
        """Empreinte INT8 < FP32 pour les prototypes et le buffer d'encodage."""
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

        proto_after_task1 = model.prototypes_acc.copy()  # [N_CLASSES, D]
        model.on_task_end(task_id=1, dataloader=None)

        x2 = rng.standard_normal((8, 4)).astype(np.float32)
        y2 = np.array([1, 1, 0, 0, 0, 1, 0, 1], dtype=np.float32)
        model.update(x2, y2)

        assert not np.allclose(model.prototypes_acc, proto_after_task1), (
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
        # Triangle supérieur = NaN (tâches futures non encore vues)
        assert np.isnan(acc[0, 1])
        assert np.isnan(acc[0, 2])
        assert np.isnan(acc[1, 2])
