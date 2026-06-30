"""Tests unitaires pour src/data/paderborn_loader.py.

Aucun accès aux fichiers data/raw/ — assertions sur constantes et fixtures synthétiques.
Les tests end-to-end sont marqués skipif (données réelles requises).

Exécution :
    pytest tests/test_paderborn_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestPaderbornConstants:
    """Tests sur les constantes exportées du loader."""

    def test_domain_labels(self):
        """K001 → faulty=0, KA04 et KI04 → faulty=1."""
        from src.data.paderborn_loader import DOMAIN_LABELS

        assert DOMAIN_LABELS["K001"] == 0
        assert DOMAIN_LABELS["KA04"] == 1
        assert DOMAIN_LABELS["KI04"] == 1

    def test_domain_order(self):
        """3 domaines dans l'ordre sain → OR → IR."""
        from src.data.paderborn_loader import DOMAIN_ORDER

        assert DOMAIN_ORDER == ["K001", "KA04", "KI04"]
        assert len(DOMAIN_ORDER) == 3

    def test_n_features(self):
        """Nombre de features sélectionnées = 5."""
        from src.data.paderborn_loader import PADERBORN_N_FEATURES_SELECTED

        assert PADERBORN_N_FEATURES_SELECTED == 5

    def test_n_features_raw(self):
        """7 features brutes : rms + kurtosis + crest + 4 bandes énergie."""
        from src.data.paderborn_loader import PADERBORN_N_FEATURES_RAW

        assert PADERBORN_N_FEATURES_RAW == 7

    def test_window_size(self):
        """Taille de fenêtre FFT = 1024 points."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE

        assert PADERBORN_WINDOW_SIZE == 1024

    def test_sampling_rate(self):
        """Fréquence d'échantillonnage = 64 000 Hz."""
        from src.data.paderborn_loader import PADERBORN_SAMPLING_RATE

        assert PADERBORN_SAMPLING_RATE == 64_000

    def test_feature_names_raw_count(self):
        """FEATURE_NAMES_RAW contient exactement 7 noms."""
        from src.data.paderborn_loader import FEATURE_NAMES_RAW, PADERBORN_N_FEATURES_RAW

        assert len(FEATURE_NAMES_RAW) == PADERBORN_N_FEATURES_RAW

    def test_freq_bands_count(self):
        """4 bandes fréquentielles définies."""
        from src.data.paderborn_loader import FREQ_BANDS

        assert len(FREQ_BANDS) == 4

    def test_all_domain_labels_covered(self):
        """DOMAIN_LABELS couvre tous les domaines de DOMAIN_ORDER."""
        from src.data.paderborn_loader import DOMAIN_LABELS, DOMAIN_ORDER

        for domain in DOMAIN_ORDER:
            assert domain in DOMAIN_LABELS


class TestPaderbornFeatureExtraction:
    """Tests sur la fonction _compute_features (aucune donnée réelle requise)."""

    def test_feature_extraction_shape(self):
        """Features extraites depuis un signal synthétique : shape (1, 7)."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE, _compute_features

        rng = np.random.default_rng(42)
        signal = rng.standard_normal(PADERBORN_WINDOW_SIZE).astype(np.float32)
        windows = signal.reshape(1, PADERBORN_WINDOW_SIZE)
        features = _compute_features(windows, fs=64_000)
        assert features.shape == (1, 7)

    def test_feature_extraction_finite(self):
        """Toutes les features sont finies (pas de NaN ni Inf)."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE, _compute_features

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(PADERBORN_WINDOW_SIZE).astype(np.float32)
        windows = signal.reshape(1, PADERBORN_WINDOW_SIZE)
        features = _compute_features(windows, fs=64_000)
        assert np.isfinite(features).all()

    def test_feature_extraction_multiple_windows(self):
        """_compute_features accepte N fenêtres et retourne shape (N, 7)."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE, _compute_features

        n_windows = 8
        rng = np.random.default_rng(7)
        windows = rng.standard_normal((n_windows, PADERBORN_WINDOW_SIZE)).astype(np.float32)
        features = _compute_features(windows, fs=64_000)
        assert features.shape == (n_windows, 7)
        assert np.isfinite(features).all()

    def test_rms_positive(self):
        """La feature RMS (index 0) est toujours positive."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE, _compute_features

        rng = np.random.default_rng(3)
        windows = rng.standard_normal((4, PADERBORN_WINDOW_SIZE)).astype(np.float32)
        features = _compute_features(windows, fs=64_000)
        assert (features[:, 0] >= 0).all()  # rms ≥ 0

    def test_energy_bands_nonnegative(self):
        """Les 4 bandes d'énergie (index 3–6) sont non-négatives."""
        from src.data.paderborn_loader import PADERBORN_WINDOW_SIZE, _compute_features

        rng = np.random.default_rng(5)
        windows = rng.standard_normal((4, PADERBORN_WINDOW_SIZE)).astype(np.float32)
        features = _compute_features(windows, fs=64_000)
        assert (features[:, 3:7] >= 0).all()


@pytest.mark.skipif(
    not Path("data/raw/paderborn/K001").exists()
    and not Path(
        "data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/K001"
    ).exists(),
    reason="Données Paderborn non disponibles",
)
class TestPaderbornDataloaders:
    """Tests end-to-end avec données réelles (skipif données absentes)."""

    def test_get_cl_dataloaders_shape(self):
        """3 tâches, x.shape[1]==5, y.shape[1]==1."""
        import torch

        from src.data.paderborn_loader import get_cl_dataloaders

        data_dir = Path(
            "data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/"
        )
        tasks = get_cl_dataloaders(data_dir, Path("configs/paderborn_config.yaml"))
        assert len(tasks) == 3  # K001, KA04, KI04
        for task in tasks:
            x, y = next(iter(task["train_loader"]))
            assert x.shape[1] == 5
            assert y.shape[1] == 1
            assert x.dtype == torch.float32

    def test_task_keys(self):
        """Chaque tâche expose les clés attendues."""
        from src.data.paderborn_loader import get_cl_dataloaders

        data_dir = Path(
            "data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/"
        )
        tasks = get_cl_dataloaders(data_dir, Path("configs/paderborn_config.yaml"))
        required_keys = {"task_id", "domain", "train_loader", "val_loader", "n_train", "n_val"}
        for task in tasks:
            assert required_keys.issubset(task.keys())
