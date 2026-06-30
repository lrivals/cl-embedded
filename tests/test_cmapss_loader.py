"""Tests unitaires pour src/data/cmapss_loader.py.

Aucun accès aux fichiers data/raw/ — assertions sur constantes et logique pure.
Les tests end-to-end sont marqués skipif (données réelles requises).

Exécution :
    pytest tests/test_cmapss_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestCmapssConstants:
    """Tests sur les constantes exportées du loader."""

    def test_rul_capping(self):
        """RUL cappé à 125 : aucune valeur > 125 après capping."""
        from src.data.cmapss_loader import CMAPSS_RUL_CAP

        rul_raw = np.array([0, 50, 125, 200, 350])
        rul_capped = np.minimum(rul_raw, CMAPSS_RUL_CAP)
        assert (rul_capped <= CMAPSS_RUL_CAP).all()
        assert rul_capped[3] == CMAPSS_RUL_CAP  # 200 → 125
        assert rul_capped[4] == CMAPSS_RUL_CAP  # 350 → 125
        assert rul_capped[1] == 50               # 50 inchangé

    def test_binarization_threshold(self):
        """RUL ≤ 30 → faulty=1, RUL > 30 → faulty=0."""
        from src.data.cmapss_loader import CMAPSS_FAULTY_THRESHOLD

        rul = np.array([0, 15, 30, 31, 50, 125])
        faulty = (rul <= CMAPSS_FAULTY_THRESHOLD).astype(int)
        assert faulty[0] == 1   # RUL = 0 → faulty
        assert faulty[1] == 1   # RUL = 15 → faulty
        assert faulty[2] == 1   # RUL = 30 → faulty (borne inclusive)
        assert faulty[3] == 0   # RUL = 31 → healthy
        assert faulty[5] == 0   # RUL = 125 → healthy

    def test_domain_order(self):
        """4 domaines FD001–FD004 dans le bon ordre."""
        from src.data.cmapss_loader import DOMAIN_ORDER

        assert DOMAIN_ORDER == ["FD001", "FD002", "FD003", "FD004"]
        assert len(DOMAIN_ORDER) == 4

    def test_n_features(self):
        """Nombre de features sélectionnées = 5."""
        from src.data.cmapss_loader import CMAPSS_N_FEATURES_SELECTED

        assert CMAPSS_N_FEATURES_SELECTED == 5

    def test_rul_cap_value(self):
        """CMAPSS_RUL_CAP vaut 125 (convention littérature NASA)."""
        from src.data.cmapss_loader import CMAPSS_RUL_CAP

        assert CMAPSS_RUL_CAP == 125

    def test_faulty_threshold_value(self):
        """CMAPSS_FAULTY_THRESHOLD vaut 30 cycles."""
        from src.data.cmapss_loader import CMAPSS_FAULTY_THRESHOLD

        assert CMAPSS_FAULTY_THRESHOLD == 30

    def test_sensor_names_count(self):
        """21 capteurs s1–s21 définis dans SENSOR_NAMES."""
        from src.data.cmapss_loader import CMAPSS_N_FEATURES_RAW, SENSOR_NAMES

        assert len(SENSOR_NAMES) == CMAPSS_N_FEATURES_RAW
        assert CMAPSS_N_FEATURES_RAW == 21


@pytest.mark.skipif(
    not Path("data/raw/cmapss/train_FD001.txt").exists()
    and not Path("data/raw/CMAPSS Jet Engine Simulated Data/train_FD001.csv").exists(),
    reason="Données CMAPSS non disponibles",
)
class TestCmapssDataloaders:
    """Tests end-to-end avec données réelles (skipif données absentes)."""

    def test_get_cl_dataloaders_shape(self):
        """4 tâches, x.shape[1]==5, y.shape[1]==1, dtype float32."""
        import torch

        from src.data.cmapss_loader import get_cl_dataloaders

        data_dir = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
        tasks = get_cl_dataloaders(
            data_dir,
            Path("configs/cmapss_config.yaml"),
        )
        assert len(tasks) == 4  # FD001–FD004
        for task in tasks:
            x, y = next(iter(task["train_loader"]))
            assert x.shape[1] == 5      # top-5 features
            assert y.shape[1] == 1      # label binaire
            assert x.dtype == torch.float32

    def test_task_keys(self):
        """Chaque tâche expose les clés attendues."""
        from src.data.cmapss_loader import get_cl_dataloaders

        data_dir = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
        tasks = get_cl_dataloaders(data_dir, Path("configs/cmapss_config.yaml"))
        required_keys = {"task_id", "domain", "train_loader", "val_loader", "n_train", "n_val"}
        for task in tasks:
            assert required_keys.issubset(task.keys())

    def test_domain_names_match_order(self):
        """Les domaines des tâches correspondent à DOMAIN_ORDER."""
        from src.data.cmapss_loader import DOMAIN_ORDER, get_cl_dataloaders

        data_dir = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
        tasks = get_cl_dataloaders(data_dir, Path("configs/cmapss_config.yaml"))
        for i, task in enumerate(tasks):
            assert task["domain"] == DOMAIN_ORDER[i]
