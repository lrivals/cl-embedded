"""Tests Sprint 32 / S3207 — paramétrage du seuil RUL→`faulty`.

Verrouille :
  * non-régression : seuil par défaut == constante native de chaque loader ;
  * binarisation conforme (opérateur natif : CMAPSS/Pronostia ``<=``, Battery ``<``) ;
  * monotonie du ``positive_ratio`` avec le seuil ;
  * configs de balayage ne différant que par le champ seuil.

Les tests touchant ``data/raw/`` sont marqués ``skipif`` (données réelles requises).

Exécution :
    pytest tests/test_threshold_sweep.py -v
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

_CMAPSS_DIR = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
_BATTERY_CSV = Path("data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv")
_PRONOSTIA_DIR = Path("data/raw/Pronostia dataset/binaries")


class TestThresholdDefaultsUnchanged:
    """Le seuil par défaut de chaque loader reste la constante native (non-régression)."""

    def test_cmapss_default_equals_constant(self):
        from src.data.cmapss_loader import CMAPSS_FAULTY_THRESHOLD, _load_raw

        default = inspect.signature(_load_raw).parameters["faulty_threshold"].default
        assert default == CMAPSS_FAULTY_THRESHOLD == 30

    def test_battery_default_equals_constant(self):
        from src.data.battery_dataset import RUL_FAILURE_THRESHOLD, load_raw_dataset

        default = inspect.signature(load_raw_dataset).parameters["rul_failure_threshold"].default
        assert default == RUL_FAILURE_THRESHOLD == 200

    def test_pronostia_default_mode_is_failure_ratio(self):
        from src.data.pronostia_dataset import load_condition_features

        sig = inspect.signature(load_condition_features).parameters
        assert sig["label_mode"].default == "failure_ratio"  # rul_threshold = opt-in
        assert sig["faulty_threshold"].default is None


class TestThresholdOperators:
    """Opérateur natif conservé : CMAPSS/Pronostia inclusif ``<=``, Battery exclusif ``<``."""

    @pytest.mark.parametrize("thr", [10, 30, 50])
    def test_cmapss_inclusive(self, thr):
        rul = np.array([thr - 1, thr, thr + 1])
        faulty = (rul <= thr).astype(int)
        assert faulty.tolist() == [1, 1, 0]  # seuil inclus

    @pytest.mark.parametrize("thr", [67, 200, 333])
    def test_battery_exclusive(self, thr):
        rul = np.array([thr - 1, thr, thr + 1])
        faulty = (rul < thr).astype(int)
        assert faulty.tolist() == [1, 0, 0]  # seuil exclu


class TestSweepConfigsOnlyThresholdDiffers:
    """Deux configs de balayage d'un même dataset ne diffèrent que par le seuil."""

    @pytest.mark.parametrize("dataset,field,a,b", [
        ("cmapss", "faulty_threshold", 10, 50),
        ("battery", "rul_failure_threshold", 67, 333),
        ("pronostia", "faulty_threshold", 24, 120),
    ])
    def test_only_threshold_field_differs(self, dataset, field, a, b):
        ca = yaml.safe_load(Path(f"configs/sweep/{dataset}_thr{a}.yaml").read_text())
        cb = yaml.safe_load(Path(f"configs/sweep/{dataset}_thr{b}.yaml").read_text())
        da, db = ca.get("data", {}), cb.get("data", {})
        assert da.get(field) == a and db.get(field) == b
        # Tout le reste de la section data est identique.
        diffs = {k for k in set(da) | set(db) if da.get(k) != db.get(k)}
        assert diffs == {field}, f"champs divergents inattendus : {diffs}"


class TestBatteryFeatureSubset:
    """Le sous-ensemble board Battery sélectionne bien 5 features sur 7."""

    def test_subset_has_five(self):
        sub = yaml.safe_load(Path("configs/battery_feature_subset.yaml").read_text())
        assert sub["n_features_selected"] == 5
        assert len(sub["feature_indices"]) == 5
        assert all(0 <= i < 7 for i in sub["feature_indices"])


@pytest.mark.skipif(not _CMAPSS_DIR.exists() or not _BATTERY_CSV.exists()
                    or not _PRONOSTIA_DIR.exists(), reason="données brutes requises")
class TestPositiveRatioMonotonic:
    """``positive_ratio`` croît avec le seuil (RUL plus élevé ⇒ plus de positifs)."""

    @pytest.mark.parametrize("dataset", ["cmapss", "pronostia", "battery"])
    def test_monotonic(self, dataset):
        from scripts.generate_threshold_sweep_configs import SWEEPS
        from scripts.run_threshold_sweep import _positive_ratio

        thresholds = sorted(SWEEPS[dataset][2])
        ratios = [_positive_ratio(dataset, t) for t in thresholds]
        ratios = [r for r in ratios if r is not None]
        assert len(ratios) >= 3
        assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:])), \
            f"{dataset} : positive_ratio non monotone {ratios}"
