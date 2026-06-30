"""
tests/test_model_pair.py — Tests unitaires pour src/ensemble/model_pair.py (Sprint 30 S3001).

Vérifie :
    - import et construction (modes / règles valides et invalides) ;
    - shapes de predict_individual avec un MahalanobisDetector réel + un classifieur stub ;
    - les 4 règles de fusion (or / and / soft_vote / weighted) sur cas connus ;
    - predict_proba ∈ [0, 1] ;
    - la binarisation multiclasse → normal-vs-fault en mode 'binary'.

Exécution :
    pytest tests/test_model_pair.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.model_pair import ModelPair, _binarize_labels
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector

# ----------------------------------------------------------------------
# Stubs classifieur supervisé (interface duck-typée)
# ----------------------------------------------------------------------


class _LabelStub:
    """Classifieur supervisé minimal : predict → labels prédéfinis."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.asarray(labels)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._labels[: len(x)]


class _ProbaStub:
    """Classifieur exposant predict_proba (proba de panne ∈ [0, 1])."""

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = np.asarray(proba, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self._proba[: len(x)] >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._proba[: len(x)]


@pytest.fixture
def detector() -> MahalanobisDetector:
    """Mahalanobis fit sur un nuage gaussien centré ; seuil calculé sur Task 0."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(200, 4)).astype(np.float32)
    det = MahalanobisDetector({"anomaly_percentile": 95})
    det.fit_task(X, task_id=0)
    return det


# ----------------------------------------------------------------------
# Binarisation (convention Sprint 28)
# ----------------------------------------------------------------------


def test_binarize_multiclass_normal_vs_fault():
    """Classe normale = plus petite valeur ; tout le reste = panne (1)."""
    y = np.array([0, 1, 2, 3, 0])
    np.testing.assert_array_equal(_binarize_labels(y), [0, 1, 1, 1, 0])


def test_binarize_already_binary():
    y = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(_binarize_labels(y), [0, 1, 1, 0])


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_invalid_mode_raises(detector):
    with pytest.raises(ValueError):
        ModelPair(detector, _LabelStub(np.zeros(5)), mode="bogus")


def test_invalid_rule_raises(detector):
    with pytest.raises(ValueError):
        ModelPair(detector, _LabelStub(np.zeros(5)), fusion_rule="majority")


# ----------------------------------------------------------------------
# predict_individual
# ----------------------------------------------------------------------


def test_predict_individual_shapes_and_binarization(detector):
    x = np.zeros((6, 4), dtype=np.float32)
    sup_labels = np.array([0, 1, 2, 3, 0, 1])  # multiclasse
    pair = ModelPair(detector, _LabelStub(sup_labels), mode="binary")
    pred_maha, pred_sup = pair.predict_individual(x)

    assert pred_maha.shape == (6,)
    assert pred_sup.shape == (6,)
    # mode binary : multiclasse réduit à normal-vs-fault.
    np.testing.assert_array_equal(pred_sup, [0, 1, 1, 1, 0, 1])
    assert set(np.unique(pred_maha)) <= {0, 1}


# ----------------------------------------------------------------------
# Règles de fusion
# ----------------------------------------------------------------------


def test_fusion_or_and_logic(detector):
    """OR = union des détections, AND = intersection (cas connus construits à la main)."""
    x = np.zeros((4, 4), dtype=np.float32)

    # Force des prédictions Mahalanobis connues en remplaçant predict.
    maha_pred = np.array([1, 1, 0, 0])
    detector.predict = lambda _x: maha_pred[: len(_x)]  # type: ignore[method-assign]
    sup_pred = np.array([1, 0, 1, 0])

    pair = ModelPair(detector, _LabelStub(sup_pred), mode="binary")
    np.testing.assert_array_equal(pair.predict_ensemble(x, rule="or"), [1, 1, 1, 0])
    np.testing.assert_array_equal(pair.predict_ensemble(x, rule="and"), [1, 0, 0, 0])


def test_fusion_soft_vote_and_weighted_range(detector):
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    pair = ModelPair(detector, _ProbaStub(proba), mode="binary", fusion_rule="weighted")

    soft = pair.predict_ensemble(np.zeros((4, 4), np.float32), rule="soft_vote")
    weighted = pair.predict_ensemble(np.zeros((4, 4), np.float32), rule="weighted")
    assert soft.shape == (4,)
    assert set(np.unique(soft)) <= {0, 1}
    assert set(np.unique(weighted)) <= {0, 1}


def test_weighted_respects_config_weights(detector):
    """Poids ~tout sur le supervisé → l'ensemble suit la proba supervisée."""
    proba = np.array([0.95, 0.95, 0.05, 0.05])
    pair = ModelPair(
        detector,
        _ProbaStub(proba),
        mode="binary",
        config={"fusion_rule": "weighted", "weights": [0.01, 0.99]},
    )
    out = pair.predict_ensemble(np.zeros((4, 4), np.float32))
    np.testing.assert_array_equal(out, [1, 1, 0, 0])


# ----------------------------------------------------------------------
# predict_proba
# ----------------------------------------------------------------------


def test_predict_proba_in_unit_interval(detector):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(20, 4)).astype(np.float32)
    proba = rng.uniform(0, 1, size=20)
    pair = ModelPair(detector, _ProbaStub(proba), mode="binary", fusion_rule="weighted")
    p = pair.predict_proba(x)
    assert p.shape == (20,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
