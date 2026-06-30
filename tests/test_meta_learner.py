"""
tests/test_meta_learner.py — Tests unitaires pour src/ensemble/meta_learner.py (Sprint 31 S3101/S3112).

Vérifie :
    - import et construction (kinds valides / invalides, lecture via config) ;
    - shapes de fit / predict / predict_proba (logreg et mlp) ;
    - build_meta_features : colonnes attendues, bornes [0, 1], features inconnues rejetées ;
    - anti-fuite : méta entraîné sur un split disjoint du split d'évaluation ;
    - méta ≥ meilleure base individuelle sur un cas synthétique où chaque base est partiellement
      correcte (le méta doit pouvoir apprendre l'arbitrage) ;
    - export_weights : clés et shapes correctes pour logreg et mlp.

Exécution :
    pytest tests/test_meta_learner.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.ensemble.meta_learner import (
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
    MetaLearner,
    build_meta_features,
)
from src.ensemble.model_pair import ModelPair
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector

# ----------------------------------------------------------------------
# Stub supervisé (interface duck-typée, identique à test_model_pair.py)
# ----------------------------------------------------------------------


class _ProbaStub:
    """Classifieur exposant predict_proba (proba de panne ∈ [0, 1])."""

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = np.asarray(proba, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self._proba[: len(x)] >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._proba[: len(x)]


@pytest.fixture
def pair() -> ModelPair:
    """Paire Mahalanobis réel + stub supervisé proba, alignée sur 40 échantillons."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0.0, 1.0, size=(200, 4)).astype(np.float32)
    det = MahalanobisDetector({"anomaly_percentile": 95})
    det.fit_task(X_train, task_id=0)
    proba = rng.uniform(0.0, 1.0, size=40)
    return ModelPair(detector=det, classifier=_ProbaStub(proba), mode="binary")


@pytest.fixture
def eval_X() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 1.5, size=(40, 4)).astype(np.float32)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValueError):
        MetaLearner(kind="svm")


def test_config_overrides_kwargs() -> None:
    ml = MetaLearner(kind="logreg", config={"kind": "mlp", "hidden_size": 4})
    assert ml.kind == "mlp"
    assert ml.hidden_size == 4


def test_default_features() -> None:
    ml = MetaLearner()
    assert ml.input_features == list(DEFAULT_FEATURES)


# ----------------------------------------------------------------------
# build_meta_features
# ----------------------------------------------------------------------


def test_build_meta_features_shape_and_bounds(pair: ModelPair, eval_X: np.ndarray) -> None:
    meta_X, names = build_meta_features(pair, eval_X, AVAILABLE_FEATURES)
    assert meta_X.shape == (40, len(AVAILABLE_FEATURES))
    assert names == list(AVAILABLE_FEATURES)
    assert meta_X.dtype == np.float32
    # Toutes les features sont bornées [0, 1] (portabilité MCU, pas de scaler).
    assert meta_X.min() >= 0.0 - 1e-6
    assert meta_X.max() <= 1.0 + 1e-6


def test_build_meta_features_unknown_raises(pair: ModelPair, eval_X: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_meta_features(pair, eval_X, ["p_maha", "bogus"])


# ----------------------------------------------------------------------
# fit / predict / predict_proba
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_fit_predict_shapes(kind: str) -> None:
    rng = np.random.default_rng(0)
    meta_X = rng.uniform(0.0, 1.0, size=(100, 4)).astype(np.float32)
    y = (meta_X[:, 0] + meta_X[:, 1] > 1.0).astype(int)
    ml = MetaLearner(kind=kind, hidden_size=8).fit(meta_X, y)
    preds = ml.predict(meta_X)
    proba = ml.predict_proba(meta_X)
    assert preds.shape == (100,)
    assert proba.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        MetaLearner().predict(np.zeros((2, 4), dtype=np.float32))


# ----------------------------------------------------------------------
# Anti-fuite : méta fit sur split disjoint du split d'évaluation
# ----------------------------------------------------------------------


def test_no_leakage_disjoint_splits() -> None:
    rng = np.random.default_rng(1)
    meta_X = rng.uniform(0.0, 1.0, size=(120, 4)).astype(np.float32)
    y = (meta_X[:, 0] > 0.5).astype(int)
    idx = rng.permutation(120)
    fit_idx, eval_idx = idx[:80], idx[80:]
    # Les deux splits sont disjoints (aucun échantillon de fit dans l'éval).
    assert set(fit_idx.tolist()).isdisjoint(set(eval_idx.tolist()))
    ml = MetaLearner(kind="logreg").fit(meta_X[fit_idx], y[fit_idx])
    # Le méta produit bien des prédictions sur des données jamais vues à l'entraînement.
    preds = ml.predict(meta_X[eval_idx])
    assert preds.shape == eval_idx.shape


# ----------------------------------------------------------------------
# Méta ≥ meilleure base individuelle (cas synthétique arbitrable)
# ----------------------------------------------------------------------


def test_meta_beats_or_equals_best_individual() -> None:
    """Construit un cas où chaque base est correcte sur une moitié des échantillons et où une
    feature d'arbitrage indique laquelle suivre. Un méta capable d'apprendre doit ≥ max(base)."""
    rng = np.random.default_rng(123)
    n = 400
    y = rng.integers(0, 2, size=n)
    region = rng.integers(0, 2, size=n)  # 0 → base A fiable ; 1 → base B fiable

    # Base A : correcte si region == 0, sinon bruitée ; idem B en miroir.
    pred_a = np.where(region == 0, y, rng.integers(0, 2, size=n))
    pred_b = np.where(region == 1, y, rng.integers(0, 2, size=n))

    # Features méta : prédictions des 2 bases + indicateur de région (l'arbitre).
    meta_X = np.column_stack([pred_a, pred_b, region]).astype(np.float32)

    # Split out-of-fold.
    idx = rng.permutation(n)
    fit_idx, eval_idx = idx[:280], idx[280:]
    ml = MetaLearner(kind="mlp", hidden_size=8).fit(meta_X[fit_idx], y[fit_idx])
    f1_meta = f1_score(y[eval_idx], ml.predict(meta_X[eval_idx]), zero_division=0)
    f1_a = f1_score(y[eval_idx], pred_a[eval_idx], zero_division=0)
    f1_b = f1_score(y[eval_idx], pred_b[eval_idx], zero_division=0)
    assert f1_meta >= max(f1_a, f1_b) - 1e-9


# ----------------------------------------------------------------------
# export_weights
# ----------------------------------------------------------------------


def test_export_weights_logreg() -> None:
    rng = np.random.default_rng(2)
    meta_X = rng.uniform(0.0, 1.0, size=(60, 4)).astype(np.float32)
    y = (meta_X[:, 0] > 0.5).astype(int)
    w = MetaLearner(kind="logreg").fit(meta_X, y).export_weights()
    assert w["kind"] == "logreg"
    assert w["w"].shape == (4,)
    assert w["w"].dtype == np.float32
    assert isinstance(w["b"], float)
    assert w["feature_names"] == list(DEFAULT_FEATURES)


def test_export_weights_mlp() -> None:
    rng = np.random.default_rng(3)
    meta_X = rng.uniform(0.0, 1.0, size=(80, 4)).astype(np.float32)
    y = (meta_X[:, 0] + meta_X[:, 1] > 1.0).astype(int)
    w = MetaLearner(kind="mlp", hidden_size=6).fit(meta_X, y).export_weights()
    assert w["kind"] == "mlp"
    assert w["w1"].shape == (6, 4)
    assert w["b1"].shape == (6,)
    assert w["w2"].shape == (1, 6)
    assert isinstance(w["b2"], float)
    assert all(arr.dtype == np.float32 for arr in (w["w1"], w["b1"], w["w2"]))
