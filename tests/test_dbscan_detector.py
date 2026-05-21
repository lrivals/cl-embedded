"""
tests/test_dbscan_detector.py — Tests unitaires pour DBSCANDetector.

Valide : initialisation, fit_task (refit + accumulate), predict_score,
predict, on_task_end, et comportement multi-tâches.

Aucun accès aux données brutes — tableaux numpy synthétiques uniquement.

Exécution :
    pytest tests/test_dbscan_detector.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.unsupervised.dbscan_detector import DBSCANDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DIM = 4
_N_TRAIN = 60
_N_TEST = 20


def _make_config(cl_strategy: str = "refit") -> dict:
    return {
        "eps": None,  # auto-estimation kNN elbow
        "min_samples": 3,
        "metric": "euclidean",
        "algorithm": "auto",
        "anomaly_threshold": None,
        "anomaly_percentile": 95,
        "cl_strategy": cl_strategy,
        "eps_knn_k": 4,
    }


@pytest.fixture
def X_train() -> np.ndarray:
    rng = np.random.default_rng(10)
    return rng.normal(0.0, 0.3, size=(_N_TRAIN, _DIM)).astype(np.float32)


@pytest.fixture
def X_test() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(0.0, 1.0, size=(_N_TEST, _DIM)).astype(np.float32)


@pytest.fixture
def detector_refit() -> DBSCANDetector:
    return DBSCANDetector(_make_config("refit"))


@pytest.fixture
def detector_accumulate() -> DBSCANDetector:
    return DBSCANDetector(_make_config("accumulate"))


# ---------------------------------------------------------------------------
# Tests d'initialisation
# ---------------------------------------------------------------------------


def test_init_attributes() -> None:
    """Les attributs publics sont None / -1 avant tout entraînement."""
    det = DBSCANDetector(_make_config())
    assert det.core_points_ is None
    assert det.task_id_ == -1
    assert det.n_features_ == 0


def test_init_cl_strategy() -> None:
    """cl_strategy est correctement stocké."""
    assert DBSCANDetector(_make_config("refit")).cl_strategy == "refit"
    assert DBSCANDetector(_make_config("accumulate")).cl_strategy == "accumulate"


# ---------------------------------------------------------------------------
# Tests fit_task
# ---------------------------------------------------------------------------


def test_fit_task_sets_core_points(detector_refit: DBSCANDetector, X_train: np.ndarray) -> None:
    """Après fit_task, core_points_ est un array numpy."""
    detector_refit.fit_task(X_train, task_id=0)
    assert detector_refit.core_points_ is not None
    assert isinstance(detector_refit.core_points_, np.ndarray)


def test_fit_task_sets_n_features(detector_refit: DBSCANDetector, X_train: np.ndarray) -> None:
    """fit_task fixe n_features_ à la dimension d'entrée."""
    detector_refit.fit_task(X_train, task_id=0)
    assert detector_refit.n_features_ == _DIM


def test_fit_task_sets_threshold(detector_refit: DBSCANDetector, X_train: np.ndarray) -> None:
    """Après fit_task sur task_id=0, threshold_ est calculé."""
    detector_refit.fit_task(X_train, task_id=0)
    assert detector_refit.threshold_ is not None


def test_fit_task_returns_self(detector_refit: DBSCANDetector, X_train: np.ndarray) -> None:
    """fit_task retourne self (interface fluente)."""
    result = detector_refit.fit_task(X_train, task_id=0)
    assert result is detector_refit


def test_fit_task_refit_task1(X_train: np.ndarray) -> None:
    """Stratégie refit : fit_task sur task_id=1 repart de zéro sans erreur."""
    det = DBSCANDetector(_make_config("refit"))
    det.fit_task(X_train, task_id=0)
    rng = np.random.default_rng(20)
    X_task1 = rng.normal(2.0, 0.3, size=(_N_TRAIN, _DIM)).astype(np.float32)
    det.fit_task(X_task1, task_id=1)
    assert det.task_id_ == 1


def test_fit_task_accumulate_grows_data(detector_accumulate: DBSCANDetector, X_train: np.ndarray) -> None:
    """Stratégie accumulate : les données s'accumulent entre les tâches."""
    detector_accumulate.fit_task(X_train, task_id=0)
    n_after_task0 = len(detector_accumulate._X_accumulated)
    rng = np.random.default_rng(21)
    X_task1 = rng.normal(0.0, 0.3, size=(_N_TRAIN, _DIM)).astype(np.float32)
    detector_accumulate.fit_task(X_task1, task_id=1)
    n_after_task1 = len(detector_accumulate._X_accumulated)
    assert n_after_task1 > n_after_task0


# ---------------------------------------------------------------------------
# Tests predict_score
# ---------------------------------------------------------------------------


def test_predict_score_shape(detector_refit: DBSCANDetector, X_train: np.ndarray, X_test: np.ndarray) -> None:
    """predict_score retourne un array de forme (N_test,)."""
    detector_refit.fit_task(X_train, task_id=0)
    scores = detector_refit.predict_score(X_test)
    assert scores.shape == (len(X_test),)


def test_predict_score_non_negative(detector_refit: DBSCANDetector, X_train: np.ndarray, X_test: np.ndarray) -> None:
    """Les scores de distance sont ≥ 0."""
    detector_refit.fit_task(X_train, task_id=0)
    scores = detector_refit.predict_score(X_test)
    assert np.all(scores >= 0.0)


# ---------------------------------------------------------------------------
# Tests predict
# ---------------------------------------------------------------------------


def test_predict_binary_output(detector_refit: DBSCANDetector, X_train: np.ndarray, X_test: np.ndarray) -> None:
    """predict retourne un array binaire (0 = normal, 1 = anomalie)."""
    detector_refit.fit_task(X_train, task_id=0)
    preds = detector_refit.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(preds.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests on_task_end
# ---------------------------------------------------------------------------


def test_on_task_end_no_crash(detector_refit: DBSCANDetector, X_train: np.ndarray) -> None:
    """on_task_end() s'exécute sans erreur après fit_task."""
    detector_refit.fit_task(X_train, task_id=0)
    detector_refit.on_task_end()  # ne doit pas lever d'exception
