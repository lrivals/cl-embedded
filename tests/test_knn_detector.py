"""Tests unitaires pour KNNDetector (S5-03)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def X0() -> np.ndarray:
    """Données synthétiques pour Task 0 (100 échantillons, 4 features)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 4)).astype(np.float32)


@pytest.fixture
def X1() -> np.ndarray:
    """Données synthétiques pour Task 1 (80 échantillons, 4 features)."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((80, 4)).astype(np.float32)


@pytest.fixture
def y0(X0: np.ndarray) -> np.ndarray:
    """Labels binaires synthétiques pour Task 0."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=len(X0)).astype(np.int64)


@pytest.fixture
def base_config() -> dict:
    return {
        "n_neighbors": 5,
        "metric": "euclidean",
        "anomaly_percentile": 95,
        "cl_strategy": "accumulate",
    }


@pytest.fixture
def fitted_detector(base_config: dict, X0: np.ndarray):
    """KNNDetector entraîné sur Task 0."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector(base_config)
    det.fit_task(X0, task_id=0)
    return det


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import() -> None:
    """Import sans erreur."""
    from src.models.unsupervised import KNNDetector  # noqa: F401


def test_fit_task0_builds_index_and_threshold(base_config: dict, X0: np.ndarray) -> None:
    """fit_task sur Task 0 construit l'index KNN et calcule le seuil."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector(base_config)
    det.fit_task(X0, task_id=0)

    assert det.nn_ is not None
    assert det.threshold_ is not None
    assert det.X_ref_ is not None
    assert det.task_id_ == 0


def test_accumulate_grows_xref(base_config: dict, X0: np.ndarray, X1: np.ndarray) -> None:
    """Stratégie accumulate : X_ref_ croît entre tâches."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector({**base_config, "cl_strategy": "accumulate"})
    det.fit_task(X0, task_id=0)
    det.fit_task(X1, task_id=1)

    assert det.X_ref_ is not None
    assert len(det.X_ref_) == len(X0) + len(X1)


def test_refit_replaces_xref(base_config: dict, X0: np.ndarray, X1: np.ndarray) -> None:
    """Stratégie refit : X_ref_ est remplacé par la tâche courante."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector({**base_config, "cl_strategy": "refit"})
    det.fit_task(X0, task_id=0)
    det.fit_task(X1, task_id=1)

    assert det.X_ref_ is not None
    assert len(det.X_ref_) == len(X1)


def test_predict_shape_and_values(fitted_detector, X0: np.ndarray) -> None:
    """predict() retourne shape [N], dtype int64, valeurs ∈ {0, 1}."""
    preds = fitted_detector.predict(X0)

    assert preds.shape == (len(X0),)
    assert preds.dtype == np.int64
    assert set(np.unique(preds)).issubset({0, 1})


def test_anomaly_score_dtype(fitted_detector, X0: np.ndarray) -> None:
    """anomaly_score() retourne dtype float32."""
    scores = fitted_detector.anomaly_score(X0)

    assert scores.shape == (len(X0),)
    assert scores.dtype == np.float32


def test_score_in_range(fitted_detector, X0: np.ndarray, y0: np.ndarray) -> None:
    """score() retourne un float ∈ [0.0, 1.0]."""
    acc = fitted_detector.score(X0, y0)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_predict_without_fit_raises(base_config: dict, X0: np.ndarray) -> None:
    """predict() lève RuntimeError si fit_task n'a pas été appelé."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector(base_config)
    with pytest.raises(RuntimeError):
        det.predict(X0)


def test_save_load_roundtrip(
    fitted_detector, X0: np.ndarray, tmp_path: Path
) -> None:
    """save + load produit des prédictions identiques."""
    save_path = tmp_path / "knn_task0.pkl"
    fitted_detector.save(save_path)

    from src.models.unsupervised import KNNDetector

    loaded = KNNDetector.load(save_path)
    np.testing.assert_array_equal(
        fitted_detector.predict(X0),
        loaded.predict(X0),
    )


def test_count_parameters_before_fit(base_config: dict) -> None:
    """count_parameters() retourne 0 avant fit_task."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector(base_config)
    assert det.count_parameters() == 0


def test_count_parameters_accumulate(
    base_config: dict, X0: np.ndarray, X1: np.ndarray
) -> None:
    """count_parameters() = n_ref × n_features, croît avec accumulate."""
    from src.models.unsupervised import KNNDetector

    det = KNNDetector({**base_config, "cl_strategy": "accumulate"})
    det.fit_task(X0, task_id=0)
    params_after_t0 = det.count_parameters()
    assert params_after_t0 == len(X0) * X0.shape[1]

    det.fit_task(X1, task_id=1)
    params_after_t1 = det.count_parameters()
    assert params_after_t1 == (len(X0) + len(X1)) * X0.shape[1]
    assert params_after_t1 > params_after_t0


def test_summary_returns_string(fitted_detector) -> None:
    """summary() retourne une chaîne non vide."""
    s = fitted_detector.summary()
    assert isinstance(s, str)
    assert len(s) > 0
