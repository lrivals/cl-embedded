"""
tests/test_ewc_oneclass.py — Tests unitaires pour EWCOneClassDetector.

Valide : initialisation, fit_task (tâche 0 et 1+), seuil d'anomalie,
predict_score, predict, on_task_end (Fisher), pénalité EWC, from_config.

Aucun accès aux données brutes — tenseurs synthétiques uniquement.

Exécution :
    pytest tests/test_ewc_oneclass.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ewc.ewc_oneclass import EWCOneClassDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INPUT_DIM = 4
_N_NORMAL = 80
_N_TEST = 30


@pytest.fixture
def detector() -> EWCOneClassDetector:
    """Détecteur minimal avec n_epochs=2 pour garder les tests rapides."""
    return EWCOneClassDetector(input_dim=_INPUT_DIM, hidden_dim=16, latent_dim=4, n_epochs=2)


@pytest.fixture
def X_normal() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 0.5, size=(_N_NORMAL, _INPUT_DIM)).astype(np.float32)


@pytest.fixture
def X_test() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(0.0, 1.0, size=(_N_TEST, _INPUT_DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests d'initialisation
# ---------------------------------------------------------------------------


def test_init_attributes(detector: EWCOneClassDetector) -> None:
    """Les attributs publics sont None / -1 avant tout entraînement."""
    assert detector.fisher_ is None
    assert detector.params_star_ is None
    assert detector.threshold_ is None
    assert detector.task_id_ == -1


def test_init_input_dim() -> None:
    """input_dim est respecté (pas d'erreur à l'instanciation)."""
    for dim in [4, 9, 13]:
        det = EWCOneClassDetector(input_dim=dim, n_epochs=1)
        assert det._input_dim == dim


# ---------------------------------------------------------------------------
# Tests fit_task
# ---------------------------------------------------------------------------


def test_fit_task_0_sets_threshold(detector: EWCOneClassDetector, X_normal: np.ndarray) -> None:
    """Après fit_task sur task_id=0, threshold_ est initialisé."""
    detector.fit_task(X_normal, task_id=0)
    assert detector.threshold_ is not None
    assert isinstance(detector.threshold_, float)


def test_fit_task_0_sets_task_id(detector: EWCOneClassDetector, X_normal: np.ndarray) -> None:
    """fit_task met à jour task_id_."""
    detector.fit_task(X_normal, task_id=0)
    assert detector.task_id_ == 0


def test_fit_task_returns_self(detector: EWCOneClassDetector, X_normal: np.ndarray) -> None:
    """fit_task retourne self (interface fluente)."""
    result = detector.fit_task(X_normal, task_id=0)
    assert result is detector


def test_fit_task_1_with_ewc(X_normal: np.ndarray) -> None:
    """fit_task sur task_id=1 avec lambda_ewc>0 s'exécute sans erreur."""
    det = EWCOneClassDetector(input_dim=_INPUT_DIM, lambda_ewc=400.0, n_epochs=2)
    det.fit_task(X_normal, task_id=0)
    rng = np.random.default_rng(2)
    X_task1 = rng.normal(1.0, 0.5, size=(_N_NORMAL, _INPUT_DIM)).astype(np.float32)
    det.fit_task(X_task1, task_id=1)
    assert det.task_id_ == 1


def test_fit_task_1_no_ewc(X_normal: np.ndarray) -> None:
    """lambda_ewc=0 (pas de régularisation EWC) s'exécute sans erreur."""
    det = EWCOneClassDetector(input_dim=_INPUT_DIM, lambda_ewc=0.0, n_epochs=2)
    det.fit_task(X_normal, task_id=0)
    rng = np.random.default_rng(3)
    X_task1 = rng.normal(0.0, 1.0, size=(_N_NORMAL, _INPUT_DIM)).astype(np.float32)
    det.fit_task(X_task1, task_id=1)


# ---------------------------------------------------------------------------
# Tests on_task_end
# ---------------------------------------------------------------------------


def test_on_task_end_sets_fisher(detector: EWCOneClassDetector, X_normal: np.ndarray) -> None:
    """on_task_end() calcule fisher_ (dict non vide)."""
    detector.fit_task(X_normal, task_id=0)
    detector.on_task_end()
    assert detector.fisher_ is not None
    assert len(detector.fisher_) > 0


def test_on_task_end_sets_params_star(detector: EWCOneClassDetector, X_normal: np.ndarray) -> None:
    """on_task_end() sauvegarde params_star_ (snapshot θ*)."""
    detector.fit_task(X_normal, task_id=0)
    detector.on_task_end()
    assert detector.params_star_ is not None
    assert len(detector.params_star_) > 0


# ---------------------------------------------------------------------------
# Tests predict_score et predict
# ---------------------------------------------------------------------------


def test_predict_score_shape(detector: EWCOneClassDetector, X_normal: np.ndarray, X_test: np.ndarray) -> None:
    """predict_score retourne un array de forme (N_test,)."""
    detector.fit_task(X_normal, task_id=0)
    scores = detector.predict_score(X_test)
    assert scores.shape == (len(X_test),)


def test_predict_score_non_negative(detector: EWCOneClassDetector, X_normal: np.ndarray, X_test: np.ndarray) -> None:
    """Les scores MSE de reconstruction sont ≥ 0."""
    detector.fit_task(X_normal, task_id=0)
    scores = detector.predict_score(X_test)
    assert np.all(scores >= 0.0)


def test_predict_binary_output(detector: EWCOneClassDetector, X_normal: np.ndarray, X_test: np.ndarray) -> None:
    """predict retourne un array binaire (0 ou 1)."""
    detector.fit_task(X_normal, task_id=0)
    preds = detector.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(preds.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests from_config
# ---------------------------------------------------------------------------


def test_from_config_basic() -> None:
    """from_config instancie correctement depuis un dictionnaire minimal."""
    config = {
        "input_dim": _INPUT_DIM,
        "MODEL": {"HIDDEN_DIM": 16, "LATENT_DIM": 4, "LAMBDA_EWC": 100.0, "THRESHOLD_PERCENTILE": 90.0},
        "TRAINING": {"N_EPOCHS": 2, "LR": 1e-3, "DEVICE": "cpu", "BATCH_SIZE": 16},
    }
    det = EWCOneClassDetector.from_config(config)
    assert det._input_dim == _INPUT_DIM
    assert det._hidden_dim == 16
    assert det._lambda_ewc == 100.0
