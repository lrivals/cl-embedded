"""
tests/test_drift_detectors.py — Sprint 44 (S4402/S4403) : contrat + comportement des détecteurs.

Vérifie l'interface commune (`requires_label`, état borné constant), la détection après un saut connu
pour les deux familles, l'indépendance de W pour PSI (O(bins)), et le recouvrement des drift_points
EXACTS du synthétique S43 par ADWIN/KSWIN — dans une tolérance documentée.

    pytest tests/test_drift_detectors.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import DRIFT_CONFIGS, DRIFT_LOADERS
from src.models.drift import (
    ADWIN,
    DDM,
    EDDM,
    KSWIN,
    MMD,
    PSI,
    BaseDriftDetector,
    DriftVerdict,
    KSTest,
    PageHinkley,
    error_stream,
)
from src.models.drift.multivariate import MultiFeatureDriftDetector
from src.utils.config_loader import load_config

CHANGE_POINT = 1000
SEED = 0

UNSUPERVISED_FACTORIES = {
    "psi": lambda: PSI(bins=10),
    "ks_test": lambda: KSTest(),
    "kswin": lambda: KSWIN(),
    "mmd": lambda: MMD({"calib_percentile": 99}),
    "adwin": lambda: ADWIN(),
}


def _error_stream_jump(seed: int = SEED) -> np.ndarray:
    """Flux d'erreur à saut 0.1 → 0.5 en t=CHANGE_POINT."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.random(CHANGE_POINT) < 0.1, rng.random(CHANGE_POINT) < 0.5]
    ).astype(int)


def _feature_stream_jump(seed: int = SEED) -> np.ndarray:
    """Flux feature à saut de moyenne 0 → 3 en t=CHANGE_POINT."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, CHANGE_POINT), rng.normal(3, 1, CHANGE_POINT)])


def _drift_indices(detector: BaseDriftDetector, stream: np.ndarray) -> list[int]:
    return [i for i, x in enumerate(stream) if detector.update(float(x)) == DriftVerdict.DRIFT]


# --------------------------------------------------------------------------- #
# Contrat d'interface
# --------------------------------------------------------------------------- #
def test_verdict_enum_three_levels():
    assert {v.name for v in DriftVerdict} == {"NORMAL", "WARNING", "DRIFT"}


@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley])
def test_supervised_requires_label(cls):
    assert cls().requires_label is True


@pytest.mark.parametrize("factory", list(UNSUPERVISED_FACTORIES.values()))
def test_unsupervised_no_label(factory):
    assert factory().requires_label is False


@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley])
def test_supervised_state_o1_constant(cls):
    """État O(1) : get_state_bytes() constant, indépendant du nombre d'échantillons vus."""
    d = cls()
    before = d.get_state_bytes()
    rng = np.random.default_rng(1)
    for _ in range(500):
        d.update(float(rng.random() < 0.2))
    assert d.get_state_bytes() == before
    assert before <= 64  # borne O(1) stricte


@pytest.mark.parametrize("factory", list(UNSUPERVISED_FACTORIES.values()))
def test_unsupervised_state_bounded_constant(factory):
    """État borné : get_state_bytes() constant (capacité fixe) au fil du flux."""
    d = factory()
    d.set_params_from_reference(_feature_stream_jump()[:500])
    before = d.get_state_bytes()
    for x in _feature_stream_jump()[:600]:
        d.update(float(x))
    assert d.get_state_bytes() == before


# --------------------------------------------------------------------------- #
# Détection après saut — supervisés
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley])
def test_supervised_detects_after_error_jump(cls):
    stream = _error_stream_jump()
    idx = _drift_indices(cls(), stream)
    post = [i for i in idx if i >= CHANGE_POINT]
    assert post, f"{cls.__name__} n'a détecté aucun DRIFT après le saut d'erreur"


def test_page_hinkley_not_slower_than_ddm_on_sudden_jump():
    """Sur drift soudain, Page-Hinkley détecte au plus tard aussi vite que DDM (délai ≤)."""
    stream = _error_stream_jump()
    ph = [i for i in _drift_indices(PageHinkley(delta=0.005, lambda_=5), stream) if i >= CHANGE_POINT]
    ddm = [i for i in _drift_indices(DDM(), stream) if i >= CHANGE_POINT]
    assert ph and ddm
    assert (ph[0] - CHANGE_POINT) <= (ddm[0] - CHANGE_POINT)


def test_error_stream_helper():
    class _Model:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 1])
    e = error_stream(_Model(), X, y)
    assert e.tolist() == [0, 1, 0, 1, 1]


# --------------------------------------------------------------------------- #
# Détection après saut — non-supervisés
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factory", list(UNSUPERVISED_FACTORIES.items()))
def test_unsupervised_detects_after_feature_jump(name, factory):
    stream = _feature_stream_jump()
    d = factory()
    d.set_params_from_reference(stream[:500])
    verdicts = np.array([d.update(float(x)) == DriftVerdict.DRIFT for x in stream])
    assert verdicts[CHANGE_POINT:].any(), f"{name} : aucun DRIFT après le saut"
    # densité de drift post-changement >= pré-changement
    assert verdicts[CHANGE_POINT:].mean() >= verdicts[:CHANGE_POINT].mean()


def test_psi_state_independent_of_window():
    """PSI : état O(bins) — get_state_bytes() ne dépend que de bins, pas d'une taille de fenêtre."""
    small = PSI({"bins": 10, "block_size": 50})
    large = PSI({"bins": 10, "block_size": 5000})
    assert small.get_state_bytes() == large.get_state_bytes()
    assert PSI({"bins": 20}).get_state_bytes() > PSI({"bins": 10}).get_state_bytes()


def test_psi_requires_calibration():
    with pytest.raises(RuntimeError):
        PSI().update(1.0)


# --------------------------------------------------------------------------- #
# Recouvrement des drift_points EXACTS du synthétique S43
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory,tol", [(lambda: ADWIN(), 100), (lambda: KSWIN(), 100)])
def test_adwin_kswin_recover_synthetic_drift_points(factory, tol):
    """ADWIN/KSWIN retrouvent [1500,3000,4500] (feature 0) dans une tolérance documentée."""
    d = DRIFT_LOADERS["synthetic"](DRIFT_CONFIGS["synthetic"])
    X, drift_points = d.X, d.drift_points
    det = factory()
    det.set_params_from_reference(X[: drift_points[0], 0])
    detected = [
        i for i, x in enumerate(X[:, 0]) if det.update(float(x)) == DriftVerdict.DRIFT
    ]
    assert detected
    for gt in drift_points:
        nearest = min(abs(p - gt) for p in detected)
        assert nearest <= tol, f"drift_point {gt} non retrouvé (plus proche = {nearest} > {tol})"


# --------------------------------------------------------------------------- #
# Multivarié + config
# --------------------------------------------------------------------------- #
def test_multifeature_aggregation():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, (500, 3))
    stream = np.vstack([rng.normal(0, 1, (500, 3)), rng.normal(4, 1, (500, 3))])
    det = MultiFeatureDriftDetector(lambda: PSI(bins=10), n_features=3, config={"aggregation": "max"})
    det.set_params_from_reference(ref)
    assert det.requires_label is False
    drift = any(det.update(stream[i]) == DriftVerdict.DRIFT for i in range(len(stream)))
    assert drift


def test_config_has_all_detector_sections():
    cfg = load_config("configs/sprint44_drift_detection.yaml")
    for section in ("ddm", "eddm", "page_hinkley", "adwin", "kswin", "ks_test", "mmd", "psi"):
        assert section in cfg, f"section {section} absente de la config"
    assert cfg["seed"] == 42


# --------------------------------------------------------------------------- #
# Contrat d'interface exhaustif + déterminisme (S4406)
# --------------------------------------------------------------------------- #
from src.models.drift import DRIFT_DETECTORS  # noqa: E402


@pytest.mark.parametrize("name,cls", list(DRIFT_DETECTORS.items()))
def test_all_registered_detectors_conform_interface(name, cls):
    """Chaque détecteur du registre respecte l'interface BaseDriftDetector (S4401)."""
    det = cls()
    assert isinstance(det, BaseDriftDetector)
    for method in ("update", "reset", "get_state_bytes", "set_params_from_reference"):
        assert callable(getattr(det, method)), f"{name} : {method} manquant"
    assert isinstance(det.requires_label, bool)
    assert isinstance(det.get_state_bytes(), int)


@pytest.mark.parametrize("cls", [DDM, EDDM, PageHinkley])
def test_supervised_determinism_seed(cls):
    """Deux passes sur le même flux d'erreur (seed fixe) → verdicts identiques (déterminisme)."""
    stream = _error_stream_jump(seed=42)
    assert _drift_indices(cls(), stream) == _drift_indices(cls(), stream)


@pytest.mark.parametrize("name,factory", list(UNSUPERVISED_FACTORIES.items()))
def test_unsupervised_determinism_seed(name, factory):
    """Non-supervisés : même référence + même flux → verdicts identiques (déterminisme seed 42)."""
    stream = _feature_stream_jump(seed=42)

    def _run():
        d = factory()
        d.set_params_from_reference(stream[:500])
        return [d.update(float(x)) == DriftVerdict.DRIFT for x in stream]

    assert _run() == _run()


def test_kswin_state_grows_with_window():
    """État borné O(W) : KSWIN à plus grande fenêtre = plus d'octets (distinct de l'O(1) supervisé)."""
    small = KSWIN({"window_size": 50})
    large = KSWIN({"window_size": 200})
    assert large.get_state_bytes() > small.get_state_bytes()
    # …tandis que les supervisés restent strictement O(1) (borne 64 B).
    assert DDM().get_state_bytes() <= 64 and PageHinkley().get_state_bytes() <= 64
