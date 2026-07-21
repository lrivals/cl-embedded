"""
src/models/drift/ — Sprint 44 : détecteurs de drift en ligne derrière une interface commune.

Deux familles sur l'axe « supervisé ∥ non-supervisé à parité » :

- **Supervisés** (flux d'erreur ``0/1``, état O(1), ``requires_label = True``) : ``DDM``, ``EDDM``,
  ``PageHinkley`` — S4402.
- **Non-supervisés** (features/score, fenêtre bornée, ``requires_label = False``) : ``ADWIN``,
  ``KSWIN``, ``KSTest``, ``MMD``, ``PSI`` (+ JS) — S4403.

Interface commune : ``BaseDriftDetector`` / verdict ``DriftVerdict`` (S4401). Le baseline
``SlidingWindowDriftDetector`` (``src/evaluation/drift_detector.py``) est catalogué à part, pas dupliqué.

Voir ``docs/context/drift_detectors.md`` (source de vérité) et
``configs/sprint44_drift_detection.yaml`` (hyperparamètres).
"""

from __future__ import annotations

from .adwin import ADWIN
from .base import BaseDriftDetector, DriftVerdict, error_stream
from .ddm import DDM
from .eddm import EDDM
from .ks_test import KSTest
from .kswin import KSWIN
from .mmd import MMD
from .multivariate import MultiFeatureDriftDetector
from .page_hinkley import PageHinkley
from .psi import PSI

#: Registre nom → classe (miroir des sections de configs/sprint44_drift_detection.yaml).
DRIFT_DETECTORS = {
    "ddm": DDM,
    "eddm": EDDM,
    "page_hinkley": PageHinkley,
    "adwin": ADWIN,
    "kswin": KSWIN,
    "ks_test": KSTest,
    "mmd": MMD,
    "psi": PSI,
}

__all__ = [
    "BaseDriftDetector",
    "DriftVerdict",
    "error_stream",
    "DDM",
    "EDDM",
    "PageHinkley",
    "ADWIN",
    "KSWIN",
    "KSTest",
    "MMD",
    "PSI",
    "MultiFeatureDriftDetector",
    "DRIFT_DETECTORS",
]
