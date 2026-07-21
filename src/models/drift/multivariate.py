"""
src/models/drift/multivariate.py — Sprint 44 (S4403) : agrégation multivariée par feature.

Enveloppe un **détecteur scalaire** (un par feature) et agrège les verdicts. Politique d'agrégation
en config : ``max`` (drift dès qu'une feature dérive) ou ``fraction`` (drift si la fraction de features
en drift dépasse ``fraction_threshold``). MMD étant nativement multivarié, il n'utilise **pas** cette
enveloppe.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import BaseDriftDetector, DriftVerdict

AGGREGATION_DEFAULT: str = "max"  # "max" | "fraction"
FRACTION_THRESHOLD_DEFAULT: float = 0.3


class MultiFeatureDriftDetector(BaseDriftDetector):
    """Applique un détecteur scalaire par feature puis agrège les verdicts.

    Parameters
    ----------
    factory : Callable[[], BaseDriftDetector]
        Fabrique un détecteur scalaire neuf (un par feature).
    n_features : int
        Nombre de features surveillées.
    config : dict
        Section commune : ``aggregation`` ("max"|"fraction"), ``fraction_threshold`` (0.3).

    Notes
    -----
    ``requires_label`` hérité du détecteur scalaire fabriqué.
    """

    def __init__(self, factory: Callable[[], BaseDriftDetector], n_features: int,
                 config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.aggregation: str = cfg.get("aggregation", AGGREGATION_DEFAULT)
        self.fraction_threshold: float = cfg.get("fraction_threshold", FRACTION_THRESHOLD_DEFAULT)
        self.n_features = n_features
        self._detectors = [factory() for _ in range(n_features)]
        self._REQUIRES_LABEL = self._detectors[0].requires_label if self._detectors else False

    def set_params_from_reference(self, reference_values: np.ndarray) -> None:
        ref = np.asarray(reference_values)
        if ref.ndim == 1:
            ref = ref[:, None]
        for j, det in enumerate(self._detectors):
            det.set_params_from_reference(ref[:, j])

    def update(self, value) -> DriftVerdict:  # value = vecteur [d]
        vec = np.atleast_1d(np.asarray(value, dtype=np.float64))
        verdicts = [det.update(float(vec[j])) for j, det in enumerate(self._detectors)]
        n_drift = sum(v == DriftVerdict.DRIFT for v in verdicts)
        n_warn = sum(v == DriftVerdict.WARNING for v in verdicts)
        if self.aggregation == "fraction":
            if n_drift / max(self.n_features, 1) > self.fraction_threshold:
                return DriftVerdict.DRIFT
        elif n_drift > 0:  # "max"
            return DriftVerdict.DRIFT
        return DriftVerdict.WARNING if n_warn > 0 else DriftVerdict.NORMAL

    def reset(self) -> None:
        for det in self._detectors:
            det.reset()

    def get_state_bytes(self) -> int:
        return sum(det.get_state_bytes() for det in self._detectors)
