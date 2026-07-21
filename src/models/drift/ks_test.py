"""
src/models/drift/ks_test.py — Sprint 44 (S4403) : test de Kolmogorov-Smirnov glissant deux-échantillons.

Détecteur **non-supervisé** : compare périodiquement (tous les ``stride`` échantillons) une fenêtre
courante bornée à une **fenêtre de référence figée** à l'enrôlement, via ``scipy.stats.ks_2samp``
(réutilisé — même primitive que ``characterize_drift.py`` S4303). ``DRIFT`` si la p-valeur < α.
État O(W) (fenêtre courante à **capacité fixe**, miroir de ``ring_buffer.h``).

Référence
---------
    Kolmogorov-Smirnov two-sample test · scipy.stats.ks_2samp.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from .base import BaseDriftDetector, DriftVerdict

WINDOW_SIZE_DEFAULT: int = 100
STRIDE_DEFAULT: int = 50  # évaluer le test tous les `stride` échantillons
ALPHA_DEFAULT: float = 0.01  # DRIFT si p-valeur < alpha
REF_SIZE_DEFAULT: int = 200  # taille max. de la fenêtre de référence figée


class KSTest(BaseDriftDetector):
    """KS glissant deux-échantillons (référence figée vs fenêtre courante bornée).

    Parameters
    ----------
    config : dict
        Section ``ks_test`` : ``window_size`` (100), ``stride`` (50), ``alpha`` (0.01),
        ``ref_size`` (200).

    Notes
    -----
    État = ref (≤ ref_size) + fenêtre courante (window_size).  # MEM: (ref_size+window_size)·4 B @ FP32
    """

    _REQUIRES_LABEL = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.window_size: int = cfg.get("window_size", WINDOW_SIZE_DEFAULT)
        self.stride: int = cfg.get("stride", STRIDE_DEFAULT)
        self.alpha: float = cfg.get("alpha", ALPHA_DEFAULT)
        self.ref_size: int = cfg.get("ref_size", REF_SIZE_DEFAULT)
        self._ref: np.ndarray | None = None  # MEM: ref_size·4 B @ FP32
        self.last_stat_: float | None = None
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.window_size)  # MEM: W·4 B @ FP32
        self._since_test = 0

    def set_params_from_reference(self, reference_values: np.ndarray) -> None:
        ref = np.asarray(reference_values, dtype=np.float64).ravel()
        self._ref = ref[: self.ref_size]
        self.reset()

    def update(self, value: float) -> DriftVerdict:
        if self._ref is None:
            raise RuntimeError("Référence non calibrée : appeler set_params_from_reference().")
        self._window.append(float(value))
        self._since_test += 1
        if len(self._window) < self.window_size or self._since_test < self.stride:
            return DriftVerdict.NORMAL

        self._since_test = 0
        stat, pval = ks_2samp(self._ref, np.asarray(self._window))
        self.last_stat_ = float(stat)
        return DriftVerdict.DRIFT if pval < self.alpha else DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        return (self.ref_size + self.window_size) * 4
