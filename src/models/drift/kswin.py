"""
src/models/drift/kswin.py — Sprint 44 (S4403) : Kolmogorov-Smirnov Windowing (Raab 2020).

Détecteur **non-supervisé** auto-adaptatif : maintient une fenêtre glissante bornée de taille W ;
compare les ``r`` échantillons **les plus récents** à un tirage aléatoire de ``r`` échantillons du
**reste** de la fenêtre (réservoir), par ``ks_2samp``. ``DRIFT`` si p-valeur < α ; on purge alors la
partie ancienne (adaptation). État O(W) + tri (dans ks_2samp).

Référence
---------
    C. Raab, M. Heusinger, F.-M. Schleif, « Reactive Soft Prototype Computing for Concept Drift »,
    Neurocomputing 2020 (KSWIN).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from .base import BaseDriftDetector, DriftVerdict

WINDOW_SIZE_DEFAULT: int = 100
STAT_SIZE_DEFAULT: int = 30  # r : nombre d'échantillons récents comparés
ALPHA_DEFAULT: float = 0.005  # DRIFT si p-valeur < alpha
SEED_DEFAULT: int = 42


class KSWIN(BaseDriftDetector):
    """Kolmogorov-Smirnov Windowing — fenêtre adaptative bornée.

    Parameters
    ----------
    config : dict
        Section ``kswin`` : ``window_size`` (100), ``stat_size`` (30 = r), ``alpha`` (0.005),
        ``seed`` (42).

    Notes
    -----
    ``window_size > stat_size`` requis. État = fenêtre (W).  # MEM: W·4 B @ FP32
    """

    _REQUIRES_LABEL = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.window_size: int = cfg.get("window_size", WINDOW_SIZE_DEFAULT)
        self.stat_size: int = cfg.get("stat_size", STAT_SIZE_DEFAULT)
        self.alpha: float = cfg.get("alpha", ALPHA_DEFAULT)
        self.seed: int = cfg.get("seed", SEED_DEFAULT)
        if self.window_size <= self.stat_size:
            raise ValueError("window_size doit être > stat_size.")
        self.last_stat_: float | None = None
        self.reset()

    def reset(self) -> None:
        self._window: deque[float] = deque(maxlen=self.window_size)  # MEM: W·4 B @ FP32
        self._rng = np.random.default_rng(self.seed)

    def update(self, value: float) -> DriftVerdict:
        self._window.append(float(value))
        if len(self._window) < self.window_size:
            return DriftVerdict.NORMAL

        arr = np.asarray(self._window)
        recent = arr[-self.stat_size:]
        older = arr[: -self.stat_size]
        # Tirage aléatoire de r échantillons dans la partie ancienne (réservoir).
        sample = self._rng.choice(older, self.stat_size, replace=True)
        stat, pval = ks_2samp(recent, sample)
        self.last_stat_ = float(stat)

        if pval < self.alpha:
            # Adaptation : purge la partie ancienne, conserve les r récents.
            self._window = deque(recent.tolist(), maxlen=self.window_size)
            return DriftVerdict.DRIFT
        return DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        return self.window_size * 4
