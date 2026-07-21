"""
src/models/drift/mmd.py — Sprint 44 (S4403) : Maximum Mean Discrepancy (noyau RBF).

Détecteur **non-supervisé nativement multivarié** : distance MMD² à noyau RBF entre une fenêtre de
référence figée et une fenêtre courante bornée. **Variante linéaire** (estimateur non biaisé O(W),
Gretton 2012 §6) privilégiée à la forme quadratique O(W²) pour la viabilité MCU. Seuil calibré par
**percentile d'enrôlement** (portable) plutôt que par permutation (réservée au diagnostic PC).

Réutilise la logique noyau/heuristique-médiane de ``_mmd_rbf`` (``characterize_drift.py`` S4303).

Référence
---------
    A. Gretton et al., « A Kernel Two-Sample Test », JMLR 2012.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import BaseDriftDetector, DriftVerdict

WINDOW_SIZE_DEFAULT: int = 100
STRIDE_DEFAULT: int = 50
GAMMA_DEFAULT = None  # None → heuristique de la médiane sur la référence
ESTIMATOR_DEFAULT: str = "linear"  # "linear" (O(W)) | "quadratic" (O(W²))
CALIB_PERCENTILE_DEFAULT: float = 99.0  # seuil = percentile des MMD² inter-blocs de référence
CALIB_BLOCKS_DEFAULT: int = 20  # nombre de blocs de référence pour calibrer le seuil
SEED_DEFAULT: int = 42
_EPS: float = 1e-8


class MMD(BaseDriftDetector):
    """MMD² RBF (référence figée vs fenêtre courante), estimateur linéaire O(W) par défaut.

    Accepte un scalaire **ou** un vecteur par ``update`` (nativement multivarié). Calibration :
    ``gamma`` par heuristique de la médiane sur la référence, seuil = percentile des MMD² calculés
    entre blocs de la référence elle-même.

    Parameters
    ----------
    config : dict
        Section ``mmd`` : ``window_size`` (100), ``stride`` (50), ``gamma`` (null=médiane),
        ``estimator`` ("linear"|"quadratic"), ``calib_percentile`` (95), ``calib_blocks`` (20),
        ``seed`` (42).

    Notes
    -----
    État = ref (n_ref·d) + fenêtre courante (W·d).  # MEM: (n_ref+W)·d·4 B @ FP32
    """

    _REQUIRES_LABEL = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.window_size: int = cfg.get("window_size", WINDOW_SIZE_DEFAULT)
        self.stride: int = cfg.get("stride", STRIDE_DEFAULT)
        self.gamma = cfg.get("gamma", GAMMA_DEFAULT)
        self.estimator: str = cfg.get("estimator", ESTIMATOR_DEFAULT)
        self.calib_percentile: float = cfg.get("calib_percentile", CALIB_PERCENTILE_DEFAULT)
        self.calib_blocks: int = cfg.get("calib_blocks", CALIB_BLOCKS_DEFAULT)
        self.seed: int = cfg.get("seed", SEED_DEFAULT)
        self._ref: np.ndarray | None = None  # MEM: n_ref·d·4 B @ FP32
        self.threshold_: float | None = None
        self._n_features: int = 1
        self.last_stat_: float | None = None
        self.reset()

    def reset(self) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)  # MEM: W·d·4 B @ FP32
        self._since_test = 0

    # ---- noyau ---------------------------------------------------------------
    def _median_gamma(self, ref: np.ndarray) -> float:
        rng = np.random.default_rng(self.seed)
        sub = ref if len(ref) <= 300 else ref[rng.choice(len(ref), 300, replace=False)]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return 1.0 / max(med, _EPS)

    def _mmd2(self, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
        if self.estimator == "linear":
            return self._mmd2_linear(x, y, gamma)
        return self._mmd2_quadratic(x, y, gamma)

    def _mmd2_quadratic(self, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
        def k(a, b):
            d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
            return np.exp(-gamma * d2)

        return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())

    def _mmd2_linear(self, x: np.ndarray, y: np.ndarray, gamma: float) -> float:
        # Estimateur linéaire non biaisé (Gretton 2012, Lemma 14) : apparie les échantillons 2 à 2.
        m = min(len(x), len(y))
        m -= m % 2
        if m < 2:
            return 0.0
        x, y = x[:m], y[:m]
        x1, x2 = x[0::2], x[1::2]
        y1, y2 = y[0::2], y[1::2]

        def k(a, b):
            return np.exp(-gamma * np.sum((a - b) ** 2, axis=-1))

        h = k(x1, x2) + k(y1, y2) - k(x1, y2) - k(x2, y1)
        return float(h.mean())

    # ---- interface -----------------------------------------------------------
    def set_params_from_reference(self, reference_values: np.ndarray) -> None:
        ref = np.asarray(reference_values, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref[:, None]
        self._ref = ref
        self._n_features = ref.shape[1]
        if self.gamma is None:
            self._gamma_ = self._median_gamma(ref)
        else:
            self._gamma_ = float(self.gamma)

        # Seuil = percentile des MMD² entre blocs de la référence (détection sous H0).
        rng = np.random.default_rng(self.seed)
        block = min(self.window_size, len(ref) // 2)
        stats = []
        if block >= 2:
            for _ in range(self.calib_blocks):
                a = ref[rng.choice(len(ref), block, replace=False)]
                b = ref[rng.choice(len(ref), block, replace=False)]
                stats.append(self._mmd2(a, b, self._gamma_))
        self.threshold_ = float(np.percentile(stats, self.calib_percentile)) if stats else 0.0
        self.reset()

    def update(self, value) -> DriftVerdict:
        if self._ref is None or self.threshold_ is None:
            raise RuntimeError("Référence non calibrée : appeler set_params_from_reference().")
        vec = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self._window.append(vec)
        self._since_test += 1
        if len(self._window) < self.window_size or self._since_test < self.stride:
            return DriftVerdict.NORMAL

        self._since_test = 0
        cur = np.vstack(self._window)
        stat = self._mmd2(self._ref, cur, self._gamma_)
        self.last_stat_ = stat
        return DriftVerdict.DRIFT if stat > self.threshold_ else DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        n_ref = 0 if self._ref is None else len(self._ref)
        return (n_ref + self.window_size) * self._n_features * 4
