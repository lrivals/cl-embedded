"""
src/models/drift/psi.py — Sprint 44 (S4403) : Population Stability Index / Jensen-Shannon.

Détecteur **non-supervisé** le plus MCU-friendly : histogrammes à **bacs fixes** calibrés sur
l'enrôlement. L'état est **O(bins)** — indépendant de la taille de fenêtre W — car les échantillons
sont comptés **incrémentalement** dans les bacs (aucun stockage des valeurs brutes). Compare un bloc
courant de ``block_size`` échantillons à la référence figée : ``DRIFT`` si PSI (ou JS) > seuil.

Réutilise la définition des métriques de ``scripts/characterize_drift.py`` (``_psi`` / ``_js_divergence``,
S4303) — ici en version « streaming » sur des probabilités déjà agrégées.

Références
----------
    PSI : standard credit-scoring · Lin 1991 (Jensen-Shannon).
"""

from __future__ import annotations

import numpy as np

from .base import BaseDriftDetector, DriftVerdict

BINS_DEFAULT: int = 10
BLOCK_SIZE_DEFAULT: int = 200  # échantillons par bloc de comparaison
PSI_THRESHOLD_DEFAULT: float = 0.2  # PSI > 0.2 = changement significatif (standard)
JS_THRESHOLD_DEFAULT: float = 0.1
METRIC_DEFAULT: str = "psi"  # "psi" | "js"
_EPS: float = 1e-8


class PSI(BaseDriftDetector):
    """Détecteur PSI / Jensen-Shannon sur histogrammes à bacs fixes — état O(bins).

    Calibration (``set_params_from_reference``) : fige les bords de bacs sur ``[min, max]`` de la
    référence et la distribution de référence ``ref_probs``. Détection : compte chaque échantillon
    reçu dans son bac ; tous les ``block_size`` échantillons, calcule la métrique vs la référence et
    réinitialise le bloc.

    Parameters
    ----------
    config : dict
        Section ``psi`` : ``bins`` (10), ``block_size`` (200), ``metric`` ("psi"|"js"),
        ``psi_threshold`` (0.2), ``js_threshold`` (0.1).

    Notes
    -----
    État = edges (bins+1) + ref_probs (bins) + cur_counts (bins).  # MEM: (3·bins+1)·4 B @ FP32
    Indépendant de W → argument MCU (Sprint 45).
    """

    _REQUIRES_LABEL = False

    def __init__(self, config: dict | None = None, *, bins: int | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.bins: int = bins if bins is not None else cfg.get("bins", BINS_DEFAULT)
        self.block_size: int = cfg.get("block_size", BLOCK_SIZE_DEFAULT)
        self.metric: str = cfg.get("metric", METRIC_DEFAULT)
        self.psi_threshold: float = cfg.get("psi_threshold", PSI_THRESHOLD_DEFAULT)
        self.js_threshold: float = cfg.get("js_threshold", JS_THRESHOLD_DEFAULT)
        self._edges: np.ndarray | None = None  # MEM: (bins+1)·4 B @ FP32
        self._ref_probs: np.ndarray | None = None  # MEM: bins·4 B @ FP32
        self.last_stat_: float | None = None
        self.reset()

    def reset(self) -> None:
        self._cur_counts = np.zeros(self.bins, dtype=np.int64)  # MEM: bins·8 B (hôte)
        self._block_seen = 0

    def set_params_from_reference(self, reference_values: np.ndarray) -> None:
        ref = np.asarray(reference_values, dtype=np.float64).ravel()
        lo, hi = float(ref.min()), float(ref.max())
        if hi - lo < _EPS:
            hi = lo + 1.0
        self._edges = np.linspace(lo, hi, self.bins + 1)
        counts, _ = np.histogram(ref, bins=self._edges)
        self._ref_probs = counts / max(counts.sum(), 1) + _EPS
        self.reset()

    def _bin_index(self, x: float) -> int:
        # searchsorted sur les bords fixes, borné à [0, bins-1] (queues clampées).
        idx = int(np.searchsorted(self._edges, x, side="right")) - 1
        return min(max(idx, 0), self.bins - 1)

    def _stat(self, cur_probs: np.ndarray) -> float:
        ref = self._ref_probs
        if self.metric == "js":
            m = 0.5 * (ref + cur_probs)
            kl = lambda a, b: float(np.sum(a * np.log2(a / b)))  # noqa: E731
            return 0.5 * kl(ref, m) + 0.5 * kl(cur_probs, m)
        return float(np.sum((cur_probs - ref) * np.log(cur_probs / ref)))  # PSI

    def update(self, value: float) -> DriftVerdict:
        if self._edges is None or self._ref_probs is None:
            raise RuntimeError("Référence non calibrée : appeler set_params_from_reference().")
        self._cur_counts[self._bin_index(value)] += 1
        self._block_seen += 1
        if self._block_seen < self.block_size:
            return DriftVerdict.NORMAL

        cur_probs = self._cur_counts / self._block_seen + _EPS
        stat = self._stat(cur_probs)
        self.last_stat_ = stat
        self.reset()
        threshold = self.js_threshold if self.metric == "js" else self.psi_threshold
        return DriftVerdict.DRIFT if stat > threshold else DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        # edges (bins+1) + ref_probs (bins) + cur_counts (bins), en FP32 côté board.
        return (3 * self.bins + 1) * 4
