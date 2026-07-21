"""
src/models/drift/eddm.py — Sprint 44 (S4402) : Early Drift Detection Method (Baena-García 2006).

Variante **supervisée** de DDM plus sensible au drift **graduel** : au lieu du taux d'erreur, elle
suit la **distance moyenne entre deux erreurs consécutives** ``p'_t`` et son écart-type ``s'_t``. Quand
les erreurs se rapprochent (distance qui chute), le rapport ``(p'+2s')/(p'_max+2s'_max)`` descend sous
les seuils α (warning) / β (drift). État **O(1)** (Welford sur les distances, pas de fenêtre).

Référence
---------
    M. Baena-García et al., « Early Drift Detection Method », IWKDDS 2006.
"""

from __future__ import annotations

import math

from .base import BaseDriftDetector, DriftVerdict

ALPHA_DEFAULT: float = 0.95  # rapport < α → WARNING
BETA_DEFAULT: float = 0.90  # rapport < β → DRIFT
MIN_ERRORS_DEFAULT: int = 30  # erreurs min. avant tout verdict


class EDDM(BaseDriftDetector):
    """Early Drift Detection Method — flux d'erreur ``0/1``, état O(1).

    Suit la distance entre erreurs par Welford (moyenne ``p'`` + écart-type ``s'``), mémorise le
    maximum ``(p'_max, s'_max)`` de ``p'+2s'`` (meilleur régime = erreurs les plus espacées), puis
    compare le rapport ``r = (p'+2s')/(p'_max+2s'_max)`` : ``WARNING`` si ``r < α``, ``DRIFT`` si
    ``r < β``. Au drift, l'état est réinitialisé.

    Parameters
    ----------
    config : dict
        Section ``eddm`` : ``alpha`` (0.95), ``beta`` (0.90), ``min_errors`` (30).

    Notes
    -----
    État = {sample_count, last_error, n_errors, m_dist, mean_dist, m2, p_max, s_max} → 8 scalaires.
    # MEM: 32 B @ FP32
    """

    _REQUIRES_LABEL = True
    _N_STATE_SCALARS = 8

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.alpha: float = cfg.get("alpha", ALPHA_DEFAULT)
        self.beta: float = cfg.get("beta", BETA_DEFAULT)
        self.min_errors: int = cfg.get("min_errors", MIN_ERRORS_DEFAULT)
        self.reset()

    def reset(self) -> None:
        self._sample_count: int = 0
        self._last_error_idx: int = 0
        self.n_errors_: int = 0
        self._m_dist: int = 0  # nombre de distances vues (compteur Welford)
        self.mean_dist_: float = 0.0  # MEM: 4 B @ FP32
        self._m2: float = 0.0  # somme des carrés centrés (Welford)
        self._p_max: float = 0.0
        self._s_max: float = 0.0

    def update(self, value: float) -> DriftVerdict:
        self._sample_count += 1
        error = value >= 0.5
        if not error:
            return DriftVerdict.NORMAL

        # Distance depuis la dernière erreur → Welford (moyenne + variance).
        self.n_errors_ += 1
        distance = self._sample_count - self._last_error_idx
        self._last_error_idx = self._sample_count
        self._m_dist += 1
        delta = distance - self.mean_dist_
        self.mean_dist_ += delta / self._m_dist
        self._m2 += delta * (distance - self.mean_dist_)

        if self._m_dist < 2:
            return DriftVerdict.NORMAL

        std_dist = math.sqrt(self._m2 / self._m_dist)
        level = self.mean_dist_ + 2.0 * std_dist

        # Nouveau meilleur régime (erreurs les plus espacées) → mémorise le max.
        if level > self._p_max + 2.0 * self._s_max:
            self._p_max = self.mean_dist_
            self._s_max = std_dist
            return DriftVerdict.NORMAL

        if self.n_errors_ < self.min_errors:
            return DriftVerdict.NORMAL

        denom = self._p_max + 2.0 * self._s_max
        if denom <= 0.0:
            return DriftVerdict.NORMAL
        ratio = level / denom
        if ratio < self.beta:
            self.reset()
            return DriftVerdict.DRIFT
        if ratio < self.alpha:
            return DriftVerdict.WARNING
        return DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        return self._N_STATE_SCALARS * 4
