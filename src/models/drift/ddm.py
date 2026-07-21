"""
src/models/drift/ddm.py — Sprint 44 (S4402) : Drift Detection Method (Gama 2004).

Détecteur **supervisé** : surveille le taux d'erreur en ligne ``p_t`` d'un modèle de faute et son
écart-type binomial ``s_t = sqrt(p_t·(1−p_t)/t)``. Quand ``p_t + s_t`` s'éloigne du minimum observé,
la distribution a changé. État strictement **O(1)** (aucune fenêtre) → très portable MCU.

Référence
---------
    J. Gama et al., « Learning with Drift Detection », SBIA 2004.
"""

from __future__ import annotations

import math

from .base import BaseDriftDetector, DriftVerdict

WARNING_LEVEL_DEFAULT: float = 2.0  # p+s ≥ p_min + 2·s_min → WARNING
DRIFT_LEVEL_DEFAULT: float = 3.0  # p+s ≥ p_min + 3·s_min → DRIFT
MIN_INSTANCES_DEFAULT: int = 30  # échantillons min. avant tout verdict (stabilise s_t)


class DDM(BaseDriftDetector):
    """Drift Detection Method — flux d'erreur ``0/1``, état O(1).

    À chaque erreur observée, met à jour la moyenne courante ``p_t`` et l'écart-type binomial
    ``s_t``, mémorise le minimum ``(p_min, s_min)`` de ``p+s``, puis compare :
    ``WARNING`` si ``p+s ≥ p_min + warning_level·s_min``, ``DRIFT`` si
    ``p+s ≥ p_min + drift_level·s_min``. Au drift, l'état est réinitialisé (nouveau régime).

    Parameters
    ----------
    config : dict
        Section ``ddm`` : ``warning_level`` (défaut 2.0), ``drift_level`` (3.0),
        ``min_instances`` (30).

    Attributes
    ----------
    n_ : int
        Nombre d'échantillons vus depuis le dernier reset.
    p_ : float
        Taux d'erreur courant.
    s_ : float
        Écart-type binomial courant.

    Notes
    -----
    État = {n, p, s, p_min, s_min} → 5 scalaires.  # MEM: 20 B @ FP32 / 20 B @ INT32
    """

    _REQUIRES_LABEL = True
    _N_STATE_SCALARS = 5

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.warning_level: float = cfg.get("warning_level", WARNING_LEVEL_DEFAULT)
        self.drift_level: float = cfg.get("drift_level", DRIFT_LEVEL_DEFAULT)
        self.min_instances: int = cfg.get("min_instances", MIN_INSTANCES_DEFAULT)
        self.reset()

    def reset(self) -> None:
        self.n_: int = 1
        self.p_: float = 1.0  # MEM: 4 B @ FP32
        self.s_: float = 0.0  # MEM: 4 B @ FP32
        self._p_min: float = math.inf
        self._s_min: float = math.inf

    def update(self, value: float) -> DriftVerdict:
        error = 1.0 if value >= 0.5 else 0.0
        # Moyenne incrémentale du taux d'erreur + écart-type binomial.
        self.p_ = self.p_ + (error - self.p_) / self.n_
        self.s_ = math.sqrt(self.p_ * (1.0 - self.p_) / self.n_)
        self.n_ += 1

        if self.n_ < self.min_instances:
            return DriftVerdict.NORMAL

        # Mise à jour du minimum de (p+s).
        if self.p_ + self.s_ <= self._p_min + self._s_min:
            self._p_min = self.p_
            self._s_min = self.s_

        if self.p_ + self.s_ > self._p_min + self.drift_level * self._s_min:
            self.reset()
            return DriftVerdict.DRIFT
        if self.p_ + self.s_ > self._p_min + self.warning_level * self._s_min:
            return DriftVerdict.WARNING
        return DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        return self._N_STATE_SCALARS * 4  # FP32 (n_ tient sur 4 B en int32 côté board)
