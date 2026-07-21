"""
src/models/drift/page_hinkley.py — Sprint 44 (S4402) : test séquentiel de Page-Hinkley (CUSUM).

Détecteur à état **O(1)** applicable au **flux d'erreur** (supervisé) **ou** à une **feature scalaire**
(frontière avec S4403). Cumule l'écart à la moyenne courante moins une tolérance ``δ`` ;
``DRIFT`` quand le cumul décroche de son minimum de plus de ``λ``. Détection très rapide sur drift
soudain (souvent ≤ DDM).

Référence
---------
    E. S. Page, « Continuous Inspection Schemes », Biometrika 1954.
"""

from __future__ import annotations

from .base import BaseDriftDetector, DriftVerdict

DELTA_DEFAULT: float = 0.005  # tolérance (magnitude de changement ignorée)
LAMBDA_DEFAULT: float = 50.0  # seuil de détection sur le cumul
MIN_INSTANCES_DEFAULT: int = 30  # échantillons min. avant tout verdict


class PageHinkley(BaseDriftDetector):
    """Test de Page-Hinkley (CUSUM unilatéral, hausse) — erreur ou feature scalaire, état O(1).

    Maintient la moyenne courante ``x̄_t``, le cumul ``m_T = Σ(x_t − x̄_t − δ)`` et son minimum
    ``min_T`` ; retourne ``DRIFT`` dès que ``m_T − min_T > λ`` (puis réinitialise). Ne produit pas de
    ``WARNING`` (test binaire).

    Parameters
    ----------
    config : dict
        Section ``page_hinkley`` : ``delta`` (0.005), ``lambda_`` (50.0), ``min_instances`` (30).

    Notes
    -----
    État = {n, mean, cumulative, min} → 4 scalaires.  # MEM: 16 B @ FP32
    """

    _REQUIRES_LABEL = True  # branché par défaut sur le flux d'erreur (S4402)
    _N_STATE_SCALARS = 4

    def __init__(self, config: dict | None = None, *, delta: float | None = None,
                 lambda_: float | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        # kwargs directs (delta=, lambda_=) prioritaires — utile pour les smoke-tests de la spec.
        self.delta: float = delta if delta is not None else cfg.get("delta", DELTA_DEFAULT)
        self.lambda_: float = (
            lambda_ if lambda_ is not None else cfg.get("lambda_", LAMBDA_DEFAULT)
        )
        self.min_instances: int = cfg.get("min_instances", MIN_INSTANCES_DEFAULT)
        self.reset()

    def reset(self) -> None:
        self.n_: int = 0
        self.mean_: float = 0.0  # MEM: 4 B @ FP32
        self._cumulative: float = 0.0
        self._min: float = 0.0

    def update(self, value: float) -> DriftVerdict:
        self.n_ += 1
        # Moyenne incrémentale.
        self.mean_ += (value - self.mean_) / self.n_
        self._cumulative += value - self.mean_ - self.delta
        self._min = min(self._min, self._cumulative)

        if self.n_ < self.min_instances:
            return DriftVerdict.NORMAL

        if self._cumulative - self._min > self.lambda_:
            self.reset()
            return DriftVerdict.DRIFT
        return DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        return self._N_STATE_SCALARS * 4
