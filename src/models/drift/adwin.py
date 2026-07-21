"""
src/models/drift/adwin.py — Sprint 44 (S4403) : ADaptive WINdowing (Bifet & Gavaldà 2007).

Détecteur **non-supervisé** à fenêtre **adaptative** : maintient un histogramme exponentiel de buckets
(chaque bucket agrège moyenne + variance de 2^i points). Coupe la fenêtre — et signale ``DRIFT`` — dès
que deux sous-fenêtres diffèrent au-delà de la borne de Hoeffding ``ε_cut`` (paramétrée par ``delta``).
Nombre de buckets O(log W) ; une **borne ``max_rows``** (config) majore l'état → argument MCU.

Référence
---------
    A. Bifet, R. Gavaldà, « Learning from Time-Changing Data with Adaptive Windowing », SDM 2007.
"""

from __future__ import annotations

import math

from .base import BaseDriftDetector, DriftVerdict

DELTA_DEFAULT: float = 0.002  # confiance de la borne de Hoeffding (plus petit = moins sensible)
MAX_BUCKETS_DEFAULT: int = 5  # buckets max. par niveau (M de l'histogramme exponentiel)
MIN_WINDOW_LENGTH_DEFAULT: int = 5  # longueur min. de chaque sous-fenêtre pour tester une coupe
MIN_CLOCK_DEFAULT: int = 32  # ne tester une coupe qu'un échantillon sur `min_clock`
MAX_ROWS_DEFAULT: int = 40  # borne du nombre de niveaux → état majoré (≈ 2^40 points de capacité)


class _Row:
    """Un niveau de l'histogramme : jusqu'à ``max_buckets + 1`` buckets de capacité identique 2^i.

    L'indice 0 est le bucket le plus ancien du niveau.
    """

    __slots__ = ("max_buckets", "total", "variance", "n")

    def __init__(self, max_buckets: int) -> None:
        self.max_buckets = max_buckets
        self.total = [0.0] * (max_buckets + 1)  # MEM: (M+1)·4 B @ FP32
        self.variance = [0.0] * (max_buckets + 1)  # MEM: (M+1)·4 B @ FP32
        self.n = 0

    def insert(self, total: float, variance: float) -> None:
        self.total[self.n] = total
        self.variance[self.n] = variance
        self.n += 1

    def drop_oldest(self, count: int = 1) -> None:
        for i in range(count, self.n):
            self.total[i - count] = self.total[i]
            self.variance[i - count] = self.variance[i]
        self.n -= count


class ADWIN(BaseDriftDetector):
    """ADaptive WINdowing — fenêtre adaptative, buckets d'histogramme exponentiel bornés.

    Parameters
    ----------
    config : dict
        Section ``adwin`` : ``delta`` (0.002), ``max_buckets`` (5), ``min_window_length`` (5),
        ``min_clock`` (32), ``max_rows`` (40).

    Attributes
    ----------
    width_ : int
        Nombre d'éléments dans la fenêtre courante.

    Notes
    -----
    État majoré = max_rows · (max_buckets+1) · 2 scalaires.
    # MEM: max_rows·(M+1)·8 B @ FP32
    """

    _REQUIRES_LABEL = False

    def __init__(self, config: dict | None = None, *, delta: float | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.delta: float = delta if delta is not None else cfg.get("delta", DELTA_DEFAULT)
        self.max_buckets: int = cfg.get("max_buckets", MAX_BUCKETS_DEFAULT)
        self.min_window_length: int = cfg.get("min_window_length", MIN_WINDOW_LENGTH_DEFAULT)
        self.min_clock: int = cfg.get("min_clock", MIN_CLOCK_DEFAULT)
        self.max_rows: int = cfg.get("max_rows", MAX_ROWS_DEFAULT)
        self.reset()

    def reset(self) -> None:
        self._rows: list[_Row] = [_Row(self.max_buckets)]
        self.width_: int = 0
        self._total: float = 0.0
        self._variance: float = 0.0
        self._n_since_check: int = 0

    # ---- insertion + compression --------------------------------------------
    def _insert(self, value: float) -> None:
        # Variance incrémentale (Welford) avant mise à jour du total.
        if self.width_ >= 1:
            mean_prev = self._total / self.width_
            self._variance += self.width_ * (value - mean_prev) ** 2 / (self.width_ + 1)
        self._total += value
        self.width_ += 1
        self._rows[0].insert(value, 0.0)
        self._compress()

    def _compress(self) -> None:
        i = 0
        while i < len(self._rows):
            row = self._rows[i]
            if row.n <= self.max_buckets:
                break
            # Fusionne les deux buckets les plus anciens (indices 0, 1) → niveau i+1.
            size = float(1 << i)
            t0, t1 = row.total[0], row.total[1]
            v0, v1 = row.variance[0], row.variance[1]
            mean0, mean1 = t0 / size, t1 / size
            merged_total = t0 + t1
            merged_var = v0 + v1 + size * size * (mean0 - mean1) ** 2 / (size + size)
            row.drop_oldest(2)
            if i + 1 >= len(self._rows):
                if len(self._rows) >= self.max_rows:
                    # Borne atteinte : on abandonne la fusion (les 2 buckets sont déjà retirés →
                    # état strictement majoré, fenêtre légèrement tronquée côté ancien).
                    self.width_ -= int(2 * size)
                    self._total -= merged_total
                    self._recompute_variance()
                    return
                self._rows.append(_Row(self.max_buckets))
            self._rows[i + 1].insert(merged_total, merged_var)
            i += 1

    def _recompute_variance(self) -> None:
        # Recalcule la variance globale après troncature (rare, borne max_rows) — état borné.
        if self.width_ <= 0:
            self._variance = 0.0

    # ---- détection de coupe --------------------------------------------------
    def _epsilon_cut(self, n0: int, n1: int) -> float:
        n = self.width_
        v = self._variance / n
        dd = math.log(2.0 * math.log(n) / self.delta)
        m = 1.0 / (n0 - self.min_window_length + 1) + 1.0 / (n1 - self.min_window_length + 1)
        return math.sqrt(2.0 * m * v * dd) + 2.0 / 3.0 * dd * m

    def _delete_oldest(self) -> None:
        # Retire le bucket le plus ancien (niveau le plus haut, indice 0).
        for i in range(len(self._rows) - 1, -1, -1):
            row = self._rows[i]
            if row.n > 0:
                size = 1 << i
                t = row.total[0]
                self.width_ -= size
                self._total -= t
                row.drop_oldest(1)
                self._variance = max(self._variance, 0.0)
                return

    def _detect(self) -> bool:
        if self.width_ < 2 * self.min_window_length:
            return False
        changed = False
        shrink = True
        while shrink:
            shrink = False
            n0, n1 = 0, self.width_
            u0, u1 = 0.0, self._total
            # Parcours du plus ancien (dernier niveau, indice 0) au plus récent.
            done = False
            for i in range(len(self._rows) - 1, -1, -1):
                row = self._rows[i]
                size = 1 << i
                for k in range(row.n):
                    n0 += size
                    n1 -= size
                    u0 += row.total[k]
                    u1 -= row.total[k]
                    if n1 < self.min_window_length or n0 < self.min_window_length:
                        continue
                    if n1 <= 0:
                        continue
                    if abs(u0 / n0 - u1 / n1) > self._epsilon_cut(n0, n1):
                        self._delete_oldest()
                        changed = True
                        shrink = True
                        done = True
                        break
                if done:
                    break
        return changed

    # ---- interface -----------------------------------------------------------
    def update(self, value: float) -> DriftVerdict:
        self._insert(float(value))
        self._n_since_check += 1
        if self._n_since_check < self.min_clock:
            return DriftVerdict.NORMAL
        self._n_since_check = 0
        return DriftVerdict.DRIFT if self._detect() else DriftVerdict.NORMAL

    def get_state_bytes(self) -> int:
        # Borne majorée par max_rows niveaux × (M+1) buckets × 2 scalaires (total, variance).
        return self.max_rows * (self.max_buckets + 1) * 2 * 4
