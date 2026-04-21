"""
src/evaluation/drift_detector.py — S9-10: Détection FAULT vs DRIFT par fenêtre glissante.

Discrimine deux types d'anomalie à partir du score brut d'un détecteur non supervisé :
    FAULT : dépassement instantané du seuil panne (score_t > fault_threshold)
    DRIFT : fraction des W derniers scores > drift_threshold dépasse drift_ratio
    NORMAL : sinon

Compatible MCU : état interne = deque(maxlen=W), O(W) RAM.
Pour W=50, d=4 → 50 × 4 B = 200 B @ FP32.  # MEM: 200 B @ FP32 / 50 B @ INT8

Références
----------
    docs/sprints/sprint_9/S907_deployment_scenario_use_case.md (Q3)
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

# Multiplicateurs de seuil calibrés sur les données d'enrôlement
FAULT_MULTIPLIER_DEFAULT: float = 2.5   # fault_threshold = P95 × 2.5
DRIFT_MULTIPLIER_DEFAULT: float = 1.3   # drift_threshold = P95 × 1.3
DRIFT_RATIO_DEFAULT: float = 0.6        # >60% de la fenêtre dépasse drift_threshold
WINDOW_SIZE_DEFAULT: int = 50           # taille de la fenêtre glissante


class SlidingWindowDriftDetector:
    """
    Détection FAULT vs DRIFT par fenêtre glissante sur les scores d'anomalie.

    FAULT : score_t > fault_threshold  (dépassement instantané — panne soudaine)
    DRIFT : fraction(scores[-W:] > drift_threshold) > drift_ratio  (dérive progressive)
    NORMAL : sinon

    Compatible MCU : état = deque(maxlen=window_size), O(W) RAM.

    Parameters
    ----------
    window_size : int
        Taille de la fenêtre glissante (défaut : 50).
    fault_multiplier : float
        fault_threshold = P95_enrôlement × fault_multiplier (défaut : 2.5).
    drift_multiplier : float
        drift_threshold = P95_enrôlement × drift_multiplier (défaut : 1.3).
    drift_ratio : float
        Fraction minimale de la fenêtre au-dessus de drift_threshold pour déclencher DRIFT.

    Attributes
    ----------
    fault_threshold : float | None
        Seuil de panne (calibré par set_thresholds_from_normal).
    drift_threshold : float | None
        Seuil de dérive (calibré par set_thresholds_from_normal).

    Notes
    -----
    Compatible STM32N6 : pour W=50 → 200 B @ FP32 / 50 B @ INT8.  # MEM: 200 B @ FP32 / 50 B @ INT8
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE_DEFAULT,
        fault_multiplier: float = FAULT_MULTIPLIER_DEFAULT,
        drift_multiplier: float = DRIFT_MULTIPLIER_DEFAULT,
        drift_ratio: float = DRIFT_RATIO_DEFAULT,
    ) -> None:
        self.window_size = window_size
        self.fault_multiplier = fault_multiplier
        self.drift_multiplier = drift_multiplier
        self.drift_ratio = drift_ratio

        self.fault_threshold: float | None = None
        self.drift_threshold: float | None = None
        self._window: deque[float] = deque(maxlen=window_size)  # MEM: W×4 B @ FP32

    def set_thresholds_from_normal(self, normal_scores: np.ndarray) -> None:
        """
        Calibre les seuils FAULT et DRIFT depuis les scores d'enrôlement.

        Parameters
        ----------
        normal_scores : np.ndarray [N]
            Scores d'anomalie calculés sur les données normales d'enrôlement.
            fault_threshold = P95 × fault_multiplier
            drift_threshold = P95 × drift_multiplier
        """
        p95 = float(np.percentile(normal_scores, 95))
        self.fault_threshold = p95 * self.fault_multiplier
        self.drift_threshold = p95 * self.drift_multiplier

    def update(self, score: float) -> Literal["NORMAL", "FAULT", "DRIFT"]:
        """
        Traite un nouveau score et retourne l'état courant.

        La priorité est : FAULT > DRIFT > NORMAL.
        FAULT est détecté avant la mise à jour de la fenêtre pour rester instantané.

        Parameters
        ----------
        score : float
            Score d'anomalie courant (distance de Mahalanobis, reconstruction PCA, etc.).

        Returns
        -------
        Literal["NORMAL", "FAULT", "DRIFT"]

        Raises
        ------
        RuntimeError
            Si set_thresholds_from_normal() n'a pas été appelé.
        """
        if self.fault_threshold is None or self.drift_threshold is None:
            raise RuntimeError(
                "Seuils non calibrés. Appeler set_thresholds_from_normal() d'abord."
            )

        self._window.append(score)

        # FAULT : dépassement instantané
        if score > self.fault_threshold:
            return "FAULT"

        # DRIFT : fraction de la fenêtre au-dessus du seuil de dérive
        if len(self._window) > 0:
            above = sum(1 for s in self._window if s > self.drift_threshold)
            if above / len(self._window) > self.drift_ratio:
                return "DRIFT"

        return "NORMAL"

    def update_batch(self, scores: np.ndarray) -> list[str]:
        """
        Traite un batch de scores et retourne la liste des états.

        Parameters
        ----------
        scores : np.ndarray [N]
            Scores d'anomalie à traiter séquentiellement.

        Returns
        -------
        list[str]
            États ["NORMAL" | "FAULT" | "DRIFT"] pour chaque score.
        """
        return [self.update(float(s)) for s in scores]

    def get_window_stats(self) -> dict:
        """
        Retourne des statistiques descriptives sur la fenêtre courante.

        Returns
        -------
        dict avec : size, mean, std, max, p90, fraction_above_drift.
        """
        if not self._window:
            return {
                "size": 0,
                "mean": None,
                "std": None,
                "max": None,
                "p90": None,
                "fraction_above_drift": None,
            }
        arr = np.array(self._window)
        fraction = (
            float((arr > self.drift_threshold).mean())
            if self.drift_threshold is not None
            else None
        )
        return {
            "size": len(self._window),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "p90": float(np.percentile(arr, 90)),
            "fraction_above_drift": fraction,
        }

    def reset(self) -> None:
        """Vide la fenêtre glissante (nouvelle machine / nouveau contexte)."""
        self._window.clear()
