"""
online_metrics.py — Métriques d'évaluation incrémentales pour le contexte embarqué.

Contrainte MCU : pas de stockage de toutes les prédictions en RAM.
Toutes les métriques sont mises à jour sample par sample (O(1) mémoire
pour accuracy, fenêtre bornée pour AUROC).

Métriques implémentées :
    OnlineAccuracy  — accuracy cumulée, update O(1)
    OnlineAUROC     — approximation par fenêtre glissante bornée
    OnlineForgetting — chute d'accuracy entre tâches
"""

from __future__ import annotations

from collections import deque

import numpy as np


class OnlineAccuracy:
    """Accuracy cumulée, mise à jour sample par sample."""

    def __init__(self) -> None:
        self._n_correct: int = 0
        self._n_total: int = 0

    def update(self, y_true: int, y_pred: int) -> None:
        self._n_total += 1
        if y_true == y_pred:
            self._n_correct += 1

    def compute(self) -> float:
        if self._n_total == 0:
            return 0.0
        return self._n_correct / self._n_total

    def reset(self) -> None:
        self._n_correct = 0
        self._n_total = 0


class OnlineAUROC:
    """
    Approximation de l'AUROC par fenêtre glissante.

    Stocke au plus `window_size` paires (y_true, score) en mémoire.
    Calcule l'AUROC exacte sur la fenêtre courante (Mann-Whitney U).

    Parameters
    ----------
    window_size : int
        Taille maximale de la fenêtre (défaut 500 pour rester < 4 Ko @ float32).
    """

    def __init__(self, window_size: int = 500) -> None:
        self._window_size = window_size
        self._buffer: deque[tuple[int, float]] = deque(maxlen=window_size)

    def update(self, y_true: int, score: float) -> None:
        self._buffer.append((int(y_true), float(score)))

    def compute(self) -> float:
        if len(self._buffer) < 2:
            return 0.5

        labels = np.array([b[0] for b in self._buffer])
        scores = np.array([b[1] for b in self._buffer])

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Undefined → convention 0.5

        # Mann-Whitney U statistic
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        u_stat = sum(
            (1.0 if p > n else 0.5 if p == n else 0.0)
            for p in pos_scores
            for n in neg_scores
        )
        return float(u_stat / (n_pos * n_neg))

    def reset(self) -> None:
        self._buffer.clear()


class OnlineForgetting:
    """
    Mesure la chute d'accuracy entre la fin d'une tâche et la fin de l'entraînement.

    À appeler en fin de chaque tâche avec `record_task_end()`,
    puis en fin de session avec `compute()`.

    Compatible avec la définition AF de De Lange et al. (2021).
    """

    def __init__(self) -> None:
        self._peak_acc: dict[int, float] = {}
        self._final_acc: dict[int, float] = {}

    def record_task_end(self, task_id: int, accuracy: float) -> None:
        """Enregistre l'accuracy de pointe à la fin de l'entraînement sur cette tâche."""
        self._peak_acc[task_id] = accuracy

    def record_final(self, task_id: int, accuracy: float) -> None:
        """Enregistre l'accuracy finale sur une tâche après toutes les tâches."""
        self._final_acc[task_id] = accuracy

    def compute(self) -> dict[str, float]:
        """
        Retourne le forgetting moyen (AF) et par tâche.

        Returns
        -------
        dict avec 'af' (scalar) et 'per_task' (dict[task_id, forgetting])
        """
        tasks = [t for t in self._peak_acc if t in self._final_acc]
        if not tasks:
            return {"af": 0.0, "per_task": {}}

        per_task = {t: self._peak_acc[t] - self._final_acc[t] for t in tasks}
        af = float(np.mean(list(per_task.values())))
        return {"af": af, "per_task": per_task}
