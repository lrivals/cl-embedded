# ruff: noqa: N803, N806  — X est une convention mathématique ML (sklearn API)
"""
Baseline DBSCAN pour la détection d'anomalies en scénario domain-incremental.

Les points classés "bruit" (label=-1) par DBSCAN sont traités comme anomalies.
Score continu = distance au core point le plus proche (option C du spec S5-14).
Labels utilisés uniquement en évaluation, jamais pendant fit_task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
EPSILON_DEFAULT: float = 0.5
MIN_SAMPLES_DEFAULT: int = 5
CL_STRATEGY_DEFAULT: str = "refit"  # "refit" | "accumulate"
ANOMALY_PERCENTILE_DEFAULT: int = 95


class DBSCANDetector:
    """
    Baseline de détection d'anomalies par DBSCAN en scénario domain-incremental.

    Les points "bruit" (label=-1) sont considérés comme anomalies. Le score
    continu est la distance au core point le plus proche, permettant le calcul
    d'AUROC. Les core points sont conservés comme référence offline.

    # MEM: (n_core_samples × d) @ FP32 pour cl_strategy=refit
    # MEM: (n_samples × d) @ FP32 pour cl_strategy=accumulate

    En scénario CL (cl_strategy="refit"), DBSCAN est recalculé sur chaque
    nouvelle tâche — évalue l'oubli structurel par recalcul complet.

    Parameters
    ----------
    config : dict
        Sous-section "dbscan" de unsupervised_config.yaml.

    Attributes
    ----------
    core_points_ : np.ndarray [n_core, d] | None
        Points core conservés pour le scoring (détectés lors du dernier fit_task).
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0 si anomaly_threshold est null).
    task_id_ : int
        Index de la dernière tâche entraînée.
    n_features_ : int
        Dimension des features (fixée au premier fit_task).

    Notes
    -----
    PC-only si cl_strategy="accumulate" (RAM croissante).
    Compatible STM32N6 uniquement si cl_strategy="refit" et n_core_samples petit.
    Labels utilisés uniquement en évaluation (score, auroc), jamais pendant fit_task.
    """

    def __init__(self, config: dict) -> None:
        self.eps: float = config.get("EPSILON", EPSILON_DEFAULT)
        self.min_samples: int = config.get("MIN_SAMPLES", MIN_SAMPLES_DEFAULT)
        self.metric: str = config.get("metric", "euclidean")
        self.algorithm: str = config.get("algorithm", "auto")
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", CL_STRATEGY_DEFAULT)

        self.core_points_: np.ndarray | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1
        self.n_features_: int = 0

        # Pour cl_strategy="accumulate"
        self._X_accumulated: np.ndarray | None = None

    def fit_task(self, X: np.ndarray, task_id: int) -> "DBSCANDetector":
        """
        Ajuste DBSCAN sur les données d'une tâche (offline).

        Le seuil de décision est calculé sur Task 0 (première tâche) uniquement.

        Parameters
        ----------
        X : np.ndarray [N, d]
            Données d'entraînement (non labelisées — labels exclus).
        task_id : int
            Index de la tâche (0-based).

        Returns
        -------
        self
        """
        self.task_id_ = task_id
        self.n_features_ = X.shape[1]

        # Gestion stratégie CL
        if self.cl_strategy == "accumulate":
            if self._X_accumulated is None:
                self._X_accumulated = X.copy()
            else:
                self._X_accumulated = np.concatenate([self._X_accumulated, X], axis=0)
            X_fit = self._X_accumulated  # MEM: (n_samples × d) @ FP32 pour accumulate
        else:
            X_fit = X  # MEM: (n_core_samples × d) @ FP32 pour refit

        # Fit DBSCAN
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
        )
        dbscan.fit(X_fit)

        # Conserver les core points pour le scoring
        core_indices = dbscan.core_sample_indices_
        if len(core_indices) > 0:
            self.core_points_ = X_fit[core_indices].astype(np.float32)
        else:
            # Aucun core point — fallback sur les centroïdes par cluster
            labels = dbscan.labels_
            unique_labels = np.unique(labels[labels >= 0])
            if len(unique_labels) > 0:
                self.core_points_ = np.array(
                    [X_fit[labels == lbl].mean(axis=0) for lbl in unique_labels],
                    dtype=np.float32,
                )
            else:
                # Tout est bruit — conserver tous les points comme référence
                self.core_points_ = X_fit.astype(np.float32)

        n_noise = int((dbscan.labels_ == -1).sum())
        n_core = len(dbscan.core_sample_indices_)
        print(
            f"  [DBSCAN] Tâche {task_id} — eps={self.eps}, min_samples={self.min_samples}, "
            f"n_core={n_core}, n_noise={n_noise}/{len(X_fit)}, "
            f"RAM estimée={self._estimate_ram_bytes()} B"
        )

        # Calcul du seuil sur Task 0 uniquement (pas de leakage inter-tâches)
        if task_id == 0 and self.threshold_ is None:
            scores = self._compute_scores(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [DBSCAN] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule le score d'anomalie = distance au core point le plus proche.

        Un score élevé indique un point éloigné des clusters denses (probable anomalie).

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N]
            Distances minimales aux core points.
        """
        if self.core_points_ is None:
            raise RuntimeError("DBSCANDetector non entraîné. Appeler fit_task() d'abord.")
        # Distance à chaque core point → min par ligne
        dists = euclidean_distances(X.astype(np.float32), self.core_points_)  # [N, n_core]
        return dists.min(axis=1)  # [N]

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (distance au core point le plus proche).

        Un score élevé indique une anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N], dtype=float32
            Scores d'anomalie.
        """
        return self._compute_scores(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le label binaire (0=normal, 1=anormal) en comparant au seuil.

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N], dtype=int64
            Prédictions binaires.

        Raises
        ------
        RuntimeError
            Si threshold_ n'a pas été calculé (fit_task sur Task 0 requis).
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Seuil non calculé. Appeler fit_task(X, task_id=0) sur Task 0 d'abord."
            )
        scores = self.anomaly_score(X)
        return (scores > self.threshold_).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcule l'accuracy binaire. Labels utilisés en évaluation uniquement.

        Parameters
        ----------
        X : np.ndarray [N, d]
        y : np.ndarray [N], valeurs ∈ {0, 1}

        Returns
        -------
        float
            Accuracy.
        """
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    def _estimate_ram_bytes(self) -> int:
        """Estime la RAM modèle (core points) en octets @ FP32."""
        if self.core_points_ is None or self.n_features_ == 0:
            return 0
        return int(self.core_points_.nbytes)  # n_core × d × 4 octets

    def count_parameters(self) -> int:
        """
        Retourne le nombre de paramètres du modèle (éléments des core points).

        Returns
        -------
        int
            n_core_samples × d.
        """
        if self.core_points_ is None:
            return 0
        return int(self.core_points_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        n_core = self.core_points_.shape[0] if self.core_points_ is not None else 0
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        ram = f"{self._estimate_ram_bytes()} B" if self.core_points_ is not None else "—"
        return (
            f"DBSCANDetector | eps={self.eps}, min_samples={self.min_samples} | "
            f"n_core={n_core} | threshold={threshold} | "
            f"strategy={self.cl_strategy} | params={self.count_parameters()} | RAM={ram} @ FP32"
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle (core_points, seuil) au format pickle.

        Parameters
        ----------
        path : str | Path
            Chemin de destination (ex. experiments/exp_008/checkpoints/dbscan_task2.pkl).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[DBSCAN] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DBSCANDetector":
        """
        Charge un modèle sauvegardé.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        DBSCANDetector
        """
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)
