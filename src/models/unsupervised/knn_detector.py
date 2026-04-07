# ruff: noqa: N803, N806  — X, X_ref sont des conventions mathématiques ML (sklearn API)
"""
Baseline KNN pour la détection d'anomalies en scénario domain-incremental.

PC-only — pas de contrainte 64 Ko STM32N6.
Labels utilisés uniquement en évaluation, jamais pendant fit_task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
N_NEIGHBORS_DEFAULT: int = 5
METRIC_DEFAULT: str = "euclidean"
ANOMALY_PERCENTILE_DEFAULT: int = 95


class KNNDetector:
    """
    Baseline de détection d'anomalies par distance aux k plus proches voisins.

    Le score d'anomalie est la distance moyenne aux k plus proches voisins dans
    l'ensemble de référence. Aucun label n'est utilisé pendant l'entraînement.

    En scénario CL (cl_strategy="accumulate"), les échantillons de référence
    s'accumulent entre tâches — le modèle grandit mais ne « oublie » pas.
    En scénario CL (cl_strategy="refit"), seule la tâche courante est conservée.

    Parameters
    ----------
    config : dict
        Sous-section "knn" de unsupervised_config.yaml.

    Attributes
    ----------
    nn_ : sklearn.neighbors.NearestNeighbors | None
        Modèle KNN entraîné (None avant fit_task).
    X_ref_ : np.ndarray | None
        Données de référence stockées ([N_total, n_features]).
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0).
    task_id_ : int
        Index de la dernière tâche entraînée.

    Notes
    -----
    PC-only — pas de contrainte 64 Ko STM32N6.
    `cl_strategy="accumulate"` : RAM croissante au fil des tâches (acceptable PC-only).
    """

    def __init__(self, config: dict) -> None:
        self.n_neighbors: int = config.get("n_neighbors", N_NEIGHBORS_DEFAULT)
        self.metric: str = config.get("metric", METRIC_DEFAULT)
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", "accumulate")

        self.nn_: NearestNeighbors | None = None
        self.X_ref_: np.ndarray | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1

    def fit_task(self, X: np.ndarray, task_id: int) -> "KNNDetector":
        """
        Met à jour le modèle KNN avec les données d'une nouvelle tâche.

        Si cl_strategy=="accumulate", les données sont concaténées aux références existantes.
        Si cl_strategy=="refit", les références sont remplacées par les nouvelles données.

        Le seuil de décision est calculé une seule fois sur Task 0.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
            Données d'entraînement (sans labels).
        task_id : int
            Index de la tâche (0-based). Task 0 = Pump, 1 = Turbine, 2 = Compressor.

        Returns
        -------
        self
        """
        self.task_id_ = task_id

        if self.cl_strategy == "accumulate":
            if self.X_ref_ is None:
                self.X_ref_ = X.copy()
            else:
                self.X_ref_ = np.concatenate([self.X_ref_, X], axis=0)
            print(
                f"  [KNN] Tâche {task_id} — accumulate : "
                f"{len(X)} nouveaux + {len(self.X_ref_) - len(X)} existants = "
                f"{len(self.X_ref_)} références totales"
            )
        else:
            # refit : seule la tâche courante est conservée
            self.X_ref_ = X.copy()
            print(f"  [KNN] Tâche {task_id} — refit : {len(X)} références")

        # (Re)construire l'index KNN — k borné pour éviter erreur sur petits jeux
        k_eff = min(self.n_neighbors, len(self.X_ref_) - 1)
        self.nn_ = NearestNeighbors(
            n_neighbors=k_eff,
            metric=self.metric,
            algorithm="auto",
            n_jobs=-1,
        )
        self.nn_.fit(self.X_ref_)

        # Calcul du seuil sur Task 0 uniquement (pas de leakage inter-tâches)
        if task_id == 0 and self.threshold_ is None:
            scores = self._compute_scores(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [KNN] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les scores d'anomalie pour chaque échantillon.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N]
            Distance moyenne aux k plus proches voisins.
        """
        if self.nn_ is None:
            raise RuntimeError("KNNDetector non entraîné. Appeler fit_task() d'abord.")
        distances, _ = self.nn_.kneighbors(X)  # distances : [N, k]
        return distances.mean(axis=1)  # [N] — distance moyenne aux k voisins

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (distance moyenne aux k plus proches voisins).

        Un score élevé indique une anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

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
        X : np.ndarray [N, n_features]

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
        Calcule l'accuracy binaire. Labels utilisés uniquement en évaluation.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
        y : np.ndarray [N], valeurs ∈ {0, 1}

        Returns
        -------
        float
            Accuracy.
        """
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    def count_parameters(self) -> int:
        """
        Retourne le nombre de points de référence × n_features (taille de X_ref_).

        Returns
        -------
        int
            N_ref × n_features (nombre de valeurs float dans X_ref_).
        """
        if self.X_ref_ is None:
            return 0
        return int(self.X_ref_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        n_ref = len(self.X_ref_) if self.X_ref_ is not None else 0
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        return (
            f"KNNDetector | k={self.n_neighbors} | metric={self.metric} | "
            f"strategy={self.cl_strategy} | n_ref={n_ref} | threshold={threshold}"
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle (X_ref_ + index KNN + seuil) via pickle.

        Parameters
        ----------
        path : str | Path
            Chemin de destination (ex. experiments/exp_005/checkpoints/knn_task2.pkl).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[KNN] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "KNNDetector":
        """
        Charge un modèle sauvegardé.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        KNNDetector
        """
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)
