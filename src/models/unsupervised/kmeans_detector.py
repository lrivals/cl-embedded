# ruff: noqa: N803, N806  — X, X0 sont des conventions mathématiques ML (sklearn API)
"""
Baseline K-Means pour la détection d'anomalies en scénario domain-incremental.

PC-only — pas de contrainte 64 Ko STM32N6.
Labels utilisés uniquement en évaluation, jamais pendant fit_task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
K_METHOD_DEFAULT: str = "silhouette"  # "silhouette" | "elbow" | "fixed"
K_FIXED_DEFAULT: int = 3
K_MIN_DEFAULT: int = 2
K_MAX_DEFAULT: int = 10
ANOMALY_PERCENTILE_DEFAULT: int = 95
N_INIT_DEFAULT: int = 10
MAX_ITER_DEFAULT: int = 300
EMA_ALPHA_DEFAULT: float = 0.3  # poids de la tâche courante dans la MAJ EMA du seuil


class KMeansDetector:
    """
    Baseline de détection d'anomalies par K-Means en scénario domain-incremental.

    Le score d'anomalie est la distance euclidienne au centroïde le plus proche.
    Un échantillon est classifié comme anormal si son score dépasse `threshold_`.

    En scénario CL (cl_strategy="refit"), le modèle est réentraîné à chaque nouvelle
    tâche sur les données de cette tâche uniquement (pas de mémoire inter-tâches).
    En scénario CL (cl_strategy="accumulate"), les centroïdes sont conservés entre tâches.

    Parameters
    ----------
    config : dict
        Sous-section "kmeans" de unsupervised_config.yaml.

    Attributes
    ----------
    kmeans_ : sklearn.cluster.KMeans | None
        Modèle K-Means entraîné (None avant fit_task).
    threshold_ : float | None
        Seuil de décision (EMA inter-tâches si ema_alpha > 0, figé sinon).
    threshold_history_ : list[float]
        Seuil enregistré après chaque tâche [seuil_T0, seuil_T1, ...].
    k_selected_ : list[int]
        K sélectionné à chaque tâche (historique).
    task_id_ : int
        Index de la dernière tâche entraînée.

    Notes
    -----
    PC-only — pas de contrainte 64 Ko STM32N6.
    Labels utilisés uniquement en évaluation (score, auroc), jamais pendant fit_task.
    """

    def __init__(self, config: dict) -> None:
        self.k_method: str = config.get("k_method", K_METHOD_DEFAULT)
        self.k_fixed: int = config.get("k_fixed", K_FIXED_DEFAULT)
        self.k_min: int = config.get("k_min", K_MIN_DEFAULT)
        self.k_max: int = config.get("k_max", K_MAX_DEFAULT)
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.n_init: int = config.get("n_init", N_INIT_DEFAULT)
        self.max_iter: int = config.get("max_iter", MAX_ITER_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", "refit")
        self.ema_alpha: float = config.get("ema_alpha", EMA_ALPHA_DEFAULT)

        self.kmeans_: KMeans | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.threshold_history_: list[float] = []
        self.k_selected_: list[int] = []
        self.task_id_: int = -1

    def _select_k(self, X: np.ndarray) -> int:
        """
        Sélectionne K optimal selon la méthode configurée.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
            Données d'entraînement de la tâche courante.

        Returns
        -------
        int
            K optimal sélectionné.
        """
        if self.k_method == "fixed":
            return self.k_fixed

        ks = range(self.k_min, min(self.k_max + 1, len(X)))
        scores = []

        for k in ks:
            km = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=42,
            )
            labels = km.fit_predict(X)

            if self.k_method == "silhouette":
                # Silhouette score : maximiser (meilleure séparation inter-cluster)
                score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
                scores.append(score)
            elif self.k_method == "elbow":
                # Elbow : minimiser l'inertie (coude de la courbe)
                scores.append(km.inertia_)
            else:
                raise ValueError(
                    f"k_method inconnu : {self.k_method!r}. "
                    "Valeurs valides : 'silhouette', 'elbow', 'fixed'."
                )

        if self.k_method == "silhouette":
            k_opt = list(ks)[int(np.argmax(scores))]
        else:
            # Elbow : second dérivé discret (méthode du coude)
            scores_arr = np.array(scores)
            deltas = np.diff(scores_arr)
            delta2 = np.diff(deltas)
            k_opt = list(ks)[int(np.argmax(np.abs(delta2))) + 1] if len(delta2) > 0 else self.k_min

        return k_opt

    def fit_task(self, X: np.ndarray, task_id: int) -> "KMeansDetector":
        """
        Entraîne le modèle K-Means sur les données d'une tâche.

        Si cl_strategy=="refit", le modèle est réinitialisé à chaque tâche.
        Le seuil de décision est calculé sur Task 0 (première tâche) uniquement.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
            Données d'entraînement (non labelisées — labels exclus).
        task_id : int
            Index de la tâche (0-based). Task 0 = Pump, 1 = Turbine, 2 = Compressor.

        Returns
        -------
        self
        """
        self.task_id_ = task_id

        k_opt = self._select_k(X)
        self.k_selected_.append(k_opt)
        print(f"  [KMeans] Tâche {task_id} — K sélectionné : {k_opt} " f"(méthode={self.k_method})")

        self.kmeans_ = KMeans(
            n_clusters=k_opt,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=42,
        )
        self.kmeans_.fit(X)

        # Mise à jour EMA du seuil : figé sur Task 0, EMA sur tâches suivantes
        distances = self._compute_distances(X)
        percentile_new = float(np.percentile(distances, self.anomaly_percentile))

        if task_id == 0 and self.threshold_ is None:
            self.threshold_ = percentile_new  # MEM: 4 B @ FP32 / 4 B @ INT8 (scalaire EMA)
            print(
                f"  [KMeans] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )
        elif task_id > 0 and self.threshold_ is not None and self.ema_alpha > 0.0:
            self.threshold_ = (  # MEM: 4 B @ FP32 / 4 B @ INT8 (scalaire EMA)
                self.ema_alpha * percentile_new + (1 - self.ema_alpha) * self.threshold_
            )
            print(
                f"  [KMeans] Seuil EMA mis à jour (Task {task_id}) : "
                f"new_pct={percentile_new:.4f} → threshold={self.threshold_:.4f} "
                f"(alpha={self.ema_alpha})"
            )

        if self.threshold_ is not None:
            self.threshold_history_.append(self.threshold_)

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule la distance euclidienne au centroïde le plus proche pour chaque échantillon.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N]
            Distances min aux centroïdes.
        """
        if self.kmeans_ is None:
            raise RuntimeError("KMeansDetector non entraîné. Appeler fit_task() d'abord.")
        # sklearn transform() retourne [N, k] distances aux centroides
        dists = self.kmeans_.transform(X)  # [N, k]
        return dists.min(axis=1)  # [N] — distance au centroïde le plus proche

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (distance au centroïde le plus proche).

        Un score élevé indique une anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N], dtype=float32
            Scores d'anomalie.
        """
        return self._compute_distances(X).astype(np.float32)

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
        Calcule l'accuracy binaire. Labels utilisés en évaluation uniquement.

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
        Retourne le nombre de paramètres du modèle (centroides uniquement).

        Returns
        -------
        int
            k * n_features (nombre de valeurs float dans cluster_centers_).
        """
        if self.kmeans_ is None:
            return 0
        return int(self.kmeans_.cluster_centers_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        k = self.kmeans_.n_clusters if self.kmeans_ is not None else "—"
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        hist = [f"{t:.3f}" for t in self.threshold_history_]
        return (
            f"KMeansDetector | k={k} | method={self.k_method} | "
            f"threshold={threshold} | ema_alpha={self.ema_alpha} | "
            f"history={hist} | strategy={self.cl_strategy} | "
            f"params={self.count_parameters()}"
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle (centroides + seuil + historique) via pickle.

        Parameters
        ----------
        path : str | Path
            Chemin de destination (ex. experiments/exp_005/checkpoints/kmeans_task2.pkl).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[KMeans] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "KMeansDetector":
        """
        Charge un modèle sauvegardé.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        KMeansDetector
        """
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)
