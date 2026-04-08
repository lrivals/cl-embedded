# ruff: noqa: N803, N806  — X, Sigma sont des conventions mathématiques ML (sklearn API)
"""
Baseline Mahalanobis pour la détection d'anomalies en scénario domain-incremental.

μ et Σ⁻¹ calculés offline (fit_task). Inférence = produit matriciel uniquement.
Labels utilisés uniquement en évaluation, jamais pendant fit_task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
ANOMALY_PERCENTILE_DEFAULT: int = 95
REG_COVAR_DEFAULT: float = 1e-6  # régularisation Σ : Σ_reg = Σ + reg_covar * I
CL_STRATEGY_DEFAULT: str = "refit"  # "refit" uniquement (recalcul μ, Σ⁻¹ à chaque tâche)


class MahalanobisDetector:
    """
    Baseline de détection d'anomalies par distance de Mahalanobis en scénario domain-incremental.

    μ et Σ⁻¹ sont calculés offline (fit_task). L'inférence est un simple produit matriciel :
    score(x) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)).

    Avantage embarqué : empreinte mémoire = d + d² floats.
    Pour d=4 (Dataset 2) → 80 B @ FP32 / 20 B @ INT8.  # MEM: 80 B @ FP32 / 20 B @ INT8

    En scénario CL (cl_strategy="refit"), μ et Σ⁻¹ sont recalculés sur chaque nouvelle tâche.
    Pas de mémoire inter-tâches — évalue l'oubli catastrophique par recalcul complet.

    Parameters
    ----------
    config : dict
        Sous-section "mahalanobis" de unsupervised_config.yaml.

    Attributes
    ----------
    mu_ : np.ndarray [d] | None
        Vecteur moyen calculé sur la dernière tâche.
    sigma_inv_ : np.ndarray [d, d] | None
        Matrice inverse de covariance (calculée offline, utilisée online).
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0 si anomaly_threshold est null).
    task_id_ : int
        Index de la dernière tâche entraînée.
    n_features_ : int
        Dimension des features (fixée au premier fit_task).

    Notes
    -----
    Compatible STM32N6 (< 64 Ko RAM pour d ≤ 126).
    Labels utilisés uniquement en évaluation (score, auroc), jamais pendant fit_task.
    """

    def __init__(self, config: dict) -> None:
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.reg_covar: float = config.get("reg_covar", REG_COVAR_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", CL_STRATEGY_DEFAULT)

        self.mu_: np.ndarray | None = None
        self.sigma_inv_: np.ndarray | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1
        self.n_features_: int = 0

    def fit_task(self, X: np.ndarray, task_id: int) -> "MahalanobisDetector":
        """
        Calcule μ et Σ⁻¹ sur les données d'une tâche (offline).

        Le seuil de décision est calculé sur Task 0 (première tâche) uniquement.

        Parameters
        ----------
        X : np.ndarray [N, d]
            Données d'entraînement (non labelisées — labels exclus).
        task_id : int
            Index de la tâche (0-based). Task 0 = Pump, 1 = Turbine, 2 = Compressor.

        Returns
        -------
        self
        """
        self.task_id_ = task_id
        self.n_features_ = X.shape[1]

        # Calcul offline de μ et Σ
        self.mu_ = X.mean(axis=0)  # [d]
        cov = np.cov(X, rowvar=False)  # [d, d]
        cov_reg = cov + self.reg_covar * np.eye(self.n_features_)  # régularisation

        self.sigma_inv_ = np.linalg.inv(cov_reg)  # [d, d] — offline uniquement
        print(
            f"  [Mahalanobis] Tâche {task_id} — μ shape={self.mu_.shape}, "
            f"Σ⁻¹ shape={self.sigma_inv_.shape}, "
            f"RAM estimée={self._estimate_ram_bytes()} B"
        )

        # Calcul du seuil sur Task 0 uniquement (pas de leakage inter-tâches)
        if task_id == 0 and self.threshold_ is None:
            scores = self._compute_distances(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [Mahalanobis] Seuil calculé sur Task 0 : {self.threshold_:.4f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule la distance de Mahalanobis pour chaque échantillon.

        Distance = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) — produit matriciel vectorisé.

        Parameters
        ----------
        X : np.ndarray [N, d]

        Returns
        -------
        np.ndarray [N]
            Distances de Mahalanobis.
        """
        if self.mu_ is None or self.sigma_inv_ is None:
            raise RuntimeError("MahalanobisDetector non entraîné. Appeler fit_task() d'abord.")
        diff = X - self.mu_  # [N, d]
        # (diff @ Σ⁻¹) * diff → somme ligne → distances²
        left = diff @ self.sigma_inv_  # [N, d]
        dist_sq = (left * diff).sum(axis=1)  # [N] — (x-μ)ᵀ Σ⁻¹ (x-μ)
        return np.sqrt(np.maximum(dist_sq, 0.0))  # [N] — clip numérique

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (distance de Mahalanobis).

        Un score élevé indique une anomalie probable.

        Parameters
        ----------
        X : np.ndarray [N, d]

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
        """Estime la RAM modèle (μ + Σ⁻¹) en octets @ FP32."""
        if self.n_features_ == 0:
            return 0
        d = self.n_features_
        return (d + d * d) * 4  # float32 = 4 octets

    def count_parameters(self) -> int:
        """
        Retourne le nombre de paramètres du modèle (μ + Σ⁻¹).

        Returns
        -------
        int
            d + d² (vecteur moyen + matrice inverse de covariance).
        """
        if self.mu_ is None or self.sigma_inv_ is None:
            return 0
        return int(self.mu_.size + self.sigma_inv_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        d = self.n_features_
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "—"
        ram = f"{self._estimate_ram_bytes()} B" if d > 0 else "—"
        return (
            f"MahalanobisDetector | d={d} | "
            f"threshold={threshold} | strategy={self.cl_strategy} | "
            f"params={self.count_parameters()} | RAM={ram} @ FP32"
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle (μ, Σ⁻¹, seuil) au format pickle.

        Parameters
        ----------
        path : str | Path
            Chemin de destination (ex. experiments/exp_007/checkpoints/mahalanobis_task2.pkl).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Mahalanobis] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisDetector":
        """
        Charge un modèle sauvegardé.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        MahalanobisDetector
        """
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)
