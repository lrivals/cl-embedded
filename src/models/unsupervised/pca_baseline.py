# ruff: noqa: N803, N806  — X est une convention mathématique ML (sklearn API)
"""
Baseline PCA reconstruction error pour la détection d'anomalies en scénario CL.

PC-only — pas de contrainte 64 Ko STM32N6.
Labels utilisés uniquement en évaluation, jamais pendant fit_task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

# Valeurs par défaut — toujours passer par configs/unsupervised_config.yaml
N_COMPONENTS_DEFAULT: int = 2
MIN_EXPLAINED_VARIANCE_DEFAULT: float = 0.95
ANOMALY_PERCENTILE_DEFAULT: int = 95


class PCABaseline:
    """
    Baseline de détection d'anomalies par erreur de reconstruction PCA.

    Le score d'anomalie est l'erreur quadratique moyenne (MSE) entre l'échantillon
    original et sa reconstruction dans le sous-espace PCA. Un score élevé indique
    un échantillon difficile à représenter dans le sous-espace appris → anomalie.

    Deux stratégies CL sont supportées :
    - "refit"       : PCA complet sur les données de la tâche courante uniquement.
    - "incremental" : IncrementalPCA mis à jour batch par batch (simulation online).

    Parameters
    ----------
    config : dict
        Sous-section "pca" de unsupervised_config.yaml.

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA | IncrementalPCA | None
        Modèle PCA entraîné.
    n_components_fitted_ : int | None
        Nombre de composantes réellement utilisées après fit.
    threshold_ : float | None
        Seuil de décision (calculé sur Task 0).
    task_id_ : int
        Index de la dernière tâche entraînée.

    Notes
    -----
    PC-only — pas de contrainte 64 Ko STM32N6.
    Labels utilisés uniquement en évaluation (score, auroc), jamais pendant fit_task.
    """

    def __init__(self, config: dict) -> None:
        self.n_components: int | None = config.get("n_components", N_COMPONENTS_DEFAULT)
        self.min_explained_variance: float = config.get(
            "min_explained_variance", MIN_EXPLAINED_VARIANCE_DEFAULT
        )
        self.anomaly_threshold: float | None = config.get("anomaly_threshold", None)
        self.anomaly_percentile: int = config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT)
        self.cl_strategy: str = config.get("cl_strategy", "refit")

        self.pca_: PCA | IncrementalPCA | None = None
        self.n_components_fitted_: int | None = None
        self.threshold_: float | None = self.anomaly_threshold
        self.task_id_: int = -1

    def _resolve_n_components(self, X: np.ndarray) -> int:
        """
        Détermine le nombre de composantes à conserver.

        Si n_components est fixé → retourner n_components.
        Sinon → sélection automatique par variance expliquée cumulée.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        int
            Nombre de composantes.
        """
        if self.n_components is not None:
            return min(self.n_components, X.shape[1], X.shape[0] - 1)

        # Sélection automatique : nombre de composantes pour atteindre min_explained_variance
        pca_full = PCA(n_components=None, random_state=42)
        pca_full.fit(X)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n = int(np.searchsorted(cumvar, self.min_explained_variance) + 1)
        n = min(n, X.shape[1])
        print(
            f"  [PCA] Sélection automatique : {n} composantes "
            f"({cumvar[n-1]*100:.1f}% variance expliquée)"
        )
        return n

    def fit_task(self, X: np.ndarray, task_id: int) -> "PCABaseline":
        """
        Entraîne le modèle PCA sur les données d'une tâche.

        Si cl_strategy=="refit", un nouveau PCA est entraîné sur la tâche courante.
        Si cl_strategy=="incremental", IncrementalPCA est mis à jour avec les données.

        Le seuil de décision est calculé une seule fois sur Task 0.

        Parameters
        ----------
        X : np.ndarray [N, n_features]
            Données d'entraînement (sans labels).
        task_id : int
            Index de la tâche (0-based).

        Returns
        -------
        self
        """
        self.task_id_ = task_id
        n_comp = self._resolve_n_components(X)

        if self.cl_strategy == "refit":
            self.pca_ = PCA(n_components=n_comp, random_state=42)
            self.pca_.fit(X)
            self.n_components_fitted_ = n_comp
            print(
                f"  [PCA] Tâche {task_id} — refit : {n_comp} composantes, "
                f"variance expliquée={sum(self.pca_.explained_variance_ratio_)*100:.1f}%"
            )

        elif self.cl_strategy == "incremental":
            # IncrementalPCA : mis à jour par batches
            # TODO(dorra) : valider que IncrementalPCA est pertinent pour simulation MCU
            batch_size = max(n_comp, 32)  # IncrementalPCA requiert batch_size >= n_components
            if self.pca_ is None:
                self.pca_ = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
            self.pca_.partial_fit(X)
            self.n_components_fitted_ = n_comp
            print(f"  [PCA] Tâche {task_id} — incremental : partial_fit ({len(X)} samples)")

        else:
            raise ValueError(
                f"cl_strategy inconnu : {self.cl_strategy!r}. Valeurs valides : 'refit', 'incremental'."
            )

        # Calcul du seuil sur Task 0 uniquement
        if task_id == 0 and self.threshold_ is None:
            errors = self.reconstruction_error(X)
            self.threshold_ = float(np.percentile(errors, self.anomaly_percentile))
            print(
                f"  [PCA] Seuil calculé sur Task 0 : {self.threshold_:.6f} "
                f"(percentile {self.anomaly_percentile})"
            )

        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule l'erreur de reconstruction MSE pour chaque échantillon.

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N]
            MSE entre X original et X reconstruit.

        Notes
        -----
        reconstruction = pca.inverse_transform(pca.transform(X))
        error[i] = mean((X[i] - reconstruction[i])^2)
        """
        if self.pca_ is None:
            raise RuntimeError("PCABaseline non entraîné. Appeler fit_task() d'abord.")
        X_proj = self.pca_.transform(X)  # [N, n_components]
        X_reconstructed = self.pca_.inverse_transform(X_proj)  # [N, n_features]
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)  # [N]
        return errors

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie (erreur de reconstruction MSE).

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N], dtype=float32
        """
        return self.reconstruction_error(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit le label binaire (0=normal, 1=anormal).

        Parameters
        ----------
        X : np.ndarray [N, n_features]

        Returns
        -------
        np.ndarray [N], dtype=int64

        Raises
        ------
        RuntimeError
            Si threshold_ n'a pas été calculé.
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
        """
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    def count_parameters(self) -> int:
        """
        Retourne le nombre de paramètres (components_ + mean_).

        Returns
        -------
        int
        """
        if self.pca_ is None or not hasattr(self.pca_, "components_"):
            return 0
        return int(self.pca_.components_.size + self.pca_.mean_.size)

    def summary(self) -> str:
        """Résumé du modèle pour affichage console."""
        n_comp = self.n_components_fitted_ if self.n_components_fitted_ is not None else "—"
        threshold = f"{self.threshold_:.6f}" if self.threshold_ is not None else "—"
        var = "—"
        if self.pca_ is not None and hasattr(self.pca_, "explained_variance_ratio_"):
            var = f"{sum(self.pca_.explained_variance_ratio_)*100:.1f}%"
        return (
            f"PCABaseline | n_components={n_comp} | variance={var} | "
            f"strategy={self.cl_strategy} | threshold={threshold} | "
            f"params={self.count_parameters()}"
        )

    def save(self, path: str | Path) -> None:
        """Sauvegarde le modèle au format pickle."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[PCA] Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PCABaseline":
        """Charge un modèle sauvegardé."""
        import pickle

        with open(Path(path), "rb") as f:
            return pickle.load(f)
