"""
hdc_regressor.py — Régression linéaire sur embeddings HDC.

Tâche : prédire un RUL continu à partir d'embeddings hyperdimensionnels.
Apprentissage : descente de gradient (SGD) sur MSE — pas d'accumulation de prototypes.
Oubli catastrophique : atténué par la nature distribuite des embeddings HDC.

RAM estimée (D=1024) :
    Vecteurs de base : D × N_LEVELS × 1 = 10 240 B @ INT8
    Vecteur poids w  : D × 4 = 4 096 B @ FP32
    TOTAL            : ~14 Ko  ✅ << 256 Ko NUCLEO-F439ZI

Référence : Benatti2019HDC (encodage HDC), hdc_classifier.py (encode_observation)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.hdc.base_vectors import generate_base_hvectors
from src.models.hdc.hdc_classifier import encode_observation


class HDCRegressor:
    """
    Régression linéaire sur embeddings hyperdimensionnels.

    Réutilise l'encodage HDC existant (base vectors + quantification par niveaux)
    et apprend un vecteur de poids w par SGD sur MSE.

    Parameters
    ----------
    D : int
        Dimension des hypervecteurs. Default : 1024.
    n_levels : int
        Niveaux de quantification par feature. Default : 10.
    n_features : int
        Dimension de l'espace d'entrée. Default : 5 (CMAPSS top-5).
    lr : float
        Taux d'apprentissage SGD. Default : 0.01.
    seed : int
        Graine pour la génération des vecteurs de base. Default : 42.
    """

    def __init__(
        self,
        D: int = 1024,
        n_levels: int = 10,
        n_features: int = 5,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.D = D
        self.n_levels = n_levels
        self.n_features = n_features

        # Vecteur de poids linéaire — MEM: D × 4 = 4 096 B @ FP32
        self.w = nn.Parameter(torch.zeros(D))
        self.optimizer = torch.optim.SGD([self.w], lr=lr)

        # Vecteurs de base (int8) — MEM: D × N_LEVELS = 10 240 B @ INT8 pour D=1024
        self._H_level, self._H_pos = generate_base_hvectors(
            D=D,
            n_levels=n_levels,
            n_features=n_features,
            seed=seed,
        )

        # Bornes de features pour la quantification (initialisées via set_feature_bounds)
        self._feature_bounds: list[tuple[float, float]] | None = None

    def set_feature_bounds(self, x: np.ndarray) -> None:
        """
        Calcule et fixe les bornes de features depuis un batch de données.

        Doit être appelé avant fit_batch() ou predict(). Typiquement appelé
        sur les données de la première tâche.

        Parameters
        ----------
        x : np.ndarray [N, n_features]
            Données représentatives pour calculer min/max par feature.
        """
        self._feature_bounds = [
            (float(x[:, i].min()), float(x[:, i].max()))
            for i in range(x.shape[1])
        ]

    def _encode(self, x: np.ndarray) -> Tensor:
        """
        Encode un batch x ∈ ℝ^(N × n_features) en hypervecteurs {±1}^(N × D).

        Réutilise encode_observation() de hdc_classifier vectorisé sur le batch.

        Parameters
        ----------
        x : np.ndarray [N, n_features], dtype=float32

        Returns
        -------
        Tensor [N, D], dtype=float32

        Raises
        ------
        RuntimeError
            Si set_feature_bounds() n'a pas été appelé au préalable.
        """
        if self._feature_bounds is None:
            raise RuntimeError(
                "Feature bounds non définies. Appeler set_feature_bounds(x) d'abord."
            )
        hvecs = np.stack(
            [
                encode_observation(
                    sample,
                    self._H_level,
                    self._H_pos,
                    self._feature_bounds,
                    self.n_levels,
                    self.D,
                )
                for sample in x
            ]
        )  # [N, D] int8
        return torch.tensor(hvecs, dtype=torch.float32)

    def fit_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Met à jour w sur un mini-batch (online learning).

        Parameters
        ----------
        x : np.ndarray [N, n_features], dtype=float32
        y : np.ndarray [N], dtype=float32
            Valeurs RUL cibles.

        Returns
        -------
        float : MSE loss sur ce batch.
        """
        hvecs = self._encode(x)                        # [N, D]
        y_pred = (hvecs * self.w).sum(dim=1)           # [N]
        loss = nn.MSELoss()(y_pred, torch.tensor(y, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prédiction RUL pour un batch de features.

        Parameters
        ----------
        x : np.ndarray [N, n_features], dtype=float32

        Returns
        -------
        np.ndarray [N], dtype=float32
        """
        with torch.no_grad():
            hvecs = self._encode(x)
            return (hvecs * self.w).sum(dim=1).numpy()

    def count_parameters(self) -> int:
        return int(self.w.numel())

    def estimate_ram_bytes(self) -> int:
        """Estime la RAM totale : vecteurs de base INT8 + poids FP32."""
        base_vectors_bytes = self.D * self.n_levels * 1   # MEM: H_level @ INT8
        w_bytes = self.D * 4                               # MEM: w @ FP32
        return base_vectors_bytes + w_bytes

    def __repr__(self) -> str:
        return (
            f"HDCRegressor("
            f"D={self.D}, "
            f"n_levels={self.n_levels}, "
            f"n_features={self.n_features}, "
            f"RAM≈{self.estimate_ram_bytes() // 1024:.1f}Ko)"
        )
