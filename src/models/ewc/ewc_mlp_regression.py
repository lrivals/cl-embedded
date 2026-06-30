"""
ewc_mlp_regression.py — EWC Online + MLP pour la régression RUL.

Tâche : prédire le Remaining Useful Life continu (float) sur CMAPSS / Pronostia.
Méthode CL : Elastic Weight Consolidation Online (Schwarz et al., 2018).

RAM estimée (input_dim=5, hidden_dims=[32, 16]) :
    Poids : (5×32+32 + 32×16+16 + 16×1+1) × 4 = ~3 Ko @ FP32
    Fisher : ~3 Ko @ FP32
    θ*     : ~3 Ko @ FP32
    TOTAL  : ~9 Ko @ FP32  ✅ << 256 Ko NUCLEO-F439ZI

Références :
    Kirkpatrick2017EWC — EWC (régularisation)
    Schwarz et al., 2018 — EWC Online
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class EWCMlpRegressor(nn.Module):
    """
    MLP régression avec régularisation EWC Online.

    Architecture :
        Linear(input_dim → 32) + ReLU
        Dropout(p=dropout)
        Linear(32 → 16)        + ReLU
        Dropout(p=dropout)
        Linear(16 → 1)         [sortie linéaire — pas de Sigmoid]

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée.
    hidden_dims : list[int]
        Dimensions des couches cachées. Default : [32, 16].
    dropout : float
        Taux de dropout. Default : 0.2.
    ewc_lambda : float
        Coefficient de pénalité EWC. Default : 400.0.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        ewc_lambda: float = 400.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.ewc_lambda = ewc_lambda

        # MEM: Linear(5→32)  = (5×32 + 32) × 4 = 704 B @ FP32 / 176 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout)
        # MEM: Linear(32→16) = (32×16 + 16) × 4 = 2 112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout)
        # MEM: Linear(16→1)  = (16×1 + 1) × 4 = 68 B @ FP32 / 17 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        # Paramètres EWC Online
        self._fisher: dict[str, Tensor] = {}   # Fisher diagonale par paramètre
        self._theta_star: dict[str, Tensor] = {}  # θ* snapshot post-consolidation

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]

        Returns
        -------
        Tensor [batch_size, 1]
            Prédiction RUL ŷ ∈ ℝ (non bornée en inférence).
        """
        # MEM activations: 32 × 4 = 128 B @ FP32
        h = torch.relu(self.fc1(x))
        h = self.drop1(h)
        # MEM activations: 16 × 4 = 64 B @ FP32
        h = torch.relu(self.fc2(h))
        h = self.drop2(h)
        return self.fc3(h)  # shape [batch_size, 1]

    def ewc_penalty(self) -> Tensor:
        """Pénalité EWC = λ/2 · Σ F_i · (θ_i - θ*_i)²."""
        if not self._fisher:
            return torch.tensor(0.0, requires_grad=True)
        penalty = sum(
            (self._fisher[n] * (p - self._theta_star[n]) ** 2).sum()
            for n, p in self.named_parameters()
            if n in self._fisher
        )
        return self.ewc_lambda / 2.0 * penalty

    def consolidate(
        self,
        data_loader: DataLoader,
        n_samples: int = 200,
        rul_scale: float = 1.0,
    ) -> None:
        """
        Calcule la Fisher diagonale sur `n_samples` exemples et snapshote θ*.

        Doit être appelé APRÈS l'entraînement d'une tâche, AVANT la suivante.
        Fisher via gradient MSE (pas de log-vraisemblance binaire).

        Parameters
        ----------
        rul_scale : float
            Facteur de dé-normalisation des cibles (ex. 125 pour CMAPSS).
            Doit correspondre à la valeur utilisée pendant l'entraînement.
        """
        self.eval()
        criterion = nn.MSELoss()
        fisher_accum: dict[str, Tensor] = {
            n: torch.zeros_like(p) for n, p in self.named_parameters()
        }
        count = 0
        for x_batch, y_batch in data_loader:
            if count >= n_samples:
                break
            self.zero_grad()
            y_pred = self(x_batch)
            y_target = y_batch.float().squeeze() / rul_scale
            loss = criterion(y_pred.squeeze(), y_target)
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher_accum[n] += p.grad.data.clone() ** 2
            count += len(x_batch)

        n_batches = max(1, count // data_loader.batch_size)
        self._fisher = {n: f / n_batches for n, f in fisher_accum.items()}
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters()}
        self.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """Estime la RAM des poids + Fisher + θ* (× 3 pour l'état EWC complet)."""
        bytes_per_param = {"fp32": 4, "int8": 1}.get(dtype, 4)
        return self.count_parameters() * bytes_per_param * 3

    def __repr__(self) -> str:
        n = self.count_parameters()
        return (
            f"EWCMlpRegressor("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dims}, "
            f"params={n:,}, "
            f"RAM≈{self.estimate_ram_bytes() // 1024:.1f}Ko FP32)"
        )
