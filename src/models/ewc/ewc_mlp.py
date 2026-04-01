"""
ewc_mlp.py — EWC Online + MLP pour la classification binaire.

Méthode : Elastic Weight Consolidation Online (Schwarz et al., 2018)
Dataset cible : Dataset 2 — Industrial Equipment Monitoring
Taxonomie CL : Regularization-based

RAM estimée (FP32) :
    - Poids modèle   : ~3 Ko (769 params × 4 B)
    - Fisher diag    : ~3 Ko (769 scalaires × 4 B)
    - Snapshot θ*    : ~3 Ko (769 scalaires × 4 B)
    - Activations    : ~512 B (peak forward)
    - TOTAL          : ~9.5 Ko  ✅ << 64 Ko cible STM32N6

Références :
    Kirkpatrick et al. (2017). Overcoming catastrophic forgetting. PNAS.
    Schwarz et al. (2018). Progress & compress. ICML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWCMlpClassifier(nn.Module):
    """
    MLP binaire avec régularisation EWC Online.

    Architecture :
        Linear(input_dim → 32) + ReLU
        Linear(32 → 16)        + ReLU
        Linear(16 → 1)         + Sigmoid

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée (6 pour Dataset 2).
    hidden_dims : list[int]
        Dimensions des couches cachées. Default : [32, 16].
    dropout : float
        Taux de dropout (désactivé automatiquement à l'inférence MCU).
    ewc_lambda : float
        Coefficient de régularisation EWC.
    ewc_gamma : float
        Facteur de décroissance Fisher Online (0 < γ ≤ 1).

    Notes
    -----
    MCU compatibility :
        - ReLU uniquement (INT8-friendly CMSIS-NN)
        - Pas de BatchNorm
        - SGD optimizer recommandé dans la boucle CL
        - Annotations # MEM: présentes sur chaque couche
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        ewc_lambda: float = 1000.0,
        ewc_gamma: float = 0.9,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma

        # --- Architecture MLP ---
        # MEM: Linear(6→32)  = (6×32 + 32) × 4 = 896 B @ FP32 / 224 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout)

        # MEM: Linear(32→16) = (32×16 + 16) × 4 = 2 112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout)

        # MEM: Linear(16→1)  = (16×1 + 1) × 4 = 68 B @ FP32 / 17 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        # --- État EWC Online ---
        # Fisher diagonale accumulée (mise à jour après chaque tâche)
        self._fisher: dict[str, torch.Tensor] = {}
        # Snapshot des poids optimaux après chaque tâche
        self._params_star: dict[str, torch.Tensor] = {}
        # Indique si l'état EWC est initialisé (au moins 1 tâche vue)
        self._ewc_initialized: bool = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]

        Returns
        -------
        Tensor [batch_size, 1]
            Probabilité de défaut ŷ ∈ [0, 1].
        """
        # MEM activations: 32 × 4 = 128 B @ FP32
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        # MEM activations: 16 × 4 = 64 B @ FP32
        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        # MEM activations: 1 × 4 = 4 B @ FP32
        x = torch.sigmoid(self.fc3(x))
        return x

    # ------------------------------------------------------------------
    # Perte EWC
    # ------------------------------------------------------------------

    def ewc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Calcule la perte totale : BCE + terme de régularisation EWC.

        Parameters
        ----------
        x : Tensor [batch_size, input_dim]
        y : Tensor [batch_size, 1]

        Returns
        -------
        total_loss : Tensor (scalaire)
        loss_components : dict {"bce": float, "ewc_reg": float, "total": float}
        """
        y_hat = self.forward(x)
        bce = F.binary_cross_entropy(y_hat, y)

        ewc_reg = torch.tensor(0.0, device=x.device)

        if self._ewc_initialized:
            for name, param in self.named_parameters():
                if name in self._fisher and name in self._params_star:
                    fisher = self._fisher[name].to(x.device)
                    theta_star = self._params_star[name].to(x.device)
                    # Terme élastique : λ/2 · Σ F_i (θ_i - θ*_i)²
                    ewc_reg += (fisher * (param - theta_star) ** 2).sum()

        ewc_reg = (self.ewc_lambda / 2) * ewc_reg
        total_loss = bce + ewc_reg

        return total_loss, {
            "bce": bce.item(),
            "ewc_reg": ewc_reg.item(),
            "total": total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Mise à jour de l'état EWC (fin de tâche)
    # ------------------------------------------------------------------

    def update_ewc_state(
        self, dataloader: torch.utils.data.DataLoader, device: torch.device
    ) -> None:
        """
        Met à jour la Fisher diagonale (accumulation Online) et le snapshot θ*.

        Doit être appelé APRÈS l'entraînement sur chaque tâche.

        Parameters
        ----------
        dataloader : DataLoader
            Données de la tâche qui vient de se terminer.
        device : torch.device
        """
        new_fisher = self._compute_fisher(dataloader, device)

        if not self._ewc_initialized:
            # Première tâche : initialisation directe
            self._fisher = {n: f.clone() for n, f in new_fisher.items()}
        else:
            # Accumulation Online avec décroissance γ
            for name in self._fisher:
                self._fisher[name] = (
                    self.ewc_gamma * self._fisher[name]
                    + new_fisher.get(name, torch.zeros_like(self._fisher[name]))
                )

        # Snapshot des poids courants
        self._params_star = {
            name: param.detach().clone()
            for name, param in self.named_parameters()
        }

        self._ewc_initialized = True

    def _compute_fisher(
        self, dataloader: torch.utils.data.DataLoader, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """
        Calcule la diagonale de la matrice de Fisher par la méthode empirique.

        F_i ≈ E[(∂ log p(y|x,θ) / ∂θ_i)²]

        Returns
        -------
        dict[str, Tensor] : Fisher diagonale par nom de paramètre.
        """
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.named_parameters()
            if p.requires_grad
        }

        self.eval()
        n_batches = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            self.zero_grad()
            y_hat = self.forward(x_batch)
            loss = F.binary_cross_entropy(y_hat, y_batch)
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach() ** 2

            n_batches += 1

        # Normalisation
        if n_batches > 0:
            fisher = {n: f / n_batches for n, f in fisher.items()}

        self.train()
        return fisher

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Nombre total de paramètres (entraînables + non entraînables)."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """
        Estime la RAM des poids seuls.

        Parameters
        ----------
        dtype : "fp32" (4 B) ou "int8" (1 B)

        Returns
        -------
        int : octets estimés (poids modèle uniquement)
        """
        bytes_per_param = {"fp32": 4, "int8": 1}.get(dtype, 4)
        model_bytes = self.count_parameters() * bytes_per_param

        # Overhead EWC : Fisher + snapshot = ×2 le modèle
        ewc_bytes = model_bytes * 2 if self._ewc_initialized else 0

        return model_bytes + ewc_bytes

    def save_state(self, path: str) -> None:
        """Sauvegarde poids + état EWC."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "fisher": self._fisher,
                "params_star": self._params_star,
                "ewc_initialized": self._ewc_initialized,
                "ewc_lambda": self.ewc_lambda,
                "ewc_gamma": self.ewc_gamma,
            },
            path,
        )

    def load_state(self, path: str, device: torch.device | None = None) -> None:
        """Charge poids + état EWC."""
        checkpoint = torch.load(path, map_location=device or "cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self._fisher = checkpoint.get("fisher", {})
        self._params_star = checkpoint.get("params_star", {})
        self._ewc_initialized = checkpoint.get("ewc_initialized", False)
        self.ewc_lambda = checkpoint.get("ewc_lambda", self.ewc_lambda)
        self.ewc_gamma = checkpoint.get("ewc_gamma", self.ewc_gamma)

    def __repr__(self) -> str:
        n = self.count_parameters()
        ram = self.estimate_ram_bytes("fp32")
        return (
            f"EWCMlpClassifier("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dims}, "
            f"params={n:,}, "
            f"RAM≈{ram/1024:.1f}Ko FP32, "
            f"λ={self.ewc_lambda}, γ={self.ewc_gamma})"
        )
