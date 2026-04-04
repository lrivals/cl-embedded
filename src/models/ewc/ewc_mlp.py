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

import torch
import torch.nn as nn


class EWCMlpClassifier(nn.Module):
    """
    MLP binaire avec régularisation EWC Online.

    Architecture :
        Linear(input_dim → 32) + ReLU
        Dropout(p=dropout)
        Linear(32 → 16)        + ReLU
        Dropout(p=dropout)
        Linear(16 → 1)         + Sigmoid

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée. Default : 6 (769 params).
    hidden_dims : list[int]
        Dimensions des couches cachées. Default : [32, 16].
    dropout : float
        Taux de dropout (désactivé automatiquement à l'inférence MCU).

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
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # --- Architecture MLP ---
        # MEM: Linear(6→32)  = (6×32 + 32) × 4 = 896 B @ FP32 / 224 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout)

        # MEM: Linear(32→16) = (32×16 + 16) × 4 = 2 112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout)

        # MEM: Linear(16→1)  = (16×1 + 1) × 4 = 68 B @ FP32 / 17 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], 1)

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
        hidden = torch.relu(self.fc1(x))
        hidden = self.drop1(hidden)

        # MEM activations: 16 × 4 = 64 B @ FP32
        hidden = torch.relu(self.fc2(hidden))
        hidden = self.drop2(hidden)

        # MEM activations: 1 × 4 = 4 B @ FP32
        out = torch.sigmoid(self.fc3(hidden))
        return out

    # ------------------------------------------------------------------
    # Perte EWC
    # ------------------------------------------------------------------

    def ewc_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fisher: dict[str, torch.Tensor] | None,
        theta_star: dict[str, torch.Tensor] | None,
        ewc_lambda: float,
    ) -> torch.Tensor:
        """
        Calcule la perte totale : BCE + terme de régularisation EWC.

        L_EWC(θ) = L_BCE(θ) + λ/2 · Σᵢ Fᵢ (θᵢ - θ*ᵢ)²

        Parameters
        ----------
        x : Tensor [batch_size, input_dim]
        y : Tensor [batch_size, 1]
        fisher : dict[str, Tensor] ou None
            Diagonale de Fisher par nom de paramètre (None pour Task 1).
        theta_star : dict[str, Tensor] ou None
            Snapshot des poids après la tâche précédente (None pour Task 1).
        ewc_lambda : float
            Coefficient de régularisation λ (depuis ewc_config.yaml → ewc.lambda).

        Returns
        -------
        torch.Tensor (scalaire)
            Perte totale L_EWC, différentiable et rétropropageable.

        Notes
        -----
        Si fisher ou theta_star est None (Task 1), retourne la pure perte BCE.
        Référence : docs/models/ewc_mlp_spec.md §1, Kirkpatrick2017EWC eq. 3
        """
        y_hat = self.forward(x)
        bce = nn.functional.binary_cross_entropy(y_hat, y)

        if fisher is None or theta_star is None:
            return bce

        ewc_reg = torch.tensor(0.0, device=x.device)
        for name, param in self.named_parameters():
            if name in fisher and name in theta_star:
                f = fisher[name].to(x.device)
                ts = theta_star[name].to(x.device)
                # Terme élastique : Σ F_i (θ_i - θ*_i)²
                ewc_reg += (f * (param - ts) ** 2).sum()

        return bce + (ewc_lambda / 2.0) * ewc_reg

    # ------------------------------------------------------------------
    # Snapshot θ* (appelé après chaque tâche, avant la suivante)
    # ------------------------------------------------------------------

    def get_theta_star(self) -> dict[str, torch.Tensor]:
        """
        Retourne un snapshot détaché des poids courants (θ* après la tâche courante).

        Doit être appelé APRÈS l'entraînement sur une tâche, AVANT la suivante.

        Returns
        -------
        dict[str, Tensor]
            Copie détachée des paramètres courants (requires_grad=False).
        """
        return {name: param.detach().clone() for name, param in self.named_parameters()}

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
        Estime la RAM des poids seuls (sans état EWC).

        Parameters
        ----------
        dtype : "fp32" (4 B) ou "int8" (1 B)

        Returns
        -------
        int : octets estimés (poids modèle uniquement)

        Notes
        -----
        L'état EWC (Fisher + snapshot θ*) représente ×2 supplémentaire,
        géré en externe par cl_trainer.py (S1-06).
        """
        bytes_per_param = {"fp32": 4, "int8": 1}.get(dtype, 4)
        return self.count_parameters() * bytes_per_param

    def save_state(self, path: str) -> None:
        """Sauvegarde les poids du modèle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    def load_state(self, path: str, device: torch.device | None = None) -> None:
        """Charge les poids du modèle."""
        checkpoint = torch.load(path, map_location=device or "cpu")
        self.load_state_dict(checkpoint["model_state_dict"])

    def __repr__(self) -> str:
        n = self.count_parameters()
        ram = self.estimate_ram_bytes("fp32")
        return (
            f"EWCMlpClassifier("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dims}, "
            f"params={n:,}, "
            f"RAM≈{ram / 1024:.1f}Ko FP32)"
        )
