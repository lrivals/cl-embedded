"""
ewc_mlp_int8.py — EWC MLP avec simulation fake-quantization INT8.

Implémente Gap 3 : SGD INT8 backprop simulé par fake-quant (weights + activations).
Les gradients circulent en FP32 (straight-through estimator automatique avec PyTorch QAT).

Critère : AUROC_INT8 ≥ AUROC_FP32 - 0.02 (docs/triple_gap.md Gap 3)

Usage :
    from src.models.ewc.ewc_mlp_int8 import EWCMlpInt8Classifier
    model = EWCMlpInt8Classifier(input_dim=5)
    # Entraînement identique à EWCMlpClassifier via ewc_loss()

Référence : Ravaglia2021QLRCL (rejeu latent UINT8), ewc_mlp.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.quantization as quant


class EWCMlpInt8Classifier(nn.Module):
    """
    MLP binaire EWC avec fake-quantization INT8 (Quantization-Aware Training).

    Architecture identique à EWCMlpClassifier :
        Linear(input_dim → 32) + FakeQuant + ReLU
        Linear(32 → 16)        + FakeQuant + ReLU
        Linear(16 → 1)         + Sigmoid

    Quantization :
        Weights : per-channel symmetric INT8 (torch PerChannelMinMaxObserver)
        Activations : per-tensor affine INT8 (torch HistogramObserver)

    Notes
    -----
    MCU mapping (ewc_head_int8.c) :
        Q7  (int8_t)  — activations
        Q15 (int16_t) — accumulateurs MAC
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # --- Couches linéaires — identiques à EWCMlpClassifier ---
        # MEM: Linear(5→32) = (5×32+32)×4 = 704 B @ FP32 / 176 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # MEM: Linear(32→16) = (32×16+16)×4 = 2112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # MEM: Linear(16→1) = (16×1+1)×4 = 68 B @ FP32 / 17 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        # Fake-quantizers pour activations (per-tensor affine)
        # MEM: observers = négligeable (scalaires FP32)
        def _act_fq() -> quant.FakeQuantize:
            return quant.FakeQuantize.with_args(
                observer=quant.HistogramObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
            )()

        self.fq_input = _act_fq()
        self.fq_h1 = _act_fq()
        self.fq_h2 = _act_fq()

        # Fake-quantizers pour poids (per-channel symmetric)
        _w_fq_cls = quant.FakeQuantize.with_args(
            observer=quant.PerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )
        self.fq_w1 = _w_fq_cls()
        self.fq_w2 = _w_fq_cls()
        self.fq_w3 = _w_fq_cls()

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
        x = self.fq_input(x)

        # MEM activations: 32 × 4 = 128 B @ FP32 / 32 B @ INT8
        w1 = self.fq_w1(self.fc1.weight)
        h1 = torch.relu(F.linear(x, w1, self.fc1.bias))
        h1 = self.fq_h1(self.drop1(h1))

        # MEM activations: 16 × 4 = 64 B @ FP32 / 16 B @ INT8
        w2 = self.fq_w2(self.fc2.weight)
        h2 = torch.relu(F.linear(h1, w2, self.fc2.bias))
        h2 = self.fq_h2(self.drop2(h2))

        # MEM activations: 1 × 4 = 4 B @ FP32 / 1 B @ INT8
        w3 = self.fq_w3(self.fc3.weight)
        out = torch.sigmoid(F.linear(h2, w3, self.fc3.bias))
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
            Coefficient de régularisation λ.

        Returns
        -------
        torch.Tensor (scalaire)
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
                ewc_reg += (f * (param - ts) ** 2).sum()

        return bce + (ewc_lambda / 2.0) * ewc_reg

    # ------------------------------------------------------------------
    # Snapshot θ*
    # ------------------------------------------------------------------

    def get_theta_star(self) -> dict[str, torch.Tensor]:
        """Snapshot détaché des poids quantifiés (même interface que EWCMlpClassifier)."""
        return {name: param.detach().clone() for name, param in self.named_parameters()}

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_ram_bytes(self, dtype: str = "int8") -> int:
        """
        Estime la RAM des poids seuls (sans état EWC).

        Parameters
        ----------
        dtype : "fp32" (4 B) ou "int8" (1 B). Default "int8" pour ce modèle.
        """
        bytes_per_param = {"fp32": 4, "int8": 1}.get(dtype, 1)
        return self.count_parameters() * bytes_per_param

    def __repr__(self) -> str:
        n = self.count_parameters()
        ram = self.estimate_ram_bytes("int8")
        return (
            f"EWCMlpInt8Classifier("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dims}, "
            f"params={n:,}, "
            f"RAM≈{ram / 1024:.1f}Ko INT8)"
        )
