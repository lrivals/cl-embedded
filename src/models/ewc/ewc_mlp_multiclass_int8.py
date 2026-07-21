"""
ewc_mlp_multiclass_int8.py — EWC MLP multi-classe avec fake-quantization INT8 (QAT).

Variante QAT (fake-quant dans la boucle d'entraînement, straight-through) du head
``EWCMlpMulticlass`` (2 sorties softmax) — le head **compatible firmware** (le kernel
INT8 v2 et ``export_weights_c.py --int8-v2`` consomment un head à ``n_classes`` sorties).

Contrairement à ``EWCMlpInt8Classifier`` (binaire, 1 sortie sigmoïde), cette classe garde
l'architecture ``Linear(k→32→16→n_classes)`` du firmware. Les poids sous-jacents restent
FP32 (le fake-quant n'est appliqué qu'au forward) → ``state_dict`` avec ``fc1/fc2/fc3``
**directement exportable** via ``EWCHeadWeights.from_state_dict`` / ``export_weights_c.py``.

C'est la brique du chemin **both** board (Sprint 46, S4608) : QAT → export → kernel v2
calibré (le firmware ne fait jamais de fake-quant à l'inférence, il exécute un noyau entier).

Quantization (identique à EWCMlpInt8Classifier) :
    Weights     : per-channel symmetric INT8 (PerChannelMinMaxObserver)
    Activations : per-tensor affine INT8 (HistogramObserver)

Références : Kirkpatrick2017EWC (EWC), Ravaglia2021QLRCL (QAT/rejeu INT8),
             ewc_mlp_multiclass.py (FP32), ewc_mlp_int8.py (fake-quant binaire).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.quantization as quant
from torch import Tensor
from torch.utils.data import DataLoader


class EWCMlpMulticlassInt8(nn.Module):
    """
    MLP multi-classe EWC Online avec fake-quantization INT8 (QAT).

    Architecture (identique à EWCMlpMulticlass, compatible firmware) :
        Linear(input_dim → 32) + FakeQuant + ReLU + Dropout
        Linear(32 → 16)        + FakeQuant + ReLU + Dropout
        Linear(16 → n_classes) [logits bruts — CrossEntropyLoss applique softmax]

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée.
    n_classes : int
        Nombre de classes (board EWC : 2 = normal / faulty).
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
        n_classes: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        ewc_lambda: float = 400.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.ewc_lambda = ewc_lambda

        # --- Couches linéaires — identiques à EWCMlpMulticlass (poids FP32 sous-jacents) ---
        # MEM: Linear(5→32) = (5×32+32)×4 = 704 B @ FP32 / 176 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # MEM: Linear(32→16) = (32×16+16)×4 = 2112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # MEM: Linear(16→n_classes) = (16×n_classes+n_classes)×4 = 136 B @ FP32 / 34 B @ INT8 (n=2)
        self.fc3 = nn.Linear(hidden_dims[1], n_classes)

        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        # Fake-quantizers pour activations (per-tensor affine) — MEM: négligeable (scalaires)
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

        self._fisher: dict[str, Tensor] = {}
        self._theta_star: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Forward pass (fake-quant weights + activations, logits bruts)
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]

        Returns
        -------
        Tensor [batch_size, n_classes]
            Logits bruts (avant softmax).
        """
        x = self.fq_input(x)

        # MEM activations: 32 × 4 = 128 B @ FP32 / 32 B @ INT8
        w1 = self.fq_w1(self.fc1.weight)
        h = torch.relu(F.linear(x, w1, self.fc1.bias))
        h = self.fq_h1(self.drop1(h))

        # MEM activations: 16 × 4 = 64 B @ FP32 / 16 B @ INT8
        w2 = self.fq_w2(self.fc2.weight)
        h = torch.relu(F.linear(h, w2, self.fc2.bias))
        h = self.fq_h2(self.drop2(h))

        # MEM activations: n_classes × 4 B @ FP32 (logits)
        w3 = self.fq_w3(self.fc3.weight)
        return F.linear(h, w3, self.fc3.bias)  # shape [batch_size, n_classes]

    def predict(self, x: Tensor) -> Tensor:
        """Retourne la classe prédite (argmax des logits)."""
        with torch.no_grad():
            return self(x).argmax(dim=1)

    # ------------------------------------------------------------------
    # Régularisation EWC (interface identique à EWCMlpMulticlass)
    # ------------------------------------------------------------------

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

    def consolidate(self, data_loader: DataLoader, n_samples: int = 200) -> None:
        """Calcule la Fisher diagonale (gradient cross-entropy) et snapshote θ*."""
        self.eval()
        criterion = nn.CrossEntropyLoss()
        fisher_accum: dict[str, Tensor] = {
            n: torch.zeros_like(p) for n, p in self.named_parameters()
        }
        count = 0
        for x_batch, y_batch in data_loader:
            if count >= n_samples:
                break
            self.zero_grad()
            logits = self(x_batch)
            loss = criterion(logits, y_batch.long())
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher_accum[n] += p.grad.data.clone() ** 2
            count += len(x_batch)

        n_batches = max(1, count // data_loader.batch_size)
        self._fisher = {n: f / n_batches for n, f in fisher_accum.items()}
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters()}
        self.train()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_ram_bytes(self, dtype: str = "int8") -> int:
        """Estime la RAM des poids seuls (sans état EWC). Default int8 pour ce modèle."""
        bytes_per_param = {"fp32": 4, "int8": 1}.get(dtype, 1)
        return self.count_parameters() * bytes_per_param

    def __repr__(self) -> str:
        n = self.count_parameters()
        ram = self.estimate_ram_bytes("int8")
        return (
            f"EWCMlpMulticlassInt8("
            f"input={self.input_dim}, "
            f"n_classes={self.n_classes}, "
            f"hidden={self.hidden_dims}, "
            f"params={n:,}, "
            f"RAM≈{ram / 1024:.1f}Ko INT8)"
        )
