"""
src/models/tinyol/autoencoder.py — Backbone TinyOL : autoencoder MLP offline.

Rôle : pré-entraîner un encodeur 25→8 sur données normales (MSE),
puis le geler pour la phase continual learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constantes de configuration
# Conformes à configs/tinyol_config.yaml — ne pas modifier ici.
# ---------------------------------------------------------------------------

# Budget mémoire encodeur (référence tinyol_spec.md §2.1) :
# Linear(25→32) : 25×32 + 32 = 832 params  → MEM: 832×4 = 3 328 B @ FP32 / 832 B @ INT8
# Linear(32→16) : 32×16 + 16 = 528 params  → MEM: 528×4 = 2 112 B @ FP32 / 528 B @ INT8
# Linear(16→8)  : 16×8  +  8 = 136 params  → MEM: 136×4 =   544 B @ FP32 / 136 B @ INT8
# TOTAL encodeur : 1 496 params → ~5 828 B ≈ 5,8 Ko @ FP32 / ~1,5 Ko @ INT8

INPUT_DIM: int = 25
ENCODER_DIMS: tuple[int, ...] = (32, 16, 8)
DECODER_DIMS: tuple[int, ...] = (16, 32, 25)


class TinyOLAutoencoder(nn.Module):
    """
    Autoencoder MLP pour le pré-entraînement du backbone TinyOL.

    Séparation backbone/tête conforme à Ren2021TinyOL :
    - L'encodeur est pré-entraîné offline sur données normales, puis gelé.
    - Le décodeur est utilisé uniquement au pré-entraînement (perte MSE).
    - En phase CL, seul l'encodeur est utilisé (via encode()) ; le décodeur
      est ignoré.

    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée (défaut : 25).
    encoder_dims : tuple[int, ...]
        Dimensions des couches de l'encodeur (défaut : (32, 16, 8)).
    decoder_dims : tuple[int, ...]
        Dimensions des couches du décodeur (défaut : (16, 32, 25)).

    Notes
    -----
    Budget mémoire (conforme tinyol_spec.md §2.3) :
    - Encodeur (Flash après gel) : 1 496 params → ~5,8 Ko @ FP32 / ~1,5 Ko @ INT8
    - Décodeur (pré-entraînement seulement) : 1 513 params → ~6,5 Ko @ FP32
    - Activations forward : ~512 B @ FP32 / ~128 B @ INT8 (batch=1, MCU)
    - TOTAL RAM dynamique (inférence seule) : < 600 B (cible 64 Ko — très confortable)

    Références
    ----------
    Ren2021TinyOL, docs/models/tinyol_spec.md §2
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        encoder_dims: tuple[int, ...] = ENCODER_DIMS,
        decoder_dims: tuple[int, ...] = DECODER_DIMS,
    ) -> None:
        super().__init__()

        # --- Encodeur ---
        self.enc1 = nn.Linear(input_dim, encoder_dims[0])       # MEM: 3 328 B @ FP32 / 832 B @ INT8
        self.enc2 = nn.Linear(encoder_dims[0], encoder_dims[1]) # MEM: 2 112 B @ FP32 / 528 B @ INT8
        self.enc3 = nn.Linear(encoder_dims[1], encoder_dims[2]) # MEM:   544 B @ FP32 / 136 B @ INT8

        # --- Décodeur (utilisé au pré-entraînement uniquement) ---
        self.dec1 = nn.Linear(encoder_dims[2], decoder_dims[0])
        self.dec2 = nn.Linear(decoder_dims[0], decoder_dims[1])
        self.dec3 = nn.Linear(decoder_dims[1], decoder_dims[2])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait l'embedding latent z depuis les features d'entrée.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim], dtype float32

        Returns
        -------
        z : torch.Tensor, shape [batch, 8], dtype float32
            Embedding latent (8D).
            # MEM: batch × 8 × 4 B @ FP32 / batch × 8 B @ INT8
        """
        z = F.relu(self.enc1(x))  # MEM: batch × 32 × 4 B = 128 B @ FP32 (batch=1)
        z = F.relu(self.enc2(z))  # MEM: batch × 16 × 4 B = 64 B @ FP32 (batch=1)
        z = self.enc3(z)          # MEM: batch × 8 × 4 B = 32 B @ FP32 (batch=1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruit les features depuis l'embedding latent.

        Parameters
        ----------
        z : torch.Tensor, shape [batch, 8], dtype float32

        Returns
        -------
        x_hat : torch.Tensor, shape [batch, input_dim], dtype float32
            Reconstruction des features d'entrée.
        """
        x_hat = F.relu(self.dec1(z))
        x_hat = F.relu(self.dec2(x_hat))
        x_hat = self.dec3(x_hat)  # Pas d'activation finale (MSE sur features normalisées)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass complet (encodeur + décodeur).

        Utilisé pendant le pré-entraînement batch uniquement.
        En phase CL, utiliser encode() directement.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim], dtype float32

        Returns
        -------
        z : torch.Tensor, shape [batch, 8]
            Embedding latent.
        x_hat : torch.Tensor, shape [batch, input_dim]
            Reconstruction pour le calcul de la perte MSE.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la perte de reconstruction MSE.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim]
            Features d'entrée (normalisées).
        x_hat : torch.Tensor, shape [batch, input_dim]
            Reconstruction de l'encodeur-décodeur.

        Returns
        -------
        torch.Tensor (scalaire) — perte MSE différentiable.

        Notes
        -----
        Conforme à tinyol_spec.md §5 : PRE_TRAIN_LOSS = "mse".
        """
        return F.mse_loss(x_hat, x)

    def freeze_encoder(self) -> None:
        """
        Gèle les paramètres de l'encodeur (requires_grad = False).

        À appeler APRÈS le pré-entraînement, AVANT la phase CL.
        Après le gel, seule la tête OtO (oto_head.py) est entraînable.

        Analogie MCU : l'encodeur est stocké en Flash (lecture seule),
        la tête OtO est en RAM (lecture/écriture).

        Références
        ----------
        Ren2021TinyOL, tinyol_spec.md §1
        """
        for param in [
            *self.enc1.parameters(),
            *self.enc2.parameters(),
            *self.enc3.parameters(),
        ]:
            param.requires_grad = False

    def n_encoder_params(self) -> int:
        """Retourne le nombre de paramètres de l'encodeur (attendu : 1 496)."""
        return sum(
            p.numel()
            for p in [
                *self.enc1.parameters(),
                *self.enc2.parameters(),
                *self.enc3.parameters(),
            ]
        )
