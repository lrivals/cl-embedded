"""
Tête OtO (One-to-One) pour TinyOL.

Architecture : Linear(9→1) + Sigmoid
Paramètres   : 10 (9 poids + 1 biais) → 40 octets @ FP32
MCU          : SGD uniquement, pas de momentum, pas d'Adam

Référence : Ren2021TinyOL, tinyol_spec.md §2.2 et §6
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class OtOHead(nn.Module):
    """
    Tête One-to-One (OtO) entraînable du modèle TinyOL.

    Parameters
    ----------
    input_dim : int
        Dimension de l'entrée = embed_dim + 1 (MSE scalaire).
        Valeur attendue : 9 (8D embedding + 1D MSE).

    Notes
    -----
    Conformité MCU :
    - Pas d'Adam — état (m, v) trop coûteux en RAM.
    - ReLU absent (sortie = Sigmoid pour probabilité binaire).
    - Taille fixe : 10 paramètres = 40 octets @ FP32.
    """

    def __init__(self, input_dim: int = 9) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)  # MEM: 40 B @ FP32 / 10 B @ INT8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [input_dim] ou [batch, input_dim]

        Returns
        -------
        torch.Tensor, shape [1] ou [batch, 1]
            Probabilité de panne dans [0, 1].
        """
        return torch.sigmoid(self.fc(x))  # MEM: 4 B @ FP32 (scalaire)

    def n_params(self) -> int:
        """Retourne le nombre de paramètres (attendu : 10)."""
        return sum(p.numel() for p in self.parameters())


class TinyOLOnlineTrainer:
    """
    Boucle d'apprentissage online pour TinyOL.

    Encapsule le backbone gelé (TinyOLAutoencoder) et la tête OtO,
    et expose une méthode `update` pour la mise à jour échantillon par échantillon.

    Parameters
    ----------
    autoencoder : TinyOLAutoencoder
        Backbone pré-entraîné et gelé.
    oto_head : OtOHead
        Tête OtO entraînable.
    config : dict
        Configuration YAML complète (section ``oto_head`` utilisée).

    Notes
    -----
    Conformité MCU — tinyol_spec.md §6 :
    - Optimiseur : SGD pur (pas de momentum → oto_head.momentum=0.0)
    - Loss       : Binary Cross-Entropy
    - Fréquence  : 1 update par échantillon
    - Gradient   : limité à `oto_head` uniquement (backbone gelé)
    """

    def __init__(
        self,
        autoencoder: "TinyOLAutoencoder",  # noqa: F821
        oto_head: OtOHead,
        config: dict,
    ) -> None:
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)  # gel complet du backbone

        self.oto_head = oto_head
        self.optimizer = torch.optim.SGD(
            self.oto_head.parameters(),
            lr=config["oto_head"]["learning_rate"],
            momentum=config["oto_head"]["momentum"],  # doit être 0.0
        )

    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Effectue un pas d'apprentissage online sur un seul échantillon.

        Parameters
        ----------
        x : torch.Tensor, shape [25]
            Features normalisées d'une fenêtre.
        y : torch.Tensor, shape [1] ou scalaire
            Label binaire (0=normal, 1=panne).

        Returns
        -------
        float
            Valeur de la loss BCE pour cet échantillon.
        """
        # 1. Forward backbone gelé — pas de gradient
        with torch.no_grad():
            z, x_hat = self.autoencoder(x.unsqueeze(0))  # MEM: 32 B @ FP32 (z)
            mse = F.mse_loss(x_hat, x.unsqueeze(0)).unsqueeze(0)  # MEM: 4 B @ FP32

        # 2. Construction de l'entrée OtO : [embed_dim + 1] = [9]
        oto_input = torch.cat([z.squeeze(0), mse])  # MEM: 36 B @ FP32

        # 3. Forward + backward tête OtO
        y_hat = self.oto_head(oto_input)  # MEM: 4 B @ FP32
        loss = F.binary_cross_entropy(y_hat.squeeze(), y.float().squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Discard x, y — pas de stockage (online learning)
        return loss.item()

    def predict(self, x: torch.Tensor) -> tuple[float, float]:
        """
        Prédit sans mise à jour.

        Parameters
        ----------
        x : torch.Tensor, shape [25]
            Features normalisées d'une fenêtre.

        Returns
        -------
        tuple[float, float]
            (probabilité_panne, mse_reconstruction)
        """
        with torch.no_grad():
            z, x_hat = self.autoencoder(x.unsqueeze(0))
            mse = F.mse_loss(x_hat, x.unsqueeze(0))
            oto_input = torch.cat([z.squeeze(0), mse.unsqueeze(0)])  # MEM: 36 B @ FP32
            y_hat = self.oto_head(oto_input)  # MEM: 4 B @ FP32
        return y_hat.item(), mse.item()


if __name__ == "__main__":
    import tracemalloc

    from src.models.tinyol.autoencoder import TinyOLAutoencoder

    config_dummy = {"oto_head": {"learning_rate": 1e-2, "momentum": 0.0}}

    autoencoder = TinyOLAutoencoder()
    oto_head = OtOHead(input_dim=9)
    trainer = TinyOLOnlineTrainer(autoencoder, oto_head, config_dummy)

    x_dummy = torch.randn(25)
    y_dummy = torch.tensor(0.0)

    tracemalloc.start()
    _ = trainer.update(x_dummy, y_dummy)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"OtO params      : {oto_head.n_params()} (attendu : 10)")
    print(f"RAM peak update : {peak} B (cible : < 100 B hors PyTorch overhead)")
    # FIXME(gap2) : overhead PyTorch non représentatif de la RAM MCU réelle
    # → utiliser memory_profiler.py pour la mesure officielle dans S3-06
