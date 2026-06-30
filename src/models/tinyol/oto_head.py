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

from src.utils.quantization import dequantize_uint8, quantize_buffer


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

        # --- Buffer UINT8 (S4-02) ---
        oto_cfg = config.get("oto_head", {})
        self.use_uint8_buffer: bool = oto_cfg.get("use_uint8_buffer", False)
        self.buffer_size: int = oto_cfg.get("buffer_size", 50)
        self.buffer_replay_ratio: float = oto_cfg.get("buffer_replay_ratio", 0.2)

        self._buffer_fp32: list[torch.Tensor] = []
        self._buffer_labels_raw: list[float] = []  # labels parallèles à _buffer_fp32
        self._buffer_uint8: torch.Tensor | None = None  # MEM: buffer_size×embed_dim×1 B @ UINT8
        self._buffer_labels: torch.Tensor | None = None  # MEM: buffer_size×4 B @ FP32
        self._buffer_scale: float = 1.0
        self._buffer_zero_point: int = 0
        self._step_counter: int = 0

    def _add_to_buffer(self, embedding: torch.Tensor, label: float) -> None:
        """
        Ajoute un embedding au buffer FIFO. Si use_uint8_buffer, requantifie le buffer complet.

        Stratégie FIFO — l'embedding le plus ancien est supprimé si buffer_size atteint.
        """
        self._buffer_fp32.append(embedding.detach())
        self._buffer_labels_raw.append(label)
        if len(self._buffer_fp32) > self.buffer_size:
            self._buffer_fp32.pop(0)
            self._buffer_labels_raw.pop(0)

        if self.use_uint8_buffer and len(self._buffer_fp32) >= 2:
            buf_uint8, scale, zp = quantize_buffer(self._buffer_fp32)
            self._buffer_uint8 = buf_uint8  # MEM: len×embed_dim×1 B @ UINT8
            self._buffer_labels = torch.tensor(
                self._buffer_labels_raw, dtype=torch.float32
            )  # MEM: len×4 B @ FP32
            self._buffer_scale = scale
            self._buffer_zero_point = zp

    def _replay_from_buffer(self) -> None:
        """
        Mini-replay : tire 1 embedding UINT8 du buffer, reconstruit en FP32,
        et effectue un step SGD sur la tête OtO.
        """
        if self._buffer_uint8 is None or len(self._buffer_uint8) == 0:
            return
        idx = int(torch.randint(0, len(self._buffer_uint8), (1,)).item())
        emb_uint8 = self._buffer_uint8[idx]  # MEM: embed_dim×1 B @ UINT8
        emb_fp32 = dequantize_uint8(  # MEM: embed_dim×4 B @ FP32
            emb_uint8.unsqueeze(0),
            self._buffer_scale,
            self._buffer_zero_point,
        ).squeeze(0)
        label = self._buffer_labels[idx]
        self.optimizer.zero_grad()
        pred = self.oto_head(emb_fp32)
        loss = F.binary_cross_entropy(pred.squeeze(), label)
        loss.backward()
        self.optimizer.step()

    def get_buffer_ram_bytes(self) -> dict[str, int | float]:
        """
        Retourne l'empreinte RAM du buffer (utile pour profile_memory.py).

        Returns
        -------
        dict avec clés : "uint8_bytes", "fp32_equivalent_bytes", "compression_ratio"
        """
        if self._buffer_uint8 is None:
            return {"uint8_bytes": 0, "fp32_equivalent_bytes": 0, "compression_ratio": 1}
        n = self._buffer_uint8.numel()
        return {
            "uint8_bytes": n,  # MEM: N×embed_dim×1 B @ UINT8
            "fp32_equivalent_bytes": n * 4,  # MEM: N×embed_dim×4 B @ FP32
            "compression_ratio": 4.0,
        }

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

        # 4. Buffer UINT8 + replay conditionnel (S4-02)
        # On stocke l'entrée OtO complète (z + MSE) pour pouvoir la rejouer directement.
        self._add_to_buffer(oto_input.detach(), float(y.item() if hasattr(y, "item") else y))

        self._step_counter += 1
        replay_every = max(1, int(1 / self.buffer_replay_ratio))
        if self.use_uint8_buffer and self._step_counter % replay_every == 0:
            self._replay_from_buffer()

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
