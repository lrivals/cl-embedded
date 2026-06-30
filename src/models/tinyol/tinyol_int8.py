"""
tinyol_int8.py — TinyOL à stockage INT8 (poids) + UINT8 (activations) (Sprint 28, S2804).

Étend le pattern UINT8 du Sprint 4 (``quantize_buffer`` dans oto_head.py) à l'autoencoder
complet et à la tête OtO. Approche fake-quantization identique à ewc_mlp_int8.py : les poids
sont calibrés et stockés en INT8 (scale/zero_point par tenseur), le forward dequantifie en
FP32 pour le calcul, et les activations intermédiaires de l'encodeur transitent en UINT8
(asymétrique, range [0, 255] après ReLU).

Architecture réelle enveloppée (cf. autoencoder.py / oto_head.py — les chiffres de la spec
S2804 « 9→32→16→9 » sont indicatifs) :
    Encodeur : Linear(25→32) → Linear(32→16) → Linear(16→8)
    Décodeur : Linear(8→16)  → Linear(16→32) → Linear(32→25)
    OtOHead  : Linear(9→1)   (entrée = embedding 8D + MSE scalaire)

Réutilise src/utils/quantization.py (compute_scale_zero_point, quantize_uint8,
dequantize_uint8).

Référence : Ren2021TinyOL, ewc_mlp_int8.py, docs/sprints/sprint_28/S2804_tinyol_int8_python.md
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.models.tinyol.autoencoder import TinyOLAutoencoder
from src.models.tinyol.oto_head import OtOHead
from src.utils.quantization import (
    compute_scale_zero_point,
    dequantize_uint8,
    quantize_uint8,
)


def _quantize_weight_fake(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantization UINT8 par tenseur d'un poids (calcul en FP32).

    Stockage réel MCU = INT8 (1 B/élément) ; ici on simule la perte de précision en
    quantifiant puis dequantifiant. Cohérent avec ewc_mlp_int8.py (fake-quant).
    """
    scale, zp = compute_scale_zero_point(w)
    return dequantize_uint8(quantize_uint8(w, scale, zp), scale, zp)


class TinyOLAutoencoderInt8:
    """Autoencoder TinyOL à poids INT8 (fake-quant) et buffers d'activation UINT8.

    Parameters
    ----------
    autoencoder : TinyOLAutoencoder
        Backbone pré-entraîné (FP32) à quantifier.

    Notes
    -----
    Le forward INT8 ne modifie pas les poids FP32 d'origine : il applique une
    fake-quantization à la volée. La calibration des activations (UINT8) doit être
    faite via ``calibrate_int8`` sur un jeu représentatif avant ``forward_int8``.
    """

    _ENC_LAYERS = ("enc1", "enc2", "enc3")
    _DEC_LAYERS = ("dec1", "dec2", "dec3")

    def __init__(self, autoencoder: TinyOLAutoencoder) -> None:
        self.ae = autoencoder
        self.ae.eval()
        # Scales/zero_points des activations encodeur (UINT8) — remplis par calibrate_int8.
        self._act_scales: dict[str, float] = {}
        self._act_zero_points: dict[str, int] = {}
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_int8(self, X_calib: np.ndarray) -> None:
        """Calibre les scales/zero_points UINT8 des activations encodeur.

        Les poids n'ont pas besoin de calibration persistante (fake-quant par-tenseur
        au forward) ; seules les activations intermédiaires (post-ReLU enc1/enc2) sont
        calibrées sur ``X_calib`` pour fixer leur range [0, 255].

        Parameters
        ----------
        X_calib : np.ndarray [N, input_dim], float32
            Échantillons représentatifs (typiquement le train set de la tâche 1).
        """
        x = torch.as_tensor(np.asarray(X_calib, dtype=np.float32))
        with torch.no_grad():
            h1 = torch.relu(self.ae.enc1(x))  # MEM: N×32×4 B @ FP32
            h2 = torch.relu(self.ae.enc2(h1))  # MEM: N×16×4 B @ FP32
        self._act_scales["enc1"], self._act_zero_points["enc1"] = compute_scale_zero_point(h1)
        self._act_scales["enc2"], self._act_zero_points["enc2"] = compute_scale_zero_point(h2)
        self._calibrated = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fake_quant_act(self, h: torch.Tensor, name: str) -> torch.Tensor:
        """Applique la fake-quant UINT8 sur une activation calibrée (sinon identité)."""
        if name not in self._act_scales:
            return h
        s, zp = self._act_scales[name], self._act_zero_points[name]
        return dequantize_uint8(quantize_uint8(h, s, zp), s, zp)

    def forward_int8(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward encodeur + décodeur avec poids INT8 (dequant FP32) et activations UINT8.

        Parameters
        ----------
        x : np.ndarray [input_dim] ou [batch, input_dim], float32

        Returns
        -------
        encoded : np.ndarray [batch, 8], float32
        reconstructed : np.ndarray [batch, input_dim], float32
        """
        xt = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=np.float32)))
        with torch.no_grad():
            # --- Encodeur (poids INT8 fake-quant, activations UINT8) ---
            # MEM activations: 32 × 1 B @ UINT8 (vs 128 B @ FP32)
            h = torch.relu(
                F.linear(xt, _quantize_weight_fake(self.ae.enc1.weight), self.ae.enc1.bias)
            )
            h = self._fake_quant_act(h, "enc1")
            # MEM activations: 16 × 1 B @ UINT8 (vs 64 B @ FP32)
            h = torch.relu(
                F.linear(h, _quantize_weight_fake(self.ae.enc2.weight), self.ae.enc2.bias)
            )
            h = self._fake_quant_act(h, "enc2")
            # MEM activations: 8 × 1 B @ UINT8 (embedding latent)
            z = F.linear(h, _quantize_weight_fake(self.ae.enc3.weight), self.ae.enc3.bias)

            # --- Décodeur (poids INT8 fake-quant) ---
            d = torch.relu(
                F.linear(z, _quantize_weight_fake(self.ae.dec1.weight), self.ae.dec1.bias)
            )
            d = torch.relu(
                F.linear(d, _quantize_weight_fake(self.ae.dec2.weight), self.ae.dec2.bias)
            )
            x_hat = F.linear(d, _quantize_weight_fake(self.ae.dec3.weight), self.ae.dec3.bias)

        return z.numpy(), x_hat.numpy()

    def reconstruction_error_int8(self, x: np.ndarray) -> float:
        """Erreur MSE de reconstruction via le forward INT8."""
        xt = np.atleast_2d(np.asarray(x, dtype=np.float32))
        _, x_hat = self.forward_int8(xt)
        return float(np.mean((x_hat - xt) ** 2))

    # ------------------------------------------------------------------
    # Empreinte mémoire
    # ------------------------------------------------------------------

    def get_memory_footprint_int8(self) -> dict[str, int]:
        """Empreinte INT8 : poids (1 B/élément) + buffers d'activation UINT8.

        Returns
        -------
        dict avec encoder_weights_bytes, decoder_weights_bytes, weights_bytes,
        activation_buffer_bytes, total_bytes.
        """
        enc_params = sum(
            getattr(self.ae, name).weight.numel() + getattr(self.ae, name).bias.numel()
            for name in self._ENC_LAYERS
        )
        dec_params = sum(
            getattr(self.ae, name).weight.numel() + getattr(self.ae, name).bias.numel()
            for name in self._DEC_LAYERS
        )
        # Buffers d'activation UINT8 calibrés (enc1=32, enc2=16) → 1 B/élément.
        act_bytes = 32 + 16
        weights_bytes = (enc_params + dec_params) * 1  # int8
        return {
            "encoder_weights_bytes": int(enc_params),
            "decoder_weights_bytes": int(dec_params),
            "weights_bytes": int(weights_bytes),
            "activation_buffer_bytes": int(act_bytes),
            "total_bytes": int(weights_bytes + act_bytes),
        }


class OtOHeadInt8:
    """Tête One-to-One à poids INT8 (fake-quant), SGD en ligne.

    Parameters
    ----------
    oto_head : OtOHead
        Tête FP32 (Linear(input_dim→1)) à quantifier.
    learning_rate : float
        Pas de la mise à jour SGD (défaut : 1e-2).

    Notes
    -----
    L'update suit le pattern fake-quant de ewc_mlp_int8.py : on calcule le gradient en
    FP32 sur les poids dequantifiés, on met à jour les poids FP32 maîtres, puis le
    forward re-quantifie à la volée. Les poids INT8 sont la *représentation de stockage*,
    le FP32 reste la copie maîtresse pour l'apprentissage (master weights).
    """

    def __init__(self, oto_head: OtOHead, learning_rate: float = 1e-2) -> None:
        self.head = oto_head
        self.lr = float(learning_rate)

    def _quant_weight(self) -> torch.Tensor:
        return _quantize_weight_fake(self.head.fc.weight)

    def predict_int8(self, encoded: np.ndarray) -> float:
        """Forward avec poids INT8 (dequant FP32). Retourne la probabilité ∈ [0, 1]."""
        xt = torch.as_tensor(np.asarray(encoded, dtype=np.float32)).reshape(-1)
        with torch.no_grad():
            logit = F.linear(xt, self._quant_weight(), self.head.fc.bias)
            return float(torch.sigmoid(logit).item())

    def update_int8(self, encoded: np.ndarray, y: int) -> float:
        """1 pas de SGD avec poids INT8 fake-quant. Retourne la loss BCE.

        Gradient calculé sur poids dequantifiés (straight-through), poids maîtres FP32
        mis à jour, requantification implicite au forward suivant.
        """
        xt = torch.as_tensor(np.asarray(encoded, dtype=np.float32)).reshape(1, -1)
        yt = torch.as_tensor([[float(y)]])

        w_fq = self._quant_weight().detach().clone().requires_grad_(True)
        b = self.head.fc.bias.detach().clone().requires_grad_(True)

        pred = torch.sigmoid(F.linear(xt, w_fq, b))
        loss = F.binary_cross_entropy(pred, yt)
        loss.backward()

        with torch.no_grad():
            self.head.fc.weight -= self.lr * w_fq.grad
            self.head.fc.bias -= self.lr * b.grad
        return float(loss.item())

    def get_memory_footprint_int8(self) -> dict[str, int]:
        """Empreinte INT8 des poids de la tête (1 B/élément)."""
        n = sum(p.numel() for p in self.head.parameters())
        return {"weights_bytes": int(n), "total_bytes": int(n)}
