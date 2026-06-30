"""
Primitives de quantification UINT8 affine pour TinyOL et modèles embarqués.

Algorithme par-tenseur (global min/max) — compatible CMSIS-NN `arm_q7_to_float`
(Cortex-M4, NUCLEO-F439ZI). Sans `torch.quantization` (non portable MCU).

Référence : CLAUDE.md §Contraintes hardware, S4-01
"""

from __future__ import annotations

import numpy as np
import torch


def compute_scale_zero_point(
    x: torch.Tensor | np.ndarray,
    n_bits: int = 8,
) -> tuple[float, int]:
    """
    Calcule scale et zero_point pour quantification affine UINT8.

    Formule :
        scale = (x_max - x_min) / (2^n_bits - 1)
        zero_point = clamp(round(-x_min / scale), 0, 255)

    Parameters
    ----------
    x : Tensor ou ndarray
        Tenseur à quantifier (activations ou poids).
    n_bits : int
        Résolution (défaut : 8 pour UINT8).

    Returns
    -------
    scale : float — facteur d'échelle
    zero_point : int — décalage entier (0–255)

    Notes
    -----
    Quantification par-tenseur (global min/max) — pas par-canal.
    Compatible CMSIS-NN `arm_q7_to_float` (Cortex-M4).
    """
    if isinstance(x, np.ndarray):
        x_min = float(x.min())
        x_max = float(x.max())
    else:
        x_min = float(x.min().item())
        x_max = float(x.max().item())

    n_levels = (1 << n_bits) - 1  # 255 pour UINT8
    x_range = x_max - x_min

    # Cas dégénéré : tenseur constant → scale arbitraire non nul
    scale = x_range / n_levels if x_range > 0 else 1.0
    zero_point = int(round(-x_min / scale))
    zero_point = max(0, min(n_levels, zero_point))  # clamp [0, 255]

    return scale, zero_point


def quantize_uint8(
    x: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """
    Quantifie un tenseur FP32 en UINT8.

    x_q = clamp(round(x / scale) + zero_point, 0, 255)  # MEM: 1 B/élément @ UINT8
    (vs 4 B/élément @ FP32 → facteur 4×)

    Parameters
    ----------
    x : Tensor [*shape], dtype float32
    scale : float
    zero_point : int

    Returns
    -------
    Tensor [*shape], dtype uint8
    """
    x_shifted = torch.round(x / scale) + zero_point  # MEM: shape×4 B @ FP32 (temporaire)
    x_clamped = torch.clamp(x_shifted, 0, 255)  # MEM: shape×4 B @ FP32 (temporaire)
    return x_clamped.to(torch.uint8)  # MEM: shape×1 B @ UINT8


def dequantize_uint8(
    x_q: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """
    Reconvertit UINT8 → FP32 (reconstruction approximative).

    x_deq = (x_q.float() - zero_point) * scale  # MEM: 4 B/élément @ FP32

    Parameters
    ----------
    x_q : Tensor [*shape], dtype uint8
    scale : float
    zero_point : int

    Returns
    -------
    Tensor [*shape], dtype float32
    """
    return (x_q.float() - zero_point) * scale  # MEM: shape×4 B @ FP32


def quantize_buffer(
    activations: list[torch.Tensor],
    n_bits: int = 8,
) -> tuple[torch.Tensor, float, int]:
    """
    Quantifie une liste d'activations (buffer de replay TinyOL).

    Calcule scale/zero_point sur l'ensemble du buffer (min/max global),
    puis quantifie chaque tenseur et les concatène.

    Parameters
    ----------
    activations : list[Tensor]
        Embeddings du backbone TinyOL, shape [(embed_dim,), ...]
    n_bits : int

    Returns
    -------
    buffer_uint8 : Tensor [N, embed_dim], dtype uint8
    scale : float
    zero_point : int

    Notes
    -----
    RAM buffer FP32 = N × embed_dim × 4 B
    RAM buffer UINT8 = N × embed_dim × 1 B  ← 4× moins
    """
    stacked = torch.stack(activations, dim=0)  # MEM: N×embed_dim×4 B @ FP32
    scale, zero_point = compute_scale_zero_point(stacked, n_bits=n_bits)
    buffer_uint8 = quantize_uint8(stacked, scale, zero_point)  # MEM: N×embed_dim×1 B @ UINT8
    return buffer_uint8, scale, zero_point
