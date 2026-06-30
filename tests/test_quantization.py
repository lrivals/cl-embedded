"""Tests unitaires — src/utils/quantization.py (S4-01)."""

import torch

from src.utils.quantization import (
    compute_scale_zero_point,
    dequantize_uint8,
    quantize_buffer,
    quantize_uint8,
)


def test_scale_zero_point_range():
    """scale > 0, zero_point dans [0, 255]."""
    x = torch.randn(100)
    scale, zp = compute_scale_zero_point(x)
    assert scale > 0
    assert 0 <= zp <= 255


def test_quantize_dtype():
    """quantize_uint8 retourne dtype uint8."""
    x = torch.randn(16, 8)
    scale, zp = compute_scale_zero_point(x)
    x_q = quantize_uint8(x, scale, zp)
    assert x_q.dtype == torch.uint8


def test_roundtrip_error_bounded():
    """Erreur de reconstruction ≤ scale (demi-intervalle de quantification)."""
    x = torch.randn(64, 8)
    scale, zp = compute_scale_zero_point(x)
    x_q = quantize_uint8(x, scale, zp)
    x_deq = dequantize_uint8(x_q, scale, zp)
    max_err = (x - x_deq).abs().max().item()
    assert max_err <= scale, f"Erreur {max_err:.4f} > scale {scale:.4f}"


def test_uint8_memory_ratio():
    """Buffer UINT8 occupe 4× moins que FP32."""
    x = torch.randn(32, 8)
    scale, zp = compute_scale_zero_point(x)
    x_q = quantize_uint8(x, scale, zp)
    fp32_bytes = x.numel() * 4
    uint8_bytes = x_q.numel() * 1
    assert uint8_bytes * 4 == fp32_bytes


def test_quantize_buffer_shape():
    """quantize_buffer concatène N activations en un tenseur [N, embed_dim]."""
    embed_dim = 9
    activations = [torch.randn(embed_dim) for _ in range(20)]
    buf, scale, zp = quantize_buffer(activations)
    assert buf.shape == (20, embed_dim)
    assert buf.dtype == torch.uint8
    assert scale > 0
    assert 0 <= zp <= 255
