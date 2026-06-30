# S4-01 — Implémenter `quantization.py` (UINT8 encoder/decoder)

| Champ | Valeur |
|-------|--------|
| **ID** | S4-01 |
| **Sprint** | Sprint 4 — Semaine 4 (6–13 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S3-05 (`oto_head.py` disponible — embeddings à quantifier) |
| **Fichiers cibles** | `src/utils/quantization.py`, `src/utils/__init__.py` |
| **Statut** | ✅ Terminé |

---

## Objectif

Implémenter les primitives de quantification UINT8 (affine, par-tenseur) utilisées par le buffer d'activations de TinyOL (S4-02) et potentiellement réutilisables pour EWC et HDC en Phase 2.

Les fonctions doivent être :
- **Sans dépendance lourde** : NumPy uniquement (pas de `torch.quantization` — non portable MCU)
- **Symétrique avec PyTorch** : accepter `torch.Tensor` ou `np.ndarray`, retourner le même type
- **Documentées avec `# MEM:`** : chaque buffer intermédiaire annoté
- **Exportables en C** : l'algorithme doit pouvoir être transcrit directement en C embarqué (pas de lookup table dynamique)

**Critère de succès** : `pytest tests/test_quantization.py -v` passe (5 tests minimum).

---

## Sous-tâches

### 1. API publique

```python
# src/utils/quantization.py

def compute_scale_zero_point(
    x: torch.Tensor | np.ndarray,
    n_bits: int = 8,
) -> tuple[float, int]:
    """
    Calcule scale et zero_point pour quantification affine UINT8.

    Formule :
        scale = (x_max - x_min) / (2^n_bits - 1)
        zero_point = round(-x_min / scale)  ← clampé dans [0, 255]

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
```

### 2. Export dans `src/utils/__init__.py`

```python
from .quantization import compute_scale_zero_point, quantize_uint8, dequantize_uint8, quantize_buffer

__all__ = [
    "compute_scale_zero_point",
    "quantize_uint8",
    "dequantize_uint8",
    "quantize_buffer",
]
```

### 3. Tests unitaires — `tests/test_quantization.py`

```python
import torch
import numpy as np
from src.utils.quantization import (
    compute_scale_zero_point, quantize_uint8, dequantize_uint8, quantize_buffer
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
    """Erreur de reconstruction ≤ scale/2 (erreur de quantification maximale théorique)."""
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
```

---

## Critères d'acceptation

- [ ] `from src.utils.quantization import quantize_uint8, dequantize_uint8` — import sans erreur
- [ ] `quantize_uint8(x, s, zp).dtype == torch.uint8` pour tout tenseur FP32
- [ ] Erreur de reconstruction ≤ `scale` (demi-intervalle de quantification)
- [ ] `quantize_buffer([...]).shape == (N, embed_dim)` pour N embeddings
- [ ] Pas de dépendance à `torch.quantization` (non portable MCU)
- [ ] Annotations `# MEM:` sur chaque opération qui alloue de la mémoire
- [ ] `ruff check src/utils/quantization.py` et `black --check` passent
- [ ] `pytest tests/test_quantization.py -v` — tous les tests passent

---

## Interface attendue par S4-02

```python
# Dans oto_head.py (S4-02)
from src.utils.quantization import quantize_buffer, dequantize_uint8

# Stocker les embeddings du backbone en UINT8
buffer_uint8, scale, zp = quantize_buffer(list_of_embeddings)
# Reconstruire pour le forward pass
embeddings_fp32 = dequantize_uint8(buffer_uint8, scale, zp)
```

---

## Questions ouvertes

- `TODO(dorra)` : La quantification par-tenseur est-elle suffisante ou faut-il per-channel pour limiter la perte de précision sur les embeddings TinyOL ?
- `FIXME(gap3)` : Mesurer le delta AA (FP32 vs UINT8 buffer) dans exp_004 — si delta > 0.005, envisager per-channel ou n_bits=16.
- `TODO(arnaud)` : Confirmer que CMSIS-NN `arm_q7_to_float` est disponible sur NUCLEO-F439ZI (Cortex-M4).
