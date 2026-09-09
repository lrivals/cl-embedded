"""Tests S4803 — export sub-INT8 (`export_weights_c.py --ewc-subint8`).

Vérifie que :
  1. ``_pack_weights`` fait un aller-retour exact avec la sémantique de dépacking C
     (LSB-first, complément à deux INT4/INT2, binaire {−1,+1}↔{0,1}) ;
  2. la quantification de l'export réutilise EXACTEMENT les primitives émulateur S47
     (``_quant_weight_mode``) → parité par construction ;
  3. le header généré est bien gardé et non vide.

PC-only, aucune carte. Aucun chiffre golden en dur (tout dérive de l'émulateur).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXPORT = _ROOT / "scripts" / "export_weights_c.py"
_CKPT = _ROOT / "experiments" / "exp_S39_matched" / "checkpoints" / "ewc_pronostia_5feat.pt"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_weights_c", _EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unpack_reference(row: np.ndarray, i: int, pack_bits: int) -> int:
    """Réplique Python de ewc_v2_unpack_weight (contrat de parité firmware)."""
    per = 8 // pack_bits
    field = (int(row[i // per]) >> ((i % per) * pack_bits)) & ((1 << pack_bits) - 1)
    if pack_bits == 1:
        return 1 if field else -1
    shift = 8 - pack_bits
    v = (field << shift) & 0xFF          # low 8 bits
    if v >= 128:
        v -= 256                          # reinterprétation (int8_t)
    return v >> shift                     # décalage arithmétique (Python >> = arithmétique)


@pytest.mark.parametrize("pack_bits", [4, 2, 1])
def test_pack_unpack_roundtrip(pack_bits: int) -> None:
    mod = _load_export_module()
    rng = np.random.default_rng(48)
    if pack_bits == 4:
        vals = rng.integers(-7, 8, size=(6, 11))
    elif pack_bits == 2:
        vals = rng.integers(-1, 2, size=(6, 11))
    else:  # binaire : uniquement {−1, +1}
        vals = rng.choice([-1, 1], size=(6, 11))
    q = vals.astype(np.int64)
    packed = mod._pack_weights(q, pack_bits)
    assert packed.dtype == np.uint8
    assert packed.shape == (q.shape[0], (q.shape[1] * pack_bits + 7) // 8)
    for j in range(q.shape[0]):
        for i in range(q.shape[1]):
            assert _unpack_reference(packed[j], i, pack_bits) == int(q[j, i])


@pytest.mark.skipif(not _CKPT.exists(), reason="checkpoint EWC absent")
@pytest.mark.parametrize("mode,bits", [("linear", 4), ("ternary", 2), ("binary", 1)])
def test_export_uses_emulator_primitives(mode: str, bits: int) -> None:
    """Les poids exportés == émulateur `_quant_weight_mode` (parité par construction)."""
    from src.utils.int8_c_emulation import _quant_weight_mode

    mod = _load_export_module()
    q = mod._ewc_subint8_quantize(_CKPT, mode, bits, "per_channel", None)
    for name in ("w1", "w2", "w3"):
        w = getattr(q["w"], name)
        q_ref, s_ref = _quant_weight_mode(w, mode, "per_channel", bits)
        assert np.array_equal(q["qw"][name], q_ref.astype(np.int32))
        assert np.allclose(q["scales"][name], s_ref.astype(np.float32))


@pytest.mark.skipif(not _CKPT.exists(), reason="checkpoint EWC absent")
def test_generated_header_guarded(tmp_path: Path) -> None:
    mod = _load_export_module()
    mod.export_ewc_subint8_to_c(_CKPT, tmp_path, "ternary", 2, "per_channel",
                                "symmetric", packed=True)
    header = (tmp_path / "ewc_head_subint8_weights.h").read_text()
    assert "EWC_SUBINT8_WEIGHTS_PROVIDED 1" in header
    assert "EWC_SUBINT8_PACK_BITS 2" in header
    assert "EWC_SUBINT8_PACKED 1" in header
    assert "uint8_t EWC_SUB_W1" in header  # stockage packé
    assert "NE PAS ÉDITER" in header
