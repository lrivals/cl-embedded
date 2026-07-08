"""test_int8_c_emulation.py — Validation de l'émulateur INT8 vs logs board (S3903).

Crédite l'émulateur ``src/utils/int8_c_emulation.py`` (S3902) : montre qu'il reproduit
la dégradation F1 **réelle** observée sur board (Sprint 29/36) **sans flasher**, et que
les schémas calibrés récupèrent l'accord. On s'appuie sur :

  - la tête EWC board pronostia 5feat réentraînée à l'identique
    (``run_s39_int8_ablation.train_ewc_head``) ;
  - le log board ``experiments/exp_S36_board_frozen_int8_5feat_ewc_pronostia`` (F1=0.138,
    agreement_int8_vs_fp32=0.736).

**Honnêteté (S3903)** : la parité *exacte* board↔émulateur dépend de l'ordre de streaming
et de la normalisation hôte. On valide donc le **mécanisme** (forte chute en ``legacy_c``,
récupération en calibré, même régime d'accord que le board) avec des tolérances larges et
documentées, pas un match au centième.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.int8_c_emulation import (
    QuantConfig,
    _sat8,
    _trunc_to_int,
    _wrap_int8,
    _wrap_int16,
    agreement,
    forward_fp32,
    forward_quant,
)

REPO = Path(__file__).resolve().parent.parent
PRONOSTIA_DATA = REPO / "data" / "raw" / "Pronostia dataset"
BOARD_LOG = REPO / "experiments" / "exp_S36_board_frozen_int8_5feat_ewc_pronostia" / "results.json"

_needs_data = pytest.mark.skipif(
    not PRONOSTIA_DATA.exists(),
    reason="data/raw/Pronostia dataset absent — test d'entraînement ignoré",
)


# ── Primitives entières bit-exactes ──────────────────────────────────────────


def test_bit_exact_primitives():
    """_wrap_int8 / _wrap_int16 / _trunc_to_int / _sat8 aux valeurs limites."""
    # (int8_t) wrap modulo 256 dans [-128, 127]
    assert int(_wrap_int8(np.array(128))) == -128      # overflow → wrap
    assert int(_wrap_int8(np.array(127))) == 127       # borne haute
    assert int(_wrap_int8(np.array(-129))) == 127      # underflow → wrap
    assert int(_wrap_int8(np.array(256))) == 0

    # (int16_t) wrap modulo 65536 dans [-32768, 32767] (overflow latent F1)
    assert int(_wrap_int16(32768)) == -32768
    assert int(_wrap_int16(-32769)) == 32767
    assert int(_wrap_int16(32767)) == 32767

    # (int) cast = troncature vers zéro (≠ floor pour négatifs)
    assert int(_trunc_to_int(np.array(1.7))) == 1
    assert int(_trunc_to_int(np.array(-1.7))) == -1    # vers 0, pas -2
    assert int(_trunc_to_int(np.array(-0.9))) == 0

    # SAT8 = saturation dans [-127, 127]
    assert int(_sat8(np.array(200))) == 127
    assert int(_sat8(np.array(-200))) == -127
    assert int(_sat8(np.array(50))) == 50


# ── Reproduction de la dégradation board (nécessite les données) ─────────────


@pytest.fixture(scope="module")
def pronostia_head():
    """Tête EWC board pronostia 5feat + données (réentraînée à l'identique S36)."""
    import torch

    from scripts.run_s39_int8_ablation import train_ewc_head
    from src.evaluation.feature_conditions import load_condition_arrays
    from src.utils.int8_c_emulation import EWCHeadWeights

    X, y, _idx, _names = load_condition_arrays("pronostia", "5feat", "ewc", seed=42)
    model = train_ewc_head(X, y, seed=42)
    with torch.no_grad():
        state = {k: v.cpu() for k, v in model.state_dict().items()}
    return EWCHeadWeights.from_state_dict(state), X, y


@_needs_data
def test_legacy_reproduces_board_degradation(pronostia_head):
    """legacy_c dégrade fortement la F1 ; les schémas calibrés récupèrent l'accord."""
    from src.evaluation.metrics import compute_fault_f1
    from src.utils.int8_c_emulation import predict

    w, X, y = pronostia_head

    def f1(logits):
        return float(compute_fault_f1(y, predict(logits))["f1_faulty"])

    fp = forward_fp32(w, X)
    f1_fp32 = f1(fp)
    f1_legacy = f1(forward_quant(w, X, QuantConfig.legacy_c()))

    # FP32 apprend bien la tâche (board FP32 ≈ 0.916).
    assert f1_fp32 > 0.5, f"F1 fp32 trop bas : {f1_fp32}"
    # legacy_c s'effondre (board INT8 = 0.138) — forte chute, mécanisme reproduit.
    assert f1_legacy < f1_fp32 - 0.4, f"pas de dégradation legacy : {f1_legacy} vs {f1_fp32}"

    # Un schéma calibré récupère l'accord de prédiction quasi exact vs FP32.
    for ctor in (QuantConfig.per_channel_int8, QuantConfig.q15):
        acc = agreement(forward_quant(w, X, ctor()), fp)
        assert acc >= 0.95, f"{ctor().name} ne récupère pas l'accord : {acc}"


@_needs_data
@pytest.mark.skipif(not BOARD_LOG.exists(), reason="log board S36 absent")
def test_agreement_matches_board_log(pronostia_head):
    """agreement(legacy, fp32) émulé est dans le même régime dégradé que le board.

    Le board loggue agreement_int8_vs_fp32 ≈ 0.736. L'émulateur reproduit un accord
    dégradé (< 0.95, nettement sous la parité) et proche du board à tolérance large
    (l'écart résiduel vient de l'ordre de streaming / normalisation hôte, S3903).
    """
    w, X, _y = pronostia_head
    fp = forward_fp32(w, X)
    emu = agreement(forward_quant(w, X, QuantConfig.legacy_c()), fp)

    board = json.loads(BOARD_LOG.read_text())["agreement_int8_vs_fp32"]

    # Même régime : dégradé (pas de parité) mais loin d'un accord aléatoire.
    assert 0.5 < emu < 0.95, f"accord émulé hors régime dégradé : {emu}"
    # Proche du board (tolérance large et documentée : ordre/normalisation).
    assert abs(emu - board) < 0.15, f"émulé {emu:.3f} vs board {board:.3f} (>0.15)"
