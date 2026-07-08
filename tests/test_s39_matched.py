"""test_s39_matched.py — Garde-fous du harnais de comparaison INT8 appariée (S3918).

Vérifie les trois invariants qui rendent une comparaison PC↔board *pertinente* :
  1. le côté PC exécute le **schéma board** (émulateur), jamais le QAT S28 ;
  2. les deux côtés partagent **une seule source de données** (``load_condition_arrays``) ;
  3. l'inférence gelée ``legacy_c`` est **déterministe** (bit-exacte, reproductible).

Les tests qui touchent aux données sont ``skip`` si le dataset n'est pas présent (règle
projet : pas d'échec dû à des données absentes en CI).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import scripts.run_s39_matched_compare as matched
from src.utils.int8_c_emulation import (
    EWCHeadWeights,
    QuantConfig,
    forward_quant,
    predict,
)


# ── 1. Le côté PC utilise le schéma board, pas le QAT S28 ─────────────────────

def test_pc_side_uses_board_scheme_not_qat():
    """Le mapping de schémas pointe sur l'émulateur (``QuantConfig``), jamais sur le QAT."""
    # Tous les schémas résolvent en QuantConfig émulateur.
    for scheme, cfg in matched.SCHEME_TO_QUANTCONFIG.items():
        assert isinstance(cfg, QuantConfig), f"{scheme} n'est pas un QuantConfig émulateur"

    # legacy_c == kernel v1 ; per_channel_int8/q15 == kernel v2.
    assert matched.SCHEME_TO_QUANTCONFIG["legacy_c"].name == "legacy_c"
    assert matched.SCHEME_TO_QUANTCONFIG["per_channel_int8"].name == "per_channel_int8"
    assert matched.SCHEME_TO_QUANTCONFIG["q15"].name == "q15"

    # Le module ne s'appuie PAS sur le QAT Sprint 28 : le forward émulé provient bien
    # de l'émulateur bit-exact, et aucune ligne d'import ne tire ``ewc_mlp_int8``.
    assert matched.forward_quant.__module__ == "src.utils.int8_c_emulation"
    import_lines = [
        ln for ln in inspect.getsource(matched).splitlines()
        if ln.lstrip().startswith(("import ", "from "))
    ]
    assert not any("ewc_mlp_int8" in ln for ln in import_lines)


# ── 2. Pipeline de données à source unique ────────────────────────────────────

def test_matched_pipeline_single_source():
    """Le harnais consomme ``load_condition_arrays`` (source unique board/PC, S3508)."""
    src = inspect.getsource(matched)
    assert "load_condition_arrays" in src
    # La fonction d'entraînement de tête est celle de l'ablation (mêmes hyperparamètres board).
    assert matched.train_ewc_head.__module__ == "scripts.run_s39_int8_ablation"


# ── 3. L'inférence gelée legacy_c est déterministe (bit-exacte) ───────────────

def _synthetic_head(seed: int = 0) -> tuple[EWCHeadWeights, np.ndarray]:
    """Tête EWC 5→32→16→2 synthétique + un lot d'entrées (pas de dataset requis)."""
    rng = np.random.default_rng(seed)
    w = EWCHeadWeights(
        w1=rng.standard_normal((32, 5)) * 0.3, b1=rng.standard_normal(32) * 0.1,
        w2=rng.standard_normal((16, 32)) * 0.3, b2=rng.standard_normal(16) * 0.1,
        w3=rng.standard_normal((2, 16)) * 0.3, b3=rng.standard_normal(2) * 0.1,
    )
    X = rng.standard_normal((64, 5)).astype(np.float64)
    return w, X


def test_frozen_is_deterministic():
    """Deux exécutions émulateur ``legacy_c`` sur mêmes données → sorties identiques."""
    w, X = _synthetic_head()
    cfg = QuantConfig.legacy_c()
    logits_a = forward_quant(w, X, cfg)
    logits_b = forward_quant(w, X, cfg)
    # Bit-exact : le chemin entier est entièrement déterministe (pas de flottant stochastique).
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(predict(logits_a), predict(logits_b))


def test_frozen_per_channel_deterministic():
    """Le schéma v2 par-canal est lui aussi déterministe (garde-fou parité gelée S3919)."""
    w, X = _synthetic_head(seed=39)
    cfg = QuantConfig.per_channel_int8()
    a = forward_quant(w, X, cfg)
    b = forward_quant(w, X, cfg)
    np.testing.assert_array_equal(a, b)


# ── 4. Structure de sortie compatible confrontation de parité (S3919) ─────────

def test_compare_scheme_output_schema():
    """``compare_scheme`` émet les clés attendues par la confrontation board (S3919)."""
    from src.utils.int8_c_emulation import calibrate_activations, forward_fp32

    w, X = _synthetic_head(seed=7)
    y = (X[:, 0] > 0).astype(np.int64)
    act_max = calibrate_activations(w, X)
    logits_fp32 = forward_fp32(w, X)
    cell = matched.compare_scheme(w, X, y, "per_channel_int8", act_max, logits_fp32)

    assert cell["parity_class"] == "exact"
    assert cell["n_compared"] == len(y)
    assert set(cell["rows"][0]) >= {"idx", "true", "pred_fp32", "pred_int8_pc", "pred_pc"}
    # pred_pc == pred_int8_pc (alias : ce que le board doit reproduire).
    assert all(r["pred_pc"] == r["pred_int8_pc"] for r in cell["rows"])
    assert 0.0 <= cell["agreement_vs_fp32"] <= 1.0
