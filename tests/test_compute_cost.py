"""
tests/test_compute_cost.py — Tests unitaires pour compute_cost.

Vérifie les formules analytiques MACs pour chacun des 6 modèles du benchmark.

Exécution :
    pytest tests/test_compute_cost.py -v
"""

from __future__ import annotations

import pytest

from src.evaluation.compute_cost import (
    compute_bops,
    compute_bops_for_model,
    compute_flops,
    compute_flops_for_model,
    compute_macs,
    compute_params_for_model,
    macs_dbscan,
    macs_ewc_mlp,
    macs_hdc,
    macs_kmeans,
    macs_mahalanobis,
    macs_tinyol,
    params_ewc_mlp,
    params_mahalanobis,
)


def test_macs_ewc_mlp_two_layers():
    assert macs_ewc_mlp(25, [32, 16], 1) == 25 * 32 + 32 * 16 + 16 * 1


def test_macs_ewc_mlp_single_layer():
    assert macs_ewc_mlp(5, [64], 2) == 5 * 64 + 64 * 2


def test_macs_tinyol():
    assert macs_tinyol(25, [32, 16, 8], 2) == 25 * 32 + 32 * 16 + 16 * 8 + 8 * 2


def test_macs_hdc():
    assert macs_hdc(5, 2048, 2) == 2048 * 5 + 2048 * 2


def test_macs_kmeans():
    assert macs_kmeans(5, 4) == 4 * 5


def test_macs_mahalanobis():
    assert macs_mahalanobis(5) == 25 + 5


def test_macs_dbscan():
    assert macs_dbscan(5, 100) == 500


def test_dispatcher_matches_direct_calls():
    assert compute_macs(
        "EWC", n_features=25, hidden_dims=[32, 16], n_classes=1
    ) == macs_ewc_mlp(25, [32, 16], 1)
    assert compute_macs(
        "HDC", n_features=5, dim_hv=2048, n_classes=2
    ) == macs_hdc(5, 2048, 2)
    assert compute_macs("Mahalanobis", n_features=5) == macs_mahalanobis(5)


def test_dispatcher_rejects_unknown_model():
    with pytest.raises(KeyError, match="Modèle inconnu"):
        compute_macs("Unknown", n_features=5)


# ── S3301 : FLOPs / BOPs / Params (non-régression + nouvelles fonctions) ──


def test_macs_unchanged_after_extension():
    """Non-régression : les MACs gardent exactement les valeurs historiques."""
    assert macs_ewc_mlp(5, [16], 2) == 5 * 16 + 16 * 2
    assert compute_macs("EWC", n_features=25, hidden_dims=[32, 16], n_classes=1) == (
        25 * 32 + 32 * 16 + 16 * 1
    )


def test_compute_flops_is_twice_macs():
    assert compute_flops(80) == 160
    assert compute_flops(0) == 0


def test_compute_bops_fp32_vs_int8_ratio():
    macs = 80
    assert compute_bops(macs, 32) == compute_flops(macs) * 1024
    assert compute_bops(macs, 8) == compute_flops(macs) * 64
    # Ratio FP32/INT8 attendu = (32/8)² = 16
    assert compute_bops(macs, 32) == 16 * compute_bops(macs, 8)


def test_flops_bops_dispatchers_consistent():
    macs = compute_macs("EWC", n_features=5, hidden_dims=[16], n_classes=2)
    assert compute_flops_for_model("EWC", n_features=5, hidden_dims=[16], n_classes=2) == (
        compute_flops(macs)
    )
    assert compute_bops_for_model("EWC", 8, n_features=5, hidden_dims=[16], n_classes=2) == (
        compute_bops(macs, 8)
    )


def test_count_params_ewc_mlp():
    # Linear(5,16): 5*16+16=96 ; Linear(16,2): 16*2+2=34 → 130
    assert params_ewc_mlp(5, [16], 2) == 96 + 34
    assert compute_params_for_model("EWC", n_features=5, hidden_dims=[16], n_classes=2) == 130


def test_count_params_mahalanobis():
    # μ (5) + Σ⁻¹ (25) = 30
    assert params_mahalanobis(5) == 30
    assert compute_params_for_model("Mahalanobis", n_features=5) == 30


def test_params_dispatcher_rejects_unknown_model():
    with pytest.raises(KeyError, match="Modèle inconnu"):
        compute_params_for_model("Unknown", n_features=5)
