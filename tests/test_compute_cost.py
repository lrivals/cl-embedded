"""
tests/test_compute_cost.py — Tests unitaires pour compute_cost.

Vérifie les formules analytiques MACs pour chacun des 6 modèles du benchmark.

Exécution :
    pytest tests/test_compute_cost.py -v
"""

from __future__ import annotations

import pytest

from src.evaluation.compute_cost import (
    compute_macs,
    macs_dbscan,
    macs_ewc_mlp,
    macs_hdc,
    macs_kmeans,
    macs_mahalanobis,
    macs_tinyol,
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
