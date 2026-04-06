# ruff: noqa: N806  — H_level, H_pos, H1, H2 sont des conventions mathématiques pour matrices HDC
from pathlib import Path

import numpy as np
import pytest

from src.models.hdc.base_vectors import generate_base_hvectors, load_base_vectors, save_base_vectors

D, N_LEVELS, N_FEATURES = 1024, 10, 4


def test_shapes():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert H_level.shape == (N_LEVELS, D)
    assert H_pos.shape == (N_FEATURES, D)


def test_dtype():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert H_level.dtype == np.int8
    assert H_pos.dtype == np.int8


def test_binary_values():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert set(np.unique(H_level)) == {-1, 1}
    assert set(np.unique(H_pos)) == {-1, 1}


def test_reproducibility():
    H1, P1 = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    H2, P2 = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    np.testing.assert_array_equal(H1, H2)
    np.testing.assert_array_equal(P1, P2)


def test_different_seeds_differ():
    H1, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    H2, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=99)
    assert not np.array_equal(H1, H2)


def test_approximate_orthogonality():
    """Les hypervecteurs de niveaux différents doivent être approximativement orthogonaux."""
    H_level, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    # Dot product normalisé entre deux vecteurs aléatoires → N(0, 1/sqrt(D))
    dots = []
    for i in range(N_LEVELS):
        for j in range(i + 1, N_LEVELS):
            dot = np.dot(H_level[i].astype(np.float32), H_level[j].astype(np.float32)) / D
            dots.append(abs(dot))
    assert np.mean(dots) < 0.1, f"Mean |dot|/D = {np.mean(dots):.4f} > 0.1"


def test_save_load_roundtrip(tmp_path):
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    path = tmp_path / "test_vectors.npz"
    save_base_vectors(H_level, H_pos, path)
    H_loaded, P_loaded = load_base_vectors(path)
    np.testing.assert_array_equal(H_level, H_loaded)
    np.testing.assert_array_equal(H_pos, P_loaded)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_base_vectors(Path("/tmp/nonexistent_vectors.npz"))
