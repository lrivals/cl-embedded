"""
tests/test_pronostia_dataset.py — Tests unitaires pour pronostia_dataset.py.

Tests unitaires : fixtures synthétiques (pas de dépendance aux vrais .npy).
Tests d'intégration : marqués ``@pytest.mark.integration``, ignorés si les .npy sont absents.

Exécution :
    pytest tests/test_pronostia_dataset.py -v
    pytest tests/test_pronostia_dataset.py -v -m integration  # avec vrais .npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.pronostia_dataset import (
    CHANNEL_NAMES,
    COL_ACC_HORIZ,
    COL_ACC_VERT,
    CONDITION_BEARING_MAP,
    FAILURE_RATIO,
    FEATURE_NAMES,
    N_CHANNELS,
    N_CONDITIONS,
    N_FEATURES,
    STEP_SIZE,
    WINDOW_SIZE,
    _compute_channel_features,
    fit_normalizer,
    get_pronostia_dataloaders,
    get_pronostia_dataloaders_single_task,
    load_bearing_features,
    load_condition_features,
    save_normalizer,
)

# ---------------------------------------------------------------------------
# Chemins pour les tests d'intégration
# ---------------------------------------------------------------------------

_NPY_DIR = Path("data/raw/Pronostia dataset/binaries")
_NORMALIZER_PATH = Path("configs/pronostia_normalizer.yaml")

_integration = pytest.mark.skipif(
    not (_NPY_DIR / "0.npy").exists(),
    reason=f"Fichiers .npy introuvables : {_NPY_DIR}",
)

# ---------------------------------------------------------------------------
# Fixtures synthétiques
# ---------------------------------------------------------------------------

_N_WINDOWS = 20  # fenêtres synthétiques par roulement fictif


@pytest.fixture
def synthetic_npy_dir(tmp_path: Path) -> Path:
    """Répertoire avec 6 fichiers .npy synthétiques de petite taille."""
    rng = np.random.default_rng(42)
    npy_dir = tmp_path / "binaries"
    npy_dir.mkdir()

    for bearing_idx in range(6):
        n_windows = _N_WINDOWS + bearing_idx  # tailles légèrement différentes
        n_samples = n_windows * WINDOW_SIZE
        # Format : [hour, min, sec, microsec, acc_horiz, acc_vert]
        data = np.zeros((n_samples, 6), dtype=np.float64)
        data[:, COL_ACC_HORIZ] = rng.normal(0, 0.5 + bearing_idx * 0.3, n_samples)
        data[:, COL_ACC_VERT] = rng.normal(0, 0.4 + bearing_idx * 0.2, n_samples)
        np.save(npy_dir / f"{bearing_idx}.npy", data)

    return npy_dir


@pytest.fixture
def synthetic_normalizer_path(tmp_path: Path, synthetic_npy_dir: Path) -> Path:
    """Normalizer YAML fité sur la Condition 1 synthétique."""
    from src.data.pronostia_dataset import load_condition_features

    feats, _ = load_condition_features(synthetic_npy_dir, condition=1)
    normalizer = fit_normalizer(feats)
    norm_path = tmp_path / "pronostia_normalizer.yaml"
    save_normalizer(norm_path, normalizer)
    return norm_path


# ---------------------------------------------------------------------------
# Tests unitaires : constantes et structure
# ---------------------------------------------------------------------------


def test_n_features_correct():
    """N_FEATURES = 6 stats × 2 canaux + 1 temporal = 13."""
    assert N_FEATURES == 13
    assert len(FEATURE_NAMES) == N_FEATURES


def test_condition_bearing_map_complete():
    """Chaque condition a exactement 2 roulements, indices 0–5 couverts."""
    all_indices = [idx for indices in CONDITION_BEARING_MAP.values() for idx in indices]
    assert sorted(all_indices) == list(range(6))
    for cond, indices in CONDITION_BEARING_MAP.items():
        assert len(indices) == 2, f"Condition {cond} doit avoir 2 roulements"


def test_compute_channel_features_shape():
    """_compute_channel_features retourne [12] pour une fenêtre [2560, 2]."""
    rng = np.random.default_rng(0)
    window = rng.normal(0, 1, (WINDOW_SIZE, N_CHANNELS)).astype(np.float32)
    feats = _compute_channel_features(window)
    assert feats.shape == (N_CHANNELS * 6,)
    assert feats.dtype == np.float32


def test_compute_channel_features_zero_window():
    """Fenêtre constante (rms=0) → crest_factor = 0.0, pas de division par zéro."""
    window = np.zeros((WINDOW_SIZE, N_CHANNELS), dtype=np.float32)
    feats = _compute_channel_features(window)
    assert np.isfinite(feats).all()


def test_temporal_position_bounds(synthetic_npy_dir: Path):
    """La feature temporal_position (index 12) doit être dans [0, 1] pour toutes les fenêtres."""
    npy_path = synthetic_npy_dir / "0.npy"
    feats, _ = load_bearing_features(npy_path)
    temporal_pos = feats[:, -1]  # index 12
    assert temporal_pos.min() >= 0.0, f"temporal_position min hors bornes : {temporal_pos.min()}"
    assert temporal_pos.max() <= 1.0, f"temporal_position max hors bornes : {temporal_pos.max()}"


def test_feature_ram_footprint(synthetic_npy_dir: Path):
    """Un vecteur de 13 features FP32 doit peser exactement 52 octets (Gap 2 — RAM embarquée)."""
    npy_path = synthetic_npy_dir / "0.npy"
    feats, _ = load_bearing_features(npy_path)
    bytes_per_sample = feats.itemsize * feats.shape[1]  # 4 B × 13 = 52 B
    assert bytes_per_sample == 13 * 4, \
        f"RAM features : attendu 52 B, obtenu {bytes_per_sample} B"


# ---------------------------------------------------------------------------
# Tests unitaires : load_bearing_features
# ---------------------------------------------------------------------------


def test_load_bearing_features_shape(synthetic_npy_dir: Path):
    """load_bearing_features retourne features [N_windows, 13] et labels [N_windows]."""
    npy_path = synthetic_npy_dir / "0.npy"
    feats, lbls = load_bearing_features(npy_path)

    n_windows = _N_WINDOWS  # fixture : bearing 0 a exactement _N_WINDOWS fenêtres
    assert feats.shape == (n_windows, N_FEATURES)
    assert lbls.shape == (n_windows,)
    assert feats.dtype == np.float32
    assert lbls.dtype == np.float32


def test_load_bearing_features_label_ratio(synthetic_npy_dir: Path):
    """Fraction de labels=1 ≈ FAILURE_RATIO (± 1 fenêtre)."""
    npy_path = synthetic_npy_dir / "0.npy"
    _, lbls = load_bearing_features(npy_path, failure_ratio=0.2)
    ratio = lbls.mean()
    assert abs(ratio - 0.2) <= 1 / _N_WINDOWS + 1e-6


def test_load_bearing_features_file_not_found():
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_bearing_features(Path("/nonexistent/0.npy"))


# ---------------------------------------------------------------------------
# Tests unitaires : load_condition_features
# ---------------------------------------------------------------------------


def test_load_condition_features_shape(synthetic_npy_dir: Path):
    """Condition 1 agrège 2 roulements → features concaténées."""
    feats, lbls = load_condition_features(synthetic_npy_dir, condition=1)
    # Bearing 0 : _N_WINDOWS, Bearing 1 : _N_WINDOWS + 1
    expected_n = _N_WINDOWS + (_N_WINDOWS + 1)
    assert feats.shape == (expected_n, N_FEATURES)
    assert lbls.shape == (expected_n,)


def test_load_condition_features_invalid_condition(synthetic_npy_dir: Path):
    with pytest.raises(ValueError, match="invalide"):
        load_condition_features(synthetic_npy_dir, condition=99)


# ---------------------------------------------------------------------------
# Tests unitaires : normalisation
# ---------------------------------------------------------------------------


def test_fit_normalizer_shape(synthetic_npy_dir: Path):
    """fit_normalizer retourne mean/std de longueur N_FEATURES."""
    feats, _ = load_condition_features(synthetic_npy_dir, condition=1)
    norm = fit_normalizer(feats)
    assert norm["mean"].shape == (N_FEATURES,)
    assert norm["std"].shape == (N_FEATURES,)
    assert (norm["std"] > 0).all()


def test_save_load_normalizer_roundtrip(
    tmp_path: Path, synthetic_npy_dir: Path
):
    """save_normalizer puis load_pronostia_normalizer → valeurs identiques."""
    from src.data.pronostia_dataset import load_pronostia_normalizer

    feats, _ = load_condition_features(synthetic_npy_dir, condition=1)
    norm = fit_normalizer(feats)
    path = tmp_path / "norm.yaml"
    save_normalizer(path, norm)
    loaded = load_pronostia_normalizer(path)
    np.testing.assert_allclose(norm["mean"], loaded["mean"], rtol=1e-5)
    np.testing.assert_allclose(norm["std"], loaded["std"], rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests unitaires : get_pronostia_dataloaders
# ---------------------------------------------------------------------------


def test_get_pronostia_dataloaders_returns_3_tasks(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
):
    """get_pronostia_dataloaders retourne exactement 3 dicts."""
    tasks = get_pronostia_dataloaders(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        batch_size=8,
    )
    assert len(tasks) == N_CONDITIONS
    for i, task in enumerate(tasks, start=1):
        assert task["task_id"] == i
        assert task["condition"] == i
        assert "train_loader" in task
        assert "val_loader" in task
        assert task["n_train"] > 0
        assert task["n_val"] > 0


def test_get_pronostia_dataloaders_tensor_shapes(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
):
    """Les tenseurs ont les bonnes dimensions : X=[batch, 13], y=[batch, 1]."""
    tasks = get_pronostia_dataloaders(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        batch_size=4,
    )
    for task in tasks:
        x_batch, y_batch = next(iter(task["train_loader"]))
        assert x_batch.shape[1] == N_FEATURES
        assert y_batch.shape[1] == 1
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests unitaires : get_pronostia_dataloaders_single_task
# ---------------------------------------------------------------------------


def test_get_pronostia_dataloaders_single_task_returns_dict(
    synthetic_npy_dir: Path,
):
    """get_pronostia_dataloaders_single_task retourne un dict (pas une liste)."""
    result = get_pronostia_dataloaders_single_task(
        npy_dir=synthetic_npy_dir, batch_size=4
    )
    assert isinstance(result, dict)
    for key in ("train_loader", "val_loader", "test_loader", "n_train", "n_val", "n_test"):
        assert key in result
    assert result["n_train"] > 0
    assert result["n_val"] > 0
    assert result["n_test"] > 0


# ---------------------------------------------------------------------------
# Tests d'intégration (vrais .npy)
# ---------------------------------------------------------------------------


@_integration
def test_integration_load_bearing_0():
    """Bearing1_1 : ~2 803 fenêtres de 2 560 points, 2 canaux."""
    feats, lbls = load_bearing_features(_NPY_DIR / "0.npy")
    assert feats.shape[1] == N_FEATURES
    assert feats.shape[0] > 2000  # ≈ 2 803 fenêtres attendues
    assert lbls.mean() > 0  # au moins quelques labels=1


@_integration
def test_integration_get_pronostia_dataloaders():
    """Pipeline complet : 3 tâches, DataLoaders fonctionnels."""
    if not _NORMALIZER_PATH.exists():
        pytest.skip("Normalizer YAML absent — générer d'abord via fit_normalizer + save_normalizer")

    tasks = get_pronostia_dataloaders(
        npy_dir=_NPY_DIR, normalizer_path=_NORMALIZER_PATH, batch_size=32
    )
    assert len(tasks) == 3
    for task in tasks:
        x, y = next(iter(task["train_loader"]))
        assert x.shape[1] == N_FEATURES
        assert torch.isfinite(x).all()
