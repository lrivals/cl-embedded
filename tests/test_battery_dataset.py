"""
tests/test_battery_dataset.py — Tests unitaires pour battery_dataset.py.

Tests unitaires : fixtures synthétiques (pas de dépendance au vrai CSV).
Tests d'intégration : marqués ``@pytest.mark.integration``, ignorés si le CSV est absent.

Exécution :
    pytest tests/test_battery_dataset.py -v
    pytest tests/test_battery_dataset.py -v -m integration  # avec vrai CSV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.battery_dataset import (
    CYCLE_COL,
    FEATURE_COLUMNS,
    N_FEATURES,
    N_TASKS,
    RUL_COL,
    RUL_FAILURE_THRESHOLD,
    VAL_RATIO,
    fit_normalizer,
    get_battery_dataloaders,
    get_battery_dataloaders_single_task,
    load_raw_dataset,
    normalize_features,
    save_normalizer,
)

# ---------------------------------------------------------------------------
# Chemins pour les tests d'intégration
# ---------------------------------------------------------------------------

_CSV_PATH = Path(
    "data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv"
)
_NORMALIZER_PATH = Path("configs/battery_normalizer.yaml")

_integration = pytest.mark.skipif(
    not _CSV_PATH.exists(),
    reason=f"CSV introuvable : {_CSV_PATH}",
)

# ---------------------------------------------------------------------------
# Fixtures synthétiques
# ---------------------------------------------------------------------------

_N_SYNTHETIC = 300  # cycles synthétiques (3 tâches × 100 cycles)


@pytest.fixture
def synthetic_battery_csv(tmp_path: Path) -> Path:
    """CSV synthétique de 300 lignes imitant Battery_RUL.csv."""
    rng = np.random.default_rng(42)
    n = _N_SYNTHETIC
    rul_values = np.arange(n - 1, -1, -1, dtype=np.float32)  # décroissant : n-1 → 0
    df = pd.DataFrame(
        {
            CYCLE_COL: np.arange(1, n + 1, dtype=np.float32),
            "Discharge Time (s)": rng.normal(7000, 500, n).astype(np.float32),
            "Decrement 3.6-3.4V (s)": rng.normal(1100, 100, n).astype(np.float32),
            "Max. Voltage Dischar. (V)": rng.normal(4.2, 0.05, n).astype(np.float32),
            "Min. Voltage Charg. (V)": rng.normal(3.2, 0.05, n).astype(np.float32),
            "Time at 4.15V (s)": rng.normal(5400, 300, n).astype(np.float32),
            "Time constant current (s)": rng.normal(6700, 200, n).astype(np.float32),
            "Charging time (s)": rng.normal(10500, 400, n).astype(np.float32),
            RUL_COL: rul_values,
        }
    )
    csv_path = tmp_path / "Battery_RUL.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_normalizer_path(tmp_path: Path, synthetic_battery_csv: Path) -> Path:
    """Normalizer YAML fité sur les données synthétiques (Task 1)."""
    df = load_raw_dataset(synthetic_battery_csv)
    n_task = len(df) // N_TASKS
    df_task1 = df.iloc[:n_task]
    normalizer = fit_normalizer(df_task1)
    norm_path = tmp_path / "battery_normalizer.yaml"
    save_normalizer(norm_path, normalizer)
    return norm_path


# ---------------------------------------------------------------------------
# Tests unitaires : constantes et structure
# ---------------------------------------------------------------------------


def test_n_features_correct():
    """N_FEATURES = 7 features électrochimiques."""
    assert N_FEATURES == 7
    assert len(FEATURE_COLUMNS) == N_FEATURES


def test_rul_failure_threshold_sensible():
    """RUL_FAILURE_THRESHOLD = 200 cycles (valeur documentée dans la spec)."""
    assert RUL_FAILURE_THRESHOLD == 200


# ---------------------------------------------------------------------------
# Tests unitaires : load_raw_dataset
# ---------------------------------------------------------------------------


def test_load_raw_dataset_adds_faulty_column(synthetic_battery_csv: Path):
    """load_raw_dataset ajoute la colonne 'faulty' binaire."""
    df = load_raw_dataset(synthetic_battery_csv)
    assert "faulty" in df.columns
    assert set(df["faulty"].unique()).issubset({0.0, 1.0})


def test_load_raw_dataset_sorted_by_cycle(synthetic_battery_csv: Path):
    """DataFrame trié par Cycle_Index croissant."""
    df = load_raw_dataset(synthetic_battery_csv)
    assert (df[CYCLE_COL].diff().dropna() >= 0).all()


def test_load_raw_dataset_label_ratio(synthetic_battery_csv: Path):
    """Ratio de labels=1 correspond à RUL < RUL_FAILURE_THRESHOLD."""
    df = load_raw_dataset(synthetic_battery_csv)
    expected_ratio = (df[RUL_COL] < RUL_FAILURE_THRESHOLD).mean()
    actual_ratio = df["faulty"].mean()
    assert abs(expected_ratio - actual_ratio) < 1e-6


def test_load_raw_dataset_file_not_found():
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_raw_dataset(Path("/nonexistent/Battery_RUL.csv"))


def test_load_raw_dataset_missing_column(tmp_path: Path):
    """CSV sans colonne RUL → ValueError."""
    df = pd.DataFrame({"Cycle_Index": [1, 2], "Discharge Time (s)": [100, 200]})
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="manquantes"):
        load_raw_dataset(bad_csv)


# ---------------------------------------------------------------------------
# Tests unitaires : normalisation
# ---------------------------------------------------------------------------


def test_fit_normalizer_shape(synthetic_battery_csv: Path):
    """fit_normalizer retourne mean/std pour chaque feature."""
    df = load_raw_dataset(synthetic_battery_csv)
    norm = fit_normalizer(df)
    assert set(norm["mean"].keys()) == set(FEATURE_COLUMNS)
    assert all(v > 0 for v in norm["std"].values())


def test_normalize_features_zero_mean(synthetic_battery_csv: Path):
    """Après normalisation sur soi-même, mean ≈ 0 et std ≈ 1 par feature."""
    df = load_raw_dataset(synthetic_battery_csv)
    norm = fit_normalizer(df)
    df_norm = normalize_features(df, norm)
    for feat in FEATURE_COLUMNS:
        assert abs(df_norm[feat].mean()) < 0.01
        assert abs(df_norm[feat].std() - 1.0) < 0.05


def test_save_load_normalizer_roundtrip(tmp_path: Path, synthetic_battery_csv: Path):
    """save_normalizer puis load_battery_normalizer → valeurs identiques."""
    from src.data.battery_dataset import load_battery_normalizer

    df = load_raw_dataset(synthetic_battery_csv)
    norm = fit_normalizer(df)
    path = tmp_path / "norm.yaml"
    save_normalizer(path, norm)
    loaded = load_battery_normalizer(path)
    for feat in FEATURE_COLUMNS:
        assert abs(norm["mean"][feat] - loaded["mean"][feat]) < 1e-5
        assert abs(norm["std"][feat] - loaded["std"][feat]) < 1e-5


# ---------------------------------------------------------------------------
# Tests unitaires : get_battery_dataloaders
# ---------------------------------------------------------------------------


def test_get_battery_dataloaders_returns_3_tasks(
    synthetic_battery_csv: Path, synthetic_normalizer_path: Path
):
    """get_battery_dataloaders retourne exactement N_TASKS=3 dicts."""
    tasks = get_battery_dataloaders(
        csv_path=synthetic_battery_csv,
        normalizer_path=synthetic_normalizer_path,
        batch_size=8,
    )
    assert len(tasks) == N_TASKS
    for i, task in enumerate(tasks, start=1):
        assert task["task_id"] == i
        assert task["temporal_window"] == i
        assert task["n_train"] > 0
        assert task["n_val"] > 0


def test_get_battery_dataloaders_tensor_shapes(
    synthetic_battery_csv: Path, synthetic_normalizer_path: Path
):
    """X=[batch, 7], y=[batch, 1], dtype float32."""
    tasks = get_battery_dataloaders(
        csv_path=synthetic_battery_csv,
        normalizer_path=synthetic_normalizer_path,
        batch_size=4,
    )
    for task in tasks:
        x_batch, y_batch = next(iter(task["train_loader"]))
        assert x_batch.shape[1] == N_FEATURES
        assert y_batch.shape[1] == 1
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32


def test_get_battery_dataloaders_no_overlap(
    synthetic_battery_csv: Path, synthetic_normalizer_path: Path
):
    """Aucun cycle n'apparaît dans deux tâches différentes (pas de fuite)."""
    tasks = get_battery_dataloaders(
        csv_path=synthetic_battery_csv,
        normalizer_path=synthetic_normalizer_path,
        batch_size=256,
    )
    total_samples = sum(t["n_train"] + t["n_val"] for t in tasks)
    assert total_samples == _N_SYNTHETIC


# ---------------------------------------------------------------------------
# Tests unitaires : get_battery_dataloaders_single_task
# ---------------------------------------------------------------------------


def test_get_battery_dataloaders_single_task_returns_dict(synthetic_battery_csv: Path):
    """get_battery_dataloaders_single_task retourne un dict (pas une liste)."""
    result = get_battery_dataloaders_single_task(
        csv_path=synthetic_battery_csv, batch_size=4
    )
    assert isinstance(result, dict)
    for key in ("train_loader", "val_loader", "test_loader", "n_train", "n_val", "n_test"):
        assert key in result
    total = result["n_train"] + result["n_val"] + result["n_test"]
    assert total == _N_SYNTHETIC


def test_get_battery_dataloaders_single_task_tensor_shapes(synthetic_battery_csv: Path):
    """X=[batch, 7], y=[batch, 1]."""
    result = get_battery_dataloaders_single_task(synthetic_battery_csv, batch_size=8)
    x, y = next(iter(result["train_loader"]))
    assert x.shape[1] == N_FEATURES
    assert y.shape[1] == 1


# ---------------------------------------------------------------------------
# Tests d'intégration (vrai CSV)
# ---------------------------------------------------------------------------


@_integration
def test_integration_load_raw_dataset():
    """Dataset réel : 15 064 lignes, label ~18% positifs."""
    df = load_raw_dataset(_CSV_PATH)
    assert len(df) == 15064
    assert df["faulty"].mean() > 0.15
    assert df["faulty"].mean() < 0.25


@_integration
def test_integration_get_battery_dataloaders():
    """Pipeline complet : 3 tâches, DataLoaders fonctionnels, tenseurs finis."""
    if not _NORMALIZER_PATH.exists():
        pytest.skip("Normalizer YAML absent")

    tasks = get_battery_dataloaders(
        csv_path=_CSV_PATH, normalizer_path=_NORMALIZER_PATH, batch_size=64
    )
    assert len(tasks) == 3
    for task in tasks:
        x, y = next(iter(task["train_loader"]))
        assert x.shape[1] == N_FEATURES
        assert torch.isfinite(x).all()
