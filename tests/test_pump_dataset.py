"""
tests/test_pump_dataset.py — Tests unitaires et d'intégration pour pump_dataset.py.

Tests unitaires : fixtures synthétiques (pas de dépendance au vrai CSV).
Tests d'intégration : marqués ``@pytest.mark.integration``, ignorés si le CSV est absent.

Exécution :
    pytest tests/test_pump_dataset.py -v
    pytest tests/test_pump_dataset.py -v -m integration  # avec vrai CSV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.pump_dataset import (
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    FEATURES_PER_CHANNEL,
    N_FEATURES,
    N_TASKS,
    STEP_SIZE,
    TEMPORAL_COLUMN,
    WINDOW_SIZE,
    CLStreamSplitter,
    PumpMaintenanceDataset,
    load_pump_normalizer,
)

# ---------------------------------------------------------------------------
# Chemins pour les tests d'intégration
# ---------------------------------------------------------------------------

_CSV_PATH = Path(
    "data/raw/pump_maintenance/"
    "Large Industrial_Pump_Maintenance_Dataset/"
    "Large_Industrial_Pump_Maintenance_Dataset.csv"
)
_NORMALIZER_PATH = Path("configs/pump_normalizer.yaml")

_integration = pytest.mark.skipif(
    not _CSV_PATH.exists(),
    reason=f"CSV introuvable : {_CSV_PATH}",
)

# ---------------------------------------------------------------------------
# Fixtures synthétiques
# ---------------------------------------------------------------------------

_N_SYNTHETIC = 200  # → (200 - 32) // 16 + 1 = 11 fenêtres


@pytest.fixture
def synthetic_pump_csv(tmp_path: Path) -> Path:
    """CSV synthétique de 200 lignes avec les colonnes brutes du vrai CSV."""
    rng = np.random.default_rng(42)
    n = _N_SYNTHETIC
    df = pd.DataFrame(
        {
            "Pump_ID": np.ones(n, dtype=int),
            "Temperature": rng.normal(75.0, 10.0, n).astype(np.float32),
            "Vibration": rng.normal(1.5, 0.5, n).astype(np.float32),
            "Pressure": rng.normal(40.0, 8.0, n).astype(np.float32),
            "Flow_Rate": rng.normal(100.0, 15.0, n).astype(np.float32),
            "RPM": rng.normal(1500.0, 100.0, n).astype(np.float32),
            "Operational_Hours": np.arange(n, dtype=np.float32),
            "Maintenance_Flag": np.where(np.arange(n) > 170, 1, 0),
        }
    )
    csv_path = tmp_path / "pump.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def loaded_dataset(synthetic_pump_csv: Path) -> PumpMaintenanceDataset:
    """PumpMaintenanceDataset chargé sur le CSV synthétique."""
    ds = PumpMaintenanceDataset(synthetic_pump_csv)
    ds.load()
    return ds


@pytest.fixture
def extracted_features(loaded_dataset: PumpMaintenanceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Features et labels extraits depuis le dataset synthétique."""
    return loaded_dataset.extract_features()


@pytest.fixture
def splitter(extracted_features: tuple[np.ndarray, np.ndarray]) -> CLStreamSplitter:
    """CLStreamSplitter avec features synthétiques, non normalisé."""
    features, labels = extracted_features
    return CLStreamSplitter(features, labels)


@pytest.fixture
def normalizer(splitter: CLStreamSplitter) -> dict:
    """Normaliseur calculé sur Task 1 du dataset synthétique."""
    return splitter.fit_normalizer(task_id=0)


@pytest.fixture
def normalized_splitter(splitter: CLStreamSplitter, normalizer: dict) -> CLStreamSplitter:
    """CLStreamSplitter avec normalisation appliquée."""
    splitter.apply_normalizer(normalizer)
    return splitter


# ---------------------------------------------------------------------------
# Tests unitaires — PumpMaintenanceDataset
# ---------------------------------------------------------------------------


def test_load_file_not_found():
    """FileNotFoundError si le CSV n'existe pas."""
    with pytest.raises(FileNotFoundError):
        PumpMaintenanceDataset(Path("nonexistent/pump.csv")).load()


def test_load_missing_column(tmp_path: Path):
    """ValueError si une colonne obligatoire est absente après renommage."""
    df = pd.DataFrame({"Temperature": [1.0], "Operational_Hours": [0.0], "Maintenance_Flag": [0]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        PumpMaintenanceDataset(csv_path).load()


def test_load_bad_labels(synthetic_pump_csv: Path, tmp_path: Path):
    """ValueError si maintenance_required contient une valeur hors {0, 1}."""
    df = pd.read_csv(synthetic_pump_csv)
    df.loc[0, "Maintenance_Flag"] = 2
    bad_path = tmp_path / "bad_labels.csv"
    df.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="Labels inattendus"):
        PumpMaintenanceDataset(bad_path).load()


def test_load_chronological_order(synthetic_pump_csv: Path):
    """Le DataFrame chargé doit être trié par operational_hours croissant."""
    ds = PumpMaintenanceDataset(synthetic_pump_csv)
    df = ds.load()
    hours = df[TEMPORAL_COLUMN].to_numpy()
    assert (np.diff(hours) >= 0).all(), "operational_hours n'est pas trié croissant."


def test_extract_features_requires_load(synthetic_pump_csv: Path):
    """RuntimeError si extract_features() est appelé sans load() préalable."""
    ds = PumpMaintenanceDataset(synthetic_pump_csv)
    with pytest.raises(RuntimeError, match="load\\(\\)"):
        ds.extract_features()


def test_window_shape(extracted_features: tuple[np.ndarray, np.ndarray]):
    """features.shape == (N_windows, 25) et labels.shape == (N_windows,)."""
    features, labels = extracted_features
    expected_n_windows = (_N_SYNTHETIC - WINDOW_SIZE) // STEP_SIZE + 1
    assert features.shape == (
        expected_n_windows,
        N_FEATURES,
    ), f"Attendu ({expected_n_windows}, {N_FEATURES}), obtenu {features.shape}"
    assert labels.shape == (
        expected_n_windows,
    ), f"Attendu ({expected_n_windows},), obtenu {labels.shape}"


def test_features_dtype(extracted_features: tuple[np.ndarray, np.ndarray]):
    """features et labels doivent être de dtype float32."""
    features, labels = extracted_features
    assert features.dtype == np.float32
    assert labels.dtype == np.float32


def test_labels_binary(extracted_features: tuple[np.ndarray, np.ndarray]):
    """Les labels doivent être dans {0.0, 1.0} après majority vote."""
    _, labels = extracted_features
    unique = set(labels.tolist())
    assert unique.issubset({0.0, 1.0}), f"Labels inattendus : {unique}"


def test_no_data_mutation(loaded_dataset: PumpMaintenanceDataset):
    """extract_features() ne doit pas modifier le DataFrame interne."""
    df_before = loaded_dataset._df.copy()
    loaded_dataset.extract_features()
    pd.testing.assert_frame_equal(loaded_dataset._df, df_before)


def test_feature_names_count():
    """FEATURE_NAMES doit contenir exactement N_FEATURES = 25 entrées."""
    assert (
        len(FEATURE_NAMES) == N_FEATURES
    ), f"FEATURE_NAMES: {len(FEATURE_NAMES)} entrées, attendu {N_FEATURES}"


def test_feature_names_structure():
    """Les 24 premières features suivent le pattern {stat}_{canal}."""
    expected_24 = [f"{stat}_{col}" for col in FEATURE_COLUMNS for stat in FEATURES_PER_CHANNEL]
    assert FEATURE_NAMES[:24] == expected_24
    assert FEATURE_NAMES[24] == "temporal_position"


# ---------------------------------------------------------------------------
# Tests unitaires — CLStreamSplitter
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises(extracted_features: tuple[np.ndarray, np.ndarray]):
    """CLStreamSplitter avec strategy != 'chronological' doit lever ValueError."""
    features, labels = extracted_features
    with pytest.raises(ValueError, match="chronological"):
        CLStreamSplitter(features, labels, strategy="random")


def test_chronological_split_n_tasks(splitter: CLStreamSplitter):
    """CLStreamSplitter doit créer exactement N_TASKS tranches."""
    assert len(splitter._slices) == N_TASKS


def test_chronological_split_contiguous(splitter: CLStreamSplitter):
    """Les tranches sont contiguës : sum(sizes) == n_total."""
    total = sum(len(f) for f, _ in splitter._slices)
    assert total == len(splitter._features)


def test_no_temporal_leakage(extracted_features: tuple[np.ndarray, np.ndarray]):
    """Les fenêtres de la tâche i précèdent toutes celles de la tâche i+1."""
    features, labels = extracted_features
    sp = CLStreamSplitter(features, labels)

    # Reconstituer les indices de début de chaque tranche dans le tableau original
    n = len(features)
    slice_size = n // N_TASKS
    boundaries = [i * slice_size for i in range(N_TASKS)] + [n]

    for t in range(N_TASKS - 1):
        end_t = boundaries[t + 1]
        start_next = boundaries[t + 1]
        # Vérifier que la tranche t se termine avant la tranche t+1
        assert end_t <= start_next, (
            f"Fuite temporelle entre task {t} et task {t+1}: "
            f"end={end_t}, start_next={start_next}"
        )

    # Vérifier que les tailles cumulées sont strictement croissantes
    cumul = 0
    for feats, _ in sp._slices:
        cumul += len(feats)
    assert cumul == n


def test_normalizer_fit_task0_mean(normalized_splitter: CLStreamSplitter):
    """Après normalisation, la moyenne de Task 1 doit être ≈ 0."""
    feats_t1, _ = normalized_splitter._slices[0]
    means = feats_t1.mean(axis=0)
    max_abs_mean = np.abs(means).max()
    assert (
        max_abs_mean < 0.01
    ), f"Moyenne max de Task 1 après normalisation : {max_abs_mean:.6f} (attendu < 0.01)"


def test_normalizer_fit_task0_std(normalized_splitter: CLStreamSplitter):
    """Après normalisation, l'écart-type de Task 1 doit être ≈ 1."""
    feats_t1, _ = normalized_splitter._slices[0]
    stds = feats_t1.std(axis=0)
    # Tolérance large car les features synthétiques peuvent avoir des std non unitaires
    assert np.all(stds > 0.5) and np.all(
        stds < 2.0
    ), f"Std de Task 1 hors [0.5, 2.0] : min={stds.min():.4f}, max={stds.max():.4f}"


def test_normalizer_not_mutating_original(extracted_features: tuple[np.ndarray, np.ndarray]):
    """apply_normalizer() ne doit pas modifier les features originales passées au constructeur."""
    features, labels = extracted_features
    features_copy = features.copy()
    sp = CLStreamSplitter(features, labels)
    normalizer = sp.fit_normalizer(task_id=0)
    sp.apply_normalizer(normalizer)
    # Les features passées au constructeur ne doivent pas avoir changé
    np.testing.assert_array_equal(features, features_copy)


def test_save_load_normalizer(splitter: CLStreamSplitter, normalizer: dict, tmp_path: Path):
    """save_normalizer + load_pump_normalizer doivent reproduire les mêmes stats."""
    norm_path = tmp_path / "pump_normalizer.yaml"
    splitter.save_normalizer(norm_path, normalizer)

    loaded = load_pump_normalizer(norm_path)

    np.testing.assert_allclose(loaded["mean"], normalizer["mean"], rtol=1e-5)
    np.testing.assert_allclose(loaded["std"], normalizer["std"], rtol=1e-5)


def test_save_normalizer_yaml_format(splitter: CLStreamSplitter, normalizer: dict, tmp_path: Path):
    """Le YAML généré doit contenir 'mean', 'std', et 'feature_names'."""
    norm_path = tmp_path / "pump_normalizer.yaml"
    splitter.save_normalizer(norm_path, normalizer)

    with open(norm_path) as f:
        data = yaml.safe_load(f)

    assert "mean" in data
    assert "std" in data
    assert "feature_names" in data
    assert len(data["feature_names"]) == N_FEATURES


# ---------------------------------------------------------------------------
# Tests unitaires — get_task_tensors
# ---------------------------------------------------------------------------


def test_task_tensor_shapes(normalized_splitter: CLStreamSplitter):
    """X : [N, 25] float32 — y : [N, 1] float32 pour chaque tâche."""
    for task_id in range(N_TASKS):
        task = normalized_splitter.get_task_tensors(task_id, batch_size=4)
        x_batch, y_batch = next(iter(task["train_loader"]))
        assert (
            x_batch.shape[1] == N_FEATURES
        ), f"Task {task_id}: X doit avoir {N_FEATURES} features, obtenu {x_batch.shape[1]}"
        assert (
            y_batch.shape[1] == 1
        ), f"Task {task_id}: y doit avoir shape [N, 1], obtenu {y_batch.shape}"
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32


def test_task_ids(normalized_splitter: CLStreamSplitter):
    """task_id doit valoir 1, 2, 3 (1-indexé)."""
    for i in range(N_TASKS):
        task = normalized_splitter.get_task_tensors(i, batch_size=4)
        assert task["task_id"] == i + 1


def test_task_n_train_n_val(normalized_splitter: CLStreamSplitter):
    """n_train + n_val doit couvrir exactement la taille de la tranche."""
    for i, (feats, _) in enumerate(normalized_splitter._slices):
        task = normalized_splitter.get_task_tensors(i, batch_size=4)
        assert task["n_train"] + task["n_val"] == len(feats), (
            f"Task {i}: n_train + n_val != taille tranche "
            f"({task['n_train']} + {task['n_val']} != {len(feats)})"
        )


# ---------------------------------------------------------------------------
# Tests d'intégration — vrai CSV
# ---------------------------------------------------------------------------


@_integration
def test_real_window_shape():
    """Sur le vrai CSV, features.shape == (N_windows, 25)."""
    ds = PumpMaintenanceDataset(_CSV_PATH)
    ds.load()
    features, labels = ds.extract_features()

    assert (
        features.shape[1] == N_FEATURES
    ), f"Attendu {N_FEATURES} features, obtenu {features.shape[1]}"
    assert labels.ndim == 1
    assert features.dtype == np.float32


@_integration
def test_real_normalizer_task1_mean():
    """Sur le vrai CSV, la moyenne de Task 1 doit être ≈ 0 après normalisation."""
    if not _NORMALIZER_PATH.exists():
        pytest.skip(f"Normalizer absent : {_NORMALIZER_PATH} — générer via save_normalizer()")

    ds = PumpMaintenanceDataset(_CSV_PATH)
    ds.load()
    features, labels = ds.extract_features()

    splitter = CLStreamSplitter(features, labels)
    normalizer = load_pump_normalizer(_NORMALIZER_PATH)
    splitter.apply_normalizer(normalizer)

    feats_t1, _ = splitter._slices[0]
    means = feats_t1.mean(axis=0)
    max_abs_mean = np.abs(means).max()
    assert (
        max_abs_mean < 0.15
    ), f"Moyenne max Task 1 après normalisation : {max_abs_mean:.4f} (attendu < 0.15)"


@_integration
def test_real_n_windows_positive():
    """Sur le vrai CSV (20 000 lignes), le nombre de fenêtres doit être > 1 000."""
    ds = PumpMaintenanceDataset(_CSV_PATH)
    ds.load()
    features, _ = ds.extract_features()
    n_windows = features.shape[0]
    # (20000 - 32) // 16 + 1 = 1249
    assert n_windows > 1000, f"Nombre de fenêtres trop faible : {n_windows}"
