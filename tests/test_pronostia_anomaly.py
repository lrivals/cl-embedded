"""
tests/test_pronostia_anomaly.py — Tests unitaires pour get_pronostia_dataloaders_anomaly_detection().

Valide le comportement du loader anomaly detection Pronostia :
    - structure de retour (3 tâches, 1-indexed)
    - train_loader contient uniquement des données normales
    - test_loader_mixed contient les deux classes
    - la dimension d'entrée est 13
    - le failure_ratio influe sur la proportion de données faulty en test

Aucun accès aux fichiers data/raw/ — fixtures synthétiques (tmp_path uniquement).

Exécution :
    pytest tests/test_pronostia_anomaly.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.pronostia_dataset import (
    COL_ACC_HORIZ,
    COL_ACC_VERT,
    N_FEATURES,
    WINDOW_SIZE,
    fit_normalizer,
    get_pronostia_dataloaders_anomaly_detection,
    load_condition_features,
    save_normalizer,
)

# ---------------------------------------------------------------------------
# Constantes de fixture
# ---------------------------------------------------------------------------

_N_WINDOWS = 40  # fenêtres synthétiques par roulement — plus grand que défaut pour tests stables


# ---------------------------------------------------------------------------
# Fixtures synthétiques
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_npy_dir(tmp_path: Path) -> Path:
    """Répertoire avec 6 fichiers .npy synthétiques au format PRONOSTIA."""
    rng = np.random.default_rng(42)
    npy_dir = tmp_path / "binaries"
    npy_dir.mkdir()

    for bearing_idx in range(6):
        n_windows = _N_WINDOWS + bearing_idx
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
    feats, _ = load_condition_features(synthetic_npy_dir, condition=1)
    normalizer = fit_normalizer(feats)
    norm_path = tmp_path / "pronostia_normalizer.yaml"
    save_normalizer(norm_path, normalizer)
    return norm_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_three_tasks(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """get_pronostia_dataloaders_anomaly_detection retourne exactement 3 tâches."""
    tasks = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        failure_ratio=0.10,
        seed=42,
    )
    assert len(tasks) == 3


def test_task_ids_are_ordered(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """Les task_id sont ordonnés 1, 2, 3 (convention 1-indexed)."""
    tasks = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
    )
    task_ids = [t["task_id"] for t in tasks]
    assert task_ids == [1, 2, 3]


def test_train_loader_only_normal(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """Le train_loader ne contient que des échantillons normaux (y == 0) sur 13 features."""
    tasks = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        failure_ratio=0.10,
    )
    for task in tasks:
        for X_batch, y_batch in task["train_loader"]:
            assert X_batch.shape[1] == N_FEATURES
            assert y_batch.max().item() == 0.0, "train_loader contient des labels faulty (y=1)"


def test_test_loader_has_both_classes(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """Le test_loader_mixed contient à la fois des normaux (0) et des faulty (1)."""
    tasks = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        failure_ratio=0.10,
    )
    for task in tasks:
        all_labels: list[float] = []
        for _X_batch, y_batch in task["test_loader_mixed"]:
            all_labels.extend(y_batch.tolist())
        unique = set(all_labels)
        assert 0.0 in unique, f"task {task['task_id']}: aucun échantillon normal dans test_loader_mixed"
        assert 1.0 in unique, f"task {task['task_id']}: aucun échantillon faulty dans test_loader_mixed"


def test_failure_ratio_affects_split(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """Un failure_ratio plus élevé produit davantage de données faulty dans les test loaders."""
    tasks_10 = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        failure_ratio=0.10,
    )
    tasks_20 = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
        failure_ratio=0.20,
    )

    def count_faulty(tasks: list[dict]) -> int:
        total = 0
        for task in tasks:
            for _X, y in task["test_loader_mixed"]:
                total += int(y.sum().item())
        return total

    n_faulty_10 = count_faulty(tasks_10)
    n_faulty_20 = count_faulty(tasks_20)
    assert n_faulty_20 > n_faulty_10, (
        f"failure_ratio=0.20 devrait produire plus de faulty ({n_faulty_20}) "
        f"que failure_ratio=0.10 ({n_faulty_10})"
    )


def test_input_dim_is_13(
    synthetic_npy_dir: Path, synthetic_normalizer_path: Path
) -> None:
    """Les tenseurs X ont bien 13 features (6 stats × 2 canaux + position temporelle)."""
    tasks = get_pronostia_dataloaders_anomaly_detection(
        npy_dir=synthetic_npy_dir,
        normalizer_path=synthetic_normalizer_path,
    )
    for task in tasks:
        for X_batch, _ in task["train_loader"]:
            assert X_batch.shape[1] == N_FEATURES
            assert X_batch.dtype == torch.float32
            break
        for X_batch, _ in task["test_loader_mixed"]:
            assert X_batch.shape[1] == N_FEATURES
            break
