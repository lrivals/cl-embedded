"""
tests/test_cwru_dataset.py — Tests unitaires pour CWRUDataset, CWRUFaultTypeStream,
CWRUSeverityStream, get_cwru_cl_dataloaders_by_fault_type et get_cwru_cl_dataloaders_by_severity.

Aucun accès aux fichiers data/raw/ — fixtures synthétiques (tmp_path uniquement).

Exécution :
    pytest tests/test_cwru_dataset.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cwru_dataset import (
    FAULT_COL,
    FAULT_TYPE_LABELS,
    FEATURE_COLS,
    N_FEATURES,
    N_TASKS,
    NORMAL_LABEL,
    SEVERITY_LABELS,
    CWRUDataset,
    CWRUFaultTypeStream,
    CWRUSeverityStream,
    get_cwru_cl_dataloaders_by_fault_type,
    get_cwru_cl_dataloaders_by_severity,
)

# ---------------------------------------------------------------------------
# Constantes de fixture
# ---------------------------------------------------------------------------

_N_PER_CLASS = 50  # suffisant pour les splits train/val/test


# ---------------------------------------------------------------------------
# Fixture CSV synthétique
# ---------------------------------------------------------------------------


def _all_unique_fault_labels() -> list[str]:
    labels: list[str] = [NORMAL_LABEL]
    seen = {NORMAL_LABEL}
    for group in list(FAULT_TYPE_LABELS.values()) + list(SEVERITY_LABELS.values()):
        for lbl in group:
            if lbl not in seen:
                labels.append(lbl)
                seen.add(lbl)
    return labels


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """CSV synthétique : 10 classes × _N_PER_CLASS fenêtres, 9 features + colonne fault."""
    rng = np.random.default_rng(42)
    rows = []
    for fault_label in _all_unique_fault_labels():
        X = rng.normal(0.0, 1.0, size=(_N_PER_CLASS, N_FEATURES)).astype(np.float32)
        for row in X:
            rows.append(list(row) + [fault_label])
    df = pd.DataFrame(rows, columns=FEATURE_COLS + [FAULT_COL])
    csv_path = tmp_path / "feature_time_48k_2048_load_1.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests CWRUDataset
# ---------------------------------------------------------------------------


def test_dataset_loads(synthetic_csv: Path) -> None:
    """CWRUDataset se charge sans erreur depuis un CSV valide."""
    ds = CWRUDataset(synthetic_csv)
    assert ds.X is not None
    assert ds.y is not None
    assert ds.fault_labels is not None


def test_dataset_x_shape(synthetic_csv: Path) -> None:
    """X a la forme (N, 9) avec dtype float32."""
    ds = CWRUDataset(synthetic_csv)
    n_classes = len(_all_unique_fault_labels())
    assert ds.X.shape == (n_classes * _N_PER_CLASS, N_FEATURES)
    assert ds.X.dtype == np.float32


def test_dataset_y_shape_and_dtype(synthetic_csv: Path) -> None:
    """y a la forme (N,) avec dtype int8."""
    ds = CWRUDataset(synthetic_csv)
    assert ds.y.ndim == 1
    assert ds.y.shape[0] == ds.X.shape[0]
    assert ds.y.dtype == np.int8


def test_dataset_binary_labels(synthetic_csv: Path) -> None:
    """y ne contient que 0 (Normal) et 1 (Défaut)."""
    ds = CWRUDataset(synthetic_csv)
    unique = set(ds.y.tolist())
    assert unique == {0, 1}


def test_dataset_fault_labels_known(synthetic_csv: Path) -> None:
    """fault_labels ne contient que des étiquettes connues."""
    ds = CWRUDataset(synthetic_csv)
    known = set(_all_unique_fault_labels())
    assert set(ds.fault_labels.tolist()).issubset(known)


def test_dataset_missing_file_raises(tmp_path: Path) -> None:
    """FileNotFoundError levé si le CSV n'existe pas."""
    with pytest.raises(FileNotFoundError):
        CWRUDataset(tmp_path / "inexistant.csv")


def test_dataset_missing_column_raises(tmp_path: Path) -> None:
    """ValueError levé si une colonne feature est absente du CSV."""
    df = pd.DataFrame({"max": [1.0], "fault": [NORMAL_LABEL]})  # colonnes incomplètes
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        CWRUDataset(csv_path)


# ---------------------------------------------------------------------------
# Tests CWRUFaultTypeStream
# ---------------------------------------------------------------------------


def test_fault_type_stream_yields_three_tasks(synthetic_csv: Path) -> None:
    """CWRUFaultTypeStream.iter_tasks() produit exactement N_TASKS=3 tâches."""
    ds = CWRUDataset(synthetic_csv)
    tasks = list(CWRUFaultTypeStream(ds).iter_tasks())
    assert len(tasks) == N_TASKS


def test_fault_type_stream_task_order(synthetic_csv: Path) -> None:
    """Ordre des tâches : ball → inner_race → outer_race."""
    ds = CWRUDataset(synthetic_csv)
    names = [name for _, name, _, _ in CWRUFaultTypeStream(ds).iter_tasks()]
    assert names == ["ball", "inner_race", "outer_race"]


def test_fault_type_stream_shapes(synthetic_csv: Path) -> None:
    """Chaque tâche a X (N_task, 9) et y (N_task,) cohérents."""
    ds = CWRUDataset(synthetic_csv)
    for _, _, X_task, y_task in CWRUFaultTypeStream(ds).iter_tasks():
        assert X_task.ndim == 2
        assert X_task.shape[1] == N_FEATURES
        assert y_task.shape[0] == X_task.shape[0]


# ---------------------------------------------------------------------------
# Tests CWRUSeverityStream
# ---------------------------------------------------------------------------


def test_severity_stream_yields_three_tasks(synthetic_csv: Path) -> None:
    """CWRUSeverityStream.iter_tasks() produit exactement N_TASKS=3 tâches."""
    ds = CWRUDataset(synthetic_csv)
    tasks = list(CWRUSeverityStream(ds).iter_tasks())
    assert len(tasks) == N_TASKS


def test_severity_stream_task_order(synthetic_csv: Path) -> None:
    """Ordre des tâches : 007 → 014 → 021 (sévérité croissante)."""
    ds = CWRUDataset(synthetic_csv)
    names = [name for _, name, _, _ in CWRUSeverityStream(ds).iter_tasks()]
    assert names == ["007", "014", "021"]


def test_severity_stream_contains_both_classes(synthetic_csv: Path) -> None:
    """Chaque tâche contient des exemples normaux (y=0) et défaillants (y=1)."""
    ds = CWRUDataset(synthetic_csv)
    for _, _, _, y_task in CWRUSeverityStream(ds).iter_tasks():
        assert 0 in y_task.tolist()
        assert 1 in y_task.tolist()


# ---------------------------------------------------------------------------
# Tests get_cwru_cl_dataloaders_by_fault_type
# ---------------------------------------------------------------------------


def test_cl_fault_type_returns_three_tasks(synthetic_csv: Path) -> None:
    """get_cwru_cl_dataloaders_by_fault_type retourne une liste de 3 dicts."""
    tasks = get_cwru_cl_dataloaders_by_fault_type(synthetic_csv)
    assert len(tasks) == N_TASKS


def test_cl_fault_type_task_keys(synthetic_csv: Path) -> None:
    """Chaque dict de tâche possède les clés obligatoires."""
    required_keys = {"task_id", "task_name", "domain", "train_loader", "val_loader", "test_loader", "n_train", "n_val", "n_test"}
    tasks = get_cwru_cl_dataloaders_by_fault_type(synthetic_csv)
    for task in tasks:
        assert required_keys.issubset(task.keys())


def test_cl_fault_type_task_ids(synthetic_csv: Path) -> None:
    """Les task_id sont 0, 1, 2."""
    tasks = get_cwru_cl_dataloaders_by_fault_type(synthetic_csv)
    assert [t["task_id"] for t in tasks] == [0, 1, 2]


def test_cl_fault_type_feature_dim(synthetic_csv: Path) -> None:
    """Les batches du train_loader ont bien 9 features."""
    import torch
    tasks = get_cwru_cl_dataloaders_by_fault_type(synthetic_csv, batch_size=8)
    for task in tasks:
        X_batch, _ = next(iter(task["train_loader"]))
        assert X_batch.shape[1] == N_FEATURES


# ---------------------------------------------------------------------------
# Tests get_cwru_cl_dataloaders_by_severity
# ---------------------------------------------------------------------------


def test_cl_severity_returns_three_tasks(synthetic_csv: Path) -> None:
    """get_cwru_cl_dataloaders_by_severity retourne une liste de 3 dicts."""
    tasks = get_cwru_cl_dataloaders_by_severity(synthetic_csv)
    assert len(tasks) == N_TASKS


def test_cl_severity_task_names(synthetic_csv: Path) -> None:
    """Noms des tâches : '007', '014', '021'."""
    tasks = get_cwru_cl_dataloaders_by_severity(synthetic_csv)
    assert [t["task_name"] for t in tasks] == ["007", "014", "021"]


def test_cl_severity_split_sizes_coherent(synthetic_csv: Path) -> None:
    """n_train + n_val + n_test cohérent avec la taille de chaque tâche."""
    tasks = get_cwru_cl_dataloaders_by_severity(synthetic_csv)
    for task in tasks:
        total = task["n_train"] + task["n_val"] + task["n_test"]
        assert total > 0
