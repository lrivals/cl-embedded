"""
tests/test_cwru_anomaly.py — Tests unitaires pour get_cwru_dataloaders_anomaly_detection().

Valide le comportement du loader anomaly detection CWRU :
    - structure de retour (3 tâches, 0-indexées)
    - train_loader contient uniquement des données normales
    - test_loader_mixed contient les deux classes
    - la dimension d'entrée est 9
    - les deux scénarios (by_severity, by_fault_type) fonctionnent
    - UserWarning émis si < 100 échantillons normaux d'entraînement

Aucun accès aux fichiers data/raw/ — fixtures synthétiques (tmp_path uniquement).

Exécution :
    pytest tests/test_cwru_anomaly.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.cwru_dataset import (
    FAULT_COL,
    FAULT_TYPE_LABELS,
    FEATURE_COLS,
    N_FEATURES,
    NORMAL_LABEL,
    SEVERITY_LABELS,
    get_cwru_dataloaders_anomaly_detection,
)

# ---------------------------------------------------------------------------
# Constantes de fixture
# ---------------------------------------------------------------------------

_N_WINDOWS_PER_CLASS = 230  # identique au vrai dataset


# ---------------------------------------------------------------------------
# Fixture CSV synthétique
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """
    CSV synthétique reproduisant la structure du vrai CWRU CSV.

    2 300 lignes × (9 features + colonne fault) :
    - 230 lignes Normal_1
    - 230 lignes par classe faulty (9 classes)
    """
    rng = np.random.default_rng(42)

    all_fault_labels = [NORMAL_LABEL] + [
        label
        for labels in list(FAULT_TYPE_LABELS.values()) + list(SEVERITY_LABELS.values())
        for label in labels
        if label not in [NORMAL_LABEL]
    ]
    # Dédupliquer tout en conservant l'ordre
    seen: set[str] = set()
    unique_labels: list[str] = []
    for lbl in all_fault_labels:
        if lbl not in seen:
            unique_labels.append(lbl)
            seen.add(lbl)

    rows = []
    for fault_label in unique_labels:
        X = rng.normal(0.0, 1.0, size=(_N_WINDOWS_PER_CLASS, N_FEATURES)).astype(np.float32)
        for row in X:
            rows.append(list(row) + [fault_label])

    df = pd.DataFrame(rows, columns=FEATURE_COLS + [FAULT_COL])
    csv_path = tmp_path / "feature_time_48k_2048_load_1.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_three_tasks(synthetic_csv: Path) -> None:
    """get_cwru_dataloaders_anomaly_detection retourne exactement 3 tâches."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    assert len(tasks) == 3


def test_task_ids_are_0_1_2(synthetic_csv: Path) -> None:
    """Les task_id sont 0, 1, 2 (convention 0-indexed CWRU)."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    assert [t["task_id"] for t in tasks] == [0, 1, 2]


def test_train_loader_only_normal(synthetic_csv: Path) -> None:
    """Le train_loader ne contient que des échantillons normaux (y == 0) sur 9 features."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    for task in tasks:
        for X_batch, y_batch in task["train_loader"]:
            assert X_batch.shape[1] == N_FEATURES
            assert y_batch.max().item() == 0.0, (
                f"task {task['task_id']}: train_loader contient des labels faulty (y=1)"
            )


def test_test_loader_has_both_classes(synthetic_csv: Path) -> None:
    """Le test_loader_mixed contient à la fois des normaux (0) et des faulty (1)."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    for task in tasks:
        all_labels: list[float] = []
        for _X_batch, y_batch in task["test_loader_mixed"]:
            all_labels.extend(y_batch.tolist())
        unique = set(all_labels)
        assert 0.0 in unique, f"task {task['task_id']}: aucun normal dans test_loader_mixed"
        assert 1.0 in unique, f"task {task['task_id']}: aucun faulty dans test_loader_mixed"


def test_input_dim_is_9(synthetic_csv: Path) -> None:
    """Les tenseurs X ont bien 9 features, dtype float32."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    for task in tasks:
        for X_batch, _ in task["train_loader"]:
            assert X_batch.shape[1] == N_FEATURES
            assert X_batch.dtype == torch.float32
            break
        for X_batch, _ in task["test_loader_mixed"]:
            assert X_batch.shape[1] == N_FEATURES
            break


def test_both_scenarios_work(synthetic_csv: Path) -> None:
    """Les deux scénarios by_severity et by_fault_type retournent 3 tâches valides."""
    for scenario in ("by_severity", "by_fault_type"):
        tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv, scenario=scenario)
        assert len(tasks) == 3, f"scenario={scenario!r} : attendu 3 tâches"
        for task in tasks:
            assert task["n_train"] > 0
            assert task["n_test_faulty"] > 0


def test_warning_on_few_normal_samples(synthetic_csv: Path) -> None:
    """UserWarning émis car < 100 échantillons normaux par tâche (~62 avec 230 normaux total)."""
    with pytest.warns(UserWarning, match="échantillons normaux d'entraînement"):
        get_cwru_dataloaders_anomaly_detection(synthetic_csv)


def test_n_test_faulty_consistent(synthetic_csv: Path) -> None:
    """n_test_faulty correspond au nombre de fenêtres faulty dans test_loader_mixed."""
    tasks = get_cwru_dataloaders_anomaly_detection(synthetic_csv)
    for task in tasks:
        n_faulty_counted = 0
        for _, y_batch in task["test_loader_mixed"]:
            n_faulty_counted += int(y_batch.sum().item())
        assert n_faulty_counted == task["n_test_faulty"], (
            f"task {task['task_id']}: n_test_faulty déclaré={task['n_test_faulty']}, "
            f"compté={n_faulty_counted}"
        )


def test_invalid_scenario_raises(synthetic_csv: Path) -> None:
    """Un scénario inconnu lève ValueError."""
    with pytest.raises(ValueError, match="scenario"):
        get_cwru_dataloaders_anomaly_detection(synthetic_csv, scenario="unknown")  # type: ignore[arg-type]
