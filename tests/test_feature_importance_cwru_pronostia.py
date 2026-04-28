"""
tests/test_feature_importance_cwru_pronostia.py — Tests S11-21.

Valide les constantes CWRU/Pronostia, resolve_feature_names(), et les fonctions
d'importance sur données synthétiques sans accès aux datasets réels.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.feature_importance import (
    CHANNEL_GROUPS_PRONOSTIA,
    FEATURE_NAMES_CWRU,
    FEATURE_NAMES_MONITORING,
    FEATURE_NAMES_PRONOSTIA,
    feature_masking_importance,
    permutation_importance,
    permutation_importance_per_task,
    resolve_feature_names,
)


# ── A. Constantes ─────────────────────────────────────────────────────────────


def test_cwru_feature_names_length() -> None:
    assert len(FEATURE_NAMES_CWRU) == 9


def test_pronostia_feature_names_length() -> None:
    assert len(FEATURE_NAMES_PRONOSTIA) == 13


def test_channel_groups_cover_all_pronostia_features() -> None:
    all_in_groups = {f for feats in CHANNEL_GROUPS_PRONOSTIA.values() for f in feats}
    assert all_in_groups == set(FEATURE_NAMES_PRONOSTIA)


# ── B. resolve_feature_names ──────────────────────────────────────────────────


def test_resolve_monitoring() -> None:
    assert resolve_feature_names("monitoring") is FEATURE_NAMES_MONITORING


def test_resolve_cwru() -> None:
    assert resolve_feature_names("cwru") is FEATURE_NAMES_CWRU


def test_resolve_pronostia() -> None:
    assert resolve_feature_names("pronostia") is FEATURE_NAMES_PRONOSTIA


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_feature_names("pump")


# ── C. permutation_importance — CWRU ─────────────────────────────────────────


def test_permutation_importance_cwru_returns_all_features(
    cwru_synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = cwru_synthetic_data
    predict_fn = lambda arr: (arr[:, 4] > 0.5).astype(float)  # noqa: E731
    result = permutation_importance(predict_fn, X, y, FEATURE_NAMES_CWRU, n_repeats=3)
    assert set(result.keys()) == set(FEATURE_NAMES_CWRU)


def test_permutation_importance_cwru_sorted_descending(
    cwru_synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = cwru_synthetic_data
    predict_fn = lambda arr: (arr[:, 4] > 0.5).astype(float)  # noqa: E731
    result = permutation_importance(predict_fn, X, y, FEATURE_NAMES_CWRU, n_repeats=3)
    values = list(result.values())
    assert values == sorted(values, reverse=True)


def test_permutation_importance_cwru_top_feature_is_rms(
    cwru_synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """La feature 'rms' (col 4) pilote les labels → doit avoir la plus haute importance."""
    X, y = cwru_synthetic_data
    predict_fn = lambda arr: (arr[:, 4] > 0.5).astype(float)  # noqa: E731
    result = permutation_importance(predict_fn, X, y, FEATURE_NAMES_CWRU, n_repeats=5)
    top_feature = next(iter(result))
    assert top_feature == "rms"


# ── D. permutation_importance_per_task — Pronostia ───────────────────────────


def test_per_task_pronostia_returns_three_tasks(
    pronostia_synthetic_tasks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    tasks = [
        {"task_name": name, "X": X, "y": y}
        for name, (X, y) in pronostia_synthetic_tasks.items()
    ]
    predict_fn = lambda arr: (arr[:, 2] > 1.5).astype(float)  # noqa: E731
    result = permutation_importance_per_task(predict_fn, tasks, FEATURE_NAMES_PRONOSTIA, n_repeats=3)
    assert set(result.keys()) == {"condition_1", "condition_2", "condition_3"}


def test_per_task_pronostia_all_features_in_each_task(
    pronostia_synthetic_tasks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    tasks = [
        {"task_name": name, "X": X, "y": y}
        for name, (X, y) in pronostia_synthetic_tasks.items()
    ]
    predict_fn = lambda arr: (arr[:, 2] > 1.5).astype(float)  # noqa: E731
    result = permutation_importance_per_task(predict_fn, tasks, FEATURE_NAMES_PRONOSTIA, n_repeats=3)
    for task_scores in result.values():
        assert set(task_scores.keys()) == set(FEATURE_NAMES_PRONOSTIA)


def test_per_task_consistency_with_global(
    pronostia_synthetic_tasks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """La moyenne des scores par tâche doit être cohérente avec le score global (±20 pp)."""
    all_X = np.vstack([X for X, _ in pronostia_synthetic_tasks.values()])
    all_y = np.concatenate([y for _, y in pronostia_synthetic_tasks.values()])
    tasks = [
        {"task_name": name, "X": X, "y": y}
        for name, (X, y) in pronostia_synthetic_tasks.items()
    ]
    predict_fn = lambda arr: (arr[:, 2] > 1.5).astype(float)  # noqa: E731

    global_scores = permutation_importance(predict_fn, all_X, all_y, FEATURE_NAMES_PRONOSTIA, n_repeats=3)
    per_task_scores = permutation_importance_per_task(predict_fn, tasks, FEATURE_NAMES_PRONOSTIA, n_repeats=3)

    for feat in FEATURE_NAMES_PRONOSTIA:
        avg_per_task = float(np.mean([per_task_scores[t][feat] for t in per_task_scores]))
        assert abs(avg_per_task - global_scores[feat]) < 0.20, (
            f"Feature '{feat}': avg_per_task={avg_per_task:.4f}, global={global_scores[feat]:.4f}"
        )


# ── E. feature_masking_importance ─────────────────────────────────────────────


def test_masking_cwru_returns_correct_keys(
    cwru_synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = cwru_synthetic_data
    predict_fn = lambda arr: (arr[:, 4] > 0.5).astype(float)  # noqa: E731
    result = feature_masking_importance(predict_fn, X, y, FEATURE_NAMES_CWRU)
    assert set(result.keys()) == set(FEATURE_NAMES_CWRU)


def test_masking_pronostia_dominant_feature_detected(
    pronostia_synthetic_tasks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """rms_acc_horiz (col 2) pilote les labels → son masquage doit dégrader les perf."""
    X, y = pronostia_synthetic_tasks["condition_1"]
    predict_fn = lambda arr: (arr[:, 2] > 1.5).astype(float)  # noqa: E731
    result = feature_masking_importance(predict_fn, X, y, FEATURE_NAMES_PRONOSTIA)
    assert result["rms_acc_horiz"] > 0.0
