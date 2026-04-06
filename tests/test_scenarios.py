"""
tests/test_scenarios.py — Tests unitaires pour src/training/scenarios.py.

Fixtures utilisées : hdc_model_mock, cl_tasks (définies dans conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.scenarios import evaluate_task_generic, run_cl_scenario


def test_acc_matrix_shape(hdc_model_mock, cl_tasks):
    """acc_matrix doit être [T, T] avec NaN au-dessus de la diagonale."""
    acc = run_cl_scenario(hdc_model_mock, cl_tasks, {})
    T = len(cl_tasks)
    assert acc.shape == (T, T)
    assert np.isnan(acc[0, 1])  # tâche 2 pas encore vue après task 1
    assert not np.isnan(acc[T - 1, 0])  # tâche 1 vue depuis la fin


def test_evaluate_task_generic_range(hdc_model_mock, cl_tasks):
    """accuracy doit être dans [0, 1]."""
    acc = evaluate_task_generic(hdc_model_mock, cl_tasks[0]["val_loader"])
    assert 0.0 <= acc <= 1.0


def test_compatible_with_compute_cl_metrics(hdc_model_mock, cl_tasks):
    """La matrice doit être compatible avec compute_cl_metrics()."""
    from src.evaluation.metrics import compute_cl_metrics

    acc = run_cl_scenario(hdc_model_mock, cl_tasks, {})
    metrics = compute_cl_metrics(acc)
    assert "aa" in metrics
    assert "af" in metrics
    assert "bwt" in metrics
