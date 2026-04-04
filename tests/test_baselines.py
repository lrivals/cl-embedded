"""
tests/test_baselines.py — Tests unitaires pour src/training/baselines.py.

Vérifie :
    - La forme de acc_matrix pour train_naive_sequential et train_joint
    - La propriété borne supérieure : aa_joint >= aa_naive

Exécution :
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ewc import EWCMlpClassifier


def test_naive_acc_matrix_shape(cl_tasks, ewc_config):
    """acc_matrix doit être [T, T] avec NaN au-dessus de la diagonale."""
    from src.training.baselines import train_naive_sequential

    model = EWCMlpClassifier(input_dim=6)
    M = train_naive_sequential(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    assert M.shape == (T, T)
    assert np.isnan(M[0, 1])  # tâche 2 pas encore vue après task 1


def test_joint_acc_matrix_shape(cl_tasks, ewc_config):
    """Joint training : seule la dernière ligne est non-NaN."""
    from src.training.baselines import train_joint

    model = EWCMlpClassifier(input_dim=6)
    M = train_joint(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    assert M.shape == (T, T)
    assert not np.isnan(M[T - 1, :]).any()


def test_naive_diagonal_non_nan(cl_tasks, ewc_config):
    """La diagonale de acc_matrix (naive) doit être non-NaN."""
    from src.training.baselines import train_naive_sequential

    model = EWCMlpClassifier(input_dim=6)
    M = train_naive_sequential(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    for i in range(T):
        assert not np.isnan(M[i, i]), f"M[{i},{i}] est NaN"


def test_naive_upper_triangle_nan(cl_tasks, ewc_config):
    """Les entrées au-dessus de la diagonale doivent être NaN (tâche pas encore vue)."""
    from src.training.baselines import train_naive_sequential

    model = EWCMlpClassifier(input_dim=6)
    M = train_naive_sequential(model, cl_tasks, ewc_config)
    T = len(cl_tasks)
    for i in range(T):
        for j in range(i + 1, T):
            assert np.isnan(M[i, j]), f"M[{i},{j}] devrait être NaN"


def test_joint_aa_geq_naive_aa(cl_tasks, ewc_config):
    """Joint training doit avoir AA ≥ fine-tuning naïf (borne supérieure)."""
    from src.evaluation.metrics import compute_cl_metrics
    from src.training.baselines import train_joint, train_naive_sequential

    M_naive = train_naive_sequential(EWCMlpClassifier(input_dim=6), cl_tasks, ewc_config)
    M_joint = train_joint(EWCMlpClassifier(input_dim=6), cl_tasks, ewc_config)
    aa_naive = compute_cl_metrics(M_naive)["aa"]
    aa_joint = compute_cl_metrics(M_joint)["aa"]
    assert aa_joint >= aa_naive, f"Joint AA ({aa_joint:.3f}) < Naive AA ({aa_naive:.3f})"
