"""
tests/test_metrics.py — Tests unitaires pour les métriques CL.

Exécution :
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    accuracy_binary,
    compute_cl_metrics,
    format_metrics_report,
)


# ------------------------------------------------------------------
# Matrices de test (cas connus)
# ------------------------------------------------------------------

# Cas idéal : aucun oubli, transfert positif
MATRIX_NO_FORGETTING = np.array([
    [0.90, np.nan, np.nan],
    [0.90, 0.88,  np.nan],
    [0.90, 0.88,  0.91 ],
])

# Cas oubli catastrophique : performance s'effondre sur les tâches passées
MATRIX_CATASTROPHIC = np.array([
    [0.90, np.nan, np.nan],
    [0.50, 0.88,  np.nan],
    [0.30, 0.45,  0.92 ],
])

# Cas réaliste EWC : oubli modéré
MATRIX_EWC_REALISTIC = np.array([
    [0.91, np.nan, np.nan],
    [0.88, 0.85,  np.nan],
    [0.86, 0.83,  0.89 ],
])


# ------------------------------------------------------------------
# Tests de base
# ------------------------------------------------------------------

def test_aa_no_forgetting():
    """AA = moyenne de la dernière ligne (tous cas sans NaN)."""
    m = compute_cl_metrics(MATRIX_NO_FORGETTING)
    expected_aa = np.mean([0.90, 0.88, 0.91])
    assert abs(m["aa"] - expected_aa) < 1e-6


def test_af_no_forgetting():
    """AF = 0 si aucun oubli."""
    m = compute_cl_metrics(MATRIX_NO_FORGETTING)
    assert abs(m["af"]) < 1e-6, f"AF devrait être 0, obtenu {m['af']}"


def test_af_catastrophic():
    """AF élevé en cas d'oubli catastrophique."""
    m = compute_cl_metrics(MATRIX_CATASTROPHIC)
    assert m["af"] > 0.2, f"AF devrait être > 0.2, obtenu {m['af']}"


def test_bwt_no_forgetting():
    """BWT = 0 si les performances sur les tâches passées ne changent pas."""
    m = compute_cl_metrics(MATRIX_NO_FORGETTING)
    assert abs(m["bwt"]) < 1e-6


def test_bwt_negative_catastrophic():
    """BWT < 0 en cas d'oubli (performances régressent)."""
    m = compute_cl_metrics(MATRIX_CATASTROPHIC)
    assert m["bwt"] < 0


def test_ewc_realistic_metrics():
    """Test sur une matrice réaliste EWC."""
    m = compute_cl_metrics(MATRIX_EWC_REALISTIC)
    assert 0.80 < m["aa"] < 0.95, f"AA inattendu : {m['aa']}"
    assert 0.0 < m["af"] < 0.1, f"AF inattendu : {m['af']}"
    assert m["bwt"] < 0, f"BWT doit être négatif : {m['bwt']}"


def test_n_tasks():
    """n_tasks = T dans la matrice."""
    m = compute_cl_metrics(MATRIX_EWC_REALISTIC)
    assert m["n_tasks"] == 3


def test_single_task():
    """Avec une seule tâche, AF et BWT sont 0."""
    M = np.array([[0.85]])
    m = compute_cl_metrics(M)
    assert m["af"] == 0.0
    assert m["bwt"] == 0.0
    assert abs(m["aa"] - 0.85) < 1e-6


# ------------------------------------------------------------------
# Tests accuracy_binary
# ------------------------------------------------------------------

def test_accuracy_binary_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    assert accuracy_binary(y_true, y_pred) == 1.0


def test_accuracy_binary_threshold():
    y_true = np.array([0, 1])
    y_pred = np.array([0.4, 0.6])
    assert accuracy_binary(y_true, y_pred, threshold=0.5) == 1.0
    assert accuracy_binary(y_true, y_pred, threshold=0.7) == 0.5


# ------------------------------------------------------------------
# Test format rapport
# ------------------------------------------------------------------

def test_format_report_runs():
    """Le rapport se génère sans erreur."""
    m = compute_cl_metrics(MATRIX_EWC_REALISTIC)
    report = format_metrics_report(m, model_name="EWC Online")
    assert "AA" in report
    assert "AF" in report
    assert "BWT" in report
    assert "EWC Online" in report
