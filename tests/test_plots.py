"""Tests unitaires pour src/evaluation/plots.py (S2-06)."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_acc_matrix() -> np.ndarray:
    return np.array(
        [
            [0.91, np.nan, np.nan],
            [0.88, 0.85, np.nan],
            [0.86, 0.83, 0.89],
        ]
    )


def test_plot_accuracy_matrix_returns_figure(sample_acc_matrix):
    import matplotlib.pyplot as plt

    from src.evaluation.plots import plot_accuracy_matrix

    fig = plot_accuracy_matrix(sample_acc_matrix, task_names=["Pump", "Turbine", "Compressor"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_forgetting_curve_returns_figure(sample_acc_matrix):
    import matplotlib.pyplot as plt

    from src.evaluation.plots import plot_forgetting_curve

    fig = plot_forgetting_curve(sample_acc_matrix)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metrics_comparison_returns_figure():
    import matplotlib.pyplot as plt

    from src.evaluation.plots import plot_metrics_comparison

    results = {
        "EWC": {"aa": 0.98, "af": 0.001, "bwt": 0.0},
        "HDC": {"aa": 0.95, "af": 0.0, "bwt": 0.002},
    }
    fig = plot_metrics_comparison(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_figure_creates_file(sample_acc_matrix, tmp_path):
    from src.evaluation.plots import plot_accuracy_matrix, save_figure

    fig = plot_accuracy_matrix(sample_acc_matrix)
    out = tmp_path / "figures" / "test_matrix.png"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0
