"""
tests/test_memory_profiler.py — Tests unitaires pour le profiler mémoire CL.

Exécution :
    pytest tests/test_memory_profiler.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.evaluation.memory_profiler import (
    compare_models_memory,
    full_memory_report,
    profile_cl_update,
    profile_forward_pass,
)
from src.models.ewc.ewc_mlp import EWCMlpClassifier


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def ewc_model() -> EWCMlpClassifier:
    return EWCMlpClassifier(input_dim=6)


@pytest.fixture
def tiny_model() -> nn.Module:
    """MLP minimal pour les tests rapides."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())


# ------------------------------------------------------------------
# Tests profile_forward_pass
# ------------------------------------------------------------------


def test_forward_pass_returns_expected_keys(ewc_model):
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=10)
    expected_keys = {
        "ram_peak_bytes",
        "ram_current_bytes",
        "inference_latency_ms",
        "inference_latency_std_ms",
        "n_params",
        "params_fp32_bytes",
        "params_int8_bytes",
        "within_budget_64ko",
    }
    assert expected_keys.issubset(result.keys())


def test_forward_pass_within_budget_ewc(ewc_model):
    """EWCMlpClassifier (769 params, ~3Ko FP32) doit tenir dans 64Ko."""
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=10)
    assert result["within_budget_64ko"] is True


def test_forward_pass_ram_peak_positive(ewc_model):
    """ram_peak_bytes doit être > 0 pour un vrai forward pass."""
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=5)
    assert result["ram_peak_bytes"] > 0


def test_forward_pass_n_params_ewc(ewc_model):
    """EWCMlpClassifier avec input_dim=6, hidden=[32,16] doit avoir 769 params."""
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=5)
    assert result["n_params"] == 769


def test_forward_pass_params_bytes_consistency(ewc_model):
    """params_fp32_bytes = n_params * 4, params_int8_bytes = n_params * 1."""
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=5)
    assert result["params_fp32_bytes"] == result["n_params"] * 4
    assert result["params_int8_bytes"] == result["n_params"] * 1


def test_forward_pass_latency_positive(ewc_model):
    result = profile_forward_pass(ewc_model, (1, 6), n_runs=20)
    assert result["inference_latency_ms"] > 0
    assert result["inference_latency_std_ms"] >= 0


def test_forward_pass_tiny_model(tiny_model):
    result = profile_forward_pass(tiny_model, (1, 4), n_runs=5)
    assert result["within_budget_64ko"] is True


# ------------------------------------------------------------------
# Tests profile_cl_update
# ------------------------------------------------------------------


def test_cl_update_returns_expected_keys(ewc_model):
    optimizer = torch.optim.SGD(ewc_model.parameters(), lr=0.01)

    def update_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy(ewc_model(x), y)
        loss.backward()
        optimizer.step()
        return loss.item()

    result = profile_cl_update(update_fn, (1, 6), label_shape=(1, 1), n_runs=10)
    expected_keys = {
        "ram_peak_bytes_update",
        "update_latency_ms",
        "update_latency_std_ms",
        "within_budget_64ko_update",
    }
    assert expected_keys.issubset(result.keys())


def test_cl_update_within_budget(ewc_model):
    optimizer = torch.optim.SGD(ewc_model.parameters(), lr=0.01)

    def update_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy(ewc_model(x), y)
        loss.backward()
        optimizer.step()
        return loss.item()

    result = profile_cl_update(update_fn, (1, 6), label_shape=(1, 1), n_runs=10)
    assert result["within_budget_64ko_update"] is True


def test_cl_update_latency_positive(ewc_model):
    optimizer = torch.optim.SGD(ewc_model.parameters(), lr=0.01)

    def update_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy(ewc_model(x), y)
        loss.backward()
        optimizer.step()
        return loss.item()

    result = profile_cl_update(update_fn, (1, 6), label_shape=(1, 1), n_runs=10)
    assert result["update_latency_ms"] > 0


# ------------------------------------------------------------------
# Tests full_memory_report
# ------------------------------------------------------------------


def test_full_memory_report_returns_dict(ewc_model, capsys):
    report = full_memory_report(ewc_model, (1, 6), model_name="EWCMlpClassifier")
    assert isinstance(report, dict)
    assert "model" in report
    assert "forward" in report
    assert report["model"] == "EWCMlpClassifier"


def test_full_memory_report_stdout(ewc_model, capsys):
    full_memory_report(ewc_model, (1, 6), model_name="EWCTest")
    captured = capsys.readouterr()
    assert "EWCTest" in captured.out
    assert "Paramètres" in captured.out
    assert "RAM" in captured.out


def test_full_memory_report_with_update_fn(ewc_model):
    optimizer = torch.optim.SGD(ewc_model.parameters(), lr=0.01)

    def update_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy(ewc_model(x), y)
        loss.backward()
        optimizer.step()
        return loss.item()

    report = full_memory_report(
        ewc_model, (1, 6), update_fn=update_fn, model_name="EWC+Update"
    )
    assert "update" in report
    assert "ram_peak_bytes_update" in report["update"]


# ------------------------------------------------------------------
# Tests compare_models_memory
# ------------------------------------------------------------------


def test_compare_models_memory_returns_string(ewc_model, tiny_model):
    r1 = full_memory_report(ewc_model, (1, 6), model_name="EWC MLP")
    r2 = full_memory_report(tiny_model, (1, 4), model_name="Tiny MLP")
    table = compare_models_memory([r1, r2])
    assert isinstance(table, str)
    assert "EWC MLP" in table
    assert "Tiny MLP" in table


def test_compare_models_memory_single_model(ewc_model):
    r = full_memory_report(ewc_model, (1, 6), model_name="Solo")
    table = compare_models_memory([r])
    assert "Solo" in table
