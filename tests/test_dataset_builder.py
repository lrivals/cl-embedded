"""
Tests pour scripts/board_dataset_builder.py (dry-run, sans board).

Valide la création des fichiers de sortie (CSV, results.json, config_snapshot.yaml)
et la conformité du format résultats avec les champs obligatoires Phase 1.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

# ── Imports du module sous test ──────────────────────────────────────────────
import importlib.util

_SCRIPT = Path(__file__).parent.parent / "scripts" / "board_dataset_builder.py"
spec = importlib.util.spec_from_file_location("board_dataset_builder", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

_save_csv              = _mod._save_csv
_compute_results_json  = _mod._compute_results_json

import argparse
import numpy as np
from datetime import datetime


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_results():
    """20 résultats fictifs couvrant 2 tâches."""
    rng = np.random.default_rng(0)
    results = []
    for i in range(20):
        task_id = i % 2
        label = int(rng.integers(0, 2))
        results.append({
            "task_id":       task_id,
            "ts_ms":         i * 100,
            "true":          label,
            "pred":          label if rng.random() > 0.1 else 1 - label,
            "confidence":    float(rng.uniform(0.5, 1.0)),
            "latency_us":    3,
            "ram_bytes":     200,
            "throughput_ips": 333333,
            "status":        0,
        })
    return results


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── Tests CSV ────────────────────────────────────────────────────────────────

def test_save_csv_creates_file(mock_results, output_dir):
    csv_path = output_dir / "dataset.csv"
    _save_csv(mock_results, csv_path)
    assert csv_path.exists()


def test_save_csv_correct_row_count(mock_results, output_dir):
    csv_path = output_dir / "dataset.csv"
    _save_csv(mock_results, csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(mock_results)


def test_save_csv_required_columns(mock_results, output_dir):
    csv_path = output_dir / "dataset.csv"
    _save_csv(mock_results, csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
    for col in ["task_id", "true", "pred", "latency_us", "ram_bytes"]:
        assert col in fieldnames, f"Colonne manquante : {col}"


def test_save_csv_empty(output_dir):
    csv_path = output_dir / "empty.csv"
    _save_csv([], csv_path)
    assert not csv_path.exists()


# ── Tests results.json ───────────────────────────────────────────────────────

def test_results_json_required_fields(mock_results, output_dir):
    """Les 6 champs obligatoires de evaluate_all.py doivent être présents."""
    results_json = _compute_results_json(mock_results, "cwru", "nucleo_f439zi", output_dir)
    required = ["acc_final", "avg_forgetting", "backward_transfer",
                "ram_peak_bytes", "inference_latency_ms", "n_params"]
    # Certains champs peuvent être None (renseignés par board_experiment_recorder)
    for field in required:
        assert field in results_json, f"Champ manquant : {field}"


def test_results_json_acc_range(mock_results, output_dir):
    results_json = _compute_results_json(mock_results, "monitoring", "nucleo_f439zi", output_dir)
    acc = results_json["acc_final"]
    assert 0.0 <= acc <= 1.0


def test_results_json_ram_positive(mock_results, output_dir):
    results_json = _compute_results_json(mock_results, "cwru", "nucleo_f439zi", output_dir)
    assert results_json["ram_peak_bytes"] > 0


def test_results_json_latency_ms(mock_results, output_dir):
    results_json = _compute_results_json(mock_results, "cwru", "nucleo_f439zi", output_dir)
    lat = results_json["inference_latency_ms"]
    assert lat > 0.0
    assert lat < 100.0  # Gap 2 : < 100 ms


def test_results_json_empty():
    results_json = _compute_results_json([], "cwru", "nucleo_f439zi", Path("/tmp/x"))
    assert results_json == {}
