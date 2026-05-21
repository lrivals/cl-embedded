"""
Tests pour scripts/board_experiment_recorder.py (dry-run, sans board).

Valide que le fichier results.json produit contient exactement les 6 champs
obligatoires de evaluate_all.py (acc_final, avg_forgetting, backward_transfer,
ram_peak_bytes, inference_latency_ms, n_params) et que les valeurs sont dans
des plages valides.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Chargement du module ─────────────────────────────────────────────────────
_SCRIPT = Path(__file__).parent.parent / "scripts" / "board_experiment_recorder.py"
spec = importlib.util.spec_from_file_location("board_experiment_recorder", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

_compute_per_task_acc  = _mod._compute_per_task_acc
_compute_forgetting    = _mod._compute_forgetting
_build_results_json    = _mod._build_results_json
_N_PARAMS              = _mod._N_PARAMS


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_results(n_tasks: int = 3, n_per_task: int = 20,
                  acc: float = 0.9, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for task_id in range(n_tasks):
        for _ in range(n_per_task):
            label = int(rng.integers(0, 2))
            pred  = label if rng.random() < acc else 1 - label
            results.append({
                "task_id":        task_id,
                "ts_ms":          len(results) * 10,
                "true":           label,
                "pred":           pred,
                "confidence":     float(rng.uniform(0.5, 1.0)),
                "latency_us":     3,
                "ram_bytes":      200,
                "throughput_ips": 333333,
                "status":         0,
            })
    return results


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "exp_test_01"


# ── Tests métriques CL ───────────────────────────────────────────────────────

def test_per_task_acc_correct():
    results = _make_results(n_tasks=2, n_per_task=100, acc=1.0)
    per_task = _compute_per_task_acc(results)
    for tid in [0, 1]:
        assert abs(per_task[tid] - 1.0) < 0.01


def test_per_task_acc_count():
    results = _make_results(n_tasks=3, n_per_task=10)
    per_task = _compute_per_task_acc(results)
    assert len(per_task) == 3


def test_forgetting_no_drift():
    per_task = {0: 0.9, 1: 0.9, 2: 0.9}
    af = _compute_forgetting(per_task)
    assert af == pytest.approx(0.0, abs=1e-6)


def test_forgetting_with_drift():
    per_task = {0: 0.9, 1: 0.7}  # task 1 a chuté vs task 0 peak
    af = _compute_forgetting(per_task)
    assert af > 0.0


def test_forgetting_single_task():
    per_task = {0: 0.85}
    af = _compute_forgetting(per_task)
    assert af == 0.0


# ── Tests format résultats ────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "acc_final", "avg_forgetting", "backward_transfer",
    "ram_peak_bytes", "inference_latency_ms", "n_params",
]


def test_required_fields_present(output_dir):
    results = _make_results()
    json_out = _build_results_json("mahalanobis", "cwru", "nucleo_f439zi",
                                    output_dir, results, 1.0, 3)
    for field in REQUIRED_FIELDS:
        assert field in json_out, f"Champ manquant : {field}"


def test_acc_final_in_range(output_dir):
    results = _make_results(acc=0.9)
    json_out = _build_results_json("mahalanobis", "cwru", "nucleo_f439zi",
                                    output_dir, results, 1.0, 3)
    assert 0.0 <= json_out["acc_final"] <= 1.0


def test_n_params_known_models(output_dir):
    for model in ["mahalanobis", "ewc", "tinyol"]:
        results = _make_results()
        json_out = _build_results_json(model, "monitoring", "nucleo_f439zi",
                                        output_dir, results, 1.0, 3)
        assert json_out["n_params"] == _N_PARAMS[model]


def test_gap2_compliant_normal_values(output_dir):
    results = _make_results()
    json_out = _build_results_json("mahalanobis", "cwru", "nucleo_f439zi",
                                    output_dir, results, 1.0, 3)
    assert json_out["gap2_ram_compliant"] is True
    assert json_out["gap2_latency_compliant"] is True


def test_empty_results(output_dir):
    json_out = _build_results_json("mahalanobis", "cwru", "nucleo_f439zi",
                                    output_dir, [], 0.0, 3)
    assert json_out == {}


def test_results_json_serializable(output_dir):
    results = _make_results()
    json_out = _build_results_json("ewc", "monitoring", "nucleo_f439zi",
                                    output_dir, results, 2.5, 3)
    # Doit être sérialisable sans erreur
    serialized = json.dumps(json_out)
    recovered = json.loads(serialized)
    assert recovered["model"] == "ewc"
