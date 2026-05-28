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


# ── Tests intégration CLI (subprocess dry-run) ───────────────────────────────

import subprocess
import sys

_REPO_ROOT = Path(__file__).parent.parent
_RECORDER_CLI = _REPO_ROOT / "scripts" / "board_experiment_recorder.py"
_DATA_ROOTS = {
    "cwru":       _REPO_ROOT / "data" / "raw" / "cwru",
    "monitoring": _REPO_ROOT / "data" / "raw" / "equipment_monitoring",
}

REQUIRED_KEYS = {
    "acc_final", "avg_forgetting", "backward_transfer",
    "ram_peak_bytes", "inference_latency_ms", "n_params",
}
BOARD_KEYS = {
    "exp_id", "model", "dataset", "platform",
    "date", "n_tasks", "n_samples_total", "config_snapshot",
}


def _dataset_available(dataset: str) -> bool:
    p = _DATA_ROOTS.get(dataset, Path("/nonexistent"))
    try:
        return p.is_dir() and any(p.iterdir())
    except PermissionError:
        return False


def _run_recorder_cli(model: str, dataset: str, output: Path) -> Path:
    result = subprocess.run(
        [sys.executable, str(_RECORDER_CLI),
         "--model", model,
         "--dataset", dataset,
         "--dry-run",
         "--output", str(output)],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"recorder CLI failed:\n{result.stderr}"
    return output / "results.json"


class TestDryRunOutput:

    def test_json_file_created(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent — test nécessite les données CWRU")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        assert json_path.exists(), "results.json non créé"

    def test_json_is_valid(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert isinstance(data, dict)

    def test_required_keys_present(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        missing = REQUIRED_KEYS - data.keys()
        assert not missing, f"Clés obligatoires manquantes : {missing}"

    def test_board_keys_present(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        missing = BOARD_KEYS - data.keys()
        assert not missing, f"Clés board manquantes : {missing}"

    def test_metric_types(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert isinstance(data["acc_final"], float)
        assert isinstance(data["avg_forgetting"], float)
        assert isinstance(data["backward_transfer"], float)
        assert isinstance(data["ram_peak_bytes"], (int, float))
        assert isinstance(data["inference_latency_ms"], float)
        assert isinstance(data["n_params"], int)

    def test_metric_ranges(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert 0.0 <= data["acc_final"] <= 1.0
        assert data["avg_forgetting"] >= 0.0
        assert data["ram_peak_bytes"] > 0
        assert data["inference_latency_ms"] > 0.0
        assert data["n_params"] > 0

    def test_n_params_mahalanobis(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["n_params"] == 30

    def test_n_params_ewc(self, tmp_path):
        if not _dataset_available("monitoring"):
            pytest.skip("data/raw/equipment_monitoring absent")
        json_path = _run_recorder_cli("ewc", "monitoring", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["n_params"] == 1538

    def test_config_snapshot_exists(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        out = tmp_path / "exp"
        _run_recorder_cli("mahalanobis", "cwru", out)
        assert (out / "config_snapshot.yaml").exists(), "config_snapshot.yaml non copié"

    def test_platform_field(self, tmp_path):
        if not _dataset_available("cwru"):
            pytest.skip("data/raw/cwru absent")
        json_path = _run_recorder_cli("mahalanobis", "cwru", tmp_path / "exp")
        data = json.loads(json_path.read_text())
        assert data["platform"] == "nucleo_f439zi"

    @pytest.mark.parametrize("model,dataset", [
        ("mahalanobis", "cwru"),
        ("ewc", "monitoring"),
        ("tinyol", "cwru"),
    ])
    def test_all_models_dry_run(self, tmp_path, model, dataset):
        if not _dataset_available(dataset):
            pytest.skip(f"data/raw/{dataset} absent")
        json_path = _run_recorder_cli(model, dataset, tmp_path / f"exp_{model}")
        data = json.loads(json_path.read_text())
        assert REQUIRED_KEYS <= data.keys()
