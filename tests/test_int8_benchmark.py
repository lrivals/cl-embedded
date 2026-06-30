"""
tests/test_int8_benchmark.py — Tests du benchmark INT8 vs FP32 (S2801).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_int8_fp32", _ROOT / "scripts" / "benchmark_int8_fp32.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()


# ----------------------------------------------------------------------------
# Schéma JSON
# ----------------------------------------------------------------------------
def test_build_result_dict_schema():
    res = bench.build_result_dict(
        model_name="ewc",
        dataset="monitoring",
        config_path="configs/ewc_int8_monitoring.yaml",
        metric_name="auroc",
        fp32_metric=0.912,
        fp32_ram=9728,
        fp32_latency=0.045,
        int8_metric=0.899,
        int8_ram=2432,
        int8_latency=0.051,
    )
    for key in (
        "model", "dataset", "config_path", "timestamp",
        "fp32", "int8", "delta_metric", "ram_ratio",
        "gap3_metric_ok", "gap3_ram_ok",
    ):
        assert key in res
    for sub in ("metric_name", "metric_value", "ram_bytes", "latency_ms"):
        assert sub in res["fp32"]
        assert sub in res["int8"]
    assert isinstance(res["fp32"]["ram_bytes"], int)
    assert isinstance(res["gap3_metric_ok"], bool)
    assert isinstance(res["gap3_ram_ok"], bool)


def test_gap3_flags():
    # |delta| < 0.02 → metric_ok True ; ram_ratio > 1 → ram_ok True
    ok = bench.build_result_dict(
        "ewc", "d", "c", "auroc", 0.90, 4000, 0.01, 0.89, 1000, 0.01
    )
    assert ok["delta_metric"] == pytest.approx(-0.01)
    assert ok["gap3_metric_ok"] is True
    assert ok["ram_ratio"] == pytest.approx(4.0)
    assert ok["gap3_ram_ok"] is True

    # delta trop grand + pas de gain RAM → deux flags False
    bad = bench.build_result_dict(
        "ewc", "d", "c", "auroc", 0.90, 1000, 0.01, 0.80, 1000, 0.01
    )
    assert bad["gap3_metric_ok"] is False
    assert bad["gap3_ram_ok"] is False


def test_unknown_model_raises(tmp_path):
    with pytest.raises(ValueError, match="Mod"):
        bench.run_benchmark(
            model_name="does_not_exist",
            config_path="configs/ewc_int8_monitoring.yaml",
            output_path=str(tmp_path / "x.json"),
        )


# ----------------------------------------------------------------------------
# Smoke EWC réel (nécessite le CSV monitoring)
# ----------------------------------------------------------------------------
_MONITORING_CSV = _ROOT / (
    "data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset/"
    "equipment_anomaly_data.csv"
)


@pytest.mark.skipif(
    not _MONITORING_CSV.exists(), reason="Dataset monitoring absent (data/ gitignored)"
)
def test_ewc_benchmark_smoke(tmp_path):
    out = tmp_path / "bench.json"
    res = bench.run_benchmark(
        model_name="ewc",
        config_path="configs/ewc_int8_monitoring.yaml",
        output_path=str(out),
        n_samples=64,
    )
    assert out.exists()
    loaded = json.load(open(out))
    assert loaded == res
    assert loaded["model"] == "ewc"
    assert loaded["fp32"]["metric_name"] == "auroc"
    assert loaded["ram_ratio"] > 1.0
    assert loaded["gap3_ram_ok"] is True


@pytest.mark.skipif(
    not _MONITORING_CSV.exists(), reason="Dataset monitoring absent (data/ gitignored)"
)
@pytest.mark.parametrize(
    "model_name,config",
    [
        ("tinyol", "configs/tinyol_int8_monitoring.yaml"),
        ("mahalanobis", "configs/mahalanobis_int8_monitoring.yaml"),
    ],
)
def test_anomaly_int8_benchmark_smoke(model_name, config, tmp_path):
    """TinyOL et Mahalanobis : variants INT8 câblés (S2804/S2805) → plus de NotImplementedError."""
    out = tmp_path / f"{model_name}.json"
    res = bench.run_benchmark(
        model_name=model_name,
        config_path=config,
        output_path=str(out),
        n_samples=128,
    )
    assert out.exists()
    loaded = json.load(open(out))
    assert loaded == res
    assert loaded["model"] == model_name
    assert loaded["fp32"]["metric_name"] == "auroc"
    # Compression mémoire attendue (INT8 < FP32) sur les deux détecteurs.
    assert loaded["ram_ratio"] > 1.0
    assert loaded["gap3_ram_ok"] is True
