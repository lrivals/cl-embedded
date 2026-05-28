"""
Tests pour scripts/sensor_stream.py (protocole v2, dry-run).

Tests sans board : valide le protocole UART v2, la construction des trames,
le CRC, le parsing des réponses et les statistiques globales.
"""

import importlib.util
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Chargement du module sensor_stream ──────────────────────────────────────
_SCRIPT = Path(__file__).parent.parent / "scripts" / "sensor_stream.py"
spec = importlib.util.spec_from_file_location("sensor_stream", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

build_frame_v2 = _mod.build_frame_v2
crc8           = _mod.crc8
MAGIC          = _mod.MAGIC
PROTO_VERSION  = _mod.PROTO_VERSION
RESPONSE_V2_SIZE  = _mod.RESPONSE_V2_SIZE
RESPONSE_V3_SIZE  = _mod.RESPONSE_V3_SIZE
parse_response    = _mod.parse_response


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_features():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)


@pytest.fixture
def mock_xy():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.int64)
    return X, y


# ── Tests protocole v2 ───────────────────────────────────────────────────────

def test_frame_magic(simple_features):
    frame = build_frame_v2(simple_features, 0, task_id=0, ts_ms=0)
    magic = struct.unpack_from("<H", frame, 0)[0]
    assert magic == MAGIC


def test_frame_version(simple_features):
    frame = build_frame_v2(simple_features, 0, task_id=0, ts_ms=0)
    version = frame[2]
    assert version == PROTO_VERSION


def test_frame_task_id(simple_features):
    for tid in range(4):
        frame = build_frame_v2(simple_features, 0, task_id=tid, ts_ms=0)
        assert frame[3] == tid


def test_frame_crc_valid(simple_features):
    frame = build_frame_v2(simple_features, 1, task_id=2, ts_ms=1000)
    payload, recv_crc = frame[:-1], frame[-1]
    assert crc8(payload) == recv_crc


def test_frame_crc_detects_corruption(simple_features):
    frame = bytearray(build_frame_v2(simple_features, 0, task_id=0, ts_ms=0))
    frame[5] ^= 0xFF  # corrompt un octet de features
    payload, recv_crc = bytes(frame[:-1]), frame[-1]
    assert crc8(payload) != recv_crc


def test_frame_length(simple_features):
    n = len(simple_features)
    frame = build_frame_v2(simple_features, 0, task_id=0, ts_ms=0)
    # HDR: 2(magic)+1(ver)+1(task)+4(ts)+1(n) = 9, features: n×4, tail: 2 (label+flags), crc: 1
    expected = 9 + n * 4 + 2 + 1
    assert len(frame) == expected


def test_frame_all_features_preserved(simple_features):
    frame = build_frame_v2(simple_features, 0, task_id=0, ts_ms=0)
    n = len(simple_features)
    offset = 9  # après header
    features_bytes = frame[offset:offset + n * 4]
    recovered = np.frombuffer(features_bytes, dtype=np.float32)
    np.testing.assert_array_almost_equal(simple_features, recovered, decimal=6)


# ── Tests dry-run streaming ─────────────────────────────────────────────────

def test_dry_run_returns_correct_count(mock_xy):
    X, y = mock_xy
    results = _mod._stream_dry_run(X, y, n_samples=30, n_tasks=3,
                                    request_update=False, verbose=False)
    assert len(results) == 30


def test_dry_run_all_status_ok(mock_xy):
    X, y = mock_xy
    results = _mod._stream_dry_run(X, y, n_samples=20, n_tasks=2,
                                    request_update=False, verbose=False)
    STATUS_CRC_ERR = _mod.STATUS_CRC_ERR
    for r in results:
        assert r["status"] & STATUS_CRC_ERR == 0


def test_dry_run_task_ids_distributed(mock_xy):
    X, y = mock_xy
    results = _mod._stream_dry_run(X, y, n_samples=30, n_tasks=3,
                                    request_update=False, verbose=False)
    task_ids = {r["task_id"] for r in results}
    assert len(task_ids) == 3


def test_dry_run_perfect_accuracy(mock_xy):
    X, y = mock_xy
    results = _mod._stream_dry_run(X, y, n_samples=20, n_tasks=2,
                                    request_update=False, verbose=False)
    acc = sum(r["pred"] == r["true"] for r in results) / len(results)
    assert acc == 1.0  # dry-run loopback → pred = true


# ── Tests statistiques ───────────────────────────────────────────────────────

def test_compute_stats_fields(mock_xy):
    X, y = mock_xy
    results = _mod._stream_dry_run(X, y, 20, 2, False, False)
    stats = _mod._compute_stats(results)
    required = ["n_samples", "n_tasks", "accuracy", "latency_mean_us",
                "latency_p50_us", "latency_p99_us", "ram_mean_bytes",
                "throughput_mean_ips", "crc_errors"]
    for field in required:
        assert field in stats, f"Champ manquant : {field}"


def test_compute_stats_empty():
    stats = _mod._compute_stats([])
    assert stats["n_samples"] == 0


# ── Tests parse_response v3 / rétrocompatibilité v2 ─────────────────────────

def test_sensor_stream_parse_v3():
    data = struct.pack("<BfIfff", 1, 0.9, 1234, 0.8, 0.75, 0.02)
    assert len(data) == RESPONSE_V3_SIZE
    result = parse_response(data)
    assert "acc" in result and "auroc" in result and "forgetting" in result
    assert abs(result["acc"] - 0.8) < 1e-5
    assert abs(result["auroc"] - 0.75) < 1e-5
    assert abs(result["forgetting"] - 0.02) < 1e-5
    assert result["pred"] == 1


def test_sensor_stream_backward_compat():
    data = struct.pack("<BfIHHB", 0, 0.95, 5000, 200, 333, 0)
    assert len(data) == RESPONSE_V2_SIZE
    result = parse_response(data)
    assert "pred" in result
    assert "acc" not in result
    assert "auroc" not in result
    assert "forgetting" not in result
    assert result["ram_bytes"] == 200
