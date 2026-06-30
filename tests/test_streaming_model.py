"""
tests/test_streaming_model.py — Tests du modèle de streaming/buffer (S3401/S3409).

Vérifie Débit_max = 1/latence, Débit_streaming = f_acq×S/W, la marge temps-réel, le budget
buffer W×sizeof(sample) et la contrainte SRAM, ainsi que le chargement du profil YAML.

Exécution :
    pytest tests/test_streaming_model.py -v
"""

from __future__ import annotations

import pytest

from src.evaluation.streaming_model import (
    budget_buffer_bytes,
    check_sram_budget,
    debit_max,
    debit_streaming,
    load_profile,
    marge_temps_reel,
)


# ── Débit_max = 1 / latence ──────────────────────────────────────────────────
def test_debit_max_inverse_latency():
    """Débit_max = 1/latence : 5 µs → 200 000 Hz."""
    assert debit_max(5e-6) == pytest.approx(200_000.0)


def test_debit_max_ewc():
    """EWC 50 µs → 20 000 Hz."""
    assert debit_max(50e-6) == pytest.approx(20_000.0)


def test_debit_max_rejects_nonpositive():
    with pytest.raises(ValueError):
        debit_max(0.0)
    with pytest.raises(ValueError):
        debit_max(-1.0)


# ── Débit_streaming = f_acq × S / W ──────────────────────────────────────────
def test_debit_streaming_basic():
    """f_acq=100, stride=1, window=5 → 20 Hz."""
    assert debit_streaming(100.0, 1, 5) == pytest.approx(20.0)


def test_debit_streaming_stride_increases_rate():
    """Augmenter le stride augmente le débit de fenêtres."""
    base = debit_streaming(100.0, 1, 5)
    assert debit_streaming(100.0, 5, 5) == pytest.approx(5 * base)


def test_debit_streaming_window_decreases_rate():
    """Une fenêtre plus grande diminue le débit de fenêtres."""
    assert debit_streaming(100.0, 1, 10) < debit_streaming(100.0, 1, 5)


def test_debit_streaming_rejects_bad_args():
    with pytest.raises(ValueError):
        debit_streaming(100.0, 1, 0)
    with pytest.raises(ValueError):
        debit_streaming(100.0, 0, 5)


# ── Marge temps-réel ─────────────────────────────────────────────────────────
def test_marge_ok_when_streaming_below_max():
    """20 Hz produit vs 200 000 Hz soutenable → ok, marge ≈ 1.0."""
    res = marge_temps_reel(20.0, 200_000.0)
    assert res["ok"] is True
    assert res["marge_pct"] == pytest.approx(1.0 - 20.0 / 200_000.0)


def test_marge_not_ok_when_saturated():
    """Débit produit > débit max → not ok, marge négative."""
    res = marge_temps_reel(300_000.0, 200_000.0)
    assert res["ok"] is False
    assert res["marge_pct"] < 0.0


def test_marge_boundary_equal_is_ok():
    """Égalité exacte = limite acceptable (≤)."""
    res = marge_temps_reel(200_000.0, 200_000.0)
    assert res["ok"] is True
    assert res["marge_pct"] == pytest.approx(0.0)


# ── Budget buffer + contrainte SRAM ──────────────────────────────────────────
def test_budget_buffer_bytes():
    """W=50, float32 → 200 B."""
    assert budget_buffer_bytes(50, 4) == 200


def test_check_sram_budget_within():
    assert check_sram_budget(200, 65536) is True


def test_check_sram_budget_exceeds():
    assert check_sram_budget(70_000, 65536) is False


def test_check_sram_budget_boundary():
    assert check_sram_budget(65536, 65536) is True


# ── Chargement profil YAML ───────────────────────────────────────────────────
def test_load_profile_reads_streaming_section(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "streaming:\n  f_acq_hz: 100\n  window_w: 5\n  sram_budget_bytes: 65536\n"
    )
    prof = load_profile(cfg)
    assert prof["f_acq_hz"] == 100
    assert prof["window_w"] == 5
    assert prof["sram_budget_bytes"] == 65536


def test_load_profile_real_config():
    """Le profil livré du projet est lisible et cohérent."""
    prof = load_profile("configs/streaming_profile.yaml")
    assert prof["sizeof_sample_bytes"] == 4
    assert "sweep" in prof
    assert "latences_inf_us" in prof
    # Cohérence bout-en-bout : budget de la fenêtre par défaut tient en SRAM.
    buf = budget_buffer_bytes(prof["window_w"], prof["sizeof_sample_bytes"])
    assert check_sram_budget(buf, prof["sram_budget_bytes"])
