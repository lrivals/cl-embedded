"""
tests/test_hw_cost_model.py — Tests du modèle de coût matériel (S3302/S3309).

Vérifie les formules proxy de hw_cost_model.py et la cohérence avec le profil HW
configs/hw_profile_f439zi.yaml (bornes efficacité, FLOPS/W positif, throughput).

Exécution :
    pytest tests/test_hw_cost_model.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.hw_cost_model import (
    EFFICACITE_MAX,
    EFFICACITE_MIN,
    estimate_inference_time,
    flops_per_watt,
    load_hw_profile,
    power_watts,
    throughput,
)

HW_PROFILE = Path("configs/hw_profile_f439zi.yaml")


def test_efficacite_bounds():
    """L'efficacité de la config par défaut respecte [0.1, 0.6] (CR 19 mai)."""
    profile = load_hw_profile(HW_PROFILE)
    eff = profile["hardware"]["efficacite"]
    for enc in ("fp32", "int8"):
        assert EFFICACITE_MIN <= eff[enc] <= EFFICACITE_MAX


def test_flops_per_watt_positive():
    assert flops_per_watt(180e6, power_watts(50.0, 3.3)) > 0


def test_flops_per_watt_rejects_zero_power():
    with pytest.raises(ValueError):
        flops_per_watt(180e6, 0.0)


def test_throughput_consistent_with_latency():
    """throughput(t) == 1/t, cohérent avec estimate_inference_time."""
    t = estimate_inference_time(macs=1000, flops_peak=180e6, efficacite=0.3)
    assert throughput(t) == pytest.approx(1.0 / t)


def test_estimate_inference_time_is_flops_over_peak_eff():
    macs = 1000
    flops_peak = 180e6
    eff = 0.3
    expected = (2 * macs) / (flops_peak * eff)
    assert estimate_inference_time(macs, flops_peak, eff) == pytest.approx(expected)


def test_estimate_inference_time_rejects_nonphysical():
    with pytest.raises(ValueError):
        estimate_inference_time(1000, 0.0, 0.3)
    with pytest.raises(ValueError):
        estimate_inference_time(1000, 180e6, 0.0)


def test_power_watts_formula():
    # 50 mA × 3.3 V = 0.165 W
    assert power_watts(50.0, 3.3) == pytest.approx(0.165)
