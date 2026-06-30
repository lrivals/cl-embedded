"""
tests/test_autonomy.py — Tests du module d'autonomie batterie (S3307/S3309).

Vérifie I_moy (courant moyen pondéré), autonomie = capacité/courant, balayage de
capacités, lecture config et propagation du placeholder « à mesurer ».

Exécution :
    pytest tests/test_autonomy.py -v
"""

from __future__ import annotations

import pytest

from src.evaluation.autonomy import (
    A_MESURER,
    autonomy_hours,
    average_current_ma,
    load_battery_capacities,
    sweep_capacities,
)


def test_average_current_matches_manual_calc():
    """I_moy = Σ(E_phase/V) / T_cycle × 1000, vérifié à la main.

    Une seule phase : E = 100 µJ = 1e-4 J, V = 3.3 V, t = 0.0005 s.
    charge = 1e-4 / 3.3 A·s ; T_cycle = 0.0005 s ; I = charge/T en A → ×1000 mA.
    """
    phases_uj = {"inference": 100.0}
    durations = {"inference": 0.0005}
    expected_ma = ((100.0 / 1e6) / 3.3) / 0.0005 * 1000.0
    assert average_current_ma(phases_uj, durations) == pytest.approx(expected_ma)


def test_average_current_weights_by_duration():
    """Deux phases : le courant moyen est pondéré par les durées."""
    phases_uj = {"inference": 100.0, "idle": 10.0}
    durations = {"inference": 0.0005, "idle": 0.01}
    charge = (100.0 / 1e6) / 3.3 + (10.0 / 1e6) / 3.3
    t_cycle = 0.0005 + 0.01
    expected_ma = charge / t_cycle * 1000.0
    assert average_current_ma(phases_uj, durations) == pytest.approx(expected_ma)


def test_average_current_rejects_placeholder():
    """Refuse de calculer si une énergie n'est pas mesurée (aucun chiffre inventé)."""
    with pytest.raises(ValueError):
        average_current_ma({"inference": A_MESURER}, {"inference": 0.0005})


def test_autonomy_hours_basic():
    assert autonomy_hours(220.0, 22.0) == pytest.approx(10.0)


def test_autonomy_hours_decreases_with_higher_current():
    a_low = autonomy_hours(1000.0, 10.0)
    a_high = autonomy_hours(1000.0, 50.0)
    assert a_high < a_low


def test_autonomy_hours_rejects_zero_current():
    with pytest.raises(ValueError):
        autonomy_hours(220.0, 0.0)


def test_sweep_capacities_monotonic():
    sweep = sweep_capacities(i_moy_ma=10.0, capacites_mah=[220.0, 1000.0, 3000.0])
    vals = list(sweep.values())
    assert vals == sorted(vals)  # autonomie croît avec la capacité
    assert sweep[3000.0] == pytest.approx(300.0)


def test_load_battery_capacities_from_config():
    caps = load_battery_capacities("configs/hw_profile_f439zi.yaml")
    assert len(caps) >= 1
    assert all(c > 0 for c in caps)
