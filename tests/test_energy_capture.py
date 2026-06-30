"""tests/test_energy_capture.py — Tests du pilote énergie LPM01A (S3305/S3309).

Couvre l'intégration µJ (E = Σ I·V·dt), le parsing CSV tolérant, la déduction des
fenêtres de phase depuis le signal de sync PA8 (S3304), la segmentation par phase,
l'export JSON (chemin mesuré vs placeholder « à mesurer ») et la chaîne bout-en-bout
sur une trace synthétique.

Toutes les valeurs numériques ci-dessous sont des **fixtures de test** calculées à la
main — jamais des résultats projet (aucune écriture dans `experiments/`).

Exécution :
    pytest tests/test_energy_capture.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.energy_capture import (
    A_MESURER,
    PHASES,
    _capture_one,
    _load_csv,
    derive_phase_windows,
    export_energy_json,
    integrate_energy_uj,
    segment_by_phase,
)


# ── integrate_energy_uj ───────────────────────────────────────────────────


def test_integrate_constant_trace_manual():
    """Trace constante : E = I·V·dt·N × 1e6 µJ, vérifié à la main.

    I = 0.01 A, V = 3.3 V, dt = 0.001 s, N = 5 échantillons.
    E = 0.01·3.3·0.001·5 = 1.65e-4 J = 165 µJ.
    """
    courant = np.full(5, 0.01)
    energy_uj = integrate_energy_uj(courant, 3.3, 0.001)
    assert energy_uj == pytest.approx(165.0)


def test_integrate_scalar_vs_vector_voltage():
    """Tension scalaire et tension par échantillon donnent le même résultat."""
    courant = np.array([0.01, 0.02, 0.03])
    scalar = integrate_energy_uj(courant, 3.3, 0.001)
    vector = integrate_energy_uj(courant, np.full(3, 3.3), 0.001)
    assert scalar == pytest.approx(vector)


def test_integrate_vector_dt():
    """dt par intervalle : E = Σ I·V·dt_i × 1e6."""
    courant = np.array([0.01, 0.02])
    dt = np.array([0.001, 0.002])
    expected = (0.01 * 3.3 * 0.001 + 0.02 * 3.3 * 0.002) * 1e6
    assert integrate_energy_uj(courant, 3.3, dt) == pytest.approx(expected)


# ── _load_csv (parsing tolérant) ──────────────────────────────────────────


def _write_csv(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_csv_named_header_full(tmp_path):
    """Header nommé avec time/current/voltage/sync → toutes colonnes lues."""
    csv = _write_csv(
        tmp_path / "trace.csv",
        ["time_s,current_a,voltage_v,sync", "0.0,0.01,3.3,1", "0.001,0.02,3.3,0"],
    )
    trace = _load_csv(csv)
    assert np.allclose(trace["time_s"], [0.0, 0.001])
    assert np.allclose(trace["current_a"], [0.01, 0.02])
    assert np.allclose(trace["voltage_v"], [3.3, 3.3])
    assert np.allclose(trace["sync"], [1, 0])


def test_load_csv_positional_no_header(tmp_path):
    """Sans header : ordre positionnel [temps, courant] ; tension/sync absentes."""
    csv = _write_csv(tmp_path / "trace.csv", ["0.0,0.01", "0.001,0.02"])
    trace = _load_csv(csv)
    assert np.allclose(trace["current_a"], [0.01, 0.02])
    assert trace["voltage_v"] is None
    assert trace["sync"] is None


def test_load_csv_ignores_comments(tmp_path):
    """Les lignes commençant par # sont ignorées."""
    csv = _write_csv(
        tmp_path / "trace.csv",
        ["# LPM01A export", "time,current", "0.0,0.01", "0.001,0.02"],
    )
    trace = _load_csv(csv)
    assert trace["current_a"].size == 2


# ── derive_phase_windows ──────────────────────────────────────────────────


def test_derive_phase_windows_square_signal():
    """Signal PA8 carré bas→haut→bas → fenêtres idle/inference/idle."""
    trace = {
        "time_s": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        "sync": np.array([0, 1, 1, 0, 0]),
    }
    windows = derive_phase_windows(trace)
    assert windows == [
        ("idle", 0.0, 0.1),
        ("inference", 0.1, 0.3),
        ("idle", 0.3, 0.4),
    ]


def test_derive_phase_windows_constant_signal():
    """Signal constant haut → une seule fenêtre inference."""
    trace = {
        "time_s": np.array([0.0, 0.1, 0.2]),
        "sync": np.array([1, 1, 1]),
    }
    assert derive_phase_windows(trace) == [("inference", 0.0, 0.2)]


def test_derive_phase_windows_requires_sync():
    """Sans colonne sync → ValueError (pas de fabrication)."""
    with pytest.raises(ValueError, match="synchronisation"):
        derive_phase_windows({"time_s": np.array([0.0]), "sync": None})


# ── segment_by_phase ──────────────────────────────────────────────────────


def test_segment_by_phase_accumulates(tmp_path):
    """Deux fenêtres inference + une idle → µJ sommés par phase, vérifié main.

    Échantillonnage uniforme dt=0.1 s, V=3.3 V, I=0.01 A.
    np.gradient sur un axe uniforme rend dt=0.1 partout.
    Fenêtre inference [0.0,0.2) → échantillons à t=0.0,0.1 :
        E = 0.01·3.3·0.1·2 = 6.6e-3 J = 6600 µJ.
    """
    csv = _write_csv(
        tmp_path / "trace.csv",
        [
            "time,current,voltage",
            "0.0,0.01,3.3",
            "0.1,0.01,3.3",
            "0.2,0.01,3.3",
            "0.3,0.01,3.3",
        ],
    )
    windows = [("inference", 0.0, 0.2), ("idle", 0.2, 0.4)]
    phases = segment_by_phase(csv, windows)
    assert set(phases) == set(PHASES)
    assert phases["inference"] == pytest.approx(6600.0)
    assert phases["acquisition"] == 0.0  # fenêtre vide → 0


def test_segment_by_phase_unknown_phase(tmp_path):
    """Une phase hors PHASES → ValueError."""
    csv = _write_csv(tmp_path / "t.csv", ["time,current,voltage", "0.0,0.01,3.3"])
    with pytest.raises(ValueError, match="Phase inconnue"):
        segment_by_phase(csv, [("bogus", 0.0, 1.0)])


def test_segment_by_phase_requires_voltage(tmp_path):
    """Trace sans tension → ValueError (fabrication interdite)."""
    csv = _write_csv(tmp_path / "t.csv", ["time,current", "0.0,0.01"])
    with pytest.raises(ValueError, match="fabrication"):
        segment_by_phase(csv, [("inference", 0.0, 1.0)])


# ── export_energy_json ────────────────────────────────────────────────────


def test_export_placeholder_path(tmp_path):
    """phases_uj=None → tous les champs énergie == 'à mesurer', source placeholder."""
    out = tmp_path / "ewc_fp32.json"
    export_energy_json(None, "ewc", "fp32", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_uj"] == A_MESURER
    assert all(payload["phases_uj"][p] == A_MESURER for p in PHASES)
    assert payload["source"] == "placeholder"


def test_export_measured_path(tmp_path):
    """phases_uj chiffré → champs numériques, total = somme, source lpm01a_csv."""
    out = tmp_path / "ewc_fp32.json"
    phases = {"inference": 100.0, "idle": 25.0}
    export_energy_json(phases, "ewc", "fp32", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total_uj"] == pytest.approx(125.0)
    assert payload["phases_uj"]["inference"] == pytest.approx(100.0)
    assert payload["source"] == "lpm01a_csv"


# ── Bout-en-bout : _capture_one sur CSV synthétique ───────────────────────


def test_capture_one_end_to_end(tmp_path):
    """CSV synthétique avec sync → JSON aux µJ chiffrés (la chaîne tourne).

    Niveau sync haut sur [0.0,0.2) (2 échantillons), bas ensuite.
    Phase inference : E = 0.01·3.3·0.1·2 = 6600 µJ (dt=0.1 uniforme).
    """
    csv = _write_csv(
        tmp_path / "ewc_fp32.csv",
        [
            "time,current,voltage,sync",
            "0.0,0.01,3.3,1",
            "0.1,0.01,3.3,1",
            "0.2,0.01,3.3,0",
            "0.3,0.01,3.3,0",
        ],
    )
    out = tmp_path / "ewc_fp32.json"
    _capture_one("ewc", "fp32", out, duration_s=10.0, sampling_rate_hz=0.0, csv_in=csv)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "lpm01a_csv"
    assert isinstance(payload["total_uj"], float)
    assert payload["phases_uj"]["inference"] == pytest.approx(6600.0)


def test_capture_one_refuses_csv_without_sync(tmp_path):
    """CSV réel sans colonne sync → ValueError (pas de segmentation fabriquée)."""
    csv = _write_csv(
        tmp_path / "ewc_fp32.csv", ["time,current,voltage", "0.0,0.01,3.3"]
    )
    out = tmp_path / "ewc_fp32.json"
    with pytest.raises(ValueError, match="synchronisation"):
        _capture_one(
            "ewc", "fp32", out, duration_s=10.0, sampling_rate_hz=0.0, csv_in=csv
        )
