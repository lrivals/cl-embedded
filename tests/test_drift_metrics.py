"""
tests/test_drift_metrics.py — Sprint 44 (S4404/S4406) : harnais de métriques de drift.

Cas canoniques (spec S4404) sur des verdicts **synthétiques** et des points de drift **connus**,
sans détecteur réel :

- **oracle** (alarme exactement aux ``drift_points``) → délai=0, FAR=0, MDR=0, F1=1.0 ;
- **paresseux** (jamais d'alarme) → MDR=1.0, FAR=0, délai=``None`` ;
- **paranoïaque** (alarme partout) → FAR élevé, délai≈0, précision faible ;
- **sans vérité-terrain ponctuelle** (``drift_points=None``) → délai/MDR/P/R/F1 = ``None`` honnête ;
- ``state_bytes`` du profil cohérent avec l'annotation ``# MEM:`` du détecteur.

    pytest tests/test_drift_metrics.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.evaluation.drift_metrics import (
    alarms_from_verdicts,
    build_comparison_table,
    compute_drift_metrics,
    profile_drift_detector,
    save_drift_metrics,
)
from src.models.drift import DDM, PSI, DriftVerdict

N = 6000
DRIFT_POINTS = [1500, 3000, 4500]
TOL = 200


def _alarms_at(indices: list[int], n: int = N) -> list[bool]:
    a = np.zeros(n, dtype=bool)
    a[indices] = True
    return a.tolist()


# --------------------------------------------------------------------------- #
# Mapping verdict → alarme (source unique)
# --------------------------------------------------------------------------- #
def test_alarms_from_verdicts_handles_enum_str_int():
    verdicts = [
        DriftVerdict.NORMAL,
        DriftVerdict.WARNING,
        DriftVerdict.DRIFT,  # enum
        "NORMAL",
        "FAULT",
        "DRIFT",  # baseline strings
        0,
        1,
        2,  # ints
        True,
        False,  # bool
    ]
    assert alarms_from_verdicts(verdicts) == [
        False, False, True,
        False, False, True,
        False, False, True,
        True, False,
    ]


# --------------------------------------------------------------------------- #
# Cas oracle : alarme exactement aux points de drift
# --------------------------------------------------------------------------- #
def test_oracle_perfect_detection():
    alarms = _alarms_at(DRIFT_POINTS)
    m = compute_drift_metrics(alarms, DRIFT_POINTS, N, TOL)
    assert m["mean_detection_delay"] == 0.0
    assert m["missed_detection_rate"] == 0.0
    assert m["false_alarm_rate"] == 0.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["n_detected"] == len(DRIFT_POINTS)
    assert m["n_false_alarms"] == 0


def test_oracle_with_delay():
    """Alarme 50 échantillons après chaque point (dans la tolérance) → délai=50, MDR=0."""
    alarms = _alarms_at([dp + 50 for dp in DRIFT_POINTS])
    m = compute_drift_metrics(alarms, DRIFT_POINTS, N, TOL)
    assert m["mean_detection_delay"] == 50.0
    assert m["mtd"] == 50.0
    assert m["missed_detection_rate"] == 0.0
    assert m["false_alarm_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Cas paresseux : jamais d'alarme
# --------------------------------------------------------------------------- #
def test_lazy_never_alarms():
    alarms = _alarms_at([])
    m = compute_drift_metrics(alarms, DRIFT_POINTS, N, TOL)
    assert m["missed_detection_rate"] == 1.0
    assert m["false_alarm_rate"] == 0.0
    assert m["mean_detection_delay"] is None  # rien à moyenner (honnête)
    assert m["recall"] == 0.0
    assert m["n_detected"] == 0


# --------------------------------------------------------------------------- #
# Cas paranoïaque : alarme partout
# --------------------------------------------------------------------------- #
def test_paranoid_alarms_everywhere():
    alarms = [True] * N
    m = compute_drift_metrics(alarms, DRIFT_POINTS, N, TOL)
    assert m["mean_detection_delay"] == 0.0  # une alarme au point même
    assert m["missed_detection_rate"] == 0.0
    assert m["false_alarm_rate"] > 0.5  # énormément de fausses alarmes
    assert m["precision"] < 0.01  # précision effondrée
    assert m["n_false_alarms"] > 0


# --------------------------------------------------------------------------- #
# Cas honnête : pas de vérité-terrain ponctuelle (Electricity/NOAA)
# --------------------------------------------------------------------------- #
def test_no_ground_truth_null_fields():
    alarms = _alarms_at([100, 2000, 4000])
    m = compute_drift_metrics(alarms, None, N, TOL)
    assert m["mean_detection_delay"] is None
    assert m["missed_detection_rate"] is None
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None
    assert m["mtd"] is None
    # FAR / MTFA restent calculés sur le flux réputé stable.
    assert m["false_alarm_rate"] is not None
    assert m["false_alarm_rate"] > 0.0
    assert m["mtfa"] is not None


def test_missed_when_alarm_outside_tolerance():
    """Alarme 500 échantillons après le point (> tolérance 200) → point manqué + fausse alarme."""
    alarms = _alarms_at([dp + 500 for dp in DRIFT_POINTS])
    m = compute_drift_metrics(alarms, DRIFT_POINTS, N, TOL)
    assert m["missed_detection_rate"] == 1.0
    assert m["n_false_alarms"] == len(DRIFT_POINTS)


# --------------------------------------------------------------------------- #
# Profil de coût : state_bytes cohérent avec # MEM: du détecteur
# --------------------------------------------------------------------------- #
def test_profile_state_bytes_matches_mem_annotation():
    # DDM : # MEM: 20 B @ FP32 (5 scalaires × 4 B) — état O(1).
    ddm = DDM()
    prof = profile_drift_detector(ddm, np.zeros(200).tolist())
    assert prof["state_bytes"] == 5 * 4 == 20
    assert prof["state_bytes_source"] == "get_state_bytes"
    assert prof["requires_label"] is True
    assert prof["_proxy"] is True
    assert prof["latency_us_per_update"] is not None
    assert prof["n_updates"] == 200


def test_profile_state_bytes_o_bins_psi():
    # PSI : # MEM: (3·bins+1)·4 B — indépendant de la taille de bloc.
    psi = PSI({"bins": 10, "block_size": 100})
    psi.set_params_from_reference(np.random.default_rng(0).normal(size=500))
    prof = profile_drift_detector(psi, np.random.default_rng(1).normal(size=300).tolist())
    assert prof["state_bytes"] == (3 * 10 + 1) * 4
    assert prof["requires_label"] is False


def test_profile_baseline_state_bytes_from_annotation():
    """Baseline sans get_state_bytes : state_bytes fourni par l'appelant (# MEM: 200 B)."""

    class _NoStateApi:
        requires_label = False

        def update(self, v):
            return "NORMAL"

    prof = profile_drift_detector(_NoStateApi(), [0.0] * 10, state_bytes=200)
    assert prof["state_bytes"] == 200
    assert prof["state_bytes_source"] == "mem_annotation"


# --------------------------------------------------------------------------- #
# Table comparative + sauvegarde
# --------------------------------------------------------------------------- #
def test_build_comparison_table_and_save(tmp_path):
    results = {
        "ddm": {
            "synthetic": {
                "requires_label": True,
                "viabilite_mcu": "haute",
                "drift_metrics": {"f1": 0.8, "false_alarm_rate": 0.01, "mtfa": 500.0,
                                  "mtd": 30.0, "missed_detection_rate": 0.0},
                "cost": {"state_bytes": 20, "latency_us_per_update": 1.2},
            }
        },
        "psi": {
            "electricity": {
                "requires_label": False,
                "viabilite_mcu": "haute",
                "drift_metrics": {"f1": None, "false_alarm_rate": 0.05, "mtfa": 100.0,
                                  "mtd": None, "missed_detection_rate": None},
                "cost": {"state_bytes": 124, "latency_us_per_update": 3.4},
            }
        },
    }
    table = build_comparison_table(results)
    assert table["columns"][0] == "detector"
    assert len(table["rows"]) == 2
    row = next(r for r in table["rows"] if r["detector"] == "ddm")
    assert row["state_bytes"] == 20 and row["f1"] == 0.8

    out = tmp_path / "table.json"
    save_drift_metrics(table, out, extra_info={"sprint": 44})
    import json

    loaded = json.loads(out.read_text())
    assert loaded["sprint"] == 44 and loaded["rows"]


# --------------------------------------------------------------------------- #
# 0 chiffre de résultat en dur dans le catalogue de figures S4405 (miroir S4207)
# --------------------------------------------------------------------------- #
def test_no_hardcoded_results_drift_pc() -> None:
    """Scan AST de drift_detection_pc.py : aucun flottant hors liste blanche de layout/style."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src/figures/catalogs/drift_detection_pc.py"
    # Constantes de mise en page/style (positions, largeurs, seuil de couleur de texte, figsize) —
    # AUCUN résultat : toute valeur tracée vient d'un results.json.
    layout_whitelist: set[float] = {0.0, 0.005, 0.01, 0.14, 0.2, 0.5, 0.6, 0.86, 1.2, 1.4, 4.5}
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in layout_whitelist
    }
    assert not offending, (
        f"Littéraux flottants suspects dans drift_detection_pc.py : {sorted(offending)} — "
        "toute valeur tracée doit venir d'un results.json, pas d'un littéral."
    )
