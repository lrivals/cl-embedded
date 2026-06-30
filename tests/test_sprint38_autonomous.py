"""test_sprint38_autonomous.py — S3809 : verrou comportemental Sprint 38.

Couvre : calibration des seuils du gate, logique des 4 politiques de mise à jour,
déterminisme du gate, structure de ``exp_S38_summary.json`` + ``economy_table`` (deltas vs
``always``), et Gap 2 (latences board < 100 ms). Lecture seule sur les expériences déjà produites
(skip propre si absentes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_sprint38 import _delta, _economy_table  # noqa: E402
from src.evaluation.drift_detector import SlidingWindowDriftDetector  # noqa: E402

EXP = ROOT / "experiments"
GAP2_US = 100_000
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")


# ── Calibration des seuils ────────────────────────────────────────────────────

def test_threshold_calibration_p95_multipliers():
    rng = np.random.default_rng(0)
    normal = rng.normal(1.0, 0.2, size=2000)
    det = SlidingWindowDriftDetector(window_size=50, fault_multiplier=2.5,
                                     drift_multiplier=1.3, drift_ratio=0.6)
    det.set_thresholds_from_normal(normal)
    p95 = float(np.percentile(normal, 95))
    assert det.fault_threshold == pytest.approx(p95 * 2.5)
    assert det.drift_threshold == pytest.approx(p95 * 1.3)
    assert det.fault_threshold > det.drift_threshold


# ── Logique des 4 politiques (mapping verdict → action) ───────────────────────

def _policy_update_counts(verdicts: list[str], policy: str) -> dict:
    """Réplique le mapping documenté (run_sprint38_pc / gate firmware) → compteurs."""
    ewc_updates, maha_updates = 0, 0
    for v in verdicts:
        if policy == "frozen":
            pass
        elif policy == "always":
            ewc_updates += 1
        elif policy == "gated_truelabel":
            if v != "NORMAL":
                ewc_updates += 1
        elif policy == "gated_pseudolabel":
            if v == "FAULT":
                ewc_updates += 1
            elif v == "DRIFT":
                maha_updates += 1
    return {"ewc_updates": ewc_updates, "maha_updates": maha_updates}


def test_policy_logic_mapping():
    verdicts = ["NORMAL", "NORMAL", "DRIFT", "FAULT", "DRIFT", "NORMAL", "FAULT"]
    n = len(verdicts)
    assert _policy_update_counts(verdicts, "frozen")["ewc_updates"] == 0
    assert _policy_update_counts(verdicts, "always")["ewc_updates"] == n
    # gated_truelabel : SGD sur tout verdict != NORMAL
    gt = _policy_update_counts(verdicts, "gated_truelabel")
    assert gt["ewc_updates"] == sum(v != "NORMAL" for v in verdicts)
    # gated_pseudolabel : FAULT→SGD, DRIFT→maha_update, NORMAL→rien
    gp = _policy_update_counts(verdicts, "gated_pseudolabel")
    assert gp["ewc_updates"] == sum(v == "FAULT" for v in verdicts)
    assert gp["maha_updates"] == sum(v == "DRIFT" for v in verdicts)


def test_policy_update_ordering():
    """frozen (0) ≤ gated ≤ always (N) — l'arbitrage cœur du sprint."""
    verdicts = ["NORMAL"] * 90 + ["DRIFT"] * 7 + ["FAULT"] * 3
    n = len(verdicts)
    fr = _policy_update_counts(verdicts, "frozen")["ewc_updates"]
    al = _policy_update_counts(verdicts, "always")["ewc_updates"]
    gt = _policy_update_counts(verdicts, "gated_truelabel")["ewc_updates"]
    assert fr == 0 <= gt <= al == n
    assert gt < al   # le gate filtre réellement


# ── Déterminisme du gate ──────────────────────────────────────────────────────

def test_gate_determinism():
    rng = np.random.default_rng(7)
    scores = rng.normal(1.0, 0.5, size=300).tolist()
    normal = rng.normal(1.0, 0.2, size=500)

    def run():
        d = SlidingWindowDriftDetector(window_size=50, drift_ratio=0.6)
        d.set_thresholds_from_normal(normal)
        return [d.update(s) for s in scores]

    assert run() == run()


def test_fault_priority_and_normal():
    det = SlidingWindowDriftDetector(window_size=10, drift_ratio=0.6)
    det.fault_threshold = 5.0
    det.drift_threshold = 1.0
    assert det.update(100.0) == "FAULT"          # dépassement instantané
    det2 = SlidingWindowDriftDetector(window_size=10, drift_ratio=0.6)
    det2.fault_threshold = 5.0
    det2.drift_threshold = 1.0
    assert det2.update(0.1) == "NORMAL"          # sous les deux seuils


# ── economy_table : arithmétique des deltas vs always ─────────────────────────

def test_economy_table_math():
    cells = {
        "frozen": {"board": {"mean_latency_us": 50.0, "f1_faulty": 0.20,
                             "update_rate": 0.0, "bss_delta_vs_default": None}},
        "always": {"board": {"mean_latency_us": 250.0, "f1_faulty": 0.90,
                             "update_rate": 1.0, "bss_delta_vs_default": 0}},
        "gated_truelabel": {"board": {"mean_latency_us": 80.0, "f1_faulty": 0.88,
                                      "update_rate": 0.025, "bss_delta_vs_default": 300}},
        "gated_pseudolabel": {"board": {"mean_latency_us": 80.0, "f1_faulty": 0.50,
                                        "update_rate": 0.025, "bss_delta_vs_default": 300}},
    }
    eco = _economy_table(cells)
    assert eco["always"]["latency_saved_us"] == 0.0
    assert eco["gated_truelabel"]["latency_saved_us"] == pytest.approx(170.0)
    assert eco["gated_truelabel"]["f1_lost"] == pytest.approx(0.02)
    assert eco["gated_truelabel"]["ram_added_bytes"] == 300
    assert eco["gated_truelabel"]["updates_saved_pct"] == pytest.approx(1 - 0.025 / 1.0)
    assert _delta(None, 1.0) is None


# ── Structure du summary + Gap 2 (si produit) ─────────────────────────────────

def _summary():
    p = EXP / "exp_S38_summary.json"
    if not p.exists():
        pytest.skip("exp_S38_summary.json absent — lancer aggregate_sprint38.py")
    return json.loads(p.read_text())


def test_summary_structure():
    s = _summary()
    assert s["sprint"] == 38
    for ds in s["datasets"]:
        for init in s["init_modes"]:
            cell = s["results"][ds][init]
            assert "economy_table" in cell
            for pol in POLICIES:
                assert "pc" in cell[pol] and "board" in cell[pol]
                assert set(cell["economy_table"][pol]) == {
                    "latency_saved_us", "ram_added_bytes", "f1_lost", "updates_saved_pct"}


def test_summary_update_rate_ordering():
    """Sur chaque cellule board complète : frozen=0 ≤ gated < always=1."""
    s = _summary()
    checked = 0
    for ds in s["datasets"]:
        for init in s["init_modes"]:
            cell = s["results"][ds][init]
            rates = {p: cell[p]["board"]["update_rate"] for p in POLICIES}
            if any(rates[p] is None for p in POLICIES):
                continue
            assert rates["frozen"] == 0
            assert rates["always"] == pytest.approx(1.0)
            assert 0 <= rates["gated_truelabel"] < rates["always"]
            assert 0 <= rates["gated_pseudolabel"] < rates["always"]
            checked += 1
    assert checked > 0, "aucune cellule board complète trouvée"


def test_gap2_latencies_under_100ms():
    s = _summary()
    for ds in s["datasets"]:
        for init in s["init_modes"]:
            for pol in POLICIES:
                b = s["results"][ds][init][pol]["board"]
                for key in ("mean_latency_us", "inference_latency_us", "gate_overhead_us"):
                    v = b.get(key)
                    if isinstance(v, (int, float)):
                        assert v < GAP2_US, f"{ds}/{init}/{pol}/{key}={v} ≥ 100 ms"


def test_verdict_parity_exact_when_present():
    """Le gate embarqué reproduit la décision PC (mêmes seuils exportés)."""
    s = _summary()
    seen = 0
    for ds in s["datasets"]:
        for init in s["init_modes"]:
            for pol in ("gated_truelabel", "gated_pseudolabel"):
                vp = s["results"][ds][init][pol]["board"]["verdict_parity_rate"]
                if vp is not None:
                    assert vp == pytest.approx(1.0), f"{ds}/{init}/{pol} verdict_parity={vp}"
                    seen += 1
    if seen == 0:
        pytest.skip("aucune cellule gated board produite")
