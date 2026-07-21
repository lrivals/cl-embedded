"""
tests/test_sprint45_board.py — Sprint 45 (S4503) : garde-fous du portage des détecteurs de drift.

Vérifie, SANS board :
  - la parité PC des détecteurs Python déterministes sur une séquence connue (référence des
    tests Unity C, S4502) ;
  - la génération du header `inc/drift_methods_params.h` (structure + garde PROVIDED) ;
  - la calibration PSI (miroir set_params_from_reference) reproductible ;
  - l'honnêteté N/A / « à mesurer » (aucun chiffre inventé) ;
  - Gap 2 sur un résultat board déjà mesuré s'il existe (latence p99 < 100 ms).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.drift import DDM, PSI, PageHinkley  # noqa: E402

FW_INC = ROOT / "firmware" / "stm32f4_blink" / "inc"


# ── Parité déterministe des détecteurs (mêmes séquences que test_drift_methods.c) ──

def test_page_hinkley_drift_on_mean_shift():
    ph = PageHinkley({"delta": 0.0, "lambda_": 5.0, "min_instances": 5})
    seq = [0.0] * 8 + [3.0] * 8
    verdicts = [ph.update(x).name for x in seq]
    assert verdicts[9] == "DRIFT"
    assert verdicts.count("DRIFT") == 1


def test_ddm_reaches_warning_and_drift():
    ddm = DDM({"warning_level": 2.0, "drift_level": 3.0, "min_instances": 5})
    seq = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    verdicts = [ddm.update(float(e)).name for e in seq]
    assert "WARNING" in verdicts
    assert "DRIFT" in verdicts


def test_psi_drift_on_collapsed_block():
    psi = PSI({"bins": 4, "block_size": 5, "metric": "psi", "psi_threshold": 0.2})
    ref = np.array([0.05, 0.15, 0.2, 0.3, 0.4, 0.55, 0.6, 0.7, 0.85, 0.95,
                    0.1, 0.5, 0.9, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55, 0.5])
    psi.set_params_from_reference(ref)
    stream = [0.1, 0.4, 0.6, 0.9, 0.5, 0.95, 0.95, 0.95, 0.95, 0.95]
    verdicts = [psi.update(x).name for x in stream]
    assert verdicts[-1] == "DRIFT"
    assert verdicts[:9] == ["NORMAL"] * 9


def test_psi_calibration_reproducible():
    """Calibration PSI = miroir exact de set_params_from_reference (déterministe)."""
    rng = np.random.default_rng(42)
    ref = rng.normal(size=500)
    a = PSI({"bins": 10}); a.set_params_from_reference(ref)
    b = PSI({"bins": 10}); b.set_params_from_reference(ref)
    np.testing.assert_array_equal(a._edges, b._edges)
    np.testing.assert_array_equal(a._ref_probs, b._ref_probs)
    # ref_probs > 0 (eps ajouté) → log défini côté C.
    assert np.all(a._ref_probs > 0)


# ── Export du header de paramètres ────────────────────────────────────────────────

def test_export_drift_methods_header(tmp_path):
    params = {
        "page_hinkley": {"delta": 0.005, "lambda": 50.0, "min_instances": 30},
        "psi": {"bins": 4, "block_size": 5, "threshold": 0.2,
                "edges": [0.0, 0.25, 0.5, 0.75, 1.0], "ref_probs": [0.25, 0.25, 0.25, 0.25]},
    }
    pj = tmp_path / "params.json"
    pj.write_text(json.dumps(params))
    out = tmp_path / "inc"
    out.mkdir()
    r = subprocess.run(
        [sys.executable, "scripts/export_weights_c.py", "--drift-methods", str(pj),
         "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    header = (out / "drift_methods_params.h").read_text()
    assert "DRIFT_METHODS_PARAMS_PROVIDED 1" in header
    assert "PAGE_HINKLEY_DELTA" in header and "PAGE_HINKLEY_LAMBDA" in header
    assert "PSI_BIN_EDGES" in header and "PSI_REF_PROBS" in header
    assert "PSI_REF_BINS         4" in header


def test_generated_params_header_present():
    """Le header (généré par --drift-methods) définit toutes les macros consommées par pipeline.c."""
    header = (FW_INC / "drift_methods_params.h").read_text()
    for macro in ("PAGE_HINKLEY_DELTA", "PAGE_HINKLEY_LAMBDA", "DDM_WARN_SIGMA",
                  "DDM_DRIFT_SIGMA", "DRIFT_MIN_INSTANCES", "PSI_REF_BINS",
                  "PSI_BLOCK_SIZE_PARAM", "PSI_THRESHOLD_PARAM", "PSI_BIN_EDGES", "PSI_REF_PROBS"):
        assert macro in header, f"macro manquante : {macro}"


# ── Honnêteté N/A + Gap 2 (si mesuré) ─────────────────────────────────────────────

def _board_results():
    return sorted((ROOT / "experiments").glob("exp_S45_board_*/results.json"))


def test_board_results_are_honest():
    """Aucun chiffre inventé : métrique = float mesuré, null+na_reason, ou « à mesurer »."""
    for rj in _board_results():
        r = json.loads(rj.read_text())
        mv = r.get("metric_value")
        if mv is None:
            assert r.get("na_reason"), f"{rj} : metric_value null sans na_reason"
        elif isinstance(mv, str):
            assert mv == "à mesurer", f"{rj} : chaîne métrique inattendue {mv!r}"
        else:
            assert isinstance(mv, (int, float))


@pytest.mark.parametrize("rj", _board_results(), ids=lambda p: p.parent.name)
def test_gap2_latency_under_100ms(rj):
    r = json.loads(rj.read_text())
    p99 = r.get("latency_us_p99")
    if p99 is None:
        pytest.skip("latence non mesurée (streaming différé)")
    assert p99 < 100_000, f"Gap 2 violé : p99={p99} µs"


def test_parity_when_present():
    """Si une parité a été calculée, verdict_parity ∈ [0,1] et table cohérente."""
    for pj in sorted((ROOT / "experiments").glob("exp_S45_parity_*.json")):
        p = json.loads(pj.read_text())
        vp = p.get("verdict_parity")
        if vp is not None:
            assert 0.0 <= vp <= 1.0
            assert len(p["table"]) == p["n_samples"]


# ── Agrégat S4504 (exp_S45_summary.json) ──────────────────────────────────────────

SUMMARY = ROOT / "experiments" / "exp_S45_summary.json"


def _summary():
    if not SUMMARY.exists():
        pytest.skip("exp_S45_summary.json absent (lancer scripts/aggregate_sprint45.py)")
    return json.loads(SUMMARY.read_text())


def _measured_board_cells(summary):
    """Itère les cellules board effectivement mesurées (latence présente)."""
    for ds, per_det in summary["results"].items():
        for det, cell in per_det.items():
            b = cell["board"]
            if b.get("measured"):
                yield ds, det, cell


def test_summary_structure():
    """`exp_S45_summary.json` : [dataset][detector][platform] + clés attendues."""
    s = _summary()
    assert s["sprint"] == 45
    assert set(s["detectors"]) == {"page_hinkley", "ddm", "psi"}
    for ds in s["datasets"]:
        for det in s["detectors"]:
            cell = s["results"][ds][det]
            assert "board" in cell and "pc_proxy" in cell
            for k in ("measured", "latency_us_p50", "latency_us_p99", "bss_bytes",
                      "verdict_parity", "gap2_ok", "gap3_ram_ok"):
                assert k in cell["board"], f"clé board manquante : {k}"
            assert cell["pc_proxy"]["is_proxy"] is True   # proxy jamais confondu avec board


def test_summary_gap2_all_measured():
    """Gap 2 : toute cellule board mesurée a p99 < 100 ms."""
    s = _summary()
    measured = list(_measured_board_cells(s))
    if not measured:
        pytest.skip("aucune cellule board mesurée")
    for ds, det, cell in measured:
        b = cell["board"]
        assert b["latency_us_p99"] < 100_000, f"Gap 2 violé : {det}×{ds} p99={b['latency_us_p99']}"
        assert b["gap2_ok"] is True


def test_summary_gap3_ram_budget():
    """Gap 3 : `.bss` mesuré < 256 Ko ; delta build par défaut = constantes documentées (0 régr.)."""
    s = _summary()
    assert s["bss_default"] == 105_036            # build défaut invariant (S4502)
    assert s["bss_delta_by_method"] == {"page_hinkley": 36, "ddm": 40, "psi": 132}
    for ds, det, cell in _measured_board_cells(s):
        b = cell["board"]
        assert b["bss_bytes"] < s["ram_budget_bytes"], f"Gap 3 violé : {det}×{ds}"
        assert b["gap3_ram_ok"] is True


def test_summary_verdict_parity_deterministic():
    """Parité board↔PC = 1.000 sur les cellules mesurées déterministes."""
    s = _summary()
    measured = list(_measured_board_cells(s))
    if not measured:
        pytest.skip("aucune cellule board mesurée")
    for ds, det, cell in measured:
        vp = cell["board"]["verdict_parity"]
        if vp is not None:
            assert vp == 1.0, f"parité non exacte : {det}×{ds} = {vp}"


def test_summary_no_hardcoded_values():
    """0 chiffre en dur : les valeurs du summary proviennent bien des JSON board/parité sources."""
    s = _summary()
    for ds, det, cell in _measured_board_cells(s):
        raw = json.loads(
            (ROOT / "experiments" / f"exp_S45_board_{det}_{ds}" / "results.json").read_text())
        assert cell["board"]["latency_us_p50"] == raw["latency_us_p50"]
        assert cell["board"]["bss_bytes"] == raw["bss_bytes"]
