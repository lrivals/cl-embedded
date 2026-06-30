"""Tests Sprint 36 / S3608 — comparaison appariée PC ↔ board (EWC).

Vérifie la **structure** et la **cohérence** des artefacts produits (pas les valeurs mesurées,
qui dépendent de la board) :

  * ``exp_S36_summary.json`` indexé ``[dataset][condition][platform]`` ;
  * parité **frozen exacte** (== 1.0) quand renseignée (poids gelés) ;
  * forme des tables parité (``match == (pred_pc == pred_board)``) ;
  * présence des clés métriques PC/board ;
  * Gap 2 : toutes latences renseignées < 100 000 µs ;
  * (rework) plateformes INT8 ``board_{frozen,online}_int8`` présentes, ratio RAM ≈ 4×
    (Gap 3), accord INT8↔FP32 ∈ [0, 1] — fonction de poids testée sans board.

Robuste aux champs ``null`` (skip si l'artefact amont n'a pas encore tourné). Le firmware EWC
étant inchangé par ce sprint, Unity ``make test`` doit rester à 0 régression (hors périmètre ici).

Exécution :
    pytest tests/test_sprint36_comparison.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXP = Path("experiments")
DATASETS = ["pronostia", "monitoring"]
CONDITIONS = ["5feat", "all"]
PROTOCOLS = ["frozen", "online"]
GAP2_LATENCY_US = 100_000

SUMMARY = EXP / "exp_S36_summary.json"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _parity_files():
    return [
        EXP / f"exp_S36_parity_{cond}_{proto}_{ds}.json"
        for ds in DATASETS for cond in CONDITIONS for proto in PROTOCOLS
    ]


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummaryStructure:
    def test_summary_indexed_by_dataset_condition_platform(self):
        s = _load(SUMMARY)
        if s is None:
            pytest.skip("exp_S36_summary.json absent (lancer aggregate_sprint36.py)")
        res = s["results"]
        assert set(res) == set(DATASETS)
        for ds in DATASETS:
            assert set(res[ds]) == set(CONDITIONS)
            for cond in CONDITIONS:
                cell = res[ds][cond]
                assert set(cell) >= {"pc", "board_frozen", "board_online", "delta_pc_board"}

    def test_metrics_keys_present(self):
        s = _load(SUMMARY)
        if s is None:
            pytest.skip("exp_S36_summary.json absent")
        for ds in DATASETS:
            for cond in CONDITIONS:
                cell = s["results"][ds][cond]
                assert {"acc_final", "af", "f1_faulty", "roc_auc", "ram_peak_bytes"} <= set(cell["pc"])
                assert {"latency_us_p50", "parity_rate"} <= set(cell["board_frozen"])
                assert {"latency_us_p50", "parity_rate"} <= set(cell["board_online"])

    def test_gap2_latencies(self):
        s = _load(SUMMARY)
        if s is None:
            pytest.skip("exp_S36_summary.json absent")
        seen = 0
        for ds in DATASETS:
            for cond in CONDITIONS:
                cell = s["results"][ds][cond]
                for plat in ("board_frozen", "board_online"):
                    for key in ("latency_us_p50", "latency_us_p99"):
                        lat = cell[plat].get(key)
                        if isinstance(lat, (int, float)):
                            seen += 1
                            assert lat < GAP2_LATENCY_US, f"{ds}/{cond}/{plat}/{key}={lat}"
        if seen == 0:
            pytest.skip("aucune latence renseignée (board non exécutée)")


# ── Parité ────────────────────────────────────────────────────────────────────

class TestParity:
    def test_parity_frozen_is_exact_when_present(self):
        checked = 0
        for ds in DATASETS:
            for cond in CONDITIONS:
                r = _load(EXP / f"exp_S36_parity_{cond}_frozen_{ds}.json")
                if r is None or r.get("parity_rate") is None:
                    continue
                checked += 1
                assert r["parity_rate"] == 1.0, f"{cond}/{ds} frozen parity={r['parity_rate']}"
                assert r["mismatch_count"] == 0
        if checked == 0:
            pytest.skip("aucun fichier parité frozen (lancer board_pc_parity.py)")

    def test_parity_table_shape(self):
        checked = 0
        for path in _parity_files():
            r = _load(path)
            if r is None or not r.get("rows"):
                continue
            checked += 1
            assert {"protocol", "condition", "dataset", "parity_rate", "rows"} <= set(r)
            for row in r["rows"][:50]:
                assert {"idx", "true", "pred_pc", "pred_board", "match"} <= set(row)
                assert row["match"] == (row["pred_pc"] == row["pred_board"])
        if checked == 0:
            pytest.skip("aucun fichier parité (lancer board_pc_parity.py)")

    def test_parity_online_is_approx_class(self):
        for ds in DATASETS:
            for cond in CONDITIONS:
                r = _load(EXP / f"exp_S36_parity_{cond}_online_{ds}.json")
                if r is None:
                    continue
                assert r["parity_class"] == "approx"
                if r.get("parity_rate") is not None:
                    assert 0.0 <= r["parity_rate"] <= 1.0


# ── INT8 vs FP32 board (rework S3610–S3613) ─────────────────────────────────────

class TestInt8Fp32:
    def test_summary_has_int8_platforms(self):
        s = _load(SUMMARY)
        if s is None:
            pytest.skip("exp_S36_summary.json absent")
        for ds in DATASETS:
            for cond in CONDITIONS:
                cell = s["results"][ds][cond]
                assert {"board_frozen_int8", "board_online_int8"} <= set(cell)

    def test_int8_weight_bytes_ratio_is_four(self):
        # Fonction pure (sans board) : ratio FP32/INT8 structurel == 4.0.
        from scripts.run_sprint36_board import _ewc_weight_bytes
        for k in (1, 5, 13, 21):
            fp32_b, int8_b = _ewc_weight_bytes(k)
            assert fp32_b == 4 * int8_b
            assert int8_b == k * 32 + 32 * 16 + 16 * 2

    def test_int8_cells_consistent_when_present(self):
        s = _load(SUMMARY)
        if s is None:
            pytest.skip("exp_S36_summary.json absent")
        seen = 0
        for ds in DATASETS:
            for cond in CONDITIONS:
                for plat in ("board_frozen_int8", "board_online_int8"):
                    cell = s["results"][ds][cond][plat]
                    # Clés toujours présentes (valeurs None tant que board non exécutée).
                    assert {"ram_ratio_fp32_over_int8", "gap3_ram_ok",
                            "agreement_int8_vs_fp32", "latency_us_p50"} <= set(cell)
                    ratio = cell.get("ram_ratio_fp32_over_int8")
                    if ratio is not None:
                        seen += 1
                        assert ratio >= 3.5            # Gap 3 RAM (≈ 4×)
                        assert cell["gap3_ram_ok"] is True
                    lat = cell.get("latency_us_p50")
                    if isinstance(lat, (int, float)):
                        assert lat < GAP2_LATENCY_US    # Gap 2
                    agree = cell.get("agreement_int8_vs_fp32")
                    if agree is not None:
                        assert 0.0 <= agree <= 1.0
        if seen == 0:
            pytest.skip("aucune cellule INT8 mesurée (board non exécutée)")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
