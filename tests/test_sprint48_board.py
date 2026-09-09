"""test_sprint48_board.py — Clôture Sprint 48 : Gap 2/Gap 3 board, parité, honnêteté (S4807).

Tests **déterministes** avec **skip honnête** tant que la board n'a pas streamé (aucun JSON) :
ils passent en dur dès que ``run_s48_board_depth.py`` + ``aggregate_sprint48.py`` ont produit
les cellules. Aucun chiffre attendu n'est écrit — tout est vérifié depuis les JSON mesurés.

Couvre : structure de l'agrégat, Gap 2 (latence dépacking < 100 ms), Gap 3 (`.bss` packé <
non-packé), parité exacte board↔émulateur, N/A honnête, garde AST 0-chiffre sur le catalogue.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "experiments" / "exp_S48_board"
SUMMARY = ROOT / "experiments" / "exp_S48_summary.json"
CATALOG_SRC = ROOT / "src" / "figures" / "catalogs" / "quant_depth_board.py"
GAP2_LATENCY_US = 100_000
RAM_BUDGET_BYTES = 256 * 1024

# Liste blanche de flottants de mise en page (miroir test_figures_library.py).
LAYOUT_WHITELIST: set[float] = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.12, 0.15, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.5, 0.55, 0.6, 0.72, 0.78, 0.8, 0.82, 0.86, 0.9, 0.92, 0.94, 0.98, 1.0, 1.05, 1.2, 1.4,
    1.5, 2.0, 4.5, 5.0, 8.0, 8.5, 9.0, 11.0,
}


def _summary() -> dict:
    if not SUMMARY.exists():
        pytest.skip("exp_S48_summary.json absent (lancer scripts/aggregate_sprint48.py)")
    return json.loads(SUMMARY.read_text())


def _cells(summary: dict):
    """Itère (dataset, bits, cell) de l'agrégat."""
    for ds, per_bits in summary["results_by_condition"].items():
        for bits, per_gran in per_bits.items():
            for _gran, cell in per_gran.items():
                yield ds, bits, cell


def _board_results():
    return sorted(BOARD_DIR.glob("exp_S48_*/results.json"))


# ── Structure de l'agrégat ────────────────────────────────────────────────────

def test_summary_structure():
    """`exp_S48_summary.json` : [dataset][weight_bits][granularity] + board/pc/deltas."""
    s = _summary()
    assert s["sprint"] == 48
    assert "results_by_condition" in s
    for _ds, _bits, cell in _cells(s):
        assert {"board", "pc", "deltas"} <= set(cell), f"clés manquantes : {cell.keys()}"
        assert "nonpacked" in cell["board"] and "packed" in cell["board"]


# ── Gap 2 : latence dépacking < 100 ms ────────────────────────────────────────

def test_gap2_latency():
    """Toute latence DWT board renseignée < 100 000 µs (y compris chemin packé)."""
    s = _summary()
    seen = 0
    for _ds, _bits, cell in _cells(s):
        for packing in ("nonpacked", "packed"):
            sub = cell["board"].get(packing)
            if not sub:
                continue
            for key in ("latency_dwt_p50_us", "latency_dwt_p99_us"):
                lat = sub.get(key)
                if isinstance(lat, (int, float)):
                    assert lat < GAP2_LATENCY_US, f"Gap 2 violé : {key}={lat} µs"
                    seen += 1
    if seen == 0:
        pytest.skip("aucune latence board mesurée")


# ── Gap 3 : .bss packé < .bss non-packé ───────────────────────────────────────

def test_gap3_ram_packed():
    """Le packing matérialise le gain : .bss packé < .bss non-packé quand mesuré."""
    s = _summary()
    checked = 0
    for _ds, _bits, cell in _cells(s):
        b = cell["board"]
        np_bss = (b.get("nonpacked") or {}).get("bss_bytes")
        pk_bss = (b.get("packed") or {}).get("bss_bytes")
        if isinstance(np_bss, (int, float)) and isinstance(pk_bss, (int, float)):
            assert pk_bss < np_bss, f"packé {pk_bss} ≥ non-packé {np_bss}"
            assert pk_bss < RAM_BUDGET_BYTES
            checked += 1
    if checked == 0:
        pytest.skip("aucune paire packé/non-packé mesurée")


# ── Parité exacte board↔émulateur ─────────────────────────────────────────────

def test_parity_exact():
    """parity_pred == 1.000 sur les cellules mesurées (schéma émulateur == kernel)."""
    s = _summary()
    checked = 0
    for _ds, _bits, cell in _cells(s):
        for packing in ("nonpacked", "packed"):
            sub = cell["board"].get(packing)
            if not sub:
                continue
            p = sub.get("parity_pred")
            if isinstance(p, (int, float)):
                assert p == 1.0, f"parité {p} ≠ 1.000 ({_bits} {packing})"
                checked += 1
    if checked == 0:
        pytest.skip("aucune parité board mesurée")


# ── N/A honnête ───────────────────────────────────────────────────────────────

def test_na_honesty():
    """Cellules non mesurables → na_reason renseigné, jamais un chiffre fabriqué."""
    results = _board_results()
    if not results:
        pytest.skip("aucune cellule board écrite")
    for rj in results:
        r = json.loads(rj.read_text())
        p50 = r.get("latency_dwt_p50_us")
        if isinstance(p50, (int, float)):
            continue  # cellule mesurée
        # non mesurée : soit pending (« à mesurer »), soit N/A explicite
        auroc = r.get("auroc_board")
        if r.get("na_reason"):
            assert r.get("parity_pred") is None
        else:
            assert auroc == "à mesurer" or r.get("stream_mode"), f"{rj} : ni mesuré ni honnête"


# ── Garde AST 0-chiffre sur le catalogue ──────────────────────────────────────

def test_no_hardcoded_numbers():
    """Aucun flottant de résultat en dur dans le catalogue quant_depth_board."""
    tree = ast.parse(CATALOG_SRC.read_text(encoding="utf-8"))
    offending = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
        and node.value not in LAYOUT_WHITELIST
    }
    assert not offending, (
        f"Littéraux flottants suspects : {sorted(offending)} — "
        "toute valeur de résultat doit venir de load_experiment."
    )
