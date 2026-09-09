"""Tests de la mesure RAM complète `.data + .bss + pic de pile` (Sprint 49, S4907).

Verrouille la cohérence de l'agrégat `experiments/exp_S49_ram/summary.json` et de la doc :

* invariant `total == data + bss + max(pic_inf, pic_update)` (cellules board mesurées) ;
* positivité / monotonie du pic (`pic_update ≥ pic_inference` — la MAJ CL creuse plus la pile) ;
* honnêteté N/A (cellules `status:"na"` sans champ métrique fabriqué) ;
* schéma indexé `[model][dataset][condition][encoding][platform]` ;
* garde AST 0-chiffre-en-dur sur `src/figures/catalogs/ram_full.py` ;
* terminologie : plus aucune occurrence de « modulable » hors CR / specs de sprint.

Tests **indépendants de la carte** : ils valident structure et invariants, pas des valeurs de
résultat. `summary.json` est régénéré (lecture seule) au besoin par `scripts/aggregate_ram.py`.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "exp_S49_ram" / "summary.json"
RAM_FULL_SRC = ROOT / "src" / "figures" / "catalogs" / "ram_full.py"
CONDITION = "5feat"

# Littéraux de mise en page autorisés dans ram_full.py (positions, largeurs, alpha) — AUCUN
# n'est un résultat. Cohérent avec la garde AST de test_figures_library.py.
LAYOUT_WHITELIST: set[float] = {
    0.0, 0.005, 0.25, 0.35, 0.5, 0.9, 1.0,
}


def _ensure_summary() -> dict:
    if not SUMMARY.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "aggregate_ram.py")],
            check=True, cwd=ROOT, capture_output=True,
        )
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _iter_cells(summary: dict):
    """Itère (model, dataset, encoding, platform, block) sur les cellules réelles."""
    for model, ds_node in summary.items():
        if model.startswith("_"):
            continue
        for dataset, cond_node in ds_node.items():
            for encoding, plat_node in cond_node.get(CONDITION, {}).items():
                for platform, block in plat_node.items():
                    yield model, dataset, encoding, platform, block


@pytest.fixture(scope="module")
def summary() -> dict:
    return _ensure_summary()


def test_summary_indexed(summary: dict) -> None:
    """`summary.json` indexé [model][dataset][condition][encoding][platform]."""
    models = [k for k in summary if not k.startswith("_")]
    assert models, "aucun modèle indexé"
    for model in models:
        for dataset, cond_node in summary[model].items():
            assert CONDITION in cond_node, f"condition {CONDITION} absente pour {model}/{dataset}"
            for encoding, plat_node in cond_node[CONDITION].items():
                assert encoding in ("fp32", "int8")
                assert set(plat_node) <= {"board", "pc"}


def test_total_equals_components(summary: dict) -> None:
    """total == data + bss + max(pic_inf, pic_update) sur cellules board `done`."""
    checked = 0
    for model, ds, enc, plat, b in _iter_cells(summary):
        if plat != "board" or b.get("status") != "done":
            continue
        data, bss = b.get("data"), b.get("bss")
        peaks = [b.get("stack_peak_inference"), b.get("stack_peak_update")]
        peaks = [p for p in peaks if isinstance(p, (int, float))]
        if not (isinstance(data, (int, float)) and isinstance(bss, (int, float)) and peaks):
            continue
        expected = data + bss + max(peaks)
        assert b["total"] == expected, (
            f"{model}/{ds}/{enc}/{plat} : total {b['total']} != {expected}"
        )
        checked += 1
    assert checked > 0, "aucune cellule board complète vérifiée"


def test_stack_peak_positive(summary: dict) -> None:
    """pic ≥ 0 ; pic_update ≥ pic_inference (la MAJ CL creuse plus la pile)."""
    for model, ds, enc, plat, b in _iter_cells(summary):
        pi = b.get("stack_peak_inference")
        pu = b.get("stack_peak_update")
        if isinstance(pi, (int, float)):
            assert pi >= 0
        if isinstance(pu, (int, float)):
            assert pu >= 0
        if isinstance(pi, (int, float)) and isinstance(pu, (int, float)):
            assert pu >= pi, f"{model}/{ds}/{enc}/{plat} : pic MAJ {pu} < pic inférence {pi}"


def test_na_honesty(summary: dict) -> None:
    """Cellules status:"na" : aucun champ métrique fabriqué (tous null)."""
    METRIC_KEYS = ("data", "bss", "stack_peak_inference", "stack_peak_update", "total")
    seen_na = False
    for model, ds, enc, plat, b in _iter_cells(summary):
        if b.get("status") != "na":
            continue
        seen_na = True
        for k in METRIC_KEYS:
            assert b.get(k) is None, f"{model}/{ds}/{enc}/{plat} : champ {k} fabriqué sur N/A"
        assert b.get("na_reason"), "cellule N/A sans na_reason"
    assert seen_na, "aucune cellule N/A trouvée (schéma attendu en contient)"


def test_no_hardcoded_numbers() -> None:
    """Garde AST : aucun flottant de résultat en dur dans ram_full.py."""
    tree = ast.parse(RAM_FULL_SRC.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in LAYOUT_WHITELIST
    }
    assert not offending, (
        f"Littéraux flottants suspects dans ram_full.py : {sorted(offending)} — "
        "toute valeur de résultat doit venir de load_experiment."
    )


def test_terminology() -> None:
    """Plus aucune occurrence de « modulable » hors CR et specs de sprint."""
    pat = re.compile(r"modulable", re.IGNORECASE)
    offenders: list[str] = []
    for sub in ("docs", "src", "scripts"):
        for path in (ROOT / sub).rglob("*"):
            if path.suffix not in (".md", ".py") or not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            # Le CR énonce l'instruction ; les specs sprint_49 la décrivent — exclus.
            if "CR_reunion_16juillet2026" in rel or "sprint_49" in rel:
                continue
            if pat.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)
    assert not offenders, f"« modulable » subsiste hors CR/specs : {offenders}"
