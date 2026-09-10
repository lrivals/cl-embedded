"""Tests du catalogue pédagogique `energy_pedagogy` (mesure d'énergie expliquée).

Le point sensible de ce catalogue n'est pas l'esthétique mais la **non-fusion des trois
estimateurs** de µJ par inférence : leur écart est un résultat de la campagne, pas un
désaccord à moyenner. Les tests vérifient que la figure les charge séparément depuis
trois JSON distincts, et que rien n'est comblé par un zéro.

La garde AST « 0 chiffre en dur » du module vit dans ``tests/test_figures_library.py``
(liste ``HARDCODE_GUARDED_SRCS``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.figures.catalogs import energy_pedagogy as cat
from src.figures.loaders import A_MESURER
from src.figures.registry import list_catalogs
from src.figures.style import apply_style
from src.utils.reproducibility import set_seed

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "docs" / "presentation_encadrants_sept2026" / "01_slides.md"
FIGURE_RE = re.compile(r"!\[[^\]]*\]\((\.\./figures/[^)]+)\)")


def test_catalog_is_registered() -> None:
    assert cat.CATALOG in list_catalogs()


def test_build_produces_every_figure(tmp_path: Path) -> None:
    """``build`` rend exactement les figures déclarées, dans l'ordre, toutes non vides."""
    set_seed(42)
    apply_style("slide")
    paths = cat.build(tmp_path)

    assert [p.stem for p in paths] == cat.FIGURES
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_num_never_invents_a_zero() -> None:
    """Sentinel, booléen et absence deviennent ``None`` ; un vrai zéro est préservé."""
    assert cat._num(None) is None
    assert cat._num(A_MESURER) is None
    assert cat._num("na") is None
    assert cat._num(True) is None
    assert cat._num(0) == 0.0


def test_three_estimators_come_from_three_distinct_sources() -> None:
    """Les trois estimateurs sont chargés séparément — jamais fusionnés ni moyennés."""
    sources = {rel for _, rel, _, _, _ in cat.ESTIMATORS}
    assert len(sources) == len(cat.ESTIMATORS) == 3

    values = [cat._num(cat._dig(cat._load(rel), dotted))
              for _, rel, dotted, _, _ in cat.ESTIMATORS]
    if any(v is None for v in values):
        pytest.skip("agrégats S53 non produits")
    assert len(set(values)) == len(values), (
        "trois estimateurs distincts doivent rendre trois valeurs distinctes ; "
        "des valeurs égales signaleraient une fusion accidentelle des sources"
    )


def test_estimator_gap_is_ordered_by_what_each_includes() -> None:
    """delta ≥ régression ≥ lot : chaque estimateur retire un coût étranger de plus."""
    values = [cat._num(cat._dig(cat._load(rel), dotted))
              for _, rel, dotted, _, _ in cat.ESTIMATORS]
    if any(v is None for v in values):
        pytest.skip("agrégats S53 non produits")
    delta, regression, batch = values
    assert delta >= regression >= batch > 0


def test_saturation_points_are_flagged_not_dropped_silently() -> None:
    """Un point saturé reste dans le JSON avec son drapeau : il est écarté, pas effacé."""
    cell = (cat._load("experiments/exp_S53_wfi/batch_sweep.json") or {}).get("ewc_fp32")
    if cell is None:
        pytest.skip("exp_S53_wfi/batch_sweep.json non produit")
    saturated = [p for p in cell["points"] if p.get("saturated")]
    assert saturated, "le balayage de lot doit contenir au moins un point saturé"
    for point in saturated:
        assert cat._num(point.get("achieved_rate_hz")) is not None
        assert cat._num(point.get("i_mean_a")) is not None


def test_dynamic_mode_refusal_is_recorded() -> None:
    """Le refus du mode dynamique est une mesure (courant crête), pas une omission."""
    summary = cat._load("experiments/exp_S53_freq_sweep/summary.json")
    if summary is None:
        pytest.skip("exp_S53_freq_sweep/summary.json non produit")
    modes = summary["acqmode_dyn_by_mhz"]
    assert modes, "le balayage de fréquence doit consigner les tentatives d'acquisition"
    for node in modes.values():
        assert isinstance(node["succeeded"], bool)
        assert cat._num(node.get("i_max_ma")) is not None


def test_annex_figures_are_all_used_in_slides() -> None:
    """Les 10 figures de l'annexe sont toutes projetées, et pointent un PNG existant."""
    cited = set(FIGURE_RE.findall(SLIDES.read_text(encoding="utf-8")))
    own = {f"../figures/{cat.CATALOG}/{name}.png" for name in cat.FIGURES}
    assert own <= cited, f"figures non utilisées : {sorted(own - cited)}"
    for rel in own:
        assert (SLIDES.parent / rel).resolve().exists()
