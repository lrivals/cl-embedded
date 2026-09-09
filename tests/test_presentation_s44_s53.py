"""Tests du support de présentation encadrants — sprints 44 → 53.

Vérifie le catalogue de figures ``seminaire_s44_s53`` (enregistrement, complétude,
honnêteté des cellules non mesurées) et la cohérence entre les slides Markdown et les
PNG qu'elles citent. La garde AST « 0 chiffre en dur » du module de catalogue vit dans
``tests/test_figures_library.py`` (liste ``HARDCODE_GUARDED_SRCS``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.figures.catalogs import seminaire_s44_s53 as cat
from src.figures.loaders import A_MESURER
from src.figures.registry import list_catalogs
from src.figures.style import apply_style
from src.utils.reproducibility import set_seed

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "docs" / "presentation_encadrants_sept2026"
SLIDES = DOC_DIR / "01_slides.md"
INDEX = DOC_DIR / "00_index.md"
FIGURE_RE = re.compile(r"!\[[^\]]*\]\((\.\./figures/[^)]+)\)")


def test_catalog_is_registered() -> None:
    """Le catalogue s'auto-enregistre à l'import de ``src.figures.catalogs``."""
    assert cat.CATALOG in list_catalogs()


def test_build_produces_every_figure(tmp_path: Path) -> None:
    """``build`` rend exactement les figures déclarées, toutes non vides et uniques."""
    set_seed(42)
    apply_style("slide")
    paths = cat.build(tmp_path)

    assert len(paths) == len(cat.FIGURES)
    names = [p.stem for p in paths]
    assert names == cat.FIGURES, "l'ordre de build doit suivre FIGURES"
    assert len(set(names)) == len(names)
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_missing_value_is_not_drawn_as_zero() -> None:
    """Une grandeur absente ou « à mesurer » devient ``None`` — jamais 0.0."""
    assert cat._num(None) is None
    assert cat._num(A_MESURER) is None
    assert cat._num("na") is None
    assert cat._num(True) is None, "un booléen n'est pas une mesure"
    assert cat._num(0) == 0.0, "un zéro réel doit être préservé"
    assert cat._is_na(A_MESURER) and not cat._is_na("na")


def test_dig_returns_none_on_broken_path() -> None:
    """Le descendeur de chemin pointé ne fabrique rien quand un maillon manque."""
    data = {"a": {"b": 1}}
    assert cat._dig(data, "a.b") == 1
    assert cat._dig(data, "a.c") is None
    assert cat._dig(data, "a.b.c") is None
    assert cat._dig(None, "a") is None


def test_na_honesty_in_sources() -> None:
    """Au moins une cellule non mesurée porte sa raison dans le JSON source."""
    s45 = cat._load("experiments/exp_S45_summary.json")
    if s45 is None:
        pytest.skip("exp_S45_summary.json non produit")
    psi = s45["results"]["gas_sensor_drift"]["psi"]["board"]
    assert psi["measured"] is False
    assert psi["na_reason"], "une cellule non mesurée doit dire pourquoi"
    assert psi["latency_us_p50"] is None, "N/A ne doit jamais être écrit comme 0"


def test_gap2_latencies_stay_under_budget() -> None:
    """Toutes les latences carte citées restent très loin du budget de 100 ms."""
    s45 = cat._load("experiments/exp_S45_summary.json")
    s48 = cat._load("experiments/exp_S48_summary.json")
    if s45 is None or s48 is None:
        pytest.skip("agrégats S45 / S48 non produits")

    budget_us = cat._num(s45["gap2_latency_us"])
    assert budget_us is not None

    measured: list[float] = []
    for detectors in s45["results"].values():
        for cell in detectors.values():
            value = cat._num((cell.get("board") or {}).get("latency_us_p50"))
            if value is not None:
                measured.append(value)
    for by_bits in s48["results_by_condition"].values():
        for cell in by_bits.values():
            board = cell["per_channel"]["board"]
            for packing in ("nonpacked", "packed"):
                value = cat._num((board.get(packing) or {}).get("latency_dwt_p50_us"))
                if value is not None:
                    measured.append(value)

    assert measured, "aucune latence carte trouvée"
    assert max(measured) < budget_us


def test_every_figure_cited_in_slides_exists() -> None:
    """Chaque ``![](...)`` des slides pointe un PNG réellement généré."""
    cited = FIGURE_RE.findall(SLIDES.read_text(encoding="utf-8"))
    assert cited, "les slides doivent citer des figures"
    for rel in cited:
        assert (DOC_DIR / rel).resolve().exists(), f"figure manquante : {rel}"


def test_own_figures_are_all_used() -> None:
    """Les 14 figures propres au catalogue sont toutes projetées au moins une fois."""
    cited = set(FIGURE_RE.findall(SLIDES.read_text(encoding="utf-8")))
    own = {f"../figures/{cat.CATALOG}/{name}.png" for name in cat.FIGURES}
    assert own <= cited, f"figures non utilisées : {sorted(own - cited)}"


def test_index_documents_traceability() -> None:
    """L'index rappelle la commande de régénération et la règle « jamais un zéro »."""
    index = INDEX.read_text(encoding="utf-8")
    assert "generate_figures.py --catalog seminaire_s44_s53" in index
    assert A_MESURER in index
