"""Tests de la bibliothèque de figures `src/figures/` (Sprint 42, S4207).

Couvre : registre de catalogues, absence de chiffres de résultat en dur dans le
catalogue d'impact, honnêteté des loaders (``metric_or_na``), idempotence de la
génération, erreur claire sur expérience absente, placeholder « à mesurer », et
chemins de sortie normalisés.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import src.figures.catalogs  # noqa: F401 — auto-enregistrement des catalogues
from src.figures import registry
from src.figures.loaders import A_MESURER, load_experiment, metric_or_na
from src.figures.style import apply_style, savefig_png
from src.utils.reproducibility import set_seed

QUANT_CATALOGS = {
    "quantization/pedagogy",
    "quantization/pipeline",
    "quantization/impact",
    "quantization/moment",
    "quant_depth",
}
_CATALOGS_DIR = Path(__file__).resolve().parents[1] / "src/figures/catalogs"
IMPACT_SRC = _CATALOGS_DIR / "quant_impact.py"

# Constantes de mise en page autorisées (positions, largeurs de barres, alpha, tailles de
# police, limites d'axes, figsizes) — AUCUNE n'est un résultat. Un littéral hors de cette
# liste blanche fait échouer le test : c'est le garde-fou « aucun chiffre de résultat en
# dur » (règle Sprints 33/40). Scanné sur quant_impact.py (S4205) ET quant_moment.py (S4606).
LAYOUT_WHITELIST: set[float] = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.12, 0.15, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.5, 0.55, 0.6, 0.72, 0.78, 0.8, 0.82, 0.86, 0.9, 0.92, 0.94, 0.98, 1.0, 1.05, 1.2, 1.4,
    1.5, 2.0, 4.5, 5.0, 8.0, 8.5, 9.0, 11.0,
}

# Modules de catalogue soumis à la garde AST « 0 chiffre en dur ».
HARDCODE_GUARDED_SRCS: list[Path] = [
    IMPACT_SRC,
    _CATALOGS_DIR / "quant_moment.py",
    _CATALOGS_DIR / "quant_depth.py",
]


def test_registry_lists_catalogs() -> None:
    """Les 3 catalogues quantification sont enregistrés ; un catalogue jouet apparaît."""
    listed = set(registry.list_catalogs())
    assert QUANT_CATALOGS <= listed, f"catalogues manquants : {QUANT_CATALOGS - listed}"

    toy_name = "test/_toy_catalog"

    @registry.register_catalog(toy_name)
    def _toy(out_root: Path) -> list[Path]:  # pragma: no cover - jamais exécuté ici
        return []

    try:
        assert toy_name in registry.list_catalogs()
        assert registry.get_catalog(toy_name) is _toy
    finally:
        registry._CATALOGS.pop(toy_name, None)  # nettoyage de l'état global


@pytest.mark.parametrize("src", HARDCODE_GUARDED_SRCS, ids=lambda p: p.name)
def test_no_hardcoded_results(src: Path) -> None:
    """Scan AST des catalogues gardés : aucun flottant hors liste blanche de layout."""
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in LAYOUT_WHITELIST
    }
    assert not offending, (
        f"Littéraux flottants suspects dans {src.name} : {sorted(offending)} — "
        "toute valeur de résultat doit être chargée via load_experiment, pas écrite en dur."
    )


def test_loaders_na_honest() -> None:
    """metric_or_na : None/sentinel sur absent/null, jamais 0 par défaut ; 0 réel préservé."""
    assert metric_or_na({}, "x") is None                     # champ absent
    assert metric_or_na({"x": None}, "x") is None            # null (na_reason)
    assert metric_or_na({"a": {"b": None}}, "a.b") is None   # chemin pointé null
    assert metric_or_na({"x": A_MESURER}, "x") == A_MESURER  # sentinel conservé
    assert metric_or_na({"x": 0.0}, "x") == 0.0              # 0 réel non confondu avec absent
    assert metric_or_na({"a": {"b": 0.7}}, "a.b") == 0.7


def test_generate_idempotent(tmp_path: Path) -> None:
    """Deux exécutions d'un catalogue (seed fixé) → mêmes fichiers, contenu identique."""
    build = registry.get_catalog("quantization/pipeline")

    set_seed(42)
    apply_style("slide")
    out1 = tmp_path / "run1"
    paths1 = build(out1)

    set_seed(42)
    apply_style("slide")
    out2 = tmp_path / "run2"
    paths2 = build(out2)

    names1 = sorted(p.name for p in paths1)
    names2 = sorted(p.name for p in paths2)
    assert names1 == names2 and len(names1) == 5
    for name in names1:
        b1 = (out1 / "quantization/pipeline" / name).read_bytes()
        b2 = (out2 / "quantization/pipeline" / name).read_bytes()
        assert b1 == b2, f"figure non idempotente : {name}"


def test_missing_experiment_raises(tmp_path: Path) -> None:
    """Expérience source absente → FileNotFoundError clair, pas de valeur par défaut."""
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "n_existe_pas.json")


def test_a_mesurer_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans exp_S40_board_v2/, I6 rend le sentinel « à mesurer » (jamais 0)."""
    from src.figures.catalogs import quant_impact

    # Redirige la racine des expériences vers un dossier vide
    monkeypatch.setattr(quant_impact, "EXPERIMENTS_DIR", tmp_path)
    assert quant_impact._board_v2_f1("pronostia", "frozen") == A_MESURER
    assert quant_impact._board_v2_f1("monitoring", "frozen") == A_MESURER


def test_figures_output_paths(tmp_path: Path) -> None:
    """savefig_png écrit sous <out_root>/<catalog>/ et retourne le chemin produit."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = savefig_png(fig, "test/cat", "demo", out_root=tmp_path)
    assert out == tmp_path / "test/cat" / "demo.png"
    assert out.exists() and out.stat().st_size > 0
