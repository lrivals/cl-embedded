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

# Constantes de layout propres à un seul catalogue. Elles restent **locales** : les
# inscrire dans LAYOUT_WHITELIST globale affaiblirait la garde des autres fichiers,
# car des valeurs comme 0.93 ou 0.97 y seraient des AUROC parfaitement plausibles.
EXTRA_LAYOUT_WHITELIST: dict[str, set[float]] = {
    # manuscrit_final.py : figsizes, largeurs de barres, bornes d'axes, marges de
    # tight_layout/bbox_to_anchor, facteurs de headroom, epsilon d'échelle.
    "manuscrit_final.py": {
        0.001, 0.015, 0.09, 0.1, 0.27, 0.38, 0.88, 0.93, 0.97, 0.99,
        1.01, 1.1, 1.12, 1.15, 1.3, 1.45, 3.4, 3.6, 3.8, 4.0, 4.2, 7.2, 7.6, 9.6,
    },
    # soutenance.py : coordonnées de schéma, décalages d'étiquettes, alphas de mise en
    # retrait, bornes d'axes et facteurs de headroom. Aucune n'est un résultat — toutes
    # les valeurs tracées passent par src.figures.sources.
    # article_ewc.py : positions de barres, largeurs, bornes d'axes, figsizes et facteur de
    # conversion Hz→MHz. Toutes les valeurs tracées viennent de l'agrégat S4008.
    "article_ewc.py": {
        0.27, 0.85, 1.01, 4.2, 4.4, 4.6, 7.5, 10.0, 11.0, 1000000.0,
    },
    # seminaire_s44_s53.py : coordonnées des trois schémas de cadrage (frise, carte
    # axes×gaps, carte de quantification), hauteurs/largeurs de boîtes, pas de rangée,
    # marges et wspace. Aucune n'est un résultat — toutes les valeurs tracées passent
    # par load_experiment sur experiments/.
    # energy_pedagogy.py : coordonnées des schémas de principe (chaîne de mesure,
    # chronogramme, encarts de méthode), largeurs/hauteurs de boîtes, marges et
    # positions d'annotation. Aucune n'est un résultat — courants, µJ, pentes et r²
    # viennent tous de load_experiment sur experiments/exp_S53_*.
    "energy_pedagogy.py": {
        0.018, 0.04, 0.07, 0.078, 0.08, 0.09, 0.1, 0.11, 0.13, 0.135, 0.16, 0.17,
        0.18, 0.22, 0.225, 0.24, 0.26, 0.27, 0.28, 0.283, 0.34, 0.38, 0.42, 0.47,
        0.475, 0.48, 0.512, 0.543, 0.56, 0.62, 0.63, 0.645, 0.68, 0.7, 0.74, 0.76,
        0.77, 0.845, 0.85, 0.88, 0.93, 0.95, 0.97, 0.99, 1.01, 1.6,
    },
    "seminaire_s44_s53.py": {
        0.04, 0.09, 0.115, 0.125, 0.13, 0.14, 0.155, 0.185, 0.22, 0.24, 0.245, 0.26,
        0.32, 0.45, 0.62, 0.7, 0.74, 0.84, 0.95, 1.12,
    },
    "soutenance.py": {
        2e-05, 0.001, 0.0015, 0.0025, 0.015, 0.032, 0.055, 0.065, 0.08, 0.09, 0.1, 0.13, 0.14, 0.16,
        0.17, 0.18, 0.21, 0.22, 0.235, 0.24, 0.243, 0.26, 0.27, 0.275, 0.28, 0.31, 0.32,
        0.34, 0.36, 0.37, 0.38, 0.41, 0.42, 0.44, 0.45, 0.46, 0.47, 0.49, 0.51, 0.52, 0.53,
        0.565, 0.57, 0.575, 0.58, 0.62, 0.63, 0.66, 0.665, 0.68, 0.69, 0.7, 0.75, 0.76,
        0.77, 0.775, 0.795, 0.83, 0.84, 0.85, 0.87, 0.88, 0.93, 0.97, 1.02, 1.06, 1.08,
        1.1, 1.12, 1.15, 1.18, 1.42, 1.6, 1.7, 1.75, 1.8, 2.4, 2.5, 3.9, 4.4, 5.1, 5.3,
        5.6,
    },
}

# Modules de catalogue soumis à la garde AST « 0 chiffre en dur ».
HARDCODE_GUARDED_SRCS: list[Path] = [
    IMPACT_SRC,
    _CATALOGS_DIR / "quant_moment.py",
    _CATALOGS_DIR / "quant_depth.py",
    _CATALOGS_DIR / "quant_depth_board.py",
    _CATALOGS_DIR / "ram_full.py",
    _CATALOGS_DIR / "energy_real.py",
    _CATALOGS_DIR / "manuscrit_final.py",
    _CATALOGS_DIR / "soutenance.py",
    _CATALOGS_DIR / "article_ewc.py",
    _CATALOGS_DIR / "seminaire_s44_s53.py",
    _CATALOGS_DIR / "energy_pedagogy.py",
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
    allowed = LAYOUT_WHITELIST | EXTRA_LAYOUT_WHITELIST.get(src.name, set())
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in allowed
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
