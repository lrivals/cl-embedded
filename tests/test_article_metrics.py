"""Tests de l'agrégat de métriques et du catalogue de figures de l'article EWC (S4008/S4009).

Ce que ces tests protègent, au-delà du schéma :

  * **la non-fusion des estimateurs d'énergie** — le dépôt interdit de moyenner la régression de
    cadence, le delta/WFI et le lot `INFER_BATCH_N` : leur désaccord est un résultat, et une
    régression logicielle qui les agrégerait passerait sinon inaperçue ;
  * **l'honnêteté des N/A** — une mesure absente vaut ``None`` ou le sentinel ``"à mesurer"``,
    jamais 0 ; c'est la règle qui distingue « pas encore mesuré » de « mesuré à zéro » ;
  * **l'absence de chiffre en dur** dans le catalogue de figures (garde AST), les valeurs devant
    toutes transiter par l'agrégat.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "experiments" / "exp_S40_article_metrics" / "summary.json"
CATALOG_SRC = ROOT / "src" / "figures" / "catalogs" / "article_ewc.py"

DATASETS = ("monitoring", "pronostia")
AXES = ("performance", "ablation_int8", "recovery_board", "moment", "depth_pc",
        "depth_board", "ram", "latency", "compute_cost")
PLATFORMS = {"mesuré board", "émulé PC", "mesuré PC", "théorique"}
A_MESURER = "à mesurer"
#: Les trois estimateurs d'énergie qui doivent rester séparés (S5302/S5304).
REQUIRED_ESTIMATORS = {"regression", "delta_wfi", "batch"}


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip("agrégat absent (lancer scripts/aggregate_article_ewc.py)")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _iter_cells(node, prefix=""):
    """Parcourt les cellules ``{value, source_json, platform, na_reason}`` de l'agrégat."""
    if isinstance(node, dict):
        if "value" in node and "platform" in node:
            yield prefix, node
            return
        for k, v in node.items():
            if not k.startswith("_"):
                yield from _iter_cells(v, f"{prefix}.{k}" if prefix else k)


def test_schema_covers_the_five_metric_families(summary: dict) -> None:
    """Les 9 axes existent pour les 2 jeux de l'article, et couvrent les 5 familles demandées."""
    for ds in DATASETS:
        assert ds in summary, f"jeu manquant : {ds}"
        for axis in AXES:
            assert axis in summary[ds], f"axe manquant : {ds}.{axis}"
            assert summary[ds][axis], f"axe vide : {ds}.{axis}"
    assert "energy" in summary and "estimators" in summary["energy"]
    assert "context" in summary and "forgetting" in summary["context"]


def test_every_cell_is_traceable(summary: dict) -> None:
    """Chaque cellule porte sa source et une plateforme reconnue — pas de valeur orpheline."""
    bad = []
    for path, cell in _iter_cells({d: summary[d] for d in DATASETS}):
        if cell["platform"] not in PLATFORMS:
            bad.append(f"{path} : plateforme {cell['platform']!r}")
        if not cell.get("source_json"):
            bad.append(f"{path} : source absente")
    assert not bad, "cellules non traçables : " + "; ".join(bad[:10])


def test_energy_estimators_are_never_merged(summary: dict) -> None:
    """Les estimateurs restent séparés, chacun avec sa note de méthode.

    Un agrégat qui fusionnerait deux méthodes (moyenne, valeur « énergie » unique hors
    ``estimators``) violerait la consigne portée par les JSON S53 eux-mêmes.
    """
    est = summary["energy"]["estimators"]
    assert REQUIRED_ESTIMATORS <= set(est), f"estimateurs manquants : {REQUIRED_ESTIMATORS - set(est)}"
    for name in REQUIRED_ESTIMATORS:
        assert est[name].get("method_note"), f"{name} sans note de méthode"
        assert est[name].get("cells"), f"{name} sans cellule"
    # Aucune clé d'énergie agrégée ne doit exister à côté du bloc des estimateurs.
    assert set(summary["energy"]) <= {"_note", "estimators"}, \
        "une valeur d'énergie vit hors des estimateurs (fusion ?)"
    # Les valeurs des estimateurs diffèrent : les confondre effacerait ce constat.
    values = {
        name: est[name]["cells"].get("ewc_fp32_energy_uj_per_inference", {}).get("value")
        for name in REQUIRED_ESTIMATORS
    }
    measured = [v for v in values.values() if isinstance(v, (int, float))]
    assert len(set(measured)) == len(measured), f"valeurs identiques entre estimateurs : {values}"


def test_na_is_honest_never_zero(summary: dict) -> None:
    """Une cellule non mesurée vaut None ou « à mesurer » + raison — jamais 0."""
    for path, cell in _iter_cells(summary["energy"]["estimators"], "energy"):
        if cell["value"] is None or cell["value"] == A_MESURER:
            assert cell.get("na_reason"), f"{path} : N/A sans raison"
    # Le sentinel littéral est préservé tel quel (il n'est pas converti en None ni en 0).
    updates = summary["energy"]["estimators"]["delta_wfi"]["cells"]
    per_update = updates["ewc_fp32_energy_uj_per_update"]["value"]
    assert per_update is None or per_update == A_MESURER
    assert per_update != 0 and per_update != 0.0


def test_missing_lists_only_unmeasured_cells(summary: dict) -> None:
    """`missing` énumère exactement les cellules sans mesure — ni plus, ni moins."""
    missing = set(summary["missing"])
    assert missing, "aucune cellule manquante listée : la campagne est-elle vraiment complète ?"
    for path in missing:
        node = summary
        for key in path.split("."):
            node = node[key]
        assert node["value"] is None or node["value"] == A_MESURER, \
            f"{path} listée manquante alors qu'elle porte {node['value']!r}"


def test_compute_cost_matches_measured_params(summary: dict) -> None:
    """Le coût de calcul décrit bien la tête réellement portée (garde-fou d'architecture).

    Le compte de paramètres doit coïncider avec celui mesuré au Sprint 39 : sinon les MACs/BOPs
    décriraient une autre architecture que celle quantifiée et flashée.
    """
    for ds in DATASETS:
        cc = summary[ds]["compute_cost"]
        assert cc["n_params_matches_s39"]["value"] is True, \
            f"{ds} : n_params {cc['n_params']['value']} != S39 {cc['n_params_s39_reference']['value']}"
        assert cc["bops_ratio_fp32_over_int8"]["value"] == 16
        assert cc["bops_fp32"]["platform"] == "théorique"


def test_forgetting_context_is_not_mixed_in(summary: dict) -> None:
    """L'oubli (S54) est mesuré sur CWRU : il reste hors des lignes Pronostia/Monitoring."""
    ctx = summary["context"]["forgetting"]
    assert ctx["ewc_dataset"]["value"] == "cwru"
    for ds in DATASETS:
        joined = json.dumps(summary[ds], ensure_ascii=False)
        assert "avg_forgetting_f1" not in joined, f"métrique d'oubli CWRU mélangée dans {ds}"


def test_catalog_has_no_hardcoded_results() -> None:
    """Garde AST : aucun littéral flottant de résultat dans le catalogue de figures."""
    tree = ast.parse(CATALOG_SRC.read_text(encoding="utf-8"))
    # Constantes de mise en page uniquement (positions, largeurs de barres, bornes, figsizes).
    allowed = {0.0, 0.2, 0.25, 0.27, 0.3, 0.35, 0.5, 0.55, 0.6, 0.8, 0.85, 0.9, 1.0, 1.01,
               1.5, 4.2, 4.4, 4.6, 7.5, 10.0, 11.0, 1e6}
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node.value if isinstance(node, ast.Constant) else None, float)
        and node.value not in allowed
    }
    assert not offending, (
        f"littéraux flottants suspects dans {CATALOG_SRC.name} : {sorted(offending)} — "
        "toute valeur de résultat doit venir de l'agrégat, pas être écrite en dur."
    )


def test_catalog_is_idempotent(tmp_path: Path) -> None:
    """Deux exécutions vers un répertoire temporaire produisent des PNG identiques.

    L'appel hors répertoire canonique ne doit rien synchroniser vers l'article : c'est ce qui
    permet de tester la génération sans écraser les figures publiées.
    """
    import matplotlib
    matplotlib.use("Agg")

    import src.figures.catalogs  # noqa: F401 — auto-enregistrement
    from src.figures import registry
    from src.figures.style import apply_style
    from src.utils.reproducibility import set_seed

    build = registry.get_catalog("article_ewc_int8")
    runs = []
    for name in ("run1", "run2"):
        set_seed(42)
        apply_style("manuscript")
        runs.append(build(tmp_path / name))

    assert len(runs[0]) == 10, f"10 figures attendues, {len(runs[0])} produites"
    for a, b in zip(runs[0], runs[1]):
        assert a.name == b.name
        assert a.read_bytes() == b.read_bytes(), f"{a.name} non déterministe"
    # Rien n'a fui vers les répertoires publiés depuis un out_root temporaire.
    assert not (ROOT / "docs" / "figures" / "run1").exists()
