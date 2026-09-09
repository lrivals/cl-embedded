"""Tests du catalogue `soutenance` et de la matrice d'oubli qui l'alimente.

Trois propriétés sont vérifiées, dans l'ordre de ce qui casserait le plus discrètement :

1. **La matrice d'oubli est honnête** — les tâches pas encore vues portent ``None``,
   jamais ``0.0``. Un zéro se tracerait comme un F1 nul mesuré et raconterait un oubli
   qui n'a pas eu lieu.
2. **La figure B9 ne montre que les politiques à gate** — c'est la demande explicite
   (retirer les colonnes N/A des deux panneaux), et elle est vérifiable sur les données.
3. **Le catalogue produit bien ses figures** et ne trace pas 0 pour une donnée absente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.figures import sources
from src.figures.catalogs import soutenance
from src.figures.registry import list_catalogs

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


# ── 1. La matrice d'oubli mesurée ────────────────────────────────────────────

FORGETTING_ARMS = ["ewc", "naive"]


def _forgetting(arm: str) -> dict:
    path = EXPERIMENTS / f"exp_S54_forgetting_{arm}" / "results.json"
    if not path.exists():
        pytest.skip(f"exp_S54_forgetting_{arm} non produit")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", FORGETTING_ARMS)
def test_forgetting_matrix_is_triangular_with_none(arm: str) -> None:
    """Hors du triangle vu, la matrice porte ``None`` — jamais un 0 de remplissage."""
    data = _forgetting(arm)
    matrix = data["task_f1_per_epoch"]
    n_tasks = data["n_tasks"]
    n_epochs = data["n_epochs_per_task"]
    assert len(matrix) == n_tasks * n_epochs
    for epoch, row in enumerate(matrix):
        assert len(row) == n_tasks
        seen = epoch // n_epochs          # tâches déjà rencontrées à cette époque
        for task, value in enumerate(row):
            if task <= seen:
                assert isinstance(value, (int, float)), (epoch, task, value)
            else:
                assert value is None, (
                    f"époque {epoch}, tâche {task} : {value!r} — une tâche pas encore "
                    "vue doit être None, pas un F1 de remplissage"
                )


@pytest.mark.parametrize("arm", FORGETTING_ARMS)
def test_forgetting_matrix_shows_a_collapse(arm: str) -> None:
    """Le F1 de la tâche 0 s'effondre à la bascule : c'est le message de la slide 4."""
    data = _forgetting(arm)
    matrix = data["task_f1_per_epoch"]
    n_epochs = data["n_epochs_per_task"]
    task0 = [row[0] for row in matrix]
    peak_before = max(task0[:n_epochs])
    after = task0[n_epochs]
    assert peak_before > after, "aucune chute mesurée — la figure n'aurait rien à montrer"


def test_forgetting_arms_differ_by_lambda_only() -> None:
    """Les deux bras ne diffèrent que par λ : sinon le contraste tracé serait confondu."""
    ewc, naive = _forgetting("ewc"), _forgetting("naive")
    assert ewc["ewc_lambda"] != naive["ewc_lambda"]
    for key in ("n_tasks", "n_classes", "n_epochs_per_task", "dataset"):
        assert ewc[key] == naive[key]


# ── 2. La figure de secours B9 ───────────────────────────────────────────────

def _s38() -> dict:
    path = EXPERIMENTS / "exp_S38_summary.json"
    if not path.exists():
        pytest.skip("exp_S38_summary.json absent")
    return json.loads(path.read_text(encoding="utf-8"))


def test_b9_keeps_only_gated_policies() -> None:
    """Les politiques retirées sont exactement celles sans verdict — pas un tri arbitraire.

    ``frozen`` et ``always`` n'ont pas de gate : leur ``verdict_parity_rate`` est ``null``
    par construction. La figure les écarte ; ce test vérifie que le critère de tri
    coïncide bien avec l'absence de verdict dans les données.
    """
    summary = _s38()
    results = summary["results"]
    gated = [p for p in summary["policies"] if p.startswith("gated")]
    dropped = [p for p in summary["policies"] if p not in gated]
    assert gated and dropped

    for dataset in summary["datasets"]:
        for init in summary["init_modes"]:
            for policy in gated:
                cell = results[dataset][init][policy]["board"]
                assert cell.get("verdict_parity_rate") is not None, (
                    f"{dataset}/{init}/{policy} : politique gardée mais sans verdict"
                )
            for policy in dropped:
                cell = results[dataset][init][policy]["board"]
                assert cell.get("verdict_parity_rate") is None, (
                    f"{dataset}/{init}/{policy} : politique retirée alors qu'elle a un "
                    "verdict — le critère de tri de la figure serait faux"
                )


# ── 3. Le catalogue ──────────────────────────────────────────────────────────

def test_catalog_is_registered() -> None:
    assert soutenance.CATALOG in list_catalogs()


def test_build_produces_every_figure(tmp_path: Path) -> None:
    """Le catalogue produit une PNG non vide par entrée de :data:`FIGURES`."""
    paths = soutenance.build(tmp_path)
    assert len(paths) == len(soutenance.FIGURES)
    assert len({p.name for p in paths}) == len(paths), "deux figures partagent un nom"
    for path in paths:
        assert path.exists() and path.stat().st_size > 0, path


def test_missing_value_is_not_drawn_as_zero() -> None:
    """``None`` traverse les helpers sans jamais devenir 0."""
    assert soutenance._fr(None) == "N/A"
    assert soutenance._int_fr(None) == "N/A"
    assert soutenance._bytes(None) == "N/A"
    assert soutenance._pct(None, 100) == ""
    assert sources.num(None) is None
    assert sources.num(True) is None, "un booléen tracé comme hauteur serait inventé"
    assert sources.first_num({"a": None, "b": 3}, "a", "b") == 3


def test_stacked_row_skips_unknown_segments() -> None:
    """Un segment de largeur inconnue est absent, pas complété par une valeur inventée."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    total = soutenance._stacked_row(ax, 0, [(10, "#000000", ""), (None, "#111111", "")])
    plt.close(fig)
    assert total == 10


def test_figures_are_named_after_their_slide(tmp_path: Path) -> None:
    """Chaque PNG porte le numéro de la slide qu'elle sert (``s13_…``, ``b9_…``).

    C'est ce qui permet de vérifier d'un coup d'œil, avant la soutenance, qu'aucune
    slide ne pointe vers une figure qui n'existe plus.
    """
    import re

    names = sorted(p.stem for p in soutenance.build(tmp_path))
    pattern = re.compile(r"^(s\d{1,2}|b\d{1,2})_[a-z0-9_]+$")
    for name in names:
        assert pattern.match(name), f"{name} ne porte pas de numéro de slide exploitable"
    slides = {int(re.match(r"^[sb](\d{1,2})_", n).group(1)) for n in names}
    assert {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24} <= slides, (
        "les douze slides reprises du manuscrit doivent toutes avoir leur version soutenance"
    )


# ── 4. Intégrité des renvois de figures dans docs/soutenance/ ────────────────

#: Captures de `Kirkpatrick2017EWC` déposées à la main dans le dossier du catalogue.
#: Elles ne sont pas régénérables : si elles disparaissent, trois slides perdent leur
#: appui et personne ne s'en apercevrait avant la projection.
HAND_PROVIDED_ASSETS = {
    "docs/figures/soutenance/KirkArticle_Frontpage.png",
    "docs/figures/soutenance/KirkArticle_Plot.png",
    "docs/figures/soutenance/KirkArticle_SchemaLoss.png",
    "docs/figures/soutenance/MaintenancePredictive_Intro.png",
}

#: Assets encore attendus. Ils sont cités par les documents mais pas encore déposés :
#: le test d'intégrité des renvois les tolère, celui de présence ne les couvre pas encore.
#: À la livraison, les déplacer vers HAND_PROVIDED_ASSETS.
PENDING_ASSETS: set[str] = set()


def test_hand_provided_assets_are_present() -> None:
    """Les captures non régénérables sont là — une régénération ne les recrée pas."""
    missing = [a for a in sorted(HAND_PROVIDED_ASSETS) if not (ROOT / a).exists()]
    assert not missing, (
        "captures fournies à la main absentes (le catalogue ne peut pas les reproduire) :\n  "
        + "\n  ".join(missing)
    )


def test_generated_figures_never_shadow_hand_assets(tmp_path: Path) -> None:
    """Aucune figure générée ne porte le nom d'une capture déposée à la main."""
    generated = {p.name for p in soutenance.build(tmp_path)}
    clashes = generated & {Path(a).name for a in HAND_PROVIDED_ASSETS}
    assert not clashes, f"le catalogue écraserait : {sorted(clashes)}"


def test_every_figure_cited_in_docs_exists() -> None:
    """Aucune slide ne renvoie vers une figure absente.

    C'est le test qui protège le jour J : une figure renommée ou supprimée casserait
    silencieusement un renvoi de `S02`/`S03`, et cela ne se verrait qu'en projection.
    """
    import re

    docs = sorted((ROOT / "docs" / "soutenance").glob("*.md"))
    assert docs, "docs/soutenance/ est vide"
    pattern = re.compile(r"`(docs/(?:figures|soutenance)/[^`]+\.png)`")

    missing: list[str] = []
    for doc in docs:
        for cited in pattern.findall(doc.read_text(encoding="utf-8")):
            if cited in PENDING_ASSETS:
                continue
            if not (ROOT / cited).exists():
                missing.append(f"{doc.name} → {cited}")
    assert not missing, "figures citées mais absentes :\n  " + "\n  ".join(missing)


def test_docs_no_longer_cite_manuscript_figures() -> None:
    """Les slides ne renvoient plus vers `manuscrit_overleaf/images/`.

    Le but de la refonte est que le jury ne revoie pas en projection les figures qu'il a
    déjà lues dans le manuscrit : un renvoi qui y retournerait annulerait le travail.
    """
    offenders: list[str] = []
    for doc in sorted((ROOT / "docs" / "soutenance").glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "manuscrit_overleaf/images/" in line or ".../images/" in line:
                # Une mention explicative reste permise si elle ne sert pas de renvoi.
                if "Figure" in line or "Figures" in line:
                    offenders.append(f"{doc.name}:{line_no}")
    assert not offenders, "renvois vers les figures du manuscrit :\n  " + "\n  ".join(offenders)


# ── 5. Non-régression des données consolidées (volet H) ─────────────────────

def _comparison() -> dict:
    path = EXPERIMENTS / "comparison_sprint23.json"
    if not path.exists():
        pytest.skip("comparison_sprint23.json absent")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "dataset", ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"]
)
def test_b2_tinyol_not_duplicate_of_mahalanobis(dataset: str) -> None:
    """Garde-fou du bug Sprint 52 : TinyOL carte ne doit pas répliquer Mahalanobis.

    Le drapeau UART TinyOL manquant faisait emprunter à la campagne le chemin
    d'inférence par défaut du firmware — Mahalanobis. Les deux colonnes devenaient
    alors identiques, et la grille B2 projetait des valeurs invalidées. Si ce test
    échoue, `comparison_sprint23.json` a été régénéré depuis des cellules périmées.
    """
    grid = _comparison()["results_by_condition"]["5feat"][dataset]
    maha = grid["mahalanobis"]["nucleo_f439zi"]["f1_faulty"]
    tinyol = grid["tinyol"]["nucleo_f439zi"]["f1_faulty"]
    if maha is None or tinyol is None:
        pytest.skip(f"{dataset} : cellule non mesurée")
    assert maha != tinyol, (
        f"{dataset} : TinyOL carte ({tinyol}) réplique Mahalanobis ({maha}) — "
        "régénérer experiments/comparison_sprint23.json"
    )


def test_s11_latencies_come_from_the_same_source_as_s20() -> None:
    """Les latences de la slide 11 sont celles de la slide 20, pas des valeurs figées.

    La figure héritée affichait « 130 µs / 403 µs » (Sprint 26) alors que le reste de
    l'exposé annonce 48–50 µs / 239–251 µs. Les deux slides lisent désormais le même
    nœud, donc ne peuvent plus diverger.
    """
    node = sources.s36_node("monitoring", "board_online", condition="5feat")
    assert node, "exp_S36_summary absent"
    inference = sources.num(node.get("latency_inference_only_us_p50"))
    total = sources.num(node.get("latency_us_p50"))
    assert inference is not None and total is not None
    assert total > inference, "le total doit inclure le surcoût de mise à jour"
    # Les valeurs héritées ne doivent plus apparaître nulle part.
    assert inference != 130 and total != 403


def test_s6_gap2_wording_is_total_ram() -> None:
    """Le libellé Gap 2 dit « RAM totale », pas « `.bss` ».

    Annoncer `.bss` en ouverture contredirait les slides 17-18, qui démontrent que
    `.bss` seul sous-estime la RAM réellement occupée.
    """
    gap2 = dict((tag, (title, body)) for tag, title, body in soutenance.GAPS)["Gap 2"]
    joined = " ".join(gap2).lower()
    assert "ram totale" in joined
    assert "pic de pile" in joined
