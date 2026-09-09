"""Tests de l'isolation à variable unique de l'effet « build » (S5306, B3/B8).

Ce que ces tests verrouillent : la règle de décision, pas la mesure. Un verdict qui
conclurait sur une variable que la paire n'a pas fait varier, ou une différence de pentes
présentée comme une énergie, sont exactement les deux façons dont cette expérience peut
mentir — les deux sont testées ici, hors banc.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.evaluation import build_isolation as bi
from src.evaluation.rate_regression import A_MESURER, R2_MIN

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp_S53_build_isolation"


def _cellule(slope: float, std: float, r2: float, energie=A_MESURER) -> dict:
    return {"slope_ua_per_hz": slope, "slope_std_ua_per_hz": std, "r2": r2,
            "energy_uj_per_inference": energie, "tension_v": 3.26}


# ── 1. Les dimensions du binaire ────────────────────────────────────────────

def test_les_quatre_dimensions_bougent_ensemble():
    """C'est le JEU de dimensions du build S38 qui est mis en cause, pas EWC_IN seul."""
    dims = bi.make_dims(6)
    assert dims == ["EWC_IN=6", "MAHA_DIM=6", "TINYOL_IN=6", "HDC_N_FEATURES=6"]


def test_proto_max_n_seulement_au_dela_de_seize():
    assert not any(d.startswith("PROTO_MAX_N") for d in bi.make_dims(16))
    assert "PROTO_MAX_N=21" in bi.make_dims(21)


def test_la_paire_b8_ne_fait_varier_que_la_dimension():
    a, b = bi.PAIRS["B8"]["cells"]
    va, vb = bi.VARIANTS[a], bi.VARIANTS[b]
    assert va["ewc_in"] != vb["ewc_in"], "la dimension doit différer"
    assert va["weights_state"] == vb["weights_state"] == "repli Xavier", \
        "les poids doivent être dans le même état des deux côtés"
    assert va["extra_cflags"] == vb["extra_cflags"] == ""


# ── 2. Lecture d'un coût marginal dans une cellule ──────────────────────────

def test_une_regression_publiable_porte_une_dependance_presente():
    presence = bi.slope_presence(_cellule(20.73, 0.21, 0.9996, energie=67.7))
    assert presence["state"] == bi.PRESENTE
    assert presence["sub_state"] is None


@pytest.mark.parametrize("slope,std,r2,sous", [
    (0.30, 0.62, 0.36, "pente non séparable de zéro"),
    (-1.071, 0.406, 0.635, "pente négative"),
    (1.673, 0.750, 0.554, "nuage non linéaire"),
])
def test_une_regression_non_publiable_ne_livre_aucun_cout_marginal(slope, std, r2, sous):
    """Trois régimes distincts, un seul état : aucun ne permet de lire un coût marginal."""
    presence = bi.slope_presence(_cellule(slope, std, r2))
    assert presence["state"] == bi.ABSENTE
    assert presence["sub_state"] == sous
    assert f"{r2:.3f}" in presence["rationale"], "la raison doit être CHIFFRÉE"


def test_cellule_absente_ne_conclut_pas():
    assert bi.slope_presence(None)["state"] == bi.INDETERMINEE
    assert bi.slope_presence({"slope_ua_per_hz": 1.0})["state"] == bi.INDETERMINEE


def test_le_seuil_de_linearite_est_celui_de_s5304():
    """Une seule règle gouverne « une pente décrit-elle un coût marginal ? »."""
    juste_sous = bi.slope_presence(_cellule(20.0, 0.2, R2_MIN - 0.001, energie=65.0))
    assert juste_sous["state"] == bi.ABSENTE


# ── 3. Verdict apparié ──────────────────────────────────────────────────────

def test_present_dun_cote_absent_de_lautre_fait_suivre_la_variable():
    verdict = bi.paired_verdict(_cellule(20.7, 0.21, 0.9996, energie=67.7),
                                _cellule(1.67, 0.75, 0.554),
                                "k6", "k4", varied="la dimension compilée",
                                same_session=True)
    assert verdict["verdict"] == "l'effet suit la dimension compilée"
    assert "dimension" in verdict["rationale"]


def test_meme_etat_des_deux_cotes_ecarte_la_variable():
    """C'est la conclusion mesurée de B3 : les poids sont écartés comme cause."""
    verdict = bi.paired_verdict(_cellule(-1.071, 0.406, 0.635),
                                _cellule(1.673, 0.750, 0.554),
                                "poids", "xavier",
                                varied="les poids de la tête EWC", same_session=False)
    assert verdict["verdict"] == "l'effet ne suit pas les poids de la tête EWC"
    assert "dimension" not in verdict["verdict"], \
        "une paire ne conclut que sur ce qu'elle a fait varier"


def test_une_cellule_manquante_laisse_le_verdict_indetermine():
    verdict = bi.paired_verdict(_cellule(20.7, 0.21, 0.9996, energie=67.7), None,
                                "k6", "k4", varied="la dimension compilée")
    assert verdict["verdict"] == bi.VERDICT_INDETERMINE


def test_la_difference_de_pentes_nest_jamais_publiee_comme_une_energie():
    """Règle A4 : une différence de pentes ne devient un µJ que pour une paire de
    politiques dont chaque régression est linéaire. Ici l'objet comparé est un binaire."""
    verdict = bi.paired_verdict(_cellule(20.7, 0.21, 0.9996, energie=67.7),
                                _cellule(1.67, 0.75, 0.554),
                                "k6", "k4", varied="la dimension compilée")
    assert verdict["delta_publishable_as_energy"] is False
    assert verdict["delta_uj_equivalent"] == A_MESURER
    assert verdict["delta_na_reason"]
    assert verdict["delta_sigma"] > 0


def test_une_paire_inter_seance_porte_sa_reserve():
    verdict = bi.paired_verdict(_cellule(20.7, 0.21, 0.9996, energie=67.7),
                                _cellule(0.2, 0.6, 0.30),
                                "k6", "k4", varied="la dimension compilée",
                                same_session=False)
    assert "séances distinctes" in verdict["reserve"]


# ── 4. Garde AST et honnêteté des sorties mesurées ──────────────────────────

def test_pilote_sans_resultat_en_dur():
    """Aucune valeur de mesure figée dans le pilote : les seuils vivent dans le module."""
    src = (ROOT / "scripts" / "run_s53_build_isolation.py").read_text(encoding="utf-8")
    autorises = {0.0, 1.0, 2.0, 3.3, 10.0, 100.0}
    suspects = {node.value for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.Constant) and isinstance(node.value, float)
                and node.value not in autorises}
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"


def test_le_module_ne_fige_aucun_seuil_de_son_cru():
    """Les seuils sont IMPORTÉS de rate_regression : une seule règle par campagne."""
    src = (ROOT / "src" / "evaluation" / "build_isolation.py").read_text(encoding="utf-8")
    suspects = {node.value for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.Constant) and isinstance(node.value, float)}
    assert not suspects, f"seuil local au lieu d'un import : {sorted(suspects)}"


def test_summary_mesure_reste_coherent_avec_ses_cellules():
    path = EXP_DIR / "summary.json"
    if not path.is_file():
        pytest.skip("aucune isolation mesurée")
    summary = json.loads(path.read_text(encoding="utf-8"))
    for nom, paire in summary["pairs"].items():
        assert paire["verdict"] in (
            bi.VERDICT_INDETERMINE,
            bi.VERDICT_SUIT.format(varied=paire["varied"]),
            bi.VERDICT_NE_SUIT_PAS.format(varied=paire["varied"]),
        ), nom
        assert paire["rationale"], nom
        if "delta_slope_ua_per_hz" in paire:
            assert paire["delta_publishable_as_energy"] is False, nom
            assert paire["delta_uj_equivalent"] == A_MESURER, nom
