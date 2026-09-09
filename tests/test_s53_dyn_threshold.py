"""Tests du plafond du mode dynamique de la sonde (S5303, B6).

Deux règles sont verrouillées ici, toutes deux hors banc :

    1. « acquisition aboutie » — une acquisition interrompue en surintensité ne lève pas :
       elle rend ce qu'elle a eu le temps de décoder. Une seule définition doit gouverner
       toute la campagne, sans quoi une trace tronquée passe pour un profil valide ;
    2. le verdict sur le déclencheur — il doit refuser de nommer un candidat quand la
       matrice ne les a pas décorrélés. C'est l'issue la plus probable, et elle doit
       sortir CHIFFRÉE au lieu d'être arrondie en « non concluant ».
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.evaluation import dyn_threshold as dt

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp_S53_dyn_threshold"


def _essai(succeeded: bool, i_max: float, i_mean: float, duration: float = 1.0) -> dict:
    return {"succeeded": succeeded, "i_max_ma": i_max, "i_mean_ma": i_mean,
            "duration_s": duration}


# ── 1. Acquisition aboutie ou interrompue ───────────────────────────────────

def test_acquisition_complete_est_un_succes():
    bloc = dt.acquisition_outcome(1_000_000, 1_000_000, "summary beg … summary end")
    assert bloc["succeeded"] is True
    assert "na_reason" not in bloc


def test_acquisition_tronquee_nest_pas_un_succes():
    """Le cas MESURÉ à 90 MHz : 21 894 échantillons sur 100 000 demandés."""
    bloc = dt.acquisition_outcome(21_894, 100_000,
                                  "Measurement interrupted due to errors")
    assert bloc["succeeded"] is False
    assert bloc["interrupted_by_probe"] is True
    assert "21894/100000" in bloc["na_reason"], "la raison doit être CHIFFRÉE"


def test_acquisition_courte_sans_message_reste_refusee():
    """Une troncature silencieuse compte autant qu'une interruption annoncée."""
    bloc = dt.acquisition_outcome(50_000, 100_000, "summary end")
    assert bloc["succeeded"] is False
    assert bloc["interrupted_by_probe"] is False
    assert bloc["na_reason"]


def test_le_seuil_de_completude_est_celui_du_balayage_de_frequence():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fs", ROOT / "scripts" / "run_s53_freq_sweep.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    assert dt.COMPLETENESS == fs.DYN_COMPLETENESS, \
        "une seule définition d'« acquisition aboutie » dans la campagne"


# ── 2. Séparabilité d'un candidat ───────────────────────────────────────────

def test_un_seuil_qui_separe_est_encadre_et_sa_marge_chiffree():
    essais = [_essai(True, 60.0, 20.0), _essai(True, 65.0, 22.0),
              _essai(False, 70.0, 21.0), _essai(False, 75.0, 20.5)]
    sep = dt.separability(essais, "i_max_ma")
    assert sep["separable"] is True
    assert sep["threshold_between"] == [65.0, 70.0]
    assert sep["margin"] == pytest.approx(5.0)
    assert sep["direction"] == "échec au-dessus du seuil"


def test_un_recouvrement_est_chiffre_pas_arrondi():
    """Deux essais aboutis encadrent un échec : aucun seuil ne les sépare."""
    essais = [_essai(True, 60.0, 27.2), _essai(True, 70.0, 28.0),
              _essai(False, 65.0, 41.1), _essai(False, 75.0, 42.0)]
    sep = dt.separability(essais, "i_max_ma")
    assert sep["separable"] is False
    assert "60.000" in sep["reason"] and "65.000" in sep["reason"]
    assert sep["overlap"] > 0


def test_sans_echec_ou_sans_succes_rien_nest_testable():
    assert dt.separability([_essai(True, 60.0, 20.0)], "i_max_ma")["separable"] is False
    assert dt.separability([_essai(False, 60.0, 20.0)], "i_max_ma")["separable"] is False


def test_le_sens_du_seuil_est_deduit_des_donnees():
    """Rien n'impose que l'échec soit au-dessus : la direction se mesure."""
    essais = [_essai(True, 70.0, 20.0), _essai(False, 60.0, 20.0)]
    sep = dt.separability(essais, "i_max_ma")
    assert sep["separable"] is True
    assert sep["direction"] == "échec en-dessous du seuil"


def test_aucune_donnee_nest_pas_un_arret():
    """0/5000 échantillons à 0,05 s : l'essai ne dit rien du déclencheur."""
    bloc = dt.acquisition_outcome(0, 5000, "summary end")
    assert bloc["succeeded"] is False
    assert bloc["no_data"] is True
    assert "ne renseigne pas sur le déclencheur" in bloc["na_reason"]
    assert dt.acquisition_outcome(22000, 100000, "interrupted")["no_data"] is False


def test_un_essai_sans_donnee_ne_pese_pas_dans_le_verdict():
    """Sans cette exclusion, un plancher d'instrument se lirait comme un seuil franchi."""
    essais = [{"succeeded": True, "i_max_ma": 60.0, "i_mean_ma": 20.0, "duration_s": 1.0},
              {"succeeded": False, "i_max_ma": 70.0, "i_mean_ma": 30.0, "duration_s": 1.0},
              {"succeeded": False, "no_data": True, "i_max_ma": None, "i_mean_ma": None,
               "duration_s": 0.05}]
    verdict = dt.classify(essais)
    assert verdict["n_attempts"] == 3
    assert verdict["n_no_data"] == 1
    assert verdict["n_considered"] == 2
    assert verdict["candidates"]["durée"]["n_ok"] + \
        verdict["candidates"]["durée"]["n_ko"] == 2


def test_une_trace_decodee_en_courants_absurdes_nest_pas_un_courant():
    """3 338 A relevés après une réouverture de session : c'est du tampon, pas une mesure."""
    assert dt.current_trustworthy(0.0698) is True
    assert dt.current_trustworthy(3338.0) is False
    assert dt.current_trustworthy(0.0) is False, "une trace nulle n'est pas un courant"


def test_un_essai_sans_courant_fiable_reste_dans_laxe_duree():
    """L'arrêt est l'observable principal : il ne se perd pas avec le courant."""
    essais = [{"succeeded": True, "duration_s": 0.05, "i_max_ma": None, "i_mean_ma": None},
              {"succeeded": False, "duration_s": 10.0, "i_max_ma": None, "i_mean_ma": None}]
    assert dt.separability(essais, "duration_s")["separable"] is True
    assert dt.separability(essais, "i_max_ma")["separable"] is False


# ── 3. Verdict ──────────────────────────────────────────────────────────────

def test_un_seul_candidat_separant_est_nomme():
    """Pic tenu constant, moyenne et durée variables : seule la durée doit sortir."""
    essais = [_essai(True, 70.0, 30.0, duration=0.05),
              _essai(True, 70.0, 40.0, duration=0.2),
              _essai(False, 70.0, 35.0, duration=1.0),
              _essai(False, 70.0, 30.0, duration=10.0)]
    verdict = dt.classify(essais)
    assert verdict["verdict"] == "durée"
    assert "durée" in verdict["rationale"]


def test_deux_candidats_confondus_ne_designent_personne():
    """L'état actuel des mesures : 68,6 mA passe, 69,6 mA échoue, pic ET moyenne montent."""
    essais = [_essai(True, 68.6, 27.2), _essai(False, 69.6, 41.1)]
    verdict = dt.classify(essais)
    assert verdict["verdict"] == dt.VERDICT_INDETERMINE
    assert set(verdict["competing"]) == {"pic", "moyenne"}


def test_aucun_candidat_separant_sort_avec_ses_recouvrements():
    essais = [_essai(True, 70.0, 40.0), _essai(False, 60.0, 30.0),
              _essai(True, 55.0, 25.0), _essai(False, 75.0, 45.0)]
    verdict = dt.classify(essais)
    assert verdict["verdict"] == dt.VERDICT_INDETERMINE
    assert "aucun" in verdict["rationale"]
    assert all(not t["separable"] for t in verdict["candidates"].values())


def test_le_verdict_compte_les_essais():
    essais = [_essai(True, 60.0, 20.0), _essai(False, 70.0, 30.0)]
    verdict = dt.classify(essais)
    assert verdict["n_attempts"] == 2 and verdict["n_succeeded"] == 1


# ── 4. Garde AST et cohérence des sorties mesurées ──────────────────────────

def test_pilote_sans_resultat_en_dur():
    """Les seules valeurs flottantes tolérées dans le pilote sont sa grille d'essais."""
    src = (ROOT / "scripts" / "run_s53_dyn_threshold.py").read_text(encoding="utf-8")
    autorises = {0.0, 0.05, 0.2, 1.0, 2.0, 3.3, 10.0, 25.0, 50.0, 100.0, 200.0, 1000.0}
    suspects = {node.value for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.Constant) and isinstance(node.value, float)
                and node.value not in autorises}
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"


def test_cellules_mesurees_portent_un_verdict_calculable():
    if not EXP_DIR.is_dir():
        pytest.skip("aucun essai mesuré (carte + sonde requises)")
    cellules = [p for p in EXP_DIR.glob("*.json") if p.name != "summary.json"]
    if not cellules:
        pytest.skip("aucun essai mesuré")
    for path in cellules:
        cell = json.loads(path.read_text(encoding="utf-8"))
        recalcule = dt.classify(cell["attempts"])
        assert recalcule["verdict"] == cell["classification"]["verdict"], \
            f"{path.name} : verdict écrit ≠ verdict recalculé depuis les essais"
        for essai in cell["attempts"]:
            assert essai["succeeded"] in (True, False)
            if not essai["succeeded"]:
                assert essai.get("na_reason"), f"{path.name} : échec sans raison"
