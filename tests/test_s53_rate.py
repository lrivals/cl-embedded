"""test_s53_rate.py — Balayage de cadence et µJ par inférence par régression (S5304).

Ce que ces tests protègent, et qui ne se voit PAS sur le banc :

1. **La reproduction du pilote réel.** Les 20 points mesurés le 2026-08-05
   (`experiments/exp_S53_rate_sweep/pilote_2026-08-05.json`) repassent dans la chaîne de
   calcul et doivent redonner les mêmes pentes, r² et incertitudes. C'est la seule
   validation possible sur des données de banc réelles sans carte ni sonde — et c'est
   aussi le garde-fou du jour où quelqu'un « améliore » la régression.
2. **L'honnêteté de la publication.** Une pente négative, une pente sous le bruit ou un
   nuage non linéaire DOIVENT sortir ``"à mesurer"`` + raison chiffrée. Jamais un zéro,
   jamais une énergie négative.
3. **La saturation silencieuse.** Au-delà du plafond de transport UART, le flux ne perd
   aucune trame et ne lève aucun CRC : il émet simplement moins vite. Sans la détection
   par cadence atteinte, la pente serait sous-estimée sans le moindre symptôme.
4. **La non-régression du pilote de fréquence** (S5303), dont l'ajustement délègue
   désormais au même module.

Les tests de schéma des cellules `skip` tant que le balayage n'a pas tourné : la carte et
la sonde sont requises pour les produire.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp_S53_rate_sweep"
PILOTE = EXP_DIR / "pilote_2026-08-05.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")


def _points(pairs, sigma=0.0):
    return [{"rate_hz": float(x), "i_mean_a": float(y), "i_std_a": float(sigma),
             "n_repeats": 1} for x, y in pairs]


# ── 1. Reproduction du pilote de banc du 2026-08-05 ──────────────────────────

@pytest.mark.skipif(not PILOTE.is_file(), reason="pilote du 2026-08-05 absent")
@pytest.mark.parametrize("cellule", ["hdc_int8", "maha_fp32"])
def test_reproduit_le_pilote_mesure(cellule):
    """La chaîne de calcul redonne EXACTEMENT les chiffres du banc, pente et incertitude.

    Le pilote reste la seule mesure réelle disponible de cette méthode : s'en écarter
    signifierait que les 173 ± 24 µJ de calcul pur de HDC INT8 ne sont plus reproductibles.
    """
    ref = json.loads(PILOTE.read_text(encoding="utf-8"))[cellule]
    fit = rr.weighted_linear_fit(ref["points_hz_a"])

    assert fit.slope == pytest.approx(ref["pente_a_par_hz"], rel=1e-9)
    assert fit.slope_std == pytest.approx(ref["se_pente_a_par_hz"], rel=1e-9)
    assert fit.intercept == pytest.approx(ref["ordonnee_a"], rel=1e-9)
    assert fit.r2 == pytest.approx(ref["r2"], rel=1e-9)
    assert not fit.weighted, "points bruts sans σ : la pondération 1/σ² doit être écartée"

    energie, raison = rr.energy_uj_per_inference(fit, ref["tension_v"])
    assert raison is None
    assert energie == pytest.approx(ref["energie_uj_par_inference"], rel=1e-9)
    assert rr.energy_uncertainty_uj(fit, ref["tension_v"]) == pytest.approx(
        ref["incertitude_uj"], rel=1e-9)


@pytest.mark.skipif(not PILOTE.is_file(), reason="pilote du 2026-08-05 absent")
def test_double_difference_isole_le_calcul():
    """Le raffinement §2a : témoin Mahalanobis = coût de trame, l'écart = le calcul pur.

    Le témoin dont le calcul vaut ~0,4 µJ sort une énergie non nulle : ce qu'elle mesure
    est la trame UART. La différence des deux cellules est donc le calcul de HDC, et ce
    test garantit que le raisonnement reste arithmétiquement vrai (il ne fige aucun
    chiffre : les deux valeurs viennent du JSON mesuré).
    """
    ref = json.loads(PILOTE.read_text(encoding="utf-8"))
    energies = {}
    for cellule, bloc in ref.items():
        fit = rr.weighted_linear_fit(bloc["points_hz_a"])
        energies[cellule], _ = rr.energy_uj_per_inference(fit, bloc["tension_v"])
    calcul_pur = energies["hdc_int8"] - energies["maha_fp32"]
    assert calcul_pur > 0, "la cellule lente doit coûter plus que le témoin quasi instantané"
    assert calcul_pur == pytest.approx(
        ref["hdc_int8"]["energie_uj_par_inference"]
        - ref["maha_fp32"]["energie_uj_par_inference"], rel=1e-9)


# ── 2. Ajustement : exactitude et pondération ────────────────────────────────

def test_pente_exacte_sur_droite_parfaite():
    pente, base = 3e-5, 0.044
    fit = rr.weighted_linear_fit(
        _points([(r, base + pente * r) for r in (0, 25, 50, 100, 200)]))
    assert fit.slope == pytest.approx(pente)
    assert fit.intercept == pytest.approx(base)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.slope_std == pytest.approx(0.0, abs=1e-15)


def test_ponderation_inverse_variance_effective():
    """Un point bruité pèse moins s'il porte un grand σ — c'est tout l'intérêt des répétitions."""
    pente, base = 3e-5, 0.044
    droits = [(r, base + pente * r) for r in (0, 25, 50, 100)]
    aberrant = (200, base + pente * 200 + 0.01)

    confiant = rr.weighted_linear_fit(
        [{"rate_hz": float(x), "i_mean_a": y, "i_std_a": 1e-5, "n_repeats": 3}
         for x, y in droits]
        + [{"rate_hz": float(aberrant[0]), "i_mean_a": aberrant[1],
            "i_std_a": 1e-5, "n_repeats": 3}])
    mefiant = rr.weighted_linear_fit(
        [{"rate_hz": float(x), "i_mean_a": y, "i_std_a": 1e-5, "n_repeats": 3}
         for x, y in droits]
        + [{"rate_hz": float(aberrant[0]), "i_mean_a": aberrant[1],
            "i_std_a": 1e-2, "n_repeats": 3}])

    assert confiant.weighted and mefiant.weighted
    assert abs(mefiant.slope - pente) < abs(confiant.slope - pente)


def test_deux_cadences_refusees():
    """Deux points passent toujours par une droite : publier leur r²=1 serait un mensonge."""
    with pytest.raises(ValueError):
        rr.weighted_linear_fit(_points([(0, 0.044), (100, 0.047)]))


def test_repetitions_du_meme_point_ne_comptent_pas_comme_cadences():
    with pytest.raises(ValueError):
        rr.weighted_linear_fit(_points([(0, 0.044), (100, 0.047), (100, 0.0471)]))


# ── 3. N/A honnête (spec §5) ─────────────────────────────────────────────────

def test_pente_negative_jamais_publiee_comme_energie():
    """Le cas que le contrôle de signe existe pour attraper : une droite DÉCROISSANTE.

    Son r² est excellent (le nuage est parfaitement linéaire) : seul le test de
    significativité l'écarte. C'est exactement la situation où le protocole delta du
    Sprint 50 rendait des µJ négatifs.
    """
    fit = rr.weighted_linear_fit(
        _points([(0, 0.050), (25, 0.049), (50, 0.048), (100, 0.046), (200, 0.042)]))
    assert fit.slope < 0 and fit.r2 > rr.R2_MIN
    energie, raison = rr.energy_uj_per_inference(fit, 3.3)
    assert energie == rr.A_MESURER
    assert raison and "σ" in raison


def test_pente_sous_le_bruit_est_na():
    """Une cellule à faible taux d'occupation : le courant ne suit pas la cadence.

    C'est le cas annoncé par la spec pour EWC, TinyOL et Mahalanobis à 200 Hz (≤ 2 %
    d'occupation) — le nuage est plat et bruité, donc son r² s'effondre avant même que le
    signe de la pente n'ait un sens.
    """
    fit = rr.weighted_linear_fit(
        _points([(0, 0.0440), (25, 0.0446), (50, 0.0441), (100, 0.0447), (200, 0.0443)]))
    assert fit.r2 < rr.R2_MIN
    energie, raison = rr.energy_uj_per_inference(fit, 3.3)
    assert energie == rr.A_MESURER
    assert raison and "non significative" in raison


def test_nuage_non_lineaire_est_na():
    fit = rr.weighted_linear_fit(
        _points([(0, 0.044), (25, 0.070), (50, 0.045), (100, 0.075), (200, 0.046)]))
    assert fit.r2 < rr.R2_MIN
    energie, raison = rr.energy_uj_per_inference(fit, 3.3)
    assert energie == rr.A_MESURER
    assert "r²" in raison


def test_na_n_est_jamais_zero():
    """Aucune sortie N/A ne doit pouvoir passer pour une mesure nulle."""
    fit = rr.weighted_linear_fit(
        _points([(0, 0.050), (50, 0.048), (100, 0.046), (200, 0.042)]))
    energie, _ = rr.energy_uj_per_inference(fit, 3.3)
    assert isinstance(energie, str) and energie == "à mesurer"
    assert energie != 0


# ── 4. Saturation silencieuse ────────────────────────────────────────────────

def test_saturation_detectee_a_la_borne_haute():
    points = _points([(0, 0.044), (50, 0.046), (100, 0.048), (200, 0.050)])
    for point in points:
        point["achieved_rate_hz"] = point["rate_hz"]
    points[-1]["achieved_rate_hz"] = 150.0     # plafond de transport atteint
    assert rr.saturation_rate_hz(points) == 200.0
    assert [p["rate_hz"] for p in rr.usable_points(points)] == [0.0, 50.0, 100.0]


def test_pas_de_saturation_sans_decrochage():
    points = _points([(0, 0.044), (50, 0.046), (100, 0.048)])
    for point in points:
        point["achieved_rate_hz"] = point["rate_hz"]
    assert rr.saturation_rate_hz(points) is None
    assert len(rr.usable_points(points)) == 3


def test_cadence_atteinte_absente_ne_vaut_pas_non_sature():
    """Sans cadence atteinte relevée, on ne conclut rien — surtout pas « ça va »."""
    points = _points([(0, 0.044), (50, 0.046), (100, 0.048)])
    assert rr.saturation_rate_hz(points) is None
    assert all("achieved_rate_hz" not in p for p in points)


def test_saturation_sous_estimerait_la_pente():
    """Justification chiffrée du filtrage : un point saturé aplatit la droite."""
    pente, base = 3e-5, 0.044
    points = _points([(r, base + pente * r) for r in (0, 50, 100, 200)])
    for point in points:
        point["achieved_rate_hz"] = point["rate_hz"]
    points[-1]["i_mean_a"] = base + pente * 150   # la carte n'a tenu que 150 Hz
    points[-1]["achieved_rate_hz"] = 150.0

    avec = rr.weighted_linear_fit(points).slope
    sans = rr.weighted_linear_fit(rr.usable_points(points)).slope
    assert avec < sans
    assert sans == pytest.approx(pente)


# ── 5. Régression de second niveau et verdicts ───────────────────────────────

def _cellule(nom, energie, latence, sigma=1.0):
    modele, _, encodage = nom.rpartition("_")
    return {"model": modele, "encoding": encodage,
            "energy_uj_per_inference": energie, "energy_uncertainty_uj": sigma,
            "dwt_latency_us_p50": latence}


def test_second_niveau_separe_calcul_et_trame():
    """La droite `µJ(latence)` rend le coût par µs de calcul ET le coût d'une trame UART."""
    uj_par_us, uj_trame = 0.09, 60.0
    cells = {nom: _cellule(nom, uj_trame + uj_par_us * lat, lat)
             for nom, lat in [("maha_fp32", 5.0), ("ewc_fp32", 50.0),
                              ("tinyol_fp32", 85.0), ("hdc_int8", 1958.0)]}
    second = rr.slope_vs_latency(cells)
    assert second["uj_per_us_compute"] == pytest.approx(uj_par_us)
    assert second["uj_per_uart_frame"] == pytest.approx(uj_trame)
    assert second["r2"] == pytest.approx(1.0)
    assert not second["cells_excluded"]


def test_second_niveau_exclut_les_cellules_non_publiables():
    cells = {"maha_fp32": _cellule("maha_fp32", 60.0, 5.0),
             "ewc_fp32": _cellule("ewc_fp32", rr.A_MESURER, 50.0),
             "hdc_int8": _cellule("hdc_int8", 234.0, 1958.0)}
    cells["ewc_fp32"]["energy_na_reason"] = "pente sous le bruit"
    second = rr.slope_vs_latency(cells)
    assert second["uj_per_us_compute"] == rr.A_MESURER
    assert "ewc_fp32" in second["cells_excluded"]
    assert second["na_reason"]


def test_verdict_gap3_calcule_et_non_saisi():
    proches = {"ewc_fp32": _cellule("ewc_fp32", 64.0, 50.0, sigma=5.0),
               "ewc_int8": _cellule("ewc_int8", 66.0, 74.0, sigma=5.0)}
    verdict = rr.gap3_energy_verdict(proches)
    assert verdict["verdict"] == "non_significatif"
    assert "RAM" in verdict["rationale"]

    ecartes = {"ewc_fp32": _cellule("ewc_fp32", 64.0, 50.0, sigma=1.0),
               "ewc_int8": _cellule("ewc_int8", 90.0, 74.0, sigma=1.0)}
    assert rr.gap3_energy_verdict(ecartes)["verdict"] == "int8_plus_couteux"


def test_verdict_gap3_na_si_une_cellule_manque():
    verdict = rr.gap3_energy_verdict({"ewc_fp32": _cellule("ewc_fp32", 64.0, 50.0)})
    assert verdict["verdict"] == rr.A_MESURER
    assert verdict["rationale"]


def test_coherence_croisee():
    """Critère d'acceptation : µJ/µs × Δlatence doit retrouver l'écart de pente mesuré."""
    uj_par_us = 0.09
    cells = {"ewc_fp32": _cellule("ewc_fp32", 64.0, 50.0),
             "ewc_int8": _cellule("ewc_int8", 64.0 + uj_par_us * 24.0, 74.0)}
    controle = rr.coherence_check(cells, {"uj_per_us_compute": uj_par_us})
    assert controle["coherent"] is True
    assert controle["ratio_measured_over_expected"] == pytest.approx(1.0)

    divergent = rr.coherence_check(
        {"ewc_fp32": _cellule("ewc_fp32", 64.0, 50.0),
         "ewc_int8": _cellule("ewc_int8", 264.0, 74.0)},
        {"uj_per_us_compute": uj_par_us})
    assert divergent["coherent"] is False


def test_duty_cycle_exige_une_latence_mesuree():
    assert rr.duty_cycle(1958.0, 200.0) == pytest.approx(0.3916)
    assert rr.duty_cycle(None, 200.0) is None
    assert rr.duty_cycle(50.0, 0.0) is None


# ── 6. Pilote : garde AST et honnêteté du schéma ─────────────────────────────

def test_pilote_sans_resultat_en_dur():
    """Garde AST : aucun résultat numérique figé dans le pilote (précédent S50/S53).

    Les seuls flottants tolérés sont des valeurs par défaut de CLI et de mise en page —
    les seuils de décision vivent dans `rate_regression.py`, où ils sont testés.
    """
    src = (ROOT / "scripts" / "run_s53_rate_sweep.py").read_text(encoding="utf-8")
    autorises = {0.0, 1.0, 2.0, 3.3, 4.0, 10.0, 100.0}
    suspects = {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in autorises
    }
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"


def test_methode_distincte_des_autres_estimateurs():
    """Les µJ par régression ne doivent JAMAIS être fusionnés avec ceux du delta (S5302)."""
    assert rr.METHOD == "régression I(rate)"
    delta = (ROOT / "scripts" / "run_s50_energy_delta.py").read_text(encoding="utf-8")
    assert rr.METHOD not in delta


def test_plafond_uart_respecte_par_les_cadences_par_defaut():
    """La borne haute reste sous le plafond de transport mesuré (~209 inf/s)."""
    pilote = _load("run_s53_rate_sweep", ROOT / "scripts" / "run_s53_rate_sweep.py")
    assert max(pilote.DEFAULT_RATES) <= 209
    assert 0 in pilote.DEFAULT_RATES, "le repos est le point qui ancre I_base"


def test_plan_randomise_et_reproductible():
    pilote = _load("run_s53_rate_sweep", ROOT / "scripts" / "run_s53_rate_sweep.py")
    cells = [("ewc", "fp32"), ("hdc", "int8")]
    rates = [0.0, 50.0, 100.0]
    plan = pilote.build_schedule(cells, rates, 3, True, 42)
    nominal = pilote.build_schedule(cells, rates, 3, False, 42)

    assert sorted(plan) == sorted(nominal), "le plan randomisé couvre les mêmes points"
    assert plan != nominal, "l'ordre doit être décorrélé de la condition (S5301)"
    assert plan == pilote.build_schedule(cells, rates, 3, True, 42), "graine reproductible"
    assert len(plan) == len(cells) * len(rates) * 3


def test_cellule_a_firmware_dedie_hors_balayage_par_defaut():
    """`maha_int8` ne doit pas être mesurée sur le build FP32 (bug du drapeau, Sprint 52)."""
    pilote = _load("run_s53_rate_sweep", ROOT / "scripts" / "run_s53_rate_sweep.py")
    parser = __import__("argparse").ArgumentParser()
    assert ("maha", "int8") not in pilote.parse_cells(None, parser)
    assert ("maha", "int8") in pilote.parse_cells("maha_int8", parser)


# ── 7. Cellules mesurées : schéma (skip tant que le banc n'a pas tourné) ─────

def _cellules_mesurees() -> list[Path]:
    return [p for p in sorted(EXP_DIR.glob("*.json"))
            if p.name not in {"slope_vs_latency.json", "summary.json"}
            and "points" in json.loads(p.read_text(encoding="utf-8"))]


@pytest.mark.skipif(not _cellules_mesurees(), reason="balayage non exécuté (carte + sonde)")
def test_schema_des_cellules_mesurees():
    for path in _cellules_mesurees():
        cell = json.loads(path.read_text(encoding="utf-8"))
        for champ in ("points", "slope_ua_per_hz", "slope_std_ua_per_hz", "intercept_ma",
                      "r2", "energy_uj_per_inference", "method", "tension_v",
                      "dwt_latency_us_p50", "duty_cycle_at_max_rate"):
            assert champ in cell, f"{path.name} : champ {champ} manquant"
        assert cell["method"] == rr.METHOD
        energie = cell["energy_uj_per_inference"]
        if isinstance(energie, str):
            assert energie == rr.A_MESURER and cell.get("energy_na_reason")
        else:
            assert energie > 0, "une énergie publiée est strictement positive"


@pytest.mark.skipif(not _cellules_mesurees(), reason="balayage non exécuté (carte + sonde)")
def test_integrite_du_flux_a_toutes_les_cadences():
    for path in _cellules_mesurees():
        check = json.loads(path.read_text(encoding="utf-8")).get("protocol_check", {})
        assert check.get("crc_errors") == 0, f"{path.name} : erreurs CRC relevées"


# ── 8. Non-régression du pilote de fréquence (S5303) ─────────────────────────

def test_le_pilote_de_frequence_ne_refait_pas_son_propre_ajustement():
    """Le balayage SYSCLK ne doit plus porter de régression maison.

    Il en portait une (`_linear_fit`) qui passait des couples `(x, y)` au module commun,
    donc SANS les écarts-types : le balayage de cadence pondérait par `1/σ²`, celui de
    fréquence non, et les deux publiaient deux chiffres pour la même mesure. Le contrat est
    désormais qu'il n'existe qu'un seul ajustement dans le dépôt.
    """
    fs = _load("run_s53_freq_sweep", ROOT / "scripts" / "run_s53_freq_sweep.py")
    assert not hasattr(fs, "_linear_fit"), "l'ajustement maison est revenu"
    source = (ROOT / "scripts" / "run_s53_freq_sweep.py").read_text(encoding="utf-8")
    assert "rr.fit_cell(" in source, "la cellule doit être dérivée par rate_regression"


# ── 9. Provenance de la cadence atteinte (A7) ────────────────────────────────

def test_une_cadence_non_mesuree_ne_porte_pas_la_consigne():
    """`achieved_rate_hz` est un champ de MESURE : la consigne n'y a rien à faire."""
    points = [{"rate_hz": 0.0, "i_mean_a": 0.040},
              {"rate_hz": 50.0, "i_mean_a": 0.043, "achieved_rate_hz": 50.0},
              {"rate_hz": 200.0, "i_mean_a": 0.050, "achieved_rate_hz": 126.3}]
    rr.normalize_achieved_source(points)
    assert points[1]["achieved_rate_hz"] is None
    assert points[1]["achieved_rate_source"] == "non mesuré"
    # La borne haute est re-streamée : elle est MESURÉE, y compris quand elle décroche.
    assert points[2]["achieved_rate_source"] == "mesuré"


def test_un_plafond_propage_est_signale_comme_inference():
    points = [{"rate_hz": 100.0, "i_mean_a": 0.046, "achieved_rate_hz": 71.3},
              {"rate_hz": 200.0, "i_mean_a": 0.050, "achieved_rate_hz": 71.3}]
    rr.normalize_achieved_source(points)
    assert points[0]["achieved_rate_source"] == "inféré du plafond mesuré"
    assert points[1]["achieved_rate_source"] == "mesuré"


def test_la_provenance_ne_change_pas_lajustement():
    """Le garde-fou est de traçabilité : la saturation ignore déjà les points sans mesure."""
    points = [{"rate_hz": r, "i_mean_a": 0.04 + r * 6e-5, "i_std_a": 1e-5,
               "achieved_rate_hz": float(r)} for r in (0, 10, 25, 50, 100)]
    avant = rr.fit_cell([dict(p) for p in points], 3.3, 50.0)
    rr.normalize_achieved_source(points)
    apres = rr.fit_cell(points, 3.3, 50.0)
    assert avant["slope_ua_per_hz"] == pytest.approx(apres["slope_ua_per_hz"])
    assert avant["saturation_rate_hz"] == apres["saturation_rate_hz"]


@pytest.mark.parametrize("path", sorted(EXP_DIR.glob("*.json")))
def test_aucune_cellule_mesuree_ne_declare_la_consigne_comme_atteinte(path):
    cell = json.loads(path.read_text(encoding="utf-8"))
    for point in cell.get("points", []):
        if not isinstance(point, dict) or "rate_hz" not in point:
            continue
        atteint = point.get("achieved_rate_hz")
        if atteint is None:
            continue
        assert point.get("achieved_rate_source") == "mesuré" or \
            float(atteint) != float(point["rate_hz"]), \
            f"{path.name} : cadence atteinte = consigne sans mesure ({point['rate_hz']} Hz)"
