"""test_s53_policy_energy.py — Énergie des politiques de mise à jour P0–P3 (S5306).

Ce que ces tests protègent, et qui ne se voit PAS sur le banc :

1. **L'isolement du coût d'une mise à jour.** `energy_uj_per_update` est une DIFFÉRENCE de
   pentes (always − frozen). Sur des points synthétiques dont on connaît la réponse, la
   chaîne doit rendre exactement cette différence — et son incertitude propagée.
2. **L'honnêteté de la publication.** Une différence négative, une différence sous le
   bruit, ou une cellule dont la régression n'est pas publiable DOIVENT sortir
   ``"à mesurer"`` + raison CHIFFRÉE (la valeur relevée et sa barre d'erreur). Jamais un
   zéro, jamais une énergie négative — c'est exactement ce que le protocole delta du
   Sprint 50 produisait et que ce sprint refuse de reproduire.
3. **Le surcoût PERMANENT du gate.** Le gate coûte ~27 µs sur TOUS les échantillons et
   n'évite une mise à jour que sur ceux qu'il ne déclenche pas. Le retranchement de la
   part imputable aux mises à jour réellement effectuées est ce qui rend le verdict
   interprétable — les trois issues (économise / neutre / plus coûteux) sont toutes
   publiables.
4. **Le non-recopiage des chiffres du Sprint 38.** Les colonnes S38 du tableau croisé
   doivent PROVENIR de `exp_S38_summary.json`. Le test les lit dans un fichier factice et
   vérifie qu'elles en ressortent — un chiffre en dur ferait échouer le test.
5. **La cohérence avec le pilote du Sprint 38.** Les drapeaux de compilation et le drapeau
   UART `--update` sont miroirs de `run_sprint38_board.py`. S'ils divergeaient, une
   politique serait mesurée sur un chemin d'exécution qui n'est pas le sien — le bug du
   drapeau TinyOL du Sprint 52.

Les tests de schéma des cellules `skip` tant que le banc n'a pas tourné : la carte et la
sonde sont requises pour les produire.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp_S53_policy_energy"
ECONOMY = EXP_DIR / "economy_energy.json"
HW_PROFILE = ROOT / "configs" / "hw_profile_f439zi.yaml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pe = _load("policy_energy", ROOT / "src" / "evaluation" / "policy_energy.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")

TENSION_V = 3.3


def _cell(policy: str, dataset: str, slope_a_per_hz: float, intercept_a: float,
          sigma: float = 0.0, latency_us: float | None = None) -> dict:
    """Cellule synthétique : des points STRICTEMENT sur la droite `I = b + a·rate`.

    Points exacts ⇒ résidus nuls ⇒ erreur-type nulle : la pente est alors publiable par
    construction, ce qui isole dans chaque test la seule grandeur qu'il examine. Le bruit
    s'ajoute explicitement (`sigma`) là où c'est lui qu'on teste.
    """
    rates = [0.0, 50.0, 100.0, 150.0, 200.0]
    points = [{"rate_hz": r,
               "i_mean_a": intercept_a + slope_a_per_hz * r + (sigma if i % 2 else -sigma),
               "i_std_a": 0.0, "n_repeats": 3,
               "achieved_rate_hz": r if r > 0 else None}
              for i, r in enumerate(rates)]
    for p in points:
        if p["achieved_rate_hz"] is None:
            del p["achieved_rate_hz"]
    cell = {"policy": policy, "dataset": dataset, "points": points,
            "protocol_check": {"crc_errors": 0, "latency_p50_us": latency_us}}
    cell.update(rr.fit_cell(points, TENSION_V, latency_us))
    return cell


# ── 1. Énergie d'une mise à jour CL : la différence P1 − P0 ──────────────────

def test_energie_de_mise_a_jour_est_la_difference_des_pentes():
    """`E_update = (pente(always) − pente(frozen)) × V`, au µJ près.

    Le coût de trame UART — 60,6 µJ au pilote S5304, soit l'essentiel de la pente d'une
    cellule rapide — est commun aux deux cellules et doit se soustraire EXACTEMENT.
    """
    frozen = _cell("frozen", "monitoring", slope_a_per_hz=20e-6, intercept_a=46e-3)
    always = _cell("always", "monitoring", slope_a_per_hz=95e-6, intercept_a=46e-3)

    res = pe.update_energy_uj(always, frozen)

    attendu_uj = (95e-6 - 20e-6) * TENSION_V * 1e6
    assert res["value_uj"] == pytest.approx(attendu_uj, rel=1e-9)
    assert res["significant"] is True
    assert res["method"] == pe.METHOD


def test_energie_de_mise_a_jour_ignore_le_cout_de_trame_commun():
    """Deux paires de cellules de coûts de trame TRÈS différents, même écart de pente.

    C'est la propriété qui fait de P1 − P0 le chemin propre vers `energy_uj_per_update` :
    tout ce qui est commun aux deux politiques (trame, scrutation, PHY) s'annule.
    """
    delta = 40e-6
    a = pe.update_energy_uj(_cell("always", "monitoring", 10e-6 + delta, 46e-3),
                            _cell("frozen", "monitoring", 10e-6, 46e-3))
    b = pe.update_energy_uj(_cell("always", "monitoring", 300e-6 + delta, 51e-3),
                            _cell("frozen", "monitoring", 300e-6, 51e-3))
    assert a["value_uj"] == pytest.approx(b["value_uj"], rel=1e-9)


def test_difference_negative_jamais_publiee_comme_energie():
    """`always` moins gourmand que `frozen` est un signe de banc, pas une énergie.

    Le protocole delta du Sprint 50 rendait des µJ NÉGATIFS ; le champ doit rester
    ``"à mesurer"``, avec la valeur relevée dans sa raison pour qu'elle reste visible.
    """
    frozen = _cell("frozen", "monitoring", slope_a_per_hz=95e-6, intercept_a=46e-3)
    always = _cell("always", "monitoring", slope_a_per_hz=20e-6, intercept_a=46e-3)

    res = pe.update_energy_uj(always, frozen)

    assert res["value_uj"] == pe.A_MESURER
    assert res["significant"] is False
    assert "non significative" in res["na_reason"]
    assert "µJ" in res["na_reason"], "la valeur relevée doit rester visible dans la raison"
    assert res["delta_slope_ua_per_hz"] < 0


def test_difference_sous_le_bruit_est_na_honnete():
    """Un écart de pente inférieur à 2 σ n'est pas une énergie, même s'il est positif.

    Les deux régressions sont ici parfaitement publiables : c'est bien la RÈGLE DE LA
    DIFFÉRENCE qui refuse, pas celle d'une pente isolée.
    """
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 21e-6, 46e-3)
    frozen["slope_std_ua_per_hz"] = 3.0     # 1 µA/Hz d'écart pour ~4 µA/Hz de bruit
    always["slope_std_ua_per_hz"] = 3.0

    res = pe.update_energy_uj(always, frozen)

    assert res["value_uj"] == pe.A_MESURER
    assert "σ" in res["na_reason"]
    assert "Δpente" in res["na_reason"]


def test_cellule_non_lineaire_bloque_la_difference():
    """Sans linéarité, une pente ne décrit aucun coût marginal — leur différence non plus.

    C'est la seule condition d'entrée qui reste (règle A4) : elle porte sur le r² de la
    cellule, et la raison du refus est CHIFFRÉE par ce r².
    """
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    always["r2"] = 0.412

    res = pe.update_energy_uj(always, frozen)

    assert res["value_uj"] == pe.A_MESURER
    assert "always" in res["na_reason"]
    assert "0.412" in res["na_reason"], "le r² fautif doit être nommé, pas effacé"


def test_difference_publiable_meme_si_aucune_pente_ne_lest_seule():
    """Règle A4 — le bruit COMMUN aux deux cellules s'annule dans l'écart.

    Deux pentes noyées dans le bruit prises séparément (aucune n'est à 2σ de zéro, les deux
    cellules sortent donc « à mesurer » en absolu), mais dont la DIFFÉRENCE l'est : la règle
    antérieure refusait de publier, alors que c'est exactement le régime où vit le coût
    d'une mise à jour CL. Les deux cellules partagent binaire, trame UART, séance et
    ordonnée à l'origine.
    """
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    for cell in (frozen, always):
        # Pente non séparable de zéro prise seule (75 µA/Hz d'écart, 60 µA/Hz de bruit) …
        cell["slope_std_ua_per_hz"] = 60.0
        cell["energy_uj_per_inference"] = pe.A_MESURER
        cell["energy_na_reason"] = "pente à moins de 2 σ de zéro"
        assert not pe._publiable(cell), "prise seule, la cellule reste non publiable"
        assert pe._lineaire(cell), "elle reste parfaitement linéaire"

    res = pe.update_energy_uj(always, frozen)

    # Δ = 75 µA/Hz pour σ = hypot(60, 60) ≈ 84,9 µA/Hz → toujours PAS significatif.
    assert res["value_uj"] == pe.A_MESURER
    # … mais avec un bruit deux fois moindre, la différence, elle, l'est.
    for cell in (frozen, always):
        cell["slope_std_ua_per_hz"] = 20.0
    res = pe.update_energy_uj(always, frozen)
    assert pe.is_measured(res["value_uj"]), (
        "une différence significative doit être publiée même si aucune des deux pentes "
        "ne l'est prise isolément"
    )


def test_lenergie_absolue_reste_soumise_au_test_2_sigma():
    """Le relâchement A4 ne vaut QUE pour les différences : aucune contagion en absolu."""
    cell = _cell("always", "monitoring", 95e-6, 46e-3)
    cell["slope_std_ua_per_hz"] = 60.0
    cell["energy_uj_per_inference"] = pe.A_MESURER
    assert pe._lineaire(cell) and not pe._publiable(cell)


def test_incertitude_propagee_par_quadrature():
    """σ(Δ) = hypot(σ₁, σ₀) × V : l'incertitude de la différence n'est pas celle d'un terme."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    frozen["slope_std_ua_per_hz"] = 3.0
    always["slope_std_ua_per_hz"] = 4.0

    res = pe.update_energy_uj(always, frozen)

    assert res["delta_slope_std_ua_per_hz"] == pytest.approx(5.0, rel=1e-9)   # 3-4-5
    assert res["uncertainty_uj"] == pytest.approx(5.0 * 1e-6 * TENSION_V * 1e6, rel=1e-9)


def test_tensions_heterogenes_refusees():
    """Deux cellules mesurées à des tensions différentes ne se soustraient pas."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    always["tension_v"] = 3.0

    with pytest.raises(ValueError, match="tensions"):
        pe.update_energy_uj(always, frozen)


# ── 2. Surcoût permanent du gate ─────────────────────────────────────────────

def test_surcout_du_gate_retranche_la_part_des_mises_a_jour():
    """Le gate paie son surcoût sur TOUS les échantillons ; il n'évite que les MAJ non
    déclenchées. Le retranchement de `update_rate × E_update` est ce qui isole le gate."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    update = pe.update_energy_uj(always, frozen)

    # Le gated dépasse frozen de 10 µA/Hz, dont une part imputable à ses 2,5 % de MAJ.
    gated = _cell("gated_truelabel", "monitoring", 30e-6, 46e-3)
    res = pe.gate_energy_overhead_uj(gated, frozen, 0.025, update)

    delta_uj = (30e-6 - 20e-6) * TENSION_V * 1e6
    part_maj = 0.025 * update["value_uj"]
    assert res["value_uj"] == pytest.approx(delta_uj - part_maj, rel=1e-9)
    assert res["updates_share_uj"] == pytest.approx(part_maj, rel=1e-9)
    assert res["update_rate"] == 0.025


def test_surcout_du_gate_na_sans_energie_de_mise_a_jour():
    """Sans `E_update` chiffrée, la part des MAJ n'est pas retranchable : N/A honnête."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    gated = _cell("gated_truelabel", "monitoring", 30e-6, 46e-3)
    update = {"value_uj": pe.A_MESURER, "na_reason": "différence non significative"}

    res = pe.gate_energy_overhead_uj(gated, frozen, 0.025, update)

    assert res["value_uj"] == pe.A_MESURER
    assert pe.A_MESURER in res["na_reason"]


def test_surcout_du_gate_na_sans_taux_mesure():
    """Le taux de mise à jour doit être MESURÉ (ou chargé), jamais supposé."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 95e-6, 46e-3)
    gated = _cell("gated_truelabel", "monitoring", 30e-6, 46e-3)

    res = pe.gate_energy_overhead_uj(gated, frozen, None,
                                     pe.update_energy_uj(always, frozen))

    assert res["value_uj"] == pe.A_MESURER
    assert "taux de mise à jour" in res["na_reason"]


# ── 3. Économie et verdict : les trois issues ────────────────────────────────

def test_economie_rapportee_a_l_energie_marginale_de_always():
    """Le dénominateur du pourcentage est `pente(always) − pente(frozen)`, et il est écrit.

    Rapporter l'économie à la consommation TOTALE de la carte la diluerait dans le coût de
    trame et le PHY, que la politique ne change pas.
    """
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 100e-6, 46e-3)
    gated = _cell("gated_truelabel", "monitoring", 40e-6, 46e-3)

    res = pe.energy_saved_vs_always(gated, always, frozen)

    # marginal always = 80 µA/Hz, marginal gated = 20 → économie 60/80 = 75 %.
    assert res["saved_pct_of_always_marginal"] == pytest.approx(75.0, rel=1e-9)
    assert res["denominator_uj_always_marginal"] == pytest.approx(
        80e-6 * TENSION_V * 1e6, rel=1e-9)
    assert "MARGINALE" in res["denominator_note"]


@pytest.mark.parametrize("slope_gated_ua,attendu", [
    (25e-6, "gate_economise"),        # surcoût faible, économie franche
    (200e-6, "gate_plus_couteux"),    # surcoût permanent > MAJ évitées
])
def test_verdict_prononce_les_trois_issues(slope_gated_ua, attendu):
    """Les trois issues de la spec sont publiables — y compris « le gate coûte plus »,
    qui borne le domaine de validité de la contribution du Sprint 38."""
    frozen = _cell("frozen", "monitoring", 20e-6, 46e-3)
    always = _cell("always", "monitoring", 100e-6, 46e-3)
    gated = _cell("gated_truelabel", "monitoring", slope_gated_ua, 46e-3)
    update = pe.update_energy_uj(always, frozen)

    overhead = pe.gate_energy_overhead_uj(gated, frozen, 0.025, update)
    saved = pe.energy_saved_vs_always(gated, always, frozen)
    verdict = pe.gate_verdict(overhead, saved)

    assert verdict["verdict"] == attendu
    assert "µJ" in verdict["rationale"], "le verdict doit être chiffré, pas déclaratif"


def test_verdict_neutre_quand_le_bilan_est_dans_le_bruit():
    """Un bilan net sous 2 σ est « neutre » : la justification du gate redevient alors la
    latence pire-cas et l'autonomie de décision, pas l'énergie."""
    overhead = {"value_uj": 10.0, "uncertainty_uj": 20.0}
    saved = {"saved_uj_per_sample": 12.0, "uncertainty_uj": 20.0}

    verdict = pe.gate_verdict(overhead, saved)

    assert verdict["verdict"] == "gate_neutre"
    assert "latence pire-cas" in verdict["rationale"]


def test_verdict_non_prononcable_sans_grandeurs():
    verdict = pe.gate_verdict({"value_uj": pe.A_MESURER, "na_reason": "raison amont"},
                              {"saved_uj_per_sample": pe.A_MESURER})
    assert verdict["verdict"] == pe.A_MESURER
    assert "raison amont" in verdict["rationale"]


# ── 4. Autonomie ─────────────────────────────────────────────────────────────

def test_autonomie_sur_les_capacites_chargees_du_profil_hw():
    """Les capacités viennent de `configs/hw_profile_f439zi.yaml`, jamais du code."""
    capacites = pe.load_capacities(HW_PROFILE)
    cell = _cell("frozen", "monitoring", 20e-6, 46e-3)

    res = pe.policy_autonomy(cell, capacites)

    assert set(res["autonomy_hours"]) == {str(float(c)) for c in capacites}
    # I_moy = intercept + pente × 100 Hz, exprimé en mA.
    attendu_ma = (46e-3 + 20e-6 * pe.AUTONOMY_RATE_HZ) * 1e3
    assert res["i_moy_ma"] == pytest.approx(attendu_ma, rel=1e-9)
    for cap in capacites:
        assert res["autonomy_hours"][str(float(cap))] == pytest.approx(
            cap / attendu_ma, rel=1e-9)


def test_autonomie_porte_la_reserve_du_regime_mesure():
    """Réserve reconduite du Sprint 50 : flux CONTINU, carte jamais endormie. Lire ces
    heures comme une autonomie duty-cyclée serait un contresens."""
    res = pe.policy_autonomy(_cell("frozen", "monitoring", 20e-6, 46e-3),
                             pe.load_capacities(HW_PROFILE))
    assert res["duty_cycle"] is None
    assert "duty-cyclée" in res["regime_mesure"]


def test_autonomie_na_sans_regression():
    res = pe.policy_autonomy(None, [220.0])
    assert res["i_moy_ma"] == pe.A_MESURER
    assert res["duty_cycle"] is None


# ── 5. Croisement Sprint 38 : chargé, jamais recopié ─────────────────────────

@pytest.fixture()
def s38_factice(tmp_path: Path) -> Path:
    """Un `exp_S38_summary.json` minimal aux valeurs RECONNAISSABLES.

    Si le tableau croisé contenait un chiffre en dur, il ne pourrait pas rendre ces
    valeurs-là.
    """
    summary = {"results": {ds: {"pretrained": {
        "frozen": {"board": {"update_rate": 0.0, "f1_faulty": 0.111}},
        "always": {"board": {"update_rate": 1.0, "f1_faulty": 0.222}},
        "gated_truelabel": {"board": {"update_rate": 0.0251, "f1_faulty": 0.333}},
        "gated_pseudolabel": {"board": {"update_rate": 0.0252, "f1_faulty": 0.444}},
        "economy_table": {
            "frozen": {"updates_saved_pct": 1.0, "latency_saved_us": 190.16,
                       "ram_added_bytes": None},
            "always": {"updates_saved_pct": 0.0, "latency_saved_us": 0.0,
                       "ram_added_bytes": None},
            "gated_truelabel": {"updates_saved_pct": 0.9744, "latency_saved_us": 158.65,
                                "ram_added_bytes": 300},
            "gated_pseudolabel": {"updates_saved_pct": 0.9744, "latency_saved_us": 158.87,
                                  "ram_added_bytes": 300},
        }}} for ds in pe.DATASETS}}
    path = tmp_path / "exp_S38_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_tableau_croise_charge_les_colonnes_du_sprint_38(s38_factice):
    s38 = pe.load_s38_summary(s38_factice)
    cells = {f"{p}_{ds}": _cell(p, ds, 20e-6, 46e-3)
             for ds in pe.DATASETS for p in pe.POLICIES}

    rows = pe.cross_table(cells, s38)

    assert len(rows) == len(pe.POLICIES) * len(pe.DATASETS)
    gated = next(r for r in rows
                 if r["policy"] == "gated_truelabel" and r["dataset"] == "monitoring")
    assert gated["updates_saved_pct_s38"] == 0.9744
    assert gated["latency_saved_us_s38"] == 158.65
    assert gated["update_rate_s38"] == 0.0251
    assert gated["f1_faulty_s38"] == 0.333
    assert gated["ram_added_bytes_s38"] == 300


def test_tableau_croise_na_honnete_sans_le_fichier_s38(tmp_path):
    """Sans `exp_S38_summary.json`, les colonnes S38 sortent « à mesurer » + raison —
    elles ne sont jamais complétées de mémoire."""
    assert pe.load_s38_summary(tmp_path / "absent.json") is None
    rows = pe.cross_table({}, None)
    assert all(r["updates_saved_pct_s38"] == pe.A_MESURER for r in rows)
    assert all("exp_S38_summary.json" in r["s38_na_reason"] for r in rows)


def test_concordance_du_taux_de_mise_a_jour_est_calculee(s38_factice):
    """Critère d'acceptation n° 1 : la concordance est CALCULÉE, avec son écart relatif."""
    s38 = pe.load_s38_summary(s38_factice)
    attendu = pe.s38_update_rate(s38, "monitoring", "gated_truelabel")
    assert attendu == 0.0251

    proche = pe.update_rate_agreement(0.0255, attendu)
    assert proche["agreement"] is True
    assert proche["relative_gap"] < pe.UPDATE_RATE_TOLERANCE

    loin = pe.update_rate_agreement(0.5, attendu)
    assert loin["agreement"] is False
    assert loin["measured"] == 0.5

    inconnu = pe.update_rate_agreement(None, attendu)
    assert inconnu["agreement"] is None


def test_ordre_des_politiques_constate_jamais_postule():
    """L'ordre frozen < gated < always est structurel sur le NOMBRE de mises à jour ; sur
    l'ÉNERGIE il est constaté. Un ordre inattendu est rapporté, pas corrigé."""
    cells = {
        "frozen_monitoring": _cell("frozen", "monitoring", 20e-6, 46e-3),
        "gated_truelabel_monitoring": _cell("gated_truelabel", "monitoring", 30e-6, 46e-3),
        "gated_pseudolabel_monitoring": _cell("gated_pseudolabel", "monitoring", 31e-6, 46e-3),
        "always_monitoring": _cell("always", "monitoring", 95e-6, 46e-3),
    }
    assert pe.ordering_check(cells, "monitoring")["ordered"] is True

    cells["always_monitoring"] = _cell("always", "monitoring", 1e-6, 46e-3)
    inverse = pe.ordering_check(cells, "monitoring")
    assert inverse["ordered"] is False
    assert inverse["observed_order"][0] == "always"


# ── 6. Témoin de séance ──────────────────────────────────────────────────────

def test_temoin_declare_les_seances_comparables_ou_non():
    """P0/P1 et P2/P3 sont mesurées dans des séances séparées par un reflash : sans témoin,
    une dérive de banc se lirait comme un effet du gate."""
    stable = {"default_monitoring": {"slope_ua_per_hz": 60.0, "slope_std_ua_per_hz": 2.0},
              "gate_monitoring": {"slope_ua_per_hz": 61.0, "slope_std_ua_per_hz": 2.0}}
    assert pe.anchor_drift(stable)["comparable"] is True

    derive = {"default_monitoring": {"slope_ua_per_hz": 60.0, "slope_std_ua_per_hz": 1.0},
              "gate_monitoring": {"slope_ua_per_hz": 90.0, "slope_std_ua_per_hz": 1.0}}
    res = pe.anchor_drift(derive)
    assert res["comparable"] is False
    assert "porte cette dérive" in res["rationale"]


def test_temoin_na_avec_une_seule_seance():
    res = pe.anchor_drift({"default_monitoring": {"slope_ua_per_hz": 60.0}})
    assert res["comparable"] is None
    assert "moins de deux séances" in res["na_reason"]


# ── 7. Cohérence avec le pilote du Sprint 38 ─────────────────────────────────

def test_drapeaux_de_compilation_miroirs_du_sprint_38():
    """Les drapeaux doivent être ceux de `run_sprint38_board.build_and_flash_gated`.

    S'ils divergeaient, une politique serait mesurée sur un binaire qui ne porte pas son
    gate — et l'écart mesuré n'aurait plus rien à voir avec la politique nommée.
    """
    src = (ROOT / "scripts" / "run_sprint38_board.py").read_text(encoding="utf-8")
    assert "-DEWC_AUTO_UPDATE" in src
    assert "-DGATE_PSEUDO_LABEL" in src
    assert pe.BUILD_BY_POLICY["frozen"] == pe.BUILD_BY_POLICY["always"] == ""
    assert pe.BUILD_BY_POLICY["gated_truelabel"] == "-DEWC_AUTO_UPDATE"
    assert pe.BUILD_BY_POLICY["gated_pseudolabel"] == \
        "-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL"
    # P0 et P1 partagent le binaire : c'est ce qui rend leur différence intra-séance.
    assert pe.BUILD_NAME_BY_POLICY["frozen"] == pe.BUILD_NAME_BY_POLICY["always"]
    assert len({pe.BUILD_NAME_BY_POLICY[p] for p in pe.POLICIES}) == 3


def test_drapeau_uart_update_porte_par_always_seul():
    """Miroir de `run_sprint38_board.py` : `request_update = (policy == "always")`.

    Les politiques gated streament SANS le drapeau : le firmware décide seul. Le leur
    donner ferait mesurer une mise à jour systématique sous le nom d'une politique gated.
    """
    src = (ROOT / "scripts" / "run_sprint38_board.py").read_text(encoding="utf-8")
    assert 'request_update = (policy == "always")' in src
    assert pe.UPDATE_FLAG_BY_POLICY == {"frozen": False, "always": True,
                                        "gated_truelabel": False,
                                        "gated_pseudolabel": False}


def test_pilote_ajoute_condition_et_update_a_la_commande_de_flux():
    """`rc.stream_command` ne porte ni `--update` ni `--condition` : le pilote doit les
    ajouter, sinon P1 ≡ P0 et les colonnes diffèreraient de celles du Sprint 38."""
    pilote = _load("run_s53_policy_energy",
                   ROOT / "scripts" / "run_s53_policy_energy.py")
    frozen = pilote.policy_stream_command("frozen", "monitoring", "/dev/ttyACM0", 100, 50.0)
    always = pilote.policy_stream_command("always", "monitoring", "/dev/ttyACM0", 100, 50.0)

    assert "--update" not in frozen
    assert "--update" in always
    for cmd in (frozen, always):
        assert cmd[cmd.index("--condition") + 1] == pe.CONDITION
        assert cmd[cmd.index("--model") + 1] == pilote.EWC_MODEL_FLAG


def test_compteur_du_gate_lu_sur_le_dernier_echantillon():
    """`forgetting` est un compteur CUMULÉ : le sommer multiplierait le nombre de MAJ."""
    pilote = _load("run_s53_policy_energy",
                   ROOT / "scripts" / "run_s53_policy_energy.py")
    flux = {"samples": [{"forgetting": 0.0, "auroc": 0.0},
                        {"forgetting": 1.0, "auroc": 2.0},
                        {"forgetting": 1.0, "auroc": 0.0},
                        {"forgetting": 2.0, "auroc": 2.0}]}

    res = pilote.gate_counters(flux)

    assert res["n_updates"] == 2
    assert res["update_rate"] == pytest.approx(0.5)
    assert res["counter_monotonic"] is True
    assert res["plausible_gate_build"] is True


def test_compteur_du_gate_na_sur_le_binaire_par_defaut():
    """Sans les slots réinterprétés, le taux n'est pas déductible — et on le dit."""
    pilote = _load("run_s53_policy_energy",
                   ROOT / "scripts" / "run_s53_policy_energy.py")
    res = pilote.gate_counters({"samples": [{"pred": 0, "true": 0}]})
    assert res["n_updates"] is None
    assert "binaire par défaut" in res["na_reason"]


def test_sensor_stream_expose_les_slots_reinterpretes():
    """L'ajout à `--dump-samples` est ce qui rend le taux MESURABLE depuis un
    sous-processus. Strictement additif : le format de trame ne change pas."""
    src = (ROOT / "scripts" / "sensor_stream.py").read_text(encoding="utf-8")
    assert 'for slot in ("auroc", "forgetting")' in src
    assert "RESPONSE_V3_FMT" in src, "le format de réponse V3 doit rester en place"


# ── 8. Garde AST et honnêteté du schéma ──────────────────────────────────────

def test_pilote_sans_resultat_en_dur():
    """Garde AST : aucun résultat numérique figé dans le pilote (précédent S50/S53).

    Les seuls flottants tolérés sont des valeurs par défaut de CLI et de mise en page —
    les seuils de décision vivent dans `policy_energy.py`, où ils sont testés.
    """
    src = (ROOT / "scripts" / "run_s53_policy_energy.py").read_text(encoding="utf-8")
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
    """L'estimateur par différence de pentes ne doit JAMAIS être fusionné avec celui par
    delta (S5302) ni avec celui de la régression simple (S5304) : leur comparaison est
    elle-même un résultat."""
    assert pe.METHOD != rr.METHOD
    delta = (ROOT / "scripts" / "run_s50_energy_delta.py").read_text(encoding="utf-8")
    assert pe.METHOD not in delta


def test_plafond_uart_respecte_par_les_cadences_par_defaut():
    """La borne haute reste sous le plafond de transport mesuré (~209 inf/s), et le repos
    ancre `I_base`."""
    pilote = _load("run_s53_policy_energy",
                   ROOT / "scripts" / "run_s53_policy_energy.py")
    assert max(pilote.DEFAULT_RATES) <= 209
    assert 0 in pilote.DEFAULT_RATES
    assert 0 in pilote.ANCHOR_RATES
    assert len(set(pilote.ANCHOR_RATES)) >= 3, "trois cadences distinctes au minimum"


# ── 9. Schéma des sorties (skip tant que le banc n'a pas tourné) ─────────────

@pytest.mark.skipif(not ECONOMY.is_file(), reason="campagne S5306 non encore mesurée")
def test_schema_economy_energy():
    economy = json.loads(ECONOMY.read_text(encoding="utf-8"))
    for cle in ("method", "by_dataset", "by_cell", "anchor_drift", "cross_table",
                "battery_capacities_mah", "sources"):
        assert cle in economy
    for dataset, bloc in economy["by_dataset"].items():
        assert "energy_uj_per_update" in bloc
        update = bloc["energy_uj_per_update"]
        assert "value_uj" in update or "na_reason" in update
        if update.get("value_uj") == pe.A_MESURER:
            assert update["na_reason"], "un N/A sort TOUJOURS avec sa raison mesurée"


@pytest.mark.skipif(not ECONOMY.is_file(), reason="campagne S5306 non encore mesurée")
def test_cellules_mesurees_sans_erreur_crc():
    """Toute mesure d'énergie est précédée d'une non-régression du flux (règle du sprint)."""
    for path in EXP_DIR.glob("*.json"):
        if path.name in {"economy_energy.json", "firmware_state.json"}:
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        assert cell["protocol_check"]["crc_errors"] == 0, f"{path.name} : erreurs CRC"
        assert cell["policy"] in pe.POLICIES
        assert cell["build_name"] == pe.BUILD_NAME_BY_POLICY[cell["policy"]]
