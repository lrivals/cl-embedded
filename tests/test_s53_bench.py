"""
test_s53_bench.py — Règles de banc du Sprint 53 (contre-balancement et états de repos).

Les verdicts de banc sont CALCULÉS par `src/evaluation/counterbalance.py`, jamais saisis
dans les JSON : ces tests exercent la règle sur des données synthétiques (donc hors banc,
reproductibles) et vérifient le schéma des mesures réellement produites lorsqu'elles sont
présentes — sans jamais les fabriquer si elles sont absentes.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")

EXP_DIR = ROOT / "experiments" / "exp_S53_counterbalance"


# ── Règle du contre-balancement repos/charge (S5301) ─────────────────────────

def test_verdict_artefact_ordre():
    """Le repos part haut puis rejoint les cellules → artefact d'ordre."""
    idle = [{"session_index": i, "i_a": v}
            for i, v in enumerate([0.0548, 0.0500, 0.0462, 0.0461, 0.0460])]
    verdict, rationale = cb.counterbalance_verdict(idle, [0.0465, 0.0470])
    assert verdict == cb.VERDICT_ARTEFACT
    assert "mA" in rationale


def test_verdict_effet_reel():
    """Repos plat ET au-dessus des cellules → effet réel, cause non établie."""
    idle = [{"session_index": i, "i_a": v}
            for i, v in enumerate([0.0548, 0.0547, 0.0548, 0.0547, 0.0548])]
    verdict, _ = cb.counterbalance_verdict(idle, [0.0465, 0.0470])
    assert verdict == cb.VERDICT_EFFET_REEL


def test_verdict_derive_partielle():
    """Repos qui dérive nettement sans redescendre au niveau des cellules."""
    idle = [{"session_index": i, "i_a": v}
            for i, v in enumerate([0.0700, 0.0650, 0.0600, 0.0601, 0.0600, 0.0602])]
    verdict, _ = cb.counterbalance_verdict(idle, [0.0465, 0.0470])
    assert verdict == cb.VERDICT_DERIVE


def test_verdict_exige_des_mesures():
    """Sans mesure, pas de verdict — règle « aucun chiffre inventé »."""
    with pytest.raises(ValueError):
        cb.counterbalance_verdict([], [0.046])
    with pytest.raises(ValueError):
        cb.counterbalance_verdict([{"session_index": 0, "i_a": 0.05}], [])


def test_established_regime_ecarte_la_queue():
    """La queue d'établissement est écartée avant d'estimer la dispersion.

    Sans ce tri, la dérive gonfle sa propre barre d'erreur et le test devient
    trivialement concluant.
    """
    points = [{"session_index": i, "i_a": v}
              for i, v in enumerate([0.0620, 0.0550, 0.0461, 0.0460, 0.0461, 0.0460])]
    etabli = cb.established_regime(points)
    assert [i for i, _ in etabli] == [2, 3, 4, 5]
    assert cb.bench_dispersion_a([v for _, v in etabli]) < 1e-3


def test_idle_drift_slope_negative_si_le_repos_descend():
    points = [(0, 0.055), (1, 0.050), (2, 0.047), (3, 0.046)]
    assert cb.idle_drift_slope(points) < 0.0
    assert cb.idle_drift_slope([(0, 0.05)]) == 0.0


# ── Règle des états de repos (S5301b — de quel « repos » parle-t-on ?) ───────

def _states(port_ferme, port_ouvert, flux, apres_flux):
    return {
        cb.STATE_PORT_CLOSED: port_ferme,
        cb.STATE_PORT_OPEN: port_ouvert,
        cb.STATE_STREAM: flux,
        cb.STATE_POST_STREAM: apres_flux,
    }


def test_etats_de_repos_coherents():
    """Repos indiscernables : l'écart entre sessions vient d'ailleurs."""
    states = _states([0.0398, 0.0399], [0.0398, 0.0400], [0.0419, 0.0420], [0.0398, 0.0399])
    verdict, _ = cb.idle_state_verdict(states)
    assert verdict == cb.VERDICT_REPOS_COHERENT


def test_repos_post_flux_different():
    """Un repos post-flux franchement plus bas est signalé, pas absorbé."""
    states = _states([0.0500, 0.0501], [0.0500, 0.0499], [0.0520, 0.0521], [0.0398, 0.0399])
    verdict, rationale = cb.idle_state_verdict(states)
    assert verdict == cb.VERDICT_POST_FLUX
    assert cb.STATE_POST_STREAM in rationale


def test_etat_du_port_significatif():
    states = _states([0.0500, 0.0501], [0.0430, 0.0431], [0.0520, 0.0521], [0.0500, 0.0499])
    verdict, _ = cb.idle_state_verdict(states)
    assert verdict == cb.VERDICT_PORT_STATE


def test_residu_significatif_mais_negligeable():
    """Sur un banc très stable, 0,1 mA sort à > 3σ sans rien changer.

    Le résidu doit être signalé dans la justification, pas promu en verdict : il pèse
    moins que `MARGINAL_FRACTION` du surcoût de charge mesuré dans la même session.
    """
    states = _states([0.03982, 0.03983], [0.03984, 0.03985],
                     [0.04200, 0.04201], [0.03993, 0.03994])
    verdict, rationale = cb.idle_state_verdict(states)
    assert verdict == cb.VERDICT_REPOS_COHERENT
    assert "négligeable" in rationale


def test_pas_de_verdict_sans_dispersion_estimable():
    """Aucun état répété → refus de conclure (le repli rendrait tout significatif)."""
    states = _states([0.0398], [0.0399], [0.0420], [0.0480])
    verdict, rationale = cb.idle_state_verdict(states)
    assert verdict == cb.VERDICT_INDETERMINE
    assert "--repeats" in rationale


def test_idle_state_verdict_exige_une_reference():
    with pytest.raises(ValueError):
        cb.idle_state_verdict({cb.STATE_STREAM: [0.042, 0.042]})


def test_pooled_dispersion_intra_etat():
    """La dispersion est estimée DANS les états, pas entre eux."""
    groups = [[0.0398, 0.0399, 0.0398], [0.0500, 0.0501, 0.0500]]
    assert cb.pooled_dispersion_a(groups) < 1e-4          # écart inter-états ignoré
    assert cb.pooled_dispersion_a([[0.04], [0.05]]) == cb.FALLBACK_DISPERSION_A


# ── Schéma des mesures réellement produites (jamais fabriquées) ──────────────

@pytest.mark.parametrize("nom", ["idle_states_diagnostic.json", "idle_states_power_cycle.json"])
def test_schema_diagnostic_etats(nom):
    path = EXP_DIR / nom
    if not path.exists():
        pytest.skip(f"{nom} non mesuré sur ce poste (banc requis)")
    d = json.loads(path.read_text(encoding="utf-8"))
    for champ in ("sequence", "state_mean_a", "pooled_dispersion_a", "verdict",
                  "verdict_rationale", "board_responsive_after"):
        assert champ in d, champ
    assert d["verdict"] in {cb.VERDICT_REPOS_COHERENT, cb.VERDICT_POST_FLUX,
                            cb.VERDICT_PORT_STATE, cb.VERDICT_INDETERMINE}
    # Le rang de chaque acquisition est consigné : l'ordre ne peut pas être confondu
    # avec la condition (leçon S5301).
    assert [p["session_index"] for p in d["sequence"]] == list(range(len(d["sequence"])))
    # Le verdict du JSON est bien celui que la règle recalcule sur ces mêmes courants.
    verdict, _ = cb.idle_state_verdict(
        d["states_a"], {k: all(v) for k, v in d["board_responsive_after"].items() if v}
    )
    assert verdict == d["verdict"]


def test_schema_bascule_premier_flux():
    """Le basculement de niveau au premier flux, s'il a été mesuré, est bien tracé."""
    path = EXP_DIR / "idle_states_power_cycle.json"
    if not path.exists():
        pytest.skip("mesure de coupure d'alimentation absente (banc requis)")
    pc = json.loads(path.read_text(encoding="utf-8")).get("power_cycle_check")
    if pc is None:
        pytest.skip("session sans --with-power-cycle")
    for champ in ("settling_series_a", "post_first_stream_series_a", "step_first_stream_a",
                  "i_idle_port_open_before_any_frame_a", "board_responsive"):
        assert champ in pc, champ
    # La carte répond après la coupure : l'état mesuré est bien une carte vivante.
    assert pc["board_responsive"] is True
    # Le pas est CALCULÉ depuis les deux séries, jamais saisi.
    avant = sum(p["i_a"] for p in pc["settling_series_a"]) / len(pc["settling_series_a"])
    apres = sum(p["i_a"] for p in pc["post_first_stream_series_a"]) / len(
        pc["post_first_stream_series_a"])
    assert pc["step_first_stream_a"] == pytest.approx(apres - avant, abs=1e-9)


def test_pilote_diagnostic_sans_resultat_en_dur():
    """Garde AST : aucun résultat de mesure figé dans le pilote de diagnostic.

    Seuls sont tolérés les défauts de CLI et de mise en page.
    """
    src = (ROOT / "scripts" / "diag_s53_idle_states.py").read_text(encoding="utf-8")
    autorises = {0.0, 1.0, 2.0, 3.3, 4.0, 10.0, 100.0}
    suspects = {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in autorises
    }
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"


# ── S5302 — repos réel (WFI) et lot d'inférences ─────────────────────────────

wfi = _load("run_s53_wfi", ROOT / "scripts" / "run_s53_wfi.py")
rc50 = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")

WFI_DIR = ROOT / "experiments" / "exp_S53_wfi"
HW_PROFILE = ROOT / "configs" / "hw_profile_f439zi.yaml"


def _cell(i_mean_a: float, n_inference: int = 1000) -> dict:
    """Cellule minimale au schéma attendu par `build_delta_recovery`."""
    return {"current_measurement": {"i_mean_a": i_mean_a}, "n_inference": n_inference}


def test_delta_na_avec_raison_actualisee():
    """Delta ≤ 0 ⇒ « à mesurer », avec une raison propre à S5302.

    La raison du Sprint 50 attribuait le N/A à une cause NON établie. S5301 l'a
    levée : republier la même phrase reviendrait à publier une incertitude que la
    mesure a résolue.
    """
    out = wfi.build_delta_recovery(
        {"maha_fp32": _cell(0.0460)}, i_idle_a=0.0480, voltage_v=3.3,
        window_s=10.0, firmware_build="UART_WFI_IDLE",
    )["maha_fp32"]
    assert out["energy_uj_per_inference"] == wfi.A_MESURER
    assert "na_reason" in out
    assert out["na_reason"] != rc50.NA_PER_INFERENCE
    assert "UART_WFI_IDLE" in out["na_reason"]
    assert out["delta_a"] < 0


def test_delta_chiffre_si_repos_sous_la_charge():
    """Delta > 0 ⇒ µJ calculés, avec la méthode nommée (pas de valeur orpheline)."""
    out = wfi.build_delta_recovery(
        {"hdc_int8": _cell(0.0512, n_inference=1000)}, i_idle_a=0.0460,
        voltage_v=3.3, window_s=10.0, firmware_build="UART_WFI_IDLE",
    )["hdc_int8"]
    assert out["energy_uj_per_inference"] != wfi.A_MESURER
    assert out["method"] == "delta vs repos UART_WFI_IDLE"
    # Recalcul indépendant : (ΔI × V × T / N) × 1e6
    attendu = (0.0512 - 0.0460) * 3.3 * 10.0 / 1000 * 1e6
    assert out["energy_uj_per_inference"] == pytest.approx(attendu, rel=1e-9)
    assert "na_reason" not in out


def test_regression_lot_lineaire():
    """La régression I(N) retrouve la pente imposée sur des points alignés."""
    pente, ordonnee, r2 = wfi.linear_fit([1.0, 10.0, 50.0, 100.0],
                                         [0.0460 + 2e-6 * n for n in (1, 10, 50, 100)])
    assert pente == pytest.approx(2e-6, rel=1e-6)
    assert ordonnee == pytest.approx(0.0460, abs=1e-9)
    assert r2 >= 0.95


def test_regression_lot_refuse_un_seul_point():
    """Un point unique ne définit pas de pente — pas de r² trompeur à 1,0."""
    pente, _, r2 = wfi.linear_fit([1.0], [0.0460])
    assert pente == 0.0 and r2 == 0.0


def test_profil_hw_refuse_sans_mesure(tmp_path):
    """Le profil matériel ne se remplit pas par estimation."""
    cible = tmp_path / "hw.yaml"
    cible.write_text(HW_PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError):
        wfi.write_hw_profile_currents(cible, veille_a=None, actif_a=None, source="x")
    # Le fichier est laissé strictement intact.
    assert cible.read_text(encoding="utf-8") == HW_PROFILE.read_text(encoding="utf-8")


def test_profil_hw_ecrit_la_mesure_et_sa_provenance(tmp_path):
    """Les courants mesurés arrivent dans `puissance_watts`, commentaires préservés."""
    yaml = pytest.importorskip("yaml")
    cible = tmp_path / "hw.yaml"
    avant = HW_PROFILE.read_text(encoding="utf-8")
    cible.write_text(avant, encoding="utf-8")

    ecrites = wfi.write_hw_profile_currents(
        cible, veille_a=0.0121, actif_a=0.0465, source="idle_reference.json (WFI)")
    assert set(ecrites) == {"veille_uA", "actif_mA"}

    d = yaml.safe_load(cible.read_text(encoding="utf-8"))["hardware"]["puissance_watts"]
    assert d["veille_uA"] == pytest.approx(12100.0, rel=1e-6)
    assert d["actif_mA"] == pytest.approx(46.5, rel=1e-6)
    assert "idle_reference.json" in d["mesure_source"]
    # Le fichier reste documenté : les commentaires ne sont pas perdus.
    apres = cible.read_text(encoding="utf-8")
    assert apres.count("#") >= avant.count("#") - 2
    # Idempotent : ré-écrire ne duplique pas la provenance.
    wfi.write_hw_profile_currents(cible, veille_a=0.0121, actif_a=None, source="s2")
    assert cible.read_text(encoding="utf-8").count("mesure_source:") == 1


def test_profil_hw_ne_devine_pas_lactif(tmp_path):
    """Sans cellule mesurée, `actif_mA` est laissé TEL QUEL — il ne se déduit pas du repos.

    L'assertion porte sur « inchangé », pas sur « null » : depuis S5302 le profil
    réel porte un actif mesuré, et un test écrit contre `null` confondrait
    « la fonction respecte l'existant » avec « la clé est vide ».
    """
    yaml = pytest.importorskip("yaml")
    cible = tmp_path / "hw.yaml"
    cible.write_text(HW_PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    avant = yaml.safe_load(cible.read_text(encoding="utf-8"))["hardware"]["puissance_watts"]
    ecrites = wfi.write_hw_profile_currents(cible, veille_a=0.0121, actif_a=None,
                                            source="idle_reference.json (WFI)")
    assert ecrites == ["veille_uA"]
    d = yaml.safe_load(cible.read_text(encoding="utf-8"))["hardware"]["puissance_watts"]
    assert d["actif_mA"] == avant["actif_mA"]


def test_schema_reference_repos_wfi():
    """Schéma de `idle_reference.json`, et gain WFI recalculé depuis les deux repos."""
    path = WFI_DIR / "idle_reference.json"
    if not path.exists():
        pytest.skip("référence de repos non mesurée sur ce poste (banc requis)")
    d = json.loads(path.read_text(encoding="utf-8"))
    builds = {k: v for k, v in d.items()
              if isinstance(v, dict) and "i_idle_established_a" in v}
    assert builds, "aucun build de repos consigné"
    for nom, ref in builds.items():
        assert ref["firmware_build"] == nom
        for champ in ("i_idle_runs_a", "i_idle_established_a", "bench_dispersion_a",
                      "n_idle_discarded_as_settling", "tension_v", "window_s"):
            assert champ in ref, f"{nom}: {champ}"
        assert [p["session_index"] for p in ref["i_idle_runs_a"]] == \
            list(range(len(ref["i_idle_runs_a"])))
    if "UART_WFI_IDLE" not in builds:
        pytest.skip("build WFI pas encore mesuré (carte requise)")
    g = d["wfi_gain"]
    assert g["gain_ma"] == pytest.approx(
        (g["i_idle_scrutation_a"] - g["i_idle_wfi_a"]) * 1000.0, rel=1e-9)


def test_schema_cellules_wfi_tracent_le_build():
    """Une cellule ne se lit que si elle dit sur QUEL firmware elle a été prise."""
    # La liste des fichiers « qui ne sont pas des cellules » vit dans le pilote :
    # la dupliquer ici la ferait diverger au premier rapport ajouté.
    cellules = wfi.iter_cell_files(WFI_DIR) if WFI_DIR.is_dir() else []
    if not cellules:
        pytest.skip("cellules WFI non mesurées (carte requise)")
    for p in cellules:
        c = json.loads(p.read_text(encoding="utf-8"))
        assert c["firmware_build"], p.name
        assert isinstance(c["infer_batch_n"], int) and c["infer_batch_n"] >= 1, p.name


def test_delta_reporte_dans_les_cellules(tmp_path):
    """Deux fichiers du même répertoire ne disent pas le contraire sur une cellule.

    La cellule est écrite par le protocole « courant moyen » (S50), qui laisse
    l'énergie par inférence à « à mesurer ». Une fois le delta chiffré, la raison
    du N/A doit disparaître de la cellule — sinon c'est elle, plus lisible qu'un
    JSON annexe, qui serait recopiée.
    """
    (tmp_path / "ewc_fp32.json").write_text(json.dumps({
        "firmware_build": "UART_WFI_IDLE",
        "current_measurement": {"i_mean_a": 0.0319},
        "energy_uj_per_inference": wfi.A_MESURER,
        "energy_na_reason": "raison du Sprint 50",
    }), encoding="utf-8")
    (tmp_path / "maha_fp32.json").write_text(json.dumps({
        "firmware_build": "UART_WFI_IDLE",
        "current_measurement": {"i_mean_a": 0.0270},
        "energy_uj_per_inference": wfi.A_MESURER,
        "energy_na_reason": "raison du Sprint 50",
    }), encoding="utf-8")
    recovery = {
        "firmware_build": "UART_WFI_IDLE",
        "cells": {
            "ewc_fp32": {"energy_uj_per_inference": 146.7, "delta_a": 0.0045,
                         "method": "delta vs repos UART_WFI_IDLE"},
            "maha_fp32": {"energy_uj_per_inference": wfi.A_MESURER, "delta_a": -0.0004,
                          "na_reason": "delta négatif face au repos UART_WFI_IDLE"},
        },
    }
    wfi.propagate_delta_into_cells(recovery, tmp_path)

    chiffree = json.loads((tmp_path / "ewc_fp32.json").read_text(encoding="utf-8"))
    assert chiffree["energy_uj_per_inference"] == 146.7
    assert "energy_na_reason" not in chiffree
    assert chiffree["energy_source"] == "delta_recovery.json"

    na = json.loads((tmp_path / "maha_fp32.json").read_text(encoding="utf-8"))
    assert na["energy_uj_per_inference"] == wfi.A_MESURER
    assert na["energy_na_reason"] == "delta négatif face au repos UART_WFI_IDLE"
    assert "energy_method" not in na


def _point(n: int, i_a: float, saturated: bool = False) -> dict:
    return {"n": n, "i_mean_a": i_a, "saturated": saturated}


def test_lot_ecarte_les_points_satures():
    """Un point saturé n'entre pas dans la pente — il l'aplatirait.

    Au-delà du plafond de cadence, `sensor_stream` sature EN SILENCE : aucune trame
    perdue, aucun CRC, le flux tourne simplement moins vite. Le courant moyen porte
    alors MOINS d'inférences que « cadence × N », donc la pente — et l'énergie qui
    s'en déduit — sont sous-estimées.
    """
    droite = [_point(n, 0.030 + 1.5e-7 * n) for n in (1, 50, 100)]
    sature = _point(200, 0.030 + 1.5e-7 * 120, saturated=True)   # courbe rabattue
    entry = {"points": droite + [sature]}
    wfi.apply_fit(entry, voltage_v=3.3, rate_hz=100.0)
    assert entry["n_points_excluded_saturation"] == 1
    assert entry["r2"] > 0.99
    assert entry["slope_a_per_inference_per_frame"] == pytest.approx(1.5e-7, rel=1e-6)
    # Avec le point saturé, la pente serait plus faible : c'est bien lui qui trompe.
    biaise = {"points": droite + [dict(sature, saturated=False)]}
    wfi.apply_fit(biaise, voltage_v=3.3, rate_hz=100.0)
    assert biaise["slope_a_per_inference_per_frame"] < entry["slope_a_per_inference_per_frame"]


def test_lot_retire_la_raison_periemee_quand_il_chiffre():
    """Une cellule chiffrée ne conserve pas la `na_reason` d'un passage précédent.

    Sans ce nettoyage, un fichier porterait à la fois une valeur et la raison de
    son absence — et c'est la raison, plus lisible, qui serait recopiée.
    """
    entry = {"points": [_point(n, 0.030 + 1e-7 * n) for n in (1, 50, 100)],
             "na_reason": "raison d'un passage précédent",
             "energy_uj_per_inference": wfi.A_MESURER}
    wfi.apply_fit(entry, voltage_v=3.3, rate_hz=100.0)
    assert "na_reason" not in entry
    assert entry["energy_uj_per_inference"] != wfi.A_MESURER
    assert entry["method"].startswith("régression I(N)")


def test_lot_na_si_trop_peu_de_points_non_satures():
    """Trois points mesurés dont deux saturés ⇒ N/A honnête, pas une pente sur 1 point."""
    entry = {"points": [_point(1, 0.030), _point(50, 0.031, True), _point(100, 0.032, True)],
             "energy_uj_per_inference": 12.0, "method": "obsolète"}
    wfi.apply_fit(entry, voltage_v=3.3, rate_hz=100.0)
    assert entry["energy_uj_per_inference"] == wfi.A_MESURER
    assert "1 point(s) de N exploitable(s) sur 3" in entry["na_reason"]
    assert "method" not in entry


def test_parite_lot_appariee_par_features():
    """La parité du lot s'apparie sur les features, jamais sur l'indice.

    Deux flux successifs ne parcourent pas la même séquence d'échantillons : un
    appariement par indice comparerait des échantillons différents et inventerait
    des désaccords.
    """
    ref = {"1,2": 1, "3,4": 0}
    echantillons = [{"features": [3.0, 4.0], "pred": 0},    # commun, d'accord
                    {"features": [1.0, 2.0], "pred": 0},    # commun, en désaccord
                    {"features": [9.0, 9.0], "pred": 1}]    # hors référence
    commun, desaccords = wfi.prediction_parity(ref, echantillons)
    assert (commun, desaccords) == (2, 1)


def test_cellules_et_rapports_ne_se_confondent_pas():
    """`iter_cell_files` ne retient que les cellules — pas les rapports du répertoire."""
    noms = {p.name for p in wfi.iter_cell_files(WFI_DIR)} if WFI_DIR.is_dir() else set()
    for rapport in ("idle_reference.json", "delta_recovery.json", "batch_sweep.json",
                    "autonomy_delta.json", "wake_validation.json"):
        assert rapport not in noms


def test_schema_lot_trace_cadence_et_parite():
    """Chaque point de lot dit à quelle cadence il a été pris et s'il tient la trame."""
    path = WFI_DIR / "batch_sweep.json"
    if not path.exists():
        pytest.skip("balayage de lot non mesuré (carte requise)")
    d = json.loads(path.read_text(encoding="utf-8"))
    for cle, entry in d.items():
        assert entry["saturation_rule"]
        for p in entry["points"]:
            for champ in ("achieved_rate_hz", "saturated", "crc_errors",
                          "pred_parity_n_common"):
                assert champ in p, f"{cle} N={p['n']} : {champ}"
            assert p["crc_errors"] == 0, f"{cle} N={p['n']} : trames corrompues"
            if p["n"] > 1 and p["pred_parity_vs_n1"] is not None:
                # Invariant du lot : seule la DERNIÈRE inférence alimente la réponse,
                # la prédiction doit donc être celle de N = 1.
                assert p["pred_parity_vs_n1"] == 1.0, f"{cle} N={p['n']}"


def test_autonomie_delta_na_honnete():
    """L'autonomie ne se calcule que sur des µJ mesurés ; sinon elle reste absente."""
    path = WFI_DIR / "autonomy_delta.json"
    if not path.exists():
        pytest.skip("autonomie WFI non produite (carte requise)")
    d = json.loads(path.read_text(encoding="utf-8"))
    for cle, m in d["per_model"].items():
        if m.get("autonomy_h_by_period") is None:
            assert m.get("na_reason"), cle
        else:
            assert m["energy_uj_per_inference"] > 0.0, cle
            assert m["methode"], cle


def test_pilote_wfi_sans_resultat_en_dur():
    """Garde AST : aucun résultat de mesure figé dans le pilote S5302."""
    src = (ROOT / "scripts" / "run_s53_wfi.py").read_text(encoding="utf-8")
    # 0.95 : seuil de la règle de saturation (une RÈGLE, pas un résultat), au même
    # titre que 0.9 pour le r² minimal.
    autorises = {0.0, 1.0, 2.0, 3.3, 4.0, 10.0, 100.0, 0.9, 0.95, 1000.0, 1e6, 1e3}
    suspects = {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in autorises
    }
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"
