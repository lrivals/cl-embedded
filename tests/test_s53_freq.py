"""
test_s53_freq.py — Balayage de fréquence SYSCLK 180/90/45 MHz (Sprint 53, S5303).

Ce que ces tests protègent, et qui ne se voit PAS sur le banc :

1. **L'invariant UART.** `USART3->BRR = 0x0187` est figé en dur dans `hw_uart_init` et
   calculé pour PCLK1 = 45 MHz. Le balayage ne tient que parce que PPRE1 absorbe la
   variation de PLLP. Le jour où quelqu'un touche un prescaler sans toucher au BRR, le
   protocole tombe entièrement — et le seul symptôme serait des trames corrompues sur
   une carte. Ici, ça tombe en CI.
2. **La dérive C↔Python.** `PLL_BY_MHZ` (pilote) décrit dans le JSON la configuration
   supposée du silicium. Elle est confrontée aux `#define` réellement compilés.
3. **L'honnêteté des conclusions.** La tendance énergétique est CALCULÉE ; « à mesurer »
   sort quand elle ne l'est pas, jamais un 0.

Les tests de mesure proprement dits (latences, courant, plafond 59 mA) `skip` tant que
les cellules n'existent pas : la carte et la sonde sont requises pour les produire.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HW_INFO_C = ROOT / "firmware" / "stm32f4_blink" / "src" / "hw_info.c"
MAKEFILE = ROOT / "firmware" / "stm32f4_blink" / "Makefile"
EXP_DIR = ROOT / "experiments" / "exp_S53_freq_sweep"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fs = _load("run_s53_freq_sweep", ROOT / "scripts" / "run_s53_freq_sweep.py")
rr = _load("rate_regression", ROOT / "src" / "evaluation" / "rate_regression.py")

#: Décodage des champs RCC (RM0090) — c'est la table du manuel, pas un résultat.
PLLP_DIV = {0: 2, 1: 4, 2: 6, 3: 8}
PPRE_DIV = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 4, 6: 8, 7: 16}
WS_TOKEN = {"FLASH_ACR_LATENCY_1WS": 1, "FLASH_ACR_LATENCY_2WS": 2,
            "FLASH_ACR_LATENCY_5WS": 5}
VCO_MHZ = 360   # PLLM=8, PLLN=180 sur HSI 16 MHz — invariant du balayage


def _parse_hw_info_blocks() -> dict[int, dict]:
    """Extrait de `hw_info.c` la configuration RÉELLEMENT compilée par fréquence."""
    src = HW_INFO_C.read_text(encoding="utf-8")
    blocs = re.split(r"#(?:el)?if\s+SYSCLK_MHZ\s*==\s*(\d+)", src)
    out: dict[int, dict] = {}
    for mhz, corps in zip(blocs[1::2], blocs[2::2]):
        corps = corps.split("#else")[0]

        def defval(nom: str) -> str:
            found = re.search(rf"#define\s+{nom}\s+(\S+)", corps)
            assert found, f"{nom} absent du bloc SYSCLK_MHZ == {mhz}"
            return found.group(1)

        out[int(mhz)] = {
            "pllp": PLLP_DIV[int(defval("HWCLK_PLLP_BITS").rstrip("UL"), 0)],
            "ppre1": PPRE_DIV[int(defval("HWCLK_PPRE1_BITS").rstrip("UL"), 0)],
            "ppre2": PPRE_DIV[int(defval("HWCLK_PPRE2_BITS").rstrip("UL"), 0)],
            "flash_ws": WS_TOKEN[defval("HWCLK_FLASH_WS")],
            "overdrive": bool(int(defval("HWCLK_OVERDRIVE"))),
            "vos": defval("HWCLK_VOS"),
        }
    return out


# ── Firmware : ce que le C compile réellement ────────────────────────────────

def test_les_trois_frequences_sont_definies():
    blocs = _parse_hw_info_blocks()
    assert set(blocs) == set(fs.PLL_BY_MHZ), (
        f"le C définit {sorted(blocs)}, le pilote décrit {sorted(fs.PLL_BY_MHZ)}")


def test_pllp_produit_bien_la_frequence_annoncee():
    """SYSCLK = VCO / PLLP : la fréquence n'est pas déclarative, elle se calcule."""
    for mhz, bloc in _parse_hw_info_blocks().items():
        assert VCO_MHZ / bloc["pllp"] == mhz, (
            f"SYSCLK_MHZ == {mhz} configure PLLP=/{bloc['pllp']} → "
            f"{VCO_MHZ / bloc['pllp']:.0f} MHz")


def test_pclk1_reste_a_45_mhz_partout():
    """L'invariant qui sauve `USART3->BRR = 0x0187` — donc tout le protocole UART."""
    for mhz, bloc in _parse_hw_info_blocks().items():
        pclk1 = mhz * 1_000_000 // bloc["ppre1"]
        assert pclk1 == fs.PCLK1_HZ, (
            f"à {mhz} MHz, PPRE1=/{bloc['ppre1']} donne PCLK1={pclk1} Hz : le BRR figé "
            f"à 0x0187 n'est plus valide, le protocole tombe.")


def test_table_du_pilote_miroir_du_firmware():
    """`PLL_BY_MHZ` décrit le JSON : elle doit décrire le silicium, pas une intention."""
    for mhz, bloc in _parse_hw_info_blocks().items():
        attendu = fs.PLL_BY_MHZ[mhz]
        for champ in ("pllp", "ppre1", "ppre2", "flash_ws", "overdrive"):
            assert attendu[champ] == bloc[champ], (
                f"{mhz} MHz, champ {champ} : pilote={attendu[champ]} vs "
                f"firmware={bloc[champ]}")
        assert attendu["pclk2_hz"] == mhz * 1_000_000 // bloc["ppre2"]


def test_overdrive_et_wait_states_suivent_la_frequence():
    """L'overdrive et les wait states ne sont pas cosmétiques : c'est le gain attendu."""
    blocs = _parse_hw_info_blocks()
    assert blocs[180]["overdrive"] is True, "l'overdrive est requis au-delà de 168 MHz"
    for mhz in (90, 45):
        assert blocs[mhz]["overdrive"] is False, (
            f"overdrive laissé actif à {mhz} MHz : une part du gain énergétique est perdue")
    # Moins de fréquence ⇒ moins d'attente Flash, sinon on paye des cycles pour rien.
    ordonnees = [blocs[m]["flash_ws"] for m in sorted(blocs)]
    assert ordonnees == sorted(ordonnees), f"wait states non monotones : {ordonnees}"


def test_build_par_defaut_inchange_et_valeurs_non_supportees_rejetees():
    src = HW_INFO_C.read_text(encoding="utf-8")
    assert re.search(r"#ifndef\s+SYSCLK_MHZ\s+#define\s+SYSCLK_MHZ\s+180",
                     src, re.MULTILINE) or re.search(
        r"#ifndef\s+SYSCLK_MHZ\n#define\s+SYSCLK_MHZ\s+180", src), (
        "sans défaut à 180, un build ordinaire changerait de fréquence")
    assert "#error" in src, (
        "une valeur non supportée doit casser la compilation, pas produire une "
        "configuration silencieusement fausse")


def test_cible_makefile_verifiable_sans_carte():
    """`check-sysclk` est ce qui prouve les trois builds quand la carte est absente."""
    mk = MAKEFILE.read_text(encoding="utf-8")
    assert "check-sysclk:" in mk
    assert "SYSCLK_LIST" in mk
    assert re.search(r"\.PHONY:[^\n]*(\\\n[^\n]*)*check-sysclk", mk), (
        "check-sysclk absent de .PHONY")


# ── Pilote : verdicts calculés, N/A honnêtes ─────────────────────────────────

def test_verdict_gap2_est_chiffre():
    verdict = fs.gap2_verdict(45, 8380.0, {"hdc": 8380.0, "ewc": 200.0})
    assert "TENU" in verdict and "8380" in verdict and "hdc" in verdict
    depasse = fs.gap2_verdict(45, 120_000.0, {"hdc": 120_000.0})
    assert "DÉPASSÉ" in depasse


def test_verdict_gap2_sans_mesure_ne_conclut_pas():
    """Le verdict ne se déduit pas de la fréquence : sans latence, pas de conclusion."""
    verdict = fs.gap2_verdict(45, None, {"hdc": None})
    assert "exige une mesure" in verdict


def test_rapport_1_sur_f_est_calcule():
    cells = {
        "180": {"dwt_latency_us_p50": {"ewc": 50.0, "hdc": 2000.0}},
        "45": {"dwt_latency_us_p50": {"ewc": 200.0, "hdc": None}},
    }
    scaling = fs.latency_scaling(cells)
    assert scaling["45"]["ratio_attendu"] == 4.0
    assert scaling["45"]["ratio_mesure_par_modele"]["ewc"] == 4.0
    assert scaling["45"]["ratio_mesure_par_modele"]["hdc"] is None


def test_rapport_1_sur_f_sans_reference_reste_na():
    scaling = fs.latency_scaling({"45": {"dwt_latency_us_p50": {"ewc": 200.0}}})
    assert "na_reason" in scaling


def _ecrire_cellule(out_dir: Path, mhz: int, energie, **extra) -> None:
    cell = {
        "sysclk_mhz": mhz,
        "energy_uj_per_inference": energie,
        "slope_ua_per_hz": 1.0,
        "i_mean_ma_by_rate": [{"rate_hz": 0.0, "i_ma": 46.0}],
        "dwt_latency_us_p50": {"ewc": 50.0 * fs.REF_MHZ / mhz},
        "gap2_ok": True,
        "gap2_worst_us": 50.0 * fs.REF_MHZ / mhz,
        "gap2_verdict": "…",
        "acqmode_dyn": {"succeeded": False, "i_max_ma": 63.2},
    }
    cell.update(extra)
    (out_dir / f"{mhz}.json").write_text(json.dumps(cell), encoding="utf-8")


@pytest.mark.parametrize("e180, e45, attendu", [
    (100.0, 100.0, "constante"),
    (100.0, 50.0, "croissante_avec_f"),     # ralentir réduit le coût par inférence
    (50.0, 100.0, "decroissante_avec_f"),   # ralentir allonge sans rien gagner
])
def test_tendance_energetique_est_calculee(tmp_path, e180, e45, attendu):
    _ecrire_cellule(tmp_path, 45, e45)
    _ecrire_cellule(tmp_path, 180, e180)
    summary = fs.build_summary(tmp_path)
    assert summary["tendance_energie"] == attendu
    assert summary["tendance_rationale"], "une tendance sans justification n'est pas lisible"


def test_tendance_exige_deux_points_chiffres(tmp_path):
    """Un seul point mesuré ⇒ « à mesurer », jamais une tendance inventée."""
    _ecrire_cellule(tmp_path, 180, 100.0)
    _ecrire_cellule(tmp_path, 45, fs.A_MESURER)
    summary = fs.build_summary(tmp_path)
    assert summary["tendance_energie"] == fs.A_MESURER
    assert "au moins deux points" in summary["tendance_rationale"]


def test_summary_vide_ne_fabrique_rien(tmp_path):
    summary = fs.build_summary(tmp_path)
    assert summary["frequencies_mhz"] == []
    assert summary["tendance_energie"] == fs.A_MESURER


def test_echec_du_mode_dynamique_est_consigne_avec_sa_valeur():
    """Critère d'acceptation : le plafond 59 mA se consigne CHIFFRÉ, succès ou échec."""
    class _SondeQuiRefuse:
        pass

    def _capture_qui_echoue(*_a, **_k):
        raise RuntimeError("dynamic mode rejected: measured 63.24 mA above 59 mA limit")

    ancien = fs.lp.capture
    fs.lp.capture = _capture_qui_echoue
    try:
        dyn = fs.try_dynamic_mode(_SondeQuiRefuse(), 3300, 1.0)
    finally:
        fs.lp.capture = ancien
    assert dyn["attempted"] is True and dyn["succeeded"] is False
    assert dyn["i_max_ma"] == pytest.approx(63.24)
    assert "na_reason" in dyn


def test_controle_de_frequence_desactive_sort_un_na_explicite():
    """`--skip-hw-check` ne doit jamais produire un `sysclk_match: true` par défaut."""
    class _Args:
        skip_hw_check = True
    check = fs.check_frequency(_Args())
    assert check["sysclk_match"] is None
    assert check["sysclk_reported_mhz"] is None
    assert "na_reason" in check


def test_pilote_sans_resultat_en_dur():
    """Garde AST : aucune valeur de mesure figée dans le pilote S5303.

    Les littéraux autorisés sont tous des grandeurs de CONFIGURATION (budget Gap 2,
    conversions d'unités, seuils de décision, réglages CLI) — aucune n'est un résultat.
    """
    src = (ROOT / "scripts" / "run_s53_freq_sweep.py").read_text(encoding="utf-8")
    autorises = {
        0.0, 1.0, 2.0, 4.0, 8.0, 10.0, 100.0,      # réglages CLI (fenêtre, répétitions…)
        0.1, 0.9,                                   # seuils de décision (10 %, r² mini)
        0.95,                                       # DYN_COMPLETENESS : fraction des
                                                    # échantillons attendus sous laquelle
                                                    # une acquisition dyn est tenue pour
                                                    # interrompue (surintensité)
        3.3,                                        # tension de service par défaut
        1000.0, 1_000_000.0,                        # conversions A→mA, A·s→µJ
        100_000.0,                                  # budget Gap 2 en µs
    }
    suspects = {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in autorises
    }
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"


# ── Cellules mesurées : ne s'exécutent qu'une fois le banc passé ─────────────

def _cellules_mesurees() -> list[Path]:
    if not EXP_DIR.is_dir():
        return []
    return [p for p in sorted(EXP_DIR.glob("*.json")) if p.name != "summary.json"]


def test_cellules_declarent_la_frequence_constatee():
    cellules = _cellules_mesurees()
    if not cellules:
        pytest.skip("aucune cellule mesurée (carte + sonde requises)")
    for path in cellules:
        cell = json.loads(path.read_text(encoding="utf-8"))
        check = cell["hw_check"]
        assert check["sysclk_match"] in (True, None), path.name
        if check["sysclk_match"]:
            assert check["sysclk_reported_mhz"] == cell["sysclk_mhz"], path.name
            assert check["pclk1_reported_hz"] == fs.PCLK1_HZ, path.name
        else:
            assert check["na_reason"], path.name


def test_cellules_mesurees_tiennent_le_gap2():
    cellules = _cellules_mesurees()
    if not cellules:
        pytest.skip("aucune cellule mesurée (carte + sonde requises)")
    for path in cellules:
        cell = json.loads(path.read_text(encoding="utf-8"))
        assert cell["gap2_worst_us"] is None or \
            cell["gap2_worst_us"] < fs.GAP2_BUDGET_US, path.name
        assert cell["gap2_verdict"], path.name


def test_cellules_mesurees_sans_perte_de_trame():
    """Contrôle d'intégrité : sous V3, c'est l'échantillon MANQUANT qui trahit la perte."""
    cellules = _cellules_mesurees()
    if not cellules:
        pytest.skip("aucune cellule mesurée (carte + sonde requises)")
    for path in cellules:
        cell = json.loads(path.read_text(encoding="utf-8"))
        for model, bloc in cell["protocol_check"].items():
            assert bloc["samples_lost"] == 0, f"{path.name} / {model}"


# ── Ajustement délégué à rate_regression (A3) ────────────────────────────────

def test_la_cellule_porte_lerreur_type_de_la_pente():
    """Un verdict de tendance sans barre d'erreur n'est pas un verdict."""
    points = [{"rate_hz": r, "i_mean_a": i, "i_std_a": s} for r, i, s in
              [(0, 0.0399, 7e-6), (10, 0.0412, 5.7e-4), (25, 0.0414, 1.5e-3),
               (50, 0.0432, 1.4e-5), (100, 0.0463, 1.1e-4)]]
    cell = fs.derive_cell(points, 3.3, 50.0)
    assert cell["slope_std_ua_per_hz"] > 0
    assert cell["energy_uncertainty_uj"] > 0
    assert cell["weighted_by_inverse_variance"] is True


def test_les_ecarts_types_pesent_reellement_sur_lajustement():
    """Garde anti-régression : le retour d'un fit NON pondéré doit se voir.

    C'était le défaut d'origine — les `i_std_a` étaient jetés en chemin. Deux jeux de points
    identiques aux seuls écarts-types près doivent donner deux pentes différentes.
    """
    base = [(0, 0.0399), (10, 0.0412), (25, 0.0414), (50, 0.0432), (100, 0.0463)]
    egaux = [{"rate_hz": r, "i_mean_a": i, "i_std_a": 1e-5} for r, i in base]
    inegaux = [{"rate_hz": r, "i_mean_a": i, "i_std_a": s} for (r, i), s in
               zip(base, [1e-5, 2e-3, 2e-3, 1e-5, 1e-5])]
    assert (fs.derive_cell(egaux, 3.3, 50.0)["slope_ua_per_hz"]
            != pytest.approx(fs.derive_cell(inegaux, 3.3, 50.0)["slope_ua_per_hz"]))


def test_regression_non_concluante_sort_en_na_chiffre():
    """Sous le seuil de linéarité, aucun nombre n'est publié — une raison CHIFFRÉE l'est."""
    points = [{"rate_hz": r, "i_mean_a": i, "i_std_a": 1e-5} for r, i in
              [(0, 0.0399), (10, 0.0455), (25, 0.0402), (50, 0.0461), (100, 0.0410)]]
    cell = fs.derive_cell(points, 3.3, 50.0)
    assert cell["energy_uj_per_inference"] == fs.A_MESURER
    assert "r²=" in cell["energy_na_reason"]


def test_moins_de_trois_cadences_ne_publie_pas_de_pente():
    points = [{"rate_hz": 0.0, "i_mean_a": 0.04, "i_std_a": 1e-5},
              {"rate_hz": 100.0, "i_mean_a": 0.046, "i_std_a": 1e-5}]
    cell = fs.derive_cell(points, 3.3, 50.0)
    assert cell["energy_uj_per_inference"] == fs.A_MESURER
    assert "trois cadences" in cell["energy_na_reason"]


def test_lestimateur_garde_son_nom_propre():
    """Fréquence FIXÉE : ce n'est pas l'estimateur de S5304, il ne porte pas son nom."""
    points = [{"rate_hz": r, "i_mean_a": 0.04 + r * 6e-5, "i_std_a": 1e-5}
              for r in (0, 10, 25, 50, 100)]
    cell = fs.derive_cell(points, 3.3, 50.0)
    assert cell["method"] == fs.METHOD != rr.METHOD


def test_la_latence_par_modele_survit_a_la_derivation():
    """`fit_cell` renvoie une latence SCALAIRE sous une clé que la cellule utilise déjà
    pour un DICTIONNAIRE {modèle → latence} : l'écraser cassait le contrôle 1/f."""
    points = [{"rate_hz": r, "i_mean_a": 0.04 + r * 6e-5, "i_std_a": 1e-5}
              for r in (0, 10, 25, 50, 100)]
    cell = {"dwt_latency_us_p50": {"ewc": 50.0, "hdc": 585.0}}
    cell.update(fs.derive_cell(points, 3.3, 50.0))
    assert isinstance(cell["dwt_latency_us_p50"], dict)
    assert cell["regression_latency_us_p50"] == 50.0


def test_refit_ne_touche_pas_aux_points_mesures(tmp_path):
    """Recalcul hors banc : les grandeurs dérivées changent, les mesures jamais."""
    points = [{"rate_hz": r, "i_mean_a": 0.04 + r * 6e-5, "i_std_a": 1e-5,
               "n_repeats": 2} for r in (0, 10, 25, 50, 100)]
    cellule = {
        "sysclk_mhz": 180, "tension_v": 3.3, "current_points": points,
        "regression_model": "hdc_int8", "dwt_latency_us_p50": {"hdc": 585.0},
        # Champs d'une règle antérieure : ils doivent être remplacés, pas conservés.
        "slope_ua_per_hz": 999.0, "r2": 0.5,
    }
    (tmp_path / "180.json").write_text(json.dumps(cellule), encoding="utf-8")
    cells = fs.refit(tmp_path)
    relu = json.loads((tmp_path / "180.json").read_text(encoding="utf-8"))
    assert relu["current_points"] == points, "les mesures sont en lecture seule"
    assert relu["slope_ua_per_hz"] != 999.0 and relu["r2"] > 0.9
    assert set(cells) == {"180"}
