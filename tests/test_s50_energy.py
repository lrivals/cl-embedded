"""tests/test_s50_energy.py — Driver campagne énergie S50 (S5002/S5003).

Vérifie la couche S50 au-dessus de `energy_capture.py` :
    - placeholders honnêtes « à mesurer » quand aucun CSV (règle « aucun chiffre inventé ») ;
    - chemin mesuré sur un CSV synthétique (avec sync PA8) → µJ chiffrés, durées de
      phase déduites de la trace, µJ/inférence = E_inférence / N ;
    - autonomie recalculée (I_moy + balayage capacités) seulement si mesuré ;
    - le driver ne fabrique jamais 0 à la place de « à mesurer ».

Toutes les valeurs numériques sont des fixtures de test (aucune écriture dans experiments/).

Exécution : pytest tests/test_s50_energy.py -v
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from scripts.energy_capture import A_MESURER, PHASES
from scripts.run_s50_energy import (
    compute_cell,
    phase_durations_from_windows,
    run_from_manifest,
    write_autonomy,
)
from scripts.run_s50_int8_latency import SEGMENTS, _segment_stats

ROOT = Path(__file__).resolve().parents[1]
HW_PROFILE = ROOT / "configs" / "hw_profile_f439zi.yaml"
ENERGY_REAL_SRC = ROOT / "src" / "figures" / "catalogs" / "energy_real.py"

# Liste blanche de layout partagée avec test_figures_library (positions/largeurs/alpha).
_LAYOUT_WHITELIST: set[float] = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.12, 0.15, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.5, 0.55, 0.6, 0.72, 0.78, 0.8, 0.82, 0.86, 0.9, 0.92, 0.94, 0.98, 1.0, 1.05, 1.2, 1.4,
    1.5, 2.0, 4.5, 5.0, 8.0, 8.5, 9.0, 11.0,
}


def _write_csv(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── phase_durations_from_windows ──────────────────────────────────────────


def test_phase_durations_sum_by_phase():
    """Durées sommées par phase depuis les fenêtres PA8 ; toutes les phases présentes."""
    windows = [("idle", 0.0, 0.1), ("inference", 0.1, 0.3), ("idle", 0.3, 0.35)]
    d = phase_durations_from_windows(windows)
    assert set(d) == set(PHASES)
    assert d["inference"] == pytest.approx(0.2)
    assert d["idle"] == pytest.approx(0.15)
    assert d["acquisition"] == 0.0  # phase absente → 0, jamais fabriquée


# ── compute_cell : placeholder ────────────────────────────────────────────


def test_compute_cell_placeholder_when_no_csv():
    """Sans CSV → tous les champs énergie == « à mesurer », source placeholder."""
    cell = compute_cell("ewc", "int8", csv=None, n_inference=None,
                        n_update=None, tension_v=3.3)
    assert cell["source"] == "placeholder"
    assert cell["total_uj"] == A_MESURER
    assert cell["energy_uj_per_inference"] == A_MESURER
    assert all(cell["phases_uj"][p] == A_MESURER for p in PHASES)
    assert all(cell["phase_durations_s"][p] == A_MESURER for p in PHASES)
    assert cell["by_component"]["sensor"] == "na"  # capteur non isolable


# ── compute_cell : chemin mesuré (CSV synthétique) ────────────────────────


def test_compute_cell_measured_end_to_end(tmp_path):
    """CSV synthétique avec sync → µJ chiffrés + durées + µJ/inférence.

    Niveau sync haut sur [0.0,0.2) (2 échantillons), bas ensuite. dt=0.1 uniforme.
    Phase inference : E = 0.01·3.3·0.1·2 = 6600 µJ ; durée = 0.2 s.
    µJ/inférence avec N=4 → 6600/4 = 1650 µJ.
    """
    csv = _write_csv(
        tmp_path / "ewc_int8.csv",
        ["time,current,voltage,sync",
         "0.0,0.01,3.3,1", "0.1,0.01,3.3,1", "0.2,0.01,3.3,0", "0.3,0.01,3.3,0"],
    )
    cell = compute_cell("ewc", "int8", csv=str(csv), n_inference=4,
                        n_update=0, tension_v=3.3)
    assert cell["source"] == "lpm01a_csv"
    assert cell["phases_uj"]["inference"] == pytest.approx(6600.0)
    assert cell["phase_durations_s"]["inference"] == pytest.approx(0.2)
    assert cell["energy_uj_per_inference"] == pytest.approx(1650.0)
    assert isinstance(cell["total_uj"], float) and cell["total_uj"] > 0


def test_compute_cell_measured_without_n_inference(tmp_path):
    """Mesuré mais N inconnu → µJ total chiffré mais µJ/inférence « à mesurer »."""
    csv = _write_csv(
        tmp_path / "ewc_fp32.csv",
        ["time,current,voltage,sync", "0.0,0.01,3.3,1", "0.1,0.01,3.3,0"],
    )
    cell = compute_cell("ewc", "fp32", csv=str(csv), n_inference=None,
                        n_update=None, tension_v=3.3)
    assert isinstance(cell["total_uj"], float)
    assert cell["energy_uj_per_inference"] == A_MESURER  # pas de N → honnête


def test_compute_cell_refuses_csv_without_sync(tmp_path):
    """CSV réel sans sync → ValueError (pas de segmentation fabriquée)."""
    csv = _write_csv(tmp_path / "x.csv", ["time,current,voltage", "0.0,0.01,3.3"])
    with pytest.raises(ValueError, match="synchronisation"):
        compute_cell("ewc", "fp32", csv=str(csv), n_inference=1,
                    n_update=0, tension_v=3.3)


# ── autonomie (S5003) ─────────────────────────────────────────────────────


def test_autonomy_measured_from_durations(tmp_path):
    """Cellule mesurée (phases_uj + phase_durations_s) → I_moy + autonomie chiffrés."""
    cell = {
        "phases_uj": {"inference": 100.0, "idle": 5.0},
        "phase_durations_s": {"inference": 0.0005, "idle": 0.01},
        "tension_v": 3.3,
    }
    write_autonomy(tmp_path, {"ewc_int8": cell}, duty_cycle={"inference_period_s": 1.0})
    data = json.loads((tmp_path / "autonomy.json").read_text(encoding="utf-8"))
    entry = data["per_model"]["ewc_int8"]
    assert isinstance(entry["i_moy_ma"], float) and entry["i_moy_ma"] > 0
    assert all(isinstance(v, float) for v in entry["autonomy_h_by_mah"].values())


def test_autonomy_placeholder_when_not_measured(tmp_path):
    """Cellule placeholder → I_moy et autonomie restent « à mesurer »."""
    cell = {
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "tension_v": 3.3,
    }
    write_autonomy(tmp_path, {"ewc_int8": cell}, duty_cycle={})
    data = json.loads((tmp_path / "autonomy.json").read_text(encoding="utf-8"))
    entry = data["per_model"]["ewc_int8"]
    assert entry["i_moy_ma"] == A_MESURER
    assert all(v == A_MESURER for v in entry["autonomy_h_by_mah"].values())


# ── campagne complète depuis manifeste ────────────────────────────────────


def test_run_from_manifest_all_placeholder(tmp_path):
    """Manifeste sans CSV → 8 cellules placeholder + summary + autonomy honnêtes."""
    manifest = tmp_path / "campaign.yaml"
    manifest.write_text(
        "duty_cycle: {inference_period_s: 1.0}\n"
        "components: {mcu: null, periph: null, sensor: na}\n"
        "cells:\n"
        "  ewc_fp32: {csv: null, n_inference: null, n_update: null}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    run_from_manifest(manifest, out)
    assert (out / "ewc_fp32.json").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "autonomy.json").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["per_model"]["ewc"]["delta_int8_vs_fp32_uj"] == A_MESURER


def test_run_from_manifest_preserves_measured_cells(tmp_path):
    """Une cellule déjà mesurée sur carte n'est JAMAIS réécrite par le manifeste.

    Le protocole delta (S5008) écrit ses cellules dans le même répertoire et sous
    les mêmes noms : sans cette garde, rejouer le manifeste effacerait des mesures
    réelles au profit de placeholders (perte de données silencieuse).
    """
    manifest = tmp_path / "campaign.yaml"
    manifest.write_text(
        "duty_cycle: {inference_period_s: 1.0}\n"
        "cells:\n  ewc_fp32: {csv: null, n_inference: null, n_update: null}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    mesure = {
        "model": "ewc", "encoding": "fp32", "tension_v": 3.3,
        "source": "lpm01a_delta", "energy_uj_per_inference": 12.5,
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "delta_measurement": {"i_idle_a": 0.05, "i_active_a": 0.056,
                              "delta_a": 0.006, "window_s": 10.0},
    }
    (out / "ewc_fp32.json").write_text(json.dumps(mesure), encoding="utf-8")

    run_from_manifest(manifest, out)

    apres = json.loads((out / "ewc_fp32.json").read_text(encoding="utf-8"))
    assert apres["source"] == "lpm01a_delta"
    assert apres["energy_uj_per_inference"] == 12.5
    # …et les cellules non mesurées sont bien produites en placeholder.
    voisine = json.loads((out / "ewc_int8.json").read_text(encoding="utf-8"))
    assert voisine["source"] == "placeholder"


def test_autonomy_from_delta_measurement(tmp_path):
    """Sans profil par phase, l'autonomie se dérive du protocole delta (S5008)."""
    cell = {
        "tension_v": 3.3,
        "source": "lpm01a_delta",
        "energy_uj_per_inference": 40.0,
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "delta_measurement": {"i_idle_a": 0.05, "i_active_a": 0.058,
                              "delta_a": 0.008, "window_s": 10.0},
    }
    write_autonomy(tmp_path, {"ewc_int8": cell}, duty_cycle={"inference_period_s": 1.0})
    entry = json.loads((tmp_path / "autonomy.json").read_text(encoding="utf-8"))[
        "per_model"]["ewc_int8"]
    assert isinstance(entry["i_moy_ma"], float)
    # I_moy ≥ I_repos : l'activité ne peut que s'ajouter au courant de repos.
    assert entry["i_moy_ma"] >= 0.05 * 1e3
    assert entry["methode"].startswith("protocole delta")
    assert all(isinstance(v, float) for v in entry["autonomy_h_by_mah"].values())


def test_autonomy_delta_without_period_stays_na(tmp_path):
    """Delta mesuré mais période d'usage absente → « à mesurer », pas de période inventée."""
    cell = {
        "tension_v": 3.3,
        "source": "lpm01a_delta",
        "energy_uj_per_inference": 40.0,
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "delta_measurement": {"i_idle_a": 0.05, "i_active_a": 0.058,
                              "delta_a": 0.008, "window_s": 10.0},
    }
    write_autonomy(tmp_path, {"ewc_int8": cell}, duty_cycle={})
    entry = json.loads((tmp_path / "autonomy.json").read_text(encoding="utf-8"))[
        "per_model"]["ewc_int8"]
    assert entry["i_moy_ma"] == A_MESURER


# ── S5007 — tests d'honnêteté du sprint 50 ────────────────────────────────────


def test_uj_positive_only_if_csv(tmp_path):
    """µJ chiffré (> 0) UNIQUEMENT si un CSV réel est fourni ; sinon « à mesurer »."""
    # Sans CSV → « à mesurer » (jamais 0).
    placeholder = compute_cell("ewc", "int8", csv=None, n_inference=10,
                               n_update=0, tension_v=3.3)
    assert placeholder["total_uj"] == A_MESURER
    assert placeholder["energy_uj_per_inference"] == A_MESURER
    # Avec CSV réel synchronisé → µJ chiffré > 0.
    csv = _write_csv(
        tmp_path / "ewc_int8.csv",
        ["time,current,voltage,sync", "0.0,0.02,3.3,1", "0.1,0.02,3.3,1", "0.2,0.02,3.3,0"],
    )
    measured = compute_cell("ewc", "int8", csv=str(csv), n_inference=2,
                            n_update=0, tension_v=3.3)
    assert isinstance(measured["total_uj"], float) and measured["total_uj"] > 0
    assert isinstance(measured["energy_uj_per_inference"], float)
    assert measured["energy_uj_per_inference"] > 0


def test_component_na_honest():
    """`by_component.sensor` = « na » + raison explicite (jamais un delta fabriqué)."""
    cell = compute_cell("hdc", "fp32", csv=None, n_inference=None,
                        n_update=None, tension_v=3.3)
    bc = cell["by_component"]
    assert bc["sensor"] == "na"
    assert bc.get("sensor_na_reason")           # raison présente et non vide
    # MCU/périph restent « à mesurer » tant qu'aucun CSV séparé n'a tourné (jamais 0).
    assert bc["mcu"] == A_MESURER and bc["periph"] == A_MESURER


def test_int8_latency_segments():
    """Le breakdown latence INT8 contient les 3 segments dequant/mac/requant.

    Structure vérifiée sur fixture (aucune carte requise) ; si le JSON board réel existe
    (exp_S50_int8_latency/ewc.json), on vérifie AUSSI son contrat.
    """
    fake = [
        {"acc": 200.0, "auroc": 6800.0, "forgetting": 3700.0, "status": 0},
        {"acc": 200.0, "auroc": 6810.0, "forgetting": 3720.0, "status": 0},
    ]
    seg = _segment_stats(fake)
    assert set(seg) >= set(SEGMENTS) == {"dequant", "mac", "requant"}
    assert seg["mac"]["cycles_p50"] is not None and seg["mac"]["cycles_p50"] > 0

    board = ROOT / "experiments" / "exp_S50_int8_latency" / "ewc.json"
    if board.is_file():
        d = json.loads(board.read_text(encoding="utf-8"))
        assert set(d["segments"]) >= {"dequant", "mac", "requant"}


def test_cost_benefit_no_hardcode():
    """Garde AST « 0 chiffre en dur » sur le catalogue energy_real.py (résultats chargés)."""
    tree = ast.parse(ENERGY_REAL_SRC.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in _LAYOUT_WHITELIST
    }
    assert not offending, f"Littéraux flottants suspects dans energy_real.py : {sorted(offending)}"


def test_calibration_present():
    """`energy_calibration` présent dans le profil HW (résout TODO(dorra) S5001)."""
    cfg = yaml.safe_load(HW_PROFILE.read_text(encoding="utf-8"))
    assert "energy_calibration" in cfg
    assert cfg["energy_calibration"].get("sampling_rate_hz")   # consigne renseignée
