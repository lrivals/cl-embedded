"""Tests du protocole delta d'énergie (S5002, contrainte matérielle LPM01A).

Vérifie le calcul µJ/inférence par différence de courants moyens, et surtout
l'honnêteté du schéma : ce que le protocole delta ne peut pas mesurer doit
rester « à mesurer », jamais 0.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")
rd = _load("run_s50_energy_delta", ROOT / "scripts" / "run_s50_energy_delta.py")


class TestCalculDelta:
    def test_valeur_analytique(self):
        """(I_act−I_repos)=1 mA, V=3,3 V, T=10 s, N=1000 → 33 µJ/inférence."""
        uj = ec.energy_uj_per_inference_delta(
            i_idle_a=0.059, i_active_a=0.060, voltage_v=3.3, window_s=10.0, n_inference=1000
        )
        assert uj == pytest.approx(33.0, rel=1e-9)

    def test_proportionnel_au_delta_de_courant(self):
        a = ec.energy_uj_per_inference_delta(0.05, 0.051, 3.3, 10.0, 100)
        b = ec.energy_uj_per_inference_delta(0.05, 0.052, 3.3, 10.0, 100)
        assert b == pytest.approx(2 * a)

    def test_inverse_au_nombre_dinferences(self):
        a = ec.energy_uj_per_inference_delta(0.05, 0.051, 3.3, 10.0, 100)
        b = ec.energy_uj_per_inference_delta(0.05, 0.051, 3.3, 10.0, 200)
        assert b == pytest.approx(a / 2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"i_idle_a": 0.06, "i_active_a": 0.06},  # delta nul
            {"i_idle_a": 0.06, "i_active_a": 0.059},  # delta négatif
        ],
    )
    def test_delta_non_concluant_renvoie_none(self, kwargs):
        """Un delta non positif n'est pas une mesure : None, surtout pas 0."""
        out = ec.energy_uj_per_inference_delta(
            voltage_v=3.3, window_s=10.0, n_inference=100, **kwargs
        )
        assert out is None

    @pytest.mark.parametrize("n,window", [(0, 10.0), (-5, 10.0), (100, 0.0)])
    def test_parametres_invalides(self, n, window):
        assert ec.energy_uj_per_inference_delta(0.05, 0.06, 3.3, window, n) is None


class TestSchemaCellule:
    def test_cellule_mesuree(self):
        cell = rd.build_cell("ewc", "int8", 0.059, 0.060, 3.3, 10.0, 1000)
        assert cell["source"] == "lpm01a_delta"
        assert cell["energy_uj_per_inference"] == pytest.approx(33.0)
        assert cell["delta_measurement"]["delta_a"] == pytest.approx(1e-3)

    def test_phases_restent_a_mesurer_avec_raison(self):
        """Le profil par phase est hors de portée : « à mesurer » + raison."""
        cell = rd.build_cell("ewc", "int8", 0.059, 0.060, 3.3, 10.0, 1000)
        for phase in ec.PHASES:
            assert cell["phases_uj"][phase] == ec.A_MESURER
            assert cell["phase_durations_s"][phase] == ec.A_MESURER
        assert cell["phases_na_reason"]
        assert cell["energy_uj_per_update"] == ec.A_MESURER

    def test_cellule_non_concluante_reste_honnete(self):
        cell = rd.build_cell("hdc", "fp32", 0.060, 0.060, 3.3, 10.0, 1000)
        assert cell["source"] == "placeholder"
        assert cell["energy_uj_per_inference"] == ec.A_MESURER
        assert cell["energy_uj_per_inference"] != 0
        assert "na_reason" in cell

    def test_capteur_toujours_na(self):
        cell = rd.build_cell("maha", "fp32", 0.059, 0.061, 3.3, 10.0, 500)
        assert cell["by_component"]["sensor"] == "na"
        assert cell["by_component"]["sensor_na_reason"]

    def test_cle_compatible_avec_le_schema_existant(self):
        """Mêmes clés que le schéma de référence de `run_s50_energy.compute_cell`.

        La référence est le schéma **produit par le code**, pas un fichier de
        `experiments/` : ces fichiers appartiennent au dernier pilote qui a tourné
        (aujourd'hui `run_s50_board_current.py`, qui ajoute ses propres clés), et
        s'y comparer ferait échouer ce test au gré des campagnes.
        """
        re_ = _load("run_s50_energy", ROOT / "scripts" / "run_s50_energy.py")
        ref_keys = set(re_.compute_cell(
            model="ewc", encoding="int8", csv=None, n_inference=1000,
            n_update=None, tension_v=3.3,
        ).keys())
        cell_keys = set(rd.build_cell("ewc", "int8", 0.059, 0.06, 3.3, 10.0, 1000).keys())
        assert ref_keys <= cell_keys, f"clés manquantes : {ref_keys - cell_keys}"
