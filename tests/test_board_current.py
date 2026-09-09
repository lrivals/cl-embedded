"""tests/test_board_current.py — Campagne « courant moyen » S5008.

Ce pilote a remplacé le protocole delta après deux constats de banc mesurés :
la première acquisition d'une session est biaisée (~+8 mA), et la référence
« au repos » du firmware est plus consommatrice que tous les régimes de flux
(attente UART par scrutation active). Ces tests verrouillent les invariants qui
en découlent — surtout l'honnêteté du N/A sur les µJ/inférence, qui est ici un
**résultat mesuré** et non un « pas encore fait ».

Aucune sonde ni carte requise : le schéma et les garde-fous sont testés hors banc.

Exécution : pytest tests/test_board_current.py -v
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


rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")

A_MESURER = ec.A_MESURER
CELLS_DIR = ROOT / "experiments" / "exp_S50_energy"


def _cell(runs=(0.0465, 0.0466, 0.0464), idle=0.0548):
    return rc.build_cell("ewc", "int8", list(runs), idle, 3.26, 10.0, 100.0, 0.0628)


class TestSchema:
    def test_grandeurs_mesurees_presentes(self):
        """Le courant moyen, sa dispersion et les tirages bruts sont conservés."""
        mes = _cell()["current_measurement"]
        assert mes["i_mean_a"] == pytest.approx(0.0465, abs=1e-6)
        assert mes["i_std_a"] > 0
        assert len(mes["i_runs_a"]) == mes["n_repeats"] == 3
        assert mes["i_idle_a"] == pytest.approx(0.0548)
        assert mes["warmup_discarded_a"] == pytest.approx(0.0628)

    def test_n_inference_exact_par_construction(self):
        """N = cadence × fenêtre : la cadence imposée le rend exact, pas estimé."""
        assert _cell()["n_inference"] == 1000
        cell = rc.build_cell("hdc", "fp32", [0.05], 0.055, 3.26, 20.0, 50.0, 0.062)
        assert cell["n_inference"] == 1000

    def test_ecart_type_nul_si_une_seule_repetition(self):
        assert _cell(runs=(0.0465,))["current_measurement"]["i_std_a"] == 0.0

    def test_source_declare_la_mesure(self):
        assert _cell()["source"] == "lpm01a_current"


class TestHonnetete:
    def test_uj_par_inference_na_avec_raison_mesuree(self):
        """Le N/A des µJ porte une raison MESURÉE — jamais 0, jamais « pas fait ».

        La raison a été RÉVISÉE au Sprint 53 : le contre-balancement (S5301) a établi
        que l'écart repos↔charge du Sprint 50 était un artefact d'ordre. Le test
        vérifie donc que la raison cite ce constat et renvoie vers les protocoles qui
        produisent réellement des µJ, au lieu de l'ancienne incertitude désormais levée.
        """
        cell = _cell()
        assert cell["energy_uj_per_inference"] == A_MESURER
        assert cell["energy_uj_per_inference"] != 0
        raison = cell["energy_na_reason"]
        # La raison doit citer le constat, pas l'absence de banc.
        assert "ARTEFACT D'ORDRE" in raison
        assert "établi" in raison
        # …et ne doit plus affirmer une cause non établie (constat levé par S5301).
        assert "cause n'est pas établie" not in raison

    def test_profil_par_phase_reste_na(self):
        cell = _cell()
        assert all(v == A_MESURER for v in cell["phases_uj"].values())
        assert all(v == A_MESURER for v in cell["phase_durations_s"].values())
        assert cell["phases_na_reason"]

    def test_capteur_toujours_na(self):
        """Capteurs simulés par UART (S5001) : « na », pas 0."""
        assert _cell()["by_component"]["sensor"] == "na"
        assert _cell()["by_component"]["sensor_na_reason"]

    def test_maha_int8_exclue_du_balayage_par_defaut(self):
        """Une cellule à firmware dédié ne doit pas être mesurée sur le mauvais build.

        Le Mahalanobis INT8 se choisit à la compilation (`-DMAHA_INT8`) : la mesurer
        sur le build par défaut écrirait un fichier qui ment sur son contenu — c'est
        exactement le bug corrigé au Sprint 52 côté drapeaux UART.
        """
        assert ("maha", "int8") in rc.BUILD_SPECIFIC
        assert ("maha", "int8") not in [c for c in rc.STREAM_MODEL
                                        if c not in rc.BUILD_SPECIFIC]

    def test_stream_command_porte_toujours_le_modele(self):
        """Sans `--model`, le firmware exécuterait Mahalanobis en silence."""
        for (model, enc), flag in rc.STREAM_MODEL.items():
            cmd = rc.stream_command(flag, "monitoring", "/dev/null", 1000, 100.0)
            assert "--model" in cmd and flag in cmd, (model, enc)
            assert "--rate-hz" in cmd


class TestCellulesProduites:
    """Contrôles sur les cellules réellement écrites (ignorés si absentes)."""

    def _cells(self):
        found = {}
        for model in ec.CAMPAIGN_MODELS:
            for enc in ec.CAMPAIGN_ENCODINGS:
                p = CELLS_DIR / f"{model}_{enc}.json"
                if p.is_file():
                    found[f"{model}_{enc}"] = json.loads(p.read_text(encoding="utf-8"))
        return found

    def test_courant_positif_et_dispersion_faible(self):
        cells = {k: c for k, c in self._cells().items()
                 if c.get("source") == "lpm01a_current"}
        if not cells:
            pytest.skip("aucune cellule mesurée")
        for name, c in cells.items():
            mes = c["current_measurement"]
            assert mes["i_mean_a"] > 0, name
            # Dispersion du banc mesurée à ±0,06 mA : au-delà de 1 mA d'écart-type,
            # la cellule n'est pas exploitable pour comparer des modèles.
            assert mes["i_std_a"] < 1e-3, f"{name} dispersion trop forte"

    def test_aucune_cellule_ne_chiffre_les_uj(self):
        """Tant que la référence est la scrutation active, aucun µJ ne doit apparaître."""
        for name, c in self._cells().items():
            if c.get("source") == "lpm01a_current":
                assert c["energy_uj_per_inference"] == A_MESURER, name
                assert c["energy_uj_per_update"] == A_MESURER, name


def test_pilote_sans_resultat_en_dur():
    """Garde AST : aucun résultat numérique figé dans le pilote.

    Les seuls flottants tolérés sont des valeurs par défaut de CLI et de mise en
    page (cadence, fenêtre, délai d'établissement) — pas des mesures.
    """
    src = (ROOT / "scripts" / "run_s50_board_current.py").read_text(encoding="utf-8")
    autorises = {0.0, 1.0, 2.0, 3.3, 4.0, 10.0, 100.0}
    suspects = {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in autorises
    }
    assert not suspects, f"littéraux flottants suspects : {sorted(suspects)}"
