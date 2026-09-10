"""Tables d'annexe de l'article EWC INT8 (S4011).

Les annexes détaillent des dizaines de cellules ; l'article déclare qu'aucune valeur
n'y est saisie à la main. Ces tests verrouillent cette déclaration : les tables sont
générées depuis l'agrégat, régénérables à l'identique, et une cellule absente y sort
en tiret plutôt qu'en zéro.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "article" / "ewc_int8_mcu" / "tables"
AGG = ROOT / "experiments" / "exp_S40_article_metrics" / "summary.json"
GENERATOR = ROOT / "scripts" / "generate_article_tables.py"

NAMES = ("annex_depth", "annex_symmetry", "annex_ram", "annex_energy", "annex_missing")


def _require_tables() -> None:
    if not TABLES.exists():
        pytest.skip("tables d'annexe non générées")


class TestArticleTables:
    def test_all_tables_exist_in_both_languages(self):
        _require_tables()
        missing = [
            f"{name}_{lang}.tex"
            for name in NAMES
            for lang in ("fr", "en")
            if not (TABLES / f"{name}_{lang}.tex").exists()
        ]
        assert not missing, f"tables absentes : {missing}"

    def test_tables_are_marked_generated(self):
        """Un fichier généré doit le dire, sinon il finira édité à la main."""
        _require_tables()
        for path in sorted(TABLES.glob("*.tex")):
            head = path.read_text(encoding="utf-8").splitlines()[0]
            assert "Généré par" in head, f"{path.name} sans en-tête de génération"

    def test_regeneration_is_idempotent(self):
        """Régénérer ne doit rien changer : sinon les PDF et les sources divergent."""
        _require_tables()
        before = {p.name: p.read_text(encoding="utf-8") for p in sorted(TABLES.glob("*.tex"))}
        result = subprocess.run(
            [sys.executable, str(GENERATOR)], capture_output=True, text=True, cwd=ROOT
        )
        assert result.returncode == 0, result.stderr
        after = {p.name: p.read_text(encoding="utf-8") for p in sorted(TABLES.glob("*.tex"))}
        drift = sorted(k for k in before if before[k] != after.get(k))
        assert not drift, f"tables non reproductibles : {drift}"

    def test_missing_cells_render_as_dash_never_zero(self):
        """Règle « aucun chiffre inventé » : une cellule absente n'est pas un zéro."""
        _require_tables()
        agg = json.loads(AGG.read_text(encoding="utf-8"))
        assert agg.get("missing"), "l'agrégat ne déclare aucune cellule manquante"
        registry = (TABLES / "annex_missing_fr.tex").read_text(encoding="utf-8")
        for path in agg["missing"]:
            leaf = path.split(".")[-1]
            assert leaf.replace("_", "\\_") in registry, f"{leaf} absent du registre"
        assert "\\num{0.0000}" not in registry

    def test_fr_en_tables_have_same_shape(self):
        """Miroir strict : mêmes lignes de données des deux côtés."""
        _require_tables()
        for name in NAMES:
            fr = (TABLES / f"{name}_fr.tex").read_text(encoding="utf-8")
            en = (TABLES / f"{name}_en.tex").read_text(encoding="utf-8")
            assert fr.count("\\\\") == en.count("\\\\"), f"{name} : formes FR/EN divergentes"

    def test_values_come_from_the_aggregate(self):
        """Échantillon de contrôle : la RAM totale de la table est celle de l'agrégat."""
        _require_tables()
        agg = json.loads(AGG.read_text(encoding="utf-8"))
        table = (TABLES / "annex_ram_fr.tex").read_text(encoding="utf-8")
        for dataset in ("monitoring", "pronostia"):
            for scheme in ("fp32", "int8"):
                total = agg[dataset]["ram"][f"{scheme}_board_total"]["value"]
                assert f"\\num{{{int(total)}}}" in table, \
                    f"{dataset}/{scheme} total {total} absent de la table"
