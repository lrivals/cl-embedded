"""Tests Sprint 40 / S4003 — validation board kernel INT8 v2 + notebook de synthèse.

Vérifie la **structure** et la **cohérence** des artefacts (pas les valeurs mesurées, qui
dépendent de la carte) :

  * ``exp_S40_board_v2/results_{scheme}_{ds}_{proto}.json`` : clés requises, CRC == 0,
    Gap 2 (latence < 100 ms), ratios RAM (per_channel/int8_legacy → 4.0, q15 → 2.0),
    accord INT8↔FP32 ∈ [0, 1] et ≥ 0.95 pour les schémas v2 frozen ;
  * parité S4002 : ``parity_{scheme}_frozen_{ds}.json`` v2 == 1.0 (``exact_vs_emulator``),
    forme des rows (``match == (pred_pc == pred_board)``) ;
  * fonctions pures (sans board) : ratios RAM des schémas, référence émulateur ;
  * notebook ``synthesis.ipynb`` : 5 PNG référencés, ``save_figure`` utilisé, garde ``HAS_S40``
    et constante ``NA`` présents (règle « aucun chiffre inventé »), pas de littéral métrique
    en dur dans les cellules qui chargent la campagne board v2.

Robuste aux artefacts absents (skip). Le firmware EWC v1 est inchangé (chemin v2 sous
``#ifdef EWC_INT8_V2``) → Unity ``make test`` reste à 0 régression (hors périmètre ici).

Exécution :
    pytest tests/test_sprint40_article.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXP = Path("experiments")
S40_DIR = EXP / "exp_S40_board_v2"
NB = Path("notebooks/cl_eval/article_ewc_int8/synthesis.ipynb")
FIGS = Path("docs/figures/sprint40_article")
PROVENANCE = FIGS / "provenance_table.csv"
ARTICLE = Path("docs/article/ewc_int8_mcu")
TEX_FR = ARTICLE / "main_fr.tex"
TEX_EN = ARTICLE / "main_en.tex"

DATASETS = ["pronostia", "monitoring"]
SCHEMES = ["per_channel", "q15", "int8_legacy"]
V2_SCHEMES = ["per_channel", "q15"]
PROTOCOLS = ["frozen", "online"]
GAP2_LATENCY_US = 100_000
RAM_RATIO = {"per_channel": 4.0, "int8_legacy": 4.0, "q15": 2.0}
EXPECTED_FIGS = [
    "fig1_parity_fp32_pc_board.png",
    "fig2_latency_gap2.png",
    "fig3_ablation_ladder.png",
    "fig4_int8_recovery_board.png",
    "fig5_pareto_ram_f1_latency.png",
]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ── Résultats board v2 (skip si campagne non exécutée) ────────────────────────

class TestS40Results:
    def test_results_keys_present_when_measured(self):
        seen = 0
        for scheme in SCHEMES:
            for ds in DATASETS:
                for proto in PROTOCOLS:
                    r = _load(S40_DIR / f"results_{scheme}_{ds}_{proto}.json")
                    if r is None:
                        continue
                    seen += 1
                    assert {"scheme", "kernel", "protocol", "dataset", "f1_faulty",
                            "latency_us_p50", "bss_bytes", "crc_errors", "parity_rate",
                            "gap2_latency_compliant"} <= set(r)
                    assert r["scheme"] == scheme and r["dataset"] == ds and r["protocol"] == proto
        if seen == 0:
            pytest.skip("exp_S40_board_v2 absent (lancer run_s40_board_v2.py)")

    def test_crc_zero_and_gap2(self):
        seen = 0
        for scheme in SCHEMES:
            for ds in DATASETS:
                for proto in PROTOCOLS:
                    r = _load(S40_DIR / f"results_{scheme}_{ds}_{proto}.json")
                    if r is None:
                        continue
                    if r.get("crc_errors") is not None:
                        assert r["crc_errors"] == 0, f"{scheme}/{ds}/{proto} CRC={r['crc_errors']}"
                    lat = r.get("latency_us_p50")
                    if isinstance(lat, (int, float)):
                        seen += 1
                        assert lat < GAP2_LATENCY_US, f"{scheme}/{ds}/{proto} lat={lat}"
                        assert r["gap2_latency_compliant"] is True
                    bss = r.get("bss_bytes")
                    if isinstance(bss, int):
                        assert bss < 256 * 1024
        if seen == 0:
            pytest.skip("aucune latence mesurée (board non exécutée)")

    def test_ram_ratio_by_scheme(self):
        seen = 0
        for scheme in SCHEMES:
            for ds in DATASETS:
                for proto in PROTOCOLS:
                    r = _load(S40_DIR / f"results_{scheme}_{ds}_{proto}.json")
                    if r is None:
                        continue
                    ratio = r.get("ram_ratio_fp32_over_quant")
                    if ratio is not None:
                        seen += 1
                        assert abs(ratio - RAM_RATIO[scheme]) < 0.01, \
                            f"{scheme} ratio={ratio} attendu {RAM_RATIO[scheme]}"
        if seen == 0:
            pytest.skip("aucune cellule RAM mesurée")

    def test_v2_frozen_agreement_high(self):
        seen = 0
        for scheme in V2_SCHEMES:
            for ds in DATASETS:
                r = _load(S40_DIR / f"results_{scheme}_{ds}_frozen.json")
                if r is None:
                    continue
                agree = r.get("agreement_int8_vs_fp32")
                if agree is not None:
                    seen += 1
                    assert 0.0 <= agree <= 1.0
                    assert agree >= 0.95, f"{scheme}/{ds} accord v2↔FP32={agree} < 0.95"
        if seen == 0:
            pytest.skip("aucune cellule v2 frozen mesurée")


# ── Parité S4002 ──────────────────────────────────────────────────────────────

class TestS40Parity:
    def test_v2_frozen_parity_exact(self):
        checked = 0
        for scheme in V2_SCHEMES:
            for ds in DATASETS:
                r = _load(S40_DIR / f"parity_{scheme}_frozen_{ds}.json")
                if r is None or r.get("parity_rate") is None:
                    continue
                checked += 1
                assert r["parity_class"] == "exact_vs_emulator"
                assert r["parity_rate"] == 1.0, f"{scheme}/{ds} parity={r['parity_rate']}"
                assert r["mismatch_count"] == 0
        if checked == 0:
            pytest.skip("aucune parité v2 frozen (lancer board_pc_parity.py --exp exp_S40_board_v2)")

    def test_parity_rows_shape(self):
        checked = 0
        for scheme in SCHEMES:
            for ds in DATASETS:
                for proto in PROTOCOLS:
                    r = _load(S40_DIR / f"parity_{scheme}_{proto}_{ds}.json")
                    if r is None or not r.get("rows"):
                        continue
                    checked += 1
                    assert {"scheme", "protocol", "dataset", "parity_rate", "rows"} <= set(r)
                    for row in r["rows"][:50]:
                        assert {"idx", "true", "pred_pc", "pred_board", "match"} <= set(row)
                        assert row["match"] == (row["pred_pc"] == row["pred_board"])
        if checked == 0:
            pytest.skip("aucun fichier parité S40")


# ── Fonctions pures (sans board) ──────────────────────────────────────────────

class TestPureFunctions:
    def test_ram_block_ratios(self):
        from scripts.run_s40_board_v2 import _ram_block
        for k in (5, 13, 21):
            assert abs(_ram_block("per_channel", k)["ram_ratio_fp32_over_quant"] - 4.0) < 0.01
            assert abs(_ram_block("int8_legacy", k)["ram_ratio_fp32_over_quant"] - 4.0) < 0.01
            assert abs(_ram_block("q15", k)["ram_ratio_fp32_over_quant"] - 2.0) < 0.01

    def test_emulator_reference_recovers_vs_collapses(self):
        # Sans board : l'émulateur PC (référence bit-exacte du kernel) doit récupérer la F1
        # en per_channel/q15 et s'effondrer en legacy (cœur du résultat de l'article).
        import numpy as np
        from src.evaluation.feature_conditions import load_condition_arrays
        from src.evaluation.metrics import compute_fault_f1
        from src.utils.int8_c_emulation import calibrate_activations
        from scripts.run_s40_board_v2 import _load_head, _emu_pred

        ck = EXP / "exp_S36_PC_5feat_ewc_pronostia" / "checkpoints" / "ewc_head.pt"
        if not ck.exists():
            pytest.skip("checkpoint S36 absent")
        X, y, _idx, _n = load_condition_arrays("pronostia", "5feat", "ewc", seed=42)
        w = _load_head(ck)
        am = calibrate_activations(w, X)
        f1 = {s: compute_fault_f1(y, _emu_pred(w, X.astype(np.float32), s, am))["f1_faulty"]
              for s in SCHEMES}
        assert f1["per_channel"] > 0.5 and f1["q15"] > 0.5
        assert f1["int8_legacy"] < 0.5
        assert f1["per_channel"] - f1["int8_legacy"] > 0.4   # récupération nette


# ── Notebook de synthèse ──────────────────────────────────────────────────────

class TestNotebook:
    def _nb_src(self):
        if not NB.exists():
            pytest.skip("synthesis.ipynb absent")
        nb = json.loads(NB.read_text())
        return nb, "\n".join("".join(c.get("source", "")) for c in nb["cells"]
                             if c["cell_type"] == "code")

    def test_five_figures_referenced(self):
        _nb, src = self._nb_src()
        for fig in EXPECTED_FIGS:
            assert fig in src, f"{fig} non référencé dans le notebook"
        assert "save_figure" in src

    def test_graceful_degradation_guard(self):
        _nb, src = self._nb_src()
        # Garde de dégradation gracieuse + règle « aucun chiffre inventé ».
        assert "HAS_S40" in src
        assert 'NA = "à mesurer"' in src or "NA=" in src

    def test_no_hardcoded_metric_in_board_v2(self):
        # Les cellules chargeant S40 ne doivent pas contenir de F1 board v2 en dur : tout passe
        # par S40[...] / json.load. Heuristique : aucune assignation de float « métrique » à côté
        # d'un accès S40 (on vérifie que la campagne est lue via la structure, pas recopiée).
        nb, _src = self._nb_src()
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            text = "".join(c.get("source", ""))
            if "S40[" in text:
                assert "json" in text or "load(" in text or "results_" in text or "S40 =" in text \
                    or "r.get(" in text or "r[" in text, \
                    "cellule S40 sans chargement JSON explicite (risque de valeur en dur)"

    def test_expected_figures_exist_after_execution(self):
        if not FIGS.exists():
            pytest.skip("docs/figures/sprint40_article absent (exécuter le notebook)")
        present = {p.name for p in FIGS.glob("*.png")}
        missing = [f for f in EXPECTED_FIGS if f not in present]
        assert not missing, f"figures manquantes : {missing}"


# ── Cohérence de l'article LaTeX FR/EN (S4004–S4007) ──────────────────────────

import re


def _tex_sources(lang: str) -> str:
    """Concatène main_{lang}.tex + sections/{lang}/*.tex (skip si article absent)."""
    main = ARTICLE / f"main_{lang}.tex"
    if not main.exists():
        pytest.skip("docs/article/ewc_int8_mcu absent (S4004 non implémenté)")
    parts = [main.read_text(encoding="utf-8")]
    sec_dir = ARTICLE / "sections" / lang
    if sec_dir.exists():
        for p in sorted(sec_dir.glob("*.tex")):
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _decimals(text: str) -> set[str]:
    """Ensemble des littéraux décimaux (grandeurs canoniques) dans le LaTeX."""
    return set(re.findall(r"\d+\.\d+", text))


def _fmt_variants(v: float) -> set[str]:
    """Variantes d'affichage d'un flottant telles qu'elles peuvent figurer dans le .tex."""
    out: set[str] = set()
    for s in (f"{v:g}", f"{v:.3f}", f"{v:.4f}"):
        out.add(s)
    return out


def _walk_floats(obj, sink: set[str]) -> None:
    if isinstance(obj, dict):
        for x in obj.values():
            _walk_floats(x, sink)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _walk_floats(x, sink)
    elif isinstance(obj, float):
        sink |= _fmt_variants(obj)
    elif isinstance(obj, int):
        sink.add(str(obj))


def _json_grounded_values() -> set[str]:
    """Ensemble des valeurs numériques présentes dans les JSON sources (S36 + S39 ablation).

    Sert de vérité terrain pour prouver qu'un chiffre de l'article n'est pas hardcodé mais
    dérive bien d'une exécution mesurée/émulée."""
    vals: set[str] = set()
    summary = _load(EXP / "exp_S36_summary.json")
    if summary is not None:
        _walk_floats(summary, vals)
    abl_dir = EXP / "exp_S39_ablation"
    if abl_dir.exists():
        for p in abl_dir.glob("*.json"):
            _walk_floats(_load(p), vals)
    if not vals:
        pytest.skip("JSON sources S36/S39 absents (grounding impossible)")
    return vals


class TestArticleCoherence:
    def test_figures_match_json(self):
        """Chaque grandeur canonique affichée dans le .tex dérive d'un JSON (via la table
        de provenance générée), donc n'est pas un chiffre inventé/hardcodé."""
        src = _tex_sources("fr")
        grounded = _json_grounded_values()
        # Grandeurs clés qui DOIVENT apparaître dans l'article ET être adossées à un JSON source.
        canonical = {"0.9164", "0.9194", "0.138", "0.1337", "0.9462", "0.9201", "0.9616"}
        present = {c for c in canonical if c in src}
        assert present, "aucune valeur canonique trouvée dans main_fr.tex"
        ungrounded = [c for c in present if c not in grounded]
        assert not ungrounded, f"valeurs non adossées à un JSON (hardcode ?) : {ungrounded}"

    def test_notebook_structure(self):
        """Le notebook de synthèse produit/référence les 5 figures attendues."""
        if not NB.exists():
            pytest.skip("synthesis.ipynb absent")
        nb = json.loads(NB.read_text())
        src = "\n".join("".join(c.get("source", "")) for c in nb["cells"]
                        if c["cell_type"] == "code")
        for fig in EXPECTED_FIGS:
            assert fig in src, f"{fig} non référencé dans le notebook"

    def test_fr_en_key_values(self):
        """Les valeurs numériques clés sont IDENTIQUES entre FR et EN (miroir strict)."""
        fr = _decimals(_tex_sources("fr"))
        en = _decimals(_tex_sources("en"))
        only_fr = fr - en
        only_en = en - fr
        assert not only_fr and not only_en, \
            f"divergence numérique FR/EN — seulement FR={sorted(only_fr)} seulement EN={sorted(only_en)}"

    def test_board_v2_na_honest(self):
        """Les grandeurs board v2 non mesurées restent « à mesurer » (aucun chiffre inventé) :
        marqueur d'honnêteté présent dans les deux versions, et lignes board v2 vides en amont."""
        fr = _tex_sources("fr")
        en = _tex_sources("en")
        assert "à mesurer" in fr, "marqueur « à mesurer » absent de main_fr.tex"
        assert "to be measured" in en, "marqueur « to be measured » absent de main_en.tex"
        # La table de provenance ne doit contenir AUCUN chiffre pour les cellules board v2 « à mesurer ».
        if PROVENANCE.exists():
            import csv
            with PROVENANCE.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (row.get("statut") or "").strip() == "à mesurer":
                        assert not (row.get("valeur") or "").strip(), \
                            f"valeur inventée pour une cellule « à mesurer » : {row}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
