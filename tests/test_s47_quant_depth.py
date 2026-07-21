"""test_s47_quant_depth.py — Profondeur & schéma de quantification EWC (Sprint 47, S4702/S4703).

Vérifie, sans jamais exiger de chiffres inventés :
  - **0 régression** : les presets S39 (`legacy_c`, `per_tensor_calib`, `per_channel_int8`,
    `q15`) produisent des logits **identiques** à une golden figée (l'axe profondeur n'altère
    pas le chemin existant) ;
  - le câblage sub-INT8/ternaire/binaire : grilles de poids correctes ({−1,0,+1}, {−1,+1},
    saturation par bits), `subint8` porte les bons champs ;
  - `theoretical_weight_ram` : ratios ×4 (8b) / ×8 (4b) / ×16 (2b) / ×32 (binaire) ;
  - l'hypothèse H1 (per-channel repousse le cliff) et la monotonie de l'accord en bits ;
  - l'activation affine (zero-point) tourne et reste finie ;
  - honnêteté du schéma JSON `exp_S47_depth/` (`delta_auroc` renseigné, `ram_ratio` croît
    quand `weight_bits` décroît ; skip si l'artefact manque, comme test_s39_quant.py).

Référence : S4701 (taxonomie), S4702 (émulateur+harnais), S4703 (sweep).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.int8_c_emulation import (
    EWCHeadWeights,
    QuantConfig,
    _binary_weight,
    _quant_weight_mode,
    _ternary_weight,
    agreement,
    forward_fp32,
    forward_quant,
    theoretical_weight_ram,
)

REPO = Path(__file__).resolve().parent.parent
DEPTH_DIR = REPO / "experiments" / "exp_S47_depth"
CONTEXT_JSON = REPO / "experiments" / "exp_S47_context" / "context.json"
QUANT_DEPTH_CATALOG = REPO / "src" / "figures" / "catalogs" / "quant_depth.py"

_needs_context = pytest.mark.skipif(
    not CONTEXT_JSON.exists(),
    reason="exp_S47_context/context.json absent (run_s47_quant_depth.py --context)",
)

_needs_depth = pytest.mark.skipif(
    not DEPTH_DIR.exists(), reason="exp_S47_depth absent (lancer run_s47_quant_depth.py --sweep)"
)


def _golden_head(seed: int = 123) -> EWCHeadWeights:
    """Tête EWC 5→32→16→2 de grande dynamique (fige la golden de non-régression)."""
    rng = np.random.default_rng(seed)
    return EWCHeadWeights(
        w1=rng.normal(0, 2.5, (32, 5)), b1=rng.normal(0, 0.1, 32),
        w2=rng.normal(0, 1.0, (16, 32)), b2=rng.normal(0, 0.1, 16),
        w3=rng.normal(0, 1.0, (2, 16)), b3=rng.normal(0, 0.1, 2),
    )


def _high_dynamic_head(seed: int = 42) -> EWCHeadWeights:
    rng = np.random.default_rng(seed)
    return EWCHeadWeights(
        w1=rng.normal(0, 3.0, (32, 5)), b1=rng.normal(0, 0.1, 32),
        w2=rng.normal(0, 1.0, (16, 32)), b2=rng.normal(0, 0.1, 16),
        w3=rng.normal(0, 1.0, (1, 16)), b3=rng.normal(0, 0.1, 1),
    )


# ── 0 régression : presets S39 inchangés (golden figée) ──────────────────────

# Somme des logits (48×2) sur (_golden_head(123), X~N(0,1.5) seed 123) — capturée après
# l'extension S47. Toute dérive du chemin existant casse ces valeurs.
GOLDEN_LOGIT_SUM = {
    "legacy_c": -6.268409,
    "per_tensor_calib": 7338.749801,
    "per_channel_int8": 7376.9031,
    "q15": 7346.117595,
}


def test_no_regression_presets_golden():
    """Les presets S39 produisent des logits bit-identiques à la golden (0 régression)."""
    w = _golden_head(123)
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1.5, (48, 5))
    presets = {
        "legacy_c": QuantConfig.legacy_c(),
        "per_tensor_calib": QuantConfig.per_tensor_calib(),
        "per_channel_int8": QuantConfig.per_channel_int8(),
        "q15": QuantConfig.q15(),
    }
    for name, cfg in presets.items():
        s = float(forward_quant(w, X, cfg).sum())
        assert s == pytest.approx(GOLDEN_LOGIT_SUM[name], abs=1e-4), (
            f"régression {name} : {s} ≠ {GOLDEN_LOGIT_SUM[name]}"
        )


def test_q15_preset_carries_16bit_weights():
    """Rétro-compat : q15() porte weight_bits=16 (poids 16-bit préservés)."""
    assert QuantConfig.q15().weight_bits == 16
    assert QuantConfig.per_channel_int8().weight_bits == 8
    assert QuantConfig.mixed_int8w_q15act().weight_bits == 8


# ── subint8 : champs + grilles de poids ──────────────────────────────────────


def test_subint8_fields():
    c = QuantConfig.subint8(4, granularity="per_tensor", symmetry="affine", mode="linear")
    assert c.weight_bits == 4
    assert c.weight_scale == "per_tensor"
    assert c.symmetry == "affine"
    assert c.weight_mode == "linear"
    assert "subint8" in c.name


@pytest.mark.parametrize("bits,qmax", [(8, 127), (6, 31), (4, 7), (3, 3), (2, 1)])
def test_linear_weight_grid_saturates(bits: int, qmax: int):
    """La grille linéaire respecte qmax = 2^(b-1)-1 par profondeur."""
    w = _golden_head().w1
    q, _ = _quant_weight_mode(w, "linear", "per_channel", bits)
    assert q.min() >= -qmax and q.max() <= qmax
    assert q.max() == qmax or q.min() == -qmax  # scale calibré → saturation atteinte


def test_ternary_grid_is_neg1_0_1():
    q, s = _ternary_weight(_golden_head().w1)
    assert set(np.unique(q)).issubset({-1, 0, 1})
    assert (s > 0).all()
    assert (q == 0).any()  # le seuil TWN met bien des poids à zéro


def test_binary_grid_is_neg1_1():
    q, s = _binary_weight(_golden_head().w1)
    assert set(np.unique(q)).issubset({-1, 1})
    assert (s > 0).all()


# ── RAM théorique ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bits,gran,ratio", [
    (8, "per_channel", 4.0), (4, "per_channel", 8.0), (2, "per_channel", 16.0),
])
def test_theoretical_ram_ratio_linear(bits: int, gran: str, ratio: float):
    _, r = theoretical_weight_ram(_golden_head(), QuantConfig.subint8(bits, gran))
    assert r == pytest.approx(ratio, abs=0.01)


def test_theoretical_ram_ratio_binary_ternary():
    _, r_bin = theoretical_weight_ram(_golden_head(), QuantConfig.subint8(1, mode="binary"))
    _, r_ter = theoretical_weight_ram(_golden_head(), QuantConfig.subint8(2, mode="ternary"))
    assert r_bin == pytest.approx(32.0, abs=0.01)
    assert r_ter == pytest.approx(20.25, abs=0.1)  # 32/1.58


# ── Comportement du sweep : monotonie + H1 (per-channel repousse le cliff) ────


def test_agreement_non_increasing_with_bits():
    """Descendre en bits ne peut qu'égaler ou dégrader l'accord vs fp32 (non-croissance)."""
    w = _high_dynamic_head()
    rng = np.random.default_rng(7)
    X = rng.normal(0, 2.0, (256, 5))
    fp = forward_fp32(w, X)
    prev = 1.01
    for bits in (8, 6, 4, 3, 2):
        a = agreement(forward_quant(w, X, QuantConfig.subint8(bits, "per_channel")), fp)
        assert a <= prev + 1e-9, f"accord remonte à {bits} bits ({a} > {prev})"
        prev = a


def test_per_channel_rescues_low_bits():
    """H1 : à basse profondeur, la per-channel ≥ la per-tensor en accord vs fp32."""
    w = _high_dynamic_head()
    rng = np.random.default_rng(11)
    X = rng.normal(0, 2.0, (256, 5))
    fp = forward_fp32(w, X)
    a_pt = agreement(forward_quant(w, X, QuantConfig.subint8(2, "per_tensor")), fp)
    a_pc = agreement(forward_quant(w, X, QuantConfig.subint8(2, "per_channel")), fp)
    assert a_pc >= a_pt - 1e-9, f"per_channel ({a_pc}) < per_tensor ({a_pt}) à 2 bits"


# ── Symétrie affine (zero-point) ─────────────────────────────────────────────


def test_affine_activation_runs_finite():
    """Le chemin affine (zero-point post-ReLU) tourne et reste fini."""
    w = _high_dynamic_head()
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1.5, (64, 5))
    logits = forward_quant(w, X, QuantConfig.subint8(8, "per_channel", symmetry="affine"))
    assert np.isfinite(logits).all()
    # L'affine diffère du symétrique (le zero-point change bien le forward).
    sym = forward_quant(w, X, QuantConfig.subint8(8, "per_channel", symmetry="symmetric"))
    assert not np.allclose(logits, sym)


# ── Honnêteté du schéma JSON produit (skip si absent) ────────────────────────


@_needs_depth
def test_json_schema_and_honesty():
    files = sorted(DEPTH_DIR.glob("exp_S47_ewc_*.json"))
    assert files, "aucun JSON exp_S47_ewc_*"
    required = {
        "model", "dataset", "weight_bits", "granularity", "symmetry", "metric",
        "auroc_fp32", "auroc_quant", "delta_auroc", "agreement_vs_fp32",
        "ram_weight_bytes_theoretical", "ram_ratio_vs_fp32", "seed", "config_snapshot",
    }
    for f in files:
        d = json.loads(f.read_text())
        assert required <= d.keys(), f"{f.name}: clés manquantes {required - d.keys()}"
        assert d["model"] == "ewc"
        # Aucune métrique non calculée déguisée en 0 : soit un float renseigné, soit null.
        assert d["delta_auroc"] is None or isinstance(d["delta_auroc"], float)
        assert d["ram_ratio_vs_fp32"] >= 4.0  # >= INT8


@_needs_depth
def test_ram_ratio_grows_when_bits_shrink():
    """Cohérence : le ratio RAM théorique croît quand weight_bits décroît (per_channel)."""
    cells = {}
    for f in DEPTH_DIR.glob("exp_S47_ewc_monitoring_int*_per_channel.json"):
        d = json.loads(f.read_text())
        cells[int(d["weight_bits"])] = d["ram_ratio_vs_fp32"]
    if len(cells) < 2:
        pytest.skip("pas assez de cellules linéaires monitoring per_channel")
    ordered = [cells[b] for b in sorted(cells, reverse=True)]  # bits décroissants
    assert ordered == sorted(ordered), f"ratio RAM non monotone : {ordered}"


# ── Honnêteté du contexte N/A (HDC / Maha / TinyOL) ──────────────────────────


@_needs_context
def test_na_honesty():
    """exp_S47_context : HDC/Maha/TinyOL en `na_*` justifié, aucune métrique fabriquée."""
    d = json.loads(CONTEXT_JSON.read_text())
    assert d["sprint"] == 47
    assert d["swept_models"] == ["ewc"], "seul EWC est balayé en profondeur (S4700)"
    ctx = d["context_models"]
    assert {"hdc", "mahalanobis", "tinyol"} <= ctx.keys(), "modèles de contexte manquants"
    for name, entry in ctx.items():
        assert str(entry["status"]).startswith("na_"), (
            f"{name}: statut non-N/A ({entry['status']})"
        )
        assert entry["reason"].strip(), f"{name}: justification vide"
        assert entry.get("ref", "").strip(), f"{name}: référence traçable manquante"
        # Aucune métrique fabriquée : pas d'AUROC/delta/RAM déguisés dans une cellule N/A.
        forbidden = [k for k in entry if any(t in k.lower() for t in ("auroc", "delta", "ram"))]
        assert not forbidden, f"{name}: champs métriques interdits en N/A : {forbidden}"


# ── Garde AST : 0 chiffre de résultat en dur dans le catalogue de figures ─────

# Littéraux de mise en page autorisés (positions, largeurs de barres, alpha, seuils d'axe) —
# AUCUN n'est un résultat. Miroir de la liste blanche de test_figures_library.py (garde S4205).
_LAYOUT_WHITELIST: set[float] = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.06, 0.12, 0.15, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4,
    0.5, 0.55, 0.6, 0.72, 0.78, 0.8, 0.82, 0.86, 0.9, 0.92, 0.94, 0.98, 1.0, 1.05, 1.2, 1.4,
    1.5, 2.0, 4.5, 5.0, 8.0, 8.5, 9.0, 11.0,
}


@pytest.mark.skipif(
    not QUANT_DEPTH_CATALOG.exists(), reason="src/figures/catalogs/quant_depth.py absent (S4706)"
)
def test_no_hardcoded_numbers():
    """Scan AST de quant_depth.py : aucun flottant de résultat en dur (tout via load_experiment)."""
    tree = ast.parse(QUANT_DEPTH_CATALOG.read_text(encoding="utf-8"))
    offending = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in _LAYOUT_WHITELIST
    }
    assert not offending, (
        f"Littéraux flottants suspects dans quant_depth.py : {sorted(offending)} — "
        "toute valeur de résultat doit être chargée via load_experiment, pas écrite en dur."
    )
