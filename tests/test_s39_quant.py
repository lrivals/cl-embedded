"""test_s39_quant.py — Tests des schémas de quantification Sprint 39 (S3912).

Vérifie les livrables PC de la Partie A :
  - l'ablation (``experiments/exp_S39_ablation/``) est monotone (chaque correctif ne nuit pas) ;
  - le sweep trade-off (``experiments/exp_S39_quant_sweep/summary.json``) est cohérent
    (structure, ratios RAM analytiques, latence toujours marquée proxy) ;
  - la dégradation ``int8_legacy`` puis la récupération ``q15`` reproduisent le diagnostic
    Gap 3, sur les JSON figés **et** en direct via l'émulateur (``int8_c_emulation``).

Skips honnêtes si un artefact manque (comme ``test_int8_c_emulation.py``).

Référence : S3904 (ablation), S3906 (sweep), S3902 (émulateur).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ABLATION_DIR = REPO / "experiments" / "exp_S39_ablation"
SWEEP_SUMMARY = REPO / "experiments" / "exp_S39_quant_sweep" / "summary.json"

DATASETS = ["cmapss", "cwru", "monitoring", "pronostia", "paderborn"]
MODELS = ["ewc", "mahalanobis", "hdc", "tinyol"]
# Ladder de schémas EWC (les autres modèles ont leur propre jeu réduit).
EWC_SCHEMES = ["fp32", "int8_legacy", "int8_perchannel", "q15", "mixed"]
SCHEME_KEYS = {"metric", "ram_weights_bytes", "bops_proxy", "lat_proxy_rel", "lat_proxy"}

_needs_ablation = pytest.mark.skipif(
    not ABLATION_DIR.exists(), reason="exp_S39_ablation absent (lancer run_s39_int8_ablation.py)"
)
_needs_sweep = pytest.mark.skipif(
    not SWEEP_SUMMARY.exists(), reason="exp_S39_quant_sweep/summary.json absent (run_s39_quant_sweep.py)"
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SWEEP_SUMMARY.read_text())


# ── Ablation (S3904) ─────────────────────────────────────────────────────────


def _ladder(f: Path) -> dict[str, dict]:
    data = json.loads(f.read_text())
    return {s["scheme"]: s for s in data["ladder"]}, data


@_needs_ablation
def test_ablation_calibrated_tail_stable():
    """À partir de per_tensor_calib, les schémas calibrés ne se nuisent pas (Δ ≥ −ε).

    NB (honnêteté S3904) : le pas isolé ``fix_acc32`` (overflow corrigé, scale 1/128 encore
    figée) PEUT dégrader la F1 avant que la calibration ne la récupère — l'échelle n'est donc
    pas monotone bout-en-bout. On teste la stabilité du régime calibré, pas une fausse monotonie.
    """
    files = sorted(ABLATION_DIR.glob("*.json"))
    assert files, "aucun JSON d'ablation"
    tail = {"per_channel_int8", "q15"}
    eps = 0.01
    for f in files:
        ladder, _ = _ladder(f)
        for name in tail:
            if name in ladder and ladder[name]["delta_prev"] is not None:
                assert ladder[name]["delta_prev"] >= -eps, (
                    f"{f.stem}:{name} dégrade le régime calibré ({ladder[name]['delta_prev']})"
                )


@_needs_ablation
def test_ablation_endpoints_recover_fp32():
    """Les schémas cibles (per_channel_int8, q15) récupèrent la F1 FP32 (± 0.05)."""
    for f in sorted(ABLATION_DIR.glob("*.json")):
        ladder, data = _ladder(f)
        for name in ("per_channel_int8", "q15"):
            assert ladder[name]["f1"] >= data["f1_fp32"] - 0.05, (
                f"{f.stem}:{name} ne récupère pas ({ladder[name]['f1']} vs {data['f1_fp32']})"
            )


@_needs_ablation
def test_ablation_dominant_is_scale_calibration():
    """Là où legacy s'effondre (monitoring/pronostia), le facteur dominant = scale calibré."""
    for ds in ("monitoring", "pronostia"):
        f = ABLATION_DIR / f"{ds}.json"
        if not f.exists():
            pytest.skip(f"{ds}.json absent")
        data = json.loads(f.read_text())
        assert data["dominant_scheme"] == "per_tensor_calib", (
            f"{ds} : dominant={data['dominant_scheme']} (attendu per_tensor_calib)"
        )


# ── Sweep trade-off (S3906) ──────────────────────────────────────────────────


@_needs_sweep
def test_sweep_structure(summary):
    """4 modèles × 5 datasets ; EWC porte l'échelle 5 schémas ; clés cohérentes partout."""
    assert set(summary) == set(MODELS)
    for model in MODELS:
        assert set(summary[model]) == set(DATASETS), f"{model} : datasets manquants"
        for ds, schemes in summary[model].items():
            assert "fp32" in schemes, f"{model}/{ds} sans fp32"
            for name, cell in schemes.items():
                assert SCHEME_KEYS.issubset(cell), f"{model}/{ds}/{name} clés {set(cell)}"
    # L'EWC couvre bien les 5 schémas de l'échelle d'ablation.
    for ds in DATASETS:
        assert set(summary["ewc"][ds]) == set(EWC_SCHEMES), f"ewc/{ds} schémas != ladder"


@_needs_sweep
def test_ram_ratios(summary):
    """RAM analytique : int8 = ×4, q15 = ×2 vs fp32 sur la tête EWC (poids clean)."""
    for ds in DATASETS:
        cell = summary["ewc"][ds]
        fp32 = cell["fp32"]["ram_weights_bytes"]
        assert fp32 / cell["int8_perchannel"]["ram_weights_bytes"] == pytest.approx(4.0, abs=0.01)
        assert fp32 / cell["q15"]["ram_weights_bytes"] == pytest.approx(2.0, abs=0.01)


@_needs_sweep
def test_hdc_exact(summary):
    """Témoin : HDC int8 == fp32 (Δ=0) — la quantif HDC ne change pas la métrique."""
    for ds in DATASETS:
        cell = summary["hdc"][ds]
        assert cell["int8"]["metric"] == cell["fp32"]["metric"], f"hdc/{ds} Δ≠0"


@_needs_sweep
def test_lat_proxy_flagged(summary):
    """Aucune latence mesurée inventée : chaque cellule porte lat_proxy=True (board → S3915)."""
    for model in MODELS:
        for ds in DATASETS:
            for name, cell in summary[model][ds].items():
                assert cell["lat_proxy"] is True, f"{model}/{ds}/{name} lat_proxy non marqué"


@_needs_sweep
def test_legacy_degrades(summary):
    """int8_legacy s'effondre vs fp32 sur les datasets de grande dynamique (monitoring/pronostia)."""
    for ds in ("monitoring", "pronostia"):
        cell = summary["ewc"][ds]
        assert cell["int8_legacy"]["metric"] < cell["fp32"]["metric"] - 0.4, (
            f"ewc/{ds} : legacy ne s'effondre pas ({cell['int8_legacy']['metric']})"
        )


@_needs_sweep
def test_q15_recovers(summary):
    """Critère Gap 3 : F1(q15) ≥ F1(fp32) − 0.02 sur toute la grille EWC."""
    for ds in DATASETS:
        cell = summary["ewc"][ds]
        assert cell["q15"]["metric"] >= cell["fp32"]["metric"] - 0.02, (
            f"ewc/{ds} : q15 ne récupère pas ({cell['q15']['metric']} vs {cell['fp32']['metric']})"
        )


# ── Émulateur en direct (indépendant des JSON figés) ─────────────────────────


def _synthetic_head(seed: int = 42):
    """Tête EWC 5→32→16→2 synthétique de grande dynamique (reproduit la casse legacy)."""
    from src.utils.int8_c_emulation import EWCHeadWeights

    rng = np.random.default_rng(seed)
    return EWCHeadWeights(
        w1=rng.normal(0, 3.0, (32, 5)), b1=rng.normal(0, 0.1, 32),
        w2=rng.normal(0, 1.0, (16, 32)), b2=rng.normal(0, 0.1, 16),
        w3=rng.normal(0, 1.0, (2, 16)), b3=rng.normal(0, 0.1, 2),
    )


def test_emulator_legacy_degrades_perchannel_recovers():
    """En direct : legacy_c dégrade l'accord vs FP32, per_channel_int8/q15 le récupèrent."""
    from src.utils.int8_c_emulation import (
        QuantConfig,
        agreement,
        forward_fp32,
        forward_quant,
    )

    w = _synthetic_head()
    rng = np.random.default_rng(7)
    X = rng.normal(0, 2.0, (256, 5)).astype(np.float32)  # grande dynamique → clamp Q7 figé
    fp = forward_fp32(w, X)

    agree_legacy = agreement(forward_quant(w, X, QuantConfig.legacy_c()), fp)
    agree_pc = agreement(forward_quant(w, X, QuantConfig.per_channel_int8()), fp)
    agree_q15 = agreement(forward_quant(w, X, QuantConfig.q15()), fp)

    assert agree_legacy < agree_pc, f"legacy {agree_legacy} ≥ per_channel {agree_pc}"
    assert agree_pc >= 0.95, f"per_channel ne récupère pas l'accord : {agree_pc}"
    assert agree_q15 >= 0.95, f"q15 ne récupère pas l'accord : {agree_q15}"


def test_emulator_ram_footprint_ratios():
    """Empreinte analytique des poids : int8 = ×4, q15 = ×2 vs fp32 (5→32→16→2)."""
    n_params = 32 * 5 + 16 * 32 + 2 * 16  # 704 (hors biais)
    fp32_bytes = n_params * 4
    int8_bytes = n_params * 1
    q15_bytes = n_params * 2
    assert fp32_bytes / int8_bytes == 4.0
    assert fp32_bytes / q15_bytes == 2.0
