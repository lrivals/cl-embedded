"""
tests/test_mahalanobis_q15.py — Q15 Mahalanobis fallback (Sprint 34, S3409).

Verrouille le mode ``quant="q15"`` de MahalanobisDetectorInt8 (S3405) :
    - recouvrement de la fidélité au FP32 sur matrices à grande dynamique (cible du bug INT8),
    - non-régression du mode ``int8`` par défaut (résultats strictement identiques à avant),
    - cohérence de l'empreinte mémoire Q15 (sigma_inv int16 → ÷2 vs FP32).

Les tests synthétiques reproduisent le régime « grande dynamique » (sigma_inv mêlant valeurs
~1e5 et ~1e0) sans dépendre des datasets bruts (data/ gitignore). Un test optionnel valide
l'agrégat exp_S34_maha_q15/summary.json quand il a été produit (run_s34_maha_q15.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8

_ROOT = Path(__file__).resolve().parent.parent
_SUMMARY = _ROOT / "experiments" / "exp_S34_maha_q15" / "summary.json"


def _high_dynamic_data(seed: int = 34, n: int = 600, d: int = 6) -> np.ndarray:
    """Données dont la covariance (et donc sigma_inv) a une grande dynamique réaliste.

    Des échelles de feature contrastées (1 … 100, régime ~Pronostia) produisent une matrice
    de précision mêlant des valeurs faibles et fortes — le régime qui casse l'INT8 affine
    global (corr ≈ 0.67) tandis que le Q15 restaure la fidélité (corr > 0.99).
    """
    rng = np.random.default_rng(seed)
    scales = np.array([1.0, 3.0, 10.0, 30.0, 60.0, 100.0][:d], dtype=np.float32)
    return (rng.standard_normal((n, d)).astype(np.float32) * scales)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@pytest.fixture(scope="module")
def fitted():
    """FP32 (référence) + détecteurs INT8 et Q15 calibrés sur les mêmes données/seuil."""
    X = _high_dynamic_data()
    cfg = {"threshold_percentile": 95}

    m_int8 = MahalanobisDetectorInt8({**cfg, "quantization": "int8"})
    m_int8.fit(X, task_id=0)

    m_q15 = MahalanobisDetectorInt8({**cfg, "quantization": "q15"})
    m_q15.fit(X, task_id=0)

    fp32 = m_int8.anomaly_score(X)        # FP32 de référence (hérité de MahalanobisDetector)
    s_int8 = m_int8.anomaly_score_int8(X)
    s_q15 = m_q15.anomaly_score_q15(X)
    return X, fp32, s_int8, s_q15, m_int8, m_q15


def test_q15_recovers_fidelity_on_high_dynamic_range(fitted):
    """Sur matrice à grande dynamique, Q15 corrèle nettement mieux au FP32 que l'INT8.

    C'est le recouvrement ciblé par le TODO(arnaud) : l'INT8 (8 bits) casse la distance,
    le Q15 (16 bits) restaure la fidélité de rang (→ AUROC/seuil), cf. S3406 (Pronostia
    ΔAUROC −0.113 → +0.013).
    """
    _, fp32, s_int8, s_q15, _, _ = fitted
    corr_int8 = _corr(s_int8, fp32)
    corr_q15 = _corr(s_q15, fp32)
    assert corr_q15 > corr_int8 + 0.2, (corr_q15, corr_int8)  # amélioration nette
    assert corr_q15 > 0.99  # Q15 ≈ FP32 sur le rang (INT8 ≈ 0.67 ici)


def test_q15_reconstructs_sigma_inv_finer_than_int8(fitted):
    """La matrice sigma_inv est reconstruite plus finement en Q15 qu'en INT8 (16 vs 8 bits)."""
    from src.models.unsupervised.mahalanobis_int8 import (
        _dequantize_affine_int8,
        _dequantize_q15,
    )

    _, _, _, _, m_int8, m_q15 = fitted
    sig = np.asarray(m_q15.sigma_inv_, dtype=np.float32)
    rec_int8 = _dequantize_affine_int8(m_int8.sigma_inv_q_, m_int8._sigma_scale, m_int8._sigma_zp)
    rec_q15 = _dequantize_q15(m_q15.sigma_inv_q15_, m_q15._sigma_q15_scale)
    err_int8 = float(np.max(np.abs(sig - rec_int8)))
    err_q15 = float(np.max(np.abs(sig - rec_q15)))
    assert err_q15 < err_int8


def test_q15_memory_footprint_halves_sigma_vs_fp32(fitted):
    """sigma_inv Q15 = d² × 2 B (÷2 vs FP32 d² × 4 B, au lieu de ÷4 en INT8)."""
    _, _, _, _, _, m_q15 = fitted
    d = int(m_q15.mu_.shape[0])
    fp = m_q15.get_memory_footprint_q15()
    assert fp["sigma_inv_bytes"] == d * d * 2
    assert fp["mu_bytes"] == d * 1
    # ÷2 vs FP32 sur sigma_inv exactement.
    assert fp["sigma_inv_bytes"] * 2 == d * d * 4


def test_int8_mode_unchanged(fitted):
    """quant="int8" (défaut) produit exactement le résultat de calibrate_int8() historique."""
    X, _, s_int8, _, _, _ = fitted

    # Référence : détecteur identique calibré explicitement en INT8 (chemin pré-Sprint 34).
    ref = MahalanobisDetectorInt8({"threshold_percentile": 95})  # défaut quant="int8"
    ref.fit_task(X, task_id=0)
    ref.calibrate_int8()
    s_ref = ref.anomaly_score_int8(X)

    np.testing.assert_array_equal(s_int8, s_ref)
    # Le défaut est bien int8, pas q15.
    assert MahalanobisDetectorInt8({"threshold_percentile": 95}).quant == "int8"


def test_q15_score_single_sample_matches_batch(fitted):
    """score_q15(x) == anomaly_score_q15([x])[0] (cohérence API unitaire/batch)."""
    X, _, _, _, _, m_q15 = fitted
    x = X[0]
    assert m_q15.score_q15(x) == pytest.approx(float(m_q15.anomaly_score_q15(x[None, :])[0]))


def test_q15_invalid_quant_rejected():
    with pytest.raises(ValueError):
        MahalanobisDetectorInt8({"quantization": "int4"})


@pytest.mark.skipif(not _SUMMARY.exists(), reason="run_s34_maha_q15.py non exécuté")
def test_experiment_summary_q15_recovers_targets():
    """Valide l'agrégat S3406 : Q15 ≥ INT8 en fidélité de rang sur tous les datasets,
    et recouvre l'AUROC sur Pronostia (cible à AUROC non-dégénérée)."""
    summary = json.loads(_SUMMARY.read_text())
    by_ds = {d["dataset"]: d for d in summary["datasets"]}

    for ds, rec in by_ds.items():
        if rec.get("q15_rank_fidelity_better_than_int8") is not None:
            assert rec["q15_rank_fidelity_better_than_int8"], ds

    if "pronostia" in by_ds:
        assert by_ds["pronostia"]["q15_recovers_auroc"] is True
