"""
tests/test_disagreement.py — Tests unitaires pour src/evaluation/disagreement_metrics.py (S3003).

Vérifie :
    - disagreement_rate exact ;
    - cohen_kappa (accord parfait → 1.0, hasard → ~0) ;
    - disagreement_confusion (partition : a_correct + b_correct + both_wrong == n_disagree) ;
    - per_sample_disagreement_mask ;
    - analyze_disagreement_origin détecte la feature discriminante injectée.

Exécution :
    pytest tests/test_disagreement.py -v
"""

from __future__ import annotations

import numpy as np

from src.evaluation.disagreement_metrics import (
    analyze_disagreement_origin,
    cohen_kappa,
    disagreement_confusion,
    disagreement_rate,
    per_sample_disagreement_mask,
)


def test_disagreement_rate_exact():
    y_a = np.array([0, 0, 1, 1])
    y_b = np.array([0, 1, 0, 1])
    assert disagreement_rate(y_a, y_b) == 0.5


def test_disagreement_rate_empty():
    assert disagreement_rate(np.array([]), np.array([])) == 0.0


def test_per_sample_mask():
    y_a = np.array([0, 1, 1, 0])
    y_b = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(
        per_sample_disagreement_mask(y_a, y_b), [False, True, False, True]
    )


def test_cohen_kappa_perfect_agreement():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert cohen_kappa(y, y) == 1.0


def test_cohen_kappa_constant_identical():
    """Cas dégénéré : identiques et constants → kappa nan ramené à 1.0."""
    y = np.zeros(5, dtype=int)
    assert cohen_kappa(y, y) == 1.0


def test_cohen_kappa_near_zero_for_independent():
    rng = np.random.default_rng(0)
    y_a = rng.integers(0, 2, size=2000)
    y_b = rng.integers(0, 2, size=2000)
    assert abs(cohen_kappa(y_a, y_b)) < 0.1


def test_disagreement_confusion_partition():
    y_true = np.array([0, 1, 0, 1, 1])
    y_a = np.array([0, 1, 1, 0, 1])  # diverge sur idx 2, 3
    y_b = np.array([1, 1, 0, 1, 1])  # diverge sur idx 0, 2, 3
    conf = disagreement_confusion(y_a, y_b, y_true)
    # Désaccords aux index 0, 2, 3.
    assert conf["n_disagree"] == 3
    assert conf["a_correct"] + conf["b_correct"] + conf["both_wrong"] == conf["n_disagree"]
    # idx0: a=0✓ b=1✗ → a_correct ; idx2: a=1✗ b=0✓ → b_correct ; idx3: a=0✗ b=1✓ → b_correct
    assert conf["a_correct"] == 1
    assert conf["b_correct"] == 2
    assert conf["both_wrong"] == 0


def test_analyze_origin_detects_discriminative_feature():
    """Feature 1 décale fortement sur le sous-ensemble en désaccord → top feature = 1."""
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(0, 1, size=(n, 3))
    mask = np.zeros(n, dtype=bool)
    mask[:50] = True
    X[mask, 1] += 10.0  # injecte un décalage net sur la feature 1
    y_true = rng.integers(0, 2, size=n)

    res = analyze_disagreement_origin(X, mask, y_true)
    assert res["n_disagree"] == 50
    assert res["top_features"][0] == 1
    assert res["feature_deltas"][1] > res["feature_deltas"][0]
    assert res["feature_deltas"][1] > res["feature_deltas"][2]


def test_analyze_origin_with_maha_and_boundary():
    rng = np.random.default_rng(2)
    n = 100
    X = rng.normal(0, 1, size=(n, 4))
    mask = np.zeros(n, dtype=bool)
    mask[:30] = True
    y_true = rng.integers(0, 2, size=n)
    maha = rng.normal(5, 1, size=n)
    maha[mask] += 3.0  # désaccord sur scores plus élevés
    boundary = rng.uniform(0, 1, size=n)
    boundary[mask] = 0.5  # désaccord pile sur la frontière

    res = analyze_disagreement_origin(X, mask, y_true, maha_scores=maha, boundary_scores=boundary)
    assert res["maha_score_in"] > res["maha_score_out"]
    # Proximité de frontière : distance ~0 dans le masque, > 0 hors masque.
    assert res["boundary_dist_in"] < res["boundary_dist_out"]


def test_analyze_origin_degenerate_mask_warns():
    X = np.random.default_rng(3).normal(0, 1, size=(50, 3))
    mask = np.zeros(50, dtype=bool)  # aucun désaccord
    res = analyze_disagreement_origin(X, mask, np.zeros(50))
    assert res["n_disagree"] == 0
    assert res["top_features"] == []
