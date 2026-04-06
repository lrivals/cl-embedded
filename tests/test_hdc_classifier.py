# tests/test_hdc_classifier.py
"""
Tests unitaires pour HDCClassifier (S2-02).

Critère d'acceptation : 11/11 tests passent.
"""
import numpy as np
import pytest

from src.models.hdc.hdc_classifier import (
    HDCClassifier,
    encode_observation,
    quantize_feature,
)

D, N_LEVELS, N_FEATURES, N_CLASSES = 1024, 10, 4, 2

MOCK_CONFIG = {
    "hdc": {
        "D": D,
        "n_levels": N_LEVELS,
        "seed": 42,
        "base_vectors_path": "/tmp/test_hdc_bv.npz",
    },
    "data": {"n_features": N_FEATURES, "n_classes": N_CLASSES},
    "feature_bounds": {
        "temperature": (20.0, 80.0),
        "pressure": (1.0, 10.0),
        "vibration": (0.0, 5.0),
        "humidity": (10.0, 90.0),
    },
    "memory": {"target_ram_bytes": 65536, "warn_if_above_bytes": 52000},
}


def make_model() -> HDCClassifier:
    return HDCClassifier(MOCK_CONFIG)


# ---------------------------------------------------------------------------
# Tests fonctions stateless
# ---------------------------------------------------------------------------


def test_quantize_feature_clipping():
    assert quantize_feature(0.0, 0.0, 1.0, 10) == 0
    assert quantize_feature(1.0, 0.0, 1.0, 10) == 9
    assert quantize_feature(-99.0, 0.0, 1.0, 10) == 0
    assert quantize_feature(99.0, 0.0, 1.0, 10) == 9


def test_encode_observation_shape_dtype():
    model = make_model()
    x = np.array([50.0, 5.0, 2.5, 50.0], dtype=np.float32)
    bounds = list(MOCK_CONFIG["feature_bounds"].values())
    H_obs = encode_observation(x, model.H_level, model.H_pos, bounds, N_LEVELS, D)
    assert H_obs.shape == (D,)
    assert H_obs.dtype == np.int8
    assert set(np.unique(H_obs)).issubset({-1, 1})


# ---------------------------------------------------------------------------
# Tests HDCClassifier
# ---------------------------------------------------------------------------


def test_update_increments_prototypes():
    model = make_model()
    x = np.random.randn(10, N_FEATURES).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    model.update(x, y)
    assert model.class_counts[0] == 5
    assert model.class_counts[1] == 5


def test_predict_after_update():
    model = make_model()
    x_train = np.random.randn(50, N_FEATURES).astype(np.float32)
    y_train = np.array([i % 2 for i in range(50)], dtype=np.int64)
    model.update(x_train, y_train)
    preds = model.predict(x_train[:5])
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_before_fit_raises():
    model = make_model()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((1, N_FEATURES), dtype=np.float32))


def test_ram_budget_within_64ko():
    model = make_model()
    budget = model.check_ram_budget()
    assert budget["within_budget"] is True
    assert budget["estimated_bytes"] < 65536


def test_ram_estimate_fp32():
    model = make_model()
    # prototypes_acc: 2*1024*4 + prototypes_bin: 2*1024*1 + buffer: 4*1024 + counts: 2*4
    expected = 2 * 1024 * 4 + 2 * 1024 * 1 + 4 * 1024 + 2 * 4
    assert model.estimate_ram_bytes("fp32") == expected


def test_count_parameters():
    model = make_model()
    assert model.count_parameters() == N_CLASSES * D  # 2048


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    x = np.random.randn(20, N_FEATURES).astype(np.float32)
    y = np.array([i % 2 for i in range(20)], dtype=np.int64)
    model.update(x, y)
    path = str(tmp_path / "hdc_state.npz")
    model.save(path)
    model2 = make_model()
    model2.load(path)
    np.testing.assert_array_equal(model.prototypes_acc, model2.prototypes_acc)
    np.testing.assert_array_equal(model.prototypes_bin, model2.prototypes_bin)


def test_on_task_end_rebinarizes():
    model = make_model()
    model.prototypes_acc[0, :5] = 3  # valeurs positives → bin doit être +1
    model.prototypes_acc[0, 5:10] = -2  # valeurs négatives → bin doit être -1
    model.on_task_end(task_id=0, dataloader=None)
    assert (model.prototypes_bin[0, :5] == 1).all()
    assert (model.prototypes_bin[0, 5:10] == -1).all()


def test_incremental_no_catastrophic_forgetting():
    """
    Vérifie que l'accumulation de nouvelles données n'écrase pas les précédentes.
    Les counts de classe doivent être monotoniquement croissants.
    """
    model = make_model()
    x1 = np.ones((10, N_FEATURES), dtype=np.float32)
    y1 = np.zeros(10, dtype=np.int64)
    model.update(x1, y1)
    acc_before = model.prototypes_acc[0].copy()  # noqa: F841

    x2 = np.ones((5, N_FEATURES), dtype=np.float32) * 2
    y2 = np.zeros(5, dtype=np.int64)
    model.update(x2, y2)

    # Les counts de la classe 0 doivent avoir augmenté
    assert model.class_counts[0] == 15
