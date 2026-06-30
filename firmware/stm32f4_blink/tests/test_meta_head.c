/**
 * test_meta_head.c — Tests unitaires Unity pour meta_head.c (Sprint 31 / S3112)
 *
 * Méta-modèle de stacking : META_N_FEATURES entrées → proba sigmoïde ∈ [0, 1].
 * Tests :
 *   - meta_init charge les poids générés (meta_weights.h)
 *   - meta_forward parité C ↔ Python sur vecteurs de référence (test_vectors_meta.h)
 *   - meta_predict retourne {0, 1} et est cohérent avec le seuil 0.5
 */

#include "unity.h"
#include "meta_head.h"
#include "test_vectors_meta.h"
#include <math.h>

#define META_TOL 1e-5f

/* ── meta_init ──────────────────────────────────────────────────────────── */

void test_meta_init_loads_weights(void)
{
    MetaHead m;
    meta_init(&m);
#if META_HIDDEN == 0
    for (int i = 0; i < META_N_FEATURES; i++) {
        TEST_ASSERT_FLOAT_WITHIN(META_TOL, META_W[i], m.w[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(META_TOL, META_B, m.b);
#else
    TEST_ASSERT_FLOAT_WITHIN(META_TOL, META_B2, m.b2);
#endif
}

/* ── meta_forward : parité C ↔ Python ───────────────────────────────────── */

void test_meta_forward_parity_python(void)
{
    MetaHead m;
    meta_init(&m);
    for (int c = 0; c < TV_META_N_CASES; c++) {
        float p = meta_forward(&m, TV_META_INPUT[c]);
        TEST_ASSERT_FLOAT_WITHIN(META_TOL, TV_META_EXPECTED[c], p);
    }
}

void test_meta_forward_in_unit_range(void)
{
    MetaHead m;
    meta_init(&m);
    for (int c = 0; c < TV_META_N_CASES; c++) {
        float p = meta_forward(&m, TV_META_INPUT[c]);
        TEST_ASSERT_TRUE(p >= 0.0f && p <= 1.0f);
    }
}

/* ── meta_predict : binaire cohérent avec le seuil 0.5 ──────────────────── */

void test_meta_predict_binary(void)
{
    MetaHead m;
    meta_init(&m);
    for (int c = 0; c < TV_META_N_CASES; c++) {
        int pred = meta_predict(&m, TV_META_INPUT[c]);
        TEST_ASSERT_TRUE(pred == 0 || pred == 1);
        int expected = TV_META_EXPECTED[c] > 0.5f ? 1 : 0;
        TEST_ASSERT_EQUAL_INT(expected, pred);
    }
}
