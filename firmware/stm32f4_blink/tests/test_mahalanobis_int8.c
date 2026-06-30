/**
 * test_mahalanobis_int8.c — Tests unitaires Unity pour mahalanobis_int8.c (Sprint 29, S2912).
 *
 * 1) Parité forward INT8 C ↔ Python : maha_int8_score doit reproduire la distance calculée par
 *    MahalanobisDetectorInt8.score_int8 (vecteurs de référence générés par
 *    export_weights_c.py --maha-int8-test-vectors → test_vectors_maha_int8.h).
 * 2) Propriétés de base : init neutre = distance euclidienne, prédiction au seuil.
 */

#include "unity.h"
#include "mahalanobis.h"
#include "mahalanobis_int8.h"
#include "test_vectors_maha_int8.h"
#include <math.h>
#include <string.h>

/* Tolérance : déquant INT8 (mu+sigma affine) ≠ FP32 exact côté Python, l'écart vient seulement
 * de l'arrondi float32/64 — quelques 1e-4 sur des distances ~O(1). */
#define TOL_MAHA_INT8 5e-3f

/* Charge le détecteur INT8 depuis les vecteurs de référence (mêmes poids quantifiés que Python). */
static void load_ref_detector(MahalanobisInt8 *m)
{
    maha_int8_init(m, 1.0f, 0.1f);
    m->mu_scale = TV_MAHA_INT8_MU_SCALE;
    m->mu_zp = TV_MAHA_INT8_MU_ZP;
    m->sigma_inv_scale = TV_MAHA_INT8_SIGMA_SCALE;
    m->sigma_inv_zp = TV_MAHA_INT8_SIGMA_ZP;
    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        m->mu_q8[i] = TV_MAHA_INT8_MU_Q8[i];
        for (int j = 0; j < MAHA_INT8_N_FEATURES; j++) {
            m->sigma_inv_q8[i][j] = TV_MAHA_INT8_SIGMA_INV[i][j];
        }
    }
}

/* ── Parité C ↔ Python ─────────────────────────────────────────────────────── */

void test_maha_int8_parity_with_python(void)
{
    TEST_ASSERT_EQUAL_INT(MAHA_INT8_N_FEATURES, TV_MAHA_INT8_DIM);

    MahalanobisInt8 m;
    load_ref_detector(&m);

    float x[MAHA_INT8_N_FEATURES];
    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) x[i] = TV_MAHA_INT8_INPUT[i];

    float dist = maha_int8_score(&m, x);
    TEST_ASSERT_FLOAT_WITHIN(TOL_MAHA_INT8, TV_MAHA_INT8_EXPECTED_DIST, dist);
}

/* ── Propriétés de base ─────────────────────────────────────────────────────── */

void test_maha_int8_init_is_euclidean(void)
{
    /* Init neutre : mu=0 (q8=0, zp=0, scale=1), Σ⁻¹ = identité (scale=1, zp=0) → distance L2. */
    MahalanobisInt8 m;
    maha_int8_init(&m, 10.0f, 0.1f);

    float x[MAHA_INT8_N_FEATURES];
    float ss = 0.0f;
    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        x[i] = (float)(i + 1);
        ss += x[i] * x[i];
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, sqrtf(ss), maha_int8_score(&m, x));
}

void test_maha_int8_zero_distance(void)
{
    MahalanobisInt8 m;
    maha_int8_init(&m, 1.0f, 0.1f);   /* mu = 0 */
    float x[MAHA_INT8_N_FEATURES] = {0};
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, maha_int8_score(&m, x));
}

void test_maha_int8_predict_threshold(void)
{
    MahalanobisInt8 m;
    maha_int8_init(&m, 2.0f, 0.1f);   /* seuil = 2.0, Σ⁻¹ = I */
    float below[MAHA_INT8_N_FEATURES] = {1.0f, 0, 0, 0, 0};  /* dist = 1 < 2 */
    float above[MAHA_INT8_N_FEATURES] = {3.0f, 0, 0, 0, 0};  /* dist = 3 > 2 */
    TEST_ASSERT_EQUAL_INT(0, maha_int8_predict(&m, below));
    TEST_ASSERT_EQUAL_INT(1, maha_int8_predict(&m, above));
}
