/**
 * test_mahalanobis_q15.c — Tests unitaires Unity pour mahalanobis_q15.c (Sprint 34, S3409).
 *
 * 1) Parité forward Q15 C ↔ Python : maha_q15_score doit reproduire la distance calculée par
 *    MahalanobisDetectorInt8.score_q15 (vecteurs de référence générés par
 *    export_weights_c.py --maha-q15-test-vectors → test_vectors_q15.h).
 * 2) Propriétés de base : init neutre = distance euclidienne, prédiction au seuil.
 */

#include "unity.h"
#include "mahalanobis.h"
#include "mahalanobis_q15.h"
#include "test_vectors_q15.h"
#include <math.h>
#include <string.h>

/* Tolérance : déquant Q15 (mu UINT8 + sigma int16) ≠ FP32 exact côté Python, mais l'écart
 * vient seulement de l'arrondi float32/64 — quelques 1e-4 sur des distances ~O(1). */
#define TOL_Q15 5e-3f

/* Charge le détecteur Q15 depuis les vecteurs de référence (mêmes poids quantifiés que Python). */
static void load_ref_detector(MahalanobisQ15 *m)
{
    maha_q15_init(m, 1.0f, 0.1f);
    m->mu_scale = TV_Q15_MU_SCALE;
    m->mu_zp = TV_Q15_MU_ZP;
    m->sigma_inv_scale = TV_Q15_SIGMA_SCALE;
    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        m->mu_q8[i] = TV_Q15_MU_Q8[i];
        for (int j = 0; j < MAHA_Q15_N_FEATURES; j++) {
            m->sigma_inv_q15[i][j] = TV_Q15_SIGMA_INV[i][j];
        }
    }
}

/* ── Parité C ↔ Python ─────────────────────────────────────────────────────── */

void test_maha_q15_parity_with_python(void)
{
    TEST_ASSERT_EQUAL_INT(MAHA_Q15_N_FEATURES, TV_Q15_DIM);

    MahalanobisQ15 m;
    load_ref_detector(&m);

    float x[MAHA_Q15_N_FEATURES];
    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) x[i] = TV_Q15_INPUT[i];

    float dist = maha_q15_score(&m, x);
    TEST_ASSERT_FLOAT_WITHIN(TOL_Q15, TV_Q15_EXPECTED_DIST, dist);
}

/* ── Propriétés de base ─────────────────────────────────────────────────────── */

void test_maha_q15_init_is_euclidean(void)
{
    /* Init neutre : mu=0 (q8=0, zp=0, scale=1), Σ⁻¹ = identité (scale=1) → distance L2. */
    MahalanobisQ15 m;
    maha_q15_init(&m, 10.0f, 0.1f);

    float x[MAHA_Q15_N_FEATURES];
    float ss = 0.0f;
    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        x[i] = (float)(i + 1);
        ss += x[i] * x[i];
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, sqrtf(ss), maha_q15_score(&m, x));
}

void test_maha_q15_zero_distance(void)
{
    MahalanobisQ15 m;
    maha_q15_init(&m, 1.0f, 0.1f);   /* mu = 0 */
    float x[MAHA_Q15_N_FEATURES] = {0};
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, maha_q15_score(&m, x));
}

void test_maha_q15_predict_threshold(void)
{
    MahalanobisQ15 m;
    maha_q15_init(&m, 2.0f, 0.1f);   /* seuil = 2.0, Σ⁻¹ = I */
    float below[MAHA_Q15_N_FEATURES] = {1.0f, 0, 0, 0, 0};  /* dist = 1 < 2 */
    float above[MAHA_Q15_N_FEATURES] = {3.0f, 0, 0, 0, 0};  /* dist = 3 > 2 */
    TEST_ASSERT_EQUAL_INT(0, maha_q15_predict(&m, below));
    TEST_ASSERT_EQUAL_INT(1, maha_q15_predict(&m, above));
}
