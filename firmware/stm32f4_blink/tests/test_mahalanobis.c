/**
 * test_mahalanobis.c — Tests unitaires Unity pour mahalanobis.c
 *
 * Valeurs de référence calculées analytiquement (numpy) :
 *   dist([1,2,3,4,5], [1,2,3,4,5], I) = 0
 *   dist([1,2,3,4,5], [0,0,0,0,0], I) = sqrt(1+4+9+16+25) = sqrt(55) ≈ 7.41619848
 */

#include "unity.h"
#include "mahalanobis.h"
#include <math.h>
#include <string.h>

#define TOL 1e-5f

/* Initialise un détecteur avec précision = identité, mean = zéro. */
static MahalanobisDetector make_identity_det(float threshold, float ema_alpha)
{
    MahalanobisDetector det;
    maha_init(&det, threshold, ema_alpha);
    return det;
}

/* ── Tests distance ─────────────────────────────────────────────────────── */

void test_mahal_zero_distance(void)
{
    MahalanobisDetector det = make_identity_det(1.0f, 0.1f);
    float mean[MAHA_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    memcpy(det.mean, mean, sizeof(mean));

    float x[MAHA_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float dist = maha_score(&det, x);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, dist);
}

void test_mahal_identity_is_euclidean(void)
{
    /* precision = identité → Mahalanobis == norme L2 euclidienne */
    MahalanobisDetector det = make_identity_det(10.0f, 0.1f);
    /* mean = [0,0,0,0,0] (déjà initialisé à zéro par maha_init) */

    float x[MAHA_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected = sqrtf(1.0f + 4.0f + 9.0f + 16.0f + 25.0f);  /* sqrt(55) */
    float dist = maha_score(&det, x);
    TEST_ASSERT_FLOAT_WITHIN(TOL, expected, dist);
}

void test_mahal_symmetric_input(void)
{
    /* dist(x, mean=0, I) doit être identique pour x et -x */
    MahalanobisDetector det = make_identity_det(10.0f, 0.1f);

    float x_pos[MAHA_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x_neg[MAHA_DIM] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
    TEST_ASSERT_FLOAT_WITHIN(TOL, maha_score(&det, x_pos), maha_score(&det, x_neg));
}

/* ── Tests predict ──────────────────────────────────────────────────────── */

void test_mahal_predict_below_threshold(void)
{
    /* dist = sqrt(5 × 1²) = sqrt(5) ≈ 2.236, threshold = 3.0 → normal (0) */
    MahalanobisDetector det = make_identity_det(3.0f, 0.1f);
    float x[MAHA_DIM] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    TEST_ASSERT_EQUAL_INT(0, maha_predict(&det, x));
}

void test_mahal_predict_above_threshold(void)
{
    /* dist = sqrt(55) ≈ 7.416, threshold = 5.0 → anomalie (1) */
    MahalanobisDetector det = make_identity_det(5.0f, 0.1f);
    float x[MAHA_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    TEST_ASSERT_EQUAL_INT(1, maha_predict(&det, x));
}

/* ── Tests EMA update ───────────────────────────────────────────────────── */

void test_mahal_ema_update_alpha_one(void)
{
    /* alpha = 1.0 : mean ← 1×x + 0×mean → mean == x après 1 update */
    MahalanobisDetector det = make_identity_det(1.0f, 1.0f);
    float x[MAHA_DIM] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    maha_update(&det, x);
    for (int i = 0; i < MAHA_DIM; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, x[i], det.mean[i]);
    }
}

void test_mahal_ema_update_alpha_zero(void)
{
    /* alpha = 0.0 : mean reste à zéro quelle que soit la mise à jour */
    MahalanobisDetector det = make_identity_det(1.0f, 0.0f);
    float x[MAHA_DIM] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    maha_update(&det, x);
    for (int i = 0; i < MAHA_DIM; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, det.mean[i]);
    }
}

void test_mahal_ema_update_convergence(void)
{
    /* Après N mises à jour avec alpha = 0.5, mean converge vers x */
    MahalanobisDetector det = make_identity_det(1.0f, 0.5f);
    float x[MAHA_DIM] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    for (int iter = 0; iter < 30; iter++) {
        maha_update(&det, x);
    }
    /* Après 30 itérations, erreur < 1e-3 */
    for (int i = 0; i < MAHA_DIM; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, x[i], det.mean[i]);
    }
}

/* ── Tests non-regression precision custom ─────────────────────────────── */

void test_mahal_custom_precision(void)
{
    /* precision = 4×I → dist = 2 × dist_euclidean */
    MahalanobisDetector det = make_identity_det(10.0f, 0.1f);
    for (int i = 0; i < MAHA_DIM; i++) {
        det.precision[i][i] = 4.0f;
    }
    float x[MAHA_DIM] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    /* d² = xᵀ (4I) x = 4 × 1 = 4 → d = 2 */
    float dist = maha_score(&det, x);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 2.0f, dist);
}
