/**
 * test_hdc.c — Tests unitaires Unity pour hdc.c
 *
 * Architecture : random projection FP32, mémoire associative, binarisation ±1
 * Tests :
 *   - test_hdc_encode_norm                    : |hv_out|² == HDC_DIM
 *   - test_hdc_predict_label                  : classe correcte sur données mock
 *   - test_hdc_update_accumulates             : accumulation correcte dans am
 *   - test_hdc_binarize_preserves_sign        : signe conservé après binarize
 *   - test_hdc_binarize_norm_is_hdcdim        : norme ±1 après binarize
 *   - test_hdc_n_trained_increments           : n_trained incrémenté à chaque update
 *   - test_hdc_update_with_sample_fills_buffer: buf_count et buf_head corrects
 *   - test_hdc_buf_wraps_at_retrain_buf       : buf_count plafonné à HDC_RETRAIN_BUF
 *   - test_hdc_retrain_predicts_correct_after_reset : retrain restaure les prédictions
 *   - test_hdc_init_zeros_all_fields          : hdc_init réinitialise tous les champs
 */

#include "unity.h"
#include "hdc.h"
#include <math.h>
#include <string.h>

#define TOL 1e-3f

/* HDCClassifier avec proj identité sur la feature 0 :
 * proj[i][0] = 1.0f pour tout i → hv_out[i] = sign(x[0]) pour tout i. */
static HDCClassifier make_identity_proj(void)
{
    HDCClassifier h;
    memset(&h, 0, sizeof(h));
    for (int i = 0; i < HDC_DIM; i++) {
        h.proj[i][0] = 1.0f;
    }
    return h;
}

/* ── test_hdc_encode_norm ─────────────────────────────────────────────────── */

void test_hdc_encode_norm(void)
{
    /* Avec x[0]=2.0 et proj[i][0]=1.0, dot[i]=2.0>0 → hv_out[i]=+1.0 pour tout i.
     * sum(hv_out[i]²) = HDC_DIM * 1² = HDC_DIM exactement. */
    HDCClassifier h = make_identity_proj();
    float x[HDC_N_FEATURES] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];

    hdc_encode(&h, x, hv);

    float norm_sq = 0.0f;
    for (int i = 0; i < HDC_DIM; i++) {
        norm_sq += hv[i] * hv[i];
    }
    TEST_ASSERT_FLOAT_WITHIN(TOL, (float)HDC_DIM, norm_sq);
}

/* ── test_hdc_predict_label ───────────────────────────────────────────────── */

void test_hdc_predict_label(void)
{
    /* Scénario :
     *   x0 = {+1, 0, 0, 0, 0} → hv0 = all +1.0f → update classe 0
     *   x1 = {-1, 0, 0, 0, 0} → hv1 = all -1.0f → update classe 1
     * Après update, am[0] = +HDC_DIM * (+1.0f), am[1] = +HDC_DIM * (-1.0f).
     * predict(hv0) : dot(am[0],hv0) = HDC_DIM >> dot(am[1],hv0) = -HDC_DIM → classe 0
     * predict(hv1) : dot(am[1],hv1) = HDC_DIM >> dot(am[0],hv1) = -HDC_DIM → classe 1 */
    HDCClassifier h = make_identity_proj();

    float x0[HDC_N_FEATURES] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float x1[HDC_N_FEATURES] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv0[HDC_DIM], hv1[HDC_DIM];

    hdc_encode(&h, x0, hv0);
    hdc_encode(&h, x1, hv1);
    hdc_update(&h, hv0, 0);
    hdc_update(&h, hv1, 1);

    TEST_ASSERT_EQUAL_INT(0, hdc_predict(&h, hv0));
    TEST_ASSERT_EQUAL_INT(1, hdc_predict(&h, hv1));
}

/* ── Nouveaux tests (Sprint 23) ───────────────────────────────────────────── */

void test_hdc_update_accumulates(void)
{
    /* hdc_update ajoute hv à am[label].
     * Après 3 updates de {+1,...} sur classe 0 : am[0][0] == 3.0 (proj identité). */
    HDCClassifier h = make_identity_proj();
    float x[HDC_N_FEATURES] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);
    hdc_update(&h, hv, 0);
    hdc_update(&h, hv, 0);
    hdc_update(&h, hv, 0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 3.0f, h.am[0][0]);
}

void test_hdc_binarize_preserves_sign(void)
{
    /* Après accumulation de vecteurs +1, am[0][i] > 0 → binarize → am[0][i] == +1.
     * Après update de vecteurs -1, am[1][i] < 0 → binarize → am[1][i] == -1. */
    HDCClassifier h = make_identity_proj();
    float xp[HDC_N_FEATURES] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float xn[HDC_N_FEATURES] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hvp[HDC_DIM], hvn[HDC_DIM];
    hdc_encode(&h, xp, hvp);
    hdc_encode(&h, xn, hvn);
    hdc_update(&h, hvp, 0);
    hdc_update(&h, hvn, 1);
    hdc_binarize(&h);
    TEST_ASSERT_FLOAT_WITHIN(TOL,  1.0f, h.am[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(TOL, -1.0f, h.am[1][0]);
}

void test_hdc_binarize_norm_is_hdcdim(void)
{
    /* Après binarize, chaque prototype am[c] est un vecteur ±1 :
     * sum(am[c][i]²) == HDC_DIM exactement. */
    HDCClassifier h = make_identity_proj();
    for (int i = 0; i < HDC_DIM; i++) {
        h.am[0][i] = (float)(i % 5) - 2.0f;
        h.am[1][i] = (float)(i % 3) - 1.5f;
    }
    hdc_binarize(&h);
    float norm0 = 0.0f, norm1 = 0.0f;
    for (int i = 0; i < HDC_DIM; i++) {
        norm0 += h.am[0][i] * h.am[0][i];
        norm1 += h.am[1][i] * h.am[1][i];
    }
    TEST_ASSERT_FLOAT_WITHIN(TOL, (float)HDC_DIM, norm0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, (float)HDC_DIM, norm1);
}

void test_hdc_n_trained_increments(void)
{
    HDCClassifier h;
    hdc_init(&h);
    TEST_ASSERT_EQUAL_INT(0, h.n_trained);
    float hv[HDC_DIM];
    memset(hv, 0, sizeof(hv));
    hdc_update(&h, hv, 0);
    TEST_ASSERT_EQUAL_INT(1, h.n_trained);
    hdc_update(&h, hv, 1);
    TEST_ASSERT_EQUAL_INT(2, h.n_trained);
}

void test_hdc_update_with_sample_fills_buffer(void)
{
    HDCClassifier h;
    hdc_init(&h);
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x[HDC_N_FEATURES]  = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);
    hdc_update_with_sample(&h, x, hv, 0);

    TEST_ASSERT_EQUAL_INT(1, h.buf_count);
    TEST_ASSERT_EQUAL_INT(1, h.buf_head % HDC_RETRAIN_BUF);
}

void test_hdc_buf_wraps_at_retrain_buf(void)
{
    /* Après HDC_RETRAIN_BUF+5 insertions, buf_count == HDC_RETRAIN_BUF (pas plus). */
    HDCClassifier h;
    hdc_init(&h);
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x[HDC_N_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);

    for (int k = 0; k < HDC_RETRAIN_BUF + 5; k++) {
        hdc_update_with_sample(&h, x, hv, k % 2);
    }
    TEST_ASSERT_EQUAL_INT(HDC_RETRAIN_BUF, h.buf_count);
}

void test_hdc_retrain_predicts_correct_after_reset(void)
{
    /* Scénario : entraîner classe 0 et classe 1, retrain depuis buffer,
     * vérifier que les prédictions sont correctes après corruption de l'AM. */
    HDCClassifier h;
    hdc_init(&h);
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x0[HDC_N_FEATURES] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float x1[HDC_N_FEATURES] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv0[HDC_DIM], hv1[HDC_DIM];
    hdc_encode(&h, x0, hv0);
    hdc_encode(&h, x1, hv1);

    for (int k = 0; k < 5; k++) {
        hdc_update_with_sample(&h, x0, hv0, 0);
        hdc_update_with_sample(&h, x1, hv1, 1);
    }

    /* Corrompre l'AM */
    memset(h.am, 0, sizeof(h.am));

    /* Retrain depuis le buffer doit restaurer les prédictions correctes */
    hdc_retrain(&h);
    TEST_ASSERT_EQUAL_INT(0, hdc_predict(&h, hv0));
    TEST_ASSERT_EQUAL_INT(1, hdc_predict(&h, hv1));
}

void test_hdc_init_zeros_all_fields(void)
{
    HDCClassifier h;
    /* Initialiser avec des déchets */
    memset(&h, 0xAB, sizeof(h));
    hdc_init(&h);
    TEST_ASSERT_EQUAL_INT(0, h.n_trained);
    TEST_ASSERT_EQUAL_INT(0, h.buf_head);
    TEST_ASSERT_EQUAL_INT(0, h.buf_count);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.am[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.am[1][0]);
}
