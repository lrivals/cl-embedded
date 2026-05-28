/**
 * test_hdc.c — Tests unitaires Unity pour hdc.c
 *
 * Architecture : random projection FP32, mémoire associative, binarisation ±1
 * Tests :
 *   - test_hdc_encode_norm     : |hv_out|² == HDC_DIM (propriété binarisation exacte)
 *   - test_hdc_predict_label   : classe correcte sur données mock synthétiques
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
