/**
 * test_ewc_regression.c — Tests Unity pour ewc_head_regression.c (S2614)
 *
 * Compilation host via test_runner :
 *   make -C firmware/stm32f4_blink test
 *
 * Invariants testés :
 *   1. Forward shape — out[0] accessible sans crash (pas de NaN/Inf)
 *   2. MSE gradient signe correct — sgd_step rapproche la prédiction du label
 *   3. EWC penalty = 0.0 avant consolidation (fisher initialisée à 0)
 *   4. EWC penalty > 0.0 après consolidation (Fisher non nulle)
 *   5. Consolidation copie w vers star_w
 */

#include "unity.h"
#include "ewc_head_regression.h"
#include <math.h>
#include <string.h>

/* ── Test 1 : forward produit un scalaire (pas de NaN, pas de crash) ──────── */
void test_ewc_reg_forward_scalar(void)
{
    EWCHeadReg h;
    ewc_reg_init(&h);

    float x[EWC_REG_IN] = {0.5f, -0.3f, 1.2f, 0.0f, -0.8f};
    float out[EWC_REG_OUT];
    ewc_reg_forward(&h, x, out);

    TEST_ASSERT_FALSE(isnan(out[0]));
    TEST_ASSERT_FALSE(isinf(out[0]));
}

/* ── Test 2 : MSE gradient — après N steps, erreur diminue ───────────────── */
void test_ewc_reg_sgd_reduces_error(void)
{
    EWCHeadReg h;
    ewc_reg_init(&h);
    h.lambda = 0.0f;   /* isoler SGD pur sans terme EWC */

    float x[EWC_REG_IN] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float y_true = 50.0f;

    float out_before[EWC_REG_OUT];
    ewc_reg_forward(&h, x, out_before);
    float err_before = fabsf(out_before[0] - y_true);

    for (int i = 0; i < 50; i++)
        ewc_reg_sgd_step(&h, x, y_true);

    float out_after[EWC_REG_OUT];
    ewc_reg_forward(&h, x, out_after);
    float err_after = fabsf(out_after[0] - y_true);

    TEST_ASSERT_LESS_THAN_FLOAT(err_before, err_after);
}

/* ── Test 3 : EWC penalty = 0 avant consolidation ────────────────────────── */
void test_ewc_reg_penalty_zero_before_consolidate(void)
{
    EWCHeadReg h;
    ewc_reg_init(&h);

    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            fisher_sum += h.fisher1[j][i];

    /* Fisher initialisée à 0 → penalty = 0 */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, fisher_sum);
}

/* ── Test 4 : EWC penalty > 0 après consolidation ────────────────────────── */
void test_ewc_reg_penalty_nonzero_after_consolidate(void)
{
    EWCHeadReg h;
    ewc_reg_init(&h);
    h.lambda = 0.0f;

    float x[EWC_REG_IN] = {1.0f, 0.5f, -0.5f, 0.2f, -0.2f};
    float y_true = 75.0f;

    for (int i = 0; i < 20; i++)
        ewc_reg_sgd_step(&h, x, y_true);

    ewc_reg_consolidate(&h, 0.0f);   /* alpha=0 : Fisher = grad² pur */

    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            fisher_sum += h.fisher1[j][i];

    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, fisher_sum);

    /* Perturber θ et vérifier penalty > 0 */
    h.lambda = 400.0f;
    h.w1[0][0] += 0.5f;

    float penalty = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++) {
            float diff = h.w1[j][i] - h.star_w1[j][i];
            penalty += h.lambda / 2.0f * h.fisher1[j][i] * diff * diff;
        }
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, penalty);
}

/* ── Test 5 : consolidate copie w vers star_w ─────────────────────────────── */
void test_ewc_reg_consolidate_copies_weights(void)
{
    EWCHeadReg h;
    ewc_reg_init(&h);
    h.lambda = 0.0f;

    float x[EWC_REG_IN] = {0.3f, -0.1f, 0.8f, -0.5f, 0.2f};
    for (int i = 0; i < 5; i++)
        ewc_reg_sgd_step(&h, x, 60.0f);

    ewc_reg_consolidate(&h, 0.0f);

    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            TEST_ASSERT_EQUAL_FLOAT(h.w1[j][i], h.star_w1[j][i]);
}
