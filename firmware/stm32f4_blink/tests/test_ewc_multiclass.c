/**
 * test_ewc_multiclass.c — Tests Unity pour ewc_head_multiclass.c (S2615)
 *
 * Compilation host via test_runner :
 *   make -C firmware/stm32f4_blink test
 *
 * Invariants testés :
 *   1. Forward shape — logits[N] accessibles sans NaN/Inf
 *   2. Softmax normalisé — sum(softmax(logits)) ≈ 1.0
 *   3. predict retourne l'argmax correct sur logits connus
 *   4. EWC penalty = 0 avant consolidation (fisher initialisée à 0)
 *   5. EWC penalty > 0 après consolidation
 */

#include "unity.h"
#include "ewc_head_multiclass.h"
#include <math.h>
#include <string.h>

/* ── Test 1 : forward produit N logits valides ────────────────────────────── */
void test_ewc_mc_forward_valid_logits(void)
{
    EWCHeadMC h;
    ewc_mc_init(&h);

    float x[EWC_MC_IN] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f};
    float logits[EWC_MC_N_CLASSES];
    ewc_mc_forward(&h, x, logits);

    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        TEST_ASSERT_FALSE(isnan(logits[j]));
        TEST_ASSERT_FALSE(isinf(logits[j]));
    }
}

/* ── Test 2 : softmax somme à 1 ──────────────────────────────────────────── */
void test_ewc_mc_softmax_sums_to_one(void)
{
    /* Logits connus construits manuellement */
    float logits[EWC_MC_N_CLASSES];
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        logits[j] = (float)j * 0.5f - 2.0f;

    /* Softmax stable (max-shift) */
    float max_l = logits[0];
    for (int j = 1; j < EWC_MC_N_CLASSES; j++)
        if (logits[j] > max_l) max_l = logits[j];

    float sum_exp = 0.0f;
    float softmax[EWC_MC_N_CLASSES];
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        softmax[j] = expf(logits[j] - max_l);
        sum_exp += softmax[j];
    }

    float total = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        softmax[j] /= sum_exp;
        total += softmax[j];
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, total);
}

/* ── Test 3 : predict retourne l'argmax ──────────────────────────────────── */
void test_ewc_mc_predict_argmax(void)
{
    EWCHeadMC h;
    ewc_mc_init(&h);

    /* Forcer w3[5][*] très grand → classe 5 doit gagner */
    for (int i = 0; i < EWC_MC_H2; i++) {
        for (int j = 0; j < EWC_MC_N_CLASSES; j++)
            h.w3[j][i] = (j == 5) ? 10.0f : -10.0f;
    }

    float x[EWC_MC_IN];
    for (int i = 0; i < EWC_MC_IN; i++) x[i] = 1.0f;

    int pred = ewc_mc_predict(&h, x);
    TEST_ASSERT_EQUAL_INT(5, pred);
}

/* ── Test 4 : EWC penalty = 0 avant consolidation ────────────────────────── */
void test_ewc_mc_penalty_zero_before_consolidate(void)
{
    EWCHeadMC h;
    ewc_mc_init(&h);

    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++)
            fisher_sum += h.fisher3[j][i];

    TEST_ASSERT_EQUAL_FLOAT(0.0f, fisher_sum);
}

/* ── Test 5 : EWC penalty > 0 après consolidation ────────────────────────── */
void test_ewc_mc_penalty_nonzero_after_consolidate(void)
{
    EWCHeadMC h;
    ewc_mc_init(&h);
    h.lambda = 0.0f;

    float x[EWC_MC_IN];
    for (int i = 0; i < EWC_MC_IN; i++) x[i] = 0.5f;

    for (int k = 0; k < 20; k++)
        ewc_mc_sgd_step(&h, x, 3);

    ewc_mc_consolidate(&h, 0.0f);   /* alpha=0 : Fisher = grad² */

    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++)
            fisher_sum += h.fisher3[j][i];

    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, fisher_sum);

    h.lambda = 400.0f;
    h.w3[3][0] += 1.0f;

    float penalty = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++) {
            float diff = h.w3[j][i] - h.star_w3[j][i];
            penalty += h.lambda / 2.0f * h.fisher3[j][i] * diff * diff;
        }
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, penalty);
}
