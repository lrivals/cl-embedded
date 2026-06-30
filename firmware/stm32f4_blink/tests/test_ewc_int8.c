/**
 * test_ewc_int8.c — Tests Unity pour ewc_head_int8.c
 *
 * Exécution sur x86 (pas de board requis).
 * Critère forward : delta output INT8 vs FP32 < 0.05 sur chaque logit.
 *
 * Référence Sprint 22 : S2221_ewc_int8_c.md
 */

#include "unity.h"
#include "ewc_head_int8.h"
#include "ewc_head.h"
#include <math.h>
#include <string.h>

/* ── Utilitaires locaux ─────────────────────────────────────────────────── */

static EWCHead make_fp32_head(void)
{
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 0.0f;
    /* Petits poids non nuls déterministes (même pattern que test_ewc_head.c) */
    for (int i = 0; i < EWC_IN; i++) {
        h.w1[i][i] = 0.1f;
    }
    for (int j = 0; j < EWC_H1; j++) h.b1[j] = 0.05f;
    for (int i = 0; i < EWC_H2; i++) {
        h.w2[i][i] = 0.1f;
    }
    for (int j = 0; j < EWC_H2; j++) h.b2[j] = 0.05f;
    for (int i = 0; i < EWC_H2; i++) {
        h.w3[0][i] =  0.1f;
        h.w3[1][i] = -0.1f;
    }
    return h;
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

void test_int8_forward_close_to_fp32(void)
{
    /* Initialiser EWCHead (FP32) et EWCHeadInt8 avec mêmes poids */
    EWCHead     fp32 = make_fp32_head();
    EWCHeadInt8 q7;
    ewc_int8_from_fp32(&q7, &fp32);

    /* Input de test : x = [0.1, -0.3, 0.5, 0.0, -0.2] */
    float  x_f[EWC_IN] = {0.1f, -0.3f, 0.5f, 0.0f, -0.2f};
    int8_t x_q7[EWC_IN];
    for (int i = 0; i < EWC_IN; i++) x_q7[i] = float_to_q7(x_f[i]);

    float out_fp32[EWC_OUT];
    ewc_forward(&fp32, x_f, out_fp32);

    float out_int8[EWC_OUT];
    ewc_int8_forward(&q7, x_q7, out_int8);

    for (int j = 0; j < EWC_OUT; j++) {
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.05f, out_fp32[j], out_int8[j],
            "INT8 vs FP32 delta > 0.05");
    }
}

void test_int8_update_does_not_crash(void)
{
    /* SGD INT8 step doit s'exécuter sans crash (pas d'assertion métrique) */
    EWCHead     fp32 = make_fp32_head();
    EWCHeadInt8 q7;
    ewc_int8_from_fp32(&q7, &fp32);

    float  x_f[EWC_IN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int8_t x_q7[EWC_IN];
    for (int i = 0; i < EWC_IN; i++) x_q7[i] = float_to_q7(x_f[i]);

    ewc_int8_update(&q7, x_q7, 0, 0.01f, 0);
    /* Pas de crash = succès */
    TEST_PASS();
}

void test_int8_consolidate_updates_star_w(void)
{
    /* Après consolidate, star_w1 doit être une copie exacte de w1 */
    EWCHead     fp32 = make_fp32_head();
    EWCHeadInt8 q7;
    ewc_int8_from_fp32(&q7, &fp32);

    /* Modifier quelques poids pour créer un écart w / star_w */
    q7.w1[0][0] = 42;
    q7.w1[1][1] = -10;

    ewc_int8_consolidate(&q7);

    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            TEST_ASSERT_EQUAL_INT8(q7.w1[j][i], q7.star_w1[j][i]);
        }
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            TEST_ASSERT_EQUAL_INT8(q7.w2[j][i], q7.star_w2[j][i]);
        }
    }
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            TEST_ASSERT_EQUAL_INT8(q7.w3[j][i], q7.star_w3[j][i]);
        }
    }
}

void test_int8_relu_q7_clamps_negative(void)
{
    /* relu_q7 doit annuler les valeurs négatives et laisser les positives */
    TEST_ASSERT_EQUAL_INT8(0,   relu_q7(-50));
    TEST_ASSERT_EQUAL_INT8(0,   relu_q7(-1));
    TEST_ASSERT_EQUAL_INT8(0,   relu_q7(0));
    TEST_ASSERT_EQUAL_INT8(50,  relu_q7(50));
    TEST_ASSERT_EQUAL_INT8(127, relu_q7(127));
}

void test_int8_float_q7_roundtrip(void)
{
    /* Aller-retour float→Q7→float avec précision Q7 (1/128 ≈ 0.0078) */
    float val = 0.5f;
    float recovered = q7_to_float(float_to_q7(val));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, val, recovered);

    val = -0.25f;
    recovered = q7_to_float(float_to_q7(val));
    TEST_ASSERT_FLOAT_WITHIN(0.01f, val, recovered);

    /* Valeur nulle */
    TEST_ASSERT_EQUAL_INT8(0, float_to_q7(0.0f));
}
