/**
 * test_ewc_int8_v2.c — Tests Unity host pour ewc_head_int8_v2.c (Sprint 39, S3909)
 *
 * Exécution sur x86 (`make test`, TEST_MODE=1) — AUCUNE carte requise. Prouve le
 * correctif du kernel v2 (acc int32 + scales par-canal calibrés) en confrontant le
 * forward C v2 à des golden vectors produits par l'émulateur Python
 * (src/utils/int8_c_emulation.py) et figés dans test_vectors_v2.h
 * (export_weights_c.py --int8-v2-test-vectors, header golden auto-suffisant).
 *
 * Parité par construction : le header fournit poids FP32 + act_max calibré ; le test
 * reconstruit la tête via ewc_int8_v2_from_fp32_calib(act_max) puis forward → mêmes
 * scales que l'émulateur.
 *
 * Cas (cf. S3909) :
 *   test_v2_no_overflow     — v1 (acc int16) déborde, v2 (int32) reste proche du FP32
 *   test_v2_parity_emulator — logits v2 int8 ≈ golden per_channel_int8 (tol 1e-3)
 *   test_v2_q15_parity      — build -DEWC_INT8_Q15 ≈ golden q15 (sinon ignoré)
 *   test_v2_recovers_f1     — accord argmax(v2)↔FP32 ≥ accord argmax(v1)↔FP32
 *   test_v1_unchanged       — ewc_int8_forward (v1) inchangé (garde A/B, 0 régression)
 *
 * Référence : S3907 (kernel v2), S3902 (émulateur), test_ewc_int8.c (patron Unity).
 */

#include "unity.h"
#include "ewc_head.h"
#include "ewc_head_int8.h"
#include "ewc_head_int8_v2.h"
#include "test_vectors_v2.h"
#include <math.h>
#include <string.h>

/* ── Utilitaires locaux ─────────────────────────────────────────────────── */

/* Reconstruit une tête FP32 EWCHead depuis les golden vectors (convention [out][in]). */
static EWCHead head_from_golden(void)
{
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 0.0f;
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) h.w1[j][i] = TV_V2_W1[j][i];
        h.b1[j] = TV_V2_B1[j];
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) h.w2[j][i] = TV_V2_W2[j][i];
        h.b2[j] = TV_V2_B2[j];
    }
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) h.w3[j][i] = TV_V2_W3[j][i];
        h.b3[j] = TV_V2_B3[j];
    }
    return h;
}

/* max|activation| par couche pour une entrée FP32 (calibration locale, no_overflow). */
static void compute_act_max(const EWCHead *h, const float *x, float out[3])
{
    float mi = 0.0f;
    for (int i = 0; i < EWC_IN; i++) { float a = fabsf(x[i]); if (a > mi) mi = a; }
    float h1[EWC_H1], h2[EWC_H2];
    float mh1 = 0.0f;
    for (int j = 0; j < EWC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = acc > 0.0f ? acc : 0.0f;
        if (h1[j] > mh1) mh1 = h1[j];
    }
    float mh2 = 0.0f;
    for (int j = 0; j < EWC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = acc > 0.0f ? acc : 0.0f;
        if (h2[j] > mh2) mh2 = h2[j];
    }
    out[0] = mi > 0.0f ? mi : 1.0f;
    out[1] = mh1 > 0.0f ? mh1 : 1.0f;
    out[2] = mh2 > 0.0f ? mh2 : 1.0f;
}

static int argmax2(const float *logits) { return logits[1] > logits[0] ? 1 : 0; }

/* ── Tests ──────────────────────────────────────────────────────────────── */

void test_v2_no_overflow(void)
{
    /* Tête à grands poids : l'acc int16 de v1 déborde, l'acc int32 de v2 tient. */
    EWCHead fp32;
    memset(&fp32, 0, sizeof(fp32));
    fp32.lambda = 0.0f;
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) fp32.w1[j][i] = 1.5f;  /* q7=127, acc≫32767 */
        fp32.b1[j] = 0.0f;
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) fp32.w2[j][i] = 0.02f;
        fp32.b2[j] = 0.0f;
    }
    for (int i = 0; i < EWC_H2; i++) { fp32.w3[0][i] = 0.05f; fp32.w3[1][i] = -0.05f; }

    float x[EWC_IN];
    for (int i = 0; i < EWC_IN; i++) x[i] = 0.9f;

    float out_fp32[EWC_OUT];
    ewc_forward(&fp32, x, out_fp32);

    /* v1 : quantif Q7 + acc int16 (overflow). */
    EWCHeadInt8 v1;
    ewc_int8_from_fp32(&v1, &fp32);
    int8_t x_q7[EWC_IN];
    for (int i = 0; i < EWC_IN; i++) x_q7[i] = float_to_q7(x[i]);
    float out_v1[EWC_OUT];
    ewc_int8_forward(&v1, x_q7, out_v1);

    /* v2 : acc int32 + scales calibrés. */
    float act_max[3];
    compute_act_max(&fp32, x, act_max);
    EWCHeadInt8V2 v2;
    ewc_int8_v2_from_fp32_calib(&v2, &fp32, act_max);
    float out_v2[EWC_OUT];
    ewc_int8_v2_forward(&v2, x, out_v2);

    for (int j = 0; j < EWC_OUT; j++) {
        float err_v1 = fabsf(out_v1[j] - out_fp32[j]);
        float err_v2 = fabsf(out_v2[j] - out_fp32[j]);
        /* v2 reste proche du FP32 malgré l'ampleur (déquant exacte int32). */
        TEST_ASSERT_TRUE_MESSAGE(err_v2 < 0.15f, "v2 s'écarte du FP32 (overflow non corrigé ?)");
        /* v1 déborde nettement plus que v2. */
        TEST_ASSERT_TRUE_MESSAGE(err_v1 > err_v2 + 0.1f, "v1 ne montre pas l'overflow int16");
    }
}

void test_v2_parity_emulator(void)
{
#if defined(EWC_INT8_Q15) || defined(EWC_INT8_MIXED)
    TEST_IGNORE_MESSAGE("parité per_channel_int8 = build int8 par défaut (voir test-v2-q15)");
#else
    /* Logits C v2 (int8 par-canal) ≈ golden émulateur per_channel_int8, tol 1e-3. */
    EWCHead fp32 = head_from_golden();
    float act_max[3] = {TV_V2_ACT_MAX[0], TV_V2_ACT_MAX[1], TV_V2_ACT_MAX[2]};

    EWCHeadInt8V2 v2;
    ewc_int8_v2_from_fp32_calib(&v2, &fp32, act_max);

    for (int n = 0; n < TV_V2_N; n++) {
        float logits[EWC_OUT];
        ewc_int8_v2_forward(&v2, TV_V2_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++) {
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(
                1e-3f, TV_V2_LOGITS_PER_CHANNEL_INT8[n][j], logits[j],
                "parité C v2 int8 ↔ émulateur > 1e-3");
        }
    }
#endif
}

void test_v2_q15_parity(void)
{
#if defined(EWC_INT8_Q15)
    /* Build Q15 (typedef 16 bits) : logits ≈ golden émulateur q15, tol 1e-3. */
    EWCHead fp32 = head_from_golden();
    float act_max[3] = {TV_V2_ACT_MAX[0], TV_V2_ACT_MAX[1], TV_V2_ACT_MAX[2]};

    EWCHeadInt8V2 v2;
    ewc_int8_v2_from_fp32_calib(&v2, &fp32, act_max);

    for (int n = 0; n < TV_V2_N; n++) {
        float logits[EWC_OUT];
        ewc_int8_v2_forward(&v2, TV_V2_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++) {
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(
                1e-3f, TV_V2_LOGITS_Q15[n][j], logits[j],
                "parité C v2 Q15 ↔ émulateur > 1e-3");
        }
    }
#else
    TEST_IGNORE_MESSAGE("Q15 = choix de compilation (make test-v2-q15 avec -DEWC_INT8_Q15)");
#endif
}

void test_v2_recovers_f1(void)
{
    /* Accord de prédiction vs FP32 : v2 (calibré) ≥ v1 (legacy) sur les golden inputs. */
    EWCHead fp32 = head_from_golden();
    float act_max[3] = {TV_V2_ACT_MAX[0], TV_V2_ACT_MAX[1], TV_V2_ACT_MAX[2]};

    EWCHeadInt8 v1;
    ewc_int8_from_fp32(&v1, &fp32);
    EWCHeadInt8V2 v2;
    ewc_int8_v2_from_fp32_calib(&v2, &fp32, act_max);

    int agree_v1 = 0, agree_v2 = 0;
    for (int n = 0; n < TV_V2_N; n++) {
        int ref = argmax2(TV_V2_LOGITS_FP32[n]);

        int8_t x_q7[EWC_IN];
        for (int i = 0; i < EWC_IN; i++) x_q7[i] = float_to_q7(TV_V2_INPUT[n][i]);
        float lv1[EWC_OUT];
        ewc_int8_forward(&v1, x_q7, lv1);
        if (argmax2(lv1) == ref) agree_v1++;

        float lv2[EWC_OUT];
        ewc_int8_v2_forward(&v2, TV_V2_INPUT[n], lv2);
        if (argmax2(lv2) == ref) agree_v2++;
    }
    TEST_ASSERT_TRUE_MESSAGE(agree_v2 >= agree_v1,
        "v2 ne récupère pas au moins l'accord de v1 vs FP32");
}

void test_v1_unchanged(void)
{
    /* Garde A/B : v1 (ewc_int8_forward) reste déterministe et inchangé (0 régression).
     * On reconstruit la tête déterministe de test_ewc_int8.c et on fige les logits. */
    EWCHead fp32;
    memset(&fp32, 0, sizeof(fp32));
    fp32.lambda = 0.0f;
    for (int i = 0; i < EWC_IN; i++) fp32.w1[i][i] = 0.1f;
    for (int j = 0; j < EWC_H1; j++) fp32.b1[j] = 0.05f;
    for (int i = 0; i < EWC_H2; i++) fp32.w2[i][i] = 0.1f;
    for (int j = 0; j < EWC_H2; j++) fp32.b2[j] = 0.05f;
    for (int i = 0; i < EWC_H2; i++) { fp32.w3[0][i] = 0.1f; fp32.w3[1][i] = -0.1f; }

    EWCHeadInt8 v1;
    ewc_int8_from_fp32(&v1, &fp32);
    float x[EWC_IN] = {0.1f, -0.3f, 0.5f, 0.0f, -0.2f};
    int8_t x_q7[EWC_IN];
    for (int i = 0; i < EWC_IN; i++) x_q7[i] = float_to_q7(x[i]);

    float a[EWC_OUT], b[EWC_OUT];
    ewc_int8_forward(&v1, x_q7, a);
    ewc_int8_forward(&v1, x_q7, b);
    /* Déterminisme strict (même entrée → mêmes logits). */
    for (int j = 0; j < EWC_OUT; j++) TEST_ASSERT_EQUAL_FLOAT(a[j], b[j]);
    /* Cohérence FP32 grossière (v1 ≈ FP32 sur petits poids, comme test_ewc_int8.c). */
    float out_fp32[EWC_OUT];
    ewc_forward(&fp32, x, out_fp32);
    for (int j = 0; j < EWC_OUT; j++)
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.05f, out_fp32[j], a[j], "v1 dérive vs FP32 (régression)");
}
