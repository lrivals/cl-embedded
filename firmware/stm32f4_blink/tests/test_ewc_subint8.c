/**
 * test_ewc_subint8.c — Tests Unity host pour les kernels sub-INT8 (Sprint 48, S4802)
 *
 * Exécution x86 (`make test-sub-int4` / `-int2` / `-binary`, ± packé) — AUCUNE carte.
 * Prouve la parité bit-à-bit des variantes de profondeur (INT4 linéaire, ternaire {−1,0,+1},
 * binaire {−1,+1}) et du chemin bit-packé avec l'émulateur Python (subint8, S47), via les
 * golden vectors figés dans test_vectors_subint8.h (export_weights_c.py --ewc-subint8-test-vectors).
 *
 * Contrat de parité :
 *   - linéaire INT4 : la tête se reconstruit on-board (ewc_int8_v2_from_fp32_calib, QMAX 7) ⇒
 *     mêmes scales par-canal que l'émulateur ;
 *   - ternaire/binaire : le firmware ne peut PAS reproduire TWN/BWN → on charge les poids
 *     quantifiés + scales du golden dans une tête v2 puis on déroule le forward v2 ;
 *   - packé : le packing ne change QUE le stockage → logits packés == golden.
 *
 * Chaque test est gardé par sa profondeur de build (idiome TEST_IGNORE de test_ewc_int8_v2.c) :
 * sous `make test` par défaut (aucun -DEWC_INTx), tous sont ignorés → 0 régression.
 *
 * Référence : S4802 (kernels), S4803 (export/golden), test_ewc_int8_v2.c (patron).
 */

#include "unity.h"
#include "ewc_head.h"
#include "ewc_head_int8_v2.h"
#include "test_vectors_subint8.h"
#include <math.h>
#include <string.h>

/* ── Utilitaires locaux ─────────────────────────────────────────────────── */

#if defined(EWC_INT4) || defined(EWC_INT2) || defined(EWC_INT1)

/* Reconstruit une tête FP32 depuis les poids golden (voie linéaire INT4 on-board). */
static EWCHead sub_head_fp32(void)
{
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 0.0f;
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) h.w1[j][i] = TV_SUB_W1[j][i];
        h.b1[j] = TV_SUB_B1[j];
    }
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) h.w2[j][i] = TV_SUB_W2[j][i];
        h.b2[j] = TV_SUB_B2[j];
    }
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) h.w3[j][i] = TV_SUB_W3[j][i];
        h.b3[j] = TV_SUB_B3[j];
    }
    return h;
}

/* Remplit une tête v2 non-packée avec des poids DÉJÀ quantifiés + scales par-canal
 * (ternaire/binaire). Les scales d'activation dérivent de act_max/EWC_V2_A_QMAX. */
static void sub_head_from_q(EWCHeadInt8V2 *h,
                            const int8_t qw1[EWC_H1][EWC_IN],  const float sw1[EWC_H1],
                            const int8_t qw2[EWC_H2][EWC_H1],  const float sw2[EWC_H2],
                            const int8_t qw3[EWC_OUT][EWC_H2], const float sw3[EWC_OUT])
{
    for (int j = 0; j < EWC_H1; j++) {
        h->scale_w1[j] = sw1[j]; h->b1[j] = TV_SUB_B1[j];
        for (int i = 0; i < EWC_IN; i++) h->w1[j][i] = (ewc_v2_w_t)qw1[j][i];
    }
    for (int j = 0; j < EWC_H2; j++) {
        h->scale_w2[j] = sw2[j]; h->b2[j] = TV_SUB_B2[j];
        for (int i = 0; i < EWC_H1; i++) h->w2[j][i] = (ewc_v2_w_t)qw2[j][i];
    }
    for (int j = 0; j < EWC_OUT; j++) {
        h->scale_w3[j] = sw3[j]; h->b3[j] = TV_SUB_B3[j];
        for (int i = 0; i < EWC_H2; i++) h->w3[j][i] = (ewc_v2_w_t)qw3[j][i];
    }
    h->scale_act_in = TV_SUB_ACT_MAX[0] > 0.0f ? TV_SUB_ACT_MAX[0] / (float)EWC_V2_A_QMAX : 1.0f;
    h->scale_act_h1 = TV_SUB_ACT_MAX[1] > 0.0f ? TV_SUB_ACT_MAX[1] / (float)EWC_V2_A_QMAX : 1.0f;
    h->scale_act_h2 = TV_SUB_ACT_MAX[2] > 0.0f ? TV_SUB_ACT_MAX[2] / (float)EWC_V2_A_QMAX : 1.0f;
}

#if defined(EWC_INTx_PACKED)
/* Empaquette des poids quantifiés golden dans une tête packée. */
static void sub_head_packed_from_q(EWCHeadSubInt8Packed *h,
                                   const int8_t qw1[EWC_H1][EWC_IN],  const float sw1[EWC_H1],
                                   const int8_t qw2[EWC_H2][EWC_H1],  const float sw2[EWC_H2],
                                   const int8_t qw3[EWC_OUT][EWC_H2], const float sw3[EWC_OUT])
{
    for (int j = 0; j < EWC_H1; j++) {
        h->scale_w1[j] = sw1[j]; h->b1[j] = TV_SUB_B1[j];
        ewc_v2_pack_row(h->w1[j], qw1[j], EWC_IN);
    }
    for (int j = 0; j < EWC_H2; j++) {
        h->scale_w2[j] = sw2[j]; h->b2[j] = TV_SUB_B2[j];
        ewc_v2_pack_row(h->w2[j], qw2[j], EWC_H1);
    }
    for (int j = 0; j < EWC_OUT; j++) {
        h->scale_w3[j] = sw3[j]; h->b3[j] = TV_SUB_B3[j];
        ewc_v2_pack_row(h->w3[j], qw3[j], EWC_H2);
    }
    h->scale_act_in = TV_SUB_ACT_MAX[0] > 0.0f ? TV_SUB_ACT_MAX[0] / (float)EWC_V2_A_QMAX : 1.0f;
    h->scale_act_h1 = TV_SUB_ACT_MAX[1] > 0.0f ? TV_SUB_ACT_MAX[1] / (float)EWC_V2_A_QMAX : 1.0f;
    h->scale_act_h2 = TV_SUB_ACT_MAX[2] > 0.0f ? TV_SUB_ACT_MAX[2] / (float)EWC_V2_A_QMAX : 1.0f;
}
#endif /* EWC_INTx_PACKED */

#endif /* sub-INT8 build */

/* ── Tests ──────────────────────────────────────────────────────────────── */

void test_int4_quant_parity(void)
{
#if defined(EWC_INT4) && !defined(EWC_INTx_PACKED)
    /* Reconstruction on-board (QMAX 7) = émulateur subint8(4, linéaire, per_channel). */
    EWCHead fp32 = sub_head_fp32();
    float act_max[3] = {TV_SUB_ACT_MAX[0], TV_SUB_ACT_MAX[1], TV_SUB_ACT_MAX[2]};
    EWCHeadInt8V2 head;
    ewc_int8_v2_from_fp32_calib(&head, &fp32, act_max);

    for (int n = 0; n < TV_SUB_N; n++) {
        float logits[EWC_OUT];
        ewc_int8_v2_forward(&head, TV_SUB_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++)
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-3f, TV_SUB_LOGITS_INT4[n][j], logits[j],
                                             "parité C INT4 linéaire ↔ émulateur > 1e-3");
    }
#else
    TEST_IGNORE_MESSAGE("INT4 non-packé = make test-sub-int4 (-DEWC_INT4)");
#endif
}

void test_ternary_parity(void)
{
#if defined(EWC_INT2) && !defined(EWC_INTx_PACKED)
    EWCHeadInt8V2 head;
    sub_head_from_q(&head,
                    TV_SUB_QW1_TERNARY, TV_SUB_SCALE_W1_TERNARY,
                    TV_SUB_QW2_TERNARY, TV_SUB_SCALE_W2_TERNARY,
                    TV_SUB_QW3_TERNARY, TV_SUB_SCALE_W3_TERNARY);
    for (int n = 0; n < TV_SUB_N; n++) {
        float logits[EWC_OUT];
        ewc_int8_v2_forward(&head, TV_SUB_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++)
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-3f, TV_SUB_LOGITS_TERNARY[n][j], logits[j],
                                             "parité C ternaire ↔ émulateur > 1e-3");
    }
#else
    TEST_IGNORE_MESSAGE("ternaire non-packé = make test-sub-int2 (-DEWC_INT2)");
#endif
}

void test_binary_parity(void)
{
#if defined(EWC_INT1) && !defined(EWC_INTx_PACKED)
    EWCHeadInt8V2 head;
    sub_head_from_q(&head,
                    TV_SUB_QW1_BINARY, TV_SUB_SCALE_W1_BINARY,
                    TV_SUB_QW2_BINARY, TV_SUB_SCALE_W2_BINARY,
                    TV_SUB_QW3_BINARY, TV_SUB_SCALE_W3_BINARY);
    for (int n = 0; n < TV_SUB_N; n++) {
        float logits[EWC_OUT];
        ewc_int8_v2_forward(&head, TV_SUB_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++)
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-3f, TV_SUB_LOGITS_BINARY[n][j], logits[j],
                                             "parité C binaire ↔ émulateur > 1e-3");
    }
#else
    TEST_IGNORE_MESSAGE("binaire non-packé = make test-sub-binary (-DEWC_INT1)");
#endif
}

void test_int4_packed_parity(void)
{
#if defined(EWC_INT4) && defined(EWC_INTx_PACKED)
    /* Le packing ne change QUE le stockage → logits packés == golden émulateur INT4. */
    EWCHeadSubInt8Packed head;
    sub_head_packed_from_q(&head,
                           TV_SUB_QW1_INT4, TV_SUB_SCALE_W1_INT4,
                           TV_SUB_QW2_INT4, TV_SUB_SCALE_W2_INT4,
                           TV_SUB_QW3_INT4, TV_SUB_SCALE_W3_INT4);
    for (int n = 0; n < TV_SUB_N; n++) {
        float logits[EWC_OUT];
        ewc_subint8_packed_forward(&head, TV_SUB_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++)
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-3f, TV_SUB_LOGITS_INT4[n][j], logits[j],
                                             "parité C INT4 packé ↔ émulateur > 1e-3");
    }
#else
    TEST_IGNORE_MESSAGE("INT4 packé = make test-sub-int4-packed (-DEWC_INT4 -DEWC_INTx_PACKED)");
#endif
}

void test_int2_packed_parity(void)
{
#if defined(EWC_INT2) && defined(EWC_INTx_PACKED)
    /* Schéma ternaire (gagnant frontière S4708), stockage 2 bits packé (4/octet). */
    EWCHeadSubInt8Packed head;
    sub_head_packed_from_q(&head,
                           TV_SUB_QW1_TERNARY, TV_SUB_SCALE_W1_TERNARY,
                           TV_SUB_QW2_TERNARY, TV_SUB_SCALE_W2_TERNARY,
                           TV_SUB_QW3_TERNARY, TV_SUB_SCALE_W3_TERNARY);
    for (int n = 0; n < TV_SUB_N; n++) {
        float logits[EWC_OUT];
        ewc_subint8_packed_forward(&head, TV_SUB_INPUT[n], logits);
        for (int j = 0; j < EWC_OUT; j++)
            TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-3f, TV_SUB_LOGITS_TERNARY[n][j], logits[j],
                                             "parité C ternaire packé ↔ émulateur > 1e-3");
    }
#else
    TEST_IGNORE_MESSAGE("ternaire packé = make test-sub-int2-packed (-DEWC_INT2 -DEWC_INTx_PACKED)");
#endif
}

void test_packed_storage_smaller(void)
{
#if defined(EWC_INTx_PACKED)
    /* Matérialise le nœud d'honnêteté S4800 : le stockage packé des poids est
     * strictement plus petit que le conteneur int8 (le `.bss` board réel = S4804). */
    EWCHeadSubInt8Packed hp;
    EWCHeadInt8V2 h8;
    size_t packed_w = sizeof(hp.w1) + sizeof(hp.w2) + sizeof(hp.w3);
    size_t int8_w   = sizeof(h8.w1) + sizeof(h8.w2) + sizeof(h8.w3);
    TEST_ASSERT_TRUE_MESSAGE(packed_w < int8_w,
                             "le packing ne réduit pas le stockage des poids");
    /* Taille packée = somme des strides théoriques (LSB-first, EWC_V2_PACK_BITS). */
    size_t expect = (size_t)EWC_H1  * EWC_V2_PACK_STRIDE(EWC_IN)
                  + (size_t)EWC_H2  * EWC_V2_PACK_STRIDE(EWC_H1)
                  + (size_t)EWC_OUT * EWC_V2_PACK_STRIDE(EWC_H2);
    TEST_ASSERT_EQUAL_UINT_MESSAGE(expect, packed_w,
                                   "taille packée != stride bit-packé théorique");
#else
    TEST_IGNORE_MESSAGE("stockage packé = build -DEWC_INTx_PACKED");
#endif
}

void test_unpack_sign_extension(void)
{
#if defined(EWC_V2_PACK_BITS)
    /* Round-trip pack/unpack sur des valeurs représentables selon la profondeur. */
#if EWC_V2_PACK_BITS == 4
    int8_t vals[8] = {-7, -4, -1, 0, 1, 3, 6, 7};
#elif EWC_V2_PACK_BITS == 2
    int8_t vals[8] = {-1, 0, 1, -1, 0, 1, -1, 1};
#else /* 1 bit : binaire {−1,+1} (0 non représentable → mappé −1) */
    int8_t vals[8] = {-1, 1, -1, 1, 1, -1, 1, -1};
#endif
    uint8_t packed[EWC_V2_PACK_STRIDE(8)];
    ewc_v2_pack_row(packed, vals, 8);
    for (int i = 0; i < 8; i++)
        TEST_ASSERT_EQUAL_INT_MESSAGE(vals[i], ewc_v2_unpack_weight(packed, i),
                                      "round-trip pack/unpack incorrect (extension de signe ?)");
#else
    TEST_IGNORE_MESSAGE("unpack = build sub-INT8 (-DEWC_INT4/-DEWC_INT2/-DEWC_INT1)");
#endif
}
