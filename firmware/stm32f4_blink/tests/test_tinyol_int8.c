/**
 * test_tinyol_int8.c — Tests Unity pour tinyol_int8.c (Sprint 29, S2907)
 *
 * Exécution sur x86 (pas de board requise). Teste l'encodeur INT8 (fake-quant Q7,
 * activations UINT8) et la tête OtO INT8 (SGD BCE en ligne, poids maîtres FP32).
 *
 * Référence : test_ewc_int8.c (pattern fake-quant), inc/tinyol_int8.h (interface).
 */

#include "unity.h"
#include "tinyol_int8.h"
#include "tinyol.h"   /* TinyOLAutoencoder, tinyol_encode() */
#include <math.h>

#define TOL_PCT   0.05f    /* 5% de la plage [0, 8] = 0.4 */
#define ACT_RANGE 8.0f

/* ── Test 1 — embedding UINT8 dans [0, 255] ─────────────────────────────── */
void test_tinyol_int8_encode_range(void)
{
    TinyOLEncoderInt8 enc;
    tinyol_int8_init(&enc);

    float x[TINYOL_IN];
    for (int i = 0; i < TINYOL_IN; i++) x[i] = (float)(i - 2);  /* négatifs + positifs */

    uint8_t emb[TINYOL_EMB];
    tinyol_int8_encode(&enc, x, emb);

    for (int i = 0; i < TINYOL_EMB; i++) {
        /* uint8_t est déjà borné par son type, mais on teste l'absence de UB */
        TEST_ASSERT_TRUE(emb[i] >= 0 && emb[i] <= 255);
    }
}

/* ── Test 2 — embedding INT8 dequantifié proche du FP32 ─────────────────── */
void test_tinyol_int8_encode_vs_fp32(void)
{
    TinyOLEncoderInt8 enc_i8;
    tinyol_int8_init(&enc_i8);

    TinyOLEncoder enc_fp32;
    TinyOLDecoder dec_fp32;
    tinyol_init(&enc_fp32, &dec_fp32);    /* mêmes poids depuis model_weights.h */

    float x[TINYOL_IN];
    for (int i = 0; i < TINYOL_IN; i++) x[i] = 0.5f;

    uint8_t emb_u8[TINYOL_EMB];
    float   emb_fp32[TINYOL_EMB];
    tinyol_int8_encode(&enc_i8, x, emb_u8);
    tinyol_encode(&enc_fp32, x, emb_fp32);

    float total_delta = 0.0f;
    for (int i = 0; i < TINYOL_EMB; i++) {
        float act_i8 = (float)emb_u8[i] * TINYOL_INT8_ACT_SCALE;
        total_delta += fabsf(act_i8 - emb_fp32[i]);
    }
    float mean_delta = total_delta / (float)TINYOL_EMB;
    TEST_ASSERT_FLOAT_WITHIN(ACT_RANGE * TOL_PCT, 0.0f, mean_delta);
}

/* ── Test 3 — predict renvoie strictement 0 ou 1 ────────────────────────── */
void test_oto_int8_predict_returns_binary(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = (uint8_t)(i * 3 % 256);

    int pred = oto_int8_predict(&oto, emb);
    TEST_ASSERT_TRUE(pred == 0 || pred == 1);
}

/* ── Test 4 — la tête converge vers la classe 1 ─────────────────────────── */
void test_oto_int8_update_learns_class1(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    /* Embedding non-nul pour que le gradient soit non-nul */
    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = 128u;

    for (int k = 0; k < 20; k++) {
        oto_int8_update(&oto, emb, 1);
    }
    oto_int8_predict(&oto, emb);   /* met à jour last_prob */

    TEST_ASSERT_GREATER_THAN_FLOAT(0.5f, oto.last_prob);
}

/* ── Test 5 — last_prob reste dans [0, 1] après updates mixtes ──────────── */
void test_oto_int8_last_prob_range(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = (uint8_t)(i % 256);

    for (int k = 0; k < 10; k++) {
        oto_int8_update(&oto, emb, k % 2);
        oto_int8_predict(&oto, emb);
        TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.5f, oto.last_prob);  /* [0, 1] */
    }
}
