/**
 * tinyol_int8.c — TinyOL encodeur INT8 + tête OtO INT8 (Sprint 29, S2902)
 *
 * Encodeur : poids INT8 (fake-quant Q7, dequant FP32 au calcul), activations UINT8.
 * Tête OtO : Linear(EMB→1) + sigmoid, SGD BCE en ligne (poids maîtres FP32 + stockage INT8).
 *
 * Approche fake-quant cohérente avec ewc_head_int8.c et tinyol_int8.py.
 * Allocation 100 % statique (pas de malloc).
 *
 * Référence : Ren2021TinyOL, tinyol_int8.py, tinyol.c, ewc_head_int8.c
 */

#include "tinyol_int8.h"
#include "model_weights.h"
#include <string.h>
#include <math.h>   /* roundf, expf */

/* Saturation Q7 (int8) — identique à ewc_head_int8.c */
#define SAT8(x)  ((int8_t)((x) > 127 ? 127 : ((x) < -127 ? -127 : (x))))

static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }
static inline float q7_to_float(int8_t v) { return (float)v * TINYOL_INT8_SCALE_W; }

/* Quantifie une activation FP32 (≥0) en UINT8 [0,255] via TINYOL_INT8_ACT_SCALE. */
static inline uint8_t act_to_u8(float v)
{
    float q = roundf(v / TINYOL_INT8_ACT_SCALE);
    if (q < 0.0f)   q = 0.0f;
    if (q > 255.0f) q = 255.0f;
    return (uint8_t)q;
}
static inline float u8_to_act(uint8_t v) { return (float)v * TINYOL_INT8_ACT_SCALE; }

/* ── Initialisation encodeur depuis Flash FP32 ──────────────────────────── */

void tinyol_int8_init(TinyOLEncoderInt8 *enc)
{
    enc->scale_w = TINYOL_INT8_SCALE_W;

#if (TINYOL_IN == WEIGHTS_NATIVE_DIM)
    for (uint32_t j = 0; j < TINYOL_H1; j++) {
        enc->b_enc1[j] = TINYOL_B_ENC1[j];
        for (uint32_t i = 0; i < TINYOL_IN; i++) {
            enc->w_enc1[j][i] = SAT8((int)(TINYOL_W_ENC1[j][i] * 128.0f));
        }
    }
    for (uint32_t j = 0; j < TINYOL_EMB; j++) {
        enc->b_enc2[j] = TINYOL_B_ENC2[j];
        for (uint32_t i = 0; i < TINYOL_H1; i++) {
            enc->w_enc2[j][i] = SAT8((int)(TINYOL_W_ENC2[j][i] * 128.0f));
        }
    }
#else
    /* TINYOL_IN ≠ dim native : poids placeholder incopiables → zéro (S3506).
     * Poids réels par condition regénérés en S3507. */
    for (uint32_t j = 0; j < TINYOL_H1; j++) {
        enc->b_enc1[j] = 0.0f;
        for (uint32_t i = 0; i < TINYOL_IN; i++) enc->w_enc1[j][i] = 0;
    }
    for (uint32_t j = 0; j < TINYOL_EMB; j++) {
        enc->b_enc2[j] = 0.0f;
        for (uint32_t i = 0; i < TINYOL_H1; i++) enc->w_enc2[j][i] = 0;
    }
#endif
}

/* ── Forward encodeur INT8 → embedding UINT8 ────────────────────────────── */

void tinyol_int8_encode(const TinyOLEncoderInt8 *enc, const float *x, uint8_t *emb_u8)
{
    float h1[TINYOL_H1];   /* MEM: 128 B @ FP32 stack local */

    /* Couche 1 : Linear(IN→H1) + ReLU (poids dequantifiés) */
    for (uint32_t j = 0; j < TINYOL_H1; j++) {
        float acc = enc->b_enc1[j];
        for (uint32_t i = 0; i < TINYOL_IN; i++) {
            acc += q7_to_float(enc->w_enc1[j][i]) * x[i];
        }
        h1[j] = relu_f(acc);
    }

    /* Couche 2 : Linear(H1→EMB) + ReLU → quantif UINT8 */
    for (uint32_t j = 0; j < TINYOL_EMB; j++) {
        float acc = enc->b_enc2[j];
        for (uint32_t i = 0; i < TINYOL_H1; i++) {
            acc += q7_to_float(enc->w_enc2[j][i]) * h1[i];
        }
        emb_u8[j] = act_to_u8(relu_f(acc));
    }
}

/* ── Tête OtO INT8 ──────────────────────────────────────────────────────── */

void oto_int8_init(OtOHeadInt8 *oto)
{
    memset(oto, 0, sizeof(*oto));
    oto->lr = 0.01f;   /* TinyOL OtO SGD lr (cf. tinyol_int8.py learning_rate) */
}

/* Forward : sigmoid(w_master·emb + b_master). Stocke last_prob, retourne classe. */
int oto_int8_predict(OtOHeadInt8 *oto, const uint8_t *emb_u8)
{
    float logit = oto->b_master;
    for (uint32_t i = 0; i < TINYOL_EMB; i++) {
        logit += oto->w_master[i] * u8_to_act(emb_u8[i]);
    }
    float prob = 1.0f / (1.0f + expf(-logit));
    oto->last_prob = prob;
    return (prob >= 0.5f) ? 1 : 0;
}

/* 1 pas SGD BCE : grad = (sigmoid - y), maj poids maîtres FP32 puis re-quantif INT8. */
void oto_int8_update(OtOHeadInt8 *oto, const uint8_t *emb_u8, int y)
{
    float logit = oto->b_master;
    for (uint32_t i = 0; i < TINYOL_EMB; i++) {
        logit += oto->w_master[i] * u8_to_act(emb_u8[i]);
    }
    float prob = 1.0f / (1.0f + expf(-logit));
    float grad = prob - (float)y;   /* d(BCE)/d(logit) pour sigmoid */

    for (uint32_t i = 0; i < TINYOL_EMB; i++) {
        oto->w_master[i] -= oto->lr * grad * u8_to_act(emb_u8[i]);
        oto->w[i] = SAT8((int)(oto->w_master[i] * 128.0f));   /* re-quantif stockage INT8 */
    }
    oto->b_master -= oto->lr * grad;
    oto->b = oto->b_master;
}
