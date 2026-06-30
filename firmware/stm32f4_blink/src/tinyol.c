/**
 * tinyol.c — TinyOL encoder/decoder C skeleton (forward pass NPU-simulated)
 *
 * Implémente uniquement le forward pass (inférence) :
 *   - tinyol_encode : x → embedding (simule NPU NeuralART Turbo)
 *   - tinyol_decode : embedding → reconstruction
 *   - tinyol_reconstruction_error : MSE(x, recon)
 *   - tinyol_predict : score > threshold → anomalie
 *
 * La tête OtO (mise à jour incrémentale) sera portée séparément via la
 * backprop Cortex-M55 SW dans un sprint suivant.
 *
 * Pas de malloc, pas de stdlib dynamique.
 * Référence : Ren2021TinyOL, src/models/tinyol_autoencoder.py
 */

#include "tinyol.h"
#include "model_weights.h"
#include <string.h>

/* ReLU scalaire */
static inline float relu_f(float v)
{
    return v > 0.0f ? v : 0.0f;
}

/* ── Encoder forward ────────────────────────────────────────────────────── */

void tinyol_encode(const TinyOLEncoder *enc, const float *x, float *emb)
{
    float h1[TINYOL_H1];  /* MEM: 128 B @ FP32 stack local */

    /* Couche 1 : Linear(n_in→H1) + ReLU */
    for (uint32_t j = 0; j < TINYOL_H1; j++) {
        float acc = enc->b_enc1[j];
        for (uint32_t i = 0; i < TINYOL_IN; i++) {
            acc += enc->w_enc1[j][i] * x[i];
        }
        h1[j] = relu_f(acc);
    }

    /* Couche 2 : Linear(H1→EMB) + ReLU */
    for (uint32_t j = 0; j < TINYOL_EMB; j++) {
        float acc = enc->b_enc2[j];
        for (uint32_t i = 0; i < TINYOL_H1; i++) {
            acc += enc->w_enc2[j][i] * h1[i];
        }
        emb[j] = relu_f(acc);
    }
}

/* ── Decoder forward ────────────────────────────────────────────────────── */

void tinyol_decode(const TinyOLDecoder *dec, const float *emb, float *recon)
{
    float h1[TINYOL_H1];  /* MEM: 128 B @ FP32 stack local */

    /* Couche 1 : Linear(EMB→H1) + ReLU */
    for (uint32_t j = 0; j < TINYOL_H1; j++) {
        float acc = dec->b_dec1[j];
        for (uint32_t i = 0; i < TINYOL_EMB; i++) {
            acc += dec->w_dec1[j][i] * emb[i];
        }
        h1[j] = relu_f(acc);
    }

    /* Couche 2 : Linear(H1→OUT), pas d'activation */
    for (uint32_t j = 0; j < TINYOL_OUT; j++) {
        float acc = dec->b_dec2[j];
        for (uint32_t i = 0; i < TINYOL_H1; i++) {
            acc += dec->w_dec2[j][i] * h1[i];
        }
        recon[j] = acc;
    }
}

/* ── Score de reconstruction (MSE) ─────────────────────────────────────── */

float tinyol_reconstruction_error(const float *x, const float *recon, int n)
{
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - recon[i];
        mse += diff * diff;
    }
    return n > 0 ? mse / (float)n : 0.0f;
}

/* ── Prédiction anomalie ────────────────────────────────────────────────── */

int tinyol_predict(const TinyOLEncoder *enc, const TinyOLDecoder *dec,
                   const float *x, float threshold)
{
    float emb[TINYOL_EMB];            /* MEM:  64 B @ FP32 stack local */
    float recon[TINYOL_OUT];          /* MEM:  20 B @ FP32 stack local */

    tinyol_encode(enc, x, emb);
    tinyol_decode(dec, emb, recon);
    float err = tinyol_reconstruction_error(x, recon, (int)TINYOL_OUT);
    return err > threshold ? 1 : 0;
}

/* ── Initialisation depuis Flash (model_weights.h) ──────────────────────── */

/* MEM: TinyOLEncoder+Decoder poids en Flash (const) — ~5.6 Ko @ FP32
 * enc_w1[32][5], enc_b1[32], enc_w2[16][32], enc_b2[16]
 * dec_w1[32][16], dec_b1[32], dec_w2[5][32], dec_b2[5] */
void tinyol_init(TinyOLEncoder *enc, TinyOLDecoder *dec)
{
#if (TINYOL_IN == WEIGHTS_NATIVE_DIM)
    memcpy(enc->w_enc1, TINYOL_W_ENC1, sizeof(enc->w_enc1));
    memcpy(enc->b_enc1, TINYOL_B_ENC1, sizeof(enc->b_enc1));
    memcpy(enc->w_enc2, TINYOL_W_ENC2, sizeof(enc->w_enc2));
    memcpy(enc->b_enc2, TINYOL_B_ENC2, sizeof(enc->b_enc2));
    memcpy(dec->w_dec1, TINYOL_W_DEC1, sizeof(dec->w_dec1));
    memcpy(dec->b_dec1, TINYOL_B_DEC1, sizeof(dec->b_dec1));
    memcpy(dec->w_dec2, TINYOL_W_DEC2, sizeof(dec->w_dec2));
    memcpy(dec->b_dec2, TINYOL_B_DEC2, sizeof(dec->b_dec2));
#else
    /* TINYOL_IN ≠ dim native : poids placeholder incopiables → zéro (S3506).
     * Poids réels par condition regénérés en S3507. */
    memset(enc, 0, sizeof(*enc));
    memset(dec, 0, sizeof(*dec));
#endif
}
