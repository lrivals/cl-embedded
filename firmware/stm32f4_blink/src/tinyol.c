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
