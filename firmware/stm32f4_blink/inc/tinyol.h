#pragma once
#include <stdint.h>

/* ── TinyOL encoder C skeleton : forward pass (NPU-simulated) ─────────────
 *
 * Architecture Phase 1 (tinyol_autoencoder.py) :
 *   Encoder : Linear(n_in→32) + ReLU → Linear(32→16) + ReLU → embedding(16)
 *   Decoder : Linear(16→32) + ReLU → Linear(32→n_in)       [inference-only]
 *
 * Sur STM32N6 : l'encoder tourne sur NeuralART Turbo (NPU, inférence uniquement).
 * Sur NUCLEO-F439ZI : simulation soft de l'inférence NPU (Cortex-M4 FPU).
 *
 * La tête OtO (One-Task-at-a-time) pour la mise à jour incrémentale est
 * implémentée séparément et tourne sur Cortex-M55 (backprop SW).
 *
 * Référence : Ren2021TinyOL, tinyol_spec.md
 */

#define TINYOL_IN     5U    /* Features d'entrée (Monitoring / CWRU) */
#define TINYOL_H1    32U    /* Couche cachée 1 */
#define TINYOL_EMB   16U    /* Dimension embedding */
#define TINYOL_OUT    5U    /* Sortie décodeur (reconstruction) */

/* MEM: encoder = (5×32+32 + 32×16+16) × 4 = ~3.1 Ko @ FP32 en Flash */
typedef struct {
    float w_enc1[TINYOL_H1][TINYOL_IN];   /* MEM: 640 B @ FP32 — Flash */
    float b_enc1[TINYOL_H1];              /* MEM: 128 B @ FP32 — Flash */
    float w_enc2[TINYOL_EMB][TINYOL_H1];  /* MEM: 2048 B @ FP32 — Flash */
    float b_enc2[TINYOL_EMB];             /* MEM:  64 B @ FP32 — Flash */
} TinyOLEncoder;

/* MEM: (16×32+32 + 32×5+5) × 4 = ~2.8 Ko @ FP32 en Flash */
typedef struct {
    float w_dec1[TINYOL_H1][TINYOL_EMB];  /* MEM: 2048 B @ FP32 — Flash */
    float b_dec1[TINYOL_H1];              /* MEM:  128 B @ FP32 — Flash */
    float w_dec2[TINYOL_OUT][TINYOL_H1];  /* MEM:  640 B @ FP32 — Flash */
    float b_dec2[TINYOL_OUT];             /* MEM:   20 B @ FP32 — Flash */
} TinyOLDecoder;

/* Encoder forward : x[TINYOL_IN] → emb[TINYOL_EMB]
 * Activations intermédiaires sur stack local (pas de malloc).
 * MEM stack : h1 = 128 B @ FP32
 */
void tinyol_encode(const TinyOLEncoder *enc, const float *x, float *emb);

/* Decoder forward : emb[TINYOL_EMB] → recon[TINYOL_OUT]
 * MEM stack : h1 = 128 B @ FP32
 */
void tinyol_decode(const TinyOLDecoder *dec, const float *emb, float *recon);

/* Score de reconstruction (MSE entre x et recon) — utilisé comme score d'anomalie */
float tinyol_reconstruction_error(const float *x, const float *recon, int n);

/* Prédiction anomalie : 1 si MSE > threshold */
int tinyol_predict(const TinyOLEncoder *enc, const TinyOLDecoder *dec,
                   const float *x, float threshold);

/* Initialise enc et dec depuis les constantes Flash de model_weights.h */
void tinyol_init(TinyOLEncoder *enc, TinyOLDecoder *dec);
