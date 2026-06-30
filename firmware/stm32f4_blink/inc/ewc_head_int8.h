/**
 * ewc_head_int8.h — Tête MLP EWC quantifiée INT8 (Q7/Q15) pour MCU
 *
 * Forward  : activations Q7 (int8_t), accumulateurs Q15 (int16_t)
 * Update   : SGD 1 step en Q7, terme EWC diagonal Q7
 * Pas de malloc — tout alloué statiquement dans EWCHeadInt8.
 *
 * Validation board : Sprint 23 (S2307).
 * Référence Python : src/models/ewc/ewc_mlp_int8.py
 */

#ifndef EWC_HEAD_INT8_H
#define EWC_HEAD_INT8_H

#include <stdint.h>
#include "ewc_head.h"   /* réutilise EWC_IN, EWC_H1, EWC_H2, EWC_OUT */

/* Constantes de quantization (calibrées sur données d'entraînement) */
#define INT8_SCALE_W1    (1.0f / 128.0f)   /* scale poids couche 1 */
#define INT8_SCALE_W2    (1.0f / 128.0f)   /* scale poids couche 2 */
#define INT8_SCALE_W3    (1.0f / 128.0f)   /* scale poids couche 3 */
#define INT8_SCALE_ACT   (1.0f / 128.0f)   /* scale activations post-ReLU */

/* Structure INT8 — mêmes dimensions que EWCHead mais types entiers */
/* MEM: EWCHeadInt8 = ~2.4 Ko @ INT8 (vs ~9.7 Ko @ FP32 pour EWCHead)
 *   w1 = 5×32 = 160 B, w2 = 32×16 = 512 B, w3 = 16×2 = 32 B
 *   b1 = 32×4 B = 128 B (biais restent FP32), b2/b3 idem
 *   fisher1..3 = INT8 × mêmes tailles = 704 B
 *   star_w1..3 = INT8 × mêmes tailles = 704 B          */
typedef struct {
    int8_t  w1[EWC_H1][EWC_IN];
    float   b1[EWC_H1];             /* biais FP32 — impact mémoire faible */
    int8_t  w2[EWC_H2][EWC_H1];
    float   b2[EWC_H2];
    int8_t  w3[EWC_OUT][EWC_H2];
    float   b3[EWC_OUT];

    int8_t  fisher1[EWC_H1][EWC_IN];
    int8_t  fisher2[EWC_H2][EWC_H1];
    int8_t  fisher3[EWC_OUT][EWC_H2];

    int8_t  star_w1[EWC_H1][EWC_IN];
    int8_t  star_w2[EWC_H2][EWC_H1];
    int8_t  star_w3[EWC_OUT][EWC_H2];

    float   lambda;
    float   scale_w;   /* scale commune poids */
    float   scale_act; /* scale activations */
    uint8_t task_id;
} EWCHeadInt8;

/* API publique */
void ewc_int8_init(EWCHeadInt8 *h);
void ewc_int8_from_fp32(EWCHeadInt8 *dst, const EWCHead *src_fp32);
void ewc_int8_forward(const EWCHeadInt8 *h, const int8_t *x_q7, float *logits);
void ewc_int8_update(EWCHeadInt8 *h, const int8_t *x_q7, uint8_t label,
                     float lr, int fisher_ema);
void ewc_int8_consolidate(EWCHeadInt8 *h);

/* Utilitaires Q7 */
static inline int8_t  float_to_q7(float v) { return (int8_t)(v * 128.0f); }
static inline float   q7_to_float(int8_t v) { return (float)v / 128.0f; }
static inline int8_t  relu_q7(int8_t v) { return v > 0 ? v : 0; }

#endif /* EWC_HEAD_INT8_H */
