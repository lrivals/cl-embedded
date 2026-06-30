/**
 * ewc_head_int8.c — Tête MLP EWC quantifiée INT8 (Q7/Q15) pour Cortex-M4
 *
 * Forward  : activations Q7 (int8_t), accumulateurs Q15 (int16_t)
 * Update   : SGD 1 step hybride — gradients FP32, poids quantifiés Q7
 * Consolidation : Fisher EMA Q7 + snapshot θ* Q7
 *
 * Approche fake-quant : gradient calculé en FP32, appliqué sur weights Q7
 * via saturation. Cohérent avec ewc_mlp_int8.py (Gap 3).
 *
 * TODO(dorra) : biais restent FP32 (cohérence fake-quant Python). Valider
 *               si quantification Q15 des biais est nécessaire pour MCU final.
 * TODO(dorra) : décalage >> 7 assume |poids| < 1. Valider après calibration.
 * FIXME(gap3) : latence INT8 vs FP32 sur NUCLEO (DWT) réservée Sprint 23.
 *
 * Référence : Kirkpatrick2017EWC, Ravaglia2021QLRCL, ewc_head.c
 */

#include "ewc_head_int8.h"
#include <string.h>
#include <math.h>   /* expf */

/* Saturation sur int8 — évite UB sur cast hors plage */
#define SAT8(x)  ((int8_t)((x) > 127 ? 127 : ((x) < -127 ? -127 : (x))))

/* ── Initialisation ─────────────────────────────────────────────────────── */

void ewc_int8_init(EWCHeadInt8 *h)
{
    memset(h, 0, sizeof(*h));
    h->scale_w   = 1.0f / 128.0f;
    h->scale_act = 1.0f / 128.0f;
    h->lambda    = 0.0f;
    h->task_id   = 0;
}

/* ── Conversion depuis FP32 ─────────────────────────────────────────────── */

void ewc_int8_from_fp32(EWCHeadInt8 *dst, const EWCHead *src)
{
    dst->lambda    = src->lambda;
    dst->scale_w   = 1.0f / 128.0f;
    dst->scale_act = 1.0f / 128.0f;
    dst->task_id   = 0;

    for (int j = 0; j < EWC_H1; j++) {
        dst->b1[j] = src->b1[j];
        for (int i = 0; i < EWC_IN; i++) {
            dst->w1[j][i]      = SAT8((int)(src->w1[j][i]      * 128.0f));
            dst->fisher1[j][i] = SAT8((int)(src->fisher1[j][i] * 128.0f));
            dst->star_w1[j][i] = SAT8((int)(src->star_w1[j][i] * 128.0f));
        }
    }
    for (int j = 0; j < EWC_H2; j++) {
        dst->b2[j] = src->b2[j];
        for (int i = 0; i < EWC_H1; i++) {
            dst->w2[j][i]      = SAT8((int)(src->w2[j][i]      * 128.0f));
            dst->fisher2[j][i] = SAT8((int)(src->fisher2[j][i] * 128.0f));
            dst->star_w2[j][i] = SAT8((int)(src->star_w2[j][i] * 128.0f));
        }
    }
    for (int j = 0; j < EWC_OUT; j++) {
        dst->b3[j] = src->b3[j];
        for (int i = 0; i < EWC_H2; i++) {
            dst->w3[j][i]      = SAT8((int)(src->w3[j][i]      * 128.0f));
            dst->fisher3[j][i] = SAT8((int)(src->fisher3[j][i] * 128.0f));
            dst->star_w3[j][i] = SAT8((int)(src->star_w3[j][i] * 128.0f));
        }
    }
}

/* ── Forward pass Q7/Q15 ────────────────────────────────────────────────── */

/**
 * ewc_int8_forward — Forward pass Q7
 *
 * Calcule h1 = relu(w1·x + b1), h2 = relu(w2·h1 + b2), logits = w3·h2 + b3
 * en accumulateurs Q15 (int16_t) pour éviter l'overflow.
 *
 * MEM: stack local h1[32] = 32 B Q7, h2[16] = 16 B Q7, acc[EWC_H1] = 64 B Q15
 */
void ewc_int8_forward(const EWCHeadInt8 *h, const int8_t *x_q7, float *logits)
{
    int8_t  h1[EWC_H1];  /* MEM: 32 B @ INT8 */
    int8_t  h2[EWC_H2];  /* MEM: 16 B @ INT8 */

    /* Couche 1 : accumulation Q15, saturation → Q7 + ReLU */
    for (int j = 0; j < EWC_H1; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_IN; i++) {
            acc += (int16_t)h->w1[j][i] * (int16_t)x_q7[i];
        }
        /* Q7×Q7 = Q14, décalage >> 7 → Q7 */
        float val = (float)(acc >> 7) / 128.0f + h->b1[j];
        h1[j] = relu_q7(float_to_q7(val));
    }

    /* Couche 2 : même logique */
    for (int j = 0; j < EWC_H2; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_H1; i++) {
            acc += (int16_t)h->w2[j][i] * (int16_t)h1[i];
        }
        float val = (float)(acc >> 7) / 128.0f + h->b2[j];
        h2[j] = relu_q7(float_to_q7(val));
    }

    /* Couche 3 : sortie FP32 (logits) pour compatibilité softmax/sigmoid */
    for (int j = 0; j < EWC_OUT; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_H2; i++) {
            acc += (int16_t)h->w3[j][i] * (int16_t)h2[i];
        }
        logits[j] = (float)(acc >> 7) / 128.0f + h->b3[j];
    }
}

/* ── SGD step hybride fake-quant ────────────────────────────────────────── */

/**
 * ewc_int8_update — SGD 1 step avec régularisation EWC en Q7
 *
 * Gradients calculés en FP32 (fake-quant), appliqués sur poids Q7 via SAT8.
 * EWC term : lambda * F_q7 * (w_q7 - star_w_q7) converti en FP32.
 * Si fisher_ema != 0 : mise à jour Fisher EMA après le step.
 *
 * MEM: stack FP32 : h1_f[32] + h2_f[16] + logits[2] + dout[2] + dh2[16] + dh1[32]
 *      = (32+16+2+2+16+32)*4 = 400 B @ FP32
 */
void ewc_int8_update(EWCHeadInt8 *h, const int8_t *x_q7, uint8_t label,
                     float lr, int fisher_ema)
{
    /* Activations forward en FP32 pour backward exact */
    float h1_f[EWC_H1];   /* MEM: 128 B @ FP32 */
    float h2_f[EWC_H2];   /* MEM:  64 B @ FP32 */
    float logits[EWC_OUT]; /* MEM:   8 B @ FP32 */

    /* ── 1. Forward FP32 depuis poids Q7 ─────────────────────────────────
     * Dequantize poids à la volée pour le calcul de gradient exact.      */
    for (int j = 0; j < EWC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_IN; i++) {
            acc += q7_to_float(h->w1[j][i]) * q7_to_float(x_q7[i]);
        }
        h1_f[j] = acc > 0.0f ? acc : 0.0f;
    }
    for (int j = 0; j < EWC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_H1; i++) {
            acc += q7_to_float(h->w2[j][i]) * h1_f[i];
        }
        h2_f[j] = acc > 0.0f ? acc : 0.0f;
    }
    for (int j = 0; j < EWC_OUT; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_H2; i++) {
            acc += q7_to_float(h->w3[j][i]) * h2_f[i];
        }
        logits[j] = acc;
    }

    /* ── 2. Softmax + gradient sortie (CE loss) ───────────────────────── */
    float dout[EWC_OUT];  /* MEM: 8 B @ FP32 */
    float max_logit = logits[0];
    for (int j = 1; j < EWC_OUT; j++) {
        if (logits[j] > max_logit) max_logit = logits[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < EWC_OUT; j++) {
        dout[j] = expf(logits[j] - max_logit);
        sum_exp += dout[j];
    }
    for (int j = 0; j < EWC_OUT; j++) {
        dout[j] = dout[j] / sum_exp - (j == (int)label ? 1.0f : 0.0f);
    }

    /* ── 3. Backward couche 3 + EWC + update Q7 ──────────────────────── */
    float dh2[EWC_H2];    /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_H2; i++) dh2[i] = 0.0f;

    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            float ewc_term = h->lambda
                           * q7_to_float(h->fisher3[j][i])
                           * (q7_to_float(h->w3[j][i]) - q7_to_float(h->star_w3[j][i]));
            float grad = dout[j] * h2_f[i] + ewc_term;
            int delta = (int)(lr * grad * 128.0f);
            h->w3[j][i] = SAT8((int)h->w3[j][i] - delta);
            dh2[i] += q7_to_float(h->w3[j][i]) * dout[j];
        }
        h->b3[j] -= lr * dout[j];
    }

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_H2; i++) {
        dh2[i] *= (h2_f[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 4. Backward couche 2 + EWC + update Q7 ──────────────────────── */
    float dh1[EWC_H1];    /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_H1; i++) dh1[i] = 0.0f;

    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            float ewc_term = h->lambda
                           * q7_to_float(h->fisher2[j][i])
                           * (q7_to_float(h->w2[j][i]) - q7_to_float(h->star_w2[j][i]));
            float grad = dh2[j] * h1_f[i] + ewc_term;
            int delta = (int)(lr * grad * 128.0f);
            h->w2[j][i] = SAT8((int)h->w2[j][i] - delta);
            dh1[i] += q7_to_float(h->w2[j][i]) * dh2[j];
        }
        h->b2[j] -= lr * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_H1; i++) {
        dh1[i] *= (h1_f[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 5. Backward couche 1 + EWC + update Q7 ──────────────────────── */
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            float ewc_term = h->lambda
                           * q7_to_float(h->fisher1[j][i])
                           * (q7_to_float(h->w1[j][i]) - q7_to_float(h->star_w1[j][i]));
            float grad = dh1[j] * q7_to_float(x_q7[i]) + ewc_term;
            int delta = (int)(lr * grad * 128.0f);
            h->w1[j][i] = SAT8((int)h->w1[j][i] - delta);
        }
        h->b1[j] -= lr * dh1[j];
    }

    /* ── 6. Fisher EMA optionnelle ────────────────────────────────────── */
    if (fisher_ema) {
        for (int j = 0; j < EWC_H1; j++) {
            for (int i = 0; i < EWC_IN; i++) {
                float w_f   = q7_to_float(h->w1[j][i]);
                float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher1[j][i])
                            + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
                h->fisher1[j][i] = float_to_q7(f_new);
            }
        }
        for (int j = 0; j < EWC_H2; j++) {
            for (int i = 0; i < EWC_H1; i++) {
                float w_f   = q7_to_float(h->w2[j][i]);
                float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher2[j][i])
                            + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
                h->fisher2[j][i] = float_to_q7(f_new);
            }
        }
        for (int j = 0; j < EWC_OUT; j++) {
            for (int i = 0; i < EWC_H2; i++) {
                float w_f   = q7_to_float(h->w3[j][i]);
                float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher3[j][i])
                            + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
                h->fisher3[j][i] = float_to_q7(f_new);
            }
        }
    }
}

/* ── Consolidation EWC : snapshot θ* + Fisher EMA ──────────────────────── */

/* MEM: ewc_int8_consolidate — 0 B stack (tout in-place dans EWCHeadInt8) */
void ewc_int8_consolidate(EWCHeadInt8 *h)
{
    /* Couche 1 */
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            float w_f   = q7_to_float(h->w1[j][i]);
            float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher1[j][i])
                        + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
            h->fisher1[j][i] = float_to_q7(f_new);
            h->star_w1[j][i] = h->w1[j][i];
        }
    }
    /* Couche 2 */
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            float w_f   = q7_to_float(h->w2[j][i]);
            float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher2[j][i])
                        + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
            h->fisher2[j][i] = float_to_q7(f_new);
            h->star_w2[j][i] = h->w2[j][i];
        }
    }
    /* Couche 3 */
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            float w_f   = q7_to_float(h->w3[j][i]);
            float f_new = EWC_FISHER_DECAY * q7_to_float(h->fisher3[j][i])
                        + (1.0f - EWC_FISHER_DECAY) * w_f * w_f;
            h->fisher3[j][i] = float_to_q7(f_new);
            h->star_w3[j][i] = h->w3[j][i];
        }
    }
}
