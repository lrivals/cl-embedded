/**
 * ewc_head_multiclass.c — Tête MLP 3 couches EWC pour classification N classes sur MCU
 *
 * Forward : Linear(9→32)+ReLU → Linear(32→16)+ReLU → Linear(16→N_CLASSES)
 * Update  : SGD 1 step, softmax cross-entropy + terme EWC diagonal
 *           Gradient sortie : dL/dlogits[j] = softmax[j] - (j == label ? 1 : 0)
 *
 * Différences vs ewc_head.c (binaire) :
 *   - EWC_MC_N_CLASSES = N (paramétrable à la compilation via Makefile -D)
 *   - EWC_MC_IN = 9 (features CWRU) vs EWC_IN = 5 (features Monitoring)
 *   - ewc_mc_predict() reste argmax — généralisation triviale de la version binaire
 *
 * Backprop entièrement en stack local : pas de malloc.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : ewc_head.c (pattern), ewc_mlp_multiclass.py (spec Python)
 */

#include "ewc_head_multiclass.h"
#include <math.h>

/* ── Utilitaires ──────────────────────────────────────────────────────────── */

static float relu(float v) { return v > 0.0f ? v : 0.0f; }

/* ── Initialisation Xavier LCG ────────────────────────────────────────────── */

void ewc_mc_init(EWCHeadMC *h)
{
    uint32_t rng = 42u;
#define LCG_NEXT(r) ((r) = (r) * 1664525u + 1013904223u)
#define LCG_F01(r)  ((float)((r) >> 8) / (float)(1u << 24))

    /* Xavier limits : sqrt(6 / (fan_in + fan_out)) */
    static const float lim1 = 0.3780f;  /* sqrt(6/(9+32)) */
    static const float lim2 = 0.3536f;  /* sqrt(6/(32+16)) */
    /* lim3 dépend de N_CLASSES — calculé statiquement pour N=10 : sqrt(6/26) = 0.4804 */
    /* Pour N=3 : sqrt(6/19) = 0.5623 — différence < 20%, tolérable avec même seed */
    static const float lim3 = 0.4804f;  /* sqrt(6/(16+10)) — CWRU default */

    for (int j = 0; j < EWC_MC_H1; j++) {
        h->b1[j] = 0.0f;
        for (int i = 0; i < EWC_MC_IN; i++) {
            LCG_NEXT(rng);
            h->w1[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim1;
            h->fisher1[j][i] = 0.0f;
            h->star_w1[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        h->b2[j] = 0.0f;
        for (int i = 0; i < EWC_MC_H1; i++) {
            LCG_NEXT(rng);
            h->w2[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim2;
            h->fisher2[j][i] = 0.0f;
            h->star_w2[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        h->b3[j] = 0.0f;
        for (int i = 0; i < EWC_MC_H2; i++) {
            LCG_NEXT(rng);
            h->w3[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim3;
            h->fisher3[j][i] = 0.0f;
            h->star_w3[j][i] = 0.0f;
        }
    }
#undef LCG_NEXT
#undef LCG_F01
}

/* ── Forward pass ─────────────────────────────────────────────────────────── */

/* MEM: h1=128B, h2=64B (stack) + logits[N]=N×4B (fourni par appelant) */
void ewc_mc_forward(const EWCHeadMC *h, const float *x, float *logits)
{
    float h1[EWC_MC_H1];   /* MEM: 128 B @ FP32 */
    float h2[EWC_MC_H2];   /* MEM:  64 B @ FP32 */

    for (int j = 0; j < EWC_MC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_MC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_MC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_MC_H2; i++) acc += h->w3[j][i] * h2[i];
        logits[j] = acc;   /* logits bruts — softmax appliqué dans sgd_step */
    }
}

int ewc_mc_predict(const EWCHeadMC *h, const float *x)
{
    float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B @ FP32 — 40 B pour N=10 */
    ewc_mc_forward(h, x, logits);
    int best = 0;
    for (int j = 1; j < EWC_MC_N_CLASSES; j++) {
        if (logits[j] > logits[best]) best = j;
    }
    return best;
}

/* ── SGD step avec softmax CE + régularisation EWC ───────────────────────── */

/* MEM stack: h1(128B)+h2(64B)+logits(40B)+dout(40B)+dh2(64B)+dh1(128B) = 464 B pour N=10 */
void ewc_mc_sgd_step(EWCHeadMC *h, const float *x, int label)
{
    float h1[EWC_MC_H1];               /* MEM: 128 B @ FP32 */
    float h2[EWC_MC_H2];               /* MEM:  64 B @ FP32 */
    float logits[EWC_MC_N_CLASSES];    /* MEM: N×4 B — 40 B pour N=10 */

    /* ── 1. Forward ──────────────────────────────────────────────────────── */
    for (int j = 0; j < EWC_MC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_MC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_MC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_MC_H2; i++) acc += h->w3[j][i] * h2[i];
        logits[j] = acc;
    }

    /* ── 2. Softmax numériquement stable + gradient CE ───────────────────
     * dout[j] = softmax[j] - one_hot(label)[j]                          */
    float dout[EWC_MC_N_CLASSES];   /* MEM: N×4 B */
    float max_logit = logits[0];
    for (int j = 1; j < EWC_MC_N_CLASSES; j++)
        if (logits[j] > max_logit) max_logit = logits[j];

    float sum_exp = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        dout[j] = expf(logits[j] - max_logit);
        sum_exp += dout[j];
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        dout[j] = dout[j] / sum_exp - (j == label ? 1.0f : 0.0f);
    }

    /* ── 3. Backward couche 3 + EWC + SGD ─────────────────────────────── */
    float dh2[EWC_MC_H2];   /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_MC_H2; i++) dh2[i] = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        for (int i = 0; i < EWC_MC_H2; i++) {
            float grad = dout[j] * h2[i]
                       + h->lambda * h->fisher3[j][i] * (h->w3[j][i] - h->star_w3[j][i]);
            h->w3[j][i] -= EWC_MC_LR * grad;
            dh2[i] += h->w3[j][i] * dout[j];
        }
        h->b3[j] -= EWC_MC_LR * dout[j];
    }

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_MC_H2; i++)
        dh2[i] *= (h2[i] > 0.0f ? 1.0f : 0.0f);

    /* ── 4. Backward couche 2 + EWC + SGD ─────────────────────────────── */
    float dh1[EWC_MC_H1];   /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_MC_H1; i++) dh1[i] = 0.0f;
    for (int j = 0; j < EWC_MC_H2; j++) {
        for (int i = 0; i < EWC_MC_H1; i++) {
            float grad = dh2[j] * h1[i]
                       + h->lambda * h->fisher2[j][i] * (h->w2[j][i] - h->star_w2[j][i]);
            h->w2[j][i] -= EWC_MC_LR * grad;
            dh1[i] += h->w2[j][i] * dh2[j];
        }
        h->b2[j] -= EWC_MC_LR * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_MC_H1; i++)
        dh1[i] *= (h1[i] > 0.0f ? 1.0f : 0.0f);

    /* ── 5. Backward couche 1 + EWC + SGD ─────────────────────────────── */
    for (int j = 0; j < EWC_MC_H1; j++) {
        for (int i = 0; i < EWC_MC_IN; i++) {
            float grad = dh1[j] * x[i]
                       + h->lambda * h->fisher1[j][i] * (h->w1[j][i] - h->star_w1[j][i]);
            h->w1[j][i] -= EWC_MC_LR * grad;
        }
        h->b1[j] -= EWC_MC_LR * dh1[j];
    }
}

/* ── Consolidation EWC ────────────────────────────────────────────────────── */

void ewc_mc_consolidate(EWCHeadMC *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    for (int j = 0; j < EWC_MC_H1; j++)
        for (int i = 0; i < EWC_MC_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }

    for (int j = 0; j < EWC_MC_H2; j++)
        for (int i = 0; i < EWC_MC_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }

    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
}
