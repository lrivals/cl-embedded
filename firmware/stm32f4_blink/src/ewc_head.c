/**
 * ewc_head.c — Tête MLP 3 couches avec régularisation EWC sur MCU
 *
 * Forward : Linear(5→32)+ReLU → Linear(32→16)+ReLU → Linear(16→2)
 * Update  : SGD 1 step, softmax cross-entropy + terme EWC diagonal
 *
 * Backprop entièrement en stack local : pas de malloc.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : Kirkpatrick2017EWC (eq. 3), ewc_mlp.py
 */

#include "ewc_head.h"
#include <math.h>   /* expf */

/* ── Utilitaires locaux ─────────────────────────────────────────────────── */

static float relu(float v)
{
    return v > 0.0f ? v : 0.0f;
}

/* ── Initialisation Xavier LCG ─────────────────────────────────────────── */

/* MEM: ewc_init — 0 B stack extra, initialise EWCHead en place (poids Xavier,
 * Fisher et star_w remis à zéro). Ne touche pas h->lambda.
 * LCG Knuth : multiplicateur=1664525, incrément=1013904223 (Numerical Recipes). */
void ewc_init(EWCHead *h)
{
    uint32_t rng = 42u;
#define LCG_NEXT(r) ((r) = (r) * 1664525u + 1013904223u)
#define LCG_F01(r)  ((float)((r) >> 8) / (float)(1u << 24))   /* [0, 1) 24 bits */

    /* Xavier uniform — limit = sqrt(6 / (fan_in + fan_out)) */
    static const float lim1 = 0.4026f;   /* sqrt(6/(5+32))  */
    static const float lim2 = 0.3536f;   /* sqrt(6/(32+16)) */
    static const float lim3 = 0.5774f;   /* sqrt(6/(16+2))  */

    for (int j = 0; j < EWC_H1; j++) {
        h->b1[j] = 0.0f;
        for (int i = 0; i < EWC_IN; i++) {
            LCG_NEXT(rng);
            h->w1[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim1;
            h->fisher1[j][i] = 0.0f;
            h->star_w1[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_H2; j++) {
        h->b2[j] = 0.0f;
        for (int i = 0; i < EWC_H1; i++) {
            LCG_NEXT(rng);
            h->w2[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim2;
            h->fisher2[j][i] = 0.0f;
            h->star_w2[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_OUT; j++) {
        h->b3[j] = 0.0f;
        for (int i = 0; i < EWC_H2; i++) {
            LCG_NEXT(rng);
            h->w3[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim3;
            h->fisher3[j][i] = 0.0f;
            h->star_w3[j][i] = 0.0f;
        }
    }
#undef LCG_NEXT
#undef LCG_F01
}

/* ── Forward pass ──────────────────────────────────────────────────────── */

void ewc_forward(const EWCHead *h, const float *x, float *out)
{
    /* MEM: h1 = 128 B @ FP32, h2 = 64 B @ FP32 (stack local) */
    float h1[EWC_H1];
    float h2[EWC_H2];

    for (int j = 0; j < EWC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_IN; i++) {
            acc += h->w1[j][i] * x[i];
        }
        h1[j] = relu(acc);
    }

    for (int j = 0; j < EWC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_H1; i++) {
            acc += h->w2[j][i] * h1[i];
        }
        h2[j] = relu(acc);
    }

    for (int j = 0; j < EWC_OUT; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_H2; i++) {
            acc += h->w3[j][i] * h2[i];
        }
        out[j] = acc;  /* logits bruts */
    }
}

int ewc_predict(const EWCHead *h, const float *x)
{
    float logits[EWC_OUT];  /* MEM: 8 B @ FP32 */
    ewc_forward(h, x, logits);

    int best = 0;
    for (int j = 1; j < EWC_OUT; j++) {
        if (logits[j] > logits[best]) best = j;
    }
    return best;
}

/* ── SGD step avec régularisation EWC ─────────────────────────────────── */

void ewc_sgd_step(EWCHead *h, const float *x, int label)
{
    /* Activations forward — MEM: (128 + 64 + 8) B @ FP32 */
    float h1[EWC_H1];      /* MEM: 128 B @ FP32 */
    float h2[EWC_H2];      /* MEM:  64 B @ FP32 */
    float logits[EWC_OUT]; /* MEM:   8 B @ FP32 */

    /* ── 1. Forward ──────────────────────────────────────────────────── */
    for (int j = 0; j < EWC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    for (int j = 0; j < EWC_OUT; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_H2; i++) acc += h->w3[j][i] * h2[i];
        logits[j] = acc;
    }

    /* ── 2. Softmax + gradient sortie (CE loss) ───────────────────────
     * dL/dlogits[j] = softmax[j] - one_hot(label)[j]              */
    float dout[EWC_OUT];   /* MEM: 8 B @ FP32 */
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
        dout[j] = dout[j] / sum_exp - (j == label ? 1.0f : 0.0f);
    }

    /* ── 3. Backward couche 3 + EWC + SGD ───────────────────────────
     * dh2[i] = sum_j w3[j][i] * dout[j]                           */
    float dh2[EWC_H2];     /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_H2; i++) dh2[i] = 0.0f;

    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            float grad = dout[j] * h2[i]
                       + h->lambda * h->fisher3[j][i] * (h->w3[j][i] - h->star_w3[j][i]);
            h->w3[j][i] -= EWC_LR * grad;
            dh2[i] += h->w3[j][i] * dout[j];
        }
        h->b3[j] -= EWC_LR * dout[j];
    }

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_H2; i++) {
        dh2[i] *= (h2[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 4. Backward couche 2 + EWC + SGD ───────────────────────────
     * dh1[i] = sum_j w2[j][i] * dh2[j]                           */
    float dh1[EWC_H1];     /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_H1; i++) dh1[i] = 0.0f;

    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            float grad = dh2[j] * h1[i]
                       + h->lambda * h->fisher2[j][i] * (h->w2[j][i] - h->star_w2[j][i]);
            h->w2[j][i] -= EWC_LR * grad;
            dh1[i] += h->w2[j][i] * dh2[j];
        }
        h->b2[j] -= EWC_LR * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_H1; i++) {
        dh1[i] *= (h1[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 5. Backward couche 1 + EWC + SGD ──────────────────────────── */
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            float grad = dh1[j] * x[i]
                       + h->lambda * h->fisher1[j][i] * (h->w1[j][i] - h->star_w1[j][i]);
            h->w1[j][i] -= EWC_LR * grad;
        }
        h->b1[j] -= EWC_LR * dh1[j];
    }
}

/* ── Consolidation EWC : Fisher EMA + snapshot θ* ──────────────────────── */

/* MEM: ewc_consolidate — 0 B stack (tout in-place dans EWCHead en SRAM)
 * EWCHead total : ~9.5 Ko @ FP32 en .bss
 *   Poids courants : 3 Ko, Fisher diagonal : 3 Ko, θ* : 3 Ko, lambda : 4 B */
void ewc_consolidate(EWCHead *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    /* Couche 1 — grad² ≈ w² (proxy Fisher diagonal online, cf. Kirkpatrick2017EWC) */
    for (int j = 0; j < EWC_H1; j++) {
        for (int i = 0; i < EWC_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }
    }

    /* Couche 2 */
    for (int j = 0; j < EWC_H2; j++) {
        for (int i = 0; i < EWC_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }
    }

    /* Couche 3 — pas de Fisher sur les biais (standard EWC) */
    for (int j = 0; j < EWC_OUT; j++) {
        for (int i = 0; i < EWC_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
    }
}
