/**
 * ewc_head_regression.c — Tête MLP 3 couches EWC pour régression RUL sur MCU
 *
 * Forward : Linear(5→32)+ReLU → Linear(32→16)+ReLU → Linear(16→1) [linéaire]
 * Update  : SGD 1 step, perte MSE + terme EWC diagonal
 *           Gradient sortie : dL/d(out) = out[0] - y_true   (MSE, pas de softmax)
 *
 * Différences vs ewc_head.c :
 *   - EWC_REG_OUT = 1  (au lieu de 2)
 *   - Pas de softmax dans sgd_step — gradient direct = ŷ - y
 *   - ewc_reg_predict() retourne un float (RUL), pas un int
 *
 * Backprop entièrement en stack local : pas de malloc.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : ewc_head.c (pattern), ewc_mlp_regression.py (spec Python)
 */

#include "ewc_head_regression.h"
#include <math.h>

/* ── Utilitaires locaux ───────────────────────────────────────────────────── */

static float relu(float v) { return v > 0.0f ? v : 0.0f; }

/* ── Initialisation Xavier LCG ────────────────────────────────────────────── */

/* MEM: ewc_reg_init — 0 B stack extra.
 * LCG Knuth (même seed=42 que ewc_head.c — valeurs Xavier différentes car EWC_REG_* dims) */
void ewc_reg_init(EWCHeadReg *h)
{
    uint32_t rng = 42u;
#define LCG_NEXT(r) ((r) = (r) * 1664525u + 1013904223u)
#define LCG_F01(r)  ((float)((r) >> 8) / (float)(1u << 24))

    /* Xavier uniform — limit = sqrt(6 / (fan_in + fan_out)) */
    static const float lim1 = 0.4026f;   /* sqrt(6/(5+32))  */
    static const float lim2 = 0.3536f;   /* sqrt(6/(32+16)) */
    static const float lim3 = 0.6667f;   /* sqrt(6/(16+1))  */

    for (int j = 0; j < EWC_REG_H1; j++) {
        h->b1[j] = 0.0f;
        for (int i = 0; i < EWC_REG_IN; i++) {
            LCG_NEXT(rng);
            h->w1[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim1;
            h->fisher1[j][i] = 0.0f;
            h->star_w1[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        h->b2[j] = 0.0f;
        for (int i = 0; i < EWC_REG_H1; i++) {
            LCG_NEXT(rng);
            h->w2[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim2;
            h->fisher2[j][i] = 0.0f;
            h->star_w2[j][i] = 0.0f;
        }
    }
    /* Couche de sortie : EWC_REG_OUT = 1 neurone */
    for (int j = 0; j < EWC_REG_OUT; j++) {
        h->b3[j] = 0.0f;
        for (int i = 0; i < EWC_REG_H2; i++) {
            LCG_NEXT(rng);
            h->w3[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim3;
            h->fisher3[j][i] = 0.0f;
            h->star_w3[j][i] = 0.0f;
        }
    }
    h->lambda = 0.0f;
#undef LCG_NEXT
#undef LCG_F01
}

/* ── Forward pass ─────────────────────────────────────────────────────────── */

/* MEM: h1 = 128 B, h2 = 64 B (stack local) */
void ewc_reg_forward(const EWCHeadReg *h, const float *x, float *out)
{
    float h1[EWC_REG_H1];  /* MEM: 128 B @ FP32 */
    float h2[EWC_REG_H2];  /* MEM:  64 B @ FP32 */

    for (int j = 0; j < EWC_REG_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_REG_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_REG_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    /* Couche de sortie — activation linéaire (pas de Sigmoid, pas de softmax) */
    float acc = h->b3[0];
    for (int i = 0; i < EWC_REG_H2; i++) acc += h->w3[0][i] * h2[i];
    out[0] = acc;   /* RUL prédit, scalaire non borné */
}

float ewc_reg_predict(const EWCHeadReg *h, const float *x)
{
    float out[EWC_REG_OUT];  /* MEM: 4 B @ FP32 */
    ewc_reg_forward(h, x, out);
    return out[0];
}

/* ── SGD step avec perte MSE + régularisation EWC ─────────────────────────── */

/* MEM stack: h1(128B) + h2(64B) + dh2(64B) + dh1(128B) = 384 B */
void ewc_reg_sgd_step(EWCHeadReg *h, const float *x, float y_true)
{
    float h1[EWC_REG_H1];   /* MEM: 128 B @ FP32 */
    float h2[EWC_REG_H2];   /* MEM:  64 B @ FP32 */
    float out[EWC_REG_OUT]; /* MEM:   4 B @ FP32 */

    /* ── 1. Forward ──────────────────────────────────────────────────────── */
    for (int j = 0; j < EWC_REG_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_REG_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_REG_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    {
        float acc = h->b3[0];
        for (int i = 0; i < EWC_REG_H2; i++) acc += h->w3[0][i] * h2[i];
        out[0] = acc;
    }

    /* ── 2. Gradient MSE sur la sortie ───────────────────────────────────
     * L = (ŷ - y)² / 2  →  dL/dŷ = ŷ - y                             */
    float dout = out[0] - y_true;  /* scalaire — 1 neurone de sortie */

    /* ── 3. Backward couche 3 + EWC + SGD ───────────────────────────────
     * dh2[i] = w3[0][i] * dout                                         */
    float dh2[EWC_REG_H2];   /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_REG_H2; i++) {
        float grad = dout * h2[i]
                   + h->lambda * h->fisher3[0][i] * (h->w3[0][i] - h->star_w3[0][i]);
        h->w3[0][i] -= EWC_REG_LR * grad;
        dh2[i] = h->w3[0][i] * dout;
    }
    h->b3[0] -= EWC_REG_LR * dout;

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_REG_H2; i++) {
        dh2[i] *= (h2[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 4. Backward couche 2 + EWC + SGD ─────────────────────────────── */
    float dh1[EWC_REG_H1];   /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_REG_H1; i++) dh1[i] = 0.0f;
    for (int j = 0; j < EWC_REG_H2; j++) {
        for (int i = 0; i < EWC_REG_H1; i++) {
            float grad = dh2[j] * h1[i]
                       + h->lambda * h->fisher2[j][i] * (h->w2[j][i] - h->star_w2[j][i]);
            h->w2[j][i] -= EWC_REG_LR * grad;
            dh1[i] += h->w2[j][i] * dh2[j];
        }
        h->b2[j] -= EWC_REG_LR * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_REG_H1; i++) {
        dh1[i] *= (h1[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 5. Backward couche 1 + EWC + SGD ─────────────────────────────── */
    for (int j = 0; j < EWC_REG_H1; j++) {
        for (int i = 0; i < EWC_REG_IN; i++) {
            float grad = dh1[j] * x[i]
                       + h->lambda * h->fisher1[j][i] * (h->w1[j][i] - h->star_w1[j][i]);
            h->w1[j][i] -= EWC_REG_LR * grad;
        }
        h->b1[j] -= EWC_REG_LR * dh1[j];
    }
}

/* ── Consolidation EWC : Fisher EMA + snapshot θ* ────────────────────────── */

/* Identique à ewc_consolidate() — seules les dimensions diffèrent */
void ewc_reg_consolidate(EWCHeadReg *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }

    for (int j = 0; j < EWC_REG_H2; j++)
        for (int i = 0; i < EWC_REG_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }

    for (int j = 0; j < EWC_REG_OUT; j++)
        for (int i = 0; i < EWC_REG_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
}
