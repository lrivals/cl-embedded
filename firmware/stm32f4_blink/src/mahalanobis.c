/**
 * mahalanobis.c — Détecteur d'anomalies Mahalanobis pour MCU
 *
 * Inférence : d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
 * Update online : EMA sur μ uniquement (Σ⁻¹ figée en Flash après fit offline)
 *
 * Contraintes : pas de malloc, pas de stdlib, FPU FP32 uniquement.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 */

#include "mahalanobis.h"

/* VSQRT.F32 directement sur cible ARM FPU ; sqrtf() standard sur host x86 */
static inline float fpu_sqrtf(float x)
{
#ifdef __arm__
    float r;
    __asm volatile("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x));
    return r;
#else
    extern float sqrtf(float);
    return sqrtf(x);
#endif
}

void maha_init(MahalanobisDetector *det, float threshold, float ema_alpha)
{
    det->threshold = threshold;
    det->ema_alpha = ema_alpha;

    for (int i = 0; i < MAHA_DIM; i++) {
        det->mean[i] = 0.0f;
        for (int j = 0; j < MAHA_DIM; j++) {
            det->precision[i][j] = (i == j) ? 1.0f : 0.0f;  /* identité par défaut */
        }
    }
}

float maha_score(const MahalanobisDetector *det, const float *x)
{
    float diff[MAHA_DIM];                  /* MEM: 20 B @ FP32 */
    float left[MAHA_DIM];                  /* MEM: 20 B @ FP32 — precision @ diff */

    for (int i = 0; i < MAHA_DIM; i++) {
        diff[i] = x[i] - det->mean[i];
    }

    for (int i = 0; i < MAHA_DIM; i++) {
        float acc = 0.0f;
        for (int j = 0; j < MAHA_DIM; j++) {
            acc += det->precision[i][j] * diff[j];
        }
        left[i] = acc;
    }

    float dist_sq = 0.0f;
    for (int i = 0; i < MAHA_DIM; i++) {
        dist_sq += left[i] * diff[i];
    }

    return fpu_sqrtf(dist_sq > 0.0f ? dist_sq : 0.0f);
}

/* EMA sur mean uniquement — Σ⁻¹ reste figée (calculée offline en Python) */
void maha_update(MahalanobisDetector *det, const float *x)
{
    float alpha = det->ema_alpha;
    float one_minus = 1.0f - alpha;
    for (int i = 0; i < MAHA_DIM; i++) {
        det->mean[i] = one_minus * det->mean[i] + alpha * x[i];
    }
}

int maha_predict(const MahalanobisDetector *det, const float *x)
{
    return maha_score(det, x) > det->threshold ? 1 : 0;
}
