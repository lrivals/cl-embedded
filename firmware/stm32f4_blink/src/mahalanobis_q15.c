/**
 * mahalanobis_q15.c — Détecteur de Mahalanobis sigma_inv int16 Q15 pour MCU (Sprint 34, S3407).
 *
 * Déquantification (mu INT8 affine + sigma_inv int16 Q15) → distance en FP32 sur la FPU.
 * Bit-identique à MahalanobisDetectorInt8.anomaly_score_q15 (Python, S3405).
 *
 * Contraintes : pas de malloc, pas de stdlib lourde, FPU FP32 (Cortex-M4 STM32F439ZI).
 */

#include "mahalanobis.h"      /* MAHA_DIM */
#include "mahalanobis_q15.h"

/* VSQRT.F32 sur cible ARM FPU ; sqrtf() standard sur host x86 (tests Unity). */
static inline float fpu_sqrtf_q15(float x)
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

void maha_q15_init(MahalanobisQ15 *m, float threshold, float ema_alpha)
{
    m->threshold = threshold;
    m->ema_alpha = ema_alpha;
    m->mu_scale = 1.0f;
    m->mu_zp = 0;
    m->sigma_inv_scale = 1.0f;

    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        m->mu_q8[i] = 0;                          /* mu = (0 - 0) * 1 = 0 */
        for (int j = 0; j < MAHA_Q15_N_FEATURES; j++) {
            m->sigma_inv_q15[i][j] = (i == j) ? 1 : 0;   /* identité (scale=1) */
        }
    }
}

float maha_q15_score(const MahalanobisQ15 *m, const float *x)
{
    float diff[MAHA_Q15_N_FEATURES];   /* MEM: d × 4 B @ FP32 (pile) */

    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        float mu_i = (float)((int32_t)m->mu_q8[i] - m->mu_zp) * m->mu_scale;
        diff[i] = x[i] - mu_i;
    }

    float dist_sq = 0.0f;
    for (int i = 0; i < MAHA_Q15_N_FEATURES; i++) {
        float left = 0.0f;
        for (int j = 0; j < MAHA_Q15_N_FEATURES; j++) {
            float s_ij = (float)m->sigma_inv_q15[i][j] * m->sigma_inv_scale;
            left += s_ij * diff[j];
        }
        dist_sq += left * diff[i];
    }

    return fpu_sqrtf_q15(dist_sq > 0.0f ? dist_sq : 0.0f);
}

int maha_q15_predict(const MahalanobisQ15 *m, const float *x)
{
    return maha_q15_score(m, x) > m->threshold ? 1 : 0;
}
