/**
 * mahalanobis_int8.c — Détecteur de Mahalanobis INT8 pour MCU (Sprint 29, S2912).
 *
 * Déquantification (mu + sigma_inv INT8 affine) → distance en FP32 sur la FPU.
 * Bit-identique à MahalanobisDetectorInt8.anomaly_score_int8 (Python, S2805).
 *
 * Contraintes : pas de malloc, pas de stdlib lourde, FPU FP32 (Cortex-M4 STM32F439ZI).
 */

#include "mahalanobis.h"      /* MAHA_DIM */
#include "mahalanobis_int8.h"

/* VSQRT.F32 sur cible ARM FPU ; sqrtf() standard sur host x86 (tests Unity). */
static inline float fpu_sqrtf_int8(float x)
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

void maha_int8_init(MahalanobisInt8 *m, float threshold, float ema_alpha)
{
    m->threshold = threshold;
    m->ema_alpha = ema_alpha;
    m->mu_scale = 1.0f;
    m->mu_zp = 0;
    m->sigma_inv_scale = 1.0f;
    m->sigma_inv_zp = 0;

    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        m->mu_q8[i] = 0;                          /* mu = (0 - 0) * 1 = 0 */
        for (int j = 0; j < MAHA_INT8_N_FEATURES; j++) {
            m->sigma_inv_q8[i][j] = (i == j) ? 1 : 0;   /* identité (scale=1, zp=0) */
        }
    }
}

float maha_int8_score(const MahalanobisInt8 *m, const float *x)
{
    float diff[MAHA_INT8_N_FEATURES];   /* MEM: d × 4 B @ FP32 (pile) */

    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        float mu_i = (float)((int32_t)m->mu_q8[i] - m->mu_zp) * m->mu_scale;
        diff[i] = x[i] - mu_i;
    }

    float dist_sq = 0.0f;
    for (int i = 0; i < MAHA_INT8_N_FEATURES; i++) {
        float left = 0.0f;
        for (int j = 0; j < MAHA_INT8_N_FEATURES; j++) {
            float s_ij = (float)((int32_t)m->sigma_inv_q8[i][j] - m->sigma_inv_zp)
                       * m->sigma_inv_scale;
            left += s_ij * diff[j];
        }
        dist_sq += left * diff[i];
    }

    return fpu_sqrtf_int8(dist_sq > 0.0f ? dist_sq : 0.0f);
}

int maha_int8_predict(const MahalanobisInt8 *m, const float *x)
{
    return maha_int8_score(m, x) > m->threshold ? 1 : 0;
}
