#pragma once
#include <stdint.h>

/* MEM: MahalanobisDetector = 128 B @ FP32 / 32 B @ INT8 (d=5) */
#define MAHA_DIM 5   /* Nombre de features — conforme configs/unsupervised_config.yaml */

typedef struct {
    float mean[MAHA_DIM];                    /* MEM: 20 B @ FP32 */
    float precision[MAHA_DIM][MAHA_DIM];     /* MEM: 100 B @ FP32 — Σ⁻¹ */
    float threshold;                         /* MEM: 4 B */
    float ema_alpha;                         /* MEM: 4 B — taux EMA update online */
} MahalanobisDetector;

void  maha_init(MahalanobisDetector *det, float threshold, float ema_alpha);
float maha_score(const MahalanobisDetector *det, const float *x);
void  maha_update(MahalanobisDetector *det, const float *x);
int   maha_predict(const MahalanobisDetector *det, const float *x);
