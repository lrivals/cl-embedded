#pragma once
#include <stdint.h>

/* MEM: MahalanobisDetector = 128 B @ FP32 / 32 B @ INT8 (d=5) */
/* Nombre de features — surchargeable au build (S3506) : `make MAHA_DIM=9`.
 * Défaut 5 (condition board 5feat) → .bss inchangé sans override. */
#ifndef MAHA_DIM
#define MAHA_DIM 5   /* Nombre de features — conforme configs/unsupervised_config.yaml */
#endif

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
