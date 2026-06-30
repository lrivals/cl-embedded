#pragma once
/* mahalanobis_q15_weights.h — GÉNÉRÉ par export_weights_c.py --maha-q15 (S3407).
 * NE PAS MODIFIER À LA MAIN (règle CLAUDE.md). mu INT8 affine + sigma_inv int16 Q15. */
#include <stdint.h>

#define MAHA_Q15_WEIGHTS_PROVIDED
#define MAHA_Q15_NATIVE_DIM 5

static const uint8_t MAHA_Q15_MU_Q8[5] = {60, 60, 255, 60, 57};
static const float   MAHA_Q15_MU_SCALE   = 0.00884553f;
static const int32_t MAHA_Q15_MU_ZP      = 0;
static const int16_t MAHA_Q15_SIGMA_INV[5][5] = {
    {32667, -32705, 4, 1, -6},
    {-32705, 32767, -7, -5, 4},
    {4, -7, 0, 0, 0},
    {1, -5, 0, 9, 0},
    {-6, 4, 0, 0, 9}
};
static const float   MAHA_Q15_SIGMA_SCALE = 1.70920343f;
static const float   MAHA_Q15_THRESHOLD   = 3.51161300f;
