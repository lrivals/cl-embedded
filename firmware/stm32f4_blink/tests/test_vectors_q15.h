#pragma once
/* test_vectors_q15.h — GÉNÉRÉ par export_weights_c.py --maha-q15-test-vectors (S3409).
 * Parité forward Q15 C (maha_q15_score) ↔ Python (score_q15). NE PAS MODIFIER À LA MAIN. */
#include <stdint.h>

#define TV_Q15_DIM 5
static const int8_t TV_Q15_MU_Q8[5] = {3, 19, 4, 255, 0};
static const float   TV_Q15_MU_SCALE   = 0.09977844f;
static const int32_t TV_Q15_MU_ZP      = 4;
static const int16_t TV_Q15_SIGMA_INV[5][5] = {
    {30695, -1, 480, 0, 46},
    {-1, 12, 15, 0, -6},
    {480, 15, 32767, 2, -109},
    {0, 0, 2, 0, 0},
    {46, -6, -109, 0, 1415}
};
static const float   TV_Q15_SIGMA_SCALE = 0.00003127f;
static const float TV_Q15_INPUT[5] = {-0.10343039f, 0.23390496f, 2.57571363f, 25.49166870f, 0.20991492f};
static const float   TV_Q15_EXPECTED_DIST = 2.60788846f;
