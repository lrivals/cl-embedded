#pragma once
/* test_vectors_maha_int8.h — GÉNÉRÉ par export_weights_c.py --maha-int8-test-vectors (S2912).
 * Parité forward INT8 C (maha_int8_score) ↔ Python (score_int8). NE PAS MODIFIER À LA MAIN. */
#include <stdint.h>

#define TV_MAHA_INT8_DIM 5
static const uint8_t TV_MAHA_INT8_MU_Q8[5] = {255, 254, 255, 0, 255};
static const float   TV_MAHA_INT8_MU_SCALE   = 0.20121360f;
static const int32_t TV_MAHA_INT8_MU_ZP      = 255;
static const uint8_t TV_MAHA_INT8_SIGMA_INV[5][5] = {
    {255, 2, 16, 2, 6},
    {2, 2, 2, 2, 2},
    {16, 2, 239, 2, 0},
    {2, 2, 2, 2, 2},
    {6, 2, 0, 2, 12}
};
static const float   TV_MAHA_INT8_SIGMA_SCALE = 0.00405372f;
static const int32_t TV_MAHA_INT8_SIGMA_ZP    = 2;
static const float TV_MAHA_INT8_INPUT[5] = {-0.38482925f, -0.21726832f, -0.22096564f, -51.44274902f, 0.09730285f};
static const float   TV_MAHA_INT8_EXPECTED_DIST = 0.45602852f;
