/**
 * test_vectors.h — Vecteurs de test générés par export_weights_c.py
 * Valider : dist C == dist Python (tolérance 1e-5 FP32).
 * NE PAS MODIFIER À LA MAIN.
 */

#pragma once

/* d = 5 features */
static const float TV_MAHA_MEAN[5] = {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};

static const float TV_MAHA_PRECISION[5][5] = {
    {1.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 1.00000000f, 0.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 1.00000000f, 0.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 1.00000000f, 0.00000000f},
    {0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 1.00000000f}
};

static const float TV_MAHA_INPUT[5] = {1.00000000f, 2.00000000f, 3.00000000f, 4.00000000f, 5.00000000f};

/* Distance numpy FP32 attendue (tolérance 1e-5 en C) */
static const float TV_MAHA_EXPECTED_DIST = 7.41619849f;
