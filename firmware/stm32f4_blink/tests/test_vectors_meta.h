/**
 * test_vectors_meta.h — Vecteurs de test méta (parité C ↔ Python).
 * Valider : meta_forward(C) == proba sigmoïde Python (tolérance 1e-5 FP32).
 * NE PAS MODIFIER À LA MAIN — généré par scripts/export_weights_c.py.
 */

#pragma once

#define TV_META_N_CASES    3
#define TV_META_N_FEATURES 4

static const float TV_META_INPUT[TV_META_N_CASES][TV_META_N_FEATURES] = {
    {0.10000000f, 0.20000000f, 0.00000000f, 0.60000002f},
    {0.89999998f, 0.80000001f, 1.00000000f, 0.30000001f},
    {0.50000000f, 0.50000000f, 0.00000000f, 0.00000000f}
};

static const float TV_META_EXPECTED[TV_META_N_CASES] = {0.11328662f, 0.98026967f, 0.45921823f};
