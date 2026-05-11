/**
 * model_weights.h — Poids placeholder pour Mahalanobis et EWC MLP
 *
 * Ce fichier est généré automatiquement par scripts/export_weights_c.py
 * après entraînement Python. Ne pas modifier à la main.
 *
 * Placeholder : μ = 0, Σ⁻¹ = I, seuil = 1.0
 * Remplacer via : python scripts/export_weights_c.py --checkpoint <path>
 */

#pragma once
#include "mahalanobis.h"

/* ── Stats Z-score (figées en Flash depuis le dataset d'entraînement) ── */
/* MEM: 40 B @ FP32 (2 × MAHA_DIM × 4 B)                               */
static const float ZSCORE_MEAN[MAHA_DIM] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static const float ZSCORE_STD[MAHA_DIM]  = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

/* ── Mahalanobis : μ initial et Σ⁻¹ (calculés offline en Python) ────── */
static const float MAHA_MEAN_INIT[MAHA_DIM] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

static const float MAHA_PRECISION_INIT[MAHA_DIM][MAHA_DIM] = {
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
};

static const float MAHA_THRESHOLD_INIT = 1.0f;
static const float MAHA_EMA_ALPHA      = 0.1f;
