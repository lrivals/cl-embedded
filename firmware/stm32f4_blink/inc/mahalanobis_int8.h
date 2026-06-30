#pragma once
#include <stdint.h>
#include "mahalanobis.h"   /* MAHA_DIM */

/* mahalanobis_int8.h — Détecteur de Mahalanobis INT8 (Sprint 29, S2912).
 *
 * Variante INT8 du détecteur de Mahalanobis pour la grille board 4×5 (extension O8).
 * Miroir exact de MahalanobisDetectorInt8(quant="int8") (Python, S2805) :
 *   mu_fp32     = (mu_q8 - mu_zp) * mu_scale              (mu INT8 affine)
 *   sigma_fp32  = (sigma_inv_q8 - sigma_inv_zp) * sigma_inv_scale  (Σ⁻¹ INT8 affine global)
 *   distance    = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))                 (déquant → FP32 sur la FPU)
 *
 * Contrairement à la variante Q15 (S3407, sigma_inv int16 symétrique), ici sigma_inv reste
 * INT8 affine (8 bits) → reproduit FIDÈLEMENT la dégradation connue sur grande dynamique
 * (Sprint 28 : AUROC −0.236 CWRU / −0.238 Pronostia). Le but est de MESURER ce comportement
 * sur board, pas de le corriger (Q15 = fallback recommandé, cf. Sprint 34).
 *
 * Stockage : mu_q8 (uint8) + sigma_inv_q8 (uint8) → RAM ÷4 vs FP32 sur les poids.
 * Déquant sur la pile à chaque score (d ≤ 16, coût négligeable ; FP32 autorisé NUCLEO).
 *
 * Pas de malloc, FPU FP32 (Cortex-M4 STM32F439ZI).
 */

#ifndef MAHA_INT8_N_FEATURES
#define MAHA_INT8_N_FEATURES MAHA_DIM   /* identique à mahalanobis.h (MAHA_DIM, défaut 5) */
#endif

typedef struct {
    uint8_t mu_q8[MAHA_INT8_N_FEATURES];                                /* MEM: d × 1 B @ UINT8 affine */
    float   mu_scale;                                                   /* MEM: 4 B — scale mu */
    int32_t mu_zp;                                                      /* MEM: 4 B — zero-point mu */
    uint8_t sigma_inv_q8[MAHA_INT8_N_FEATURES][MAHA_INT8_N_FEATURES];   /* MEM: d² × 1 B @ UINT8 affine */
    float   sigma_inv_scale;                                            /* MEM: 4 B — scale Σ⁻¹ */
    int32_t sigma_inv_zp;                                               /* MEM: 4 B — zero-point Σ⁻¹ */
    float   threshold;                                                  /* MEM: 4 B */
    float   ema_alpha;                                                  /* MEM: 4 B — taux EMA (inutilisé en INT8, fit offline) */
} MahalanobisInt8;

/* Init neutre : mu=0 (scale=1, zp=0), Σ⁻¹ = identité (scale=1, zp=0) → distance euclidienne.
 * Les poids réels sont posés via mahalanobis_int8_weights.h (généré par export_weights_c.py). */
void  maha_int8_init(MahalanobisInt8 *m, float threshold, float ema_alpha);

/* Distance de Mahalanobis (déquant mu+Σ⁻¹ INT8 affine → FP32). Parité avec anomaly_score_int8. */
float maha_int8_score(const MahalanobisInt8 *m, const float *x);

/* Prédiction binaire (0=normal, 1=anomalie) via score > threshold. */
int   maha_int8_predict(const MahalanobisInt8 *m, const float *x);
