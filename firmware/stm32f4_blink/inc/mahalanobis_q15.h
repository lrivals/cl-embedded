#pragma once
#include <stdint.h>

/* mahalanobis_q15.h — Détecteur de Mahalanobis à sigma_inv int16 Q15 (Sprint 34, S3407).
 *
 * Réponse au TODO(arnaud) : sur les matrices sigma_inv à grande dynamique, l'INT8 (8 bits)
 * casse la distance (Sprint 28 : AUROC −0.236 CWRU / −0.238 Pronostia). Q15 (int16) garde
 * 256× plus de résolution. Parité bit-à-bit avec MahalanobisDetectorInt8(quant="q15") :
 *   mu_fp32     = (mu_q8 - mu_zp) * mu_scale          (mu reste INT8 affine)
 *   sigma_fp32  = sigma_inv_q15  * sigma_inv_scale    (sigma_inv int16 symétrique)
 *   distance    = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))             (déquant → FP32 sur la FPU, S3407)
 *
 * Stockage : seuls mu_q8 (int8) et sigma_inv_q15 (int16) sont persistants → RAM ÷2 vs FP32
 * sur sigma_inv (au lieu de ÷4 en INT8). La déquantification se fait sur la pile à chaque
 * score (d ≤ 16 → coût négligeable, CLAUDE.md : FP32 autorisé partout sur la NUCLEO).
 */

#ifndef MAHA_Q15_N_FEATURES
#define MAHA_Q15_N_FEATURES MAHA_DIM   /* identique à mahalanobis.h (MAHA_DIM, défaut 5) */
#endif

typedef struct {
    uint8_t mu_q8[MAHA_Q15_N_FEATURES];                              /* MEM: d × 1 B @ UINT8 affine */
    float   mu_scale;                                                /* MEM: 4 B — scale mu */
    int32_t mu_zp;                                                   /* MEM: 4 B — zero-point mu */
    int16_t sigma_inv_q15[MAHA_Q15_N_FEATURES][MAHA_Q15_N_FEATURES]; /* MEM: d² × 2 B @ Q15 */
    float   sigma_inv_scale;                                         /* MEM: 4 B — scale Σ⁻¹ */
    float   threshold;                                               /* MEM: 4 B */
    float   ema_alpha;                                               /* MEM: 4 B */
} MahalanobisQ15;

/* Init neutre : mu=0 (mu_scale=1, mu_zp=0), Σ⁻¹ = identité (scale=1) → distance euclidienne.
 * Les poids réels sont posés via mahalanobis_q15_weights.h (généré par export_weights_c.py). */
void  maha_q15_init(MahalanobisQ15 *m, float threshold, float ema_alpha);

/* Distance de Mahalanobis (déquant mu+Σ⁻¹ → FP32). Parité avec anomaly_score_q15 Python. */
float maha_q15_score(const MahalanobisQ15 *m, const float *x);

/* Prédiction binaire (0=normal, 1=anomalie) via score > threshold. */
int   maha_q15_predict(const MahalanobisQ15 *m, const float *x);
