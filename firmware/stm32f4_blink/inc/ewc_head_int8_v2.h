/**
 * ewc_head_int8_v2.h — Tête MLP EWC quantifiée INT8 v2 (Sprint 39, S3907)
 *
 * Corrige les trois défauts de l'INT8 « legacy » (ewc_head_int8.c, audit S3901) :
 *   1. accumulateur int32_t (v1 : int16_t → overflow latent),
 *   2. scales de poids par-canal calibrés (v1 : 1/128 figé par-tenseur),
 *   3. scales d'activation calibrés (v1 : 1/128 figé, clampe les activations >1).
 *
 * Le v1 reste STRICTEMENT intact pour la comparaison A/B board (S3916). Ce fichier est
 * un nouveau kernel séparé (décision utilisateur). Forward inférence uniquement :
 * déquantification → FP32 sur FPU (parité bit-à-bit avec l'émulateur Python
 * ``forward_quant(..., per_channel_int8|q15)`` de src/utils/int8_c_emulation.py).
 *
 * Variantes de précision (build conditionnel, mutuellement exclusives) :
 *   - défaut          : poids int8 par-canal + activations int8 calibrées (per_channel_int8),
 *   - -DEWC_INT8_Q15  : poids int16 Q15 + activations int16 Q15 (q15),
 *   - -DEWC_INT8_MIXED: poids int8 par-canal + activations int16 (mixed_int8w_q15act).
 *
 * Les scales sont importés du header généré ewc_head_int8_v2_weights.h
 * (scripts/export_weights_c.py --int8-v2, S3908) — jamais saisis à la main.
 *
 * Référence Python : src/utils/int8_c_emulation.py (émulateur bit-exact).
 */

#ifndef EWC_HEAD_INT8_V2_H
#define EWC_HEAD_INT8_V2_H

#include <stdint.h>
#include "ewc_head.h"   /* réutilise EWC_IN, EWC_H1, EWC_H2, EWC_OUT, EWCHead */

/* Type de stockage des poids/activations selon la variante de build. */
#if defined(EWC_INT8_Q15)
typedef int16_t ewc_v2_w_t;      /* poids Q15 (16 bits) */
typedef int16_t ewc_v2_a_t;      /* activations Q15 (16 bits) */
typedef int64_t ewc_v2_acc_t;    /* acc 64 bits : int16×int16 sommé déborde int32 */
#define EWC_V2_W_QMAX  32767
#define EWC_V2_A_QMAX  32767
#elif defined(EWC_INT8_MIXED)
typedef int8_t  ewc_v2_w_t;      /* poids int8 par-canal */
typedef int16_t ewc_v2_a_t;      /* activations int16 (évite le clamp Q7) */
typedef int32_t ewc_v2_acc_t;    /* int8×int16 sommé tient dans int32 */
#define EWC_V2_W_QMAX  127
#define EWC_V2_A_QMAX  32767
#else
typedef int8_t  ewc_v2_w_t;      /* poids int8 par-canal (défaut) */
typedef int8_t  ewc_v2_a_t;      /* activations int8 calibrées */
typedef int32_t ewc_v2_acc_t;    /* int8×int8 sommé : int32 largement suffisant */
#define EWC_V2_W_QMAX  127
#define EWC_V2_A_QMAX  127
#endif

/* Tête INT8 v2 — scales par-canal (un par neurone de sortie) + scales d'activation.
 * MEM (défaut int8) : w1/w2/w3 = 704 B + scale_w* (≈200 B FP32) + biais FP32. */
typedef struct {
    ewc_v2_w_t w1[EWC_H1][EWC_IN];   float scale_w1[EWC_H1];   /* un scale par neurone */
    float      b1[EWC_H1];
    ewc_v2_w_t w2[EWC_H2][EWC_H1];   float scale_w2[EWC_H2];
    float      b2[EWC_H2];
    ewc_v2_w_t w3[EWC_OUT][EWC_H2];  float scale_w3[EWC_OUT];
    float      b3[EWC_OUT];
    float      scale_act_in, scale_act_h1, scale_act_h2;   /* activations calibrées */
} EWCHeadInt8V2;

/**
 * ewc_int8_v2_from_fp32_calib — Quantifie une tête FP32 avec scales par-canal calibrés.
 *
 * @param dst      tête v2 à remplir
 * @param src      tête FP32 source (poids w1/w2/w3, biais)
 * @param act_max  bornes max|activation| calibrées [in, h1, h2] (émulateur : calibrate_activations)
 *
 * scale_w*[j] = max|W[j,:]| / QMAX ; scale_act_* = act_max / A_QMAX. Poids quantifiés
 * round(W / scale[:,None]) saturés [-QMAX, QMAX].
 */
void ewc_int8_v2_from_fp32_calib(EWCHeadInt8V2 *dst, const EWCHead *src,
                                 const float act_max[3]);

/**
 * ewc_int8_v2_forward — Forward inférence (accumulateur int32, déquant par-canal exacte).
 *
 * @param h       tête v2 calibrée
 * @param x       entrée FP32 [EWC_IN]
 * @param logits  sortie FP32 [EWC_OUT]
 *
 * MEM: stack a1[EWC_H1] + a2[EWC_H2] activations quantifiées + h1/h2 FP32.
 */
void ewc_int8_v2_forward(const EWCHeadInt8V2 *h, const float *x, float *logits);

#endif /* EWC_HEAD_INT8_V2_H */
