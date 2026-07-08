/**
 * ewc_head_int8_v2.c — Tête MLP EWC quantifiée INT8 v2 (Sprint 39, S3907)
 *
 * Kernel d'inférence corrigé, séparé de ewc_head_int8.c (laissé intact pour l'A/B
 * board, S3916). Corrige les 3 défauts de l'audit S3901 :
 *   - accumulateur int32_t (plus d'overflow int16),
 *   - scales de poids PAR-CANAL calibrés (un par neurone de sortie),
 *   - scales d'activation calibrés (plus de 1/128 figé qui clampe).
 *
 * Chemin numérique = déquantification exacte vers FP32 sur FPU :
 *   acc(int32) = Σ w_q[j][i] · x_q[i]
 *   val        = acc · scale_w[j] · scale_act + b[j]      (déquant par-canal exacte)
 *   ReLU       = max(val, 0)  en FP32 (aucun clamp Q7)
 *
 * Parité bit-à-bit visée avec l'émulateur Python forward_quant(..., per_channel_int8)
 * (chemin « calibré » : round + accumulation int64 sans wrap + déquant scale_w·scale_act).
 *
 * Variantes de build : -DEWC_INT8_Q15 (16 bits) / -DEWC_INT8_MIXED (poids int8, act int16).
 * Cf. ewc_head_int8_v2.h pour les typedefs ewc_v2_w_t / ewc_v2_a_t.
 *
 * Référence : S3901 (audit), S3907 (spec), mahalanobis_q15.c (patron Q15).
 */

#include "ewc_head_int8_v2.h"
#include <math.h>    /* lroundf, fabsf */
#include <string.h>

/* Saturation générique dans [-qmax, qmax]. */
static inline int32_t sat_q(int32_t v, int32_t qmax)
{
    if (v >  qmax) return  qmax;
    if (v < -qmax) return -qmax;
    return v;
}

/* Quantifie une valeur FP32 par un scale donné, saturée à qmax. */
static inline int32_t quant_val(float v, float scale, int32_t qmax)
{
    if (scale <= 0.0f) return 0;
    return sat_q((int32_t)lroundf(v / scale), qmax);
}

/* ── Conversion FP32 → v2 (scales par-canal calibrés) ───────────────────── */

/* Calcule un scale par ligne de sortie : max|W[j,:]| / qmax. */
static void per_channel_scales(const float *w_row, int n_in, int32_t qmax, float *scale_out)
{
    float m = 0.0f;
    for (int i = 0; i < n_in; i++) {
        float a = fabsf(w_row[i]);
        if (a > m) m = a;
    }
    *scale_out = (m > 0.0f) ? (m / (float)qmax) : 1.0f;
}

void ewc_int8_v2_from_fp32_calib(EWCHeadInt8V2 *dst, const EWCHead *src,
                                 const float act_max[3])
{
    /* Scales d'activation calibrés (act_max = [in, h1, h2]). */
    dst->scale_act_in = (act_max[0] > 0.0f) ? act_max[0] / (float)EWC_V2_A_QMAX : 1.0f;
    dst->scale_act_h1 = (act_max[1] > 0.0f) ? act_max[1] / (float)EWC_V2_A_QMAX : 1.0f;
    dst->scale_act_h2 = (act_max[2] > 0.0f) ? act_max[2] / (float)EWC_V2_A_QMAX : 1.0f;

    /* Couche 1 : scales par-canal + quantif poids. */
    for (int j = 0; j < EWC_H1; j++) {
        per_channel_scales(&src->w1[j][0], EWC_IN, EWC_V2_W_QMAX, &dst->scale_w1[j]);
        for (int i = 0; i < EWC_IN; i++)
            dst->w1[j][i] = (ewc_v2_w_t)quant_val(src->w1[j][i], dst->scale_w1[j], EWC_V2_W_QMAX);
        dst->b1[j] = src->b1[j];
    }
    /* Couche 2. */
    for (int j = 0; j < EWC_H2; j++) {
        per_channel_scales(&src->w2[j][0], EWC_H1, EWC_V2_W_QMAX, &dst->scale_w2[j]);
        for (int i = 0; i < EWC_H1; i++)
            dst->w2[j][i] = (ewc_v2_w_t)quant_val(src->w2[j][i], dst->scale_w2[j], EWC_V2_W_QMAX);
        dst->b2[j] = src->b2[j];
    }
    /* Couche 3 (sortie, pas de ReLU). */
    for (int j = 0; j < EWC_OUT; j++) {
        per_channel_scales(&src->w3[j][0], EWC_H2, EWC_V2_W_QMAX, &dst->scale_w3[j]);
        for (int i = 0; i < EWC_H2; i++)
            dst->w3[j][i] = (ewc_v2_w_t)quant_val(src->w3[j][i], dst->scale_w3[j], EWC_V2_W_QMAX);
        dst->b3[j] = src->b3[j];
    }
}

/* ── Forward inférence (accumulateur int32, déquant par-canal exacte) ────── */

void ewc_int8_v2_forward(const EWCHeadInt8V2 *h, const float *x, float *logits)
{
    ewc_v2_a_t a_in[EWC_IN];   /* activations d'entrée quantifiées */
    ewc_v2_a_t a_h1[EWC_H1];   /* activations couche 1 quantifiées */
    ewc_v2_a_t a_h2[EWC_H2];   /* activations couche 2 quantifiées */

    /* Quantif entrée (activations calibrées). */
    for (int i = 0; i < EWC_IN; i++)
        a_in[i] = (ewc_v2_a_t)quant_val(x[i], h->scale_act_in, EWC_V2_A_QMAX);

    /* Couche 1 : acc entier large, déquant acc·scale_w[j]·scale_act_in + b, ReLU FP32. */
    for (int j = 0; j < EWC_H1; j++) {
        ewc_v2_acc_t acc = 0;                              /* ← int32 (int8) / int64 (Q15) */
        for (int i = 0; i < EWC_IN; i++)
            acc += (ewc_v2_acc_t)h->w1[j][i] * (ewc_v2_acc_t)a_in[i];
        float val = (float)acc * h->scale_w1[j] * h->scale_act_in + h->b1[j];
        float relu = val > 0.0f ? val : 0.0f;              /* ReLU FP32 (pas de clamp Q7) */
        a_h1[j] = (ewc_v2_a_t)quant_val(relu, h->scale_act_h1, EWC_V2_A_QMAX);
    }

    /* Couche 2. */
    for (int j = 0; j < EWC_H2; j++) {
        ewc_v2_acc_t acc = 0;
        for (int i = 0; i < EWC_H1; i++)
            acc += (ewc_v2_acc_t)h->w2[j][i] * (ewc_v2_acc_t)a_h1[i];
        float val = (float)acc * h->scale_w2[j] * h->scale_act_h1 + h->b2[j];
        float relu = val > 0.0f ? val : 0.0f;
        a_h2[j] = (ewc_v2_a_t)quant_val(relu, h->scale_act_h2, EWC_V2_A_QMAX);
    }

    /* Couche 3 : logits FP32 (softmax/argmax en aval). */
    for (int j = 0; j < EWC_OUT; j++) {
        ewc_v2_acc_t acc = 0;
        for (int i = 0; i < EWC_H2; i++)
            acc += (ewc_v2_acc_t)h->w3[j][i] * (ewc_v2_acc_t)a_h2[i];
        logits[j] = (float)acc * h->scale_w3[j] * h->scale_act_h2 + h->b3[j];
    }
}
