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
#elif defined(EWC_INT4)
/* Sprint 48 — profondeur sub-INT8 4 bits (linéaire QMAX 7). Poids pré-quantifiés
 * PC (parité émulateur subint8), conteneur int8 en non-packé (.bss ≈ int8). */
typedef int8_t  ewc_v2_w_t;
typedef int8_t  ewc_v2_a_t;
typedef int32_t ewc_v2_acc_t;
#define EWC_V2_W_QMAX  7
#define EWC_V2_A_QMAX  127
#define EWC_V2_PACK_BITS 4               /* 2 poids/octet en packé */
#elif defined(EWC_INT2)
/* Sprint 48 — 2 bits : linéaire QMAX 1 OU ternaire {−1,0,+1} (le firmware est
 * agnostique au mode : il consomme les entiers pré-quantifiés + scales par-canal).
 * Correctif doc S4802 : QMAX = (1<<(bits-1))-1 = 1 (et non 3, cf. émulateur). */
typedef int8_t  ewc_v2_w_t;
typedef int8_t  ewc_v2_a_t;
typedef int32_t ewc_v2_acc_t;
#define EWC_V2_W_QMAX  1
#define EWC_V2_A_QMAX  127
#define EWC_V2_PACK_BITS 2               /* 4 poids/octet en packé */
#elif defined(EWC_INT1)
/* Sprint 48 — 1 bit : binaire {−1,+1} (BWN). Gagnant « agressif » S4708. */
typedef int8_t  ewc_v2_w_t;
typedef int8_t  ewc_v2_a_t;
typedef int32_t ewc_v2_acc_t;
#define EWC_V2_W_QMAX  1
#define EWC_V2_A_QMAX  127
#define EWC_V2_PACK_BITS 1               /* 8 poids/octet en packé */
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

/* ── Sprint 48 — sub-INT8 : bit-packing + dépacking ─────────────────────────
 *
 * Les variantes EWC_INT4/EWC_INT2/EWC_INT1 stockent des poids PRÉ-QUANTIFIÉS côté
 * PC (parité émulateur subint8 : linéaire, ternaire ou binaire). Deux stockages :
 *   - non-packé : conteneur int8 (EWCHeadInt8V2) → `.bss` ≈ INT8 (point d'honnêteté
 *     S4800 : un sub-INT8 dans un int8_t n'économise rien) ; forward = ewc_int8_v2_forward ;
 *   - packé (-DEWC_INTx_PACKED) : 8/EWC_V2_PACK_BITS poids par octet → `.bss` ÷2 (INT4) /
 *     ÷4 (INT2) / ÷8 (INT1), au coût du dépacking (latence, mesurée DWT en S4804).
 *
 * Encodage LSB-first. PACK_BITS ∈ {4,2} : complément à deux + extension de signe.
 * PACK_BITS == 1 : binaire {−1,+1} ↔ bit {0,1} (le complément à deux 1 bit ne
 * distingue pas −1 de +1 → code dédié).
 */
#if defined(EWC_V2_PACK_BITS)

#define EWC_V2_PACK_PER_BYTE  (8 / EWC_V2_PACK_BITS)
/* Nombre d'octets pour n poids de EWC_V2_PACK_BITS bits. */
#define EWC_V2_PACK_STRIDE(n) (((n) * EWC_V2_PACK_BITS + 7) / 8)

/* Dépacke le poids signé i d'une ligne packée (LSB-first). */
static inline int ewc_v2_unpack_weight(const uint8_t *row, int i)
{
    uint8_t byte  = row[i / EWC_V2_PACK_PER_BYTE];
    int     shift = (i % EWC_V2_PACK_PER_BYTE) * EWC_V2_PACK_BITS;
    uint8_t field = (uint8_t)((byte >> shift) & ((1u << EWC_V2_PACK_BITS) - 1u));
#if EWC_V2_PACK_BITS == 1
    return field ? 1 : -1;                              /* binaire {−1,+1} */
#else
    int s = 8 - EWC_V2_PACK_BITS;                       /* extension de signe */
    return (int)((int8_t)(field << s) >> s);
#endif
}

/* Empaquette n poids signés déjà quantifiés dans dst (LSB-first). Miroir exact de
 * ewc_v2_unpack_weight et de _pack_weights (export_weights_c.py) → parité. */
static inline void ewc_v2_pack_row(uint8_t *dst, const int8_t *q, int n)
{
    int nbytes = EWC_V2_PACK_STRIDE(n);
    for (int b = 0; b < nbytes; b++) dst[b] = 0u;
    for (int i = 0; i < n; i++) {
#if EWC_V2_PACK_BITS == 1
        uint8_t field = (uint8_t)((q[i] > 0) ? 1u : 0u);
#else
        uint8_t field = (uint8_t)((uint8_t)q[i] & ((1u << EWC_V2_PACK_BITS) - 1u));
#endif
        dst[i / EWC_V2_PACK_PER_BYTE] |=
            (uint8_t)(field << ((i % EWC_V2_PACK_PER_BYTE) * EWC_V2_PACK_BITS));
    }
}

#endif /* EWC_V2_PACK_BITS */

#if defined(EWC_INTx_PACKED)
/* Tête sub-INT8 bit-packée. MEM: w* = EWC_V2_PACK_STRIDE(n) octets/ligne (÷2/÷4/÷8
 * vs int8) ; scales/biais FP32 inchangés. */
typedef struct {
    uint8_t w1[EWC_H1][EWC_V2_PACK_STRIDE(EWC_IN)];  float scale_w1[EWC_H1];  float b1[EWC_H1];
    uint8_t w2[EWC_H2][EWC_V2_PACK_STRIDE(EWC_H1)];  float scale_w2[EWC_H2];  float b2[EWC_H2];
    uint8_t w3[EWC_OUT][EWC_V2_PACK_STRIDE(EWC_H2)]; float scale_w3[EWC_OUT]; float b3[EWC_OUT];
    float scale_act_in, scale_act_h1, scale_act_h2;
} EWCHeadSubInt8Packed;

/**
 * ewc_subint8_packed_forward — Forward inférence packé (dépack → MAC FPU).
 *
 * Parité stricte avec ewc_int8_v2_forward : le packing ne change QUE le stockage
 * (mêmes entiers, mêmes scales, même déquant acc·scale_w[j]·scale_act + b).
 */
void ewc_subint8_packed_forward(const EWCHeadSubInt8Packed *h, const float *x, float *logits);
#endif /* EWC_INTx_PACKED */

#endif /* EWC_HEAD_INT8_V2_H */
