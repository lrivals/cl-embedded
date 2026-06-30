/* tinyol_int8.h — TinyOL à poids INT8 + activations UINT8 (Sprint 29, S2902)
 *
 * Variante INT8 de tinyol.c : l'encodeur stocke ses poids en int8_t (fake-quant Q7,
 * calcul dequantifié en FP32), l'embedding transite en uint8_t [0,255] (post-ReLU,
 * asymétrique). La tête OtO (One-to-One) apprend en ligne par SGD : poids INT8 de
 * stockage + copie maîtresse FP32 (straight-through, cf. OtOHeadInt8 Python).
 *
 * Approche fake-quant identique à ewc_head_int8.c. Archi firmware (5→32→16), pas
 * l'archi Python (25→32→16→8) : tinyol.c diverge déjà du Python.
 *
 * Référence : Ren2021TinyOL, src/models/tinyol/tinyol_int8.py, tinyol.c, ewc_head_int8.c
 */

#ifndef TINYOL_INT8_H
#define TINYOL_INT8_H

#include <stdint.h>
#include "tinyol.h"   /* réutilise TINYOL_IN, TINYOL_H1, TINYOL_EMB, TINYOL_OUT */

/* Scale de quantification des activations encodeur (post-ReLU → UINT8 [0,255]).
 * Constante de calibration nommée (CLAUDE.md : pas de magic number).
 * emb_u8 = clamp(round(act / ACT_SCALE), 0, 255) ; act ≈ emb_u8 × ACT_SCALE. */
#define TINYOL_INT8_ACT_SCALE   (8.0f / 255.0f)   /* range activations ≈ [0, 8] */

/* Scale poids Q7 (fake-quant) — int8 = round(w × 128), w ≈ int8 / 128. */
#define TINYOL_INT8_SCALE_W     (1.0f / 128.0f)

/* MEM: ~2.9 Ko @ INT8 (vs ~3.1 Ko @ FP32 enc de tinyol.c, biais restent FP32)
 *   w_enc1 = 32×5 = 160 B, w_enc2 = 16×32 = 512 B, biais FP32 = 192 B */
typedef struct {
    int8_t w_enc1[TINYOL_H1][TINYOL_IN];   /* MEM: 160 B @ INT8 */
    float  b_enc1[TINYOL_H1];              /* MEM: 128 B @ FP32 (biais restent FP32) */
    int8_t w_enc2[TINYOL_EMB][TINYOL_H1];  /* MEM: 512 B @ INT8 */
    float  b_enc2[TINYOL_EMB];             /* MEM:  64 B @ FP32 */
    float  scale_w;                        /* scale commune poids */
} TinyOLEncoderInt8;

/* Tête OtO INT8 : Linear(TINYOL_EMB → 1) + sigmoid, SGD BCE en ligne.
 * Poids INT8 = représentation de stockage, w_master FP32 = copie maîtresse d'apprentissage. */
typedef struct {
    int8_t w[TINYOL_EMB];                  /* MEM: 16 B @ INT8 — poids quantifiés */
    float  b;                              /* biais quantifié (FP32 stocké) */
    float  w_master[TINYOL_EMB];           /* MEM: 64 B @ FP32 — poids maîtres SGD */
    float  b_master;
    float  lr;                             /* learning rate SGD */
    float  last_prob;                      /* prob ∈ [0,1] du dernier predict (confiance) */
} OtOHeadInt8;

/* Initialise l'encodeur depuis les poids FP32 de model_weights.h (quantif Q7 SAT8). */
void tinyol_int8_init(TinyOLEncoderInt8 *enc);

/* Forward encodeur INT8 : x[TINYOL_IN] → emb_u8[TINYOL_EMB] (UINT8, post-ReLU). */
void tinyol_int8_encode(const TinyOLEncoderInt8 *enc, const float *x, uint8_t *emb_u8);

/* Initialise la tête OtO à zéro (apprentissage en ligne). */
void oto_int8_init(OtOHeadInt8 *oto);

/* Forward OtO : dequant emb → sigmoid(w·emb + b) ; stocke last_prob ; retourne classe 0/1. */
int  oto_int8_predict(OtOHeadInt8 *oto, const uint8_t *emb_u8);

/* 1 pas SGD BCE : maj poids maîtres FP32 puis re-quantif INT8 (straight-through). */
void oto_int8_update(OtOHeadInt8 *oto, const uint8_t *emb_u8, int y);

#endif /* TINYOL_INT8_H */
