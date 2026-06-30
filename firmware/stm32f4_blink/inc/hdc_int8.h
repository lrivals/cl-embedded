/* hdc_int8.h — HDC à stockage INT8 (hypervecteurs de base) + INT16 (AM) (Sprint 29, S2901)
 *
 * Variante INT8 de hdc.c : les hypervecteurs de base sont binarisés (±1, int8_t),
 * la mémoire associative (AM) accumule des bundles d'hypervecteurs ±1 → plage [-N, +N]
 * stockée en int16_t (saturation ±32767), le produit scalaire de query accumule en int32_t.
 *
 * HDC est nativement entier : après binarisation, la métrique INT8 == métrique FP32,
 * seule la RAM diffère (compression ≈ ×3 vs hypothèse FP32). Pas une quantification
 * approximative — cf. src/models/hdc/hdc_int8.py (référence Python Sprint 28).
 *
 * Schéma d'encodage : aligné sur hdc.c firmware (sign d'une projection aléatoire ±1),
 * bv stocké feature-major. Pas de parité bit-exacte au Python (hdc.c diverge déjà).
 *
 * Référence : Benatti2019HDC, hdc.c, ewc_head_int8.c (pattern INT8)
 */

#ifndef HDC_INT8_H
#define HDC_INT8_H

#include <stdint.h>

/* Architecture : n_features → binarize → int8 hypervector D → AM int16
 *
 * MEM total HDCInt8 (HDC_I_N=9, HDC_I_D=2048, HDC_I_C=4) :
 *   bv : HDC_I_N × HDC_I_D × 1B = 18 432 B @ INT8
 *   am : HDC_I_C × HDC_I_D × 2B = 16 384 B @ INT16
 *   TOTAL : ~34 816 B (~34 Ko) en .bss
 *   vs FP32 : ~106 Ko hypothétique → compression ×3.06
 */

#define HDC_I_N   9      /* Nombre de features d'entrée */
#define HDC_I_D   2048   /* Dimension des hypervecteurs */
#define HDC_I_C   4      /* Nombre de classes */

typedef struct {
    int8_t  bv[HDC_I_N][HDC_I_D];    /* Base vectors — MEM: 18 432 B @ INT8 */
    int16_t am[HDC_I_C][HDC_I_D];    /* Associative memory — MEM: 16 384 B @ INT16 */
} HDCInt8;

/* Initialise les base vectors avec binarisation LCG déterministe (±1 → int8), zéro AM. */
void hdc_int8_init(HDCInt8 *h);

/* Encode features → hypervecteur int8 (±1) par projection signée.
 * hv_out doit pointer sur un int8_t[HDC_I_D] fourni par l'appelant. */
void hdc_int8_encode(const HDCInt8 *h, const float *x, int8_t *hv_out);

/* Query AM : retourne argmax_c(dot_product(hv, am[c])) — accumulateur int32. */
int  hdc_int8_predict(const HDCInt8 *h, const int8_t *hv);

/* Online update : am[label][i] += hv[i] (accumulation int16 saturée ±32767). */
void hdc_int8_update(HDCInt8 *h, const int8_t *hv, int label);

#endif /* HDC_INT8_H */
