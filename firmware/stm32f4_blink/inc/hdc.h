#pragma once
#include <stdint.h>

/* HDC — Hyperdimensional Computing skeleton
 * Référence : Benatti2019HDC
 * Depuis configs/board_hdc.yaml — TODO(arnaud): D=512 si budget serré avec 3 modèles? */

#define HDC_DIM        1000   /* Dimension des hypervecteurs */
#define HDC_N_FEATURES    5   /* Features d'entrée (Monitoring dataset) */
#define HDC_N_CLASSES     2   /* faulty / normal */

/* MEM total HDCClassifier : ~28 Ko @ FP32
 *   am   : 2*1000*4    = 8 Ko  (.bss SRAM)
 *   proj : 1000*5*4    = 20 Ko (SRAM skeleton — TODO(dorra): Flash const ou seed?) */
typedef struct {
    float am[HDC_N_CLASSES][HDC_DIM];      /* Mémoire associative — accumulateurs FP32 */
    float proj[HDC_DIM][HDC_N_FEATURES];   /* Matrice de projection aléatoire */
    int   n_trained;
} HDCClassifier;

/* Initialise le classifieur à zéro (proj à remplir avant utilisation). */
void hdc_init   (HDCClassifier *h);

/* Encode x ∈ ℝ^HDC_N_FEATURES en hypervecteur binarisé hv_out ∈ {-1,+1}^HDC_DIM.
 * Propriété : sum(hv_out[i]²) == HDC_DIM exactement.
 * hv_out doit pointer sur un buffer de HDC_DIM floats alloué par l'appelant. */
void hdc_encode (const HDCClassifier *h, const float *x, float *hv_out);

/* Retourne la classe prédite (0 … HDC_N_CLASSES-1) par argmax dot(am[c], hv). */
int  hdc_predict(const HDCClassifier *h, const float *hv);

/* Met à jour le prototype am[label] par accumulation de hv (apprentissage incrémental).
 * Pas d'oubli catastrophique par construction — O(HDC_DIM) en temps, O(1) en mémoire. */
void hdc_update (HDCClassifier *h, const float *hv, int label);
