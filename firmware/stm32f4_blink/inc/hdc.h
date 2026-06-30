/* HDC — Hyperdimensional Computing (implémentation complète Sprint 23)
 * Référence : Benatti2019HDC
 * Depuis configs/board_hdc.yaml */

#ifndef HDC_H
#define HDC_H

#include <stdint.h>
#include "ring_buffer.h"

#define HDC_DIM           1000  /* Dimension des hypervecteurs */
/* Features d'entrée — surchargeable au build (S3506) : `make HDC_N_FEATURES=9`.
 * Défaut 5 (condition board 5feat) → .bss inchangé sans override. */
#ifndef HDC_N_FEATURES
#define HDC_N_FEATURES       5  /* Features d'entrée (top-5 sélectionnées) */
#endif
#define HDC_N_CLASSES        2  /* faulty / normal */
#define HDC_RETRAIN_BUF     50  /* Taille buffer retrain (FIFO circulaire) */
                                /* MEM buf total : 50*(5+1) = 300 B */

/* Élément du buffer retrain (S3402) : features quantifiées uint8 + label entrelacés.
 *   octets [0 .. HDC_N_FEATURES-1] = features ; octet [HDC_N_FEATURES] = label. */
#define HDC_BUF_ELEM_SIZE (HDC_N_FEATURES + 1)

/* MEM total HDCClassifier :
 *   am          : 2*1000*4         = 8 000 B (SRAM .bss)
 *   proj        : 1000*5*4         = 20 000 B (SRAM — TODO(dorra): Flash const ?)
 *   buf_storage : 50*(5+1)*1       =    300 B (features+label entrelacés)
 *   rb          : RingBuffer       = ~24 B
 *   scalars     : n_trained + buf_head + buf_count = 12 B
 *   TOTAL       : ~28 336 B ≈ 27.7 Ko @ FP32 (dans budget 64 Ko board) */
typedef struct {
    float   am[HDC_N_CLASSES][HDC_DIM];            /* Mémoire associative */
    float   proj[HDC_DIM][HDC_N_FEATURES];         /* Projection aléatoire (fixée à l'init) */
    uint8_t buf_storage[HDC_RETRAIN_BUF * HDC_BUF_ELEM_SIZE]; /* Stockage ring buffer (uint8) */
    RingBuffer rb;      /* Abstraction buffer circulaire (S3402) sur buf_storage */
    int     n_trained;
    int     buf_head;   /* Miroir de rb.head — accès direct par les tests/diagnostics */
    int     buf_count;  /* Miroir de rb.count — nb d'échantillons (≤ HDC_RETRAIN_BUF) */
} HDCClassifier;

void hdc_init               (HDCClassifier *h);
void hdc_encode             (const HDCClassifier *h, const float *x, float *hv_out);
int  hdc_predict            (const HDCClassifier *h, const float *hv);
void hdc_update             (HDCClassifier *h, const float *hv, int label);
void hdc_update_with_sample (HDCClassifier *h, const float *x,
                              const float *hv, int label);
void hdc_binarize           (HDCClassifier *h);
void hdc_retrain            (HDCClassifier *h);

#endif /* HDC_H */
