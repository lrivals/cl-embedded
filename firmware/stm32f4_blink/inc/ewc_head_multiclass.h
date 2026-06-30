#pragma once
#include <stdint.h>

/* Architecture : Input(EWC_MC_IN) → ReLU(EWC_MC_H1) → ReLU(EWC_MC_H2) → Output(EWC_MC_N_CLASSES)
 *
 * MEM total EWCHeadMC (IN=9, H1=32, H2=16, N_CLASSES=10) :
 *   Poids   : (9×32+32 + 32×16+16 + 16×10+10) × 4 = ~4.68 Ko @ FP32
 *   Fisher  : identique                             = ~4.68 Ko @ FP32
 *   star_w  : identique                             = ~4.68 Ko @ FP32
 *   TOTAL   : ~14 Ko @ FP32 en .bss  ✅ << 256 Ko NUCLEO-F439ZI
 *
 * Pour N_CLASSES=3 (Paderborn) : ~10.2 Ko total
 * Configurer via Makefile : CFLAGS += -DEWC_MC_N_CLASSES=10   (CWRU)
 *                           CFLAGS += -DEWC_MC_N_CLASSES=3    (Paderborn)
 */

#ifndef EWC_MC_N_CLASSES
#define EWC_MC_N_CLASSES 10    /* Défaut : CWRU 10 classes */
#endif

#define EWC_MC_IN    9     /* Dimension d'entrée — features CWRU/Paderborn (board_*.yaml) */
#define EWC_MC_H1   32     /* Neurons couche cachée 1 */
#define EWC_MC_H2   16     /* Neurons couche cachée 2 */
#define EWC_MC_LR           0.01f   /* Taux d'apprentissage SGD */
#define EWC_MC_FISHER_DECAY 0.99f   /* EMA decay identique à ewc_head.c */

typedef struct {
    /* Poids courants */
    float w1[EWC_MC_H1][EWC_MC_IN];              float b1[EWC_MC_H1];
    float w2[EWC_MC_H2][EWC_MC_H1];              float b2[EWC_MC_H2];
    float w3[EWC_MC_N_CLASSES][EWC_MC_H2];       float b3[EWC_MC_N_CLASSES];

    /* Fisher diagonale */
    float fisher1[EWC_MC_H1][EWC_MC_IN];
    float fisher2[EWC_MC_H2][EWC_MC_H1];
    float fisher3[EWC_MC_N_CLASSES][EWC_MC_H2];

    float lambda;   /* Coefficient EWC — depuis configs/board_*.yaml */

    /* θ* de référence */
    float star_w1[EWC_MC_H1][EWC_MC_IN];
    float star_w2[EWC_MC_H2][EWC_MC_H1];
    float star_w3[EWC_MC_N_CLASSES][EWC_MC_H2];
} EWCHeadMC;

void ewc_mc_init(EWCHeadMC *h);
        /* Xavier LCG seed=42, zero fisher/star_w */

void ewc_mc_forward(const EWCHeadMC *h, const float *x, float *logits);
        /* logits[EWC_MC_N_CLASSES] — logits bruts (avant softmax) */

int  ewc_mc_predict(const EWCHeadMC *h, const float *x);
        /* Retourne argmax(logits) ∈ [0, EWC_MC_N_CLASSES-1] */

void ewc_mc_sgd_step(EWCHeadMC *h, const float *x, int label);
        /* SGD 1 step : CE loss + terme EWC. label ∈ [0, EWC_MC_N_CLASSES-1] */

void ewc_mc_consolidate(EWCHeadMC *h, float alpha);
        /* EMA Fisher + snapshot θ* */
