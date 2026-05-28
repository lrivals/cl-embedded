#pragma once
#include <stdint.h>

/* Architecture : Input(5) → ReLU(32) → ReLU(16) → Output(2)
 * MEM total EWCHead : ~10 Ko @ FP32
 *   Poids   : (5×32+32 + 32×16+16 + 16×2+2) × 4 = ~3 Ko
 *   Fisher  : identique aux poids                  = ~3 Ko
 *   star_w  : identique aux poids                  = ~3 Ko  */

#define EWC_IN   5     /* Dimension d'entrée — features Monitoring dataset */
#define EWC_H1  32     /* Neurons couche cachée 1 */
#define EWC_H2  16     /* Neurons couche cachée 2 */
#define EWC_OUT  2     /* Sorties (logits — softmax + CE en inférence) */
#define EWC_LR           0.01f  /* Taux d'apprentissage SGD */
#define EWC_FISHER_DECAY 0.99f  /* EMA decay pour ewc_consolidate() */

typedef struct {
    /* Poids courants — MEM: 3 Ko @ FP32 */
    float w1[EWC_H1][EWC_IN];    float b1[EWC_H1];
    float w2[EWC_H2][EWC_H1];    float b2[EWC_H2];
    float w3[EWC_OUT][EWC_H2];   float b3[EWC_OUT];

    /* Fisher diagonale (régularisation EWC) — MEM: 3 Ko @ FP32 */
    float fisher1[EWC_H1][EWC_IN];
    float fisher2[EWC_H2][EWC_H1];
    float fisher3[EWC_OUT][EWC_H2];

    float lambda;   /* Coefficient EWC — depuis configs/ewc_config.yaml */

    /* Poids de référence tâche précédente θ* — MEM: 3 Ko @ FP32 */
    float star_w1[EWC_H1][EWC_IN];
    float star_w2[EWC_H2][EWC_H1];
    float star_w3[EWC_OUT][EWC_H2];
} EWCHead;

void ewc_init(EWCHead *h);   /* Xavier LCG seed=42, zero fisher/star_w, ne touche pas lambda */
void ewc_forward(const EWCHead *h, const float *x, float *out);
int  ewc_predict(const EWCHead *h, const float *x);
void ewc_sgd_step(EWCHead *h, const float *x, int label);
void ewc_consolidate(EWCHead *h, float alpha);
