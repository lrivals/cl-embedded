#pragma once
#include <stdint.h>

/* Architecture : Input(EWC_REG_IN) → ReLU(EWC_REG_H1) → ReLU(EWC_REG_H2) → Output(1)
 *
 * MEM total EWCHeadReg (input_dim=5, H1=32, H2=16) :
 *   Poids   : (5×32+32 + 32×16+16 + 16×1+1) × 4 = ~2.96 Ko @ FP32
 *   Fisher  : identique aux poids               = ~2.96 Ko @ FP32
 *   star_w  : identique aux poids               = ~2.96 Ko @ FP32
 *   TOTAL   : ~8.88 Ko @ FP32 en .bss  ✅ << 256 Ko NUCLEO-F439ZI
 */

#define EWC_REG_IN   5     /* Dimension d'entrée — top-5 features CMAPSS (cmapss_feature_subset.yaml) */
#define EWC_REG_H1  32     /* Neurons couche cachée 1 */
#define EWC_REG_H2  16     /* Neurons couche cachée 2 */
#define EWC_REG_OUT  1     /* Sortie scalaire — RUL continu (pas de Sigmoid) */
#define EWC_REG_LR           0.001f  /* Taux d'apprentissage SGD — plus faible que binaire (MSE non bornée) */
#define EWC_REG_FISHER_DECAY 0.99f   /* EMA decay identique à ewc_head.c */

typedef struct {
    /* Poids courants — MEM: ~2.96 Ko @ FP32 */
    float w1[EWC_REG_H1][EWC_REG_IN];    float b1[EWC_REG_H1];
    float w2[EWC_REG_H2][EWC_REG_H1];    float b2[EWC_REG_H2];
    float w3[EWC_REG_OUT][EWC_REG_H2];   float b3[EWC_REG_OUT];

    /* Fisher diagonale (régularisation EWC) — MEM: ~2.96 Ko @ FP32 */
    float fisher1[EWC_REG_H1][EWC_REG_IN];
    float fisher2[EWC_REG_H2][EWC_REG_H1];
    float fisher3[EWC_REG_OUT][EWC_REG_H2];

    float lambda;   /* Coefficient EWC — depuis configs/board_ewc.yaml */

    /* Poids de référence θ* — MEM: ~2.96 Ko @ FP32 */
    float star_w1[EWC_REG_H1][EWC_REG_IN];
    float star_w2[EWC_REG_H2][EWC_REG_H1];
    float star_w3[EWC_REG_OUT][EWC_REG_H2];
} EWCHeadReg;

void  ewc_reg_init(EWCHeadReg *h);
        /* Xavier LCG seed=42, zero fisher/star_w — identique à ewc_init() */

void  ewc_reg_forward(const EWCHeadReg *h, const float *x, float *out);
        /* out[0] = RUL prédit (scalaire non borné) */

float ewc_reg_predict(const EWCHeadReg *h, const float *x);
        /* Raccourci : retourne out[0] directement */

void  ewc_reg_sgd_step(EWCHeadReg *h, const float *x, float y_true);
        /* SGD 1 step : perte MSE + terme EWC. y_true = RUL réel (float) */

void  ewc_reg_consolidate(EWCHeadReg *h, float alpha);
        /* EMA Fisher + snapshot θ* — identique à ewc_consolidate() */
