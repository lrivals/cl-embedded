# S2601–S2602 — Tête C EWC Régression : `ewc_head_regression.c` + `.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2601 : 3h / S2602 : 30 min = 3h30 total |
| **Dépendances** | `firmware/stm32f4_blink/src/ewc_head.c` ✅ (pattern à copier-adapter), `firmware/stm32f4_blink/inc/ewc_head.h` ✅, Sprint 25 ✅ (`ewc_mlp_regression.py` validé, exp_S25_01 disponible) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_regression.c`, `firmware/stm32f4_blink/inc/ewc_head_regression.h` |
| **Référence** | `firmware/stm32f4_blink/src/ewc_head.c` (pattern complet : init Xavier LCG, forward, sgd_step, consolidate), `src/models/ewc/ewc_mlp_regression.py` (spec Python équivalente) |

---

## Contexte

`ewc_head.c` implémente un MLP 3 couches pour **classification binaire** (2 sorties, softmax + cross-entropy). Sprint 26 requiert une variante pour la **régression RUL continue** (1 sortie linéaire, MSE). Les différences par rapport à `ewc_head.c` sont minimales :

| Aspect | `ewc_head.c` (binaire) | `ewc_head_regression.c` (RUL) |
|--------|----------------------|------------------------------|
| Sorties | `EWC_OUT = 2` (logits) | `EWC_REG_OUT = 1` (scalaire) |
| Activation finale | aucune (logits → softmax dans sgd_step) | aucune (régression linéaire) |
| Perte | Cross-entropy (softmax – one_hot) | MSE : gradient = `ŷ - y` |
| Fisher | identique (`w²` proxy) | identique |
| `predict()` | `argmax(logits)` → 0 ou 1 | directement `out[0]` (RUL float) |

**Règle d'implémentation** : allocation entièrement statique (tableaux globaux dans `.bss`), pas de `malloc`. Chaque variable de taille (couche, dimension) passe par un `#define` dans le header.

---

## S2602 — `firmware/stm32f4_blink/inc/ewc_head_regression.h`

```c
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
```

---

## S2601 — `firmware/stm32f4_blink/src/ewc_head_regression.c`

### Spec complète

```c
/**
 * ewc_head_regression.c — Tête MLP 3 couches EWC pour régression RUL sur MCU
 *
 * Forward : Linear(5→32)+ReLU → Linear(32→16)+ReLU → Linear(16→1) [linéaire]
 * Update  : SGD 1 step, perte MSE + terme EWC diagonal
 *           Gradient sortie : dL/d(out) = out[0] - y_true   (MSE, pas de softmax)
 *
 * Différences vs ewc_head.c :
 *   - EWC_REG_OUT = 1  (au lieu de 2)
 *   - Pas de softmax dans sgd_step — gradient direct = ŷ - y
 *   - ewc_reg_predict() retourne un float (RUL), pas un int
 *
 * Backprop entièrement en stack local : pas de malloc.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : ewc_head.c (pattern), ewc_mlp_regression.py (spec Python)
 */

#include "ewc_head_regression.h"
#include <math.h>

/* ── Utilitaires locaux ───────────────────────────────────────────────────── */

static float relu(float v) { return v > 0.0f ? v : 0.0f; }

/* ── Initialisation Xavier LCG ────────────────────────────────────────────── */

/* MEM: ewc_reg_init — 0 B stack extra.
 * LCG Knuth (même seed=42 que ewc_head.c — valeurs Xavier différentes car EWC_REG_* dims) */
void ewc_reg_init(EWCHeadReg *h)
{
    uint32_t rng = 42u;
#define LCG_NEXT(r) ((r) = (r) * 1664525u + 1013904223u)
#define LCG_F01(r)  ((float)((r) >> 8) / (float)(1u << 24))

    /* Xavier uniform — limit = sqrt(6 / (fan_in + fan_out)) */
    static const float lim1 = 0.4026f;   /* sqrt(6/(5+32))  */
    static const float lim2 = 0.3536f;   /* sqrt(6/(32+16)) */
    static const float lim3 = 0.6667f;   /* sqrt(6/(16+1))  */

    for (int j = 0; j < EWC_REG_H1; j++) {
        h->b1[j] = 0.0f;
        for (int i = 0; i < EWC_REG_IN; i++) {
            LCG_NEXT(rng);
            h->w1[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim1;
            h->fisher1[j][i] = 0.0f;
            h->star_w1[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        h->b2[j] = 0.0f;
        for (int i = 0; i < EWC_REG_H1; i++) {
            LCG_NEXT(rng);
            h->w2[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim2;
            h->fisher2[j][i] = 0.0f;
            h->star_w2[j][i] = 0.0f;
        }
    }
    /* Couche de sortie : EWC_REG_OUT = 1 neurone */
    for (int j = 0; j < EWC_REG_OUT; j++) {
        h->b3[j] = 0.0f;
        for (int i = 0; i < EWC_REG_H2; i++) {
            LCG_NEXT(rng);
            h->w3[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim3;
            h->fisher3[j][i] = 0.0f;
            h->star_w3[j][i] = 0.0f;
        }
    }
#undef LCG_NEXT
#undef LCG_F01
}

/* ── Forward pass ─────────────────────────────────────────────────────────── */

/* MEM: h1 = 128 B, h2 = 64 B (stack local) */
void ewc_reg_forward(const EWCHeadReg *h, const float *x, float *out)
{
    float h1[EWC_REG_H1];  /* MEM: 128 B @ FP32 */
    float h2[EWC_REG_H2];  /* MEM:  64 B @ FP32 */

    for (int j = 0; j < EWC_REG_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_REG_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_REG_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    /* Couche de sortie — activation linéaire (pas de Sigmoid, pas de softmax) */
    float acc = h->b3[0];
    for (int i = 0; i < EWC_REG_H2; i++) acc += h->w3[0][i] * h2[i];
    out[0] = acc;   /* RUL prédit, scalaire non borné */
}

float ewc_reg_predict(const EWCHeadReg *h, const float *x)
{
    float out[EWC_REG_OUT];  /* MEM: 4 B @ FP32 */
    ewc_reg_forward(h, x, out);
    return out[0];
}

/* ── SGD step avec perte MSE + régularisation EWC ─────────────────────────── */

/* MEM stack: h1(128B) + h2(64B) + dh2(64B) + dh1(128B) = 384 B */
void ewc_reg_sgd_step(EWCHeadReg *h, const float *x, float y_true)
{
    float h1[EWC_REG_H1];   /* MEM: 128 B @ FP32 */
    float h2[EWC_REG_H2];   /* MEM:  64 B @ FP32 */
    float out[EWC_REG_OUT]; /* MEM:   4 B @ FP32 */

    /* ── 1. Forward ──────────────────────────────────────────────────────── */
    for (int j = 0; j < EWC_REG_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_REG_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_REG_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_REG_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    {
        float acc = h->b3[0];
        for (int i = 0; i < EWC_REG_H2; i++) acc += h->w3[0][i] * h2[i];
        out[0] = acc;
    }

    /* ── 2. Gradient MSE sur la sortie ───────────────────────────────────
     * L = (ŷ - y)² / 2  →  dL/dŷ = ŷ - y                             */
    float dout = out[0] - y_true;  /* scalaire — 1 neurone de sortie */

    /* ── 3. Backward couche 3 + EWC + SGD ───────────────────────────────
     * dh2[i] = w3[0][i] * dout                                         */
    float dh2[EWC_REG_H2];   /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_REG_H2; i++) {
        float grad = dout * h2[i]
                   + h->lambda * h->fisher3[0][i] * (h->w3[0][i] - h->star_w3[0][i]);
        h->w3[0][i] -= EWC_REG_LR * grad;
        dh2[i] = h->w3[0][i] * dout;
    }
    h->b3[0] -= EWC_REG_LR * dout;

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_REG_H2; i++) {
        dh2[i] *= (h2[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 4. Backward couche 2 + EWC + SGD ─────────────────────────────── */
    float dh1[EWC_REG_H1];   /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_REG_H1; i++) dh1[i] = 0.0f;
    for (int j = 0; j < EWC_REG_H2; j++) {
        for (int i = 0; i < EWC_REG_H1; i++) {
            float grad = dh2[j] * h1[i]
                       + h->lambda * h->fisher2[j][i] * (h->w2[j][i] - h->star_w2[j][i]);
            h->w2[j][i] -= EWC_REG_LR * grad;
            dh1[i] += h->w2[j][i] * dh2[j];
        }
        h->b2[j] -= EWC_REG_LR * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_REG_H1; i++) {
        dh1[i] *= (h1[i] > 0.0f ? 1.0f : 0.0f);
    }

    /* ── 5. Backward couche 1 + EWC + SGD ─────────────────────────────── */
    for (int j = 0; j < EWC_REG_H1; j++) {
        for (int i = 0; i < EWC_REG_IN; i++) {
            float grad = dh1[j] * x[i]
                       + h->lambda * h->fisher1[j][i] * (h->w1[j][i] - h->star_w1[j][i]);
            h->w1[j][i] -= EWC_REG_LR * grad;
        }
        h->b1[j] -= EWC_REG_LR * dh1[j];
    }
}

/* ── Consolidation EWC : Fisher EMA + snapshot θ* ────────────────────────── */

/* Identique à ewc_consolidate() — seules les dimensions diffèrent */
void ewc_reg_consolidate(EWCHeadReg *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }

    for (int j = 0; j < EWC_REG_H2; j++)
        for (int i = 0; i < EWC_REG_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }

    for (int j = 0; j < EWC_REG_OUT; j++)
        for (int i = 0; i < EWC_REG_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
}
```

---

## Vérification

### Compilation host (TEST_MODE)

```bash
cd firmware/stm32f4_blink
make test   # doit inclure test_ewc_regression.c — voir S2614
```

### Footprint ARM

```bash
# Après make all :
arm-none-eabi-size build/stm32f4_blink.elf
# Vérifier que la section .bss reste < 256 Ko
# EWCHeadReg contribue ~8.88 Ko en .bss
```

### Test rapide C (inline)

```c
/* Dans test_ewc_regression.c (S2614) : */
EWCHeadReg h;
ewc_reg_init(&h);
h.lambda = 400.0f;

float x[EWC_REG_IN] = {0.5f, -0.3f, 1.2f, 0.0f, -0.8f};
float out[EWC_REG_OUT];
ewc_reg_forward(&h, x, out);
/* out[0] est un float non borné — pas une proba */

/* SGD step avec y_true = 85.0 (RUL en cycles CMAPSS) */
ewc_reg_sgd_step(&h, x, 85.0f);

/* Consolidation */
ewc_reg_consolidate(&h, EWC_REG_FISHER_DECAY);
/* Vérifier fisher1[0][0] != 0.0 après consolidation */
```

---

## Budget mémoire détaillé (NUCLEO-F439ZI)

| Composant | Octets (FP32) | Commentaire |
|-----------|-------------|-------------|
| `w1[32][5]` + `b1[32]` | 704 B | couche 1 poids |
| `w2[16][32]` + `b2[16]` | 2 112 B | couche 2 poids |
| `w3[1][16]` + `b3[1]` | 68 B | couche sortie |
| Fisher (3 matrices) | identique poids | ~2 884 B |
| `star_w` (3 matrices) | identique poids | ~2 884 B |
| `lambda` (1 float) | 4 B | |
| **Total EWCHeadReg** | **~8 884 B (~8.7 Ko)** | << 256 Ko ✅ |
| Stack max `ewc_reg_sgd_step` | 388 B | h1+h2+dh2+dh1+out |

---

## Résultats d'implémentation

| Sous-tâche                       | Statut | Notes                                          |
|----------------------------------|:------:|------------------------------------------------|
| S2602 — `ewc_head_regression.h`  | ✅     | Implémenté conforme spec                       |
| S2601 — `ewc_head_regression.c`  | ✅     | Implémenté conforme spec, `make test` 57/57 ✅ |

---

## Questions ouvertes

- `TODO(dorra)` : Le taux d'apprentissage `EWC_REG_LR = 0.001f` est plus faible que `EWC_LR = 0.01f` pour éviter la divergence MSE. Valider sur board que la convergence est suffisante en 100–500 samples CMAPSS FD001 (online).
- `FIXME(gap2)` : Le gradient MSE `dout = ŷ - y` n'est pas borné contrairement à `softmax - one_hot ∈ [-1, 1]`. Vérifier l'absence de NaN/Inf en pratique avec des données CMAPSS normalisées (clip de `dout` à `[-10, 10]` si instabilité).
- `TODO(arnaud)` : La proxy Fisher `w²` (Kirkpatrick Online) est correcte pour MSE ? Pour la classification, elle approxime `E[∇² log p]`. Pour la régression, `log p(y|x) ∝ -(ŷ-y)²` — le gradient carré reste une approximation valide selon Schwarz et al. 2018. Confirmer pour le manuscrit.
