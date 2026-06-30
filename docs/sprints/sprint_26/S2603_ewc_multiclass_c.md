# S2603–S2604 — Tête C EWC Multi-classe : `ewc_head_multiclass.c` + `.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée réelle** | S2603 : ~1h / S2604 : ~15 min — 57 tests host PASS |
| **Dépendances** | `firmware/stm32f4_blink/src/ewc_head.c` ✅ (pattern binaire à généraliser), `firmware/stm32f4_blink/inc/ewc_head.h` ✅, Sprint 25 ✅ (`ewc_mlp_multiclass.py` validé, exp_S25_03 CWRU disponible) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_multiclass.c`, `firmware/stm32f4_blink/inc/ewc_head_multiclass.h` |
| **Référence** | `firmware/stm32f4_blink/src/ewc_head.c` (pattern EWC — softmax déjà implémenté pour 2 classes), `src/models/ewc/ewc_mlp_multiclass.py` (spec Python équivalente) |

---

## Contexte

`ewc_head.c` gère **2 classes** (binary : normal / faulty). Sprint 26 requiert une variante pour **N classes** configurable via `#define EWC_MC_N_CLASSES`, sans changer le pattern fondamental :

| Aspect | `ewc_head.c` (binaire) | `ewc_head_multiclass.c` (N classes) |
|--------|----------------------|--------------------------------------|
| `EWC_OUT` | 2 (fixe) | `EWC_MC_N_CLASSES` (configurable) |
| Softmax | 2 logits | N logits (même algo, numériquement stable) |
| Cross-entropy | identique | identique (gradient = `softmax[j] - one_hot[j]`) |
| Fisher proxy | `w²` | `w²` (identique) |
| `predict()` | `argmax` de 2 logits | `argmax` de N logits |

**Cas d'usage** :
- CWRU : `EWC_MC_N_CLASSES 10` (9 features, 10 classes : normal + 3 défauts × 3 sévérités)
- Paderborn : `EWC_MC_N_CLASSES 3` (9 features, 3 états : sain / outer-race / inner-race)

**Contrainte firmware** : les tableaux de la tête sont alloués statiquement. `EWC_MC_N_CLASSES` est fixé à la compilation — pas de `malloc`, pas de VLA. Pour basculer entre 10 et 3 classes, recompiler avec `CFLAGS=-DEWC_MC_N_CLASSES=3`.

---

## S2604 — `firmware/stm32f4_blink/inc/ewc_head_multiclass.h`

```c
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
```

---

## S2603 — `firmware/stm32f4_blink/src/ewc_head_multiclass.c`

### Spec complète

```c
/**
 * ewc_head_multiclass.c — Tête MLP 3 couches EWC pour classification N classes sur MCU
 *
 * Forward : Linear(9→32)+ReLU → Linear(32→16)+ReLU → Linear(16→N_CLASSES)
 * Update  : SGD 1 step, softmax cross-entropy + terme EWC diagonal
 *           Gradient sortie : dL/dlogits[j] = softmax[j] - (j == label ? 1 : 0)
 *
 * Différences vs ewc_head.c (binaire) :
 *   - EWC_MC_N_CLASSES = N (paramétrable à la compilation via Makefile -D)
 *   - EWC_MC_IN = 9 (features CWRU) vs EWC_IN = 5 (features Monitoring)
 *   - ewc_mc_predict() reste argmax — généralisation triviale de la version binaire
 *
 * Backprop entièrement en stack local : pas de malloc.
 * Compatible STM32F439ZI Cortex-M4 FPU.
 * Référence : ewc_head.c (pattern), ewc_mlp_multiclass.py (spec Python)
 */

#include "ewc_head_multiclass.h"
#include <math.h>

/* ── Utilitaires ──────────────────────────────────────────────────────────── */

static float relu(float v) { return v > 0.0f ? v : 0.0f; }

/* ── Initialisation Xavier LCG ────────────────────────────────────────────── */

void ewc_mc_init(EWCHeadMC *h)
{
    uint32_t rng = 42u;
#define LCG_NEXT(r) ((r) = (r) * 1664525u + 1013904223u)
#define LCG_F01(r)  ((float)((r) >> 8) / (float)(1u << 24))

    /* Xavier limits : sqrt(6 / (fan_in + fan_out)) */
    static const float lim1 = 0.3780f;  /* sqrt(6/(9+32)) */
    static const float lim2 = 0.3536f;  /* sqrt(6/(32+16)) */
    /* lim3 dépend de N_CLASSES — calculé statiquement pour N=10 : sqrt(6/26) = 0.4804 */
    /* Pour N=3 : sqrt(6/19) = 0.5623 — différence < 20%, tolérable avec même seed */
    static const float lim3 = 0.4804f;  /* sqrt(6/(16+10)) — CWRU default */

    for (int j = 0; j < EWC_MC_H1; j++) {
        h->b1[j] = 0.0f;
        for (int i = 0; i < EWC_MC_IN; i++) {
            LCG_NEXT(rng);
            h->w1[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim1;
            h->fisher1[j][i] = 0.0f;
            h->star_w1[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        h->b2[j] = 0.0f;
        for (int i = 0; i < EWC_MC_H1; i++) {
            LCG_NEXT(rng);
            h->w2[j][i]      = (LCG_F01(rng) * 2.0f - 1.0f) * lim2;
            h->fisher2[j][i] = 0.0f;
            h->star_w2[j][i] = 0.0f;
        }
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        h->b3[j] = 0.0f;
        for (int i = 0; i < EWC_MC_H2; i++) {
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

/* MEM: h1=128B, h2=64B (stack) + logits[N]=N×4B (fourni par appelant) */
void ewc_mc_forward(const EWCHeadMC *h, const float *x, float *logits)
{
    float h1[EWC_MC_H1];   /* MEM: 128 B @ FP32 */
    float h2[EWC_MC_H2];   /* MEM:  64 B @ FP32 */

    for (int j = 0; j < EWC_MC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_MC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_MC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_MC_H2; i++) acc += h->w3[j][i] * h2[i];
        logits[j] = acc;   /* logits bruts — softmax appliqué dans sgd_step */
    }
}

int ewc_mc_predict(const EWCHeadMC *h, const float *x)
{
    float logits[EWC_MC_N_CLASSES];   /* MEM: N×4 B @ FP32 — 40 B pour N=10 */
    ewc_mc_forward(h, x, logits);
    int best = 0;
    for (int j = 1; j < EWC_MC_N_CLASSES; j++) {
        if (logits[j] > logits[best]) best = j;
    }
    return best;
}

/* ── SGD step avec softmax CE + régularisation EWC ───────────────────────── */

/* MEM stack: h1(128B)+h2(64B)+logits(40B)+dout(40B)+dh2(64B)+dh1(128B) = 464 B pour N=10 */
void ewc_mc_sgd_step(EWCHeadMC *h, const float *x, int label)
{
    float h1[EWC_MC_H1];               /* MEM: 128 B @ FP32 */
    float h2[EWC_MC_H2];               /* MEM:  64 B @ FP32 */
    float logits[EWC_MC_N_CLASSES];    /* MEM: N×4 B — 40 B pour N=10 */

    /* ── 1. Forward ──────────────────────────────────────────────────────── */
    for (int j = 0; j < EWC_MC_H1; j++) {
        float acc = h->b1[j];
        for (int i = 0; i < EWC_MC_IN; i++) acc += h->w1[j][i] * x[i];
        h1[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_H2; j++) {
        float acc = h->b2[j];
        for (int i = 0; i < EWC_MC_H1; i++) acc += h->w2[j][i] * h1[i];
        h2[j] = relu(acc);
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        float acc = h->b3[j];
        for (int i = 0; i < EWC_MC_H2; i++) acc += h->w3[j][i] * h2[i];
        logits[j] = acc;
    }

    /* ── 2. Softmax numériquement stable + gradient CE ───────────────────
     * dout[j] = softmax[j] - one_hot(label)[j]                          */
    float dout[EWC_MC_N_CLASSES];   /* MEM: N×4 B */
    float max_logit = logits[0];
    for (int j = 1; j < EWC_MC_N_CLASSES; j++)
        if (logits[j] > max_logit) max_logit = logits[j];

    float sum_exp = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        dout[j] = expf(logits[j] - max_logit);
        sum_exp += dout[j];
    }
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        dout[j] = dout[j] / sum_exp - (j == label ? 1.0f : 0.0f);
    }

    /* ── 3. Backward couche 3 + EWC + SGD ─────────────────────────────── */
    float dh2[EWC_MC_H2];   /* MEM: 64 B @ FP32 */
    for (int i = 0; i < EWC_MC_H2; i++) dh2[i] = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        for (int i = 0; i < EWC_MC_H2; i++) {
            float grad = dout[j] * h2[i]
                       + h->lambda * h->fisher3[j][i] * (h->w3[j][i] - h->star_w3[j][i]);
            h->w3[j][i] -= EWC_MC_LR * grad;
            dh2[i] += h->w3[j][i] * dout[j];
        }
        h->b3[j] -= EWC_MC_LR * dout[j];
    }

    /* ReLU mask couche 2 */
    for (int i = 0; i < EWC_MC_H2; i++)
        dh2[i] *= (h2[i] > 0.0f ? 1.0f : 0.0f);

    /* ── 4. Backward couche 2 + EWC + SGD ─────────────────────────────── */
    float dh1[EWC_MC_H1];   /* MEM: 128 B @ FP32 */
    for (int i = 0; i < EWC_MC_H1; i++) dh1[i] = 0.0f;
    for (int j = 0; j < EWC_MC_H2; j++) {
        for (int i = 0; i < EWC_MC_H1; i++) {
            float grad = dh2[j] * h1[i]
                       + h->lambda * h->fisher2[j][i] * (h->w2[j][i] - h->star_w2[j][i]);
            h->w2[j][i] -= EWC_MC_LR * grad;
            dh1[i] += h->w2[j][i] * dh2[j];
        }
        h->b2[j] -= EWC_MC_LR * dh2[j];
    }

    /* ReLU mask couche 1 */
    for (int i = 0; i < EWC_MC_H1; i++)
        dh1[i] *= (h1[i] > 0.0f ? 1.0f : 0.0f);

    /* ── 5. Backward couche 1 + EWC + SGD ─────────────────────────────── */
    for (int j = 0; j < EWC_MC_H1; j++) {
        for (int i = 0; i < EWC_MC_IN; i++) {
            float grad = dh1[j] * x[i]
                       + h->lambda * h->fisher1[j][i] * (h->w1[j][i] - h->star_w1[j][i]);
            h->w1[j][i] -= EWC_MC_LR * grad;
        }
        h->b1[j] -= EWC_MC_LR * dh1[j];
    }
}

/* ── Consolidation EWC ────────────────────────────────────────────────────── */

void ewc_mc_consolidate(EWCHeadMC *h, float alpha)
{
    float one_minus_alpha = 1.0f - alpha;

    for (int j = 0; j < EWC_MC_H1; j++)
        for (int i = 0; i < EWC_MC_IN; i++) {
            float g2 = h->w1[j][i] * h->w1[j][i];
            h->fisher1[j][i] = alpha * h->fisher1[j][i] + one_minus_alpha * g2;
            h->star_w1[j][i] = h->w1[j][i];
        }

    for (int j = 0; j < EWC_MC_H2; j++)
        for (int i = 0; i < EWC_MC_H1; i++) {
            float g2 = h->w2[j][i] * h->w2[j][i];
            h->fisher2[j][i] = alpha * h->fisher2[j][i] + one_minus_alpha * g2;
            h->star_w2[j][i] = h->w2[j][i];
        }

    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++) {
            float g2 = h->w3[j][i] * h->w3[j][i];
            h->fisher3[j][i] = alpha * h->fisher3[j][i] + one_minus_alpha * g2;
            h->star_w3[j][i] = h->w3[j][i];
        }
}
```

---

## Vérification

```bash
# Compilation host avec N_CLASSES=10 (CWRU)
make test CFLAGS="-DEWC_MC_N_CLASSES=10"
# → test_ewc_multiclass.c doit passer (voir S2614)

# Footprint CWRU (N=10) vs Paderborn (N=3)
arm-none-eabi-size build/stm32f4_blink.elf
# EWCHeadMC N=10 : ~14 Ko .bss
# EWCHeadMC N=3  : ~10.2 Ko .bss
```

### Vérification softmax (invariant clé)

```c
/* Dans test_ewc_multiclass.c : */
float logits[EWC_MC_N_CLASSES] = {1.0f, 2.0f, 0.5f, -0.3f, ...};  /* N=10 */
/* Calculer softmax manuellement et vérifier sum ≈ 1.0 */
float max_l = logits[0];
float sum = 0.0f;
for (int j = 0; j < EWC_MC_N_CLASSES; j++) if (logits[j] > max_l) max_l = logits[j];
for (int j = 0; j < EWC_MC_N_CLASSES; j++) sum += expf(logits[j] - max_l);
/* sum != 1.0 (c'est sum_exp avant normalisation) — vérifier que dout[j] somme à 0 */
```

---

## Budget mémoire (NUCLEO-F439ZI, N=10)

| Composant | Octets | Commentaire |
|-----------|--------|-------------|
| `w1[32][9]` + `b1[32]` | 1 280 B | couche 1 |
| `w2[16][32]` + `b2[16]` | 2 112 B | couche 2 |
| `w3[10][16]` + `b3[10]` | 680 B | couche sortie |
| Fisher (3 matrices) | ~4 072 B | identique poids |
| `star_w` (3 matrices) | ~4 072 B | identique poids |
| **Total EWCHeadMC N=10** | **~14 Ko** | << 256 Ko ✅ |
| Stack max `ewc_mc_sgd_step` (N=10) | ~464 B | h1+h2+logits+dout+dh2+dh1 |

---

## Résultats d'implémentation

| Sous-tâche                       | Statut | Notes                         |
|----------------------------------|:------:|-------------------------------|
| S2604 — `ewc_head_multiclass.h`  | ✅     | Créé — 57 tests host passent  |
| S2603 — `ewc_head_multiclass.c`  | ✅     | Créé — 57 tests host passent  |

---

## Questions ouvertes

- `TODO(dorra)` : Pour la matrice de confusion on-board (`OnlineF1Macro`, S2606), 10×10 = 400 B en `int16`. Si on bascule en Paderborn N=3, économiser 328 B — la matrice pourrait être allouée dynamiquement en tenant compte de `EWC_MC_N_CLASSES`. Vérifier si `MAX_CLASSES=10` en dur est acceptable.
- `FIXME(gap2)` : Stack peak `ewc_mc_sgd_step` = 464 B pour N=10. Le linker script fixe `_Min_Stack_Size = 0x400` (1 Ko). Valider avec `arm-none-eabi-nm` ou mapfile que le stack ne déborde pas avec les autres frames actives (HDC encode → 4 Ko stack pour `hv[]` — ne pas utiliser les deux simultanément).
- `TODO(arnaud)` : L'expérience `exp_S26_02` utilise CWRU 3 tâches (task0 = outer, task1 = inner, task2 = ball). Confirmer que les 10 classes sont actives dès la tâche 0 ou si le F1-macro doit être calculé seulement sur les classes vues (partiel).
