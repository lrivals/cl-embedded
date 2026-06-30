# S2221–S2222 — Gap 3 : INT8 Portage C (validation board = Sprint 23)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 22 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h + 2h = 6h |
| **Dépendances** | S2217 ✅ (`ewc_mlp_int8.py` validé, critère Gap 3 Python atteint), `firmware/stm32f4_blink/src/ewc_head.c` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_int8.c`, `firmware/stm32f4_blink/inc/ewc_head_int8.h`, `firmware/stm32f4_blink/tests/test_ewc_int8.c` |
| **Référence** | `firmware/stm32f4_blink/src/ewc_head.c` (FP32 à adapter), `Benatti2019HDC` (Q7 sur MCU), `Ravaglia2021QLRCL` |

---

## Contexte

`ewc_head.c` implémente le forward + update EWC en FP32 sur Cortex-M4. `ewc_head_int8.c` est sa variante INT8 : les activations sont stockées en **Q7** (int8_t, range [-128, 127]) et les accumulateurs MAC en **Q15** (int16_t) pour éviter l'overflow.

**Ce sprint se limite au portage C + tests x86.** La validation board (mesure de latence INT8 vs FP32, DWT) est remise au Sprint 23 (S2307).

**Stratégie** : copier `ewc_head.c` comme base, remplacer `float` → `int8_t`/`int16_t` avec des opérations MAC entières, ajouter les fonctions de quantization/dequantization.

---

## S2221 — `firmware/stm32f4_blink/src/ewc_head_int8.c`

### Représentation en virgule fixe

| Variable | Type C | Format | Plage |
|----------|--------|--------|-------|
| Activations | `int8_t` | Q7 (1 bit signe, 7 bits fraction) | [-1.0, 0.9921875] |
| Accumulateurs | `int16_t` | Q15 | [-32768, 32767] |
| Poids | `int8_t` | Q7 | [-1.0, 0.9921875] |
| Facteurs de scale | `float` | FP32 | (calibré hors board) |

### En-tête `ewc_head_int8.h`

```c
/**
 * ewc_head_int8.h — Tête MLP EWC quantifiée INT8 (Q7/Q15) pour MCU
 *
 * Forward  : activations Q7 (int8_t), accumulateurs Q15 (int16_t)
 * Update   : SGD 1 step en Q7, terme EWC diagonal Q7
 * Pas de malloc — tout alloué statiquement dans EWCHeadInt8.
 *
 * Validation board : Sprint 23 (S2307).
 * Référence Python : src/models/ewc/ewc_mlp_int8.py
 */

#ifndef EWC_HEAD_INT8_H
#define EWC_HEAD_INT8_H

#include <stdint.h>
#include "ewc_head.h"   /* réutilise EWC_IN, EWC_H1, EWC_H2, EWC_OUT */

/* Constantes de quantization (calibrées sur données d'entraînement) */
#define INT8_SCALE_W1    (1.0f / 128.0f)   /* scale poids couche 1 */
#define INT8_SCALE_W2    (1.0f / 128.0f)   /* scale poids couche 2 */
#define INT8_SCALE_W3    (1.0f / 128.0f)   /* scale poids couche 3 */
#define INT8_SCALE_ACT   (1.0f / 128.0f)   /* scale activations post-ReLU */

/* Structure INT8 — mêmes dimensions que EWCHead mais types entiers */
/* MEM: EWCHeadInt8 = ~2.4 Ko @ INT8 (vs ~9.7 Ko @ FP32 pour EWCHead)
 *   w1 = 5×32 = 160 B, w2 = 32×16 = 512 B, w3 = 16×2 = 32 B
 *   b1 = 32×4 B = 128 B (biais restent FP32), b2/b3 idem
 *   fisher1..3 = INT8 × mêmes tailles = 704 B
 *   star_w1..3 = INT8 × mêmes tailles = 704 B          */
typedef struct {
    int8_t  w1[EWC_H1][EWC_IN];
    float   b1[EWC_H1];             /* biais FP32 — impact mémoire faible */
    int8_t  w2[EWC_H2][EWC_H1];
    float   b2[EWC_H2];
    int8_t  w3[EWC_OUT][EWC_H2];
    float   b3[EWC_OUT];

    int8_t  fisher1[EWC_H1][EWC_IN];
    int8_t  fisher2[EWC_H2][EWC_H1];
    int8_t  fisher3[EWC_OUT][EWC_H2];

    int8_t  star_w1[EWC_H1][EWC_IN];
    int8_t  star_w2[EWC_H2][EWC_H1];
    int8_t  star_w3[EWC_OUT][EWC_H2];

    float   lambda;
    float   scale_w;   /* scale commune poids */
    float   scale_act; /* scale activations */
    uint8_t task_id;
} EWCHeadInt8;

/* API publique */
void ewc_int8_init(EWCHeadInt8 *h);
void ewc_int8_from_fp32(EWCHeadInt8 *dst, const EWCHead *src_fp32);
void ewc_int8_forward(const EWCHeadInt8 *h, const int8_t *x_q7, float *logits);
void ewc_int8_update(EWCHeadInt8 *h, const int8_t *x_q7, uint8_t label,
                     float lr, int fisher_ema);
void ewc_int8_consolidate(EWCHeadInt8 *h);

/* Utilitaires Q7 */
static inline int8_t  float_to_q7(float v) { return (int8_t)(v * 128.0f); }
static inline float   q7_to_float(int8_t v) { return (float)v / 128.0f; }
static inline int8_t  relu_q7(int8_t v) { return v > 0 ? v : 0; }

#endif /* EWC_HEAD_INT8_H */
```

### Fonctions à implémenter dans `ewc_head_int8.c`

```c
/**
 * ewc_int8_forward — Forward pass Q7
 *
 * Calcule h1 = relu(w1·x + b1), h2 = relu(w2·h1 + b2), logits = w3·h2 + b3
 * en accumulateurs Q15 (int16_t) pour éviter l'overflow.
 *
 * MEM: stack local h1[32] = 32 B Q7, h2[16] = 16 B Q7, acc[EWC_H1] = 64 B Q15
 */
void ewc_int8_forward(const EWCHeadInt8 *h, const int8_t *x_q7, float *logits)
{
    int8_t  h1[EWC_H1];
    int8_t  h2[EWC_H2];

    /* Couche 1 : accumulation Q15, saturation → Q7 + ReLU */
    for (int j = 0; j < EWC_H1; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_IN; i++) {
            acc += (int16_t)h->w1[j][i] * (int16_t)x_q7[i];
        }
        /* Scale : Q7×Q7 = Q14, décalage >> 7 → Q7 */
        float val = (float)(acc >> 7) / 128.0f + h->b1[j];
        h1[j] = relu_q7(float_to_q7(val));
    }

    /* Couche 2 : même logique */
    for (int j = 0; j < EWC_H2; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_H1; i++) {
            acc += (int16_t)h->w2[j][i] * (int16_t)h1[i];
        }
        float val = (float)(acc >> 7) / 128.0f + h->b2[j];
        h2[j] = relu_q7(float_to_q7(val));
    }

    /* Couche 3 : sortie FP32 (logits) pour compatibilité softmax/sigmoid */
    for (int j = 0; j < EWC_OUT; j++) {
        int16_t acc = 0;
        for (int i = 0; i < EWC_H2; i++) {
            acc += (int16_t)h->w3[j][i] * (int16_t)h2[i];
        }
        logits[j] = (float)(acc >> 7) / 128.0f + h->b3[j];
    }
}
```

### Compilabilité `arm-none-eabi-gcc`

```bash
# Test compilabilité sans board (x86 natif)
gcc -O2 -Wall -Wextra \
    -I firmware/stm32f4_blink/inc \
    firmware/stm32f4_blink/src/ewc_head_int8.c \
    firmware/stm32f4_blink/src/ewc_head.c \
    -lm -o /tmp/test_ewc_int8_x86

# Test compilabilité croisée ARM (sans linker)
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -O2 -Wall -c \
    -I firmware/stm32f4_blink/inc \
    firmware/stm32f4_blink/src/ewc_head_int8.c \
    -o /tmp/ewc_head_int8.o

echo "Compilation ARM OK"
```

---

## S2222 — Tests Unity `firmware/stm32f4_blink/tests/test_ewc_int8.c`

### Tests à implémenter

```c
/**
 * test_ewc_int8.c — Tests Unity pour ewc_head_int8.c
 *
 * Exécution sur x86 (pas de board requis).
 * Critère : delta output INT8 vs FP32 < 0.05 sur chaque logit.
 * Build : gcc -I unity/src unity/src/unity.c test_ewc_int8.c ewc_head_int8.c ewc_head.c -lm
 */

void test_forward_output_close_to_fp32(void)
{
    /* Initialiser EWCHead (FP32) et EWCHeadInt8 avec mêmes poids */
    EWCHead     fp32;
    EWCHeadInt8 int8;
    ewc_init(&fp32);
    ewc_int8_from_fp32(&int8, &fp32);

    /* Input de test : x = [0.1, -0.3, 0.5, 0.0, -0.2] en Q7 */
    float   x_f[5]   = {0.1f, -0.3f, 0.5f, 0.0f, -0.2f};
    int8_t  x_q7[5];
    for (int i = 0; i < 5; i++) x_q7[i] = float_to_q7(x_f[i]);

    /* Forward FP32 */
    float out_fp32[EWC_OUT];
    ewc_forward(&fp32, x_f, out_fp32);

    /* Forward INT8 */
    float out_int8[EWC_OUT];
    ewc_int8_forward(&int8, x_q7, out_int8);

    /* Critère : delta < 0.05 sur chaque logit */
    for (int j = 0; j < EWC_OUT; j++) {
        float delta = fabsf(out_fp32[j] - out_int8[j]);
        TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.05f, out_fp32[j], out_int8[j],
            "INT8 vs FP32 delta > 0.05");
    }
}

void test_update_does_not_crash(void) { /* SGD INT8 step sans assert */ }
void test_consolidate_updates_fisher(void) { /* star_w1 == w1 après consolidate */ }
void test_relu_q7_clamps_negative(void) { /* relu_q7(-50) == 0 */ }
void test_float_q7_roundtrip(void) { /* q7_to_float(float_to_q7(0.5)) ≈ 0.5 ± 0.01 */ }
```

### Build et exécution x86

```bash
# Compiler les tests Unity (x86)
gcc -O0 -g \
    -I firmware/stm32f4_blink/inc \
    -I unity/src \
    unity/src/unity.c \
    firmware/stm32f4_blink/tests/test_ewc_int8.c \
    firmware/stm32f4_blink/src/ewc_head_int8.c \
    firmware/stm32f4_blink/src/ewc_head.c \
    -lm \
    -o /tmp/test_ewc_int8_runner

/tmp/test_ewc_int8_runner   # tous les tests doivent passer (OK)
```

---

## Vérification end-to-end Sprint 22

```bash
# 1. Compilabilité ARM sans board
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 \
    -I firmware/stm32f4_blink/inc -c \
    firmware/stm32f4_blink/src/ewc_head_int8.c -o /tmp/ewc_head_int8.o
echo "ARM compilation: $?"

# 2. Tests Unity x86
gcc -O0 -I firmware/stm32f4_blink/inc -I unity/src \
    unity/src/unity.c \
    firmware/stm32f4_blink/tests/test_ewc_int8.c \
    firmware/stm32f4_blink/src/ewc_head_int8.c \
    firmware/stm32f4_blink/src/ewc_head.c -lm \
    -o /tmp/test_ewc_int8_runner && /tmp/test_ewc_int8_runner
```

---

## Questions ouvertes

- `TODO(dorra)` : Les biais restent FP32 dans cette implémentation (cohérence avec fake-quant Python). Faut-il les quantifier aussi en Q15 pour la version MCU finale ?
- `TODO(dorra)` : Le décalage `>> 7` pour passer de Q14 à Q7 suppose que les poids et activations sont dans [-1, 1]. Valider cette hypothèse après calibration sur données réelles.
- `FIXME(gap3)` : La mesure de latence INT8 vs FP32 sur NUCLEO (DWT) est réservée Sprint 23 (S2307). Ne pas conclure sur la latence board sans la mesure réelle.
