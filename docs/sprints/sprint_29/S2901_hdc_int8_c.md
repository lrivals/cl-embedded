# S2901 — Firmware C `hdc_int8.c` + `hdc_int8.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (12 juin 2026) — board flashée, .bss total firmware = 104 576 B (102 Ko) < 128 Ko ✅ |
| **Durée estimée** | 3h |
| **Dépendances** | Sprint 28 ✅ (`hdc_int8.py`) · `firmware/stm32f4_blink/src/hdc.c` ✅ (pattern FP32 à adapter) · `firmware/stm32f4_blink/src/ewc_head_int8.c` ✅ (pattern INT8 à suivre) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/hdc_int8.c`, `firmware/stm32f4_blink/inc/hdc_int8.h` |
| **Références** | `firmware/stm32f4_blink/src/hdc.c` (FP32 complet) · `firmware/stm32f4_blink/src/ewc_head_int8.c` (pattern INT8) |

---

## Contexte

`hdc.c` implémente HDC FP32 avec `hdc_binarize()` + `hdc_retrain()` + `hdc_update_with_sample()`. `hdc_int8.c` est une variante avec stockage INT8 des hypervecteurs de base et AM INT16 — même comportement, empreinte mémoire réduite ~×3.

**Différences vs `hdc.c`** :

| Aspect | `hdc.c` (FP32) | `hdc_int8.c` (INT8) |
|--------|:--------------:|:-------------------:|
| Base vectors | `float bv[HDC_N][HDC_D]` | `int8_t bv[HDC_N][HDC_D]` |
| Associative memory | `float am[HDC_C][HDC_D]` | `int16_t am[HDC_C][HDC_D]` |
| Produit scalaire query | FP32 accumulation | INT32 accumulation, normalize to float |
| Bundle update | `am[c][i] += hv[i]` (float) | `am[c][i] += (int16_t)hv[i]` |

---

## Spec header `hdc_int8.h`

```c
#pragma once
#include <stdint.h>

/* Architecture : n_features → binarize → int8 hypervector D → AM int16
 *
 * MEM total HDCInt8 (HDC_I_N=9, HDC_I_D=2048, HDC_I_C=4) :
 *   base_vecs : HDC_I_N × HDC_I_D × 1B = 18 432 B @ INT8
 *   am        : HDC_I_C × HDC_I_D × 2B = 16 384 B @ INT16
 *   TOTAL     : ~34 816 B (~34 Ko) en .bss
 *   vs FP32   : ~106 Ko hypothétique → compression ×3.06
 */

#define HDC_I_N   9      /* Nombre de features d'entrée */
#define HDC_I_D   2048   /* Dimension des hypervecteurs */
#define HDC_I_C   4      /* Nombre de classes */

typedef struct {
    int8_t  bv[HDC_I_N][HDC_I_D];    /* Base vectors — MEM: 18 432 B @ INT8 */
    int16_t am[HDC_I_C][HDC_I_D];    /* Associative memory — MEM: 16 384 B @ INT16 */
} HDCInt8;

void  hdc_int8_init(HDCInt8 *h);
    /* Initialise base vectors avec binarisation LCG (±1 → int8), zéro AM */

void  hdc_int8_encode(const HDCInt8 *h, const float *x, int8_t *hv_out);
    /* Encode features → hypervecteur int8 par binding + bundling */

int   hdc_int8_predict(const HDCInt8 *h, const int8_t *hv);
    /* Query AM : retourne argmax(dot_product(hv, am[c])) */

void  hdc_int8_update(HDCInt8 *h, const int8_t *hv, int label);
    /* Online update : am[label][i] += hv[i] (accumulation int16) */
```

---

## Budget mémoire (NUCLEO-F439ZI)

| Composant | Octets | Notes |
|-----------|-------:|-------|
| `HDCInt8.bv[9][2048]` | 18 432 B | INT8 base vectors |
| `HDCInt8.am[4][2048]` | 16 384 B | INT16 AM |
| Stack `hdc_int8_encode` | ~2 Ko | hv_out int8[2048] = 2 Ko |
| Firmware existant (Sprint 26) | 66 700 B | inchangé |
| **Total .bss estimé avec HDC INT8** | **~101 Ko** | << 256 Ko ✅ |

> **Note** : hv_out `int8_t[2048]` = 2 Ko en stack local pendant l'encodage. À passer en paramètre ou en global si stack overflow (stack par défaut STM32F4 = 8 Ko).

---

## Vérification

```bash
cd firmware/stm32f4_blink

# Compilation host (TEST_MODE)
make test   # doit inclure test_hdc_int8.c (S2906)

# Footprint ARM après make all :
arm-none-eabi-size build/stm32f4_blink.elf
# Vérifier .bss < 128 Ko
```
