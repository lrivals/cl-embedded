# S2908 — Prototype CMSIS-DSP `ewc_head_int8_simd.c`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | 🟢 Nice-to-have (exploratoire — si temps disponible après S2904/S2905) |
| **Statut** | ⛔ Bloqué à l'Étape 0 — CMSIS-DSP absent de la toolchain (15 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | `firmware/stm32f4_blink/src/ewc_head_int8.c` ✅ (base à copier) · toolchain CMSIS-DSP (à vérifier) |
| **Fichier cible** | `firmware/stm32f4_blink/src/ewc_head_int8_simd.c` |
| **Références** | `firmware/stm32f4_blink/src/ewc_head_int8.c` · CMSIS-DSP `arm_math.h` (`arm_dot_prod_q7`) |

---

## Contexte

Le Cortex-M4 dispose d'une extension DSP (instructions SIMD sur registres 32 bits) permettant 4 opérations q7×q7 par cycle via les instructions `SMLAD` / `SMLABT`. La bibliothèque CMSIS-DSP expose cette capacité via `arm_dot_prod_q7()`.

**Hypothèse théorique** : la boucle du produit scalaire INT8 scalaire (N × `LDRSH` + `MLA`) pourrait être accélérée ×2–4 avec `arm_dot_prod_q7()`. Cependant, EWC FP32 bénéficie déjà du pipeline FPU (1 cycle par FMA) → le gain SIMD INT8 est incertain sur ce Cortex-M4.

**Résultat Sprint 23 déjà connu** : EWC INT8 scalaire = 0.461 ms vs EWC FP32 = 0.251 ms (INT8 **plus lent**). Cette tâche teste si CMSIS-DSP peut réduire l'écart ou même inverser le résultat.

---

## Étape 0 — Vérification prérequis toolchain (BLOQUANT)

```bash
# Vérifier disponibilité de la bibliothèque CMSIS-DSP
arm-none-eabi-gcc --print-file-name=libarm_cortexM4lf_math.a

# Résultats possibles :
# → "libarm_cortexM4lf_math.a"           : ABSENT (gcc rend le nom en entrée) → tâche bloquée
# → "/usr/lib/.../libarm_cortexM4lf_math.a" : PRÉSENT → continuer
```

Si la bibliothèque est absente : documenter dans le bilan S2908 et passer à S2909. Ne pas installer manuellement — noter `TODO(dorra)` pour que la toolchain soit complétée.

```bash
# Alternative : vérifier via apt
dpkg -l | grep cmsis
# ou chercher arm_math.h
find /usr -name "arm_math.h" 2>/dev/null
```

---

## Implémentation `ewc_head_int8_simd.c`

Copier `ewc_head_int8.c` et remplacer uniquement la boucle de produit scalaire dans le forward pass.

**Modification clé dans la couche 1 (exemple)** :

```c
/* Version scalaire (ewc_head_int8.c) : */
for (int i = 0; i < EWC_IN; i++) {
    acc += (int32_t)h->w1[j][i] * (int32_t)x_q7[i];
}

/* Version SIMD CMSIS-DSP (ewc_head_int8_simd.c) : */
#include "arm_math.h"
q31_t acc_q31 = 0;
arm_dot_prod_q7((q7_t *)h->w1[j], (q7_t *)x_q7, EWC_IN, &acc_q31);
/* acc_q31 est en Q14 (somme de EWC_IN produits Q7×Q7) */
int32_t acc = (int32_t)(acc_q31 >> 7);   /* renormaliser en Q7 */
```

**Signature de `arm_dot_prod_q7`** :

```c
void arm_dot_prod_q7(
    const q7_t  *pSrcA,    /* premier vecteur INT8 */
    const q7_t  *pSrcB,    /* second vecteur INT8 */
    uint32_t     blockSize, /* nombre d'éléments */
    q31_t       *result     /* accumulateur Q31 résultat */
);
/* Précondition : blockSize multiple de 4 recommandé pour le SIMD.
 * EWC_IN=9, EWC_H1=16 ne sont pas multiples de 4 → vérifier si la lib gère les tails. */
```

> `TODO(dorra)` : `EWC_IN=9` n'est pas multiple de 4. Vérifier que `arm_dot_prod_q7` gère les tailles arbitraires ou si un padding à 12 est nécessaire dans les tableaux `w1`.

---

## Modifications Makefile

```makefile
# Ajouter dans firmware/stm32f4_blink/Makefile (section LDFLAGS) :
LDFLAGS += -larm_cortexM4lf_math -lm

# Ajouter le fichier source (conditionnel) :
SRCS_SIMD = src/ewc_head_int8_simd.c

# Cible de comparaison (pas dans le firmware principal — prototypage séparé) :
simd_test: $(SRCS_SIMD) ...
```

> La cible `simd_test` est distincte du firmware principal — `ewc_head_int8_simd.c` n'est **pas** intégré au `make all` par défaut (prototypage exploratoire, pas en production).

---

## Protocole de mesure DWT

Mesurer sur 100 appels forward EWC INT8 (scalaire vs SIMD) avec le même input CWRU (9 features) :

```c
/* Dans un test de benchmark board — ajouter temporairement dans main.c ou test_pipeline.c */
#define N_BENCH 100

/* Scalaire */
uint32_t t0 = DWT->CYCCNT;
for (int k = 0; k < N_BENCH; k++) {
    ewc_int8_forward(&g_ewc_int8, features, logits);
}
uint32_t t_scalar = (DWT->CYCCNT - t0) / N_BENCH;

/* SIMD */
t0 = DWT->CYCCNT;
for (int k = 0; k < N_BENCH; k++) {
    ewc_int8_simd_forward(&g_ewc_int8_simd, features, logits);
}
uint32_t t_simd = (DWT->CYCCNT - t0) / N_BENCH;
```

**Format résultat JSON** :

```json
{
  "exp_id": "exp_S29_simd_bench",
  "model": "ewc_int8_simd",
  "board": "NUCLEO-F439ZI",
  "scalar_cycles_p50": 41580,
  "simd_cycles_p50": null,
  "ratio_simd_over_scalar": null,
  "conclusion": "CMSIS-DSP absent / gain X% / perte Y%",
  "notes": "EWC_IN=9 non-multiple de 4 — padding éventuel"
}
```

Sauvegarder dans `experiments/exp_S29_board_int8/results_simd_bench.json`.

---

## Résultats attendus et interprétation

| Scénario | ratio SIMD/scalaire | Interprétation |
|----------|:------------------:|----------------|
| ratio < 1.0 | SIMD **plus rapide** | CMSIS-DSP efficace → noter comme contribution positive Gap 3 latence |
| ratio ≈ 1.0 | Équivalent | Overhead `arm_dot_prod_q7` annule le gain SIMD sur petits vecteurs (EWC_IN=9, EWC_H1=16) |
| ratio > 1.0 | SIMD **plus lent** | Expected — overhead appel CMSIS + tailles non-multiples de 4. Confirme le résultat négatif S23 |

> `FIXME(gap3)` : quel que soit le résultat, le documenter honnêtement dans `docs/triple_gap.md` (section Gap 3). Si SIMD reste plus lent que FP32, la conclusion pour le manuscrit est : **"Gap 3 est comblé exclusivement sur la réduction RAM (×2.7–4.0×) ; l'accélération latence INT8 sur Cortex-M4 FPU reste un problème ouvert."**

---

## Bilan (à compléter)

| Sous-étape | Résultat | Notes |
|-----------|---------|-------|
| Vérif toolchain CMSIS-DSP | ⛔ Absente | `arm-none-eabi-gcc --print-file-name=libarm_cortexM4lf_math.a` → renvoie le nom brut (introuvable) ; `find /usr -name arm_math.h` → aucun résultat |
| `ewc_head_int8_simd.c` compilé | ❌ Non créé | Bloqué à l'Étape 0 — installation manuelle proscrite par la spec |
| Cycles scalaire P50 | N/A | Référence connue Sprint 23 : EWC INT8 = 462 µs P50 (≈ 0.461 ms) |
| Cycles SIMD P50 | N/A | Non mesurable sans CMSIS-DSP |
| Ratio SIMD/scalaire | N/A | — |

> **Conclusion S2908** : CMSIS-DSP absent de la toolchain `arm-none-eabi-gcc` installée. La
> tâche est exploratoire / nice-to-have ; conformément à l'Étape 0 (BLOQUANT), elle est
> documentée comme bloquée sans installation manuelle. Résultat consigné dans
> `experiments/exp_S29_board_int8/results_simd_bench.json`.
>
> `TODO(dorra)` : compléter la toolchain avec CMSIS-DSP (`libarm_cortexM4lf_math.a` +
> `arm_math.h`) pour débloquer le prototype `arm_dot_prod_q7`. Tant que le SIMD INT8 n'est
> pas mesuré, la conclusion manuscrit reste : **« Gap 3 est comblé exclusivement sur la
> réduction RAM (×2.7–4.0×) ; l'accélération latence INT8 sur Cortex-M4 FPU reste un
> problème ouvert. »** (cf. résultat négatif Sprint 23, EWC INT8 1.84× plus lent que FP32).
