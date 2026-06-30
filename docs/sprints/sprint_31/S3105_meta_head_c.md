# S3105 — Firmware C `meta_head.c` + `meta_head.h`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 31 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3101 (`MetaLearner` + `export_weights`) · `scripts/export_weights_c.py` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/meta_head.c`, `firmware/stm32f4_blink/inc/meta_head.h` |
| **Références** | `firmware/stm32f4_blink/src/ewc_head.c` (pattern forward), `scripts/export_weights_c.py` |

---

## Contexte

Porte le méta-modèle (logreg ou petit MLP) en C pour exécution board dans le mode triple-modèle (S3106). Très petit (quelques features en entrée) → empreinte négligeable.

---

## Spec header `meta_head.h`

```c
#pragma once
#include <stdint.h>

/* Méta-modèle stacking : META_N_FEATURES entrées → verdict binaire.
 * MEM total : poids META_N_FEATURES[(×META_HIDDEN)] FP32 → quelques centaines d'octets.
 * Poids générés par scripts/export_weights_c.py (NE PAS éditer à la main). */

#define META_N_FEATURES 4    /* [score_maha, prob_sup, disagreement, conf_sup] */
#define META_HIDDEN     0    /* 0 = logreg ; >0 = MLP 1 couche cachée */

typedef struct {
    float w[META_N_FEATURES];   /* MEM: 16 B @ FP32 (logreg) */
    float b;
} MetaHead;

void  meta_init(MetaHead *m);
float meta_forward(const MetaHead *m, const float *feats);  /* sigmoïde */
int   meta_predict(const MetaHead *m, const float *feats);  /* 0/1 */
```

- Toute taille via `#define` (règle CLAUDE.md). Allocation statique, pas de malloc.
- Si MLP : ajouter `META_HIDDEN` + matrice cachée, init depuis `model_weights.h`.

---

## Vérification

```bash
# Génère inc/meta_weights.h (+ tests/test_vectors_meta.h pour la parité)
python scripts/export_weights_c.py --meta experiments/exp_S31_PC_maha_ewc_cwru/meta_weights.json --dump-test-vectors
cd firmware/stm32f4_blink && make all && arm-none-eabi-size build/stm32f4_blink.elf
make test   # incl. test_meta_head.c (S3112) : parité forward C ↔ Python
```

---

## Bilan d'implémentation ✅

- **`inc/meta_head.h`** : struct `MetaHead` (logreg + variante MLP gardée par `#if META_HIDDEN > 0`), API `meta_init/meta_forward/meta_predict`. Inclut `meta_weights.h` (généré). Toute taille via `#define`, alloc statique.
- **`src/meta_head.c`** : `meta_forward` = `sigmoid(w·x + b)` (logreg) ou `sigmoid(w2·relu(W1·x + b1) + b2)` (MLP). Pattern d'accumulation FP32 calqué sur `ewc_forward`.
- **`scripts/export_weights_c.py`** : nouvelle option `--meta <meta_weights.json>` → `export_meta_to_c()` génère `inc/meta_weights.h` (`META_N_FEATURES`, `META_HIDDEN`, `META_W/META_B` ou `META_W1/B1/W2/B2`). `export_meta_test_vectors_h()` génère `tests/test_vectors_meta.h` (feats de référence + sortie sigmoïde numpy attendue).
- **Empreinte** : logreg `MetaHead` ≈ 20 B en `.bss` (`META_N_FEATURES=4`). `.bss` total board = **104 596 B** (39.9 % de 256 Ko, +20 B vs Sprint 30).
- **Parité C↔Python** : `test_meta_head.c` 4/4 PASS (`meta_forward` Δ < 1e-5 vs numpy).
