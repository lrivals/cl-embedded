# S2004 — Unity tests EWC + TinyOL (8 groupes sur `mock_data.h`)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 20 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 4h |
| **Dépendances** | S2001 (ewc_consolidate), S2003 (tinyol_init + poids), S1904 (mock_data.h ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_models.c` |
| **Référence** | `firmware/stm32f4_blink/tests/test_ewc_head.c` (7 tests existants) |

---

## Contexte

`test_ewc_head.c` contient 7 tests couvrant forward + sgd_step.
`test_models.c` est défini en outline (Sprint 19, S1909) mais non implémenté.
Ce fichier doit couvrir les fonctions manquantes : `ewc_consolidate()` et tout le pipeline TinyOL.

---

## Groupes de tests à implémenter

### Groupe 1 — EWC : consolidation

| Test | Assertion |
|------|-----------|
| `test_ewc_consolidate_fisher_update` | `fisher1[0][0] ≈ 0.1 * w²` avec alpha=0.9, Fisher init=0 |
| `test_ewc_consolidate_star_copy` | `star_w1 == w1` pour tous j,i après consolidation |
| `test_ewc_consolidate_fisher_nonneg` | Toutes les Fisher ≥ 0 après consolidation |

### Groupe 2 — EWC : régularisation active

| Test | Assertion |
|------|-----------|
| `test_ewc_penalty_nonzero` | Gradient de pénalité ≠ 0 après consolidation (λ=100) |
| `test_ewc_forgetting_reduced` | acc après tâche 2 > acc sans EWC (λ=0 baseline) sur mock_data task 0 |

### Groupe 3 — TinyOL : init et forward

| Test | Assertion |
|------|-----------|
| `test_tinyol_init_weights_loaded` | `enc->w1[0][0] != 0.0f` après `tinyol_init()` |
| `test_tinyol_forward_shape` | Output shape = 16 floats (embedding) |
| `test_tinyol_forward_delta_vs_ref` | max|C - ref| ≤ 1e-5 vs valeurs de référence dans `mock_data.h` |

---

## Structure `test_models.c`

```c
#include "unity.h"
#include "ewc_head.h"
#include "tinyol.h"
#include "mock_data.h"

static EWCHead  g_ewc;
static TinyOLEncoder g_enc;

void setUp(void)   { ewc_init(&g_ewc); tinyol_init(&g_enc); }
void tearDown(void) {}

/* --- EWC consolidation --- */
void test_ewc_consolidate_fisher_update(void) { ... }
void test_ewc_consolidate_star_copy(void)     { ... }
void test_ewc_consolidate_fisher_nonneg(void) { ... }

/* --- EWC régularisation --- */
void test_ewc_penalty_nonzero(void)    { ... }
void test_ewc_forgetting_reduced(void) { ... }

/* --- TinyOL --- */
void test_tinyol_init_weights_loaded(void)    { ... }
void test_tinyol_forward_shape(void)          { ... }
void test_tinyol_forward_delta_vs_ref(void)   { ... }

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_ewc_consolidate_fisher_update);
    RUN_TEST(test_ewc_consolidate_star_copy);
    RUN_TEST(test_ewc_consolidate_fisher_nonneg);
    RUN_TEST(test_ewc_penalty_nonzero);
    RUN_TEST(test_ewc_forgetting_reduced);
    RUN_TEST(test_tinyol_init_weights_loaded);
    RUN_TEST(test_tinyol_forward_shape);
    RUN_TEST(test_tinyol_forward_delta_vs_ref);
    return UNITY_END();
}
```

---

## Vérification

- [ ] `make -C firmware/stm32f4_blink/ test` : **8/8 PASS** (test_models.c) + tous les tests existants
- [ ] Total Unity suite : ≥ 32 tests PASS (24 existants + 8 nouveaux)
- [ ] CI GitHub Actions : `firmware.yml` green sur les 8 nouveaux tests
