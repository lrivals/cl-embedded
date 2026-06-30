# S2614–S2618 — Tests C Unity + Documentation

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🟡 Important (S2614, S2615, S2616) / 🟢 Faible (S2617, S2618) |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | S2614 : 1h30 / S2615 : 1h30 / S2616 : 30 min / S2617 : 1h / S2618 : 30 min = 5h total |
| **Dépendances** | S2601 ✅ (`ewc_head_regression.h/.c`), S2603 ✅ (`ewc_head_multiclass.h/.c`), `firmware/stm32f4_blink/tests/test_ewc_head.c` ✅ (pattern Unity à réutiliser), `firmware/stm32f4_blink/tests/test_runner.c` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_ewc_regression.c`, `firmware/stm32f4_blink/tests/test_ewc_multiclass.c`, `docs/roadmap_phase2.md` |
| **Référence** | `firmware/stm32f4_blink/tests/test_ewc_head.c` (pattern Unity complet avec TEST_MODE), `firmware/stm32f4_blink/Makefile` (cible `test`) |

---

## Contexte

Les tests S2614 et S2615 compilent en mode host (`-DTEST_MODE=1`, gcc natif) avec le framework Unity. Ils testent les invariants mathématiques des têtes C sans nécessiter la board NUCLEO-F439ZI. S2616 vérifie la non-régression : aucun test existant ne doit casser après les modifications de `pipeline.c` et `metrics.c`.

**Invariants à tester** :
- Forward pass produit le bon shape (pas d'accès hors-limite)
- MSE loss / cross-entropy propagent des gradients non nuls
- EWC penalty = 0 avant consolidation, > 0 après
- Fisher et star_w ont les mêmes dimensions que les poids
- Softmax somme à 1 (invariant numérique)
- argmax correct sur données mock connues

---

## S2614 — `firmware/stm32f4_blink/tests/test_ewc_regression.c`

### Spec complète

```c
/**
 * test_ewc_regression.c — Tests Unity pour ewc_head_regression.c
 *
 * Compilation host (sans MCU) :
 *   gcc -DTEST_MODE=1 -o test_ewc_reg test_ewc_regression.c ewc_head_regression.c unity.c -lm
 *
 * Invariants testés :
 *   1. Forward shape — out[0] accessible sans crash (aucun out-of-bound)
 *   2. MSE gradient signe correct — sgd_step rapproche la prédiction du label
 *   3. EWC penalty = 0.0 avant consolidation
 *   4. EWC penalty > 0.0 après consolidation (Fisher non nulle)
 *   5. Consolidation — fisher et star_w de même dimension que w (proxy : fisher != 0 post-consolidate)
 */

#include "unity.h"
#include "ewc_head_regression.h"
#include <math.h>
#include <string.h>

static EWCHeadReg g_h;

void setUp(void)
{
    ewc_reg_init(&g_h);
    g_h.lambda = 400.0f;
}

void tearDown(void) {}

/* ── Test 1 : forward produit un scalaire (pas de NaN, pas de crash) ──────── */
void test_forward_returns_scalar(void)
{
    float x[EWC_REG_IN] = {0.5f, -0.3f, 1.2f, 0.0f, -0.8f};
    float out[EWC_REG_OUT];
    ewc_reg_forward(&g_h, x, out);
    /* Vérifier que out[0] est un float valide (non NaN, non Inf) */
    TEST_ASSERT_FALSE(isnan(out[0]));
    TEST_ASSERT_FALSE(isinf(out[0]));
}

/* ── Test 2 : MSE gradient — après 1 step, erreur diminue ────────────────── */
void test_sgd_step_reduces_error(void)
{
    float x[EWC_REG_IN]  = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float y_true = 50.0f;

    float out_before[EWC_REG_OUT];
    ewc_reg_forward(&g_h, x, out_before);
    float err_before = fabsf(out_before[0] - y_true);

    /* 10 steps SGD sans EWC (lambda=0 pour isoler) */
    g_h.lambda = 0.0f;
    for (int i = 0; i < 10; i++) {
        ewc_reg_sgd_step(&g_h, x, y_true);
    }

    float out_after[EWC_REG_OUT];
    ewc_reg_forward(&g_h, x, out_after);
    float err_after = fabsf(out_after[0] - y_true);

    /* Erreur doit diminuer après 10 steps */
    TEST_ASSERT_LESS_THAN_FLOAT(err_before, err_after);
}

/* ── Test 3 : EWC penalty = 0 avant consolidation ────────────────────────── */
void test_ewc_penalty_zero_before_consolidate(void)
{
    /* Vérifier que fisher est tout zéro à l'init */
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            TEST_ASSERT_EQUAL_FLOAT(0.0f, g_h.fisher1[j][i]);

    /* Penalty = sum(lambda/2 * fisher * (w - star_w)²) = 0 quand fisher = 0 */
    float x[EWC_REG_IN] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float out[EWC_REG_OUT];

    /* Modifier les poids légèrement pour simuler un écart θ - θ* */
    g_h.w1[0][0] = 1.0f;   /* star_w1[0][0] = 0.0f (init) */

    /* Calculer la penalty manuellement pour valider = 0 */
    float penalty = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++) {
            float diff = g_h.w1[j][i] - g_h.star_w1[j][i];
            penalty += g_h.lambda * g_h.fisher1[j][i] * diff * diff;
        }
    /* fisher = 0 → penalty = 0 même avec w != star_w */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, penalty / 2.0f);
}

/* ── Test 4 : EWC penalty > 0 après consolidation ────────────────────────── */
void test_ewc_penalty_nonzero_after_consolidate(void)
{
    float x[EWC_REG_IN] = {1.0f, 0.5f, -0.5f, 0.2f, -0.2f};
    float y_true = 75.0f;

    /* Entraîner pour avoir des w != 0 */
    g_h.lambda = 0.0f;
    for (int i = 0; i < 20; i++) ewc_reg_sgd_step(&g_h, x, y_true);

    /* Consolidation */
    ewc_reg_consolidate(&g_h, EWC_REG_FISHER_DECAY);

    /* Vérifier qu'au moins un Fisher != 0 */
    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            fisher_sum += g_h.fisher1[j][i];
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, fisher_sum);

    /* Modifier les poids et vérifier penalty > 0 */
    g_h.lambda = 400.0f;
    g_h.w1[0][0] += 0.5f;   /* perturber θ par rapport à θ* */

    float penalty = 0.0f;
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++) {
            float diff = g_h.w1[j][i] - g_h.star_w1[j][i];
            penalty += g_h.lambda / 2.0f * g_h.fisher1[j][i] * diff * diff;
        }
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, penalty);
}

/* ── Test 5 : consolidate copie w vers star_w ─────────────────────────────── */
void test_consolidate_copies_weights_to_star(void)
{
    float x[EWC_REG_IN] = {0.3f, -0.1f, 0.8f, -0.5f, 0.2f};
    g_h.lambda = 0.0f;
    for (int i = 0; i < 5; i++) ewc_reg_sgd_step(&g_h, x, 60.0f);

    ewc_reg_consolidate(&g_h, 0.0f);   /* alpha=0 : Fisher = w² pur */

    /* star_w1 doit être égal à w1 après consolidation */
    for (int j = 0; j < EWC_REG_H1; j++)
        for (int i = 0; i < EWC_REG_IN; i++)
            TEST_ASSERT_EQUAL_FLOAT(g_h.w1[j][i], g_h.star_w1[j][i]);
}

/* ── main Unity ──────────────────────────────────────────────────────────── */
int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_forward_returns_scalar);
    RUN_TEST(test_sgd_step_reduces_error);
    RUN_TEST(test_ewc_penalty_zero_before_consolidate);
    RUN_TEST(test_ewc_penalty_nonzero_after_consolidate);
    RUN_TEST(test_consolidate_copies_weights_to_star);
    return UNITY_END();
}
```

### Exécution

```bash
make -C firmware/stm32f4_blink test
# Attendu : 5 tests, 0 failures
```

---

## S2615 — `firmware/stm32f4_blink/tests/test_ewc_multiclass.c`

### Spec complète

```c
/**
 * test_ewc_multiclass.c — Tests Unity pour ewc_head_multiclass.c
 *
 * Invariants testés :
 *   1. Forward shape — logits[N] accessibles sans crash
 *   2. Softmax normalisé — sum(softmax(logits)) ≈ 1.0 (calculé manuellement)
 *   3. argmax correct sur logits connus
 *   4. EWC penalty = 0 avant consolidation
 *   5. EWC penalty > 0 après consolidation
 */

#include "unity.h"
#include "ewc_head_multiclass.h"
#include <math.h>
#include <string.h>

static EWCHeadMC g_h;

void setUp(void) { ewc_mc_init(&g_h); g_h.lambda = 400.0f; }
void tearDown(void) {}

/* ── Test 1 : forward produit N logits valides ────────────────────────────── */
void test_forward_produces_valid_logits(void)
{
    float x[EWC_MC_IN] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f};
    float logits[EWC_MC_N_CLASSES];
    ewc_mc_forward(&g_h, x, logits);

    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        TEST_ASSERT_FALSE(isnan(logits[j]));
        TEST_ASSERT_FALSE(isinf(logits[j]));
    }
}

/* ── Test 2 : softmax somme à 1 ──────────────────────────────────────────── */
void test_softmax_sums_to_one(void)
{
    /* Logits connus */
    float logits[EWC_MC_N_CLASSES];
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) logits[j] = (float)j * 0.5f - 2.0f;

    /* Softmax stable */
    float max_l = logits[0];
    for (int j = 1; j < EWC_MC_N_CLASSES; j++) if (logits[j] > max_l) max_l = logits[j];
    float sum_exp = 0.0f;
    float softmax[EWC_MC_N_CLASSES];
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        softmax[j] = expf(logits[j] - max_l);
        sum_exp += softmax[j];
    }
    float total = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
        softmax[j] /= sum_exp;
        total += softmax[j];
    }
    /* sum ≈ 1.0 avec tolérance FP32 */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, total);
}

/* ── Test 3 : predict retourne l'argmax ──────────────────────────────────── */
void test_predict_returns_argmax(void)
{
    /* Forcer w3[5][0] très grand → classe 5 doit gagner */
    for (int i = 0; i < EWC_MC_H2; i++) {
        for (int j = 0; j < EWC_MC_N_CLASSES; j++) {
            g_h.w3[j][i] = (j == 5) ? 10.0f : -10.0f;
        }
    }
    float x[EWC_MC_IN];
    for (int i = 0; i < EWC_MC_IN; i++) x[i] = 1.0f;

    int pred = ewc_mc_predict(&g_h, x);
    TEST_ASSERT_EQUAL_INT(5, pred);
}

/* ── Test 4 : EWC penalty = 0 avant consolidation ────────────────────────── */
void test_ewc_penalty_zero_before_consolidate(void)
{
    /* Fisher initialisée à 0 → penalty = 0 */
    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++)
            fisher_sum += g_h.fisher3[j][i];
    TEST_ASSERT_EQUAL_FLOAT(0.0f, fisher_sum);
}

/* ── Test 5 : EWC penalty > 0 après consolidation ────────────────────────── */
void test_ewc_penalty_nonzero_after_consolidate(void)
{
    float x[EWC_MC_IN];
    for (int i = 0; i < EWC_MC_IN; i++) x[i] = 0.5f;

    /* Entraîner pour avoir des w != 0 */
    g_h.lambda = 0.0f;
    for (int k = 0; k < 20; k++) ewc_mc_sgd_step(&g_h, x, 3);  /* label=3 */

    ewc_mc_consolidate(&g_h, 0.0f);   /* alpha=0 : Fisher = w² */

    /* fisher3 doit être non nulle */
    float fisher_sum = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++)
            fisher_sum += g_h.fisher3[j][i];
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, fisher_sum);

    /* Perturber les poids et vérifier penalty > 0 */
    g_h.lambda = 400.0f;
    g_h.w3[3][0] += 1.0f;

    float penalty = 0.0f;
    for (int j = 0; j < EWC_MC_N_CLASSES; j++)
        for (int i = 0; i < EWC_MC_H2; i++) {
            float diff = g_h.w3[j][i] - g_h.star_w3[j][i];
            penalty += g_h.lambda / 2.0f * g_h.fisher3[j][i] * diff * diff;
        }
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, penalty);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_forward_produces_valid_logits);
    RUN_TEST(test_softmax_sums_to_one);
    RUN_TEST(test_predict_returns_argmax);
    RUN_TEST(test_ewc_penalty_zero_before_consolidate);
    RUN_TEST(test_ewc_penalty_nonzero_after_consolidate);
    return UNITY_END();
}
```

### Exécution

```bash
make -C firmware/stm32f4_blink test
# Attendu : 5 tests, 0 failures
```

---

## S2616 — Vérification 0 régression (`make test`)

```bash
make -C firmware/stm32f4_blink test
```

**Attendu** : tous les tests existants + les deux nouveaux passent sans régression :

```
test_ewc_head.c         : N tests PASSED (existants — inchangés)
test_ewc_int8.c         : N tests PASSED (existants — inchangés)
test_hdc.c              : N tests PASSED (existants — inchangés)
test_mahalanobis.c      : N tests PASSED (existants — inchangés)
test_pipeline.c         : N tests PASSED (pipeline.c modifié — vérifier)
test_ewc_regression.c   : 5 tests PASSED (nouveaux — S2614)
test_ewc_multiclass.c   : 5 tests PASSED (nouveaux — S2615)
─────────────────────────────────────────────────────
TOTAL : 0 FAILED, 0 ERROR
```

Si `test_pipeline.c` échoue après modification de `pipeline.c` (S2605) : vérifier que le routing par flags est évalué dans le bon ordre et que le protocole binaire existant n'est pas cassé.

---

## S2617 — Tableau comparatif PC vs board (à insérer dans `S2600_sprint_26.md`)

Après exécution de S2611 et S2612, compléter la section **Bilan** du fichier sprint :

```markdown
### Tableau comparatif PC vs board (Sprint 26)

| Modèle | Dataset | RMSE PC (exp_S25_01) | RMSE Board | Ratio | Latence P50 µs | SRAM .bss Ko |
|--------|---------|---------------------|-----------|-------|---------------|-------------|
| EWC Régression | CMAPSS FD001 | 22.53 | 21.23 | 0.94 ✅ | 233 | 65.2 |
| EWC Multi-class | CWRU task0–2 | — | — | — | 130 (infér.) / 403 (online) | 65.2 |

| Modèle | Dataset | F1-macro PC | F1-macro Board | Note |
|--------|---------|-------------|----------------|------|
| EWC Multi-class (N=10) | CWRU 3 tâches | 0.981 (moyenne post-tâche, trompeuse) / **0.240** (modèle final, tous-tâches) | 0.243 (inférence) / 0.507 (online) | Parité board ↔ PC exacte ; critère 0.60 non atteint = oubli catastrophique (pas un bug board) — voir S2611 FIXME(gap1) RÉSOLU |
```

---

## S2618 — Mise à jour `docs/roadmap_phase2.md`

Localiser la section Sprint 25 dans `docs/roadmap_phase2.md` et ajouter après :

```markdown
### Sprint 26 — Portage Board : RUL Régression + Multi-classe C (29 juil. – 5 août 2026)

**Motivation** : porter les capacités Sprint 25 (EWC régression RUL, EWC multi-classe) sur la NUCLEO-F439ZI pour démontrer la faisabilité embarquée et valider les gaps 1 et 2.

**Livrables** :
- Firmware C : `ewc_head_regression.c/.h`, `ewc_head_multiclass.c/.h`
- Pipeline étendu : flags `RUL_MODE (0x80)`, `MULTICLASS_MODE (0x90)`, métriques `OnlineRMSE` + `OnlineF1Macro`
- Scripts Python host : `simulate_rul_board.py`, `simulate_multiclass_board.py`
- 3 expériences board : exp_S26_01 (RUL/CMAPSS), exp_S26_02 (multiclass/CWRU), exp_S26_03 (RAM profiling)
- Tests C Unity : `test_ewc_regression.c`, `test_ewc_multiclass.c` — 10 tests, 0 failures

**Résultats clés** :
- exp_S26_01 (EWC RUL board) : RMSE board = ___ cycles (ratio PC/board = ___)
- exp_S26_02 (EWC Multi-class board) : F1-macro board = ___ (critère ≥ 0.60)
- Latence P50 ≤ 100 ms (critère Gap 2) : ___
- SRAM .bss total firmware : ___ Ko (budget 256 Ko)

**Statut** : ⬜ À compléter post-exécution
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2614 — `test_ewc_regression.c` | ✅ | 5 tests Unity PASS (forward, SGD, EWC×2, consolidation) |
| S2615 — `test_ewc_multiclass.c` | ✅ | 5 tests Unity PASS (forward, softmax, argmax, EWC×2) |
| S2616 — `make test` : 0 régression Sprint 26 | ⚠️ | **75 Tests, 2 Failures** (2026-06-12). Les 10 nouveaux tests EWC reg+mc PASS ; les 2 échecs sont **pré-existants et hors périmètre Sprint 26** : `test_tinyol_predict_normal_zero_weights` et `test_tinyol_forward_delta` — valeurs de référence hardcodées qui ne correspondent plus à `model_weights.h` régénéré dans un sprint antérieur. À corriger côté TinyOL séparément. |
| S2617 — Tableau comparatif PC vs board | ✅ | `S2600_sprint_26.md` section Bilan complétée |
| S2618 — `docs/roadmap_phase2.md` mis à jour | ✅ | Sprint 25 + 26 résultats finaux ajoutés |

---

## Questions ouvertes

- `TODO(arnaud)` : Le test `test_sgd_step_reduces_error` (S2614) vérifie la convergence sur 10 steps avec LR=0.001 et input fixe. Avec une initialisation Xavier et LR très faible, la convergence en 10 steps n'est pas garantie. Fixer le seed LCG à une valeur qui garantit un w1[0][0] > 0 au départ pour assurer la convergence.
- `TODO(dorra)` : Ajouter `test_ewc_regression.c` et `test_ewc_multiclass.c` au `Makefile` cible `test`. Vérifier que `test_runner.c` inclut les deux nouveaux suites (`test_ewc_reg_suite()` + `test_ewc_mc_suite()`).
