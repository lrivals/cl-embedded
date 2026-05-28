# S1909 — Tests Unity : tous modèles sur mock_data, vérification pas de malloc

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Complété — 28/28 PASS |
| **Durée estimée** | 4h |
| **Dépendances** | S1902 (ewc_consolidate), S1904 (mock_data ✅) |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_models.c` |

---

## Contexte

La vérification sans board est le **critère de succès primaire** du Sprint 19 : `make test` doit passer 100% des tests Unity sur host x86. `test_mahalanobis.c` (16/16 PASS) sert de référence de style. `test_models.c` couvre EWC et TinyOL.

---

## Objectif

Compléter `test_models.c` avec des tests couvrant :
1. EWC forward pass (logits corrects)
2. EWC sgd_step (poids modifiés dans le bon sens)
3. EWC consolidate (Fisher ≥ 0, star_w = w)
4. TinyOL encode/decode (MSE calculé correctement)
5. TinyOL predict (classification correcte sur mock_data)
6. No-malloc (aucun appel à `malloc`/`free` dans le binaire)

---

## Référence — `test_mahalanobis.c` (16/16 PASS)

```c
#include "unity.h"
#include "mahalanobis.h"
#include "mock_data.h"

void setUp(void) {}
void tearDown(void) {}

void test_maha_normal_score_below_threshold(void) {
    MahalanobisDetector det;
    /* init identité */
    for (int k = 0; k < MOCK_N_SAMPLES; k++) {
        float score = maha_score(&det, MOCK_NORMAL_T0[k]);
        TEST_ASSERT_LESS_THAN_FLOAT(MOCK_MAHA_SCORE_NORMAL_T0_MAX, score);
    }
}
```

---

## Tests à implémenter dans `test_models.c`

### Groupe 1 — EWC forward pass

```c
void test_ewc_forward_zero_weights_produces_zero_logits(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    float out[EWC_OUT];
    ewc_forward(&h, MOCK_NORMAL_T0[0], out);
    TEST_ASSERT_FLOAT_WITHIN(MOCK_EWC_LOGIT_TOLERANCE, 0.0f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(MOCK_EWC_LOGIT_TOLERANCE, 0.0f, out[1]);
}

void test_ewc_predict_returns_valid_class(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    int pred = ewc_predict(&h, MOCK_NORMAL_T0[0]);
    TEST_ASSERT(pred == 0 || pred == 1);
}
```

### Groupe 2 — EWC SGD step

```c
void test_ewc_sgd_step_modifies_weights(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.lambda = 0.0f;  /* pas de régularisation pour ce test */
    float w1_before = h.w1[0][0];
    ewc_sgd_step(&h, MOCK_NORMAL_T0[0], 0);
    /* Au moins un poids doit avoir changé */
    int changed = 0;
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            if (h.w1[j][i] != 0.0f) changed = 1;
    TEST_ASSERT_TRUE(changed);
}
```

### Groupe 3 — EWC consolidate

```c
void test_ewc_consolidate_copies_star_weights(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    /* Fixer quelques poids non-nuls */
    h.w1[0][0] = 0.5f;
    h.w3[0][0] = -0.3f;
    ewc_consolidate(&h, 0.9f);
    TEST_ASSERT_EQUAL_FLOAT(h.w1[0][0], h.star_w1[0][0]);
    TEST_ASSERT_EQUAL_FLOAT(h.w3[0][0], h.star_w3[0][0]);
}

void test_ewc_consolidate_fisher_nonnegative(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.w1[0][0] = -0.5f;  /* poids négatif → w² > 0 */
    ewc_consolidate(&h, 0.5f);
    for (int j = 0; j < EWC_H1; j++)
        for (int i = 0; i < EWC_IN; i++)
            TEST_ASSERT_GREATER_OR_EQUAL_FLOAT(0.0f, h.fisher1[j][i]);
}

void test_ewc_consolidate_ema_alpha(void) {
    EWCHead h;
    memset(&h, 0, sizeof(h));
    h.w1[0][0] = 1.0f;           /* w² = 1 */
    h.fisher1[0][0] = 0.0f;      /* Fisher initial = 0 */
    ewc_consolidate(&h, 0.9f);
    /* fisher_new = 0.9 * 0 + 0.1 * 1.0² = 0.1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.1f, h.fisher1[0][0]);
}
```

### Groupe 4 — TinyOL reconstruction error

```c
void test_tinyol_reconstruction_error_zero_weights(void) {
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    memset(&enc, 0, sizeof(enc));
    memset(&dec, 0, sizeof(dec));
    float emb[TINYOL_EMB], recon[TINYOL_OUT];
    tinyol_encode(&enc, MOCK_NORMAL_T0[0], emb);
    tinyol_decode(&dec, emb, recon);
    float err = tinyol_reconstruction_error(MOCK_NORMAL_T0[0], recon, TINYOL_OUT);
    /* recon = 0 (poids nuls), MSE = mean(x²) */
    TEST_ASSERT_LESS_THAN_FLOAT(MOCK_TINYOL_RECON_ERR_ZERO_WEIGHTS + MOCK_TINYOL_RECON_TOLERANCE,
                                 err);
}
```

### Groupe 5 — TinyOL predict

```c
void test_tinyol_predict_anomaly_higher_error(void) {
    TinyOLEncoder enc;
    TinyOLDecoder dec;
    memset(&enc, 0, sizeof(enc));
    memset(&dec, 0, sizeof(dec));
    /* Avec poids nuls : erreur normale < erreur anomalie (normes différentes) */
    float err_normal  = 0.0f, err_anomaly = 0.0f;
    for (int k = 0; k < MOCK_N_SAMPLES; k++) {
        float emb[TINYOL_EMB], recon[TINYOL_OUT];
        tinyol_encode(&enc, MOCK_NORMAL_T0[k],  emb);
        tinyol_decode(&dec, emb, recon);
        err_normal  += tinyol_reconstruction_error(MOCK_NORMAL_T0[k],  recon, TINYOL_OUT);
        tinyol_encode(&enc, MOCK_ANOMALY_T0[k], emb);
        tinyol_decode(&dec, emb, recon);
        err_anomaly += tinyol_reconstruction_error(MOCK_ANOMALY_T0[k], recon, TINYOL_OUT);
    }
    TEST_ASSERT_GREATER_THAN_FLOAT(err_normal, err_anomaly);
}
```

### Groupe 6 — No-malloc

```bash
# Dans le Makefile, ajouter une cible de vérification :
nm firmware/stm32f4_blink/build/test_runner.elf | grep -E " malloc| free" > /dev/null && echo "FAIL: malloc found" || echo "PASS: no malloc"
```

---

## Structure `test_models.c`

```c
#include "unity.h"
#include "ewc_head.h"
#include "tinyol.h"
#include "mock_data.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* [groupes 1–5 ci-dessus] */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_ewc_forward_zero_weights_produces_zero_logits);
    RUN_TEST(test_ewc_predict_returns_valid_class);
    RUN_TEST(test_ewc_sgd_step_modifies_weights);
    RUN_TEST(test_ewc_consolidate_copies_star_weights);
    RUN_TEST(test_ewc_consolidate_fisher_nonnegative);
    RUN_TEST(test_ewc_consolidate_ema_alpha);
    RUN_TEST(test_tinyol_reconstruction_error_zero_weights);
    RUN_TEST(test_tinyol_predict_anomaly_higher_error);
    return UNITY_END();
}
```

---

## Commande `make test`

```bash
make -C firmware/stm32f4_blink/ test
# Doit afficher : X Tests 0 Failures 0 Ignored
```

---

## Vérification

- [ ] `make test` → 100% PASS (0 Failures)
- [ ] `nm build/test_runner.elf | grep malloc` → vide
- [ ] Chaque groupe de tests couvre au moins une propriété mathématique vérifiable
- [ ] Les tests compilent avec `-Wall -Wextra -Wpedantic` sans warning
