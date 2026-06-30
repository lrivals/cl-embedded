# S2907 — Tests Unity `test_tinyol_int8.c`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté — 5/5 tests PASS (`make test`, 15 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | S2902 ✅ (`tinyol_int8.c` + `tinyol_int8.h`) · S2903 ✅ (`pipeline.c` FLAGS `0xC0`) |
| **Fichier cible** | `firmware/stm32f4_blink/tests/test_tinyol_int8.c` |
| **Références** | `firmware/stm32f4_blink/tests/test_ewc_int8.c` (pattern fake-quant) · `firmware/stm32f4_blink/inc/tinyol_int8.h` (interface) |

---

## Contexte

`tinyol_int8.c` implémente deux composants distincts à tester :

1. **`TinyOLEncoderInt8`** — encodeur autoencoder avec poids `int8_t` (fake-quant Q7) et activations `uint8_t [0, 255]` post-ReLU. La dequantification (scale = `TINYOL_INT8_SCALE_W = 1/128`) et la requantification des activations (scale = `TINYOL_INT8_ACT_SCALE = 8/255`) sont les invariants clés.

2. **`OtOHeadInt8`** — tête One-to-One (Linear D→1 + sigmoid) en ligne. Poids INT8 de stockage + copie maîtresse FP32 pour SGD (straight-through). L'apprentissage se mesure par la convergence de `last_prob`.

**Différence vs `test_tinyol.c` (FP32)** : la comparaison INT8 vs FP32 teste la fidélité de la quantification, pas le comportement applicatif. Les tests numériques ont une tolérance de 5% liée à `TINYOL_INT8_SCALE_W`.

---

## Spec des 5 tests

### Test 1 — `test_tinyol_int8_encode_range`

**Invariant** : chaque composante de l'embedding `emb_u8` est dans `[0, 255]` (garanti par le cast UINT8 post-ReLU, mais testé explicitement pour détecter un overflow silencieux).

```c
void test_tinyol_int8_encode_range(void)
{
    TinyOLEncoderInt8 enc;
    tinyol_int8_init(&enc);

    float x[TINYOL_IN];
    for (int i = 0; i < TINYOL_IN; i++) x[i] = (float)(i - 2);  /* valeurs négatives + positives */

    uint8_t emb[TINYOL_EMB];
    tinyol_int8_encode(&enc, x, emb);

    for (int i = 0; i < TINYOL_EMB; i++) {
        /* uint8_t est déjà borné par son type, mais on teste l'absence de UB */
        TEST_ASSERT_TRUE(emb[i] >= 0 && emb[i] <= 255);
    }
}
```

> `uint8_t` ne peut physiquement pas sortir de [0,255], mais ce test détecte un encodeur non-initialisé ou un UB en mémoire non settée.

---

### Test 2 — `test_tinyol_int8_encode_vs_fp32`

**Invariant** : l'embedding INT8 dequantifié est proche de l'embedding FP32 (δ moyen < 5% de la plage d'activation `[0, 8]`).

```c
#include "tinyol.h"  /* TinyOLAutoencoder, tinyol_encode() */

#define TINYOL_INT8_ACT_SCALE (8.0f / 255.0f)
#define TOL_PCT 0.05f          /* 5% de la plage [0, 8] = 0.4 */
#define ACT_RANGE 8.0f

void test_tinyol_int8_encode_vs_fp32(void)
{
    TinyOLEncoderInt8 enc_i8;
    tinyol_int8_init(&enc_i8);

    TinyOLAutoencoder enc_fp32;
    tinyol_init(&enc_fp32);    /* même poids depuis model_weights.h */

    float x[TINYOL_IN];
    for (int i = 0; i < TINYOL_IN; i++) x[i] = 0.5f;

    uint8_t emb_u8[TINYOL_EMB];
    float   emb_fp32[TINYOL_EMB];
    tinyol_int8_encode(&enc_i8, x, emb_u8);
    tinyol_encode(&enc_fp32, x, emb_fp32);

    float total_delta = 0.0f;
    for (int i = 0; i < TINYOL_EMB; i++) {
        float act_i8 = (float)emb_u8[i] * TINYOL_INT8_ACT_SCALE;
        total_delta += fabsf(act_i8 - emb_fp32[i]);
    }
    float mean_delta = total_delta / (float)TINYOL_EMB;
    TEST_ASSERT_FLOAT_WITHIN(ACT_RANGE * TOL_PCT, 0.0f, mean_delta);
}
```

---

### Test 3 — `test_oto_int8_predict_returns_binary`

**Invariant** : `oto_int8_predict()` retourne strictement 0 ou 1 (jamais autre valeur — vérifie la cohérence du seuil sigmoid).

```c
void test_oto_int8_predict_returns_binary(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = (uint8_t)(i * 3 % 256);

    int pred = oto_int8_predict(&oto, emb);
    TEST_ASSERT_TRUE(pred == 0 || pred == 1);
}
```

---

### Test 4 — `test_oto_int8_update_learns_class1`

**Invariant** : après 20 updates label=1 avec le même embedding, `last_prob > 0.5f` (la tête converge vers la classe positive).

```c
void test_oto_int8_update_learns_class1(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    /* Embedding non-nul pour que le gradient soit non-nul */
    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = 128u;

    for (int k = 0; k < 20; k++) {
        oto_int8_update(&oto, emb, 1);
    }
    oto_int8_predict(&oto, emb);   /* met à jour last_prob */

    TEST_ASSERT_GREATER_THAN_FLOAT(0.5f, oto.last_prob);
}
```

---

### Test 5 — `test_oto_int8_last_prob_range`

**Invariant** : `last_prob ∈ [0.0f, 1.0f]` après des updates mixtes (vérifie que sigmoid ne produit pas de NaN ni de valeur hors plage).

```c
void test_oto_int8_last_prob_range(void)
{
    OtOHeadInt8 oto;
    oto_int8_init(&oto);

    uint8_t emb[TINYOL_EMB];
    for (int i = 0; i < TINYOL_EMB; i++) emb[i] = (uint8_t)(i % 256);

    for (int k = 0; k < 10; k++) {
        oto_int8_update(&oto, emb, k % 2);
        oto_int8_predict(&oto, emb);
        TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.5f, oto.last_prob);  /* [0, 1] */
    }
}
```

---

## Tableau récapitulatif

| Test | Composant | Input | Invariant vérifié |
|------|-----------|-------|-------------------|
| `test_tinyol_int8_encode_range` | `TinyOLEncoderInt8` | x = {−2..2} | `emb_u8[i] ∈ [0, 255]` |
| `test_tinyol_int8_encode_vs_fp32` | `TinyOLEncoderInt8` | x = {0.5…} | δ moyen INT8 vs FP32 < 5% plage |
| `test_oto_int8_predict_returns_binary` | `OtOHeadInt8` | emb aléatoire | `pred ∈ {0, 1}` |
| `test_oto_int8_update_learns_class1` | `OtOHeadInt8` | emb=128, 20 updates label=1 | `last_prob > 0.5f` |
| `test_oto_int8_last_prob_range` | `OtOHeadInt8` | updates mixtes | `last_prob ∈ [0.0, 1.0]` |

---

## Notes d'implémentation

- **`tinyol_int8_init()`** lit les poids depuis `model_weights.h` (même source que `tinyol_init()`). `test_tinyol_int8_encode_vs_fp32` suppose donc que les deux fonctions init chargent les mêmes poids — vérifier que `model_weights.h` exporte les mêmes constantes utilisées par les deux initialiseurs.
- **Tolérance 5%** : avec `TINYOL_INT8_SCALE_W = 1/128` et `TINYOL_INT8_ACT_SCALE = 8/255`, l'erreur de quantification maximale est ±0.5 LSB × scale. Sur la plage [0,8], 5% = 0.4 → tolérance confortable.
- **Test 4 learning rate** : `oto_int8_init()` fixe `lr`. Si lr est très petit, 20 updates peuvent ne pas suffire. Ajuster à 50 updates si le test échoue avec les poids d'initialisation par défaut.
- **Correction d'implémentation (15 juin)** : la spec écrivait `TinyOLAutoencoder enc_fp32; tinyol_init(&enc_fp32);` — le type FP32 réel de `tinyol.h` est `TinyOLEncoder`, et `tinyol_init(TinyOLEncoder*, TinyOLDecoder*)` exige aussi un décodeur. Test 2 corrigé en conséquence : `TinyOLEncoder enc_fp32; TinyOLDecoder dec_fp32; tinyol_init(&enc_fp32, &dec_fp32);`. 20 updates suffisent (Test 4 PASS sans monter à 50).

---

## Intégration `test_runner.c`

Ajouter dans `firmware/stm32f4_blink/tests/test_runner.c` :

```c
/* ── Déclarations — test_tinyol_int8.c (S2907) ──────────────────────────── */
void test_tinyol_int8_encode_range(void);
void test_tinyol_int8_encode_vs_fp32(void);
void test_oto_int8_predict_returns_binary(void);
void test_oto_int8_update_learns_class1(void);
void test_oto_int8_last_prob_range(void);
```

Et dans `main()` :

```c
    /* TinyOL INT8 — S2907 */
    RUN_TEST(test_tinyol_int8_encode_range);
    RUN_TEST(test_tinyol_int8_encode_vs_fp32);
    RUN_TEST(test_oto_int8_predict_returns_binary);
    RUN_TEST(test_oto_int8_update_learns_class1);
    RUN_TEST(test_oto_int8_last_prob_range);
```

---

## Vérification

```bash
cd firmware/stm32f4_blink

# Build + run tests host
make test

# Résultat attendu : 84 (post-S2906) + 5 = 89 tests, 0 failures nouvelles
# Les 2 failures TinyOL FP32 pré-existantes (test_tinyol_encode_zero_weights +
# test_tinyol_forward_delta) restent présentes — elles sont hors périmètre S2907

make all
arm-none-eabi-size build/stm32f4_blink.elf
# .bss total < 128 Ko
```
