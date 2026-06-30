# S2301–S2304 — HDC C : implémentation complète + intégration pipeline

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Terminé — 2026-06-02 |
| **Durée estimée** | 4h + 1h + 2h + 1h = 8h |
| **Dépendances** | Sprint 20 ✅ — `hdc.c` skeleton (encode + predict + update basique), `hdc.h` existant |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/hdc.c`, `firmware/stm32f4_blink/inc/hdc.h`, `firmware/stm32f4_blink/tests/test_hdc.c`, `firmware/stm32f4_blink/src/pipeline.c` |
| **Référence** | `Benatti2019HDC`, `firmware/stm32f4_blink/src/ewc_head.c` (pattern branche pipeline) |

---

## Contexte

`hdc.c` (skeleton Sprint 20, S2008) implémente les trois fonctions de base : `hdc_encode()` (projection aléatoire → binarisation ±1), `hdc_predict()` (argmax dot-product sur la mémoire associative), et `hdc_update()` (accumulation brute dans `am[label]`).

Il manque deux fonctions critiques pour un apprentissage incrémental robuste :

1. **`hdc_binarize()`** : après accumulation, les prototypes `am[c]` deviennent des vecteurs réels à grande norme. Il faut périodiquement les re-binariser en ±1 pour maintenir la propriété HDC (sinon la similarité cosinus dégénère).
2. **`hdc_retrain()`** : pour corriger des erreurs de classification, HDC incrémental peut relire un buffer borné d'échantillons récents et re-accumuler depuis zéro — sans oubli catastrophique par construction.

De plus, `pipeline.c` gère EWC et Mahalanobis mais pas HDC : il faut ajouter une branche `PROTO_FLAG_HDC_MODE`.

---

## S2301 — `firmware/stm32f4_blink/src/hdc.c` : binarize + retrain

### Fonction `hdc_binarize`

```c
/**
 * hdc_binarize — Re-binarise les prototypes AM en ±1 après accumulation.
 *
 * Après N mises à jour, am[c][i] ∈ ℝ (somme de ±1).
 * Cette fonction repasse chaque am[c][i] à +1.0f si ≥ 0, -1.0f sinon.
 * À appeler lors de la consolidation de tâche (PROTO_FLAG_CONSOLIDATE).
 *
 * Complexité : O(HDC_N_CLASSES × HDC_DIM)
 * MEM stack : 0 B supplémentaire (in-place)
 */
void hdc_binarize(HDCClassifier *h)
{
    for (int c = 0; c < HDC_N_CLASSES; c++) {
        for (int i = 0; i < HDC_DIM; i++) {
            h->am[c][i] = (h->am[c][i] >= 0.0f) ? 1.0f : -1.0f;
        }
    }
}
```

### Fonction `hdc_retrain`

```c
/**
 * hdc_retrain — Réentraîne l'AM depuis le buffer interne borné.
 *
 * Remet à zéro l'AM, puis ré-accumule tous les échantillons du buffer
 * (dans l'ordre d'insertion, FIFO circulaire) et binarise.
 * Appeler après une séquence de mises à jour pour consolider la mémoire.
 *
 * MEM stack : float hv[HDC_DIM] = HDC_DIM * 4 B = 4 Ko @ FP32
 *   → Ne pas appeler depuis une ISR.
 */
void hdc_retrain(HDCClassifier *h)
{
    /* Remettre AM à zéro */
    for (int c = 0; c < HDC_N_CLASSES; c++) {
        for (int i = 0; i < HDC_DIM; i++) {
            h->am[c][i] = 0.0f;
        }
    }

    /* Ré-accumulation depuis le buffer circulaire */
    float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack) */
    int count = (h->buf_count < HDC_RETRAIN_BUF) ? h->buf_count : HDC_RETRAIN_BUF;
    for (int k = 0; k < count; k++) {
        int idx = (h->buf_head + k) % HDC_RETRAIN_BUF;
        /* Dequantize uint8 → float [-1, 1] : x = (raw / 127.5f) - 1.0f */
        float x[HDC_N_FEATURES];
        for (int j = 0; j < HDC_N_FEATURES; j++) {
            x[j] = ((float)h->buf_x[idx][j] / 127.5f) - 1.0f;
        }
        hdc_encode(h, x, hv);
        hdc_update(h, hv, (int)h->buf_y[idx]);
    }
    hdc_binarize(h);
}
```

### Mise à jour de `hdc_update` pour alimenter le buffer

Modifier `hdc_update` pour stocker les échantillons dans le buffer circulaire :

```c
void hdc_update(HDCClassifier *h, const float *hv, int label)
{
    /* Accumulation dans AM */
    for (int i = 0; i < HDC_DIM; i++) {
        h->am[label][i] += hv[i];
    }
    h->n_trained++;
    /* Le buffer est alimenté par hdc_update_with_sample (voir pipeline.c) */
}

/**
 * hdc_update_with_sample — Update AM + stocke l'échantillon brut dans le buffer.
 *
 * x     : features originales (avant encodage)
 * hv    : hypervecteur encodé depuis x
 * label : classe (0 ou 1)
 *
 * Stockage en uint8 pour économiser la RAM :
 *   raw = (uint8_t)((x[j] + 1.0f) * 127.5f)  — plage [-1,1] → [0,255]
 * MEM buf_x : HDC_RETRAIN_BUF * HDC_N_FEATURES * 1 B = 50 * 5 = 250 B
 * MEM buf_y : HDC_RETRAIN_BUF * 1 B = 50 B
 */
void hdc_update_with_sample(HDCClassifier *h, const float *x,
                             const float *hv, int label)
{
    hdc_update(h, hv, label);

    /* Stocker dans le buffer circulaire */
    int slot = h->buf_head % HDC_RETRAIN_BUF;
    for (int j = 0; j < HDC_N_FEATURES; j++) {
        float clamped = x[j] < -1.0f ? -1.0f : (x[j] > 1.0f ? 1.0f : x[j]);
        h->buf_x[slot][j] = (uint8_t)((clamped + 1.0f) * 127.5f);
    }
    h->buf_y[slot] = (uint8_t)label;
    h->buf_head = (h->buf_head + 1) % HDC_RETRAIN_BUF;
    if (h->buf_count < HDC_RETRAIN_BUF) h->buf_count++;
}
```

---

## S2302 — `firmware/stm32f4_blink/inc/hdc.h` : API complète

Ajouter dans la struct `HDCClassifier` et dans l'API publique :

```c
/* HDC — Hyperdimensional Computing (implémentation complète Sprint 23)
 * Référence : Benatti2019HDC
 * Depuis configs/board_hdc.yaml */

#ifndef HDC_H
#define HDC_H

#include <stdint.h>

#define HDC_DIM           1000  /* Dimension des hypervecteurs */
#define HDC_N_FEATURES       5  /* Features d'entrée (top-5 sélectionnées) */
#define HDC_N_CLASSES        2  /* faulty / normal */
#define HDC_RETRAIN_BUF     50  /* Taille buffer retrain (FIFO circulaire) */
                                /* MEM buf total : 50*5 + 50 = 300 B */

/* MEM total HDCClassifier :
 *   am       : 2*1000*4    = 8 000 B (SRAM .bss)
 *   proj     : 1000*5*4    = 20 000 B (SRAM — TODO(dorra): Flash const ?)
 *   buf_x    : 50*5*1      =    250 B
 *   buf_y    : 50*1        =     50 B
 *   scalars  : n_trained + buf_head + buf_count = 12 B
 *   TOTAL    : ~28 312 B ≈ 27.7 Ko @ FP32 (dans budget 64 Ko board) */
typedef struct {
    float   am[HDC_N_CLASSES][HDC_DIM];       /* Mémoire associative */
    float   proj[HDC_DIM][HDC_N_FEATURES];    /* Projection aléatoire (fixée à l'init) */
    uint8_t buf_x[HDC_RETRAIN_BUF][HDC_N_FEATURES]; /* Buffer retrain — quantifié uint8 */
    uint8_t buf_y[HDC_RETRAIN_BUF];           /* Labels buffer */
    int     n_trained;
    int     buf_head;   /* Prochain slot d'écriture (FIFO circulaire) */
    int     buf_count;  /* Nombre d'échantillons dans le buffer (≤ HDC_RETRAIN_BUF) */
} HDCClassifier;

void hdc_init                (HDCClassifier *h);
void hdc_encode              (const HDCClassifier *h, const float *x, float *hv_out);
int  hdc_predict             (const HDCClassifier *h, const float *hv);
void hdc_update              (HDCClassifier *h, const float *hv, int label);
void hdc_update_with_sample  (HDCClassifier *h, const float *x,
                               const float *hv, int label);
void hdc_binarize            (HDCClassifier *h);
void hdc_retrain             (HDCClassifier *h);

#endif /* HDC_H */
```

---

## S2303 — `firmware/stm32f4_blink/tests/test_hdc.c` : ≥ 10 tests

Étendre le fichier existant (2 tests) avec 8 tests supplémentaires :

```c
/* ── Tests existants à conserver ──────────────────────────────────────────── */
// test_hdc_encode_norm        (déjà présent)
// test_hdc_predict_label      (déjà présent)

/* ── Nouveaux tests ────────────────────────────────────────────────────────── */

void test_hdc_update_accumulates(void)
{
    /* hdc_update ajoute hv à am[label].
     * Après 3 updates de {+1,...} sur classe 0 : am[0][i] == 3.0 pour tout i. */
    HDCClassifier h = make_identity_proj();
    float x[HDC_N_FEATURES] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);
    hdc_update(&h, hv, 0);
    hdc_update(&h, hv, 0);
    hdc_update(&h, hv, 0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 3.0f, h.am[0][0]);
}

void test_hdc_binarize_preserves_sign(void)
{
    /* Après accumulation de vecteurs +1, am[0][i] > 0 → binarize → am[0][i] == +1.
     * Après update de vecteurs -1, am[1][i] < 0 → binarize → am[1][i] == -1. */
    HDCClassifier h = make_identity_proj();
    float xp[HDC_N_FEATURES] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float xn[HDC_N_FEATURES] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hvp[HDC_DIM], hvn[HDC_DIM];
    hdc_encode(&h, xp, hvp);
    hdc_encode(&h, xn, hvn);
    hdc_update(&h, hvp, 0);
    hdc_update(&h, hvn, 1);
    hdc_binarize(&h);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 1.0f,  h.am[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(TOL, -1.0f, h.am[1][0]);
}

void test_hdc_binarize_norm_is_hdcdim(void)
{
    /* Après binarize, chaque prototype am[c] est un vecteur ±1 :
     * sum(am[c][i]²) == HDC_DIM exactement. */
    HDCClassifier h = make_identity_proj();
    /* Remplir am avec des valeurs quelconques non nulles */
    for (int i = 0; i < HDC_DIM; i++) {
        h.am[0][i] = (float)(i % 5) - 2.0f;
        h.am[1][i] = (float)(i % 3) - 1.5f;
    }
    hdc_binarize(&h);
    float norm0 = 0.0f, norm1 = 0.0f;
    for (int i = 0; i < HDC_DIM; i++) {
        norm0 += h.am[0][i] * h.am[0][i];
        norm1 += h.am[1][i] * h.am[1][i];
    }
    TEST_ASSERT_FLOAT_WITHIN(TOL, (float)HDC_DIM, norm0);
    TEST_ASSERT_FLOAT_WITHIN(TOL, (float)HDC_DIM, norm1);
}

void test_hdc_n_trained_increments(void)
{
    HDCClassifier h;
    hdc_init(&h);
    TEST_ASSERT_EQUAL_INT(0, h.n_trained);
    float hv[HDC_DIM];
    memset(hv, 0, sizeof(hv));
    hdc_update(&h, hv, 0);
    TEST_ASSERT_EQUAL_INT(1, h.n_trained);
    hdc_update(&h, hv, 1);
    TEST_ASSERT_EQUAL_INT(2, h.n_trained);
}

void test_hdc_update_with_sample_fills_buffer(void)
{
    HDCClassifier h = make_identity_proj();
    hdc_init(&h);
    /* Copier la proj du make_identity_proj */
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x[HDC_N_FEATURES]  = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);
    hdc_update_with_sample(&h, x, hv, 0);

    TEST_ASSERT_EQUAL_INT(1, h.buf_count);
    TEST_ASSERT_EQUAL_INT(1, h.buf_head % HDC_RETRAIN_BUF);
}

void test_hdc_buf_wraps_at_retrain_buf(void)
{
    /* Après HDC_RETRAIN_BUF+1 insertions, buf_count == HDC_RETRAIN_BUF (pas plus). */
    HDCClassifier h;
    hdc_init(&h);
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x[HDC_N_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv[HDC_DIM];
    hdc_encode(&h, x, hv);

    for (int k = 0; k < HDC_RETRAIN_BUF + 5; k++) {
        hdc_update_with_sample(&h, x, hv, k % 2);
    }
    TEST_ASSERT_EQUAL_INT(HDC_RETRAIN_BUF, h.buf_count);
}

void test_hdc_retrain_predicts_correct_after_reset(void)
{
    /* Scénario : entraîner classe 0 et classe 1, retrain depuis buffer,
     * vérifier que les prédictions sont correctes. */
    HDCClassifier h;
    hdc_init(&h);
    for (int i = 0; i < HDC_DIM; i++) h.proj[i][0] = 1.0f;

    float x0[HDC_N_FEATURES] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float x1[HDC_N_FEATURES] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float hv0[HDC_DIM], hv1[HDC_DIM];
    hdc_encode(&h, x0, hv0);
    hdc_encode(&h, x1, hv1);

    for (int k = 0; k < 5; k++) {
        hdc_update_with_sample(&h, x0, hv0, 0);
        hdc_update_with_sample(&h, x1, hv1, 1);
    }

    /* Corrompre l'AM */
    memset(h.am, 0, sizeof(h.am));

    /* Retrain depuis le buffer doit restaurer les prédictions correctes */
    hdc_retrain(&h);
    TEST_ASSERT_EQUAL_INT(0, hdc_predict(&h, hv0));
    TEST_ASSERT_EQUAL_INT(1, hdc_predict(&h, hv1));
}

void test_hdc_init_zeros_all_fields(void)
{
    HDCClassifier h;
    /* Initialiser avec des déchets */
    memset(&h, 0xAB, sizeof(h));
    hdc_init(&h);
    TEST_ASSERT_EQUAL_INT(0, h.n_trained);
    TEST_ASSERT_EQUAL_INT(0, h.buf_head);
    TEST_ASSERT_EQUAL_INT(0, h.buf_count);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.am[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(TOL, 0.0f, h.am[1][0]);
}
```

### Build et exécution x86

```bash
gcc -O0 -g \
    -I firmware/stm32f4_blink/inc \
    -I firmware/stm32f4_blink/tests/unity/src \
    firmware/stm32f4_blink/tests/unity/src/unity.c \
    firmware/stm32f4_blink/tests/test_hdc.c \
    firmware/stm32f4_blink/src/hdc.c \
    -lm \
    -o /tmp/test_hdc_runner

/tmp/test_hdc_runner   # attendu : 10/10 PASS
```

---

## S2304 — `firmware/stm32f4_blink/src/pipeline.c` : case `MODEL_HDC`

### Ajouts dans l'en-tête du fichier

```c
/* Dans les includes (après ewc_head.h) */
#include "hdc.h"

/* Dans les globals statiques (après g_ewc_head) */
/* MEM: HDCClassifier ~27.7 Ko @ FP32 en .bss
 * ATTENTION : avec g_ewc_head (~9.5 Ko) + g_tinyol (~5.7 Ko) + g_detector (~128 B)
 * + g_hdc (~27.7 Ko) = ~43 Ko → dans le budget 64 Ko. */
HDCClassifier g_hdc;
```

### Ajout dans `pipeline_init()`

```c
/* Après ewc_init / maha_init */
hdc_init(&g_hdc);
/* proj initialisée depuis model_weights.h ou générée avec LCG seed fixe :
 * TODO(dorra): mettre les poids de projection en Flash (RODATA) pour économiser 20 Ko SRAM */
```

### Nouveau flag PROTO dans `pipeline.h`

```c
#define PROTO_FLAG_HDC_MODE     (1U << 2)   /* bit 2 — utiliser HDCClassifier */
/* Bits existants : PROTO_FLAG_EWC_MODE (1<<0), PROTO_FLAG_UPDATE (1<<1),
 *                  PROTO_FLAG_CONSOLIDATE (1<<3), PROTO_FLAG_RESET (1<<4) */
```

### Branche HDC dans `pipeline_run()`

```c
} else if (g_recv_flags & PROTO_FLAG_HDC_MODE) {
    /* ── Chemin HDC : encode → predict → update si UPDATE → binarize si CONSOLIDATE ── */
    float hv[HDC_DIM];   /* MEM: 4 Ko @ FP32 (stack — vérifier suffisance pile MCU) */
    hdc_encode(&g_hdc, raw, hv);
    pred = hdc_predict(&g_hdc, hv);

    /* Confiance : dot(am[pred], hv) normalisé par HDC_DIM → [0, 1] */
    float score = 0.0f;
    for (int i = 0; i < HDC_DIM; i++) score += g_hdc.am[pred][i] * hv[i];
    confidence = (score / (float)HDC_DIM + 1.0f) / 2.0f;

    if (g_recv_flags & PROTO_FLAG_UPDATE) {
        hdc_update_with_sample(&g_hdc, raw, hv, (int)g_recv_label);
    }
    if (g_recv_flags & PROTO_FLAG_CONSOLIDATE) {
        hdc_binarize(&g_hdc);
        g_current_task_id = g_recv_task_id;
    }
    auroc_update(&g_auroc, confidence, (int)g_recv_label);
```

> **Attention pile MCU** : `float hv[HDC_DIM]` = 4 Ko sur la pile. La pile Cortex-M4 par défaut est 8 Ko dans les projets CubeMX. Vérifier le fichier de linker (`STM32F439ZITx_FLASH.ld`) que `_Min_Stack_Size = 0x400` n'a pas été laissé à 1 Ko.

---

## Vérification end-to-end

```bash
# 1. Compilation ARM (sans board)
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 \
    -I firmware/stm32f4_blink/inc -c \
    firmware/stm32f4_blink/src/hdc.c -o /tmp/hdc.o
echo "HDC ARM: $?"

# 2. Tests Unity x86
gcc -O0 -I firmware/stm32f4_blink/inc \
    -I firmware/stm32f4_blink/tests/unity/src \
    firmware/stm32f4_blink/tests/unity/src/unity.c \
    firmware/stm32f4_blink/tests/test_hdc.c \
    firmware/stm32f4_blink/src/hdc.c \
    -lm -o /tmp/test_hdc && /tmp/test_hdc
# Attendu : 10/10 PASS

# 3. Compilation pipeline avec HDC
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 \
    -I firmware/stm32f4_blink/inc -c \
    firmware/stm32f4_blink/src/pipeline.c -o /tmp/pipeline.o
echo "Pipeline ARM: $?"
```

---

## Questions ouvertes

- `TODO(dorra)` : La matrice `proj[1000][5]` = 20 Ko est actuellement en SRAM `.bss`. Peut-elle être mise en Flash (`const` RODATA) si fixée au compile-time ? Cela économiserait 20 Ko SRAM pour les autres modèles.
- `TODO(dorra)` : La pile MCU supporte-t-elle `float hv[HDC_DIM]` = 4 Ko dans `pipeline_run()` ? Vérifier `_Min_Stack_Size` dans le linker script.
- `TODO(arnaud)` : `HDC_DIM = 1000` est-il suffisant pour Paderborn (features FFT) ? Benatti2019HDC utilise D = 4096 pour des signaux audio — valider que 1000 suffit pour 5 features tabulaires.
- `FIXME(gap2)` : Annoter `hv[HDC_DIM]` sur la pile avec son empreinte 4 Ko dans le commentaire `pipeline_run()`.

---

## Résultats d'implémentation (2026-06-02)

### Ce qui a été livré

| Sous-tâche | Statut | Notes |
| ---------- | :----: | ----- |
| S2301 — `hdc.c` : `hdc_binarize` + `hdc_retrain` + `hdc_update_with_sample` | ✅ | Bug spec corrigé : `hdc_retrain` lit depuis l'entrée la plus ancienne (`start = (buf_head + BUF - count) % BUF`) |
| S2302 — `hdc.h` : struct étendue + `HDC_RETRAIN_BUF` + 3 nouvelles déclarations | ✅ | Struct ~27.7 Ko @ FP32 ; total pipeline ~43 Ko dans budget 64 Ko |
| S2303 — `test_hdc.c` : 8 nouveaux tests + `test_runner.c` mis à jour | ✅ | **57/57 PASS** (zéro régression) |
| S2304 — `pipeline.c/h` : `PROTO_FLAG_HDC_MODE 0x20U` + global `g_hdc` + branche HDC | ✅ | Flag 0x20 (bit 5) sans conflit avec flags existants |

### Résultats tests Unity x86

```text
57 Tests 0 Failures 0 Ignored — OK
```

Tests HDC (10/10) :

- `test_hdc_encode_norm` ✅
- `test_hdc_predict_label` ✅
- `test_hdc_update_accumulates` ✅
- `test_hdc_binarize_preserves_sign` ✅
- `test_hdc_binarize_norm_is_hdcdim` ✅
- `test_hdc_n_trained_increments` ✅
- `test_hdc_update_with_sample_fills_buffer` ✅
- `test_hdc_buf_wraps_at_retrain_buf` ✅
- `test_hdc_retrain_predicts_correct_after_reset` ✅
- `test_hdc_init_zeros_all_fields` ✅

### Compilation ARM vérifiée

```bash
# hdc.c → Cortex-M4 FPv4 FP32 : OK
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Iinc -c src/hdc.c
# pipeline.c → Cortex-M4 FPv4 FP32 : OK
arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -Iinc -DTEST_MODE=1 -DSTM32F439xx -c src/pipeline.c
```

### Correction apportée vs spec

Le spec original de `hdc_retrain` lisait le buffer depuis `buf_head + k`, ce qui est incorrect lorsque le buffer n'est pas plein (les indices lus pointent vers des slots vides). La correction utilise `start = (buf_head + HDC_RETRAIN_BUF - count) % HDC_RETRAIN_BUF` pour toujours commencer depuis l'entrée la plus ancienne, que le buffer soit plein ou non.
