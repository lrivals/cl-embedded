# S2906 — Tests Unity `test_hdc_int8.c`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Implémenté — 5/5 tests PASS (`make test`, 15 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | S2901 ✅ (`hdc_int8.c` + `hdc_int8.h`) |
| **Fichier cible** | `firmware/stm32f4_blink/tests/test_hdc_int8.c` |
| **Références** | `firmware/stm32f4_blink/tests/test_hdc.c` (modèle de style) · `firmware/stm32f4_blink/inc/hdc_int8.h` (interface) |

---

## Contexte

`hdc_int8.c` implémente HDC avec stockage INT8 (base vectors `int8_t ±1`) et AM INT16 (accumulation saturée). Les tests reprennent le pattern de `test_hdc.c` (FP32) mais adaptés aux invariants entiers :

- `hdc_int8_encode()` produit des hypervecteurs `int8_t` dont chaque composante est **strictement ±1** (projection LCG binarisée, pas de valeur intermédiaire)
- `hdc_int8_update()` accumule en `int16_t` avec saturation ±32 767
- `hdc_int8_predict()` utilise un accumulateur `int32_t` pour le produit scalaire — pas de débordement même avec D=2048

**Différence vs `test_hdc.c`** : pas de `make_identity_proj()` puisque les base vectors sont initialisés de façon déterministe par LCG dans `hdc_int8_init()`. Les tests utilisent `hdc_int8_init()` directement et testent les invariants arithmétiques plutôt que les valeurs exactes.

---

## Spec des 5 tests

### Test 1 — `test_hdc_int8_init_zeros_am`

**Invariant** : après `hdc_int8_init()`, toute la mémoire associative est à zéro.

```c
void test_hdc_int8_init_zeros_am(void)
{
    HDCInt8 h;
    memset(&h, 0xAB, sizeof(h));   /* pollution initiale */
    hdc_int8_init(&h);

    for (int c = 0; c < HDC_I_C; c++)
        for (int i = 0; i < HDC_I_D; i++)
            TEST_ASSERT_EQUAL_INT16(0, h.am[c][i]);
}
```

---

### Test 2 — `test_hdc_int8_encode_bipolar`

**Invariant** : chaque composante de `hv_out` est exactement +1 ou −1 (pas 0, pas autre chose).

```c
void test_hdc_int8_encode_bipolar(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);
    float x[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) x[i] = (float)(i + 1);

    int8_t hv[HDC_I_D];
    hdc_int8_encode(&h, x, hv);

    for (int i = 0; i < HDC_I_D; i++) {
        TEST_ASSERT_TRUE(hv[i] == 1 || hv[i] == -1);
    }
}
```

---

### Test 3 — `test_hdc_int8_predict_after_updates`

**Invariant** : après 10 updates de la classe 0 avec `hv0` et 10 updates de la classe 1 avec `hv1` (orthogonaux en attendu), `predict(hv0)==0` et `predict(hv1)==1`.

```c
void test_hdc_int8_predict_after_updates(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);

    /* Deux inputs distincts → hypervecteurs distincts */
    float x0[HDC_I_N], x1[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) { x0[i] =  1.0f; x1[i] = -1.0f; }

    int8_t hv0[HDC_I_D], hv1[HDC_I_D];
    hdc_int8_encode(&h, x0, hv0);
    hdc_int8_encode(&h, x1, hv1);

    for (int k = 0; k < 10; k++) {
        hdc_int8_update(&h, hv0, 0);
        hdc_int8_update(&h, hv1, 1);
    }

    TEST_ASSERT_EQUAL_INT(0, hdc_int8_predict(&h, hv0));
    TEST_ASSERT_EQUAL_INT(1, hdc_int8_predict(&h, hv1));
}
```

> **Note** : avec D=2048 et 10 updates, am[0] et am[1] ont des profils très différents → la séparabilité est garantie par la haute dimension.

---

### Test 4 — `test_hdc_int8_update_accumulates`

**Invariant** : après N updates de la même classe avec le même hypervecteur, `am[c][i] == N * hv[i]` (tant que N×hv[i] ≤ 32 767, donc pour N≤3 sur valeurs ±1).

```c
void test_hdc_int8_update_accumulates(void)
{
    HDCInt8 h;
    hdc_int8_init(&h);

    float x[HDC_I_N];
    for (int i = 0; i < HDC_I_N; i++) x[i] = 1.0f;
    int8_t hv[HDC_I_D];
    hdc_int8_encode(&h, x, hv);

    hdc_int8_update(&h, hv, 0);
    hdc_int8_update(&h, hv, 0);
    hdc_int8_update(&h, hv, 0);

    /* am[0][0] == 3 * hv[0] == ±3 (pas de saturation pour N=3) */
    int16_t expected = (int16_t)(3 * (int)hv[0]);
    TEST_ASSERT_EQUAL_INT16(expected, h.am[0][0]);
}
```

---

### Test 5 — `test_hdc_int8_sizeof`

**Invariant** : la taille statique de `HDCInt8` correspond aux #defines (vérifie qu'aucun padding inattendu ne gonfle le .bss).

```c
void test_hdc_int8_sizeof(void)
{
    /* bv : HDC_I_N × HDC_I_D × sizeof(int8_t)  = 9 × 2048 × 1 = 18 432 B
     * am : HDC_I_C × HDC_I_D × sizeof(int16_t) = 4 × 2048 × 2 = 16 384 B
     * Total minimum attendu : 34 816 B (padding possible mais borné) */
    TEST_ASSERT_TRUE(sizeof(HDCInt8) >= (size_t)(HDC_I_N * HDC_I_D + HDC_I_C * HDC_I_D * 2));
    /* Vérif que le struct n'excède pas 36 Ko (10% de marge de padding max) */
    TEST_ASSERT_TRUE(sizeof(HDCInt8) <= 36864u);
}
```

---

## Tableau récapitulatif

| Test | Input | Invariant vérifié |
|------|-------|-------------------|
| `test_hdc_int8_init_zeros_am` | struct pollué → `hdc_int8_init()` | `am[c][i] == 0` pour tout c,i |
| `test_hdc_int8_encode_bipolar` | x = {1..9} | `hv[i] ∈ {-1, +1}` pour tout i |
| `test_hdc_int8_predict_after_updates` | x0=all+1, x1=all−1, 10 updates chacun | `predict(hv0)==0`, `predict(hv1)==1` |
| `test_hdc_int8_update_accumulates` | x=all+1, 3 updates classe 0 | `am[0][0] == 3*hv[0]` |
| `test_hdc_int8_sizeof` | — | `sizeof(HDCInt8) ∈ [34816, 36864]` |

---

## Intégration `test_runner.c`

Ajouter dans `firmware/stm32f4_blink/tests/test_runner.c` :

```c
/* ── Déclarations — test_hdc_int8.c (S2906) ─────────────────────────────── */
void test_hdc_int8_init_zeros_am(void);
void test_hdc_int8_encode_bipolar(void);
void test_hdc_int8_predict_after_updates(void);
void test_hdc_int8_update_accumulates(void);
void test_hdc_int8_sizeof(void);
```

Et dans `main()` :

```c
    /* HDC INT8 — S2906 */
    RUN_TEST(test_hdc_int8_init_zeros_am);
    RUN_TEST(test_hdc_int8_encode_bipolar);
    RUN_TEST(test_hdc_int8_predict_after_updates);
    RUN_TEST(test_hdc_int8_update_accumulates);
    RUN_TEST(test_hdc_int8_sizeof);
```

---

## Vérification

```bash
cd firmware/stm32f4_blink

# Build + run tests host (gcc x86)
make test

# Résultat attendu : 79 + 5 = 84 tests, 0 failures
# Les 2 failures TinyOL FP32 pré-existantes (hors périmètre) sont attendues

# Vérifier que le firmware cible compile sans régression
make all
arm-none-eabi-size build/stm32f4_blink.elf
# .bss total < 128 Ko
```
