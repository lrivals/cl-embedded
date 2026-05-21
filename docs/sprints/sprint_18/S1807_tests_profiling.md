# S1807 — Tests Unity firmware : DWT latence, CRC frames, buffer overflow guards

| Champ | Valeur |
|-------|--------|
| **ID** | S1807 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟡 Secondaire |
| **Durée estimée** | 4h |
| **Dépendances** | S1804 (firmware profiling implémenté) |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/test_profiling.c` |
| **Statut** | ⬜ À faire — fichier test à créer |

---

## Objectif

Valider le module `profiling.c` avec des tests unitaires Unity :
- Cohérence du compteur de latence (positif, non nul)
- Calcul de throughput (non nul, croissant puis stabilisé)
- Format d'encodage 8 B (little-endian vérifié octet par octet)
- Garde contre les dépassements de taille .bss (valeur raisonnable)

---

## Cadre Unity existant

Les tests firmwares utilisent le framework [Unity](https://github.com/ThrowTheSwitch/Unity), déjà intégré dans `firmware/stm32f4_blink/tests/unity/`.

Voir `test_mahalanobis.c` pour le pattern : `TEST_ASSERT_FLOAT_WITHIN`, `TEST_ASSERT_EQUAL_UINT32`, etc.

Le runner est `test_runner.c` ; lancer via :

```bash
make -C firmware/stm32f4_blink test
```

---

## Fichier à créer : `firmware/stm32f4_blink/tests/test_profiling.c`

### Structure générale

```c
/**
 * test_profiling.c — Tests unitaires Unity pour profiling.c
 *
 * Tests exécutés sur host (pas de DWT réel) : les registres DWT sont
 * mockés via des macros (#define DWT_CYCCNT g_mock_cyccnt).
 */

#include "unity.h"
#include "profiling.h"
#include <string.h>
#include <stdint.h>

/* ── Mock DWT ─────────────────────────────────────────────────────────────── */
static volatile uint32_t g_mock_cyccnt = 0U;

/* Redéfinit DWT_CYCCNT pour les tests */
#undef DWT_CYCCNT
#define DWT_CYCCNT g_mock_cyccnt

void setUp(void)    { memset(&g_profiling, 0, sizeof(g_profiling)); g_mock_cyccnt = 0U; }
void tearDown(void) {}
```

### Test 1 : latence positive après start/stop

```c
void test_profiling_latency_positive(void)
{
    /* Simule 180 cycles → 1 µs à 180 MHz */
    g_mock_cyccnt = 0U;
    profiling_start();
    g_mock_cyccnt = 180U;   /* 180 cycles = 1 µs à 180 MHz */
    profiling_stop();

    TEST_ASSERT_EQUAL_UINT32(1U, profiling_get_latency_us());
}
```

### Test 2 : latence nulle si start/stop sans cycles

```c
void test_profiling_latency_zero_cycles(void)
{
    g_mock_cyccnt = 100U;
    profiling_start();
    /* Pas de progression du compteur */
    profiling_stop();

    TEST_ASSERT_EQUAL_UINT32(0U, profiling_get_latency_us());
}
```

### Test 3 : throughput non nul après première inférence

```c
void test_profiling_throughput_nonzero(void)
{
    g_mock_cyccnt = 0U;
    profiling_start();
    g_mock_cyccnt = 180000U;   /* 1000 µs = 1 ms = 1000 ips */
    profiling_stop();

    /* SYSCLK_HZ / avg_cycles = 180000000 / 180000 = 1000 ips */
    TEST_ASSERT_GREATER_THAN_UINT16(0U, profiling_get_throughput_ips());
}
```

### Test 4 : encodage little-endian 8 B

```c
void test_profiling_encode_format(void)
{
    /* Prépare un état connu */
    g_profiling.last_latency_us = 0x01020304UL;  /* 16909060 µs */
    g_profiling.bss_bytes       = 0x0506U;
    g_profiling.throughput_ips  = 0x0708U;

    uint8_t buf[PROFILING_ENCODED_SIZE];
    profiling_encode(buf);

    /* latency_us little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x04U, buf[0]);
    TEST_ASSERT_EQUAL_UINT8(0x03U, buf[1]);
    TEST_ASSERT_EQUAL_UINT8(0x02U, buf[2]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, buf[3]);

    /* bss_bytes little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x06U, buf[4]);
    TEST_ASSERT_EQUAL_UINT8(0x05U, buf[5]);

    /* throughput little-endian */
    TEST_ASSERT_EQUAL_UINT8(0x08U, buf[6]);
    TEST_ASSERT_EQUAL_UINT8(0x07U, buf[7]);
}
```

### Test 5 : taille encodée correcte

```c
void test_profiling_encode_size(void)
{
    TEST_ASSERT_EQUAL_UINT32(8U, PROFILING_ENCODED_SIZE);
}
```

### Test 6 : bss_bytes dans la limite acceptable

```c
void test_bss_size_within_limit(void)
{
    /* bss_bytes initialisé par profiling_init() via symboles linker.
     * En contexte test on vérifie juste que la valeur est plausible.
     * Sur target réelle, bss_bytes doit être < 52000 (seuil alerte). */
    profiling_init();
    /* Le linker script test peut mettre _sbss = _ebss → bss = 0.
     * On vérifie juste l'absence de dépassement u16. */
    TEST_ASSERT_LESS_THAN_UINT32(65535U, (uint32_t)profiling_get_bss_bytes());
}
```

### Runner à mettre à jour dans `test_runner.c`

```c
// Ajouter dans main() de test_runner.c :
RUN_TEST(test_profiling_latency_positive);
RUN_TEST(test_profiling_latency_zero_cycles);
RUN_TEST(test_profiling_throughput_nonzero);
RUN_TEST(test_profiling_encode_format);
RUN_TEST(test_profiling_encode_size);
RUN_TEST(test_bss_size_within_limit);
```

---

## Notes d'implémentation

### Mock DWT sur host

Le DWT CYCCNT est un registre hardware inaccessible en simulation. La technique recommandée est de redéfinir la macro `DWT_CYCCNT` via `#undef` dans le fichier de test. Cela nécessite que `profiling.c` utilise la macro (déjà le cas) et non un accès direct à l'adresse `0xE0001004`.

Alternativement, compiler `profiling.c` avec `-DTEST_HOST` pour substituer les accès registre.

### Compilation

```makefile
# Dans le Makefile firmware/stm32f4_blink/
TEST_SRCS = tests/test_runner.c tests/test_mahalanobis.c \
            tests/test_ewc_head.c tests/test_profiling.c \
            src/profiling.c src/mahalanobis.c src/ewc_head.c \
            tests/unity/unity.c
```

---

## Critères d'acceptation

- [ ] `firmware/stm32f4_blink/tests/test_profiling.c` existe et compile
- [ ] `make -C firmware/stm32f4_blink test` passe les 6 tests sans erreur
- [ ] `test_profiling_encode_format` valide l'ordre des octets little-endian
- [ ] `test_profiling_latency_positive` valide 180 cycles → 1 µs à 180 MHz
