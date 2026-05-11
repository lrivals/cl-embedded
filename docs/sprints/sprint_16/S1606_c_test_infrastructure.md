# S1006 — Infrastructure de test C (Unity + CI firmware)

| Champ | Valeur |
|-------|--------|
| **ID** | S1006 |
| **Sprint** | Sprint 16 — Semaine 1b (20–27 mai 2026) |
| **Priorité** | Haute |
| **Durée estimée** | 8h |
| **Dépendances** | S1001 ✅ (toolchain ARM OK), `firmware/stm32f4_blink/` existant |
| **Fichiers cibles** | `firmware/stm32f4_blink/tests/`, `.github/workflows/firmware.yml` |

---

## Objectif

Mettre en place une infrastructure de test pour le code C embarqué afin de valider
`mahalanobis.c`, `ewc_head.c` et `pipeline.c` (créés en S1003) **avant** de flasher
sur la board.

**Deux niveaux de test** :
1. **Host build** : compilation native (x86) + tests unitaires Unity → CI GitHub Actions
2. **Board validation** : sorties firmware vs. sorties Python (bitwise match FP32)

---

## Contexte

`mahalanobis.c`, `ewc_head.c` et `pipeline.c` existent depuis S1003 mais sans
aucun test. Un bug silencieux dans le forward pass C invaliderait toutes les mesures
de S1004 (Hardware Characterization) sans être détecté.

---

## Sous-tâches

### 1. Intégrer le framework Unity

```bash
# Unity : framework de test unitaire C minimaliste, MCU-compatible
# Sources : https://github.com/ThrowTheSwitch/Unity
cd firmware/stm32f4_blink/
mkdir -p tests/unity
# Copier unity.c + unity.h + unity_internals.h depuis le dépôt officiel
```

Unity ne dépend que de la libc standard → compile en natif x86 et sur ARM Cortex-M.

### 2. Écrire les tests unitaires

#### `firmware/stm32f4_blink/tests/test_mahalanobis.c`

```c
#include "unity.h"
#include "mahalanobis.h"
#include <math.h>

void test_mahal_zero_distance(void) {
    /* Un vecteur identique à la moyenne → distance = 0 */
    float mean[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cov_inv[4][4] = { /* identité 4x4 */ };
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dist = mahal_distance(x, mean, (float*)cov_inv, 4);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, dist);
}

void test_mahal_known_value(void) {
    /* Valeur connue calculée en Python : numpy.linalg.norm */
    /* ... */
}
```

#### `firmware/stm32f4_blink/tests/test_ewc_head.c`

```c
#include "unity.h"
#include "ewc_head.h"

void test_ewc_forward_shape(void) {
    /* Forward pass : [1, 5] → [1, 3] (monitoring : 3 classes) */
}

void test_ewc_sgd_step_decreases_loss(void) {
    /* Une étape SGD doit réduire la loss sur l'échantillon courant */
}
```

### 3. Makefile — target `make test` (host build)

Ajouter dans `firmware/stm32f4_blink/Makefile` :

```makefile
# Host test build (x86 natif, pas ARM)
TEST_CC  = gcc
TEST_SRC = tests/test_mahalanobis.c tests/test_ewc_head.c \
           tests/unity/unity.c \
           src/mahalanobis.c src/ewc_head.c
TEST_INC = -Iinc -Itests/unity

test:
	$(TEST_CC) $(TEST_SRC) $(TEST_INC) -lm -o build/test_runner
	./build/test_runner

.PHONY: test
```

```bash
# Lancer les tests en local
cd firmware/stm32f4_blink/
make test
```

### 4. GitHub Actions CI

Créer `.github/workflows/firmware.yml` :

```yaml
name: Firmware CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install ARM toolchain
        run: sudo apt-get install -y gcc-arm-none-eabi binutils-arm-none-eabi
      - name: Build firmware
        run: make -C firmware/stm32f4_blink/ all
      - name: Run host unit tests
        run: make -C firmware/stm32f4_blink/ test
```

### 5. Validation bitwise Python vs. C

```bash
# 1. Exporter des vecteurs de test depuis Python
python scripts/export_weights_c.py --dump-test-vectors \
    --output firmware/stm32f4_blink/tests/test_vectors.h

# 2. Inclure test_vectors.h dans test_mahalanobis.c
# 3. Comparer : sortie C == sortie numpy (tolérance 1e-6 FP32)
```

---

## Critères d'acceptation

- [x] `make test` compile et passe tous les tests unitaires sur x86 (16/16 PASS, zero failures)
- [x] CI GitHub Actions passe sur `main` (build ARM + tests host) — `.github/workflows/firmware.yml` configuré
- [x] `test_mahalanobis.c` : distance sur vecteur connu == valeur Python (tol 1e-5)
- [x] `test_ewc_head.c` : forward pass shape correcte + loss décroît après SGD step
- [x] `tests/` ajouté au `.gitignore` uniquement pour les binaires (`build/`) — couvert par `build/` dans le `.gitignore` racine

---

## Références

- [Unity C Test Framework](https://github.com/ThrowTheSwitch/Unity)
- [Cortex-M Testing Guide — ST](https://wiki.st.com/stm32mcu/wiki/STM32StepByStep:STM32_Unit_Test)
