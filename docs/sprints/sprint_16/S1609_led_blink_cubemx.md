# S1609 — LED blink NUCLEO-F439ZI via CubeMX + CMake

| Champ | Valeur |
|-------|--------|
| **ID** | S1609 |
| **Sprint** | Sprint 16 — ajout post-clôture (20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S1608 🆕 (projet CubeMX + CMake configuré) |
| **Fichier cible** | `firmware/stm32f4_cubemx/Core/Src/main.c` |
| **Statut** | ✅ Terminé — 20 mai 2026 |

---

## Objectif

Premier programme concret sur la NUCLEO-F439ZI avec le projet généré par CubeMX : faire clignoter la LED LD2 (PA5) via le HAL STM32 et le build CMake. C'est le "Hello World" embarqué, point de départ de toutes les démos encadrants.

---

## Contexte

S1001 avait déjà un blink mais dans un projet créé manuellement (`firmware/stm32f4_blink/`). Ce blink utilise le **projet CubeMX** (`firmware/stm32f4_cubemx/`) buildé avec CMake, ce qui valide que toute la chaîne CubeMX → CMake → flash fonctionne de bout en bout.

---

## Implémentation

Dans `firmware/stm32f4_cubemx/Core/Src/main.c`, dans la boucle `while(1)` générée par CubeMX :

```c
/* USER CODE BEGIN WHILE */
while (1)
{
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);  /* PA5 — LED LD2 */
    HAL_Delay(500);                               /* 500 ms → 1 Hz */
    /* USER CODE END WHILE */
}
```

> Les macros `LD2_GPIO_Port` et `LD2_Pin` sont générées automatiquement par CubeMX dans `Core/Inc/main.h` quand le board selector NUCLEO-F439ZI est utilisé.

---

## Build et flash

```bash
# Compiler
cd firmware/stm32f4_cubemx
cmake --build build -j4

# Flash via OpenOCD
openocd -f interface/stlink.cfg \
        -f target/stm32f4x.cfg \
        -c "program build/stm32f4_cubemx.elf verify reset exit"
```

---

## Critères d'acceptation

- [x] `cmake --build build` passe sans erreur — Flash 4 384 B / RAM 1 584 B
- [x] LED LD2 (PA5) clignote à ~1 Hz sur la board physique — flashé et vérifié via OpenOCD
- [x] Seul `USER CODE` modifié dans `main.c` et `main.h` (le reste généré par CubeMX reste intact)
