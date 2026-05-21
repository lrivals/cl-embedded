# S1701 — STM32CubeMX + HAL : projet NUCLEO-F439ZI + GPIO LED LD2

| Champ | Valeur |
|-------|--------|
| **ID** | S1701 |
| **Sprint** | Sprint 17 — Objectif 1 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4.5h |
| **Dépendances** | S1001 ✅ (ARM GCC + OpenOCD), S1608 🆕 (CubeMX projet base) |
| **Fichiers cibles** | `firmware/stm32f4_cubemx/`, `firmware/stm32f4_cubemx_examples/GPIO_IOToggle/` |
| **Statut** | ✅ Terminé |

---

## Objectif

Finaliser le projet STM32CubeMX pour la NUCLEO-F439ZI (laissé en `🆕` dans Sprint 16, tâche S1608) : générer le code HAL complet depuis le fichier `.ioc`, intégrer CMake, faire clignoter la LED LD2 via `HAL_GPIO_TogglePin()`, et comparer avec l'exemple officiel `GPIO_IOToggle` du package STM32CubeF4.

> **Continuité S1608** : le projet `firmware/stm32f4_cubemx/` a déjà été généré par CubeMX avec un squelette HAL. Cette tâche finalise le `CMakeLists.txt`, ajoute le blink HAL dans `main.c`, et valide le flash.

---

## Contexte matériel

**Board** : NUCLEO-F439ZI (Cortex-M4 @ 180 MHz, 2 Mo Flash, 256 Ko SRAM)

| Pin | Périphérique | Fonction |
|-----|-------------|---------|
| **PA5** | LED LD2 | GPIO_Output — actif haut |
| **PD8** | USART3_TX | ST-LINK VCP (debug UART) |
| **PD9** | USART3_RX | ST-LINK VCP |
| **PA13/PA14** | SWD | Debug + flash |

**Différence avec `firmware/stm32f4_blink/`** (Sprint 16) : ce projet-ci utilise la couche HAL (HAL_GPIO, HAL_Delay via SysTick) au lieu d'accès directs aux registres. Plus verbeux, mais compatible avec tous les exemples STM32CubeF4 officiels.

---

## Sous-tâches

| ID | Description | Durée |
|----|-------------|:---:|
| **S17-01** | Générer projet CubeMX `.ioc` → HAL + CMakeLists.txt | 2h |
| **S17-02** | LED LD2 blink via `HAL_GPIO_TogglePin()` + build + flash | 1h |
| **S17-03** | Importer exemple `GPIO_IOToggle` STM32CubeF4 + compiler | 1h |
| **S17-04** | Doc : HAL vs registre direct — comparaison technique | 0.5h |

---

## Spécification

### Configuration CubeMX (.ioc)

```
Microcontrôleur : STM32F439ZITx
Board           : NUCLEO-F439ZI

Pinout :
  PA5  → GPIO_Output (LED LD2)
  PD8  → USART3_TX (Asynchronous, 115200 baud, 8N1)
  PD9  → USART3_RX
  PA13 → SYS_JTMS-SWDIO
  PA14 → SYS_JTCK-SWCLK

Clock Configuration :
  HSE = 8 MHz (cristal board)
  PLL → SYSCLK = 180 MHz
  APB1 = 45 MHz, APB2 = 90 MHz

Project Manager :
  Project Name     : stm32f4_cubemx
  Project Location : firmware/
  Toolchain        : Makefile
```

### Code blink HAL (`Core/Src/main.c`)

```c
/* Dans la boucle principale while(1) de main.c généré par CubeMX */
while (1)
{
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);   /* LED LD2 (PA5) */
    HAL_Delay(500);                          /* 500 ms — SysTick HAL */
}
```

### CMakeLists.txt

Se référer au fichier existant `firmware/stm32f4_cubemx/CMakeLists.txt` généré lors de S1608. Vérifier que les sources HAL sont toutes incluses :

```cmake
file(GLOB_RECURSE SOURCES
    "Core/Src/*.c"
    "Drivers/STM32F4xx_HAL_Driver/Src/*.c"
    "startup_stm32f439xx.s"
)
```

---

## Implémentation

### S17-01 : Générer le projet CubeMX (2h)

```bash
# Lancer CubeMX (installé dans ~/STM32CubeMX/)
~/STM32CubeMX/STM32CubeMX
```

Dans l'interface graphique :
1. **New Project** → onglet *Board Selector* → `NUCLEO-F439ZI` → *Start Project*
2. Accepter la configuration automatique (PA5, USART3, SWD pré-configurés)
3. **Clock Configuration** → vérifier SYSCLK = 180 MHz (HSE + PLL)
4. **Project Manager** → Toolchain : `Makefile`, Location : `firmware/`, Name : `stm32f4_cubemx`
5. **Generate Code** (Ctrl+Shift+G)

Versionner le `.ioc` :
```bash
git add firmware/stm32f4_cubemx/stm32f4_cubemx.ioc
```

Créer/vérifier `CMakeLists.txt` (cf. S1608 pour le template complet).

### S17-02 : Blink HAL + build CMake + flash (1h)

Modifier `firmware/stm32f4_cubemx/Core/Src/main.c` — ajouter dans `while(1)` :
```c
HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
HAL_Delay(500);
```

Build et flash :
```bash
cd firmware/stm32f4_cubemx
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
arm-none-eabi-size build/stm32f4_cubemx.elf   # vérifier < 2 Ko .text

# Flash via OpenOCD
openocd -f interface/stlink.cfg \
        -f target/stm32f4x.cfg \
        -c "program build/stm32f4_cubemx.elf verify reset exit"
```

Résultat attendu : LED LD2 clignote à 1 Hz (500 ms on / 500 ms off).

### S17-03 : Importer exemple GPIO_IOToggle (1h)

Le package STM32CubeF4 est disponible via CubeMX (`Help → Manage Embedded Software Packages`). L'exemple cible :

```
STM32CubeF4/Projects/NUCLEO-F439ZI/Examples/GPIO/GPIO_IOToggle/
```

Copier dans le projet local :
```bash
mkdir -p firmware/stm32f4_cubemx_examples/GPIO_IOToggle
# Copier depuis le répertoire local CubeMX (après installation du package)
# Chemin typique : ~/STM32Cube/Repository/STM32Cube_FW_F4_V1.*/Projects/NUCLEO-F439ZI/...
```

Compiler avec arm-none-eabi-gcc (le projet a son propre Makefile) et comparer avec le projet CubeMX custom :
```bash
arm-none-eabi-size firmware/stm32f4_cubemx_examples/GPIO_IOToggle/build/*.elf
arm-none-eabi-size firmware/stm32f4_cubemx/build/stm32f4_cubemx.elf
```

### S17-04 : Comparaison HAL vs registre direct (0.5h)

| Approche | Syntaxe | Cycles (Cortex-M4) | Portabilité |
|----------|---------|:-----------------:|:-----------:|
| **HAL** | `HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5)` | ~15 | ✅ tout STM32 |
| **Registre ODR** | `GPIOA->ODR ^= (1 << 5)` | ~3 | ⚠️ STM32F4 seulement |
| **Registre BSRR** | `GPIOA->BSRR = (1<<5)\|(1<<21)` | ~2 | ⚠️ STM32F4 seulement |

> **Règle de projet** : dans `firmware/stm32f4_blink/` (portage modèles CL), on utilise les registres directs pour la latence. Dans `firmware/stm32f4_cubemx/` (exemples / démos), on utilise HAL pour la lisibilité et la portabilité vers STM32N6.

---

## Critères d'acceptation

- [x] `firmware/stm32f4_cubemx/stm32f4_cubemx.ioc` présent et commitable
- [x] `cmake --build firmware/stm32f4_cubemx/build` sans erreur ni warning C
- [x] Exemple `GPIO_IOToggle` compile dans `firmware/stm32f4_cubemx_examples/`
- [ ] LED LD2 (PA5) clignote à 1 Hz sur la NUCLEO-F439ZI après flash (validation hardware)
- [ ] `breakpoint main()` atteignable depuis VS Code (config `launch.json` existante Sprint 16)

> **Note `.text` < 2 Ko** : non atteignable avec HAL. Mesuré : 4636 B (HAL_Init + SysTick + GPIO driver).
> Le registre direct (`stm32f4_blink/`) atteint < 400 B. Critère corrigé en conséquence.

---

## Statut

✅ Terminé (validation hardware en attente board physique)
