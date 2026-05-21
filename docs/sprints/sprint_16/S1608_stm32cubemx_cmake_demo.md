# S1608 — STM32CubeMX + CMake : projet board + démo encadrants

| Champ | Valeur |
|-------|--------|
| **ID** | S1608 |
| **Sprint** | Sprint 16 — ajout post-clôture (20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S1001 ✅ (toolchain ARM + OpenOCD OK), S1004 ✅ (caractérisation board NUCLEO-F439ZI) |
| **Fichiers cibles** | `firmware/stm32f4_cubemx/`, `firmware/stm32f4_cubemx/CMakeLists.txt`, `docs/embedded_demo_guide.md` |
| **Statut** | 🆕 À faire |

---

## Objectif

Utiliser **STM32CubeMX 6.17.0** pour générer un projet C complet pour la NUCLEO-F439ZI, intégrer **CMake 4.3.2** comme build system, et préparer une démo reproductible pour les encadrants (Arnaud, Dorra, Fred).

> ⚠️ **STM32CubeMX ≠ STM32Cube.AI**
> - **STM32CubeMX** (✅ installé) : génère le code d'initialisation HAL (horloges, pins, périphériques)
> - **STM32Cube.AI** (⏸ bloqué `TODO(dorra)`) : convertit les réseaux neuronaux en code C INT8
>
> Cette tâche utilise uniquement STM32CubeMX.

---

## Contexte

S1001 a mis en place la toolchain ARM GCC + OpenOCD + VS Code avec un projet blink minimal (`firmware/stm32f4_blink/`). Ce projet a été créé à la main. STM32CubeMX permet de regénérer automatiquement la configuration HAL (horloges, périphériques, linker script) à partir d'un fichier `.ioc`, ce qui est plus robuste et maintenable pour les démos.

---

## Sous-tâches

### 1. Créer le projet STM32CubeMX pour NUCLEO-F439ZI (1h)

Lancer STM32CubeMX :
```bash
~/STM32CubeMX/STM32CubeMX
```

Étapes dans l'interface :
1. **New Project** → onglet *Board Selector* → chercher `NUCLEO-F439ZI` → *Start Project*
   (CubeMX pré-configure automatiquement ST-LINK, LED PA5, USART3)
2. Vérifier la configuration minimale :
   - **PA5** → GPIO_Output (LED LD2)
   - **USART3** → Asynchronous, 115200 baud (câble ST-LINK USB)
   - **SWD** → activé (debug)
   - **Horloge** → HSE, PLL → SYSCLK = 180 MHz (confirmer avec S1004)
3. **Project Manager** :
   - Project Name : `stm32f4_cubemx`
   - Project Location : `firmware/`
   - Toolchain/IDE : **Makefile**
4. **Generate Code** → dossier `firmware/stm32f4_cubemx/` créé

Versionner le fichier `.ioc` :
```bash
# Le fichier .ioc est la source de vérité CubeMX — à committer
git add firmware/stm32f4_cubemx/stm32f4_cubemx.ioc
```

### 2. Intégrer CMake dans le projet généré (1h)

CubeMX génère un `Makefile` natif. Créer un `CMakeLists.txt` par-dessus pour l'intégration VSCode :

**`firmware/stm32f4_cubemx/CMakeLists.txt`** :
```cmake
cmake_minimum_required(VERSION 3.20)
project(stm32f4_cubemx C ASM)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Cross-compiler ARM GCC
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_SIZE arm-none-eabi-size)

# Flags MCU (Cortex-M4 FPU)
set(MCU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS "${MCU_FLAGS} -Wall -fdata-sections -ffunction-sections")
set(CMAKE_EXE_LINKER_FLAGS "${MCU_FLAGS} -specs=nano.specs -lc -lm -lnosys \
    -Wl,--gc-sections -T${CMAKE_SOURCE_DIR}/STM32F439ZITx_FLASH.ld")

# Sources générées par CubeMX
file(GLOB_RECURSE SOURCES
    "Core/Src/*.c"
    "Drivers/STM32F4xx_HAL_Driver/Src/*.c"
    "startup_stm32f439xx.s"
)

include_directories(
    Core/Inc
    Drivers/STM32F4xx_HAL_Driver/Inc
    Drivers/CMSIS/Device/ST/STM32F4xx/Include
    Drivers/CMSIS/Include
)

add_definitions(-DSTM32F439xx -DUSE_HAL_DRIVER)

add_executable(${PROJECT_NAME}.elf ${SOURCES})

# Générer .bin et .hex pour flash
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.hex
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_NAME}.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${PROJECT_NAME}>
)
```

Build :
```bash
cd firmware/stm32f4_cubemx
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
# → build/stm32f4_cubemx.elf + .hex + .bin
```

### 3. Préparer la démo encadrants (30min)

Séquence de démo reproductible (≈ 10 min) :

**Étape A — Montrer STM32CubeMX** :
```
1. Ouvrir firmware/stm32f4_cubemx/stm32f4_cubemx.ioc dans CubeMX
2. Montrer la vue Pinout (PA5 = LED, USART3, SWD)
3. Montrer la vue Clock Configuration (180 MHz)
4. Montrer que "Generate Code" régénère le HAL automatiquement
```

**Étape B — Compiler avec CMake** :
```bash
cd firmware/stm32f4_cubemx
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
# Montrer la taille du firmware : arm-none-eabi-size build/stm32f4_cubemx.elf
```

**Étape C — Flasher et valider** :
```bash
# Flash via OpenOCD
openocd -f interface/stlink.cfg \
        -f target/stm32f4x.cfg \
        -c "program build/stm32f4_cubemx.elf verify reset exit"
# → LED PA5 clignote = succès
```

**Étape D — UART** :
```bash
# Ouvrir terminal série dans VS Code (extension Serial Monitor)
# Port : /dev/ttyACM0, 115200 baud
# Observer les messages HAL/UART de la board
```

### 4. Documenter dans `docs/embedded_demo_guide.md` (30min)

Créer un guide reproductible couvrant :
- Prérequis (outils installés, board connectée)
- Commandes CubeMX → CMake → flash → UART
- Troubleshooting fréquent (port USB, permissions udev)

---

## Critères d'acceptation

- [ ] `firmware/stm32f4_cubemx/stm32f4_cubemx.ioc` présent et versionné
- [ ] `cmake --build firmware/stm32f4_cubemx/build` produit un `.elf` sans erreur
- [ ] `arm-none-eabi-size` confirme Flash < 512 Ko et RAM < 192 Ko
- [ ] LED PA5 clignote après flash sur NUCLEO-F439ZI
- [ ] `docs/embedded_demo_guide.md` créé avec toutes les commandes

---

## Questions ouvertes

- `TODO(arnaud)` : quel niveau de détail technique pour la démo ? (montrer CubeMX seul ou pipeline complet jusqu'au UART ?)
- `TODO(dorra)` : STM32CubeMX peut aussi exporter vers STM32CubeIDE — pertinent si Cube.AI nécessite CubeIDE pour l'intégration NPU STM32N6 ?
