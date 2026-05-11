# STM32F439ZI Blink — Template Phase 2 CL-Embedded

Projet de validation de la chaîne compile → flash → debug sur NUCLEO-F439ZI.
Sert de template pour le portage C des modèles CL (Phase 2 Sprint P2-03).

## Prérequis

```bash
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi openocd
# VS Code extensions : ms-vscode.cpptools, marus25.cortex-debug
```

## Build

```bash
make -j4
# → build/stm32f4_blink.elf + build/stm32f4_blink.bin
```

## Flash

```bash
make flash
# ou : Ctrl+Shift+B → "flash-stm32" dans VS Code
```

## Debug

```
F5 dans VS Code → arrêt sur main(), LED LD2 (PA5) doit clignoter
```

## Structure

```
stm32f4_blink/
├── src/main.c                        ← blink PA5, délai SW
├── startup/startup_stm32f439xx.s     ← vecteurs + Reset_Handler
├── linker/STM32F439ZITx_FLASH.ld     ← 2 Mo Flash / 256 Ko RAM
├── inc/stm32f4xx.h                   ← registres minimaux (RCC, GPIO)
├── Makefile                          ← build + flash targets
└── .vscode/
    ├── launch.json                   ← Cortex-Debug config
    ├── tasks.json                    ← build / flash tasks
    └── c_cpp_properties.json         ← IntelliSense ARM
```

## Prochain sprint (P2-03)

Ce template sera étendu pour porter `MahalanobisDetector` en C :
- `src/mahalanobis.c` / `inc/mahalanobis.h` — forward pass + update incrémental
- Mesure RAM : linker map + DWT cycle counter
