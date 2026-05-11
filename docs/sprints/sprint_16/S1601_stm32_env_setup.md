# S1001 — Setup environnement de développement STM32 (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **ID** | S1001 |
| **Sprint** | Sprint 16 — Semaine 1 (20–27 mai 2026) |
| **Priorité** | Critique |
| **Durée estimée** | 3h |
| **Dépendances** | Sprint 15 terminé (Phase 1 anomaly detection complète) |
| **Fichiers cibles** | `firmware/stm32f4_blink/` |

---

## Objectif

Disposer d'un environnement de développement embarqué opérationnel sur la **NUCLEO-F439ZI** via VS Code : compilation ARM GCC, flash OpenOCD, debug Cortex-Debug.

> **Important** : la NUCLEO-F439ZI (STM32F439ZI — Cortex-M4, 256 Ko RAM, **pas de NPU**) est une board de développement intermédiaire. La cible finale est le **STM32N6** (Cortex-M55, ~64 Ko RAM modèle, NPU inférence-only). Ce sprint valide la chaîne d'outillage (compile → flash → debug) avant d'avoir accès au hardware cible.

**Critère de succès** : LED LD2 (PA5) clignote sur la NUCLEO-F439ZI, un breakpoint est atteignable depuis VS Code via ST-LINK, le `launch.json` est reproductible.

---

## Carte de développement

**Carte** : NUCLEO-F439ZI  
**Microcontrôleur** : STM32F439ZI (gamme STM32F4)

| Caractéristique | Valeur |
|-----------------|--------|
| Cœur | ARM Cortex-M4 @ 180 MHz |
| FPU + DSP | Oui |
| Flash | 2 Mo |
| RAM | 256 Ko |
| NPU | Non |
| Debugger intégré | ST-LINK (pas de sonde externe nécessaire) |
| LED LD2 | PA5 (GPIO_Output) |

---

## Sous-tâches

### 1. Installer la toolchain ARM GCC + OpenOCD

```bash
# Ubuntu/Debian
sudo apt install -y gcc-arm-none-eabi binutils-arm-none-eabi openocd

# Vérifier
arm-none-eabi-gcc --version   # ≥ 10.x attendu
openocd --version              # ≥ 0.11 attendu
```

### 2. Installer VS Code + extensions

```bash
code --install-extension ms-vscode.cpptools
code --install-extension marus25.cortex-debug
```

| Extension | Rôle |
|-----------|------|
| C/C++ (Microsoft) | IntelliSense, débogage natif |
| Cortex-Debug | Débogage MCU via OpenOCD/GDB |

### 3. Projet blink — fichiers créés

Le projet minimal sans CubeMX est dans `firmware/stm32f4_blink/` :

```
firmware/stm32f4_blink/
├── src/main.c                        ← blink PA5, délai SW pur registre
├── startup/startup_stm32f439xx.s     ← vecteurs + Reset_Handler (CMSIS-style)
├── linker/STM32F439ZITx_FLASH.ld     ← 2 Mo Flash / 256 Ko RAM
├── inc/stm32f4xx.h                   ← registres RCC + GPIOA (minimal)
├── Makefile                          ← build + flash targets
└── .vscode/
    ├── launch.json                   ← Cortex-Debug (OpenOCD + ST-LINK)
    ├── tasks.json                    ← tasks build / flash / clean
    └── c_cpp_properties.json         ← IntelliSense ARM GCC
```

**Note architecture** : le `main.c` accède directement aux registres (pas de HAL) pour rester proche du modèle de programmation MCU nu utilisé en Sprint 16 pour le portage des modèles CL.

### 4. Compiler

```bash
cd firmware/stm32f4_blink
make -j4
# Sortie attendue :
#   arm-none-eabi-size build/stm32f4_blink.elf
#      text    data     bss     dec     hex
#       NNN       0       0     NNN     NNN  build/stm32f4_blink.elf
```

### 5. Flasher via OpenOCD

```bash
make flash
# ou manuellement :
openocd -f interface/stlink.cfg \
        -f target/stm32f4x.cfg \
        -c "program build/stm32f4_blink.elf verify reset exit"
```

### 6. Debug depuis VS Code

1. Ouvrir VS Code à la racine du repo (`cl-embedded/`)
2. `Ctrl+Shift+P` → `Tasks: Run Task` → `build-stm32` (ou `Ctrl+Shift+B`)
3. `F5` → sélectionner `Debug STM32 (NUCLEO-F439ZI)`
4. Le programme s'arrête sur `main()`
5. Poser un breakpoint dans la boucle `while(1)` → `F5` → continuer → vérifier l'arrêt

**Configuration `launch.json`** (dans `firmware/stm32f4_blink/.vscode/launch.json`) :

```json
{
    "name": "Debug STM32 (NUCLEO-F439ZI)",
    "type": "cortex-debug",
    "request": "launch",
    "servertype": "openocd",
    "configFiles": ["interface/stlink.cfg", "target/stm32f4x.cfg"],
    "cwd": "${workspaceRoot}/firmware/stm32f4_blink",
    "executable": "${workspaceRoot}/firmware/stm32f4_blink/build/stm32f4_blink.elf",
    "runToEntryPoint": "main"
}
```

---

## Critères d'acceptation

- [x] `arm-none-eabi-gcc --version` retourne une version ≥ 10.x (13.3.1 installé)
- [x] `openocd --version` retourne une version ≥ 0.11 (0.12.0)
- [x] `make -j4` compile sans erreur dans `firmware/stm32f4_blink/`
- [x] La LED LD2 (PA5) clignote sur la carte après `make flash` (ST-LINK V2J47, 680 B, "Verified OK")
- [x] Un breakpoint dans `while(1)` est atteignable depuis VS Code (`F5`)

---

## Questions ouvertes

- `TODO(arnaud)` : confirmer si des évaluations intermédiaires sont prévues sur la NUCLEO-F439ZI ou si on bascule directement sur STM32N6 dès que disponible
- `TODO(dorra)` : planning d'accès au STM32N6 réel — à partir de quel sprint ?
- `TODO(fred)` : Edge Spectrum a-t-il une board STM32N6 pour les tests de P2-06 ?

---

## Notes

- La NUCLEO-F439ZI dispose d'un ST-LINK intégré → pas besoin de sonde externe
- Pour le portage STM32N6 (cible finale) : `target/stm32f4x.cfg` → `target/stm32n6x.cfg`, le reste du workflow est identique
- Le projet blink sert de **template** pour S1003 (portage C des modèles CL)
- Pas de dépendance HAL/CubeMX → code facilement reproductible et portable

**Complété le** : 11 mai 2026

**Versions installées** :
- `arm-none-eabi-gcc` 13.3.1 20240614 (ARM GNU Toolchain 13.3.Rel1) — installé dans `~/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/`
- `openocd` 0.12.0 — déjà présent sur le système
- PATH configuré dans `~/.bashrc`

**Flash validé** :
- ST-LINK V2J47, Cortex-M4 r0p1, Flash 2048 Ko dual-bank
- Firmware 680 B (0.03% Flash, 0% RAM)
- `** Verified OK **` — LED LD2 (PA5) clignote à 500 ms
