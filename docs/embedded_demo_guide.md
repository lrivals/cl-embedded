# Guide — Démo embarqué STM32CubeMX + CMake

Pipeline complet : génération HAL → build CMake → flash OpenOCD → terminal UART.
Durée estimée : 10 minutes avec la board connectée.

---

## Prérequis

### Outils

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| `arm-none-eabi-gcc` | 13.x | `arm-none-eabi-gcc --version` |
| `cmake` | ≥ 3.20 | `cmake --version` |
| `openocd` | ≥ 0.11 | `openocd --version` |
| STM32CubeMX | 6.17.0 | `~/STM32CubeMX/STM32CubeMX` |

### Board

- NUCLEO-F439ZI connectée en USB (câble CN1 = ST-LINK)
- Vérifier la détection : `lsusb | grep STMicro`
- Port série disponible : `ls /dev/ttyACM*`

### Permissions udev (Linux, une seule fois)

Si OpenOCD retourne `Permission denied` ou `LIBUSB_ERROR_ACCESS` :

```bash
sudo tee /etc/udev/rules.d/49-stm32.rules <<'EOF'
# ST-LINK V2/V3
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374e", MODE="0666", GROUP="plugdev"
# Port série ST-LINK
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE="0666", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
# Déconnecter / reconnecter la board
```

---

## Étape 1 — Générer le projet STM32CubeMX

Lancer CubeMX :

```bash
~/STM32CubeMX/STM32CubeMX
```

Séquence dans l'interface :

1. **New Project** → onglet *Board Selector* → rechercher `NUCLEO-F439ZI` → *Start Project*
   CubeMX pré-configure ST-LINK, LED PA5, USART3 automatiquement.

2. Vérifier la configuration **Pinout & Configuration** :
   - `PA5` → GPIO_Output (LED LD2)
   - `USART3` → Mode : Asynchronous — Baud Rate : 115200
   - `SYS` → Debug : Serial Wire (SWD)

3. Vérifier **Clock Configuration** :
   - Source : HSE → PLL
   - SYSCLK = **180 MHz** (NUCLEO-F439ZI limite hardware)

4. Onglet **Project Manager** :
   - Project Name : `stm32f4_cubemx`
   - Project Location : `<chemin-absolu-vers>/cl-embedded/firmware/`
   - Toolchain / IDE : **Makefile**

5. Cliquer **GENERATE CODE** → dossier `firmware/stm32f4_cubemx/` créé.

6. Committer le fichier `.ioc` (source de vérité CubeMX) :

```bash
git add firmware/stm32f4_cubemx/stm32f4_cubemx.ioc
git commit -m "feat(firmware): add STM32CubeMX project for NUCLEO-F439ZI"
```

> Si CubeMX propose de régénérer le code après modification de la config,
> le `CMakeLists.txt` reste intact car il n'est pas un fichier généré.

---

## Étape 2 — Build avec CMake

```bash
cd firmware/stm32f4_cubemx

# Configurer (cross-compilation Cortex-M4)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Compiler (4 threads)
cmake --build build -j4
```

Sorties attendues dans `build/` :

```
build/
├── stm32f4_cubemx.elf   ← image principale (debug + symboles)
├── stm32f4_cubemx.hex   ← format Intel HEX (flash)
├── stm32f4_cubemx.bin   ← image binaire brute
└── stm32f4_cubemx.map   ← carte mémoire (sections, symboles)
```

Vérifier la taille avec les contraintes hardware :

```bash
arm-none-eabi-size build/stm32f4_cubemx.elf
```

Résultat attendu (initialisation HAL seule) :

```
   text    data     bss     dec     hex filename
  ~8000     ~20   ~1500   ~9520    2530 build/stm32f4_cubemx.elf
# Flash < 512 Ko ✓   RAM < 192 Ko ✓
```

---

## Étape 3 — Flash via OpenOCD

Board connectée en USB, depuis `firmware/stm32f4_cubemx/` :

```bash
# Option A — cible CMake (recommandé)
cmake --build build --target flash

# Option B — commande directe
openocd \
  -f interface/stlink.cfg \
  -f target/stm32f4x.cfg \
  -c "program build/stm32f4_cubemx.elf verify reset exit"
```

Sortie attendue :

```
Info : STLINK V2J45M30 (API v2) VID:PID 0483:374B
...
** Programming Started **
** Programming Finished **
** Verify Started **
** Verified OK **
** Resetting Target **
```

---

## Étape 4 — Terminal UART

La NUCLEO-F439ZI expose USART3 via le pont USB/série ST-LINK.

```bash
# Identifier le port
ls /dev/ttyACM*   # typiquement /dev/ttyACM0

# Terminal avec minicom
minicom -D /dev/ttyACM0 -b 115200

# Terminal avec picocom (alternative)
picocom -b 115200 /dev/ttyACM0
```

Dans VS Code : extension **Serial Monitor** (Microsoft) — *Start Monitoring* sur `/dev/ttyACM0` à 115200 baud.

Pour envoyer des messages depuis la board, ajouter dans `Core/Src/main.c` (après `MX_USART3_UART_Init()`) :

```c
/* USER CODE BEGIN 2 */
char msg[] = "CL-Embedded demo OK\r\n";
HAL_UART_Transmit(&huart3, (uint8_t*)msg, sizeof(msg)-1, HAL_MAX_DELAY);
/* USER CODE END 2 */
```

Puis rebuild et re-flash.

---

## Troubleshooting

### `arm-none-eabi-gcc: command not found`

Le compilateur n'est pas sur le PATH. Ajouter à `~/.bashrc` :

```bash
export PATH="$HOME/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin:$PATH"
```

Ou appeler cmake avec le chemin complet :

```bash
cmake -B build -DCMAKE_C_COMPILER=/usr/bin/arm-none-eabi-gcc \
               -DCMAKE_ASM_COMPILER=/usr/bin/arm-none-eabi-gcc
```

### `Error: No rule to make target 'startup_stm32f439xx.s'`

CubeMX n'a pas encore généré le projet. Vérifier que `firmware/stm32f4_cubemx/startup_stm32f439xx.s` existe.

### `openocd: Error: open failed` ou `LIBUSB_ERROR_ACCESS`

Appliquer les règles udev (voir section Prérequis) ou lancer openocd avec `sudo` pour tester.

### Port `/dev/ttyACM0` occupé

```bash
fuser /dev/ttyACM0          # identifier le processus
kill <PID>                  # libérer le port
```

### CubeMX écrase mes modifications dans `Core/Src/main.c`

CubeMX ne touche que le code **hors** des blocs `/* USER CODE BEGIN */` / `/* USER CODE END */`. Toujours placer le code applicatif dans ces blocs.

### CMake détecte le mauvais compilateur (x86 au lieu d'ARM)

Supprimer le cache et reconfigurer :

```bash
rm -rf build/
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

---

## HAL vs Registre Direct — Comparaison technique (S17-04)

Deux projets coexistent dans ce dépôt avec des approches différentes d'accès au GPIO :

| Approche | Syntaxe toggle PA5 | Cycles Cortex-M4 | Portabilité |
| -------- | ------------------ | :--------------: | :---------: |
| **HAL** | `HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5)` | ~15 | ✅ Tout STM32 |
| **Registre ODR** | `GPIOA->ODR ^= (1U << 5)` | ~3 | ⚠️ STM32F4 seulement |
| **Registre BSRR** | `GPIOA->BSRR = (1U<<5)\|(1U<<21)` | ~2 | ⚠️ STM32F4 seulement |

**Règle de projet** :

- `firmware/stm32f4_blink/` — registres directs : portage modèles CL, latence < 100 ms mesurée (cf. `hw_info.c` DWT)
- `firmware/stm32f4_cubemx/` — HAL : démos encadrants, compatibilité future STM32N6 / Cortex-M55

**Taille `.text` mesurée** (blink seul, `-Og -g3`) :

```text
stm32f4_cubemx    :  text=4636  data=12  bss=1572  (HAL blink 500ms, HSI 16 MHz)
gpio_iotoggle     :  text=4656  data=24  bss=1592  (HAL blink BSP, HSE 180 MHz)
```

HAL ajoute ~4 Ko de code (.text) vs < 400 B pour le registre direct. Sur le STM32N6 (2 Mo Flash),
cet overhead est négligeable pour les démos, mais significatif dans la limite de 64 Ko du modèle CL.

---

## Exemple GPIO_IOToggle (STM32CubeF4 officiel)

Source : `STM32CubeF4 V1.28.3 / Projects/STM32F429ZI-Nucleo/Examples/GPIO/GPIO_IOToggle/`  
Copié dans : `firmware/stm32f4_cubemx_examples/GPIO_IOToggle/`

Différences avec `stm32f4_cubemx` :

- Utilise la couche BSP Nucleo-144 (`LED1_GPIO_CLK_ENABLE`, `BSP_LED_Off`)
- Horloge HSE bypass → 180 MHz (F429 → compatible F439)
- Toggle LED1 + LED2 à 100 ms chacune (F429 expose deux LEDs : PG13, PG14)

Build :

```bash
cmake -S firmware/stm32f4_cubemx_examples/GPIO_IOToggle \
      -B firmware/stm32f4_cubemx_examples/GPIO_IOToggle/build \
      -DCMAKE_BUILD_TYPE=Debug
cmake --build firmware/stm32f4_cubemx_examples/GPIO_IOToggle/build -j4
arm-none-eabi-size firmware/stm32f4_cubemx_examples/GPIO_IOToggle/build/gpio_iotoggle.elf
```

> **Note** : cet exemple configure HSE bypass → 180 MHz. Sur une NUCLEO-F439ZI sans
> cristal externe câblé, `HAL_RCC_OscConfig()` peut échouer et bloquer dans `Error_Handler()`.
> Pour tester sans HSE, remplacer `SystemClock_Config()` par la version HSI du projet `stm32f4_cubemx`.

---

## Régénérer le code CubeMX (après modification du `.ioc`)

```bash
# 1. Ouvrir le .ioc dans CubeMX
~/STM32CubeMX/STM32CubeMX firmware/stm32f4_cubemx/stm32f4_cubemx.ioc

# 2. Modifier la config (pins, horloges, périphériques)

# 3. Generate Code (le CMakeLists.txt est conservé)

# 4. Rebuild
cmake --build build -j4
```
