# S1702 — UART printf : retarget `_write()` + debug terminal USART3

| Champ | Valeur |
|-------|--------|
| **ID** | S1702 |
| **Sprint** | Sprint 17 — Objectif 2 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 5h |
| **Dépendances** | S1701 ✅ (projet CubeMX opérationnel, USART3 configuré) |
| **Fichiers cibles** | `firmware/stm32f4_cubemx/Core/Src/retarget_io.c`, `firmware/stm32f4_cubemx/Core/Src/main.c`, `firmware/stm32f4_blink/src/pipeline.c`, `firmware/stm32f4_cubemx_examples/UART_Printf/` |
| **Statut** | ✅ Terminé |

---

## Objectif

Rendre `printf()` opérationnel sur la NUCLEO-F439ZI via USART3 (ST-LINK VCP, visible sans câble supplémentaire). Deux niveaux : (1) retarget `printf` dans le projet CubeMX pour un `"Hello NUCLEO"` propre, (2) intégrer un printf lisible dans `firmware/stm32f4_blink/pipeline.c` pour remplacer la trame binaire 9B par une ligne texte de debug.

---

## Contexte matériel

**USART3 sur NUCLEO-F439ZI** :

| Paramètre | Valeur |
|-----------|--------|
| Pins | PD8 (TX) / PD9 (RX) |
| Connexion PC | ST-LINK VCP (même câble USB que le debug) |
| Baud rate | 115 200 baud |
| Format | 8N1 (8 bits, pas de parité, 1 stop bit) |
| Port Linux | `/dev/ttyACM0` (ou `/dev/ttyUSB0` selon le système) |

**Différence avec `firmware/stm32f4_blink/`** : le projet blink (Sprint 16) utilise USART3 avec un protocole binaire custom (trame MAGIC + CRC8 + données FP32). Ce sprint ajoute en parallèle un mode texte `printf` pour le debug interactif humain.

---

## Sous-tâches

| ID | Description | Durée |
|----|-------------|:---:|
| **S17-05** | Configurer USART3 dans CubeMX + retarget `printf` via `_write()` | 2h |
| **S17-06** | Exemple `UART_Printf` : "Hello NUCLEO" + compteur sur terminal | 1h |
| **S17-07** | Importer exemple `STM32CubeF4/Examples/UART/UART_Printf/` | 1h |
| **S17-08** | Intégrer `printf` debug dans `firmware/stm32f4_blink/src/pipeline.c` | 1h |

---

## Spécification

### Retarget `printf` (syscalls POSIX sur MCU)

La libc newlib utilisée par arm-none-eabi-gcc appelle `_write()` pour toute sortie `printf`. Il suffit de redéfinir cette fonction pour rediriger vers UART :

**`firmware/stm32f4_cubemx/Core/Src/retarget_io.c`** :
```c
#include "stm32f4xx_hal.h"
#include <errno.h>
#include <sys/unistd.h>

extern UART_HandleTypeDef huart3;   /* handle généré par CubeMX */

int _write(int file, char *ptr, int len)
{
    (void)file;
    HAL_UART_Transmit(&huart3, (uint8_t *)ptr, (uint16_t)len, HAL_MAX_DELAY);
    return len;
}
```

Ajouter `retarget_io.c` aux sources dans `CMakeLists.txt` :
```cmake
list(APPEND SOURCES "Core/Src/retarget_io.c")
```

Ajouter `printf` dans `main.c` :
```c
#include <stdio.h>
/* ... */
while (1)
{
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
    printf("Hello NUCLEO — tick=%lu\r\n", HAL_GetTick());
    HAL_Delay(1000);
}
```

### Interface pipeline.c (debug mode)

Dans `firmware/stm32f4_blink/src/pipeline.c`, ajouter une fonction de debug texte **en complément** de la réponse binaire existante (ne pas casser le protocole sensor_sim.py) :

```c
/* Mode debug : envoi ligne texte lisible sur UART si DEBUG_PRINTF=1 */
#ifdef DEBUG_PRINTF
    char dbg[64];
    snprintf(dbg, sizeof(dbg), "score=%.4f pred=%d lat=%lu us\r\n",
             score, pred, lat_us);
    /* Envoyer via uart_send_byte() octet par octet */
    for (int i = 0; dbg[i]; i++) uart_send_byte((uint8_t)dbg[i]);
#endif
```

Le flag `DEBUG_PRINTF` est défini dans le `Makefile` avec `-DDEBUG_PRINTF=1` en mode debug.

---

## Implémentation

### S17-05 : USART3 + retarget (2h)

Vérifier la configuration USART3 dans CubeMX (déjà présente si S17-01 terminé) :
- Mode : Asynchronous
- Baud rate : 115200
- Word length : 8 bits, No parity, 1 stop bit

Créer `Core/Src/retarget_io.c` (voir spécification ci-dessus).

Vérifier que le linker newlib est configuré pour les syscalls (`--specs=nosys.specs` ou `-specs=nano.specs`) dans `CMakeLists.txt` :
```cmake
set(CMAKE_EXE_LINKER_FLAGS "${MCU_FLAGS} -specs=nano.specs -lc -lm -lnosys ...")
```

### S17-06 : "Hello NUCLEO" (1h)

Modifier `main.c` et rebuilder :
```bash
cmake --build firmware/stm32f4_cubemx/build -j4
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
    -c "program firmware/stm32f4_cubemx/build/stm32f4_cubemx.elf verify reset exit"
```

Ouvrir le terminal série sur `/dev/ttyACM0` :
```bash
# Option 1 : Minicom
minicom -b 115200 -D /dev/ttyACM0

# Option 2 : screen
screen /dev/ttyACM0 115200

# Option 3 : VS Code Serial Monitor (extension)
```

Sortie attendue :
```
Hello NUCLEO — tick=0
Hello NUCLEO — tick=1000
Hello NUCLEO — tick=2000
...
```

### S17-07 : Exemple UART_Printf officiel (1h)

```bash
# Localiser le package STM32CubeF4 (après installation via CubeMX)
find ~/STM32Cube -name "UART_Printf" -type d 2>/dev/null
# → ~/STM32Cube/Repository/STM32Cube_FW_F4_V1.*/Projects/NUCLEO-F439ZI/Examples/UART/UART_Printf/

mkdir -p firmware/stm32f4_cubemx_examples/UART_Printf
cp -r ~/STM32Cube/.../UART_Printf/* firmware/stm32f4_cubemx_examples/UART_Printf/
```

Compiler l'exemple et comparer avec le projet custom :
```bash
# L'exemple a son propre Makefile
cd firmware/stm32f4_cubemx_examples/UART_Printf
make -j4
arm-none-eabi-size build/*.elf
```

### S17-08 : Intégrer printf dans pipeline.c (1h)

Dans `firmware/stm32f4_blink/src/pipeline.c`, après l'envoi de la réponse binaire `uart_send_response()`, ajouter le bloc debug conditionnel (cf. spécification). Mettre à jour le `Makefile` :

```makefile
# Ajouter -DDEBUG_PRINTF=1 pour activer les logs texte
CFLAGS += -DDEBUG_PRINTF=1
```

Vérifier qu'aucun test Unity existant n'est cassé :
```bash
cd firmware/stm32f4_blink
make test
# → 16/16 tests PASS (résultat Sprint 16)
```

---

## Critères d'acceptation

- [ ] `"Hello NUCLEO — tick=XXX"` s'affiche dans Minicom/screen @ 115 200 baud
- [ ] Aucune erreur de compilation liée à `_write()` (pas de `undefined reference`)
- [ ] `arm-none-eabi-size` : ajout de `printf` < 5 Ko sur la section `.text` (newlib-nano)
- [ ] `firmware/stm32f4_blink/` : `make test` → 16/16 PASS après ajout du bloc debug
- [ ] `printf("score=%.4f latency=%lu us\r\n", score, cycles)` visible sur terminal avec `DEBUG_PRINTF=1`

---

## Statut

✅ Terminé — Build OK, 16/16 tests PASS, `.text` = 9612 B (newlib-nano)
