# Guide Renode — Simulation STM32F4 sur PC

Renode émule le processeur ARM Cortex-M4 + périphériques STM32F4 sur PC x86,
permettant de valider le firmware `stm32f4_blink.elf` **sans hardware physique**.

> **Références spec** : S1704 (sprint 17) — dépendance : S1001 (firmware compilé).

---

## Installation

```bash
# Ubuntu/Debian (si disponible dans apt)
sudo apt-get update && sudo apt-get install -y renode

# Sinon : .deb depuis GitHub Releases
# https://github.com/renode/renode/releases
wget https://github.com/renode/renode/releases/download/v1.15.3/renode_1.15.3_amd64.deb
sudo dpkg -i renode_1.15.3_amd64.deb

# Vérifier
renode --version  # attendu : ≥ 1.14
```

---

## Workflow rapide : make → .elf → renode → assert

```bash
# 1. Compiler le firmware (DEBUG_PRINTF=1 déjà activé dans Makefile)
make -C firmware/stm32f4_blink/ all

# 2. Lancer la simulation de validation complète
bash firmware/renode/run_mahalanobis_sim.sh
# Attendu : "PASS: simulation Mahalanobis validée" + exit 0

# 3. Mode interactif (pour explorer)
renode firmware/renode/nucleo_f439zi.resc
```

---

## Cheat sheet commandes Renode

### Navigation de base

| Commande | Description |
|----------|-------------|
| `help` | Liste toutes les commandes disponibles |
| `machine` | Affiche les machines créées |
| `peripherals` | Liste les périphériques de la machine courante |
| `start` | Démarre la simulation |
| `pause` | Met en pause |
| `quit` | Quitte Renode |

### Firmware et mémoire

| Commande | Description |
|----------|-------------|
| `sysbus LoadELF @chemin/firmware.elf` | Charge un binaire ELF |
| `sysbus LoadBinary @chemin/firmware.bin 0x08000000` | Charge un `.bin` à une adresse |
| `sysbus ReadByte 0x20000000` | Lit 1 octet en RAM |
| `sysbus ReadDoubleWord 0x40004800` | Lit 4 octets (registre USART3 DR) |

### UART et périphériques

| Commande | Description |
|----------|-------------|
| `showAnalyzer sysbus.usart3` | Ouvre terminal UART interactif |
| `sysbus.usart3 WriteChar 0xCD` | Injecte un octet dans USART3 |
| `logLevel -1 sysbus.usart3` | Active logs verbeux UART |
| `sysbus.usart3 Log` | Active logging des transactions UART |

### GPIO et LED

| Commande | Description |
|----------|-------------|
| `sysbus ReadDoubleWord 0x40020010` | Lire GPIOA IDR (bit 5 = PA5 état entrée) |
| `sysbus ReadDoubleWord 0x40020014` | Lire GPIOA ODR (bit 5 = PA5 état sortie) |
| `logLevel -1 sysbus.gpioPortA` | Logs verbeux du port GPIOA (accès registres) |

> **Note** : `sysbus.gpioPortA.5 Log` n'est pas supporté par `STM32_GPIOPort`.
> Lire ODR/IDR directement est plus fiable pour observer l'état de PA5.

### Débogage

| Commande | Description |
|----------|-------------|
| `machine StartGdbServer 3333` | Ouvre serveur GDB sur port 3333 |
| `machine EnableProfiler @prof.csv` | Profiling cycles CPU |
| `logFile @/tmp/renode.log true` | Redirige tous les logs vers fichier |
| `cpu LogFunctionNames true true` | Log les appels de fonctions (nécessite DWARF) |

### Cycle counter DWT

```
# Lire DWT CYCCNT (même adresse que le firmware : 0xE0001004)
sysbus ReadDoubleWord 0xE0001004
```

> **Précision DWT** : Renode simule le cycle counter avec une précision de ±5%.
> Les mesures de latence en simulation ne sont pas représentatives du hardware réel.

---

## Débogage avec GDB

```bash
# Terminal 1 : lancer Renode + démarrer GDB server
renode firmware/renode/nucleo_f439zi.resc
(monitor) machine StartGdbServer 3333

# Terminal 2 : connecter GDB
arm-none-eabi-gdb firmware/stm32f4_blink/build/stm32f4_blink.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) break pipeline_run
(gdb) continue
(gdb) print g_detector.threshold
(gdb) print score
```

---

## Test GPIO blink (S17-15)

```
# Dans la console Renode interactive après chargement du .resc :
(monitor) sysbus.gpioPortA.5 Log
(monitor) start

# Vérifier que "PA5 state: True → False → True" alterne dans les logs
# LED LD2 = PA5 sur NUCLEO-F439ZI
# Toggle toutes les secondes dans le firmware (pendant les attentes UART)
```

---

## Injection manuelle de trame UART (sans socket)

Si `uarthub BindSocket` n'est pas disponible dans votre version Renode,
injectez la trame de test directement dans le `.resc` :

```
# Trame test : MAGIC=0xABCD, N=5, features=[0.1,0.2,0.3,0.4,0.5], label=0
# Encodage little-endian FP32 (0.1f = 0xCD CC CC 3D)

start
sleep 1    # attendre pipeline_init

# MAGIC 0xABCD (little-endian : 0xCD 0xAB)
sysbus.usart3 WriteChar 0xCD
sysbus.usart3 WriteChar 0xAB

# N = 5
sysbus.usart3 WriteChar 0x05

# 0.1f = CD CC CC 3D
sysbus.usart3 WriteChar 0xCD
sysbus.usart3 WriteChar 0xCC
sysbus.usart3 WriteChar 0xCC
sysbus.usart3 WriteChar 0x3D

# 0.2f = CD CC 4C 3E
sysbus.usart3 WriteChar 0xCD
sysbus.usart3 WriteChar 0xCC
sysbus.usart3 WriteChar 0x4C
sysbus.usart3 WriteChar 0x3E

# 0.3f = 9A 99 99 3E
sysbus.usart3 WriteChar 0x9A
sysbus.usart3 WriteChar 0x99
sysbus.usart3 WriteChar 0x99
sysbus.usart3 WriteChar 0x3E

# 0.4f = CD CC CC 3E
sysbus.usart3 WriteChar 0xCD
sysbus.usart3 WriteChar 0xCC
sysbus.usart3 WriteChar 0xCC
sysbus.usart3 WriteChar 0x3E

# 0.5f = 00 00 00 3F
sysbus.usart3 WriteChar 0x00
sysbus.usart3 WriteChar 0x00
sysbus.usart3 WriteChar 0x00
sysbus.usart3 WriteChar 0x3F

# label = 0
sysbus.usart3 WriteChar 0x00

# CRC8 de la trame (calculé par send_test_frame.py : python3 -c "from firmware.renode.send_test_frame import build_frame; print(build_frame([0.1,0.2,0.3,0.4,0.5], 0).hex())")
# Injecter l'octet CRC ici
sysbus.usart3 WriteChar 0xXX  # à remplacer par la valeur calculée
```

---

## Limitations connues

| Limitation | Impact |
|-----------|--------|
| Vitesse : 10–100× plus lent que hardware | Timeouts à ajuster en CI |
| DWT cycle counter ≈ précis à ±5% | Mesures latence non représentatives |
| DMA non émulé | HAL DMA non disponible |
| STM32N6 (Cortex-M55 + NPU) non supporté | `TODO(dorra)` : quel simulateur pour cible finale ? |
| USART3 BindSocket : API dépendante de la version Renode | Tester avec `renode --version` ≥ 1.14 |

---

## Intégration CI (GitHub Actions)

Voir `.github/workflows/firmware.yml` — job `renode-sim` :

```bash
# Reproduire localement ce que fait la CI
sudo apt-get install -y renode gcc-arm-none-eabi
make -C firmware/stm32f4_blink/ all
bash firmware/renode/run_mahalanobis_sim.sh
```

---

## Questions ouvertes

- `TODO(dorra)` : Renode supporte-t-il STM32N6 (Cortex-M55 + NPU) ? Si non, simulateur alternatif ?
- `TODO(arnaud)` : le workflow Renode CI est-il suffisant pour le rapport, ou faut-il des mesures hardware réel ?
