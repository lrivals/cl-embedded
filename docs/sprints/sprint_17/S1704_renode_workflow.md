# S1704 — Renode : simulation PC NUCLEO-F439ZI sans hardware

| Champ | Valeur |
|-------|--------|
| **ID** | S1704 |
| **Sprint** | Sprint 17 — Objectif 4 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 6.5h |
| **Dépendances** | S1001 ✅ (`firmware/stm32f4_blink/*.elf` compilé) |
| **Fichiers cibles** | `firmware/renode/nucleo_f439zi.resc`, `firmware/renode/run_mahalanobis_sim.sh` |
| **Statut** | ✅ Terminé |

---

## Objectif

Installer Renode et créer un workflow de simulation PC qui charge le firmware `stm32f4_blink.elf` (Mahalanobis pipeline), simule les trames UART d'entrée, et vérifie que le score et la latence sont corrects — sans avoir accès à la board physique. Ce workflow alimente ensuite la CI GitHub Actions (S17-18).

> **Pourquoi Renode** : STM32Cube.AI est bloqué (`TODO(dorra)`) et l'accès au STM32N6 réel est incertain. Renode permet de valider le firmware embarqué sur PC et de créer des tests automatisés reproductibles, y compris pour la future cible STM32N6.

---

## Contexte

**Renode** émule un processeur ARM Cortex-M4 + périphériques STM32F4 (UART, GPIO, SysTick, DWT cycle counter). Il exécute le binaire `.elf` compilé pour ARM directement sur PC x86.

| Caractéristique | Valeur |
|----------------|--------|
| Version cible | ≥ 1.14 |
| Plateforme émulée | STM32F4 Discovery (proche NUCLEO-F439ZI) |
| UART simulé | UART2 ou UART3 selon le `.repl` |
| DWT émulé | Oui (cycle counter) |
| Vitesse simulation | ~10–100× plus lent que hardware réel |

**Firmware à valider** : `firmware/stm32f4_blink/build/stm32f4_blink.elf`  
Ce firmware lit des trames UART (`[MAGIC][N][features FP32][label][CRC8]`) et répond `[pred][confidence][latency_us]`.

---

## Sous-tâches

| ID | Description | Durée |
|----|-------------|:---:|
| **S17-12** | Installer Renode + vérifier `renode --version` | 0.5h |
| **S17-13** | Script `.resc` NUCLEO-F439ZI + chargement `.elf` | 2h |
| **S17-14** | Validation Mahalanobis : stimuler UART + assert score | 2h |
| **S17-15** | Test GPIO blink dans Renode (LED toggle observable) | 1h |
| **S17-16** | Doc : commandes essentielles + workflow CI | 1h |

---

## Spécification

### Installation Renode

```bash
# Option 1 : package Debian/Ubuntu
sudo apt-get install renode

# Option 2 : .deb depuis GitHub Releases (version précise)
# Télécharger renode_*.deb depuis la page releases de Renode
sudo dpkg -i renode_*.deb

# Vérifier
renode --version
```

### Script Renode `.resc` (Robot Execution Script)

**`firmware/renode/nucleo_f439zi.resc`** :

```
# Renode script pour NUCLEO-F439ZI (STM32F4 Discovery compatible)

# Créer la machine
mach create "nucleo-f439zi"

# Charger la description matérielle STM32F4 (fournie avec Renode)
machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl

# Charger le firmware
sysbus LoadELF @firmware/stm32f4_blink/build/stm32f4_blink.elf

# Configurer l'UART (UART2 dans le .repl STM32F4 Discovery)
showAnalyzer sysbus.uart2

# Démarrer la simulation
start
```

> **Note** : Le `.repl` de la STM32F4 Discovery est le plus proche disponible dans Renode pour émuler un STM32F4xx. UART2 (PA2/PA3) est câblé dans ce `.repl` — si le firmware utilise USART3 (PD8/PD9), il faudra adapter ou créer un `.repl` custom.

### Script bash validation Mahalanobis

**`firmware/renode/run_mahalanobis_sim.sh`** :

```bash
#!/bin/bash
set -e

ELF="firmware/stm32f4_blink/build/stm32f4_blink.elf"
RESC="firmware/renode/nucleo_f439zi.resc"
LOG=$(mktemp /tmp/renode_out.XXXXXX)

# Vérifier que le firmware existe
[ -f "$ELF" ] || { echo "ERROR: $ELF not found. Run 'make -C firmware/stm32f4_blink'"; exit 1; }

# Lancer Renode en mode headless
timeout 30 renode --console --disable-xwt "$RESC" > "$LOG" 2>&1 &
RENODE_PID=$!

# Attendre la stabilisation
sleep 5

# Envoyer une trame de test via le socket UART Renode
# Trame test : MAGIC=0xABCD, N=5, features=[0.1,0.2,0.3,0.4,0.5], label=0, CRC8
python3 firmware/renode/send_test_frame.py --port 3456

# Vérifier la réponse dans les logs
sleep 2
grep -q "score=" "$LOG" && echo "PASS: score visible dans logs Renode" || echo "FAIL: score absent"

kill $RENODE_PID 2>/dev/null
rm "$LOG"
```

### Script Python de stimulation UART

**`firmware/renode/send_test_frame.py`** :

```python
import socket
import struct
import argparse

def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc

def build_frame(features: list[float], label: int) -> bytes:
    n = len(features)
    payload = struct.pack('<HB', 0xABCD, n)
    payload += struct.pack(f'<{n}f', *features)
    payload += struct.pack('<B', label)
    payload += struct.pack('<B', crc8(payload))
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3456)
    args = parser.parse_args()

    features = [0.1, 0.2, 0.3, 0.4, 0.5]   # 5 features normales
    frame = build_frame(features, label=0)

    with socket.socket() as s:
        s.connect(('localhost', args.port))
        s.sendall(frame)
        response = s.recv(9)
        pred, conf, lat_us = struct.unpack('<Bfi', response[:9])
        print(f"pred={pred} conf={conf:.4f} lat_us={lat_us}")
```

---

## Implémentation

### S17-12 : Installer Renode (0.5h)

```bash
sudo apt-get update && sudo apt-get install -y renode
renode --version
# Attendu : Renode v1.14.x ou supérieur
```

Si le package n'est pas disponible dans apt :
```bash
# Chercher la dernière release .deb sur github.com/renode/renode/releases
wget https://github.com/renode/renode/releases/download/vX.Y.Z/renode_X.Y.Z_amd64.deb
sudo dpkg -i renode_X.Y.Z_amd64.deb
```

### S17-13 : Script `.resc` + chargement ELF (2h)

```bash
mkdir -p firmware/renode

# Créer nucleo_f439zi.resc (voir spécification)

# Tester en mode interactif
renode firmware/renode/nucleo_f439zi.resc
# Dans la console Renode :
(monitor) sysbus LoadELF @firmware/stm32f4_blink/build/stm32f4_blink.elf
(monitor) start
(monitor) sysbus.uart2 WaitForLine "ready" 5
```

**Adapter le .repl si nécessaire** : si Renode signale que USART3 n'est pas émulé dans le `.repl` STM32F4 Discovery, créer un `.repl` custom qui réassigne UART aux pins PD8/PD9 :

```
# firmware/renode/stm32f439zi_custom.repl
# Basé sur platforms/cpus/stm32f429.repl
# Ajouter mapping USART3 → virtual socket port 3456
```

### S17-14 : Validation Mahalanobis (2h)

```bash
chmod +x firmware/renode/run_mahalanobis_sim.sh
./firmware/renode/run_mahalanobis_sim.sh
# Attendu : "PASS: score visible dans logs Renode"
```

Si le protocole binaire de `pipeline.c` est difficile à lire dans les logs Renode, activer `DEBUG_PRINTF=1` (S17-08) pour avoir une sortie texte plus lisible :

```
score=0.1234 pred=0 lat=3 us
```

### S17-15 : Test GPIO blink dans Renode (1h)

```
# Dans la console Renode interactive
(monitor) machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl
(monitor) sysbus LoadELF @firmware/stm32f4_blink/build/stm32f4_blink.elf
(monitor) sysbus.gpioPortA.5 Log    # Observer les toggles sur PA5
(monitor) start
# Vérifier que "PA5 state: True/False" alterne dans les logs
```

### S17-16 : Documentation workflow Renode (1h)

Créer `docs/embedded_renode_guide.md` avec :
- Commandes Renode essentielles (cheat sheet)
- Workflow `make → .elf → renode → assert`
- Comment débugger : `machine StartGdbServer 3333` + `arm-none-eabi-gdb`
- Limitations connues : DWT cycle counter précision, périphériques non émulés

---

## Critères d'acceptation

- [ ] `renode --version` ≥ 1.14 sans erreur
- [ ] `firmware/renode/nucleo_f439zi.resc` charge `stm32f4_blink.elf` sans crash
- [ ] `run_mahalanobis_sim.sh` retourne exit code 0 + "PASS"
- [ ] Score Mahalanobis visible dans les logs Renode (format `score=X.XXXX`)
- [ ] Toggle GPIO PA5 observable dans la console Renode (blink test)

---

## Questions ouvertes

- `TODO(dorra)` : Renode supporte-t-il le STM32N6 (Cortex-M55 + NPU) ? Si non, quel simulateur pour la cible finale ?
- `TODO(arnaud)` : le workflow Renode CI est-il suffisant pour valider les critères du rapport de stage, ou faut-il aussi des mesures sur hardware réel ?

---

## Statut

✅ Terminé — Renode v1.16.1 installé et validé sur PC.

### Résultats de validation (2026-05-21)

- `renode --version` → `Renode v1.16.1` ✅
- `nucleo_f439zi.resc` charge `stm32f4_blink.elf` sans crash ✅
- `run_mahalanobis_sim.sh` → exit code 0 + "PASS" ✅
- Score Mahalanobis visible : `score=0.7416 pred=0 lat=0 us` ✅
- GPIO PA5 : state observable via `sysbus ReadDoubleWord 0x40020014` (ODR) ✅

### Divergences par rapport à la spec initiale

| Point | Spec | Réel |
|-------|------|------|
| Platform description | Custom `.repl` avec redéfinition USART3 | USART3 déjà dans `stm32f4.repl` — `.repl` custom minimal |
| Socket UART | `UARTHub BindSocket` | `emulation CreateServerSocketTerminal` (Renode 1.16.1) |
| GPIO Log | `sysbus.gpioPortA.5 Log` | Non supporté — utiliser `ReadDoubleWord 0x40020014` |
| `-specs=nano.specs` + `%.4f` | fonctionnel | Nécessite `-u _printf_float` + stubs `_write`/`_exit` |
| Renode dans apt | `apt install renode` | Non disponible — `.deb` GitHub Releases |
