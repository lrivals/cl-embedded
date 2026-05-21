# S1811 — Documentation protocole binaire + guide "première session carte"

| Champ | Valeur |
|-------|--------|
| **ID** | S1811 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🟢 Optionnel |
| **Durée estimée** | 3h |
| **Dépendances** | S1801 (protocole v2 firmware implémenté) |
| **Fichiers cibles** | `docs/sprints/sprint_18/S1811_protocol_guide.md` |
| **Statut** | ✅ Ce fichier est le livrable |

---

## Objectif

Fournir un guide complet pour :
1. Comprendre le protocole UART binaire v2
2. Réaliser une première session de streaming avec une NUCLEO-F439ZI
3. Diagnostiquer les erreurs courantes

---

## Partie 1 — Protocole UART v2 : référence complète

### Vue d'ensemble

Le protocole UART v2 est un échange binaire synchrone half-duplex :
- **PC → MCU** : une trame de taille variable (dépend du nombre de features)
- **MCU → PC** : une réponse fixe de **14 octets**

Toutes les valeurs multi-octets sont en **little-endian** (octet de poids faible en premier).

---

### Trame PC → MCU (v2)

```
┌──────────┬─────────┬─────────┬──────────────┬───────────┬──────────────┬───────┬───────┬──────┐
│ MAGIC    │ VERSION │ TASK_ID │ TIMESTAMP_MS  │ N_FEATURES│   FEATURES   │ LABEL │ FLAGS │ CRC8 │
│ 2 octets │ 1 octet │ 1 octet │   4 octets    │  1 octet  │  N × 4 oct.  │1 oct. │1 oct. │1 oct.│
│ 0xABCD   │  0x02   │  0–7    │ uint32 ms     │   ≤ 16    │  f32 × N     │  0/1  │ flags │      │
└──────────┴─────────┴─────────┴──────────────┴───────────┴──────────────┴───────┴───────┴──────┘
```

**Champ FLAGS** :

| Bit | Constante | Effet firmware |
|-----|-----------|---------------|
| 0 | `FLAG_UPDATE = 0x01` | Déclenche `maha_update()` si pas d'anomalie |
| 1 | `FLAG_PROFILING = 0x02` | Inclure métriques DWT dans la réponse |

**CRC8** : calculé sur tout le payload (MAGIC → FLAGS inclus), polynomial 0x07.

Taille totale : `9 + N×4 + 3` octets. Pour N=5 → **32 octets**.

---

### Réponse MCU → PC (v2, 14 octets fixes)

```
┌─────────┬───────────┬──────────────┬──────────────┬──────────────┬────────┐
│  PRED   │ CONFIDENCE│  LATENCY_US  │  RAM_USED_B  │  THROUGHPUT  │ STATUS │
│ 1 octet │  4 octets │   4 octets   │   2 octets   │   2 octets   │1 octet │
│  u8 0/1 │ f32 [0,1] │ u32 µs (DWT) │ u16 octets   │ u16 ips      │  u8    │
└─────────┴───────────┴──────────────┴──────────────┴──────────────┴────────┘
```

**Champ STATUS** :

| Bit | Constante | Signification |
|-----|-----------|--------------|
| 0 | `STATUS_CRC_ERR = 0x01` | CRC de la trame reçue invalide |
| 1 | `STATUS_OOB = 0x02` | Out-of-bounds (N > 16 ou valeurs NaN) |
| 2 | `STATUS_UPDATE_DONE = 0x04` | Mise à jour incrémentale effectuée |

**Parsing Python** :
```python
import struct
RESPONSE_V2_FMT = "<BfIHHB"  # 14 B
pred, conf, lat_us, ram_b, throughput, status = struct.unpack(RESPONSE_V2_FMT, raw_14bytes)
```

---

### Exemple de session annotée

```
PC envoie (N=4, CWRU sample, task_id=0, ts=1250 ms, FLAGS=0x03) :
  CD AB             → MAGIC 0xABCD
  02                → VERSION v2
  00                → TASK_ID = 0
  E2 04 00 00       → TIMESTAMP_MS = 0x04E2 = 1250 ms
  04                → N_FEATURES = 4
  9A 99 19 3F       → feature[0] = 0.6 (f32 LE)
  33 33 33 3F       → feature[1] = 0.7
  00 00 80 3F       → feature[2] = 1.0
  CD CC 4C 3F       → feature[3] = 0.8
  00                → LABEL = 0 (normal)
  03                → FLAGS = UPDATE | PROFILING
  A7                → CRC8

MCU répond (14 B) :
  00                → PRED = 0 (normal)
  00 00 80 3F       → CONF = 1.0 / (1 + score) = 0.82 (f32 LE)
  B2 01 00 00       → LATENCY_US = 0x01B2 = 434 µs
  38 47 00 00       → RAM_USED_B = 0x4738 = 18232 B
  CF 08 00 00       → THROUGHPUT = 0x08CF = 2255 ips
  04                → STATUS = UPDATE_DONE
```

---

## Partie 2 — Checklist "Première session carte"

### Matériel requis

- NUCLEO-F439ZI (STM32F439ZI, Cortex-M4 @ 180 MHz)
- Câble USB-A vers Mini-B (ou Micro-B selon révision)
- PC Linux (Ubuntu 22.04+) avec ce dépôt cloné

### Étape 1 : Vérification connexion UART

```bash
# Lister les ports disponibles
ls /dev/ttyACM* /dev/ttyUSB*
# Attendu : /dev/ttyACM0

# Vérifier les permissions (une fois)
sudo usermod -aG dialout $USER
# (déconnexion/reconnexion nécessaire)

# Test de communication (doit afficher des octets)
python -c "
import serial, time
s = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
s.dtr = True; time.sleep(0.05); s.dtr = False; time.sleep(0.5)
print('Port OK, reçu:', s.read(10))
s.close()
"
```

### Étape 2 : Flash du firmware

```bash
# Depuis la racine du projet
make -C firmware/stm32f4_blink flash
# ou via ST-Link :
# openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program firmware.elf verify reset exit"
```

### Étape 3 : Dry-run de validation (sans board)

```bash
python scripts/sensor_stream.py --dataset cwru --dry-run --n-samples 50 --verbose
# Attendu : 50 lignes "[task=X ts=Yms] label=Z → OK (dry-run)"
```

### Étape 4 : Test avec board (5 samples verbose)

```bash
python scripts/sensor_stream.py \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 5 \
    --verbose
```

Sortie attendue :
```
Chargement dataset 'monitoring'...
  7672 samples, 4 features
[task=0 ts=0ms] true=0 pred=0 conf=0.821 lat=434µs ram=18232B thr=2255/s
[task=0 ts=52ms] true=1 pred=1 conf=0.612 lat=441µs ram=18232B thr=2249/s
...
```

### Étape 5 : Session complète avec dataset builder

```bash
python scripts/board_dataset_builder.py \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --n-samples 500 \
    --n-tasks 3 \
    --update \
    --platform nucleo_f439zi \
    --output experiments/exp_S18_01_board

python scripts/profiling_reader.py \
    --from-csv experiments/exp_S18_01_board/dataset.csv \
    --save experiments/exp_S18_01_board/profiling.json
```

---

## Partie 3 — Troubleshooting

### Problème : Timeout UART (`len(raw) != 14`)

**Symptôme** : `[WARN] Timeout task=0 (0/14 B)` ou données tronquées.

**Causes possibles** :
1. Firmware non flashé ou planté → reset board (bouton NRST)
2. Mauvais port (`/dev/ttyACM1` au lieu de `/dev/ttyACM0`) → `ls /dev/ttyACM*`
3. Baud rate mismatch → vérifier `USART3` config dans `main.c` (115200 N81)
4. Firmware encore en v1 (réponse 9 B au lieu de 14 B) → reflasher avec firmware v2

### Problème : Erreurs CRC (`STATUS_CRC_ERR = 0x01`)

**Symptôme** : `status & 0x01 != 0`, `crc_errors > 0` dans les stats.

**Causes possibles** :
1. Câble USB défectueux ou trop long → remplacer
2. Interférence électrique → éloigner des sources de bruit
3. Bug dans l'implémentation CRC firmware → vérifier `proto_crc8()` vs `crc8()` Python

**Vérification rapide** :
```python
from scripts.sensor_stream import crc8, build_frame_v2
import numpy as np
frame = build_frame_v2(np.zeros(4, dtype=np.float32), 0, 0, 0)
assert crc8(frame[:-1]) == frame[-1]
print("CRC Python OK")
```

### Problème : Trame ignorée par firmware (`resync` en boucle)

**Symptôme** : Aucune réponse du firmware.

**Causes possibles** :
1. Firmware en v1 (pas de parsing VERSION) → reflasher v2
2. MAGIC incorrect dans la trame → vérifier `MAGIC = 0xABCD` dans `sensor_stream.py`
3. N_FEATURES > 16 → vérifier `--dataset` supporte ≤ 16 features (CWRU=9, Monitoring=4 ✅)

### Problème : `ram_used_b = 0` dans la réponse

**Symptôme** : `profiling.json` → `ram_mean_bytes = 0`.

**Cause** : `profiling_init()` non appelé dans `main()` avant la boucle principale, ou symboles linker `_sbss`/`_ebss` non définis dans le linker script.

**Correction** : Vérifier que `profiling_init()` est bien appelé dans `main()` après `pipeline_init()`.

---

## Partie 4 — Interopérabilité protocoles

| Version | Envoyé par | Reçu par | Taille réponse |
|---------|-----------|---------|---------------|
| v1 | `sensor_sim.py` (Sprint 16) | firmware v1 (`pipeline.c` actuel) | 9 B |
| v2 | `sensor_stream.py` (Sprint 18) | firmware v2 (après S1801) | 14 B |

> `sensor_sim.py` et le firmware v2 sont **incompatibles** : utiliser uniquement `sensor_stream.py` avec un firmware flashé v2.
