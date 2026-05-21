# S1802 — Script PC streaming continu : rate-limité, séquences multi-tâches, dry-run

| Champ | Valeur |
|-------|--------|
| **ID** | S1802 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S1801 (protocole v2 firmware) |
| **Fichiers cibles** | `scripts/sensor_stream.py` |
| **Statut** | ✅ Implémenté — à valider |

---

## Objectif

Fournir un script Python capable de streamer un dataset Phase 1 (CWRU ou Monitoring) vers le firmware STM32 via UART protocole v2, avec :
- découpage en tâches CL multi-tâches (séquences continues)
- rate-limiting configurable (Hz)
- mode dry-run sans board pour validation pipeline CI
- collecte des réponses firmware (14 B) et calcul de statistiques

---

## Fichier : `scripts/sensor_stream.py`

### Constantes protocole v2

```python
PROTO_VERSION     = 0x02
MAGIC             = 0xABCD
UART_TIMEOUT_S    = 2.0

FRAME_FMT_HDR     = "<HBBIB"   # magic(u16), version(u8), task_id(u8), ts_ms(u32), n(u8)

FRAME_FLAGS_UPDATE    = 0x01   # bit0 : demande mise à jour incrémentale
FRAME_FLAGS_PROFILING = 0x02   # bit1 : demande métriques profiling

RESPONSE_V2_FMT   = "<BfIHHB"  # pred(u8), conf(f32), lat_us(u32), ram(u16), thr(u16), status(u8)
RESPONSE_V2_SIZE  = 14          # octets

STATUS_OK          = 0x00
STATUS_CRC_ERR     = 0x01
STATUS_OOB         = 0x02
STATUS_UPDATE_DONE = 0x04
```

### CRC8 (polynomial 0x07)

```python
def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc
```

Identique à l'implémentation C dans `pipeline.c` : `proto_crc8()`.

### Construction de trame v2

```python
def build_frame_v2(features: np.ndarray, label: int, task_id: int,
                   ts_ms: int, flags: int = 0) -> bytes:
```

Produit une trame de taille `9 + N×4 + 2 + 1` octets (header 9B + features + label+flags + CRC).

### Séquencement multi-tâches

```python
def _make_task_splits(X, y, n_tasks) -> list[tuple[np.ndarray, np.ndarray]]:
```

Découpe le dataset en `n_tasks` tranches temporelles égales. Chaque tranche = une tâche CL. La tâche 0 est envoyée en premier, puis tâche 1, etc.

---

## Diagramme de séquence PC ↔ MCU

```
PC                                      MCU (pipeline_run)
│                                            │
│  load_dataset(cwru / monitoring)           │
│  _make_task_splits(X, y, n_tasks=3)        │
│                                            │
│  ── build_frame_v2(feat, label,            │
│         task_id=0, ts_ms, FLAGS) ────────> │
│                                            │  uart_receive_sample()
│                                            │  profiling_start()
│                                            │  normalize_zscore()
│                                            │  maha_score()
│                                            │  maha_update() si FLAG_UPDATE
│                                            │  profiling_stop()
│  <──────── response v2 (14 B) ────────────│
│  unpack: pred, conf, lat_us,               │
│          ram_b, throughput, status         │
│                                            │
│  sleep(1/rate_hz - elapsed)  [optionnel]  │
│  ... (sample suivant) ...                  │
```

---

## Modes de fonctionnement

### Mode dry-run (sans board)

```bash
python scripts/sensor_stream.py --dataset cwru --dry-run --n-samples 200
```

- Simule la réponse firmware : `pred=label`, `conf=1.0`, `lat_us=3`, `ram=200`
- CRC calculé et vérifié en boucle locale
- Aucune dépendance `pyserial`

### Mode UART (avec board)

```bash
python scripts/sensor_stream.py \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 100 \
    --rate-hz 10 \
    --update \
    --output experiments/exp_S18_01/stream.json
```

- `--rate-hz 10` : max 10 trames/s (sleep adaptatif)
- `--update` : active `FLAG_UPDATE` → firmware met à jour le modèle en ligne
- `--output` : sauve les statistiques agrégées en JSON

### Statistiques produites

```json
{
  "n_samples": 200,
  "n_tasks": 3,
  "accuracy": 0.9150,
  "latency_mean_us": 450.2,
  "latency_p50_us": 440.0,
  "latency_p99_us": 1200.0,
  "ram_mean_bytes": 18200,
  "throughput_mean_ips": 2222,
  "crc_errors": 0,
  "mode": "uart",
  "port": "/dev/ttyACM0"
}
```

---

## Arguments CLI

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--dataset` | `cwru` \| `monitoring` | — | Dataset Phase 1 à streamer |
| `--dry-run` | flag | False | Sans board, simulation locale |
| `--port` | str | `/dev/ttyACM0` | Port série UART |
| `--baud` | int | `115200` | Baud rate |
| `--n-samples` | int | `200` | Nombre total de samples |
| `--n-tasks` | int | `3` | Nombre de tâches CL simulées |
| `--rate-hz` | float | `0.0` | Rate-limit (0 = max speed) |
| `--update` | flag | False | Demande update incrémental au firmware |
| `--output` | str | None | Chemin JSON pour les statistiques |
| `--verbose` | flag | False | Affiche chaque échange |

---

## Dépendances

```
numpy           # arrays, stats
pyserial        # UART (requis uniquement hors dry-run)
```

---

## Critères d'acceptation

- [ ] `python scripts/sensor_stream.py --dataset cwru --dry-run --n-samples 200` termine sans erreur
- [ ] Sortie JSON contient tous les champs attendus (n_samples, accuracy, latency_*, crc_errors)
- [ ] CRC `crc_errors = 0` en dry-run (self-consistent)
- [ ] `--n-tasks 3` → exactement 3 `task_id` distincts dans les résultats
- [ ] Rate-limit `--rate-hz 1` → durée ≈ n_samples secondes (±10%)
