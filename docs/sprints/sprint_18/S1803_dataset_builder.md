# S1803 — Dataset builder : collecte réponses → CSV/HDF5, métadonnées exp auto

| Champ | Valeur |
|-------|--------|
| **ID** | S1803 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S1802 (`sensor_stream.py` opérationnel) |
| **Fichiers cibles** | `scripts/board_dataset_builder.py` |
| **Statut** | ✅ Implémenté — à valider |

---

## Objectif

Orchestrer `sensor_stream.py` pour collecter les réponses firmware et produire un répertoire d'expérience `experiments/exp_S18_XX/` au format **unifié Phase 1**, compatible avec `evaluate_all.py`.

---

## Pipeline interne

```
board_dataset_builder.py
│
├── _run_stream()          ← appelle sensor_stream._stream_dry_run() ou _stream_uart()
│       │
│       └── list[dict]     ← résultats bruts (task_id, ts_ms, true, pred, conf, lat, ram, thr, status)
│
├── _save_csv()            → experiments/exp_S18_XX/dataset.csv
├── _compute_results_json() → experiments/exp_S18_XX/results.json
└── _save_config_snapshot() → experiments/exp_S18_XX/config_snapshot.yaml
```

---

## Format de sortie : `experiments/exp_S18_XX/`

### `dataset.csv`

Une ligne par sample :

| Colonne | Type | Description |
|---------|------|-------------|
| `task_id` | int | Tâche CL (0, 1, 2…) |
| `ts_ms` | int | Timestamp depuis début session |
| `true` | int | Label ground truth |
| `pred` | int | Prédiction firmware |
| `confidence` | float | Score de confiance |
| `latency_us` | int | Latence DWT en µs |
| `ram_bytes` | int | RAM .bss estimée |
| `throughput_ips` | int | Inférences/s |
| `status` | int | Code statut (0=OK) |

### `results.json`

Contient les **6 champs obligatoires Phase 1** (compatibles `evaluate_all.py`) :

```json
{
  "exp_id": "exp_S18_01",
  "model": "streaming_pipeline",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-05-25",
  "acc_final": 0.9200,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": 19456,
  "inference_latency_ms": 0.4502,
  "n_params": null,
  "n_samples": 500,
  "n_tasks": 3,
  "latency_p99_ms": 1.200,
  "throughput_mean_ips": 2222,
  "config_snapshot": "experiments/exp_S18_01/config_snapshot.yaml",
  "collection_time_s": 12.4
}
```

> `avg_forgetting`, `backward_transfer`, `n_params` sont renseignés ultérieurement par `board_experiment_recorder.py`.

### `config_snapshot.yaml`

Snapshot de la configuration utilisée à l'exécution :

```yaml
dataset: cwru
n_samples: 500
n_tasks: 3
rate_hz: 0.0
update_requested: false
dry_run: true
port: null
baud: 115200
platform: nucleo_f439zi
date: '2026-05-25T14:32:11.123456'
protocol_version: 2
```

---

## Plateformes supportées

| Valeur `--platform` | Description |
|---------------------|-------------|
| `nucleo_f439zi` | NUCLEO-F439ZI, Cortex-M4 @ 180 MHz, 192 Ko SRAM (Phase 2 actuelle) |
| `stm32n6_eval` | Kit éval STM32N6, Cortex-M55 + NPU (cible finale) |
| `edge_spectrum` | Hardware propriétaire Edge Spectrum (Phase 3) |

---

## Usage CLI

```bash
# Dry-run — critère de succès du sprint
python scripts/board_dataset_builder.py \
    --dataset cwru \
    --dry-run \
    --n-samples 500 \
    --output experiments/exp_S18_01

# Avec board NUCLEO
python scripts/board_dataset_builder.py \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --n-samples 200 \
    --n-tasks 3 \
    --update \
    --platform nucleo_f439zi \
    --output experiments/exp_S18_02
```

---

## Arguments CLI

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--dataset` | `cwru` \| `monitoring` | — | Dataset Phase 1 |
| `--dry-run` | flag | False | Mode sans board |
| `--port` | str | `/dev/ttyACM0` | Port UART |
| `--baud` | int | `115200` | Baud rate |
| `--n-samples` | int | `200` | Nombre de samples |
| `--n-tasks` | int | `3` | Nombre de tâches CL |
| `--rate-hz` | float | `0.0` | Rate-limit (0=max) |
| `--update` | flag | False | Update incrémental |
| `--platform` | str | `nucleo_f439zi` | Plateforme cible |
| `--output` | str | — | Répertoire de sortie (obligatoire) |
| `--verbose` | flag | False | Affichage détaillé |

---

## Critères d'acceptation

- [ ] `python scripts/board_dataset_builder.py --dataset cwru --dry-run --n-samples 500 --output experiments/exp_S18_01` produit les 3 fichiers en moins de 5 minutes
- [ ] `results.json` contient les 6 champs Phase 1 (`acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes`, `inference_latency_ms`, `n_params`)
- [ ] `dataset.csv` a exactement 500 lignes et les colonnes obligatoires
- [ ] `config_snapshot.yaml` enregistre `protocol_version: 2`
- [ ] Pas d'import `pyserial` en dry-run (testable en environnement CI sans USB)
