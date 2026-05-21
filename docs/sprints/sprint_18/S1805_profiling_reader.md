# S1805 — Host profiling reader : parse métriques UART, sauve JSON experiments/ auto

| Champ | Valeur |
|-------|--------|
| **ID** | S1805 |
| **Sprint** | Sprint 18 — 25 mai – 1er juin 2026 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S1804 (firmware profiling), S1803 (dataset builder) |
| **Fichiers cibles** | `scripts/profiling_reader.py` |
| **Statut** | ✅ Implémenté — à valider |

---

## Objectif

Parser les métriques de profiling firmware (latence DWT, RAM .bss, throughput) et produire un `profiling.json` standardisé dans `experiments/exp_S18_XX/`.

Fonctionne en **deux modes** :
- **Live** : collecte directement via UART (board connectée)
- **Parse-only** : relit un `dataset.csv` produit par `board_dataset_builder.py`

---

## Fichier : `scripts/profiling_reader.py`

### Modes de fonctionnement

```
profiling_reader.py
├── --port /dev/ttyACM0    → collect_from_uart()  → records[]
└── --from-csv dataset.csv → parse_from_csv()      → records[]
                                        │
                                _compute_profiling_stats()
                                        │
                                profiling.json
```

### `parse_from_csv()` — Mode parse-only

Relit `dataset.csv` (produit par `board_dataset_builder.py`) et extrait les colonnes `latency_us`, `ram_bytes`, `throughput_ips`.

```python
def parse_from_csv(csv_path: Path) -> list[dict]:
    # Colonnes requises : latency_us, ram_bytes, throughput_ips
```

### `collect_from_uart()` — Mode live

Délègue à `sensor_stream._stream_uart()` pour collecter N inférences sur le port UART, puis extrait les métriques de profiling des réponses v2.

---

## Format de sortie : `profiling.json`

```json
{
  "platform": "nucleo_f439zi",
  "date": "2026-05-25 14:35:02",
  "n_samples": 100,
  "latency_mean_us": 445.2,
  "latency_p50_us": 440.0,
  "latency_p99_us": 1200.0,
  "latency_mean_ms": 0.4452,
  "latency_p99_ms": 1.2000,
  "ram_mean_bytes": 18200,
  "ram_peak_bytes": 19456,
  "throughput_mean_ips": 2247,
  "throughput_min_ips": 833,
  "alerts": [],
  "gap2_compliant": true
}
```

### Champ `gap2_compliant`

```python
"gap2_compliant": ram_peak < 64000 and lat_mean_ms < 100.0
```

Ce booléen répond directement au **Gap 2** du projet : *"Opération sous 100 Ko RAM avec chiffres précis mesurés"* ET latence < 100 ms (contrainte hardware STM32N6).

Un `profiling.json` avec `gap2_compliant: true` est une contribution directe au manuscrit.

---

## Seuils d'alerte

Lus depuis `configs/profiling_config.yaml` (voir [S1806](S1806_profiling_config.md)) :

| Métrique | Seuil alerte | Justification |
|----------|-------------|---------------|
| `latency_mean_ms` | > 10 ms | 10% du budget 100 ms |
| `ram_peak_bytes` | > 52 000 B | Marge 12 Ko vs contrainte 64 Ko |
| `throughput_mean_ips` | < 10 ips | Minimum fonctionnel |

Si un seuil est dépassé, le champ `alerts` contient des messages explicatifs :

```json
"alerts": [
  "LATENCY: 12.50 ms > seuil 10.0 ms",
  "RAM: 55000 B > seuil 52000 B"
]
```

---

## Usage CLI

```bash
# Mode parse-only (depuis CSV board_dataset_builder — recommandé en dev)
python scripts/profiling_reader.py \
    --from-csv experiments/exp_S18_01/dataset.csv \
    --save experiments/exp_S18_01/profiling.json

# Mode live (board connectée)
python scripts/profiling_reader.py \
    --port /dev/ttyACM0 \
    --n-samples 100 \
    --platform nucleo_f439zi \
    --save experiments/exp_S18_01/profiling.json
```

### Arguments CLI

| Argument | Type | Description |
|----------|------|-------------|
| `--port` | str | Port UART (exclusif avec `--from-csv`) |
| `--from-csv` | str | CSV existant de board_dataset_builder |
| `--baud` | int | Baud rate (défaut 115200) |
| `--n-samples` | int | Nombre de samples en mode live (défaut 100) |
| `--platform` | str | Plateforme cible (défaut `nucleo_f439zi`) |
| `--save` | str | Chemin de sortie `profiling.json` (obligatoire) |
| `--verbose` | flag | Affichage détaillé |

---

## Workflow intégré Sprint 18

```bash
# Étape 1 : collecte + CSV
python scripts/board_dataset_builder.py \
    --dataset cwru --dry-run --n-samples 500 \
    --output experiments/exp_S18_01

# Étape 2 : profiling depuis le CSV
python scripts/profiling_reader.py \
    --from-csv experiments/exp_S18_01/dataset.csv \
    --save experiments/exp_S18_01/profiling.json

# Résultat attendu en dry-run
# gap2_compliant: true (lat~0.003ms, ram~200B — valeurs simulées)
```

---

## Critères d'acceptation

- [ ] `--from-csv experiments/exp_S18_01/dataset.csv --save profiling.json` fonctionne sans board
- [ ] `profiling.json` contient tous les champs : mean/P50/P99 latence, RAM peak, throughput min/mean, alerts, gap2_compliant
- [ ] `gap2_compliant: true` en dry-run (lat~3µs, ram~200B)
- [ ] Mode `--port` refuse de démarrer si `pyserial` est absent (ImportError clair)
- [ ] `alerts` vide si toutes les métriques sont dans les seuils
