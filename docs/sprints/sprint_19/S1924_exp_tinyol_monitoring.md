# S1924 — Expérience E19-11 : TinyOL board — Equipment Monitoring

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_11` |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé — 2026-06-09 |
| **Durée estimée** | 3h (après déblocage S1903) |
| **Dépendances** | S1903 (TinyOL encoder C skeleton), S1923 (TinyOL/CWRU validé) |
| **Fichiers cibles** | `experiments/exp_S19_11/` |

---

## Contexte

Extension de S1923 sur **Equipment Monitoring** : domain-incremental 3 tâches (pump → turbine → compressor). Monitoring est le dataset de référence du projet pour EWC et HDC — TinyOL permet de compléter la matrice et d'évaluer si l'autoencoder incrémental se comporte différemment sur des features tabulaires vs vibratoires.

> **Bloquer jusqu'à S1903 ✅ et S1923 ✅.**

---

## Objectif

Produire `experiments/exp_S19_11/results.json`. Comparer TinyOL vs EWC vs HDC vs Mahalanobis sur Monitoring (`exp_S19_02`, `exp_S19_04`, `exp_S19_08`).

---

## Setup expérimental

### Dataset Equipment Monitoring — segmentation 3 tâches

| Tâche | Domaine | Samples |
|-------|---------|---------|
| Task 0 | Pompe (pump) | ~100 |
| Task 1 | Turbine | ~100 |
| Task 2 | Compresseur | ~100 |

**Features** (5 dims tabulaires) — `configs/monitoring_normalizer.yaml`.

### Paramètres firmware

Depuis `configs/board_tinyol.yaml` — mêmes que S1923 (firmware non recompilé si même config).

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
# Exporter poids TinyOL entraînés sur Monitoring
python scripts/export_weights_tinyol.py \
    --config configs/tinyol_monitoring_config.yaml \
    --dataset monitoring

make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model tinyol \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_11 \
    --verbose
```

### Tableau récapitulatif Monitoring (4 modèles)

```python
import json

results = {
    "TinyOL":      json.load(open("experiments/exp_S19_11/results.json")),
    "EWC":         json.load(open("experiments/exp_S19_02/results.json")),
    "HDC":         json.load(open("experiments/exp_S19_08/results.json")),
    "Mahalanobis": json.load(open("experiments/exp_S19_04/results.json")),
}

for name, r in results.items():
    print(f"{name:12s} acc={r['acc_final']:.3f}  AF={r['avg_forgetting']:.3f}  lat={r['inference_latency_ms']:.3f}ms")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_11",
  "model": "tinyol",
  "dataset": "monitoring",
  "platform": "nucleo_f439zi",
  "date": "2026-06-XX",
  "acc_final": null,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_params": null,
  "n_tasks": 3,
  "n_samples_total": 300,
  "config_snapshot": "configs/tinyol_monitoring_config.yaml"
}
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_11/results.json` | Créer |
| `experiments/exp_S19_11/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_11/run_log.txt` | Log verbose |

---

## Résultats (2026-06-09)

| Métrique | Valeur |
|----------|--------|
| `acc_final` | **0.9567** |
| `avg_forgetting` | 0.02 |
| `backward_transfer` | -0.02 |
| `ram_peak_bytes` | 5 800 B (5.7 Ko) |
| `inference_latency_ms` | **0.126 ms** |
| `n_params` | 881 |

`per_task_acc` : task0=0.97, task1=0.95, task2=0.95

Notes :
- Monitoring a 4 features brutes → zero-padded à 5 par pipeline.c (ligne 238)
- Threshold calibré P95×1.5 = 0.002518 (valeurs brutes, pas normalisées)
- Fix : `pipeline.c` utilise maintenant `TINYOL_THRESHOLD` de `model_weights.h`

## Vérification

- [x] S1903 et S1923 complétés
- [x] Poids TinyOL générés sur Monitoring (raw, task0-only, 150 epochs)
- [x] JSON créé avec 6 métriques obligatoires
- [x] Gap 2 RAM ✅ et latence ✅
