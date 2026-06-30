# S1925 — Expérience E19-12 : TinyOL board — Pronostia

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_12` |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé — 2026-06-09 |
| **Durée estimée** | 3h (après déblocage S1903) |
| **Dépendances** | S1903 (TinyOL encoder C skeleton), S1923 (TinyOL/CWRU validé) |
| **Fichiers cibles** | `experiments/exp_S19_12/` |

---

## Contexte

Dernière expérience TinyOL, sur **Pronostia (FEMTO Bearing)** : class-incremental 2 tâches (normal / fault). Pronostia inclut une dimension de température — TinyOL autoencoder est potentiellement plus adapté que les modèles discriminatifs pour capturer la dégradation progressive des roulements via le score de reconstruction.

> **Bloquer jusqu'à S1903 ✅ et S1923 ✅.**

---

## Objectif

Produire `experiments/exp_S19_12/results.json`. Comparer TinyOL vs EWC vs HDC vs Mahalanobis sur Pronostia (respectivement `exp_S19_06`, `exp_S19_09`, `exp_S19_03`).

---

## Setup expérimental

### Dataset Pronostia — segmentation 2 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (roulements sains) | ~150 |
| Task 1 | Fault (roulements dégradés) | ~150 |

**Features** (5 dims) — `configs/pronostia_feature_subset.yaml` :
- RMS vibration X, RMS vibration Y, kurtosis X, kurtosis Y, température

### Paramètres firmware

Depuis `configs/board_tinyol.yaml` + `configs/tinyol_pronostia_by_condition_config.yaml`.

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
python scripts/export_weights_tinyol.py \
    --config configs/tinyol_pronostia_by_condition_config.yaml \
    --dataset pronostia

make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model tinyol \
    --dataset pronostia \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 2 \
    --request-update \
    --output experiments/exp_S19_12 \
    --verbose
```

### Tableau récapitulatif Pronostia (4 modèles)

```python
import json

results = {
    "TinyOL":      json.load(open("experiments/exp_S19_12/results.json")),
    "EWC":         json.load(open("experiments/exp_S19_06/results.json")),
    "HDC":         json.load(open("experiments/exp_S19_09/results.json")),
    "Mahalanobis": json.load(open("experiments/exp_S19_03/results.json")),
}

for name, r in results.items():
    print(f"{name:12s} acc={r['acc_final']:.3f}  AF={r['avg_forgetting']:.3f}  RAM={r['ram_peak_bytes']}B")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_12",
  "model": "tinyol",
  "dataset": "pronostia",
  "platform": "nucleo_f439zi",
  "date": "2026-06-XX",
  "acc_final": null,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_params": null,
  "n_tasks": 2,
  "n_samples_total": 300,
  "config_snapshot": "configs/tinyol_pronostia_by_condition_config.yaml"
}
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_12/results.json` | Créer |
| `experiments/exp_S19_12/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_12/run_log.txt` | Log verbose |

---

## Résultats (2026-06-09)

| Métrique | Valeur |
|----------|--------|
| `acc_final` | **0.9200** |
| `avg_forgetting` | 0.04 |
| `backward_transfer` | -0.04 |
| `ram_peak_bytes` | 5 800 B (5.7 Ko) |
| `inference_latency_ms` | **0.126 ms** |
| `n_params` | 881 |

`per_task_acc` : task0=0.94, task1=0.90

Notes :
- Threshold calibré P95×1.5 = 0.000029 sur MSE training (Pronostia très bien reconstruit)
- 5 features (std_acc_horiz, rms_acc_horiz, peak_acc_horiz, rms_acc_vert, temporal_position)

## Vérification

- [x] S1903 et S1923 complétés
- [x] Poids TinyOL générés sur Pronostia (raw, task0-only, 150 epochs)
- [x] JSON créé avec 6 métriques obligatoires
- [x] Gap 2 RAM ✅ et latence ✅
