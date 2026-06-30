# S1923 — Expérience E19-10 : TinyOL board — CWRU

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_10` |
| **Priorité** | 🟢 Nice-to-have |
| **Statut** | ✅ Terminé — 2026-06-09 |
| **Durée estimée** | 3h (après déblocage S1903) |
| **Dépendances** | S1903 (TinyOL encoder C skeleton + poids Flash) |
| **Fichiers cibles** | `experiments/exp_S19_10/` |

---

## Contexte

TinyOL (M1) est l'autoencoder incrémental du projet avec une tête OtO (One-Task-One-Head). Le firmware `tinyol.c` est en cours de finalisation (S1903 : forward pass + export poids). Cette expérience valide TinyOL C sur **CWRU** en 3 tâches de défaut — mais est conditionnée à la completion de S1903.

> **Bloquer jusqu'à S1903 ✅**. Ne pas lancer si `tinyol.c` n'est pas compilable avec les poids exportés.

---

## Objectif

Produire `experiments/exp_S19_10/results.json`. TinyOL étant un autoencoder, l'`acc_final` est calculée via le score de reconstruction (anomaly score), pas via une classification directe. Comparer avec HDC et EWC sur CWRU.

---

## Setup expérimental

### Dataset CWRU — segmentation 3 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (0 HP) | ~167 |
| Task 1 | Inner race fault 0.014" | ~167 |
| Task 2 | Outer race fault 0.014" | ~166 |

**Features** (5 dims) — mêmes que les autres modèles CWRU pour comparabilité.

### Paramètres firmware

Depuis `configs/board_tinyol.yaml` :
- Architecture encoder : couches définies dans `inc/tinyol.h`
- OtO head : 1 tête par tâche
- `n_tasks: 3`, `n_samples: 500`

---

## Prérequis : completion S1903

```bash
# Vérifier que tinyol.c est compilable
make -C firmware/stm32f4_blink/ all 2>&1 | grep -i tinyol

# Vérifier que les poids sont exportés
python scripts/export_weights_tinyol.py \
    --config configs/board_tinyol.yaml \
    --dataset cwru
grep "TINYOL_W" firmware/stm32f4_blink/inc/model_weights.h | head -3
```

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model tinyol \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_10 \
    --verbose
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_10",
  "model": "tinyol",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-06-XX",
  "acc_final": null,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_params": null,
  "n_tasks": 3,
  "n_samples_total": 500,
  "config_snapshot": "configs/board_tinyol.yaml"
}
```

---

## Points de vigilance

### Score TinyOL = reconstruction error

`acc_final` pour TinyOL est dérivé du score de reconstruction par rapport à un seuil. Si le seuil est mal calibré, l'acc sera biaisée. Vérifier le seuil dans `configs/board_tinyol.yaml`.

### Latence TinyOL — forward pass autoencoder

Estimé : 2 000–4 500 µs (beaucoup plus lent que Mahalanobis/HDC). Acceptable pour la contrainte < 100 ms.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_10/results.json` | Créer (après S1903) |
| `experiments/exp_S19_10/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_10/run_log.txt` | Log verbose |

---

## Résultats (2026-06-09)

| Métrique | Valeur |
|----------|--------|
| `acc_final` | **0.9056** |
| `avg_forgetting` | 0.1416 |
| `backward_transfer` | -0.1416 |
| `ram_peak_bytes` | 5 800 B (5.7 Ko) |
| `inference_latency_ms` | **0.126 ms** |
| `n_params` | 881 |

`per_task_acc` : task0=1.000, task1=1.000, task2=0.717

Notes :
- Poids entraînés via `export_weights_tinyol.py --train-dataset cwru --task0-only` (150 epochs MSE)
- Threshold calibré : P95 × 1.5 sur MSE training
- Update board = no-op (pas de backprop bare-metal) — inference fixe

## Vérification

- [x] S1903 complété (tinyol.c compilable + poids exportés)
- [x] JSON créé avec 6 métriques obligatoires
- [x] `inference_latency_ms` = 0.126 ms < 5 ms ✅
- [x] `ram_peak_bytes` = 5 800 B < 64 Ko ✅
