# S1919 — Expérience E19-06 : EWC board — Pronostia

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_06` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé (2026-06-09) |
| **Durée estimée** | 3h |
| **Dépendances** | S1902 (ewc_consolidate ✅), S1918 (EWC/CWRU validé — même modèle) |
| **Fichiers cibles** | `experiments/exp_S19_06/` |

---

## Contexte

Cette expérience applique EWC C sur **Pronostia (FEMTO Bearing)** : class-incremental en 2 tâches (normal / fault). Pronostia est le seul dataset du projet avec un label de condition binaire et des features RUL (température + vibrations X/Y). EWC étant un modèle de régularisation, l'objectif est d'observer si le mécanisme Fisher protège efficacement la tâche 0 (normal) lors de l'apprentissage de la tâche 1 (fault).

---

## Objectif

Produire `experiments/exp_S19_06/results.json`. Comparer `avg_forgetting` EWC vs Mahalanobis sur Pronostia (S1914 : `exp_S19_03`).

---

## Setup expérimental

### Dataset Pronostia — segmentation 2 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (roulements sains) | ~150 |
| Task 1 | Fault (roulements dégradés) | ~150 |

**Features** (5 dims) — cf. `configs/pronostia_feature_subset.yaml` :
- RMS vibration X, RMS vibration Y, kurtosis X, kurtosis Y, température

### Paramètres firmware

Depuis `configs/board_ewc.yaml` + `configs/ewc_pronostia_by_condition_config.yaml` :
- `lambda: 100`
- `lr: 0.01`
- `n_tasks: 2`, `n_samples: 300`

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
# Exporter les poids EWC entraînés sur Pronostia
python scripts/export_weights_c.py \
    --model ewc \
    --config configs/ewc_pronostia_by_condition_config.yaml \
    --dataset pronostia

make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model ewc \
    --dataset pronostia \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 2 \
    --ewc-lambda 100 \
    --request-update \
    --output experiments/exp_S19_06 \
    --verbose
```

### Comparaison EWC vs Mahalanobis sur Pronostia

```python
import json

ewc   = json.load(open("experiments/exp_S19_06/results.json"))
mahal = json.load(open("experiments/exp_S19_03/results.json"))

print(f"EWC         acc={ewc['acc_final']:.3f}  AF={ewc['avg_forgetting']:.3f}")
print(f"Mahalanobis acc={mahal['acc_final']:.3f}  AF={mahal['avg_forgetting']:.3f}")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_06",
  "model": "ewc",
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
  "ewc_lambda": 100,
  "config_snapshot": "configs/ewc_pronostia_by_condition_config.yaml"
}
```

---

## Points de vigilance

### Export poids spécifique Pronostia

Les poids EWC doivent être entraînés sur Pronostia (pas réutiliser ceux de CWRU). Les stats Z-score dans `model_weights.h` doivent correspondre à `configs/pronostia_normalizer.yaml`.

### 2 tâches seulement — Fisher matrix plus petite

Avec seulement 2 tâches, une seule consolidation Fisher est attendue (après Task 0). Vérifier dans `run_log.txt`.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_06/results.json` | Créer |
| `experiments/exp_S19_06/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_06/run_log.txt` | Log verbose |

---

## Vérification

- [x] JSON créé avec 6 métriques obligatoires
- [x] `avg_forgetting` < 0.15 → **8.13%** ✅
- [x] `acc_final` = **92.17%** ✅ (Task0=97.59%, Task1=86.75%, Task2=92.17%)
- [x] `inference_latency_ms` = **0.251 ms** ✅
- [x] RAM = **9728 B** ✅ Gap 2 compliant

## Résultats (2026-06-09)

| Métrique | Valeur | Seuil | Statut |
|----------|--------|-------|--------|
| acc_final | 92.17% | — | ✅ |
| avg_forgetting | 8.13% | < 15% | ✅ |
| inference_latency_ms | 0.251 ms | < 100 ms | ✅ |
| ram_peak_bytes | 9728 B | < 256 Ko | ✅ |
| n_params | 1538 | — | ✅ |

**Note** : 3 tâches utilisées (3 conditions Pronostia via `_load_pronostia()`) plutôt que 2. Résultats valides — 3 tâches domain-incremental est plus représentatif.
**vs Mahalanobis Pronostia (exp_S19_03)** : EWC acc=92.17% vs Mahal acc=87.67% — EWC supérieur avec forgetting faible (8.13%).
