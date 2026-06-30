# S1918 — Expérience E19-05 : EWC board — CWRU

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_05` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé (2026-06-09) |
| **Durée estimée** | 3h |
| **Dépendances** | S1902 (ewc_consolidate ✅), S1911 (pipeline UART validé sur CWRU) |
| **Fichiers cibles** | `experiments/exp_S19_05/` |

---

## Contexte

S1912 a validé EWC C sur Monitoring (3 tâches, acc=8% ⚠️ — problème d'initialisation des poids identifié). Cette expérience applique EWC sur **CWRU** : classification domain-incremental en 3 tâches par type de défaut (normal / inner / outer). L'objectif est d'observer le comportement anti-oubli de l'EWC sur un dataset vibratoire plus structuré, et de corriger l'initialisation des poids identifiée en S1912.

---

## Objectif

Produire `experiments/exp_S19_05/results.json`. L'`acc_final` doit dépasser 50% (S1912 avait 8% à cause d'une mauvaise init — vérifier avant de lancer). Comparer `avg_forgetting` EWC vs Mahalanobis sur CWRU (S1911 : acc=68.7%).

---

## Setup expérimental

### Dataset CWRU — segmentation 3 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (0 HP) | ~167 |
| Task 1 | Inner race fault 0.014" | ~167 |
| Task 2 | Outer race fault 0.014" | ~166 |

**Features** (5 dims, Z-score) — cf. `configs/cwru_by_fault_config.yaml` :
- Variance temporelle, RMS, kurtosis, pic-à-pic, énergie spectrale

### Paramètres firmware

Depuis `configs/board_ewc.yaml` :
- `lambda: 100` (régularisation EWC)
- `lr: 0.01`
- `n_tasks: 3`, `n_samples: 500`

---

## Prérequis : correction initialisation poids (S1912)

Le problème d'acc=8% en S1912 était lié à une mauvaise initialisation des poids EWC head. Vérifier avant le flash :

```bash
# Régénérer les poids depuis le modèle Python entraîné
python scripts/export_weights_c.py \
    --model ewc \
    --config configs/board_ewc.yaml \
    --dataset cwru

# Vérifier que model_weights.h est à jour
grep "EWC_W1" firmware/stm32f4_blink/inc/model_weights.h | head -3
```

---

## Procédure (board réel uniquement — pas de dry-run)

### Flash et run

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model ewc \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --ewc-lambda 100 \
    --request-update \
    --output experiments/exp_S19_05 \
    --verbose
```

### Comparaison EWC vs Mahalanobis sur CWRU

```python
import json

ewc   = json.load(open("experiments/exp_S19_05/results.json"))
mahal = json.load(open("experiments/exp_S19_01/results.json"))

print(f"EWC         acc={ewc['acc_final']:.3f}  AF={ewc['avg_forgetting']:.3f}")
print(f"Mahalanobis acc={mahal['acc_final']:.3f}  AF={mahal['avg_forgetting']:.3f}")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_05",
  "model": "ewc",
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
  "ewc_lambda": 100,
  "config_snapshot": "configs/board_ewc.yaml"
}
```

---

## Points de vigilance

### Init poids — leçon de S1912

L'acc=8% de S1912 est attribuée à une mauvaise initialisation des poids. Toujours régénérer `model_weights.h` via `export_weights_c.py` avant un nouvel experiment. Ne jamais modifier `model_weights.h` à la main.

### Fisher matrix — 3 tâches CWRU

La consolidation Fisher après chaque tâche (`ewc_consolidate()`) doit être appelée exactement une fois par tâche. Vérifier dans `run_log.txt` que les 3 consolidations sont loguées.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_05/results.json` | Créer |
| `experiments/exp_S19_05/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_05/run_log.txt` | Log verbose |

---

## Vérification

- [x] Feature selection 9→5 via `configs/cwru_feature_subset.yaml` + `sensor_sim.py`
- [x] JSON créé avec 6 métriques obligatoires
- [x] `acc_final` > 50% → **87.55%** ✅ (Task0=100%, Task1=100%, Task2=62.65%)
- [x] `inference_latency_ms` < 100 ms → **0.251 ms** ✅
- [x] `avg_forgetting` = **18.67%** (EWC λ=400)
- [x] RAM = **9728 B** ✅ Gap 2 compliant

## Résultats (2026-06-09)

| Métrique | Valeur | Seuil | Statut |
|----------|--------|-------|--------|
| acc_final | 87.55% | > 50% | ✅ |
| avg_forgetting | 18.67% | — | ℹ️ |
| inference_latency_ms | 0.251 ms | < 100 ms | ✅ |
| ram_peak_bytes | 9728 B | < 256 Ko | ✅ |
| n_params | 1538 | — | ✅ |

**vs Mahalanobis CWRU (exp_S19_01)** : EWC acc=87.55% > Mahal acc=62.85% — EWC nettement supérieur sur CWRU.
**Note** : `--ewc-lambda 400` utilisé (board_ewc.yaml), pas 100 comme dans le sprint doc.
