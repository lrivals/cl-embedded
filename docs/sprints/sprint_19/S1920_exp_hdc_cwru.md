# S1920 — Expérience E19-07 : HDC board — CWRU

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_07` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | `firmware/stm32f4_blink/src/hdc.c` compilable, `configs/hdc_base_vectors_cwru.npz` existant |
| **Fichiers cibles** | `experiments/exp_S19_07/` |

---

## Contexte

HDC (Hyperdimensional Computing) est le modèle M3 du projet — architecture-based, non-neuronal, avec mise à jour incrémentale par accumulation de hypervecteurs. Le firmware `hdc.c` est implémenté sur NUCLEO-F439ZI. Cette expérience valide HDC C sur **CWRU** en 3 tâches de défaut. HDC est particulièrement adapté à CWRU car la classification par type de défaut bénéficie de la séparabilité des hypervecteurs dans l'espace HD.

---

## Objectif

Produire `experiments/exp_S19_07/results.json`. Comparer `acc_final` HDC vs Mahalanobis vs EWC sur CWRU (respectivement `exp_S19_01` et `exp_S19_05`).

---

## Setup expérimental

### Dataset CWRU — segmentation 3 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (0 HP) | ~167 |
| Task 1 | Inner race fault 0.014" | ~167 |
| Task 2 | Outer race fault 0.014" | ~166 |

**Features** (5 dims) + base vectors HDC depuis `configs/hdc_base_vectors_cwru.npz`.

### Paramètres firmware

Depuis `configs/board_hdc.yaml` :
- `hd_dim: 1024` (dimension de l'espace hyperdimensionnel)
- `n_tasks: 3`, `n_samples: 500`

---

## Prérequis : vérification base vectors

```bash
# Vérifier que les base vectors CWRU sont générés
python -c "import numpy as np; d = np.load('configs/hdc_base_vectors_cwru.npz'); print(d.files, d['base_vectors'].shape)"

# Si absent, régénérer :
python scripts/export_weights_c.py \
    --model hdc \
    --config configs/board_hdc.yaml \
    --dataset cwru
```

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model hdc \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_07 \
    --verbose
```

### Comparaison 3 modèles sur CWRU

```python
import json

hdc   = json.load(open("experiments/exp_S19_07/results.json"))
mahal = json.load(open("experiments/exp_S19_01/results.json"))
ewc   = json.load(open("experiments/exp_S19_05/results.json"))

for name, r in [("HDC", hdc), ("Mahalanobis", mahal), ("EWC", ewc)]:
    print(f"{name:12s} acc={r['acc_final']:.3f}  AF={r['avg_forgetting']:.3f}  lat={r['inference_latency_ms']:.3f}ms")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_07",
  "model": "hdc",
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
  "hd_dim": 1024,
  "config_snapshot": "configs/board_hdc.yaml"
}
```

---

## Points de vigilance

### RAM HDC — plus importante que Mahalanobis

HDC avec `hd_dim=1024` et 3 classes requiert ~28 Ko de RAM (cf. budget sprint 19). Vérifier que le `ram_peak_bytes` mesuré est cohérent avec `make size`.

### Latence HDC — produit scalaire 1024-dim

La latence HDC est dominée par les produits scalaires de dimension 1024 sur Cortex-M4 sans NPU. Estimé : 400–800 µs (acceptable < 100 ms).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_07/results.json` | Créer |
| `experiments/exp_S19_07/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_07/run_log.txt` | Log verbose |

---

## Résultats mesurés (2026-06-09)

| Métrique | HDC | Mahalanobis | EWC |
|----------|-----|-------------|-----|
| `acc_final` | **0.900** | 0.628 | 0.875 |
| `avg_forgetting` | 0.145 | 0.000 | 0.187 |
| `inference_latency_ms` | 0.652 | 0.004 | 0.251 |
| `ram_peak_bytes` | ~28 312 B† | 0 | 9 728 |

† `ram_peak_bytes=0` dans JSON : protocole v3 ne rapporte pas la RAM ; valeur théorique 28 312 B depuis `hdc.h`.

**Per-task accuracy HDC** : task 0 (Normal) = 0.994 · task 1 (Inner race) = 1.000 · task 2 (Outer race) = 0.705

HDC meilleur que Mahalanobis (+27 pts) sur CWRU. Forgetting modéré (0.145) dû à la tâche outer race.

## Vérification

- [x] Base vectors CWRU présents dans `configs/hdc_base_vectors_cwru.npz`
- [x] JSON créé avec 6 métriques obligatoires
- [x] `ram_peak_bytes` cohérent avec budget HDC (~28 Ko)
- [x] `inference_latency_ms` < 2 ms (0.652 ms ✅)
- [x] Comparaison 3 modèles sur CWRU notée
