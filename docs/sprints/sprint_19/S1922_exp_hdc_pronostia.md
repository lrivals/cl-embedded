# S1922 — Expérience E19-09 : HDC board — Pronostia

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_09` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S1920 (HDC/CWRU validé), S1914 (Pronostia pipeline validé) |
| **Fichiers cibles** | `experiments/exp_S19_09/` |

---

## Contexte

Troisième experiment HDC, sur **Pronostia (FEMTO Bearing)** : class-incremental 2 tâches (normal / fault). Les base vectors Pronostia sont disponibles dans `configs/hdc_base_vectors_pronostia.npz`. Pronostia avec HDC est intéressant car les features RUL (vibrations + température) sont continues, testant la capacité d'encodage hyperdimensionnel sur des features de dégradation progressive.

---

## Objectif

Produire `experiments/exp_S19_09/results.json`. Comparer `acc_final` HDC vs EWC vs Mahalanobis sur Pronostia (respectivement `exp_S19_06` et `exp_S19_03`).

---

## Setup expérimental

### Dataset Pronostia — segmentation 2 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (roulements sains) | ~150 |
| Task 1 | Fault (roulements dégradés) | ~150 |

**Features** (5 dims) + base vectors depuis `configs/hdc_base_vectors_pronostia.npz`.

### Paramètres firmware

Depuis `configs/board_hdc.yaml` :
- `hd_dim: 1024`
- `n_tasks: 2`, `n_samples: 300`

---

## Prérequis : vérification base vectors Pronostia

```bash
python -c "
import numpy as np
d = np.load('configs/hdc_base_vectors_pronostia.npz')
print(d.files, d['base_vectors'].shape)
"
```

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model hdc \
    --dataset pronostia \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 2 \
    --request-update \
    --output experiments/exp_S19_09 \
    --verbose
```

### Tableau récapitulatif Pronostia (3 modèles)

```python
import json

hdc   = json.load(open("experiments/exp_S19_09/results.json"))
ewc   = json.load(open("experiments/exp_S19_06/results.json"))
mahal = json.load(open("experiments/exp_S19_03/results.json"))

for name, r in [("HDC", hdc), ("EWC", ewc), ("Mahalanobis", mahal)]:
    print(f"{name:12s} acc={r['acc_final']:.3f}  AF={r['avg_forgetting']:.3f}  RAM={r['ram_peak_bytes']}B")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_09",
  "model": "hdc",
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
  "hd_dim": 1024,
  "config_snapshot": "configs/hdc_pronostia_by_condition_config.yaml"
}
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_09/results.json` | Créer |
| `experiments/exp_S19_09/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_09/run_log.txt` | Log verbose |

---

## Résultats mesurés (2026-06-09)

| Métrique | HDC | EWC | Mahalanobis |
|----------|-----|-----|-------------|
| `acc_final` | 0.887 | **0.922** | 0.877 |
| `avg_forgetting` | 0.000 | 0.081 | 0.007 |
| `inference_latency_ms` | 0.652 | 0.251 | 0.005 |

HDC et Mahalanobis sans forgetting sur Pronostia (2 tâches). EWC légèrement meilleur en acc mais avec forgetting modéré (0.081). HDC confirme la séparabilité hyperdimensionnelle sur des features de dégradation progressive.

## Vérification

- [x] Base vectors Pronostia présents dans `configs/hdc_base_vectors_pronostia.npz`
- [x] JSON créé avec 6 métriques obligatoires
- [x] Tableau comparatif HDC / EWC / Mahalanobis sur Pronostia noté
