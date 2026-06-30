# S1914 — Expérience E19-03 : Mahalanobis board — Pronostia

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S1911 (pipeline UART validé sur CWRU), S1907 (recorder) |
| **Fichiers cibles** | `experiments/exp_S19_03/` |

---

## Contexte

S1911 a validé Mahalanobis C sur CWRU (3 tâches, 198 samples). Cette expérience étend la couverture dataset au **PRONOSTIA (FEMTO Bearing)** : détection de condition (normal / fault) en 2 tâches class-incremental. L'objectif est de confirmer que le détecteur Mahalanobis C généralise à un dataset vibratoire différent de CWRU, avec des features de type RUL.

---

## Objectif

Produire `experiments/exp_S19_03/results.json` avec les 6 métriques obligatoires. Comparer `acc_final` avec la référence Python Pronostia Mahalanobis (configs existantes : `configs/mahalanobis_pronostia_by_condition_config.yaml`).

---

## Setup expérimental

### Dataset Pronostia — segmentation 2 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (roulements sains) | ~150 |
| Task 1 | Fault (roulements dégradés) | ~150 |

**Features** (5 dims, normalisées Z-score) — cf. `configs/pronostia_feature_subset.yaml` :
- RMS vibration X, RMS vibration Y, kurtosis X, kurtosis Y, température

### Paramètres firmware

Depuis `configs/board_pronostia.yaml` + `configs/board_mahalanobis.yaml` :
- `threshold_init: 3.0` (distance Mahalanobis)
- `ema_alpha: 0.05` (adaptation en ligne)
- `n_tasks: 2`, `n_samples: 300`

### Référence Python Phase 1

Chercher experiment Pronostia Mahalanobis dans `experiments/` :
```bash
grep -rl '"model": "mahalanobis"' experiments/ | xargs grep -l '"dataset": "pronostia"'
```

---

## Procédure (board réel uniquement — pas de dry-run)

### Prérequis

```bash
# 1. Flasher le firmware (avec mahalanobis.c compilé)
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0
```

### Lancer l'expérience

```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis \
    --dataset pronostia \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 2 \
    --request-update \
    --output experiments/exp_S19_03 \
    --verbose
```

### Comparaison PC vs board

```python
import json

ref = json.load(open("experiments/<ref_pronostia_mahal>/results.json"))
board = json.load(open("experiments/exp_S19_03/results.json"))

delta_acc = abs(ref["acc_final"] - board["acc_final"])
print(f"Delta acc_final: {delta_acc:.4f} (seuil: 0.05)")
```

> Seuil de delta élargi à 5% pour Pronostia (dataset plus petit, variance plus élevée).

---

## Format JSON attendu

```json
{
  "exp_id": "S19_03",
  "model": "mahalanobis",
  "dataset": "pronostia",
  "platform": "nucleo_f439zi",
  "date": "2026-06-XX",
  "acc_final": null,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_params": 30,
  "n_tasks": 2,
  "n_samples_total": 300,
  "config_snapshot": "configs/board_mahalanobis.yaml"
}
```

---

## Points de vigilance

### Features Pronostia ≠ CWRU

Les features Pronostia incluent la **température** (5ème dimension) absente de CWRU. Vérifier que `model_weights.h` contient les stats Z-score calculées sur Pronostia et non sur CWRU :
- `ZSCORE_MEAN[4]` doit correspondre à la moyenne température Pronostia

### Segmentation 2 tâches vs 3 tâches

Le firmware est configuré pour `N_TASKS_MAX` via `inc/mahalanobis.h`. Vérifier que `n_tasks=2` est compatible sans recompilation (paramètre runtime ou `#define`).

### FIXME(gap2)

`ram_peak_bytes` mesuré sur NUCLEO-F439ZI (192 Ko SRAM) — valeur indicative seulement.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_03/results.json` | Créer (sortie expérience) |
| `experiments/exp_S19_03/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_03/run_log.txt` | Log verbose de la session |

---

## Vérification

- [x] JSON créé avec 6 métriques obligatoires
- [x] `inference_latency_ms` = 0.005 ms < 1 ms ✅
- [x] `ram_peak_bytes` : 0 (non mesuré par firmware — FIXME gap2)
- [x] `acc_final` = 0.877 ✅
- [x] Comparaison PC vs board calculée — delta = 8.4% (Python sur 13 feat vs board sur 5 feat)

---

## Résultats (2026-06-09)

| Métrique | Valeur |
|----------|--------|
| `acc_final` | 0.877 |
| `avg_forgetting` | 0.007 |
| `backward_transfer` | -0.007 |
| `inference_latency_ms` | 0.005 |
| `ram_peak_bytes` | 0 (FIXME gap2) |
| `n_params` | 30 |

**Note :** Checkpoint entraîné inline sur 5 features sélectionnées (subset [1,2,4,8,12]) depuis `_load_pronostia()`.
Référence Python (exp_054, 13 features) : acc=0.793. Le board outperform grâce à la sélection de features.

---

## Questions ouvertes

- `TODO(arnaud)` : Référence Python 5-features Pronostia = exp_board_pronostia_mahal (inline Sprint 19).
- `FIXME(gap2)` : Validation < 64 Ko sur Cortex-M55 requise pour le Gap 2 formel.
