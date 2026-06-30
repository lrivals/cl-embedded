# S1915 — Expérience E19-04 : Mahalanobis board — Equipment Monitoring

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_04` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S1911 (pipeline UART validé), S1912 (EWC Monitoring validé — même dataset) |
| **Fichiers cibles** | `experiments/exp_S19_04/` |

---

## Contexte

S1911 a validé Mahalanobis C sur CWRU et S1914 le valide sur Pronostia. Cette expérience complète la couverture Mahalanobis en testant le détecteur C sur **Equipment Monitoring** : classification domain-incremental en 3 tâches (pump → turbine → compressor). Ce dataset a déjà été utilisé pour EWC (S1912), ce qui permettra une comparaison directe entre les deux approches sur le même dataset.

---

## Objectif

Produire `experiments/exp_S19_04/results.json` avec les 6 métriques obligatoires. Comparer `acc_final` Mahalanobis vs EWC sur Monitoring (S1912), sur le même board.

---

## Setup expérimental

### Dataset Equipment Monitoring — segmentation 3 tâches

| Tâche | Domaine | Samples |
|-------|---------|---------|
| Task 0 | Pompe (pump) | ~100 |
| Task 1 | Turbine | ~100 |
| Task 2 | Compresseur | ~100 |

**Features** (5 dims, tabulaire, normalisées Z-score) — cf. `configs/monitoring_normalizer.yaml` :
- Température, pression, vibration, humidité, type équipement (encodé)

### Paramètres firmware

Depuis `configs/board_mahalanobis.yaml` :
- `threshold_init: 3.0`
- `ema_alpha: 0.05`
- `n_tasks: 3`, `n_samples: 300`

### Référence Python Phase 1

```bash
grep -rl '"model": "mahalanobis"' experiments/ | xargs grep -l '"dataset": "monitoring"'
```

---

## Procédure (board réel uniquement — pas de dry-run)

### Prérequis

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0
```

### Lancer l'expérience

```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_04 \
    --verbose
```

### Comparaison Mahalanobis vs EWC sur Monitoring

```python
import json

mahal = json.load(open("experiments/exp_S19_04/results.json"))
ewc   = json.load(open("experiments/exp_S19_02/results.json"))

print(f"Mahalanobis acc_final : {mahal['acc_final']:.3f}")
print(f"EWC         acc_final : {ewc['acc_final']:.3f}")
print(f"Mahalanobis AF        : {mahal['avg_forgetting']:.3f}")
print(f"EWC         AF        : {ewc['avg_forgetting']:.3f}")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_04",
  "model": "mahalanobis",
  "dataset": "monitoring",
  "platform": "nucleo_f439zi",
  "date": "2026-06-XX",
  "acc_final": null,
  "avg_forgetting": null,
  "backward_transfer": null,
  "ram_peak_bytes": null,
  "inference_latency_ms": null,
  "n_params": 30,
  "n_tasks": 3,
  "n_samples_total": 300,
  "config_snapshot": "configs/board_mahalanobis.yaml"
}
```

---

## Points de vigilance

### Z-score Monitoring ≠ CWRU ≠ Pronostia

Les stats de normalisation doivent être celles du dataset Monitoring (`configs/monitoring_normalizer.yaml`). Vérifier que `model_weights.h` contient les bonnes constantes avant le flash.

### Comparaison avec EWC (S1912)

Le dataset Monitoring est utilisé pour les deux modèles. Si Mahalanobis obtient un `avg_forgetting` plus faible qu'EWC, noter que Mahalanobis n'a pas de mécanisme d'oubli catastrophique par conception (pas de gradient update).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_04/results.json` | Créer |
| `experiments/exp_S19_04/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_04/run_log.txt` | Log verbose |

---

## Vérification

- [x] JSON créé avec 6 métriques obligatoires
- [x] `inference_latency_ms` = 0.004 ms < 1 ms ✅
- [x] `ram_peak_bytes` : 0 (non mesuré par firmware — FIXME gap2)
- [x] Comparaison Mahalanobis vs EWC sur Monitoring (exp_S19_02 = dry-run, acc=8%)

---

## Résultats (2026-06-09)

| Métrique | Mahalanobis board | EWC board (S19_02) |
|----------|-------------------|-------------------|
| `acc_final` | **0.960** | 0.080 (dry-run) |
| `avg_forgetting` | **0.005** | 0.000 |
| `inference_latency_ms` | 0.004 | 0.004 |
| `n_params` | 30 | 1538 |

**Note MAHA_DIM :** Monitoring = 4 features → recompilation avec `MAHA_DIM=4` pour cette expérience.
MAHA_DIM restauré à 5 après. Checkpoint entraîné inline (exp_board_monitoring_mahal).
Mahalanobis montre un forgetting quasi-nul (0.5%) par conception (pas de gradient update).
