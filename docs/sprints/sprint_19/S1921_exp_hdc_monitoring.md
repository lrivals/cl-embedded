# S1921 — Expérience E19-08 : HDC board — Equipment Monitoring

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **exp_ID** | `exp_S19_08` |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Terminé |
| **Durée estimée** | 3h |
| **Dépendances** | S1920 (HDC/CWRU validé — même firmware) |
| **Fichiers cibles** | `experiments/exp_S19_08/` |

---

## Contexte

Après validation HDC sur CWRU (S1920), cette expérience teste HDC C sur **Equipment Monitoring** : domain-incremental en 3 tâches (pump → turbine → compressor). Monitoring est un dataset tabulaire — les features sont moins riches spectralement que CWRU, ce qui permet d'évaluer si la séparabilité hyperdimensionnelle est robuste aux features non-vibratoires.

---

## Objectif

Produire `experiments/exp_S19_08/results.json`. Comparer HDC vs EWC vs Mahalanobis sur Monitoring (respectivement `exp_S19_02` et `exp_S19_04`).

---

## Setup expérimental

### Dataset Equipment Monitoring — segmentation 3 tâches

| Tâche | Domaine | Samples |
|-------|---------|---------|
| Task 0 | Pompe (pump) | ~100 |
| Task 1 | Turbine | ~100 |
| Task 2 | Compresseur | ~100 |

**Features** (5 dims tabulaires) — `configs/monitoring_normalizer.yaml` :
- Température, pression, vibration, humidité, type équipement

### Paramètres firmware

Depuis `configs/board_hdc.yaml` :
- `hd_dim: 1024`
- `n_tasks: 3`, `n_samples: 300`

> Pas de base vectors spécifiques Monitoring : les base vectors HDC sont générés aléatoirement (seed fixée) et indépendants du dataset.

---

## Procédure (board réel uniquement — pas de dry-run)

```bash
make -C firmware/stm32f4_blink/ all
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

python scripts/board_experiment_recorder.py \
    --model hdc \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 300 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_08 \
    --verbose
```

### Comparaison 3 modèles sur Monitoring

```python
import json

hdc   = json.load(open("experiments/exp_S19_08/results.json"))
ewc   = json.load(open("experiments/exp_S19_02/results.json"))
mahal = json.load(open("experiments/exp_S19_04/results.json"))

for name, r in [("HDC", hdc), ("EWC", ewc), ("Mahalanobis", mahal)]:
    print(f"{name:12s} acc={r['acc_final']:.3f}  AF={r['avg_forgetting']:.3f}")
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_08",
  "model": "hdc",
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
  "hd_dim": 1024,
  "config_snapshot": "configs/board_hdc.yaml"
}
```

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_08/results.json` | Créer |
| `experiments/exp_S19_08/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_08/run_log.txt` | Log verbose |

---

## Résultats mesurés (2026-06-09)

| Métrique | HDC | EWC | Mahalanobis |
|----------|-----|-----|-------------|
| `acc_final` | 0.910 | 0.080† | **0.960** |
| `avg_forgetting` | 0.000 | 0.000 | 0.005 |
| `inference_latency_ms` | 0.652 | 0.004 | 0.004 |

† EWC/Monitoring acc très bas — état SRAM non remis à zéro lors de l'expérience d'origine (exp_S19_02).

HDC atteint 0.91 sur Monitoring sans forgetting. Le reflash entre chaque expérience HDC est requis pour remettre l'AM à zéro.

## Vérification

- [x] JSON créé avec 6 métriques obligatoires
- [x] `inference_latency_ms` < 2 ms (0.652 ms ✅)
- [x] Comparaison HDC / EWC / Mahalanobis sur Monitoring notée
