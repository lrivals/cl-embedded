# S1911 — Expérience E19-01 : Mahalanobis 500 samples CWRU, auto-enregistré

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Complété sur board réel (2026-05-27) |
| **Durée estimée** | 3h |
| **Dépendances** | S1907 (recorder validé), S1908 (config) |
| **Fichiers cibles** | `experiments/exp_S19_01/` |

---

## Contexte

Première expérience board Sprint 19 : valider le détecteur Mahalanobis C sur 500 samples CWRU répartis en 3 tâches (domain-incremental), avec enregistrement automatique des métriques. Les résultats doivent être comparables à la Phase 1 Python (exp_100–111).

Cette expérience sert aussi à valider l'ensemble de la chaîne (S1901–S1907) avant de passer à l'EWC (S1912).

---

## Objectif

Produire `experiments/exp_S19_01/results.json` avec les 6 métriques obligatoires, comparer `acc_final` entre C et Python (delta < 2%), et documenter les résultats.

---

## Setup expérimental

### Dataset CWRU — segmentation 3 tâches

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Normal (0 HP, aucun défaut) | ~167 |
| Task 1 | Roulement extérieur 0.014" (léger) | ~167 |
| Task 2 | Roulement intérieur 0.014" (léger) | ~166 |

**Features** (5 dims après PCA/extraction, normalisées Z-score) :
- Variance temporelle, RMS, kurtosis, pic-à-pic, énergie spectrale

### Paramètres firmware

Depuis `configs/board_mahalanobis.yaml` :
- `threshold_init: 3.0` (distance Mahalanobis)
- `ema_alpha: 0.05` (adaptation en ligne)
- `n_tasks: 3`, `n_samples: 500`

### Référence Python Phase 1

Experiment(s) CWRU Mahalanobis : `experiments/exp_1XX/` (exp_100–105 selon naming Phase 1)
- Lire `results.json` Phase 1 pour récupérer `acc_final`, `avg_forgetting` de référence

---

## Procédure

### Dry-run (sans board)

```bash
python scripts/board_experiment_recorder.py \
    --model mahalanobis \
    --dataset cwru \
    --n-samples 500 \
    --n-tasks 3 \
    --dry-run \
    --output experiments/exp_S19_01
```

Vérifier : `experiments/exp_S19_01/results.json` créé avec 6 métriques.

### Avec board NUCLEO-F439ZI

```bash
# 1. Flasher le firmware
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

# 2. Lancer l'expérience
python scripts/board_experiment_recorder.py \
    --model mahalanobis \
    --dataset cwru \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_01 \
    --verbose
```

### Comparaison PC vs board

```python
import json, numpy as np

# Charger résultats Phase 1 (Python)
ref = json.load(open("experiments/exp_100/results.json"))  # adapter l'ID

# Charger résultats board
board = json.load(open("experiments/exp_S19_01/results.json"))

delta_acc = abs(ref["acc_final"] - board["acc_final"])
print(f"Delta acc_final: {delta_acc:.4f} (seuil: 0.02)")
assert delta_acc < 0.02, "Delta > 2% — vérifier preprocessing ou firmware"
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_01",
  "model": "mahalanobis",
  "dataset": "cwru",
  "platform": "nucleo_f439zi",
  "date": "2026-06-02",
  "acc_final": 0.94,
  "avg_forgetting": 0.02,
  "backward_transfer": -0.02,
  "ram_peak_bytes": 220,
  "inference_latency_ms": 0.003,
  "n_params": 30,
  "n_tasks": 3,
  "n_samples_total": 500,
  "config_snapshot": "configs/board_mahalanobis.yaml"
}
```

---

## Points de vigilance

### Preprocessing identique PC ↔ C

Le Z-score firmware (`normalize_zscore` dans `pipeline.c`) utilise `ZSCORE_MEAN` et `ZSCORE_STD` figés en Flash depuis `model_weights.h`. Ces constantes doivent être calculées sur le même split que la Phase 1.

Si les stats divergent → bias systématique sur les scores → delta acc élevé.

**Vérification** : comparer `model_weights.h:ZSCORE_MEAN[i]` avec `configs/board_mahalanobis.yaml` et les stats Phase 1.

### Latence

La latence Mahalanobis est dominée par le produit matrice-vecteur 5×5. Sur NUCLEO (Cortex-M4 @ 180 MHz) :
- Estimé : ~50 cycles × 25 multiplications = ~14 µs (< 100 ms contrainte projet)

### FIXME(gap2)

`ram_peak_bytes` dans le JSON est mesuré sur NUCLEO-F439ZI (192 Ko SRAM). Ce chiffre **n'est pas** la valeur cible STM32N6 (64 Ko). Il sert uniquement de validation intermédiaire.

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_01/results.json` | Créer (sortie expérience) |
| `experiments/exp_S19_01/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_01/run_log.txt` | Log verbose de la session |

---

## Résultats board réel (2026-05-27)

198 samples CWRU, 3 tâches, NUCLEO-F439ZI, protocol v3 :

| Métrique | Valeur | Seuil |
|----------|--------|-------|
| `inference_latency_ms` | **0.004 ms** | < 100 ms ✅ |
| `accuracy` globale | 68.7% | — |
| RAM build (statique) | 1 768 B | < 64 Ko ✅ |
| Erreurs CRC | 0/198 | ✅ |

> `acc=68.7%` sans `--update` : le seuil Mahalanobis n'est pas adapté. À relancer avec `--update` pour valider l'EMA online.

---

## Vérification

- [x] JSON créé avec 6 métriques (`experiments/exp_S19_01/results.json`)
- [x] `inference_latency_ms` < 1 ms ✅
- [x] `ram_peak_bytes` : 1 768 B ✅
- [ ] `acc_final` ≥ 0.85 : non atteint sans `--update` — relancer avec adaptation EMA
- [ ] Delta `acc_final` PC vs board < 2% : non mesuré (`compare_mahalanobis_pc_vs_board.py` à lancer)
- [ ] `FIXME(gap2)` : validation STM32N6 (64 Ko cible) requise

---

## Questions ouvertes

- `FIXME(gap2)` : La mesure `ram_peak_bytes` sur NUCLEO (192 Ko) n'est qu'indicative. La validation Gap 2 (< 64 Ko) requiert un run sur STM32N6 réel ou une simulation Cortex-M55.
- `TODO(arnaud)` : Quel experiment Phase 1 Python sert de référence pour la comparaison (quel exp_ID CWRU Mahalanobis) ?
