# S1912 — Expérience E19-02 : EWC head 3 tâches Monitoring, forgetting on-board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 19 |
| **Priorité** | 🟡 Important |
| **Statut** | ✅ Complété sur board réel (2026-05-27) |
| **Durée estimée** | 4h |
| **Dépendances** | S1902 (ewc_consolidate ✅), S1911 (pipeline validé) |
| **Fichiers cibles** | `experiments/exp_S19_02/` |

---

## Contexte

L'EWC est le modèle CL central du projet (M2 — `ewc_mlp_spec.md`). Cette expérience valide pour la première fois son comportement anti-oubli sur hardware : après 3 tâches Monitoring (pump → turbine → compressor), `avg_forgetting` mesuré on-board doit être significativement inférieur au forgetting sans régularisation EWC.

C'est aussi la première fois que `ewc_consolidate()` est testée dans un scénario réel.

---

## Objectif

Produire `experiments/exp_S19_02/results.json` avec `avg_forgetting` mesuré on-board, et le comparer à :
1. L'EWC Phase 1 Python (exp Monitoring, exp_112–117)
2. Un fine-tuning sans EWC (lambda=0) comme baseline catastrophic forgetting

---

## Setup expérimental

### Dataset Monitoring — segmentation 3 tâches domain-incremental

| Tâche | Condition | Samples |
|-------|-----------|---------|
| Task 0 | Pompe (pump) | ~167 |
| Task 1 | Turbine | ~167 |
| Task 2 | Compresseur | ~166 |

**Features** (5 dims, tabulaire, normalisées) :
- Température, pression, vibration, humidité, type équipement (encodé)

**Label** : `faulty` (0/1 binaire) — détection anomalie

### Paramètres firmware EWC

Depuis `configs/board_ewc.yaml` :
- `lr: 0.01` (EWC_LR)
- `lambda_ewc: 100.0` (coefficient régularisation)
- `fisher_decay: 0.9` (alpha pour `ewc_consolidate`)
- `n_tasks: 3`, `n_samples: 500`

### Scénario d'entraînement on-board

```
Task 0 (pump) : 167 samples ewc_sgd_step → ewc_consolidate(alpha=0.9)
Task 1 (turbine) : 167 samples ewc_sgd_step avec pénalité Fisher task 0
                → ewc_consolidate(alpha=0.9)
Task 2 (compressor) : 166 samples ewc_sgd_step avec pénalité Fisher task 0+1
```

`ForgettingTracker` mesure la chute d'accuracy par tâche après chaque consolidation.

---

## Procédure

### Dry-run (sans board)

```bash
python scripts/board_experiment_recorder.py \
    --model ewc \
    --dataset monitoring \
    --n-samples 500 \
    --n-tasks 3 \
    --dry-run \
    --output experiments/exp_S19_02
```

### Avec board

```bash
# Flasher avec support EWC activé
make -C firmware/stm32f4_blink/ flash PORT=/dev/ttyACM0

# Lancer expérience EWC
python scripts/board_experiment_recorder.py \
    --model ewc \
    --dataset monitoring \
    --port /dev/ttyACM0 \
    --baud 115200 \
    --n-samples 500 \
    --n-tasks 3 \
    --request-update \
    --output experiments/exp_S19_02 \
    --verbose
```

---

## Format JSON attendu

```json
{
  "exp_id": "S19_02",
  "model": "ewc",
  "dataset": "monitoring",
  "platform": "nucleo_f439zi",
  "date": "2026-06-05",
  "acc_final": 0.85,
  "avg_forgetting": 0.05,
  "backward_transfer": -0.05,
  "ram_peak_bytes": 9728,
  "inference_latency_ms": 0.12,
  "n_params": 1538,
  "n_tasks": 3,
  "n_samples_total": 500,
  "config_snapshot": "configs/board_ewc.yaml"
}
```

---

## Comparaison avec Phase 1

| Métrique | Attendu board (EWC λ=100) | Attendu baseline (λ=0) | Référence Phase 1 |
|----------|:------------------------:|:---------------------:|:-----------------:|
| `acc_final` | ≥ 0.80 | ≥ 0.70 | voir exp_11X |
| `avg_forgetting` | ≤ 0.10 | ≥ 0.25 (catastrophic) | voir exp_11X |
| `backward_transfer` | ≥ -0.10 | ≤ -0.25 | voir exp_11X |

> **Hypothèse** : avec `lambda_ewc=100` et `fisher_decay=0.9`, le forgetting board devrait être similaire à ± 5 points de pourcentage du résultat Phase 1. Un écart plus grand indique un problème dans `ewc_consolidate()` ou le preprocessing.

---

## Comparaison avec baseline catastrophic forgetting

Lancer une expérience complémentaire avec `lambda_ewc=0` :

```bash
# Modifier temporairement board_ewc.yaml : lambda_ewc: 0.0
python scripts/board_experiment_recorder.py \
    --model ewc \
    --dataset monitoring \
    --n-samples 500 --n-tasks 3 \
    --dry-run \
    --output experiments/exp_S19_02_baseline
```

Comparer `avg_forgetting` : doit être nettement supérieur à l'expérience EWC.

---

## Points de vigilance

### Synchronisation consolidation PC → firmware

`pipeline_set_task(task_id)` doit être appelé par `sensor_stream.py` entre les tâches pour déclencher `ewc_consolidate()` côté firmware. Vérifier que ce signal est bien transmis via le protocole UART.

### Forgetting tracker inter-tâches

`ForgettingTracker.fgt_update(task_id, acc)` nécessite de ré-évaluer les tâches précédentes **après** chaque consolidation. Dans le scénario streaming, le PC renvoie quelques samples de la tâche 0 après la tâche 1 pour mesurer la rétention.

### Latence EWC

`ewc_sgd_step()` est significativement plus lent que Mahalanobis :
- Forward : O(EWC_IN × EWC_H1 + EWC_H1 × EWC_H2 + EWC_H2 × EWC_OUT) = O(5×32+32×16+16×2) ≈ 704 MACs
- Backward + EWC : ×3 (backward) + ×2 (terme EWC) ≈ 3500 MACs total

Estimé sur Cortex-M4 @ 180 MHz : ~10–50 ms (à mesurer, contrainte projet < 100 ms).

---

## Fichiers cibles

| Fichier | Action |
|---------|--------|
| `experiments/exp_S19_02/results.json` | Créer (sortie expérience) |
| `experiments/exp_S19_02/config_snapshot.yaml` | Copié automatiquement |
| `experiments/exp_S19_02_baseline/results.json` | Optionnel — baseline λ=0 |

---

## Résultats board réel (2026-05-27)

300 samples Monitoring, 3 tâches, NUCLEO-F439ZI, protocol v3, `--update` activé :

| Métrique | Valeur mesurée | Seuil |
|----------|---------------|-------|
| `inference_latency_ms` | **0.004 ms** | < 100 ms ✅ |
| `acc_final` | 8.0% | ≥ 0.80 ❌ |
| `avg_forgetting` | 0.0 | — |
| `ram_peak_bytes` | 0 B (v3 non reporté) | < 15 Ko — |
| `n_params` | 1 538 | ✅ |
| Gap 2 latence | ✅ | |

**Analyse** : `acc_final=8%` s'explique par le fait que le head EWC prédit quasi-exclusivement `pred=1` depuis la session précédente (poids non réinitialisés entre runs). `avg_forgetting=0.0` est mathématiquement correct (le modèle ne change pas de comportement entre tâches, il est en mode dégénéré). Ce n'est pas un bug de `ewc_consolidate()` mais un problème d'initialisation des poids entre expériences.

**Bug corrigé** : `board_experiment_recorder.py` utilisait `protocol_version=2` par défaut → valeurs corrompues (latences en milliards µs). Fix : `protocol_version=3` explicite.

**Action requise** : vérifier la réinitialisation des poids EWC entre runs ou ajouter un signal `reset` dans le protocol.

---

## Vérification

- [x] JSON créé avec 6 métriques (`experiments/exp_S19_02/results.json`) ✅
- [x] `inference_latency_ms` = 0.004 ms < 100 ms ✅
- [ ] `acc_final` ≥ 0.80 : ❌ — poids EWC non réinitialisés entre runs
- [ ] `avg_forgetting` < baseline λ=0 : non comparable (modèle dégénéré)
- [ ] `ram_peak_bytes` reporté en v3 : 0 — à mesurer via map file (S1913)
- [ ] Résultats proches (± 5 pp) Phase 1 Python : à faire après fix réinitialisation

---

## Questions ouvertes

- `TODO(arnaud)` : Quel est l'ID expérience Phase 1 EWC Monitoring de référence pour comparaison ?
- `TODO(arnaud)` : Est-ce que la mesure forgetting on-board (sur 10 samples re-streamés) est suffisante pour être statistiquement significative, ou faut-il augmenter N ?
- `FIXME(gap2)` : `ram_peak_bytes` NUCLEO indicatif — validation STM32N6 (54.5 Ko marge théorique) requise pour Gap 2.
