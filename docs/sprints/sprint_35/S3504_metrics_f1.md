# S3504 — Métrique F1 (classe `faulty`) PC + board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — sans F1, pas de heatmap F1 (objectif central du sprint) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | `src/evaluation/anomaly_metrics.py` ✅ (F1/précision/rappel), `src/evaluation/online_metrics.py` ✅, `src/evaluation/metrics.py` ✅ |
| **Fichiers cibles** | `src/evaluation/metrics.py`, `scripts/sensor_stream.py` |
| **Références** | Sprint 26 (`F1_MC=0.243` — pourquoi l'accuracy seule trompe) |

---

## Contexte

Les `results.json` existants exposent `acc_final` partout mais **pas F1 systématiquement**.
La détection de panne est déséquilibrée → la heatmap F1 est l'apport scientifique du sprint.
F1 doit être calculé **identiquement PC et board** pour permettre une comparaison honnête.

## Spec

- **PC** : dans le pipeline d'éval (`metrics.py` / `run_feature_condition_sweep.py`), calculer
  `f1_faulty` (classe positive = `faulty`) et `f1_macro` à partir des prédictions vs labels
  (réutiliser `anomaly_metrics.py` ; ne pas réimplémenter). Stocker dans `results.json`.
- **Board** : `sensor_stream.py` accumule déjà prédictions/labels par échantillon
  (cf. `--dump-samples`, Sprint 32). Ajouter le calcul `f1_faulty`/`f1_macro` côté hôte
  à partir du flux board, écrit dans le `results_*.json` board. **Aucun changement du protocole UART**
  (F1 calculé hôte, pas firmware).

```json
// ajout aux results.json PC et board
"f1_faulty": 0.71,
"f1_macro": 0.78,
"precision_faulty": ..., "recall_faulty": ...
```

**Règles** :
- F1 cohérent PC↔board (même définition de classe positive, même binarisation cmapss).
- Ne pas toucher au protocole UART (règle CLAUDE.md) — F1 est dérivé côté hôte.

## Implémentation (✅)

- **Définition unique partagée** : `src/evaluation/metrics.py::compute_fault_f1(y_true, y_pred)`
  → `{f1_faulty, f1_macro, precision_faulty, recall_faulty}` (sklearn, `pos_label=1`,
  `zero_division=0`). C'est **la** référence appelée PC **et** board.
- **PC** : `feature_conditions._eval_f1` délègue à `compute_fault_f1` ; `train_and_evaluate`
  expose `precision_faulty`/`recall_faulty` ; `run_feature_condition_sweep.run_cell` les stocke
  dans `results.json` (en plus de `f1_faulty`/`f1_macro` déjà présents).
- **Board** : `sensor_stream._compute_stats` calcule les 4 champs côté hôte depuis les
  `preds`/`trues` accumulés (propagation automatique dry-run/uart/cl-sequence) — **aucun
  changement du protocole UART**.
- **Tests** : `tests/test_feature_selection.py` → `test_compute_fault_f1_{perfect,known_case,monoclass_no_raise}`
  (3/3 PASS via `pytest -k f1`). Vérif sweep : `exp_S35_PC_best_ewc_cwru/results.json` contient
  bien les 4 champs ; dry-run board idem.

## Vérification

```bash
pytest tests/test_feature_selection.py -k f1 -v   # F1 cohérent sur cas connu
python scripts/run_feature_condition_sweep.py --condition 5feat --model ewc --dataset cwru
python -c "import json; r=json.load(open('experiments/exp_S35_PC_5feat_ewc_cwru/results.json')); assert 'f1_faulty' in r"
```
