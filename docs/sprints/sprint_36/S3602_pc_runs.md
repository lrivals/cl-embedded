# S3602 — Runs PC de référence (EWC, Pronostia + Monitoring)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — la référence PC est le point de comparaison ; sans elle, aucune parité ni delta board↔PC. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 5h |
| **Dépendances** | S3601 ✅ · `scripts/train_ewc.py` ✅ · `src/models/ewc/ewc_mlp.py` ✅ (`EWCMlpMulticlass`) · `src/evaluation/feature_conditions.py` ✅ (`load_condition_arrays`) · `src/evaluation/metrics.py` ✅ (`compute_cl_metrics`, `compute_fault_f1`) · `src/evaluation/anomaly_metrics.py` ✅ (ROC-AUC) · `src/evaluation/memory_profiler.py` ✅ (RAM/latence PC) |
| **Fichiers cibles** | `scripts/train_ewc.py` (ou wrapper), `experiments/exp_S36_PC_{condition}_ewc_{dataset}/results.json` |
| **Références** | Sprint 26 (oubli catastrophique EWC — rapporter AF, pas seulement acc_final) · Sprint 35 S3503 (`run_feature_condition_sweep.py` comme modèle d'orchestration) |

---

## Contexte

La référence PC doit consommer **exactement** les mêmes colonnes et le même split que le
board (parité par construction). On réutilise `load_condition_arrays(dataset, condition,
"ewc", seed=42)` — la **source unique** introduite au Sprint 35 — et l'entraînement EWC
existant. Les prédictions par échantillon sont dumpées pour la comparaison S3605.

## Spec

Pour chaque `(condition ∈ {5feat, all}, dataset ∈ {pronostia, monitoring})` :

1. **Données** : `X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=42)` (split test complet, mêmes indices que board).
2. **Entraînement CL** : EWC séquentiel par tâche (domain-incremental Monitoring ; class-incremental Pronostia), hyperparamètres ← `configs/board_ewc.yaml`.
3. **Métriques** stockées dans `results.json` :
   - `acc_matrix` (T×T) → `compute_cl_metrics` → `aa`, `af` (oubli), `bwt`, `acc_final`.
   - `f1_faulty`, `f1_macro`, `precision_faulty`, `recall_faulty` via `compute_fault_f1`.
   - `roc_auc` via `anomaly_metrics`.
   - `n_params`, `ram_peak_bytes` (tracemalloc), `inference_latency_ms` (PC).
   - `per_task_acc`.
4. **Dump prédictions** par échantillon → `samples` : `[{idx, true, pred, confidence, features[k]}]` (pour parité S3605).

```json
// exp_S36_PC_all_ewc_pronostia/results.json (extrait)
{
  "exp_id": "exp_S36_PC_all_ewc_pronostia",
  "platform": "pc", "model": "ewc", "dataset": "pronostia",
  "condition": "all", "n_features": 13,
  "acc_matrix": [[...], ...],
  "aa": null, "af": null, "bwt": null, "acc_final": null,
  "f1_faulty": null, "f1_macro": null, "roc_auc": null,
  "n_params": null, "ram_peak_bytes": null, "inference_latency_ms": null,
  "per_task_acc": {}, "samples": []
}
```

**Règles** :
- Valeurs `null`/« à mesurer » tant que non exécuté (aucun chiffre inventé).
- Hyperparamètres EWC **jamais** dans le code → `board_ewc.yaml` (CLAUDE.md).
- Même seed (42) et même `load_condition_arrays` que le board ⇒ split identique.

## Vérification

```bash
# Un run smoke par condition
python scripts/train_ewc.py --config configs/sprint36_ewc_comparison.yaml \
  --condition all --dataset pronostia            # → exp_S36_PC_all_ewc_pronostia/

python -c "import json; r=json.load(open('experiments/exp_S36_PC_all_ewc_pronostia/results.json')); \
assert {'acc_final','af','f1_faulty','roc_auc','ram_peak_bytes','samples'} <= set(r); print('PC results OK')"
```

## Implémentation (✅)

- [x] Driver dédié `scripts/run_sprint36_pc.py` (plutôt qu'un patch de `train_ewc.py`, qui fait des
      scénarios CL natifs sans `load_condition_arrays` → inadapté à la parité). Consomme
      `sprint36_ewc_comparison.yaml` + `--condition/--dataset`.
- [x] `load_condition_arrays(dataset, condition, "ewc", seed=42)` branché (split identique board).
- [x] 4 `results.json` produits (2 conditions × 2 datasets) avec AA/AF/BWT/F1/ROC-AUC + dump `samples`.
- [x] **AF rapporté** explicitement (lien Sprint 26).

### Décision de conception

Référence PC = **miroir board** : `EWCMlpMulticlass(k,2,[32,16])` entraîné sur un **split temporel
3 tâches** (identique à `run_feature_condition_board.train_ewc_board`), LR/LAMBDA ← `board_ewc.yaml`.
Le checkpoint PC `exp_S36_PC_*/checkpoints/ewc_head.pt` est **réutilisé tel quel** par le board en
S3603 → un seul modèle, parité exacte par construction. `acc_matrix[T×T]` bâtie en évaluant les
tâches vues (split test 80/20 par tâche) après chaque tâche.

### Résultats (12 juin 2026) — `experiments/exp_S36_PC_{cond}_ewc_{ds}/`

| Cellule | k | AA | AF | BWT | F1_faulty | ROC-AUC | n_params | lat_inf |
|---------|---|----|----|-----|-----------|---------|----------|---------|
| 5feat·pronostia | 5 | 0.989 | 0.010 | −0.001 | 0.916 | 0.995 | 1010 | 0.036 ms |
| all·pronostia | 13 | 0.983 | 0.005 | −0.001 | 0.918 | 0.997 | 1266 | — |
| 5feat·monitoring | 4 | 0.979 | 0.000 | +0.001 | 0.919 | 0.988 | 978 | — |
| all·monitoring | 4 | 0.979 | 0.000 | +0.001 | 0.919 | 0.988 | 978 | — |

- AF faible partout (init aléatoire → CL court 3 tâches ; pas d'oubli catastrophique sur ces splits
  domain/class-incremental, contrairement au scénario Sprint 26).
- Monitoring `5feat`==`all` identiques (4 features natives, cf. S3601).
- `samples` dumpés (idx/true/pred/confidence/features) sur le split complet pour la parité S3603/S3604.
