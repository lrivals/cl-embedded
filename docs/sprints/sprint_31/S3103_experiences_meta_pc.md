# S3103 / S3104 — Entraînement & expériences méta-modèle PC

| Champ | Valeur |
|-------|--------|
| **Sprint** | 31 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h (S3103) + 3h (S3104) |
| **Dépendances** | S3101 (`MetaLearner`) · Sprint 30 ✅ (paires + benchmark) |
| **Fichiers cibles** | `scripts/train_meta_learner.py` ✅, `configs/meta_stacking.yaml` ✅, `experiments/exp_S31_PC_*/` ✅ (14 runs + 1 skip honnête) |
| **Références** | `scripts/train_model_pair.py` (S3005), `src/training/scenarios.py` |

---

## Contexte

Démontre que l'arbitrage appris **bat ou égale** les meilleures alternatives du Sprint 30 (meilleur modèle individuel + meilleure règle d'ensemble). C'est la preuve de valeur du méta-modèle avant le portage board.

## S3103 — `scripts/train_meta_learner.py`

- Réutilise `ModelPair` + `run_cl_scenario_full` pour collecter les sorties des 2 modèles de base (split out-of-fold).
- Construit `meta_X`, entraîne le `MetaLearner`, évalue contre :
  - (a) chaque modèle individuel (Sprint 30) ;
  - (b) chaque règle d'ensemble fixe (OR/AND/soft-vote/weighted, Sprint 30).
- `results.json` :

```json
{
  "exp_id": "exp_S31_PC_maha_ewc_monitoring",
  "meta": {"kind": "logreg", "f1": null, "auroc": null},
  "baselines": {"model_a": null, "model_b": null, "ensemble_or": null, "ensemble_soft": null},
  "delta_vs_best_individual": null,
  "delta_vs_ensemble": null
}
```

## S3104 — Expériences

Un répertoire `experiments/exp_S31_PC_{pair}_{dataset}/` par run + `config_snapshot.yaml`. Couvrir les paires×datasets retenues du benchmark fixe.

---

## Résultats (✅)

14 runs produits + 1 skip honnête (`maha_hdc×paderborn` : `feature_bounds` non calibrés — aucun
chiffre inventé, règle CLAUDE.md). Chaque run entraîne **logreg + mlp** et retient `best_meta`.
Méthodologie de mesure (F1 seuil-optimal pour les modèles individuels, F1 seuil 0.5 pour ensemble
et méta) **identique au Sprint 30** → deltas comparables.

- **vs meilleure règle d'ensemble fixe** : le méta **≥ ensemble dans 12/14** runs (comparaison
  équitable seuil-0.5 ↔ seuil-0.5). Gains notables : `maha_tinyol×cwru` (+0.478),
  `maha_ewc×pronostia` (+0.227), `maha_tinyol×monitoring` (+0.213).
- **vs meilleur modèle individuel** : mixte (6/14 gains), car les baselines individuelles
  bénéficient d'un **seuil oracle** via `compute_anomaly_metrics` que le méta n'a pas. Gains :
  `maha_tinyol×cwru` (+0.089), `maha_ewc×paderborn` / `maha_tinyol×paderborn` (+0.078).
- `class_weight="balanced"` indispensable : sans lui, effondrement (F1=0) sur
  cmapss/pronostia déséquilibrés.

`results.json` enrichi : `meta:{logreg,mlp}`, `best_meta`, `baselines` (model_a/b + 4 règles),
`delta_vs_best_individual`, `delta_vs_ensemble`, bloc `oof`. `meta_weights.json` (poids du
`best_meta`) écrit pour S3105.

## Vérification

```bash
python scripts/train_meta_learner.py --config configs/meta_stacking.yaml --pair maha_ewc --dataset monitoring
ls experiments/exp_S31_PC_maha_ewc_monitoring/   # results.json + config_snapshot.yaml + meta_weights.json
```
