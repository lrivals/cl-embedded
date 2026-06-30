# S3005 / S3006 — Entraînement & expériences PC Partie A (binarisé)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🔴 Critique (PRIORITAIRE du sprint) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h (S3005) + 4h (S3006) |
| **Dépendances** | S3001 (`ModelPair`) · S3003 (désaccord) · S3002 (configs) · loaders 5 datasets ✅ |
| **Fichiers cibles** | `scripts/train_model_pair.py`, `experiments/exp_S30_PC_*/` |
| **Références** | `scripts/train_ewc.py`, `scripts/train_mahalanobis.py`, `src/training/scenarios.py` |

---

## Contexte

Cœur du benchmark fixe. Pour chaque **paire × dataset** (cadre **binarisé normal-vs-fault**) : entraîner les 2 modèles, évaluer entraînement ET inférence, et **distinguer explicitement** les métriques individuelles des métriques d'ensemble + désaccord.

---

## S3005 — `scripts/train_model_pair.py`

- Charge la config paire (`configs/board_pair_*.yaml`) + le loader dataset.
- Entraîne Mahalanobis (non-supervisé) et le modèle supervisé via la boucle CL (`run_cl_scenario_full`).
- Évalue :
  - **Entraînement CL** : AA / AF / BWT par modèle (`compute_cl_metrics`).
  - **Inférence** : AUROC, F1, précision/rappel par modèle **et** pour l'ensemble (`predict_ensemble`).
  - **Désaccord** : `disagreement_rate`, `cohen_kappa`, `disagreement_confusion`, origine.
- Sort un `results.json` structuré :

```json
{
  "exp_id": "exp_S30_PC_maha_ewc_monitoring",
  "pair": "maha_ewc", "dataset": "monitoring", "frame": "binary",
  "model_a": {"name": "mahalanobis", "auroc": null, "f1": null, "af": null},
  "model_b": {"name": "ewc",        "auroc": null, "f1": null, "af": null},
  "ensemble": {"rule": "or", "auroc": null, "f1": null},
  "disagreement": {"rate": null, "kappa": null, "a_correct": null,
                   "b_correct": null, "both_wrong": null, "origin": {}}
}
```

> Champs `null` = « à mesurer » : remplis par l'exécution, jamais à la main.

## S3006 — Expériences (3 paires × 5 datasets = 15)

Paires : `maha_hdc`, `maha_ewc`, `maha_tinyol`. Datasets : Pronostia, Monitoring, CWRU, CMAPSS, Paderborn. Un répertoire `experiments/exp_S30_PC_{pair}_{dataset}/` par run avec `config_snapshot.yaml` + `results.json`.

---

## Vérification

```bash
python scripts/train_model_pair.py --config configs/board_pair_maha_ewc.yaml --dataset monitoring
ls experiments/exp_S30_PC_maha_ewc_monitoring/   # config_snapshot.yaml + results.json
```

---

## Bilan (S3005 + S3006 ✅)

`scripts/train_model_pair.py` implémenté. Architecture : 3 fines couches d'adaptation
réutilisant l'existant — Mahalanobis (`fit_task`), supervisé EWC (`train_ewc.train_ewc`),
HDC (`run_cl_scenario_full`), TinyOL (`TinyOLAnomalyDetector` BaseCLModel). Les configs de
paires (`configs/board_pair_{maha_ewc,maha_hdc,maha_tinyol}.yaml`, S3002 ✅) référencent les
configs supervisées par dataset déjà validées au Sprint 28 (`{model}_int8_{dataset}.yaml`) :
input_dim / feature_bounds / chemins corrects, pas de duplication.

`results.json` produit avec sections **distinctes** `model_a` (Mahalanobis : AA/AF/BWT +
AUROC/F1/précision/rappel), `model_b` (supervisé : idem), `ensemble.by_rule` (**4 règles**
`or`/`and`/`soft_vote`/`weighted` + `best_rule` = max F1) et `disagreement` (taux, kappa,
qui-a-raison, origine features/score Mahalanobis).

**S3006 — 15 runs Partie A produits** (`experiments/exp_S30_PC_{pair}_{dataset}/`) : 14 mesurés
et 1 N/A honnête (maha_hdc×paderborn : `feature_bounds` non calibrés, déjà N/A au Sprint 28 →
artefact `status: skipped`, aucun chiffre inventé).

| Paire×dataset | A=maha AUROC/F1 | B=sup AUROC/F1 | best_rule (F1) | désaccord |
|---------------|-----------------|----------------|----------------|-----------|
| maha_ewc · monitoring | 0.971 / 0.891 | 0.974 / 0.816 | or (0.905) | 0.033 |
| maha_ewc · cwru | 0.560 / 0.409 | **0.999 / 0.998** | or (0.995) | 0.684 |
| maha_ewc · pronostia | 0.752 / 0.388 | 0.997 / 0.880 | and (0.721) | 0.210 |
| maha_ewc · cmapss | 0.550 / 0.303 | 0.779 / 0.473 | and (0.452) | 0.310 |
| maha_ewc · paderborn | 0.498 / 0.734 | 0.496 / 0.001 | or (0.800) | 0.951 |
| maha_hdc · monitoring | 0.971 / 0.891 | 0.860 / 0.565 | and (0.661) | 0.153 |
| maha_hdc · cwru | 0.560 / 0.409 | 0.984 / 0.984 | soft_vote (0.984) | 0.665 |
| maha_hdc · pronostia | 0.752 / 0.388 | 0.810 / 0.524 | soft_vote (0.523) | 0.165 |
| maha_hdc · cmapss | 0.550 / 0.303 | 0.500 / 0.000 | or (0.224) | 0.358 |
| maha_hdc · paderborn | — | — | N/A (skipped) | — |
| maha_tinyol · monitoring | 0.971 / 0.891 | 0.777 / 0.583 | or (0.689) | 0.052 |
| maha_tinyol · cwru | 0.560 / 0.409 | 0.756 / 0.690 | or (0.451) | 0.145 |
| maha_tinyol · pronostia | 0.752 / 0.388 | 0.675 / 0.360 | soft_vote (0.350) | 0.099 |
| maha_tinyol · cmapss | 0.550 / 0.303 | 0.501 / 0.208 | soft_vote (0.224) | 0.342 |
| maha_tinyol · paderborn | 0.498 / 0.734 | 0.494 / 0.000 | or (0.751) | 0.813 |

**Lecture** (TODO(arnaud) règle de fusion de référence) : pas de règle unique gagnante —
`or` (priorité anomalie) maximise le F1 d'ensemble sur 8/14 runs, `soft_vote`/`and` sur les
autres. Complémentarité nette là où les forces divergent : EWC/HDC dominent en classification
de panne (cwru/pronostia, F1 sup ≫ maha) tandis que Mahalanobis domine la détection générique
(monitoring/paderborn) ; le désaccord est faible quand les 2 convergent (monitoring 0.03–0.15)
et explose quand un seul est fiable (paderborn 0.81–0.95). Détail par échantillon (origine du
désaccord) dans chaque `results.json` → `disagreement.origin`.
