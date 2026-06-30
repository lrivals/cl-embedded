# S3503 — `scripts/run_feature_condition_sweep.py` (re-run PC, 3 conditions)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — produit les données PC de toutes les heatmaps |
| **Statut** | ✅ Implémenté (smoke validé) |
| **Durée estimée** | 5h |
| **Dépendances** | S3501 (`configs/best_features/*`), S3502 (`configs/all_features/*`), S3504 (F1), subsets 5-feat existants ✅ |
| **Fichiers cibles** | `scripts/run_feature_condition_sweep.py`, `experiments/exp_S35_PC_{condition}_{model}_{dataset}/results.json` |
| **Références** | `experiments/exp_S33_PC_{model}_{dataset}/results.json` (format de sortie existant) |

---

## Contexte

Re-lancer **toutes** les expériences de fault detection sur PC, pour les 3 conditions de features,
en produisant **F1 ET acc_final** (S3504). Les runs S33 PC existants servent de référence de format
mais ne couvrent ni les 3 conditions ni F1 systématique.

## Spec

Driver paramétré balayant `condition × modèle × dataset` :

- `condition ∈ {5feat, all, best}` :
  - `5feat` → `configs/*_feature_subset.yaml` (existant)
  - `all`   → `configs/all_features/{dataset}.yaml` (S3502)
  - `best`  → `configs/best_features/{model}_{dataset}.yaml` (S3501)
- `model ∈ {mahalanobis, ewc, tinyol, hdc}`
- `dataset ∈ {cwru, monitoring, pronostia, cmapss, paderborn}` (cmapss binarisé au seuil de réf. — `TODO(arnaud)`)

Chaque cellule → `experiments/exp_S35_PC_{condition}_{model}_{dataset}/results.json` :

```json
{
  "exp_id": "exp_S35_PC_best_ewc_cwru",
  "condition": "best", "model": "ewc", "dataset": "cwru", "platform": "pc",
  "n_features": <k*>,
  "acc_final": ..., "f1_faulty": ..., "f1_macro": ...,
  "avg_forgetting": ..., "ram_peak_bytes": ..., "n_params": ...
}
```

**Règles** :
- `set_seed(42)`, `config_snapshot.yaml` par expérience (reproductibilité CLAUDE.md).
- Pas de résultats hardcodés — tout sort du run.
- Réutiliser les boucles CL existantes (`src/training/`) et les loaders ; ne pas dupliquer.
- `--dry-run` pour valider la matrice sans entraîner.

## Vérification

```bash
python scripts/run_feature_condition_sweep.py --dry-run        # liste 3×4×5 = 60 cellules
python scripts/run_feature_condition_sweep.py --condition best --model ewc --dataset cwru
ls experiments/ | grep exp_S35_PC_   # 60 dossiers après run complet
```
