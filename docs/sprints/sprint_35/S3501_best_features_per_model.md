# S3501 — `scripts/select_best_features_per_model.py` + `configs/best_features/*.yaml`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — bloquant pour la condition `best` (S3503, S3508) |
| **Statut** | ✅ Implémenté (smoke validé) |
| **Durée estimée** | 4h |
| **Dépendances** | `src/evaluation/feature_importance.py` ✅ (`permutation_importance`) · loaders `src/data/` ✅ · modèles `src/models/` ✅ |
| **Fichiers cibles** | `scripts/select_best_features_per_model.py`, `configs/best_features/{model}_{dataset}.yaml` |
| **Références** | `configs/cmapss_feature_subset.yaml` (format subset existant, par dataset), `configs/cwru_feature_subset.yaml` |

---

## Contexte

Les sous-ensembles de features existants (`configs/*_feature_subset.yaml`) sont **par dataset**
(top-5 mutual-info), **pas par modèle**. Le sprint veut la condition `best` = **meilleures features
spécifiques à chaque modèle**, car un détecteur de distance (Mahalanobis) et un MLP (EWC) ne
valorisent pas les mêmes features.

## Spec

Pour chaque `(modèle ∈ {mahalanobis, ewc, tinyol, hdc}, dataset ∈ {cwru, monitoring, pronostia, cmapss, paderborn})` :

1. Charger le dataset (loader `src/data/`), split train/val (seed 42, `set_seed`).
2. Entraîner le modèle sur **toutes** les features natives.
3. Calculer la **permutation importance** par feature sur le set de validation
   (`src/evaluation/feature_importance.py:permutation_importance`), métrique de référence = **F1 classe `faulty`**.
4. Trier les features par importance décroissante.
5. **Balayer k** (k=1..n_features) : ré-entraîner sur le top-k, mesurer F1 val ;
   retenir le **k\*** maximisant F1 val (avec règle de parcimonie : plus petit k à <1% du max).
6. Écrire `configs/best_features/{model}_{dataset}.yaml`.

```yaml
# configs/best_features/{model}_{dataset}.yaml — généré par select_best_features_per_model.py
# NE PAS éditer manuellement.
model: ewc
dataset: cwru
method: permutation_importance
metric: f1_faulty
n_features_total: 9
n_features_selected: <k*>            # k optimisé sur F1 val
selected_indices: [...]
selected_features: [...]
val_f1_by_k: {1: ..., 2: ..., ...}   # courbe pour l'analyse S3512
fit_split: train (seed 42)
```

**Règles** :
- Sélection **fittée sur train/val uniquement** (pas de fuite test).
- Réutiliser `permutation_importance` — ne pas réimplémenter.
- Pas d'hyperparamètres en dur : seuils (parcimonie, n_repeats permutation) → CLI / config.

## Vérification

```bash
python scripts/select_best_features_per_model.py --model ewc --dataset cwru
ls configs/best_features/   # 20 fichiers (4 modèles × 5 datasets) après run complet
pytest tests/test_feature_selection.py -k best_features -v   # S3513
```
