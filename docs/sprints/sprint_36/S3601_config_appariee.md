# S3601 — Configuration appariée PC ↔ board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — toute l'étude découle de cette config unique (datasets, conditions, protocoles, split, débit). Sans elle, paramètres dispersés/non reproductibles. |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 2h |
| **Dépendances** | `src/evaluation/feature_conditions.py` ✅ (`resolve_feature_indices`/`load_condition_arrays`) · `configs/pronostia_config.yaml` ✅ · `configs/board_ewc.yaml` ✅ · loaders `src/data/pronostia_dataset.py`, `src/data/monitoring_*` ✅ |
| **Fichiers cibles** | `configs/sprint36_ewc_comparison.yaml` |
| **Références** | Sprint 35 (conditions `5feat`/`all`) · règle CLAUDE.md « aucun hyperparamètre dans le code → configs/ » |

---

## Contexte

L'étude apparie **deux plateformes × deux datasets × deux conditions × deux protocoles**.
Pour rester reproductible et conforme à CLAUDE.md (« tout paramètre de taille/exécution a
une constante nommée en config »), tous ces axes sont déclarés dans **une seule config**,
consommée identiquement par les runs PC (S3602), board gelé (S3603) et board online (S3604).

## Spec

`configs/sprint36_ewc_comparison.yaml` — structure proposée :

```yaml
# Sprint 36 — comparaison appariée EWC PC↔board
model: ewc
seed: 42

datasets: [pronostia, monitoring]      # D4 class-incremental · D2 domain-incremental
conditions: [5feat, all]               # résolues via feature_conditions.resolve_feature_indices
protocols: [frozen, online]            # frozen = sans --update (parité) · online = --update (latence inf+MAJ)

# Échantillons appariés : split test/inférence COMPLET des deux côtés (pas de troncature)
n_inference: full                      # streamer tout le split test
match_train_inference: true            # mêmes indices train/test PC et board (load_condition_arrays, seed=42)

uart:
  rate_hz: 50
  proto: 3                             # réponse v3 (acc/auroc/forgetting embarqués)
  dump_samples: true                   # prédictions par échantillon → parité S3605

# Hyperparamètres EWC hérités de configs/board_ewc.yaml (ne PAS dupliquer ici)
ewc_base_config: configs/board_ewc.yaml
```

**Règles** :
- Les indices de features ne sont **pas** listés ici : ils sortent de
  `resolve_feature_indices(condition, "ewc", dataset)` (source unique, parité par construction).
- Aucun hyperparamètre EWC redéfini (référence `board_ewc.yaml`).
- `n_inference: full` ⇒ S3602/S3603/S3604 streament l'intégralité du split.

## Vérification

```bash
python -c "import yaml; c=yaml.safe_load(open('configs/sprint36_ewc_comparison.yaml')); \
assert c['datasets']==['pronostia','monitoring'] and c['conditions']==['5feat','all'] \
and c['protocols']==['frozen','online']; print('config OK')"

# Les indices de features se résolvent pour chaque (condition, dataset)
python -c "from src.evaluation.feature_conditions import resolve_feature_indices; \
print(resolve_feature_indices('all','ewc','pronostia')); \
print(resolve_feature_indices('5feat','ewc','monitoring'))"
```

## Implémentation (✅)

- [x] Écrire `configs/sprint36_ewc_comparison.yaml` (consommée par `run_sprint36_pc.py` + `run_sprint36_board.py`).
- [x] Vérifier que chaque `(condition, dataset)` résout des indices valides via `feature_conditions`.
- [x] Confirmer Pronostia=13 / Monitoring=4 features natives (condition `all`).

### Résultats (12 juin 2026)

- Config écrite + bloc `training:` ajouté (`n_tasks=3`, `epochs_per_task=15`, `batch_size=32`,
  `test_ratio=0.2`) pour apparier exactement l'entraînement board (`train_ewc_board`) → checkpoint
  PC réutilisé par le board ⇒ **parité exacte par construction**. LR/LAMBDA hérités de `board_ewc.yaml`.
- Indices résolus : Pronostia `all`=`[0..12]` (k=13) · `5feat`=`[1,2,4,8,12]` (k=5) ;
  Monitoring `all`=`5feat`=`[0,1,2,3]` (k=4).
- **Découverte** : pas de `monitoring_feature_subset.yaml` → **`5feat ≡ all` pour Monitoring**
  (4 features natives). Les deux cellules Monitoring sont donc identiques par construction
  (documenté, pas un bug). Seul Pronostia donne deux conditions distinctes.
