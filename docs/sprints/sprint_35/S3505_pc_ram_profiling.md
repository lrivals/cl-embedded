# S3505 — RAM profiling des conditions de features (PC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🟡 Important — exigence CLAUDE.md (RAM profiling à chaque nouveau modèle/condition) |
| **Statut** | ✅ Implémenté (smoke) |
| **Durée estimée** | 2h |
| **Dépendances** | S3503 (modèles entraînés par condition), `scripts/profile_memory.py` ✅, `src/evaluation/memory_profiler.py` ✅ |
| **Fichiers cibles** | `experiments/exp_S35_PC_{condition}_{model}_{dataset}/ram.json` |

---

## Contexte

Changer le nombre de features change l'empreinte mémoire (poids d'entrée, projection HDC,
matrice `sigma_inv_` Mahalanobis). La règle CLAUDE.md impose un RAM profiling pour toute
nouvelle condition. Cela alimente aussi l'analyse coût/gain (S3512).

## Spec

Pour chaque `(condition, modèle, dataset)`, mesurer `ram_peak_bytes` via `tracemalloc`
(`profile_memory.py` / `memory_profiler.py`) → écrit dans `ram.json` à côté du `results.json`.

```json
{ "condition": "all", "model": "ewc", "dataset": "cmapss",
  "n_features": 21, "ram_peak_bytes": ..., "n_params": ... }
```

**Règles** : mesure réelle (tracemalloc), pas d'estimation analytique ; annotations `# MEM:`
conservées dans le code modèle.

## Vérification

```bash
python scripts/profile_memory.py --model ewc --dataset cmapss --condition all
ls experiments/exp_S35_PC_all_ewc_cmapss/ram.json
```

## Implémentation (✅ smoke)

- `scripts/profile_memory.py` : nouveau flag `--condition {5feat,all,best}` (+ `--seed`) et
  `ewc`/`tinyol` ajoutés aux choix `--model`. Dispatch `_run_condition_ram_profiling(...)`
  **avant** la validation des configs historiques.
- Mesure **réelle** (tracemalloc) réutilisée depuis `feature_conditions.train_and_evaluate`
  (la même qui alimente `ram_peak_bytes` du sweep S3503) — **pas d'estimation analytique**.
  Annotations `# MEM:` des modèles conservées.
- Sortie : `experiments/exp_S35_PC_{condition}_{model}_{dataset}/ram.json` =
  `{exp_id, condition, model, dataset, platform, sprint, n_features, ram_peak_bytes, n_params, measure}`.
- **Smoke produit** : `exp_S35_PC_best_ewc_cwru/ram.json` (k*=1, n_params=609) et
  `exp_S35_PC_all_mahalanobis_cwru/ram.json` (9 feat, n_params=90). Génération complète
  des 60 `ram.json` à lancer avec le sweep complet S3503 (cohérent avec l'état S3503).
