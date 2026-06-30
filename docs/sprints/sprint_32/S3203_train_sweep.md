# S3203 — Orchestrateur d'entraînement du balayage

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Dépendances** | S3202 (configs/sweep/) |
| **Résultat** | `scripts/run_threshold_sweep.py` (config **fusionnée** base modèle×dataset ⊕ seuil par run → contourne le re-read YAML CMAPSS/Maha ; échecs isolés ; fusion résumé par `exp_id`). **60/60 runs OK** (`experiments/exp_S32_sweep_summary.json`). Prérequis découverts & traités : Battery n'était câblé dans **aucun** train script → branche `battery_rul` ajoutée aux 4 + `configs/battery_normalizer.yaml` généré + 3 configs Battery par modèle créés (`{hdc,tinyol,mahalanobis}_battery_config.yaml`). Fix `tinyol_battery` : `oto_head.input_dim = embedding+1 = 5` (pas n_features). |
| **Fichiers cibles** | `scripts/run_threshold_sweep.py`, `configs/*battery*.yaml`, `experiments/exp_S32_*/` |
| **Références** | `scripts/train_{mahalanobis,hdc,ewc,tinyol}.py`, `configs/cmapss_config.yaml` (archi de référence) |

---

## Contexte

Boucler `{modèle} × {dataset} × {seuil}` en réutilisant les scripts d'entraînement existants (jamais réimplémenter la boucle CL). 4 modèles × 3 datasets × 5 seuils = jusqu'à 60 runs PC.

---

## Spec

```python
# scripts/run_threshold_sweep.py
MODELS   = ["mahalanobis", "hdc", "ewc", "tinyol"]
DATASETS = ["cmapss", "pronostia", "battery"]
# CLI : --models --datasets --thresholds (sous-ensembles pour smoke-test)

# pour chaque (model, dataset, thr) :
#   config = f"configs/sweep/{dataset}_thr{thr}.yaml"
#   subprocess: python scripts/train_{model}.py --config {config} --profile_memory \
#               --exp_id exp_S32_{model}_{dataset}_thr{thr}
#   -> experiments/exp_S32_{model}_{dataset}_thr{thr}/ (snapshot + results/)
```

- **Battery** : si une combinaison modèle×Battery n'a pas de config existante, la créer par analogie aux configs CMAPSS (même archi, adapter `input_dim`/loader). Ne pas inventer d'hyperparamètres exotiques.
- `--profile_memory` obligatoire (RAM/latence par run, cf. S3204).
- Échecs isolés : un run qui échoue ne stoppe pas le balayage (log + continue).

---

## Vérification

```bash
# smoke-test 1 combinaison
python scripts/run_threshold_sweep.py --models ewc --datasets cmapss --thresholds 30 --profile_memory
ls experiments/exp_S32_ewc_cmapss_thr30/results/   # metrics_*.json + memory_report.json présents
```
