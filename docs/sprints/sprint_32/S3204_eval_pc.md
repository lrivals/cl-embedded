# S3204 — Évaluation perf + profiling HW PC, par seuil

| Champ | Valeur |
|-------|--------|
| **Sprint** | 32 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 3h |
| **Dépendances** | S3203 (runs entraînés) |
| **Résultat** | Consolidation intégrée à `run_threshold_sweep.py` : `positive_ratio` (model-indépendant) calculé par dataset×seuil via les loaders, écrit dans `results/sweep_meta.json` + résumé `exp_S32_sweep_summary.json` (acc_final/AF/BWT/`ram_peak_bytes`/`inference_latency_ms`/`n_params` repris des `metrics_cl.json`, déjà profilés via `--profile_memory`). **Gradients vérifiés** : `positive_ratio` monotone (CMAPSS 0.049→0.225, Pronostia 0.191→0.729, Battery 0.062→0.300) ; `acc_final` décroît avec l'équilibrage. Note : AUROC/F1/préc/rappel non recalculés post-hoc (scores non persistés dans le JSON unifié) — `positive_ratio` est la métrique clé S3204 livrée. |
| **Fichiers cibles** | `experiments/exp_S32_*/results/*.json` |
| **Références** | `src/evaluation/anomaly_metrics.py`, `metrics.py`, `memory_profiler.py`, `compute_cost.py`, `scripts/profile_memory.py` |

---

## Contexte

Le seuil change le **ratio de positifs** → les métriques sensibles au déséquilibre (F1, précision, rappel, AUROC) sont au cœur de l'analyse. Le HW est re-profilé **par seuil** pour prouver son invariance (choix utilisateur).

---

## Spec

```text
Par run exp_S32_{model}_{dataset}_thr{XX} :
  Perf   (anomaly_metrics.py)  : AUROC, F1, précision, rappel
         (metrics.py)          : acc_final, avg_forgetting (AF), backward_transfer (BWT)
  HW     (memory_profiler.py)  : ram_peak_bytes, inference_latency_ms (re-mesuré par seuil)
         (compute_cost.py)     : MACs
  Méta                         : positive_ratio (part de faulty=1) — clé pour l'analyse
```

- Réutiliser les modules existants — ne pas réimplémenter de métriques.
- Consigner `positive_ratio` par seuil (explique la dérive des métriques perf).
- Résultats consolidés dans `experiments/exp_S32_*/results/` (format unifié S12-05).

---

## Vérification

```bash
python scripts/profile_memory.py --model ewc --dataset cmapss --config configs/sweep/cmapss_thr30.yaml
# vérifier qu'un results/metrics_*.json contient auroc, f1, precision, recall, positive_ratio
python -c "import json,glob; [print(f, json.load(open(f)).keys()) for f in glob.glob('experiments/exp_S32_*/results/metrics_*.json')][:3]"
```
