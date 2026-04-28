# Skill : Amélioration incrémentale des modèles non supervisés (v2)

> **Déclencheur** : "améliore KMeans avec EMA" / "active le Welford sur Mahalanobis" /
> "lance les expériences v2 comparatives"

---

## 1. Contexte — État v1 et limitations

### KMeans (M4a) — limitation du seuil figé

Le seuil de détection est calculé une seule fois sur Task 0 (percentile 95 des distances)
et reste constant pour toutes les tâches suivantes. En scénario domain-incremental, le
shift de distribution inter-domaines fait dériver ce seuil :
- Résultats v1 CWRU by_fault_type : AA=0.152, AF=0.019 (exp_077)
- v2 per-task threshold sans mémoire : AA=0.273, AF=0.208 (exp_090) — instable

**Cause** : recalcul brutal à chaque tâche sans mémoire de la distribution précédente.

### Mahalanobis (M6) — recalcul batch redondant

`fit_task()` recalcule μ et Σ⁻¹ en batch complet (numpy) à chaque nouvelle tâche.
La méthode `partial_fit()` avec l'algorithme de Welford est déjà implémentée dans
[mahalanobis_detector.py:150](../src/models/unsupervised/mahalanobis_detector.py#L150)
mais n'est pas activée dans la boucle CL officielle (cl_strategy="refit" uniquement).

**Conséquence** : impossibilité d'accumuler la connaissance inter-tâches sans stocker
toutes les données brutes.

---

## 2. KMeans — Spec EMA threshold

### Paramètre à ajouter

Dans `configs/unsupervised_config.yaml`, sous la clé `kmeans:` :
```yaml
ema_alpha: 0.3        # poids de la tâche courante dans la MAJ EMA du seuil
                      # 0.0 = seuil figé (comportement v1), 1.0 = refit complet
```

### Constante dans le fichier source

```python
EMA_ALPHA_DEFAULT: float = 0.3
```

### Logique dans `fit_task()`

```
Task 0  → threshold_ = percentile(distances_T0, anomaly_percentile)   # identique v1
Task > 0 :
    percentile_new = percentile(distances_current, anomaly_percentile)
    threshold_     = ema_alpha * percentile_new + (1 - ema_alpha) * threshold_
```

Le `percentile_new` est calculé sur les scores de la tâche courante (données normales).

### Attributs à ajouter

```python
self.ema_alpha: float           # lu depuis config
self.threshold_history_: list[float]  # [seuil_T0, seuil_T1, ...]
```

Annotation mémoire à placer sur la ligne de mise à jour :
```python
self.threshold_ = ...  # MEM: 4 B @ FP32 / 4 B @ INT8 (scalaire EMA)
```

### Champ de métriques JSON

```json
"threshold_history": [3.88, 4.12, 5.60]
```

---

## 3. Mahalanobis — Spec Welford CL

### `partial_fit()` — déjà implémenté

Méthode disponible à
[mahalanobis_detector.py:150](../src/models/unsupervised/mahalanobis_detector.py#L150).
Elle met à jour `mu_`, `_M2_`, `sigma_inv_` sans stocker les données brutes.

### Nouveau `cl_strategy` : `"welford"`

Dans `fit_task()`, conditionner le comportement sur `self.cl_strategy` :

```
"refit"   → comportement actuel (reset + recalcul batch)
"welford" → NE PAS réinitialiser _n_seen_, mu_, _M2_
             appeler partial_fit(X) sur les données de la tâche courante
             threshold_ inchangé (fixé sur Task 0, jamais modifié)
```

Dans `configs/unsupervised_config.yaml` :
```yaml
mahalanobis:
  cl_strategy: "welford"   # v2 — accumulation incrémentale inter-tâches
```

### Attribut de tracking

```python
self.welford_updates_per_task_: list[int]  # [n_samples_T0, n_samples_T1, ...]
```

### Champ de métriques JSON

```json
"welford_updates_per_task": [412, 412, 413]
```

---

## 4. Expériences v2 à lancer

Dernier exp existant : **exp_093** (DBSCAN CWRU by_severity v2).

| Exp | Modèle | Dataset | Scénario | Amélioration |
|-----|--------|---------|----------|--------------|
| exp_094 | KMeans EMA | CWRU | by_fault_type | `ema_alpha=0.3` |
| exp_095 | KMeans EMA | CWRU | by_severity | `ema_alpha=0.3` |
| exp_096 | KMeans EMA | Pronostia | by_condition | `ema_alpha=0.3` |
| exp_097 | Mahalanobis Welford | CWRU | by_fault_type | `cl_strategy=welford` |
| exp_098 | Mahalanobis Welford | CWRU | by_severity | `cl_strategy=welford` |
| exp_099 | Mahalanobis Welford | Pronostia | by_condition | `cl_strategy=welford` |

Scripts à utiliser (existants) :
```bash
python scripts/train_kmeans.py      --config configs/cwru_by_fault_config.yaml   --exp_id exp_094
python scripts/train_kmeans.py      --config configs/cwru_by_severity_config.yaml --exp_id exp_095
python scripts/train_kmeans.py      --config configs/pronostia_config.yaml        --exp_id exp_096
python scripts/train_mahalanobis.py --config configs/cwru_by_fault_config.yaml   --exp_id exp_097
python scripts/train_mahalanobis.py --config configs/cwru_by_severity_config.yaml --exp_id exp_098
python scripts/train_mahalanobis.py --config configs/pronostia_config.yaml        --exp_id exp_099
```

Métriques obligatoires dans chaque `experiments/exp_0XX/results/metrics_cl.json` :
`acc_final`, `avg_forgetting`, `backward_transfer`, `ram_peak_bytes`,
`inference_latency_ms`, `n_params` + champs spécifiques listés ci-dessus.

---

## 5. Notebooks comparatifs v1 vs v2

Un notebook par couple (dataset, scénario) :
```
notebooks/cl_eval/comparison_v1_v2_cwru_by_fault_type.ipynb
notebooks/cl_eval/comparison_v1_v2_cwru_by_severity.ipynb
notebooks/cl_eval/comparison_v1_v2_pronostia_by_condition.ipynb
```

### Structure obligatoire (6 sections)

```
1. Markdown — Introduction : versions comparées (v1 exp_0XX vs v2 exp_0YY)
2. Code — Tableau AA / AF / BWT v1 vs v2 (DataFrame pandas)
3. Code — Plot threshold_history (KMeans) ou welford_coverage (Mahalanobis) par tâche
4. Code — Accuracy matrix heatmap côte à côte (v1 | v2) via plot_accuracy_matrix()
5. Markdown — Analyse : gain/perte et interprétation CL
6. Markdown — Conclusion : recommandation usage embarqué
```

Fonctions à réutiliser depuis [src/evaluation/plots.py](../src/evaluation/plots.py) :
`plot_accuracy_matrix()`, `plot_forgetting_curve()`

---

## 6. Checklist de livraison

```
[ ] configs/unsupervised_config.yaml : ema_alpha + cl_strategy="welford" ajoutés
[ ] KMeansDetector.fit_task() : logique EMA + threshold_history_ + annotation MEM
[ ] MahalanobisDetector.fit_task() : branche cl_strategy="welford" + welford_updates_per_task_
[ ] exp_094–099 exécutées, metrics_cl.json dans experiments/
[ ] 3 notebooks comparatifs créés et "Run All Cells" exécuté
[ ] MAJ docs/roadmap_phase1.md : section Sprint 13 ajoutée avec exp_094–099
[ ] MAJ skills/README.md : model_improvement_v2.md dans l'index
```
