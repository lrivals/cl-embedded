# Spec — Modèles non supervisés (Sprint 5)

> Version : 1.0 | Créé : 7 avril 2026  
> Modèles : `KMeansDetector` (M4a), `KNNDetector` (M4b), `PCABaseline` (M5)  
> Fichiers source : `src/models/unsupervised/`

---

## 1. Interface commune

Tous les modèles implémentent le contrat suivant (pas de classe de base abstraite — duck typing) :

| Méthode | Signature | Description |
|---------|-----------|-------------|
| `fit_task` | `(X: np.ndarray, task_id: int) -> self` | Entraînement sur une tâche (sans labels) |
| `predict` | `(X: np.ndarray) -> np.ndarray[int64]` | Prédiction binaire {0 : normal, 1 : anomalie} |
| `anomaly_score` | `(X: np.ndarray) -> np.ndarray[float32]` | Score continu (plus élevé = plus anormal) |
| `score` | `(X: np.ndarray, y: np.ndarray) -> float` | Accuracy (labels utilisés uniquement en évaluation) |
| `save` | `(path: str \| Path) -> None` | Sérialisation pickle |
| `load` | `(path: str \| Path) -> Self` | Désérialisation pickle (**classmethod**) |
| `summary` | `() -> str` | Résumé texte de l'état du modèle |
| `count_parameters` | `() -> int` | Nombre de valeurs float stockées (pour estimation RAM) |

> **Note** : `load` est un `@classmethod` — appeler `Model.load(path)`, pas `instance.load(path)`.

### Seuil de décision

Le seuil (`threshold_`) est calculé automatiquement sur Task 0 au percentile `anomaly_percentile` (défaut : 95e). Il n'est **pas** recalculé sur les tâches suivantes pour éviter le leakage inter-tâches. Il peut être fixé manuellement via `anomaly_threshold` dans la config.

---

## 2. Stratégies CL

| Modèle | Stratégie par défaut | Stratégie alternative | Description |
|--------|---------------------|----------------------|-------------|
| `KMeansDetector` | `"refit"` | `"accumulate"` | Refit : clustering réinitialisé sur la tâche courante uniquement. Accumulate : centroides conservés entre tâches (expérimental). |
| `KNNDetector` | `"accumulate"` | `"refit"` | Accumulate : X_ref_ croît à chaque tâche (borné par `max_ref_samples`). Refit : X_ref_ remplacé par la tâche courante. |
| `PCABaseline` | `"refit"` | `"incremental"` | Refit : nouvelle PCA par tâche. Incremental : `IncrementalPCA.partial_fit()` — conserve une mémoire des tâches passées. |

---

## 3. Empreinte mémoire estimée

### Dataset 2 — Equipment Monitoring (4 features, 3 tâches)

| Modèle | Paramètres stockés | `count_parameters()` | RAM FP32 | RAM INT8 |
|--------|-------------------|:-------------------:|:--------:|:--------:|
| KMeans (K=2) | 2 centroides × 4 features | 8 | 32 B | 8 B |
| KNN (max_ref=200) | 200 échantillons × 4 features | 800 | 3 200 B | 800 B |
| PCA (n_comp=2) | 2 vecteurs × 4 features + mean | 12 | 48 B | 12 B |

### Dataset 1 — Pump Maintenance (~50 features après fenêtrage)

| Modèle | Paramètres stockés | RAM FP32 | Remarque |
|--------|-------------------|:--------:|---------|
| KMeans (K=2) | 2 × 50 | 400 B | ✅ STM32N6 |
| KNN (max_ref=200) | 200 × 50 | 40 000 B | ✅ STM32N6 (< 64 Ko) |
| PCA (n_comp=5) | 5 × 50 + 50 | 1 200 B | ✅ STM32N6 |

> Pour KNN sur Dataset 1 : `max_ref_samples ≤ 327` pour rester sous 64 Ko à FP32 (50 features × 327 × 4 = 65 400 B).

---

## 4. Attributs publics post-fit

### KMeansDetector

| Attribut | Type | Description |
|----------|------|-------------|
| `kmeans_` | `sklearn.cluster.KMeans \| None` | Modèle KMeans entraîné |
| `threshold_` | `float \| None` | Seuil de décision |
| `k_selected_` | `list[int]` | K optimal sélectionné à chaque tâche |
| `task_id_` | `int` | Dernier task_id vu |

### KNNDetector

| Attribut | Type | Description |
|----------|------|-------------|
| `nn_` | `sklearn.neighbors.NearestNeighbors \| None` | Index KNN |
| `X_ref_` | `np.ndarray \| None` | Données de référence ([N_ref, n_features]) |
| `threshold_` | `float \| None` | Seuil de décision |
| `task_id_` | `int` | Dernier task_id vu |

### PCABaseline

| Attribut | Type | Description |
|----------|------|-------------|
| `pca_` | `sklearn.decomposition.PCA \| IncrementalPCA \| None` | Modèle PCA |
| `n_components_fitted_` | `int \| None` | Nombre de composantes effectivement utilisées |
| `threshold_` | `float \| None` | Seuil de décision |
| `task_id_` | `int` | Dernier task_id vu |

---

## 5. Critères d'acceptation (mémoire)

- `count_parameters() * 4 ≤ 65 536` pour tous les modèles sur Dataset 2 (4 features)
- `count_parameters() * 4 ≤ 65 536` pour KNN sur Dataset 1 avec `max_ref_samples ≤ 327`
- RAM peak mesurée (tracemalloc) cohérente avec `count_parameters() * 4` à ±30 % (overhead Python/joblib)

---

## 6. Résultats exp_005 — Dataset 2 (7 avril 2026, seed=42)

| Modèle | AA | AF | BWT | AUROC | RAM peak | Latence |
|--------|:--:|:--:|:---:|:-----:|:--------:|:-------:|
| K-Means (K=2, silhouette) | 0.9433 | 0.0049 | -0.0040 | 0.9621 | 5.2 Ko ✅ | 0.399 ms |
| KNN (accumulate, k=5) | 0.9524 | 0.0275 | -0.0275 | 0.9728 | 110.5 Ko ⚠️ | 15.755 ms |
| PCA (2 composantes, refit) | 0.9504 | 0.0020 | -0.0010 | 0.9078 | 2.1 Ko ✅ | 0.115 ms |

> ⚠️ KNN dépasse 64 Ko en stratégie `accumulate` sur 6 137 échantillons — PC-only sans `max_ref_samples`.
