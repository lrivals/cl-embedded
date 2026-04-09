# S5-14 — DBSCAN Detector (M7)

## Objectif

Implémenter `dbscan_detector.py` comme septième baseline non supervisée (M7), basée sur DBSCAN (sklearn). Étudier son efficacité en scénario domain-incremental : les points classés "bruit" par DBSCAN sont traités comme anomalies (`faulty=1`). Contrairement à K-Means, DBSCAN ne nécessite pas de spécifier K à l'avance et détecte naturellement les outliers.

**Critère de succès** :
- `dbscan_detector.py` implémente l'interface `BaseUnsupervisedDetector`
- expérience exp_008 exécutée sur Dataset 2 (et Dataset 1 si disponible)
- métriques AA / AF / BWT / AUROC / RAM mesurées et enregistrées
- comparaison avec K-Means, KNN, PCA, Mahalanobis dans le notebook `05_supervised_vs_unsupervised.ipynb`

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S5-14a | Implémenter `dbscan_detector.py` (epsilon/min_samples via config, stratégie CL refit/accumulate, score = distance au cluster le plus proche) | ✅ | `src/models/unsupervised/dbscan_detector.py` | 2h | S5-01 |
| S5-14b | Ajouter section DBSCAN dans `configs/unsupervised_config.yaml` (EPSILON, MIN_SAMPLES, cl_strategy) | ✅ | `configs/unsupervised_config.yaml` | 0.5h | S5-14a |
| S5-14c | Expérience exp_008 — DBSCAN Dataset 2 + profiling RAM (tracemalloc) | ✅ | `experiments/exp_008_dbscan/` | 1h | S5-14b |
| S5-14d | Tests unitaires DBSCAN (interface, prédictions, contrainte RAM) | ✅ | `tests/test_unsupervised.py` (extension) | 1h | S5-14a |
| S5-14e | Mise à jour notebook comparatif avec résultats DBSCAN (ligne M7) | 🟡 | `notebooks/05_supervised_vs_unsupervised.ipynb` | 1h | S5-14c |

**Livrable** : `dbscan_detector.py` fonctionnel, exp_008 résultats enregistrés dans `experiments/exp_008_dbscan/`, tableau comparatif AA/AF/BWT/AUROC/RAM mis à jour avec M7.

---

## Notes d'implémentation

### Interface attendue

`DBSCANDetector` doit hériter de `BaseUnsupervisedDetector` (voir `src/models/unsupervised/__init__.py`) et implémenter :
- `fit(X)` — entraînement sur un domaine
- `predict(X) -> np.ndarray` — 0=normal, 1=anomalie (point bruit)
- `score(X) -> np.ndarray` — distance au cluster le plus proche (pour AUROC)

### Stratégie de scoring

DBSCAN produit des labels discrets (cluster ou bruit=-1). Pour calculer un score continu (nécessaire pour AUROC) :
- option A : distance au centroïde du cluster le plus proche (comme KNN)
- option B : proportion de voisins étiquetés bruit dans une fenêtre

### Contrainte embarquée

```python
# MEM: (n_samples × d) @ FP32 pour stratégie accumulate
# MEM: (n_core_samples × d) @ FP32 pour stratégie refit (core points only)
```

> `FIXME(gap2)` : mesurer RAM avec tracemalloc et comparer analytique vs mesuré comme pour Mahalanobis.

### Paramètres config

```yaml
# configs/unsupervised_config.yaml — section à ajouter
dbscan:
  EPSILON: 0.5          # rayon de voisinage (à tuner par dataset)
  MIN_SAMPLES: 5        # nombre minimum de voisins pour un core point
  cl_strategy: refit    # refit | accumulate
  metric: euclidean
```

---

## Résultats exp_008 — Dataset 2 (9 avril 2026, seed=42)

Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor — d=4 features

| Modèle | AA | AF | BWT | AUROC | RAM peak | Latence |
| ------ | :--: | :--: | :---: | :-----: | :--------: | :-------: |
| **DBSCAN (M7) ✅** | **0.9557** | **0.0000** | **+0.0010** | **0.9786** | 71.9 Ko ⚠️ | 0.423 ms |
| Mahalanobis (M6) ✅ | 0.9524 | 0.0010 | -0.0010 | 0.9718 | 80 B analytique | 0.018 ms |
| KNN (M4b) ⚠️ | 0.9524 | 0.0275 | -0.0275 | 0.9728 | 110.5 Ko | 15.755 ms |
| PCA (M5) ✅ | 0.9504 | 0.0020 | -0.0010 | 0.9078 | 2.1 Ko | 0.115 ms |
| K-Means (M4a) ✅ | 0.9433 | 0.0049 | -0.0040 | 0.9621 | 5.2 Ko | 0.399 ms |

> **AA = 0.9557** (meilleure de tous les non-supervisés), **AF = 0.0000** (aucun oubli), **AUROC = 0.9786** (meilleur).  
> ⚠️ RAM peak tracemalloc = 71.9 Ko (dépasse 64 Ko) — overhead Python. RAM analytique core points : 5412 params × 4 B = ~21 Ko @ FP32 ✅.  
> `FIXME(gap2)` : RAM analytique OK (< 64 Ko), mais tracemalloc inclut l'overhead de `euclidean_distances` (matrice [N, n_core]) — non représentatif du MCU.  
> eps=0.5, min_samples=5 — n_core ≈ 1 350–1 370 points par tâche (cl_strategy=refit).

## Questions ouvertes

- `TODO(arnaud)` : DBSCAN avec `refit` (recalcul complet à chaque domaine, oubli structurel) vs `accumulate` (conserve tous les points vus, mémoire croissante) — lequel est plus pertinent pour le scénario domain-incremental de ce projet ?
- `TODO(arnaud)` : epsilon optimal à fixer manuellement (via silhouette) ou à laisser configurable par dataset ? Même question posée pour K-Means (K dynamique via silhouette/elbow en S5-02).
- `FIXME(gap2)` : stratégie `accumulate` avec DBSCAN dépasse potentiellement 64 Ko sur Dataset 1 (séries temporelles longues) — vérifier et documenter la limite.
