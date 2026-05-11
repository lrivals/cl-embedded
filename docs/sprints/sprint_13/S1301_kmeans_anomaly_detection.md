# S13-01 — `KMeansDetector` : détection d'anomalies one-class (mode CL)

| Champ | Valeur |
|-------|--------|
| **ID** | S13-01 |
| **Sprint** | Sprint 13 |
| **Priorité** | 🔴 Critique |
| **Fichier source** | `src/models/unsupervised/kmeans_detector.py` |
| **Config** | `configs/unsupervised_config.yaml` — section `kmeans:` |
| **Expériences** | exp_100, exp_101, exp_102, exp_112, exp_116 |
| **Statut** | ✅ Complété |

---

## Principe

`KMeansDetector` opère en scénario **one-class** : il est entraîné uniquement sur des données normales et détecte les anomalies par distance au centroïde le plus proche.

**Score d'anomalie** = distance euclidienne minimale aux centroïdes :

```
score(x) = min_j ||x − c_j||₂    pour j ∈ {0, …, K−1}
```

Un échantillon est classifié comme anormal si `score(x) > threshold_`.

---

## Sélection de K

La méthode `_select_k(X)` choisit K automatiquement selon `k_method` (configurable) :

| `k_method` | Critère | Usage recommandé |
|-----------|---------|-----------------|
| `"silhouette"` (défaut) | Maximise le score silhouette — meilleure séparation inter-cluster | Données avec structure de cluster nette |
| `"elbow"` | Minimise l'inertie (méthode du coude, 2ème dérivé) | Données homogènes sans structure claire |
| `"fixed"` | K = `k_fixed` fixé dans la config | Quand K est connu a priori |

Sur tous les datasets testés (Monitoring d=4, Pronostia d=13, CWRU d=9), K=2 est systématiquement sélectionné par la méthode silhouette : cela reflète la structure binaire du problème (normal / anormal).

---

## Stratégie CL

| `cl_strategy` | Comportement | Forgetting attendu |
|--------------|--------------|-------------------|
| `"refit"` (défaut) | KMeans réinitialisé à chaque nouvelle tâche — apprend uniquement la distribution courante | AF faible (oubli intentionnel — modèle toujours adapté à la tâche courante) |
| `"accumulate"` | Centroides conservés entre tâches (expérimental) | AF plus faible théoriquement, mais K croît avec le nombre de tâches |

**Seuil adaptatif (EMA)** : le seuil de décision est mis à jour entre tâches via une moyenne exponentielle :

```
threshold_(t) = ema_alpha × threshold_task_t + (1 − ema_alpha) × threshold_(t−1)
```

avec `ema_alpha=0.3` (configurable). Le seuil initial est calculé au percentile 95 des scores sur Task 0.

---

## Empreinte mémoire

`count_parameters() = k × n_features` (centroides uniquement) :

| Dataset | d | K | Modèle (FP32) | Modèle (INT8) |
|---------|---|---|--------------|--------------|
| Monitoring | 4 | 2 | 32 B | 8 B |
| Pronostia | 13 | 2 | 104 B | 26 B |
| CWRU | 9 | 2 | 72 B | 18 B |

> La RAM peak mesurée (~5.3 KB) reflète l'overhead Python/sklearn, pas la taille réelle du modèle. Le modèle porté sur STM32N6 n'occupera que les octets du tableau de centroides.

---

## Résultats expérimentaux

| Exp | Dataset | Scénario | Accuracy | AF | BWT | RAM peak | Latence | STM32N6 |
|-----|---------|----------|:--------:|:--:|:---:|:--------:|:-------:|:-------:|
| exp_100 | CWRU | by_fault_type | 0.312 | 0.065 | +0.201 | 5.3 KB | 0.200 ms | ✅ |
| exp_101 | CWRU | by_severity | 0.450 | — | — | 5.3 KB | 0.245 ms | ✅ |
| exp_102 | Pronostia | by_condition | 0.872 | 0.059 | −0.059 | 5.3 KB | 0.188 ms | ✅ |
| exp_112 | Monitoring | by_equipment | 0.943 | 0.005 | −0.004 | 5.3 KB | 0.281 ms | ✅ |
| exp_116 | Monitoring | by_location | 0.947 | — | — | 5.3 KB | 0.293 ms | ✅ |

**AUROC de référence** : 0.9621 (Monitoring by_equipment, exp_005 sprint 5).

### Analyse CWRU (exp_100–101)

La faible accuracy sur CWRU (0.31–0.45) s'explique par le ratio déséquilibré du dataset : ~20% de données normales en entraînement. Avec si peu de données normales, le percentile 95 du score normal sous-estime le vrai seuil. À explorer : abaisser `anomaly_percentile` à 80 ou augmenter `k_fixed`.

### Analyse Pronostia (exp_102)

Accuracy 0.87 — AF modéré (0.059) dû au refit par tâche : le modèle "oublie" intentionnellement la distribution de la tâche précédente. BWT négatif (-0.059) confirme l'absence de transfert positif entre tâches.

### Analyse Monitoring (exp_112, exp_116)

Meilleurs résultats (0.943–0.947) grâce à la structure claire du dataset (4 features bien séparées). AF quasi-nul (~0.005) malgré le refit — les 3 équipements/locations ont des distributions normales similaires.

---

## Configuration

Section `kmeans:` dans `configs/unsupervised_config.yaml` :

```yaml
kmeans:
  K_METHOD: "silhouette"     # "silhouette" | "elbow" | "fixed"
  K_FIXED: 3
  K_MIN: 2
  K_MAX: 10
  ANOMALY_PERCENTILE: 95
  N_INIT: 10
  MAX_ITER: 300
  CL_STRATEGY: "refit"
  EMA_ALPHA: 0.3
```

---

## Conclusion STM32N6

**KMeans est compatible STM32N6** sur tous les datasets testés.

- RAM modèle réel : 32–104 B @ FP32 (largement sous 64 Ko)
- RAM peak Python : 5.3 KB (overhead non présent sur MCU)
- Latence < 0.3 ms — sous la contrainte 100 ms
- Candidat pour portage après Mahalanobis (voir [S1302](S1302_mahalanobis_anomaly_detection.md))

> `FIXME(gap2)` : mesurer la RAM réelle sur STM32N6 avec le tableau de centroides en FP32 ou INT8, sans overhead Python.
