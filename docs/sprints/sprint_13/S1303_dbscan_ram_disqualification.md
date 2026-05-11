# S13-03 — `DBSCANDetector` : analyse RAM et décision d'exclusion STM32N6

| Champ | Valeur |
|-------|--------|
| **ID** | S13-03 |
| **Sprint** | Sprint 13 |
| **Priorité** | 🔴 Critique (décision architecture) |
| **Fichier source** | `src/models/unsupervised/dbscan_detector.py` |
| **Config** | `configs/unsupervised_config.yaml` — section `dbscan:` |
| **Expériences** | exp_120, exp_121, exp_122, exp_160 |
| **Décision** | ❌ **DBSCAN écarté du portage STM32N6** |
| **Statut** | ✅ Complété (analyse conclusive) |

---

## Principe

`DBSCANDetector` utilise DBSCAN pour identifier les **core points** — points en zone dense — sur les données normales. Le score d'anomalie est la distance euclidienne au core point le plus proche :

```
score(x) = min_{c ∈ core_points} ||x − c||₂
```

Un échantillon est classifié comme anormal si `score(x) > threshold_` (calculé au percentile 95 sur Task 0).

---

## Problème RAM fondamental

### Croissance non bornée avec la taille du dataset

Le modèle stocke **tous les core points** après l'entraînement :

```
RAM_modèle = n_core_samples × d × 4 octets (FP32)
```

`n_core_samples` n'est pas borné théoriquement — il croît avec la taille du dataset d'entraînement. Il n'existe pas de borne fixe dérivable a priori sans contrainte explicite sur `min_samples` et `eps`.

**Comparaison avec KMeans et Mahalanobis** :

| Modèle | RAM modèle | Dépend de N ? | Borne théorique |
|--------|:----------:|:-------------:|:---------------:|
| Mahalanobis | d + d² floats | ❌ Non | ✅ Oui — fixe |
| KMeans | K × d floats | ❌ Non | ✅ Oui — K max borné |
| DBSCAN | n_core × d floats | ✅ Oui | ❌ Non sans contrainte |

### Tentative de compression

Il n'est pas possible de réduire `n_core_samples` sans modifier la structure de voisinage DBSCAN :
- Réduire `min_samples` : diminue légèrement les core points mais augmente le bruit de classification
- Augmenter `eps` : peut regrouper des régions qui ne devraient pas l'être
- Sous-échantillonner les core points a posteriori : détruit la garantie de détection (les nouveaux points se retrouvent plus éloignés des core points restants)

---

## Résultats mesurés

| Exp | Dataset | Scénario | n_core_samples | RAM peak | Limite 64 Ko | Statut |
|-----|---------|----------|:-------------:|:--------:|:------------:|:------:|
| exp_120 | Monitoring | by_equipment | 5 412 | **73.6 KB** | 64 KB | ❌ +15% |
| exp_121 | Monitoring | by_location | 2 476 | **40.4 KB** | 64 KB | ✅ (63%) |
| exp_122 | Pronostia | by_condition | 11 479 | **120.9 KB** | 64 KB | ❌ +89% |
| exp_160 | CWRU | by_severity | 4 797 | **56.3 KB** | 64 KB | ❌ +14% |

**3 scénarios sur 4 dépassent la limite.** Le seul scénario conforme (Monitoring by_location, 40.4 KB) n'est pas garanti si la taille du train set varie — absence de borne théorique.

### Détail des seuils et eps par tâche (exp_122 — Pronostia)

```
eps par tâche       : [0.5, 0.5, 0.5]
threshold par tâche : {Task 0: 6.41, Task 1: 2.85, Task 2: 2.42}
```

La variabilité des seuils entre tâches illustre que les distributions de core points sont instables inter-tâches en mode refit — conséquence attendue du domain shift.

---

## Comparaison AUROC vs RAM

Pour un AUROC comparable (~0.95) sur Monitoring by_equipment :

| Modèle | AUROC | RAM peak | Rapport |
|--------|:-----:|:--------:|:-------:|
| DBSCAN | ~0.987 | 73.6 KB | — |
| KMeans | ~0.962 | 5.3 KB | 14× |
| Mahalanobis | ~0.988 | 1.2 KB | **61×** |

Mahalanobis atteint un AUROC supérieur ou équivalent à DBSCAN pour **61 fois moins de RAM**.

---

## Décision

**DBSCAN est écarté du portage STM32N6.** Motivations :

1. **RAM non bornée** : aucune garantie de rester sous 64 Ko sans contrainte applicative
2. **Dépassement mesuré** : 3/4 scénarios testés excèdent la limite
3. **Alternatives supérieures** : Mahalanobis et KMeans offrent un AUROC comparable pour un coût RAM 14–61× inférieur
4. **Compatibilité MCU** : DBSCAN requiert des opérations de recherche de voisinage (k-NN) non natives sur Cortex-M55

**DBSCAN est conservé** comme baseline de référence PC-only dans les notebooks d'évaluation pour comparer les AUROC.

---

## Piste d'amélioration (PC uniquement)

`TODO(dorra)` : explorer une variante **CoreSet-DBSCAN** :
- Après fit DBSCAN, appliquer un algorithme de CoreSet (greedy k-center) pour sous-échantillonner les core points à un budget fixe, ex. 512 points maximum
- RAM bornée : 512 × d × 4 octets = 18 KB max pour d=9 (CWRU) — sous 64 Ko
- Coût : perte potentielle de précision dans les régions denses sous-représentées
- Référence à consulter : `Ravaglia2021QLRCL` (rejeu latent borné par buffer UINT8)

---

## Configuration actuelle

Section `dbscan:` dans `configs/unsupervised_config.yaml` :

```yaml
dbscan:
  EPS: 0.5                  # null pour estimation automatique via k-NN elbow
  MIN_SAMPLES: 5
  EPS_KNN_K: 5
  ANOMALY_PERCENTILE: 95
  CL_STRATEGY: "refit"
  METRIC: "euclidean"
```

> `EPS: null` déclenche l'estimation automatique via la méthode du coude sur la courbe des distances k-NN. Utile en exploration mais instable sur des datasets de petite taille.
