# S15-14 — Mise à jour config pour CWRU anomaly detection

| Champ | Valeur |
|-------|--------|
| **ID** | S15-14 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S15-13 |
| **Fichier cible** | `configs/unsupervised_anomaly_detection_config.yaml` |

---

## Objectif

Ajouter un bloc `cwru:` dans `configs/unsupervised_anomaly_detection_config.yaml` (section `DATASETS`) avec les paramètres spécifiques CWRU. Les overrides des modèles sont justifiés par le faible nombre de données normales (~62 train/tâche), contrairement à Pronostia (~750 train/tâche) ou Monitoring (~1500 train/tâche).

---

## Différence avec Pronostia

Pronostia utilise `configs/unsupervised_config.yaml` (fichier généraliste). CWRU utilise `configs/unsupervised_anomaly_detection_config.yaml` (fichier dédié anomaly detection), déjà utilisé pour Equipment Monitoring (Sprint 14). Le bloc CWRU est donc ajouté à ce fichier existant.

---

## Contenu ajouté dans `unsupervised_anomaly_detection_config.yaml`

```yaml
# --- Blocs dataset-specific (ajout Sprint 16) ---
DATASETS:
  cwru:
    INPUT_DIM: 9
    N_TASKS: 3
    SPLIT_STRATEGY: "by_severity"   # décision S16-01 — by_severity retenu (par défaut)
    NORMAL_RATIO: 0.10              # ~230 normaux / 2300 total
    csv_path: "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv"
    batch_size: 32
    # ~77 normaux/tâche → ~62 train, ~15 test normaux
    # MEM: KMeans (k=1, d=9) : 1 × 9 × 4 = 36 B @ FP32
    # MEM: Mahalanobis (d=9, cov 9×9) : (9 + 81) × 4 = 360 B @ FP32
    # Overrides justifiés par le faible nombre de normaux (~62 train/tâche)
    kmeans:
      N_CLUSTERS: 1           # 1 centroïde normal (données insuffisantes pour k>1)
    mahalanobis:
      REG_COVAR: 1.0e-4       # régularisation plus forte (cov instable sur ~62 échantillons)
    dbscan:
      EPS: 0.8
      MIN_SAMPLES: 3          # min_samples réduit car peu de normaux
    ewc_oneclass:
      THRESHOLD_PERCENTILE: 80  # seuil abaissé (entraînement sur peu de normaux)
```

---

## Justification des overrides

| Paramètre | Valeur Pronostia | Valeur CWRU | Raison |
|-----------|:---------------:|:-----------:|--------|
| `N_CLUSTERS` | 3 | **1** | Insuffisant pour apprendre plusieurs clusters avec ~62 normaux |
| `REG_COVAR` | 1e-6 | **1e-4** | Matrice de covariance 9×9 instable sur ~62 échantillons |
| `MIN_SAMPLES` | 5 | **3** | DBSCAN nécessite au moins MIN_SAMPLES points dans l'ε-voisinage — trop restrictif avec peu de normaux |
| `EPS` | 1.5 | **0.8** | Données 9D normalisées — ε plus petit adapté à l'espace de features CWRU |
| `THRESHOLD_PERCENTILE` | 95 | **80** | EWC one-class : seuil abaissé car le modèle voit peu de normaux pendant l'entraînement |

---

## Critères d'acceptation

- [x] Bloc `cwru:` présent sous `DATASETS:` dans `unsupervised_anomaly_detection_config.yaml`
- [x] `INPUT_DIM: 9` et `NORMAL_RATIO: 0.10` présents
- [x] Commentaires RAM pour KMeans (36 B) et Mahalanobis (360 B) sur 9D
- [x] `N_CLUSTERS: 1` documenté avec justification
- [x] Config charge sans erreur après modification

## Statut

✅ Terminé

## Bilan

Le bloc `cwru:` a été ajouté à `configs/unsupervised_anomaly_detection_config.yaml` (ligne ~53). Il contient `INPUT_DIM: 9`, `N_TASKS: 3`, `NORMAL_RATIO: 0.10`, les commentaires RAM pour KMeans (36 B) et Mahalanobis (360 B), ainsi que les overrides `N_CLUSTERS: 1`, `REG_COVAR: 1e-4`, `EPS: 0.8` / `MIN_SAMPLES: 3` et `THRESHOLD_PERCENTILE: 80`. Tous les overrides sont justifiés par le faible nombre de données normales (~62 train/tâche).
