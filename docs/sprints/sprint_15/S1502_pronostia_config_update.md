# S15-02 — Mise à jour `configs/unsupervised_config.yaml` pour Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S15-02 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S15-01 |
| **Fichier cible** | `configs/unsupervised_config.yaml` |

---

## Objectif

Ajouter un bloc `pronostia:` dans `configs/unsupervised_config.yaml` avec les paramètres spécifiques au dataset (input_dim=13, FAILURE_RATIO, condition_ids). Les hyperparamètres des modèles peuvent avoir des overrides dataset-spécifiques.

---

## Contenu à ajouter dans `unsupervised_config.yaml`

```yaml
# --- Bloc Pronostia (ajout Sprint 15) ---
DATASETS:
  # ... bloc monitoring existant ...
  pronostia:
    INPUT_DIM: 13
    N_TASKS: 3
    SCENARIO: "by_condition"
    FAILURE_RATIO: 0.10        # 10% derniers de chaque séquence = faulty
    CONDITION_IDS: [1, 2, 3]   # conditions opératoires Pronostia
    # RAM estimée modèles @ FP32 (input_dim=13) :
    #   KMeans (k=3, 13 features) : 3 * 13 * 4 = 156 B
    #   Mahalanobis (cov 13×13) : 13 * 13 * 4 = 676 B
    #   DBSCAN : dépend du nombre de points (variable)

    # Overrides modèles pour input_dim=13
    kmeans:
      N_CLUSTERS: 3             # 1 centroïde par condition opératoire
    mahalanobis:
      REG_COVAR: 1.0e-6         # régularisation covariance (13D plus sensible)
    dbscan:
      EPS: 1.5                  # adapter eps à la dimension 13
      MIN_SAMPLES: 5
```

---

## Critères d'acceptation

- [x] Bloc `pronostia:` présent sous `DATASETS:` dans `unsupervised_config.yaml`
- [x] `INPUT_DIM: 13` et `FAILURE_RATIO: 0.10` présents
- [x] Commentaires RAM pour les modèles statiques (KMeans, Mahalanobis) sur 13D
- [x] Config charge sans erreur après modification

## Statut

✅ Terminé

## Bilan

Le bloc `pronostia:` a été ajouté à `configs/unsupervised_config.yaml` (ligne 89). Il contient `INPUT_DIM: 13`, `N_TASKS: 3`, `FAILURE_RATIO: 0.10`, `CONDITION_IDS: [1, 2, 3]`, les commentaires RAM pour KMeans (156 B) et Mahalanobis (676 B), ainsi que les overrides `N_CLUSTERS: 3`, `REG_COVAR: 1e-6` et `EPSILON: 1.5` / `MIN_SAMPLES: 5` pour DBSCAN. Note : la clé est nommée `EPSILON` (conforme au code) plutôt que `EPS` comme spécifié initialement dans ce doc.
