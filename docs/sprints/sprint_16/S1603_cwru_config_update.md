# S16-03 — Mise à jour `configs/unsupervised_config.yaml` pour CWRU

| Champ | Valeur |
|-------|--------|
| **ID** | S16-03 |
| **Sprint** | Sprint 16 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S16-01 (scénario), S16-02 (loader) |
| **Fichier cible** | `configs/unsupervised_config.yaml` |

---

## Objectif

Ajouter le bloc `cwru:` dans `configs/unsupervised_config.yaml` avec les paramètres spécifiques CWRU pour l'anomaly detection (input_dim=9, stratégie retenue, ratio normal).

---

## Contenu à ajouter

```yaml
# --- Bloc CWRU Anomaly Detection (ajout Sprint 16) ---
DATASETS:
  # ... blocs monitoring et pronostia existants ...
  cwru:
    INPUT_DIM: 9
    N_TASKS: 3
    SPLIT_STRATEGY: "by_severity"   # ou "by_fault_type" selon décision S16-01
    NORMAL_RATIO: 0.10              # ~10% de données normales (230 / 2300)
    # Attention : seulement ~77 échantillons normaux par tâche
    # RAM estimée modèles @ FP32 (input_dim=9) :
    #   KMeans (k=3, 9 features) : 3 * 9 * 4 = 108 B
    #   Mahalanobis (cov 9×9) : 9 * 9 * 4 = 324 B

    # Overrides modèles pour CWRU (9D, peu de normaux)
    kmeans:
      N_CLUSTERS: 1               # 1 seul centroïde normal par tâche (peu de données)
    mahalanobis:
      REG_COVAR: 1.0e-4           # régularisation plus forte (77 échantillons → cov instable)
    dbscan:
      EPS: 0.8
      MIN_SAMPLES: 3              # min_samples réduit car peu de normaux
    ewc_oneclass:
      THRESHOLD_PERCENTILE: 80    # seuil abaissé car entraînement sur peu de normaux
```

---

## Critères d'acceptation

- [ ] Bloc `cwru:` présent sous `DATASETS:` dans `unsupervised_config.yaml`
- [ ] `INPUT_DIM: 9`, `SPLIT_STRATEGY` et `NORMAL_RATIO` présents
- [ ] Overrides modèles justifiés par le faible nombre d'échantillons normaux (~77)
- [ ] Config charge sans erreur après modification

## Statut

⬜ À faire (après S16-01)
