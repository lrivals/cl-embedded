# S19-03 — Mise à jour `configs/unsupervised_config.yaml` pour Pronostia

| Champ | Valeur |
|-------|--------|
| **ID** | S19-03 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S19-01 (scénario), S19-02 (loader vérifié) |
| **Fichier cible** | `configs/unsupervised_config.yaml` |

---

## Objectif

Ajouter le bloc `pronostia:` dans `configs/unsupervised_config.yaml` avec les paramètres spécifiques Pronostia pour l'anomaly detection (INPUT_DIM=13, scénario by_bearing_condition, ratio ~90% normal).

---

## Contenu à ajouter

```yaml
# --- Bloc Pronostia Anomaly Detection (ajout Sprint 19) ---
DATASETS:
  # ... blocs monitoring, equipment_monitoring, cwru existants ...
  pronostia:
    INPUT_DIM: 13
    N_TASKS: 3
    SPLIT_STRATEGY: "by_bearing_condition"   # Early life → Mid life → End of life
    NORMAL_RATIO: 0.90                        # ~90% de données normales (début de vie)
    # RAM estimée modèles @ FP32 (input_dim=13) :
    #   KMeans (k=1, 13 features) : 1 * 13 * 4 = 52 B
    #   Mahalanobis (cov 13×13) : 13 * 13 * 4 = 676 B  MEM: 676 B @ FP32 / 169 B @ INT8

    # Overrides modèles pour Pronostia (13D, ~90% normaux mais cov instable en end_of_life)
    kmeans:
      N_CLUSTERS: 1               # 1 centroïde normal par condition
    mahalanobis:
      REG_COVAR: 1.0e-5           # régularisation légèrement renforcée (cov 13×13)
    dbscan:
      EPS: 0.6
      MIN_SAMPLES: 5
    ewc_oneclass:
      THRESHOLD_PERCENTILE: 95    # seuil standard (beaucoup de normaux en early_life)
```

---

## Critères d'acceptation

- [ ] Bloc `pronostia:` présent sous `DATASETS:` dans `unsupervised_config.yaml`
- [ ] `INPUT_DIM: 13`, `SPLIT_STRATEGY: "by_bearing_condition"` et `NORMAL_RATIO: 0.90` présents
- [ ] Annotation `# MEM:` présente sur la ligne Mahalanobis (676 B @ FP32 / 169 B @ INT8)
- [ ] Config charge sans erreur après modification

## Statut

⬜ À faire (après S19-01)
