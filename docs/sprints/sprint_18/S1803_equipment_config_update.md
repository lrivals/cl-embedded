# S18-03 — Mise à jour `configs/unsupervised_config.yaml` pour Equipment Monitoring

| Champ | Valeur |
|-------|--------|
| **ID** | S18-03 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 0.5h |
| **Dépendances** | S18-01 (scénario), S18-02 (loader) |
| **Fichier cible** | `configs/unsupervised_config.yaml` |

---

## Objectif

Ajouter le bloc `equipment_monitoring:` dans `configs/unsupervised_config.yaml` avec les paramètres spécifiques Equipment Monitoring pour l'anomaly detection (INPUT_DIM=4, scénario by_equipment_type, ratio ~50% normal).

---

## Contenu à ajouter

```yaml
# --- Bloc Equipment Monitoring Anomaly Detection (ajout Sprint 18) ---
DATASETS:
  # ... blocs monitoring, pronostia, cwru existants ...
  equipment_monitoring:
    INPUT_DIM: 4
    N_TASKS: 3
    SPLIT_STRATEGY: "by_equipment_type"   # Pump → Turbine → Compressor
    NORMAL_RATIO: 0.50                    # ~50% de données normales (condition favorable)
    # RAM estimée modèles @ FP32 (input_dim=4) :
    #   KMeans (k=1, 4 features) : 1 * 4 * 4 = 16 B
    #   Mahalanobis (cov 4×4) : 4 * 4 * 4 = 64 B  MEM: 64 B @ FP32 / 16 B @ INT8

    # Overrides modèles pour Equipment Monitoring (4D, ~50% normaux)
    kmeans:
      N_CLUSTERS: 1               # 1 centroïde normal (données suffisantes)
    mahalanobis:
      REG_COVAR: 1.0e-6           # régularisation standard (assez de données)
    dbscan:
      EPS: 0.5
      MIN_SAMPLES: 5              # min_samples standard (assez de normaux)
    ewc_oneclass:
      THRESHOLD_PERCENTILE: 95    # seuil standard (assez de normaux d'entraînement)
```

---

## Critères d'acceptation

- [ ] Bloc `equipment_monitoring:` présent sous `DATASETS:` dans `unsupervised_config.yaml`
- [ ] `INPUT_DIM: 4`, `SPLIT_STRATEGY: "by_equipment_type"` et `NORMAL_RATIO: 0.50` présents
- [ ] Annotation `# MEM:` présente sur la ligne Mahalanobis (contrainte CLAUDE.md)
- [ ] Config charge sans erreur après modification

## Statut

⬜ À faire (après S18-01)
