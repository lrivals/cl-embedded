# S14-09 — exp_131–136 : 6 modèles × Monitoring by_location refit

| Champ | Valeur |
|-------|--------|
| **ID** | S14-09 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S14-03 (DBSCAN), S14-01 (EWC OC), S14-08 (loader by_location) |
| **Fichiers cibles** | `experiments/exp_131/` → `experiments/exp_136/` |

---

## Objectif

Exécuter les 6 modèles d'anomaly detection sur le scénario Monitoring by_location (5 tâches) en stratégie refit. Permet de comparer la robustesse des modèles à une distribution de tâches différente de by_equipment.

---

## Expériences

| Exp | Modèle | Scénario | Stratégie | Config |
|-----|--------|----------|-----------|--------|
| exp_131 | HDC | monitoring by_location | refit | `configs/hdc_config.yaml` |
| exp_132 | TinyOL AE | monitoring by_location | refit | `configs/tinyol_config.yaml` |
| exp_133 | KMeans | monitoring by_location | refit | `configs/unsupervised_config.yaml` |
| exp_134 | Mahalanobis | monitoring by_location | refit | `configs/unsupervised_config.yaml` |
| exp_135 | DBSCAN | monitoring by_location | refit | `configs/unsupervised_config.yaml` |
| exp_136 | EWC one-class | monitoring by_location | refit | `configs/ewc_oneclass_config.yaml` |

---

## Commande type

```bash
for model in hdc tinyol_ae kmeans mahalanobis dbscan ewc_oneclass; do
    python scripts/run_anomaly_detection.py \
        --model $model \
        --dataset monitoring \
        --scenario by_location \
        --strategy refit \
        --config configs/$(model_to_config $model) \
        --exp_id exp_$(exp_id_for $model)
done
```

---

## Critères d'acceptation

- [ ] exp_131–136 : `metrics_anomaly.json` présents, avec `auroc_per_task` de longueur 5
- [ ] `config_snapshot.yaml` présents dans les 6 répertoires
- [ ] Aucun crash sur les 5 tâches (DBSCAN potentiellement lent sur tâche 4-5 accumulate — ici refit donc OK)
- [ ] AUROC moyen > 0.5 pour tous les modèles

## Statut

⬜ À faire
