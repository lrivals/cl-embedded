# S14-10 — Notebook 6 modèles Monitoring (by_equipment + by_location)

| Champ | Valeur |
|-------|--------|
| **ID** | S14-10 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S14-04 → S14-09 (toutes les expériences Monitoring terminées) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_anomaly_detection/notebook_anomaly_detection_6models.ipynb` |

---

## Objectif

Produire le notebook de synthèse Monitoring Anomaly Detection comparant les 6 modèles sur les deux scénarios (by_equipment, by_location) avec les deux stratégies (refit, accumulate).

---

## Structure du notebook

### Cellule 1 — Imports et chargement des résultats

```python
import json, pathlib
import pandas as pd
import matplotlib.pyplot as plt

EXP_RESULTS = {
    "hdc_equip_refit":      "experiments/exp_086/results/metrics_anomaly.json",
    "tinyol_equip_refit":   "experiments/exp_087/results/metrics_anomaly.json",
    "kmeans_equip_refit":   "experiments/exp_088/results/metrics_anomaly.json",
    "mahal_equip_refit":    "experiments/exp_089/results/metrics_anomaly.json",
    "dbscan_equip_refit":   "experiments/exp_123/results/metrics_anomaly.json",
    "ewcoc_equip_refit":    "experiments/exp_125/results/metrics_anomaly.json",
    "hdc_equip_accum":      "experiments/exp_127/results/metrics_anomaly.json",
    # ... etc.
}
```

### Cellule 2 — Tableau AUROC synthèse (6×4)

| Modèle | by_equip refit | by_equip accum | by_loc refit |
|--------|:--------------:|:--------------:|:------------:|
| HDC | — | — | — |
| TinyOL AE | — | — | — |
| KMeans | — | — | — |
| Mahalanobis | — | — | — |
| DBSCAN | — | — | — |
| EWC one-class | — | — | — |

### Cellule 3 — Figures par modèle

Pour chaque modèle : courbe AUROC par tâche (refit vs accumulate) sur by_equipment.

### Cellule 4 — Comparaison RAM (barplot)

Barplot `ram_peak_bytes` pour les 6 modèles, ligne de référence 64 Ko.

### Cellule 5 — Analyse avg_forgetting

Barplot `avg_forgetting` refit vs accumulate pour les 6 modèles.

### Cellule 6 — Conclusions préliminaires

Synthèse : quel(s) modèle(s) dominent sur Monitoring ? Quel impact de la stratégie CL ?

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/monitoring/
├── auroc_table_monitoring.png
├── auroc_per_task_{model}.png  (×6)
├── ram_comparison_monitoring.png
└── forgetting_comparison_monitoring.png
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC 6×3 (ou 6×4 avec by_loc accum) présent
- [ ] Figures sauvegardées dans `notebooks/figures/anomaly_detection/monitoring/`
- [ ] Section "Conclusions préliminaires" rédigée (≥ 3 observations)
- [ ] Référence aux numéros d'expériences dans chaque cellule

## Statut

⬜ À faire
