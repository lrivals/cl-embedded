# S17-05 — Notebook CWRU Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S17-05 |
| **Sprint** | Sprint 17 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S17-04 (exp_143–148 terminées) |
| **Fichier cible** | `notebooks/cl_eval/cwru_anomaly_detection/notebook_cwru_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook d'analyse CWRU Anomaly Detection avec focus sur l'impact du faible ratio de données normales (~20% global, ~77 echantillons/tâche) sur les performances AUROC des 6 modèles.

---

## Structure du notebook

### Section 1 — Contexte CWRU Anomaly Detection

- Rappel du ratio normal/faulty (~10% normaux)
- Nombre de données normales par tâche (~77)
- Différence avec Monitoring (~50% normaux) et Pronostia (~90% normaux)

### Section 2 — Tableau AUROC synthèse

| Modèle | Tâche 1 | Tâche 2 | Tâche 3 | Moyenne |
|--------|:-------:|:-------:|:-------:|:-------:|
| HDC | — | — | — | — |
| TinyOL AE | — | — | — | — |
| KMeans | — | — | — | — |
| Mahalanobis | — | — | — | — |
| DBSCAN | — | — | — | — |
| EWC one-class | — | — | — | — |

### Section 3 — Analyse impact ratio normal/faulty

Comparaison des AUROC moyens par dataset :

```python
auroc_by_dataset = {
    "Monitoring (~50% normal, 4D)": auroc_monitoring_mean,
    "Pronostia (~90% normal, 13D)": auroc_pronostia_mean,
    "CWRU (~10% normal, 9D)": auroc_cwru_mean,
}
```

Barplot AUROC moyen par modèle × dataset — révèle l'impact du ratio normal.

### Section 4 — Distribution des scores

Pour le meilleur et le pire modèle CWRU : histogramme des scores d'anomalie pour les données normales vs faulty. Visualise le recouvrement des distributions.

### Section 5 — Conclusions CWRU

Points clés pour le manuscrit :
- Impact du ratio normal sur la qualité de détection
- Quel modèle est le plus robuste au faible nombre d'échantillons d'entraînement ?
- Recommandations pour un déploiement embarqué avec peu de données normales

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/cwru/
├── auroc_table_cwru.png
├── score_distribution_best_model.png
├── score_distribution_worst_model.png
└── auroc_vs_normal_ratio.png
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC présent avec toutes valeurs remplies (exp_143–148)
- [ ] Section analyse ratio normal/faulty présente avec barplot cross-dataset
- [ ] Section "Conclusions CWRU" rédigée avec ≥ 3 observations pour le manuscrit

## Statut

⬜ À faire
