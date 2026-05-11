# S18-05 — Notebook Equipment Monitoring Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S18-05 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S18-04 (exp_149–154 terminées) |
| **Fichier cible** | `notebooks/cl_eval/equipment_monitoring_anomaly_detection/notebook_equipment_monitoring_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook d'analyse Equipment Monitoring Anomaly Detection avec focus sur le comportement des modèles dans des conditions favorables (~50% normal) et comparaison avec CWRU (~10% normal).

---

## Structure du notebook

### Section 1 — Contexte Equipment Monitoring Anomaly Detection

- Rappel du dataset : 4D (température, pression, vibration, humidité), 3 types d'équipement
- Ratio normal/faulty (~50%) — conditions favorables par rapport à CWRU
- Scénario by_equipment_type : déploiement industriel incrémental réaliste

### Section 2 — Tableau AUROC synthèse

| Modèle | Pump | Turbine | Compressor | Moyenne |
|--------|:----:|:-------:|:----------:|:-------:|
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
    "CWRU (~10% normal, 9D)": auroc_cwru_mean,
    "Equipment Monitoring (~50% normal, 4D)": auroc_equipment_mean,
}
```

Barplot AUROC moyen par modèle × dataset — illustre le gain apporté par un ratio favorable.

### Section 4 — Analyse de l'oubli catastrophique (forgetting)

Avec by_equipment_type et refit : le modèle ré-apprend à chaque tâche. Documenter si avg_forgetting est significatif ou proche de zéro (attendu : proche de zéro en refit, car pas de mémorisation des tâches précédentes).

### Section 5 — RAM et latence

Barplot `ram_peak_bytes` pour les 6 modèles sur Equipment Monitoring (4D).
Ligne de référence : 64 Ko. Tous les modèles devraient être sous 64 Ko avec input_dim=4.

### Section 6 — Conclusions Equipment Monitoring

Points clés pour le manuscrit :
- Quel AUROC est atteignable avec un ratio favorable (~50%) ?
- Mahalanobis (64 B RAM FP32) atteint-il des performances comparables aux modèles plus complexes ?
- Recommandations pour un déploiement embarqué sur equipment_monitoring

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/equipment_monitoring/
├── auroc_table_equipment.png
├── auroc_vs_normal_ratio_equipment_vs_cwru.png
├── ram_equipment_monitoring.png
└── forgetting_equipment_monitoring.png
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC présent avec toutes valeurs remplies (exp_149–154)
- [ ] Section analyse ratio normal/faulty présente avec barplot cross-dataset
- [ ] Section RAM présente avec comparaison 64 Ko
- [ ] Section "Conclusions" rédigée avec ≥ 3 observations pour le manuscrit

## Statut

⬜ À faire
