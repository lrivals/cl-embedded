# S18-06 — Mise à jour notebook récapitulatif cross-dataset (Sprints 17+18)

| Champ | Valeur |
|-------|--------|
| **ID** | S18-06 |
| **Sprint** | Sprint 18 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S17-06 (summary CWRU créé), S18-05 (notebook Equipment Monitoring terminé) |
| **Fichier cible** | `notebooks/cl_eval/summary_anomaly_detection.ipynb` |

---

## Objectif

Mettre à jour `summary_anomaly_detection.ipynb` pour intégrer les résultats Equipment Monitoring (exp_149–154) aux côtés des résultats CWRU (exp_143–148). Cette version intermédiaire du summary couvre Sprints 17 et 18 ; la version finale sera produite en S19-06 après l'ajout de Pronostia.

---

## Modifications à apporter

### Section 1 — Tableau AUROC cross-dataset (mise à jour)

Ajouter la colonne Equipment Monitoring :

```
                    CWRU           Equipment Monitoring
Modèle           refit | accum    refit | accum
──────────────────────────────────────────────────
HDC               X.XX  | —       X.XX  | —
TinyOL AE         X.XX  | —       X.XX  | —
KMeans            X.XX  | —       X.XX  | —
Mahalanobis       X.XX  | —       X.XX  | —
DBSCAN            X.XX  | —       X.XX  | —
EWC one-class     X.XX  | —       X.XX  | —
```

> Note : les colonnes `accum` seront complétées si S17-08 et S18-08 sont exécutées.

### Section 2 — RAM cross-dataset (mise à jour)

Ajouter les barres Equipment Monitoring au barplot RAM.

### Section 3 — Impact ratio normal (nouvelle section)

Scatter plot AUROC moyen vs ratio normal par dataset :
```python
ratios = {"CWRU": 0.10, "Equipment Monitoring": 0.50}
aurocs = {"CWRU": auroc_cwru_mean, "Equipment Monitoring": auroc_equip_mean}
```

Tendance attendue : AUROC croissant avec le ratio normal.

---

## Critères d'acceptation

- [ ] `summary_anomaly_detection.ipynb` exécutable après mise à jour
- [ ] Colonne Equipment Monitoring présente dans le tableau AUROC cross-dataset
- [ ] Section "Impact ratio normal" ajoutée avec scatter plot CWRU vs Equipment
- [ ] `TODO` marqué pour l'ajout Pronostia (S19-06)

## Statut

⬜ À faire (après S18-05)
