# S19-05 — Notebook Pronostia Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S19-05 |
| **Sprint** | Sprint 19 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S19-04 (exp_155–160 terminées) |
| **Fichier cible** | `notebooks/cl_eval/pronostia_anomaly_detection/notebook_pronostia_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook d'analyse Pronostia Anomaly Detection avec focus sur l'impact de la haute dimensionnalité (13D) et du ratio très favorable (~90% normal) sur les performances AUROC des 6 modèles.

---

## Structure du notebook

### Section 1 — Contexte Pronostia Anomaly Detection

- Rappel du dataset : 13D (features spectrales + temporelles vibration roulement)
- Ratio normal/faulty (~90%) — conditions très favorables pour le one-class learning
- Scénario by_bearing_condition : early_life → mid_life → end_of_life (gradient de dégradation)
- Comparaison avec CWRU (~10% normal, 9D) et Equipment Monitoring (~50%, 4D)

### Section 2 — Tableau AUROC synthèse

| Modèle | Early life | Mid life | End of life | Moyenne |
|--------|:----------:|:--------:|:-----------:|:-------:|
| HDC | — | — | — | — |
| TinyOL AE | — | — | — | — |
| KMeans | — | — | — | — |
| Mahalanobis | — | — | — | — |
| DBSCAN | — | — | — | — |
| EWC one-class | — | — | — | — |

### Section 3 — Impact de la dimensionnalité (13D vs 4D vs 9D)

```python
auroc_by_input_dim = {
    "Equipment Monitoring (4D, ~50% normal)": auroc_equipment_mean,
    "CWRU (9D, ~10% normal)": auroc_cwru_mean,
    "Pronostia (13D, ~90% normal)": auroc_pronostia_mean,
}
```

Deux effets antagonistes en Pronostia : ratio très favorable (+) mais dimensionnalité élevée (-). Analyser lequel domine.

### Section 4 — Évolution AUROC par condition (early → mid → end)

Lineplot AUROC par modèle en fonction de la condition (early_life, mid_life, end_of_life). Tendance attendue : AUROC croissant avec la dégradation (anomalies plus marquées en fin de vie).

### Section 5 — RAM Mahalanobis 13D

Focus sur Mahalanobis (exp_158) : `ram_peak_bytes` pour la cov 13×13 (676 B @ FP32). Vérifier que ce modèle reste sous 64 Ko même en 13D.

### Section 6 — Conclusions Pronostia

Points clés pour le manuscrit :
- Pronostia confirme-t-il la tendance AUROC ↑ quand ratio normal ↑ ?
- L'espace 13D est-il un obstacle pour les modèles géométriques (KMeans, Mahalanobis, DBSCAN) ?
- Quel modèle est le plus robuste à la haute dimensionnalité ?

---

## Figures à sauvegarder

```
notebooks/figures/anomaly_detection/pronostia/
├── auroc_table_pronostia.png
├── auroc_evolution_by_condition.png
├── auroc_vs_input_dim.png
└── ram_mahalanobis_13d.png
```

---

## Critères d'acceptation

- [ ] Notebook exécutable end-to-end sans erreur
- [ ] Tableau AUROC présent avec toutes valeurs remplies (exp_155–160)
- [ ] Section analyse dimensionnalité présente avec comparaison 4D/9D/13D
- [ ] Section "Conclusions Pronostia" rédigée avec ≥ 3 observations pour le manuscrit

## Statut

⬜ À faire
