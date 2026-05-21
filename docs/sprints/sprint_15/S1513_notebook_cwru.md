# S15-18 — Notebook CWRU Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **ID** | S15-18 |
| **Sprint** | Sprint 15 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S15-15, S15-16, S15-17 |
| **Fichier cible** | `notebooks/cl_eval/cwru_anomaly_detection/notebook_cwru_anomaly_detection.ipynb` |

---

## Objectif

Produire le notebook d'analyse CWRU Anomaly Detection : comparaison des 6 modèles sur les deux scénarios (by_severity + by_fault_type), analyse RAM, et conclusions pour le manuscrit.

---

## Structure du notebook

### Section 1 — Chargement résultats

Chargement des `metrics_anomaly.json` de :
- exp_143–148 (by_severity, refit)
- exp_143b–148b (by_severity, accumulate)
- exp_149–154 (by_fault_type, refit)

### Section 2 — Tableau AUROC by_severity

| Modèle | T0 (007") | T1 (014") | T2 (021") | Moyenne | RAM |
|--------|:---------:|:---------:|:---------:|:-------:|:---:|
| HDC | 0.9983 | 0.9761 | 0.9975 | **0.9906** | 8 Ko |
| TinyOL AE | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.9 Ko |
| KMeans | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 5.3 Ko |
| Mahalanobis | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.6 Ko |
| DBSCAN | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 10.4 Ko |
| EWC one-class | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.4 Ko |

### Section 3 — Comparaison refit vs accumulate (by_severity)

Tableau Δ = AUROC accumulate − AUROC refit. Résultat attendu : Δ = 0 pour tous les modèles.

Point clé manuscrit : sur CWRU, la stratégie CL est indifférente pour l'AUROC — le refit simple suffit.

### Section 4 — Tableau AUROC by_fault_type

Même structure que Section 2, avec tâches ball / inner_race / outer_race.

### Section 5 — Comparaison by_severity vs by_fault_type

Scatter ou heatmap AUROC cross-scénario : confirme que les features spectrales sont indépendantes du type de variation (sévérité vs type de défaut).

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left : by_severity AUROC par tâche
# Right : by_fault_type AUROC par tâche
# Les deux graphes devraient être quasi-identiques
```

### Section 6 — Analyse HDC (seul modèle non-parfait)

HDC atteint avg_AUROC = 0.9906 (severity) et 0.9934 (fault_type). Analyse de la tâche difficile :
- by_severity : T1 (0.014") est la sévérité intermédiaire — signatures spectrales moins distinctes
- by_fault_type : T1 (inner_race) légèrement plus difficile à détecter

```python
# Heatmap AUROC matrix pour HDC
plot_auroc_matrix(exp_143_matrix, title="HDC — CWRU by_severity AUROC matrix")
```

### Section 7 — RAM tous modèles (9D)

Barplot RAM pour les 6 modèles sur CWRU (input_dim=9). Ligne de référence à 64 Ko.

Point clé : **tous les modèles respectent 64 Ko sur CWRU** (y compris DBSCAN = 10.4 Ko). Contraste avec Pronostia où DBSCAN dépassait 197 Ko.

```python
# Barplot RAM
models = ["HDC", "TinyOL AE", "KMeans", "Mahalanobis", "DBSCAN", "EWC one-class"]
ram_bytes = [8104, 1992, 5432, 1644, 10674, 1480]
ax.axhline(65536, color="red", linestyle="--", label="64 Ko STM32N6")
```

### Section 8 — Conclusions CWRU

Points de synthèse pour le manuscrit :
1. CWRU est le cas le plus favorable malgré seulement ~10% de données normales
2. Les features spectrales 9D permettent une séparation quasi-parfaite (5/6 modèles à AUROC = 1.0)
3. Le scénario by_severity et by_fault_type donnent des résultats identiques — les features sont indépendantes des axes de variation
4. HDC seul est légèrement en dessous de 1.0 — signature d'une représentation hyperdimensionnelle moins précise pour les sévérités intermédiaires
5. Tous les modèles respectent la contrainte RAM 64 Ko — CWRU est le seul dataset où DBSCAN est déployable sur STM32N6

---

## Figures à sauvegarder

```
notebooks/figures/cl_evaluation/anomaly_detection/cwru/
├── auroc_table_cwru_severity.png
├── auroc_table_cwru_fault_type.png
├── auroc_refit_vs_accumulate_cwru.png
├── severity_vs_fault_type_comparison.png
├── hdc_auroc_matrix_cwru.png
└── ram_cwru_all_models.png
```

---

## Critères d'acceptation

- [x] Notebook exécutable end-to-end sans erreur
- [x] Tableaux AUROC présents pour les deux scénarios (exp_143–148, exp_149–154)
- [x] Tableau refit vs accumulate présent (section 3)
- [x] Barplot RAM produit avec ligne de référence 64 Ko (section 7)
- [x] Section "Conclusions CWRU" rédigée avec ≥ 4 observations pour le manuscrit

## Statut

✅ Terminé

## Bilan

Notebook exécuté end-to-end (toutes les cellules ont des outputs). Figures générées dans `notebooks/figures/cl_evaluation/anomaly_detection/cwru/`. Les deux scénarios (by_severity et by_fault_type) sont couverts avec comparaison croisée. Section 8 "Conclusions CWRU" rédigée — 5 observations clés pour le manuscrit, dont la confirmation que tous les modèles respectent la contrainte RAM 64 Ko sur CWRU (inclus DBSCAN, exclu sur Pronostia).
