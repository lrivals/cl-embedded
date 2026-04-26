# S12-03 — EDA CWRU (`01D_data_exploration_cwru.ipynb`)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-03 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 2h |
| **Dépendances** | S12-01 (`CWRUDataset` importable) |
| **Fichiers cibles** | `notebooks/01D_data_exploration_cwru.ipynb`, `notebooks/figures/eda/cwru/` |

---

## Objectif

Produire une analyse exploratoire complète du CWRU Bearing Dataset dans un notebook dédié, couvrant la distribution des défauts, les propriétés statistiques des features, la séparabilité entre classes et la cohérence avec les scénarios CL by_fault_type et by_severity. Ce notebook constitue la section 5 de l'EDA globale (01A = Equipment Monitoring, 01B = PRONOSTIA, 01C = Battery RUL, 01D = CWRU).

---

## Structure du notebook

| Section | Titre | Contenu |
|---------|-------|---------|
| 1 | Setup | Imports, constantes, `CWRUDataset(csv_path=...)` |
| 2 | Vue d'ensemble | Shape, dtypes, valeurs manquantes, équilibre des classes |
| 3 | Distribution des labels | Barplot Normal vs Défaillant par type de défaut et par sévérité |
| 4 | Statistiques descriptives | `describe()` par classe, boxplots des 9 features |
| 5 | Corrélations | Heatmap Spearman des 9 features |
| 6 | Scénario by_fault_type | Boxplots par tâche (Ball / IR / OR), shift de distribution |
| 7 | Scénario by_severity | Boxplots par tâche (0.007" / 0.014" / 0.021"), gradient de sévérité |
| 8 | Séparabilité | PCA 2D (Normal vs Défaillant), coloré par type, coloré par sévérité |
| 9 | Comparaison avec PRONOSTIA | Tableau de bord : N échantillons, N features, déséquilibre, type CL |

---

## Figures attendues

Toutes sauvegardées dans `notebooks/figures/eda/cwru/` :

```
class_distribution.png          # barplot Normal vs Défaillant global
class_by_fault_type.png         # barplot par type (Ball, IR, OR)
class_by_severity.png           # barplot par sévérité (007, 014, 021)
boxplot_features_by_class.png   # 9 features × 2 classes
correlation_heatmap.png         # heatmap Spearman 9×9
pca_2d_by_class.png             # PCA coloré Normal/Défaillant
pca_2d_by_fault_type.png        # PCA coloré Ball/IR/OR/Normal
pca_2d_by_severity.png          # PCA coloré par sévérité
domain_shift_fault_type.png     # distributions features Task 1→2→3 (by_fault)
domain_shift_severity.png       # distributions features Task 1→2→3 (by_severity)
```

---

## Critères d'acceptation

- [x] Notebook exécuté sans erreur de bout en bout (`Run All`)
- [x] Section 2 : aucune valeur manquante confirmée, shape `(2300, 9)` affiché (note : la spec mentionnait 2299, la valeur réelle est 2300 — voir docstring `CWRUDataset`)
- [x] 10 figures sauvegardées dans `notebooks/figures/eda/cwru/`
- [x] Section 9 : tableau comparatif CWRU / PRONOSTIA présent (12 métriques)
- [x] Aucune cellule avec `display()` de DataFrame brute > 20 lignes

## Statut

✅ Terminé — 24 avril 2026
