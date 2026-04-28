# S11-18 à S11-20 — Notebooks Feature Importance CWRU, Pronostia, Cross-Dataset

| Champ | Valeur |
|-------|--------|
| **IDs** | S11-18, S11-19, S11-20 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟡 (S11-18, S11-19), 🟢 (S11-20) |
| **Durée estimée** | 7h total |
| **Dépendances** | S11-16 (exp_100–105), S11-17 (exp_106–111) |
| **Fichiers cibles** | `notebooks/cl_eval/cwru_feature_importance/`, `notebooks/cl_eval/pronostia_feature_importance/`, `notebooks/cl_eval/cross_dataset_feature_importance.ipynb` |

---

## S11-18 — Notebooks CWRU feature importance (2 notebooks)

### Fichiers

```
notebooks/cl_eval/cwru_feature_importance/by_fault_type.ipynb   ← exp_100, 103, 106, 109
notebooks/cl_eval/cwru_feature_importance/by_severity.ipynb     ← exp_101, 104, 107, 110
```

### Structure commune (7 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, chargement des 4 JSONs du scénario (KMeans, Maha, EWC, HDC) |
| 2 | Tableau récapitulatif global — ranking des 9 features par modèle (DataFrame) |
| 3 | Barplot groupé global — `plot_feature_importance_comparison()` (4 modèles × 9 features) |
| 4 | Heatmap per-task — importance de chaque feature selon la tâche (3 tâches × 9 features) par modèle |
| 5 | Comparaison méthodes pour EWC — permutation vs gradient saliency (normalisés) |
| 6 | Comparaison méthodes pour HDC — permutation vs masking |
| 7 | Markdown — Analyse : convergence entre modèles ? Stabilité inter-tâches ? Features MCU à retenir ? |

### Questions clés à répondre dans la section 7

**by_fault_type** :

- Les 3 types de défaut (ball, inner_race, outer_race) ont-ils le même top-3 de features ?
- `kurtosis` est-il stable (indicateur classique de choc mécanique) ?

**by_severity** :

- Les features importantes changent-elles quand le défaut s'aggrave (0.007" → 0.021") ?
- `crest_factor` ou `peak` montent-ils avec la sévérité ?

---

## S11-19 — Notebook Pronostia feature importance

### Fichier

```
notebooks/cl_eval/pronostia_feature_importance/by_condition.ipynb   ← exp_102, 105, 108, 111
```

### Structure (8 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, chargement des 4 JSONs (KMeans, Maha, EWC, HDC) |
| 2 | Tableau récapitulatif global — ranking des 13 features par modèle |
| 3 | Barplot groupé global — `plot_feature_importance_comparison()` |
| 4 | Barplot groupé par canal (`acc_horiz` vs `acc_vert`) — contribution relative des deux axes |
| 5 | Heatmap per-task — importance × condition opératoire (3 conditions × 13 features) par modèle |
| 6 | Focus `temporal_position` — importance à travers les 4 modèles + discussion déploiement MCU |
| 7 | Comparaison méthodes EWC (permutation vs gradient) et HDC (permutation vs masking) |
| 8 | Markdown — Analyse : canal dominant ? temporal_position utilisable en MCU ? top features par condition ? |

### Points spécifiques à traiter

- **Canal horizontal vs vertical** : le roulement PRONOSTIA subit la charge radiale sur l'axe
  horizontal → attendu que `acc_horiz` domine. À vérifier empiriquement.
- **`temporal_position`** : si importante, l'annoter `FIXME(gap1)` — non disponible en déploiement
  MCU sans horodatage de durée de vie → retirer pour les expériences de publication.
- **Conditions opératoires** : rpm différents entre conditions → si l'importance change,
  c'est un signe de domain shift réel (justification du scénario CL by_condition).

---

## S11-20 — Notebook cross-dataset comparison

### Fichier

```
notebooks/cl_eval/cross_dataset_feature_importance.ipynb
```

### Objectif

Comparer les classements de features entre Monitoring, CWRU et Pronostia pour identifier
des **invariants cross-domaine** et des **spécificités dataset**. Argument scientifique
pour le manuscrit : les features retenues pour le MCU sont robustes à la source de données.

### Structure (5 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, chargement de tous les JSONs d'importance (exp_030-035 + 100-111) |
| 2 | Tableau de correspondance features — normalisation des noms par famille statistique (`rms`, `kurtosis`, `crest`) |
| 3 | Heatmap rank-based — rang de chaque famille de features par dataset × modèle (Spearman) |
| 4 | Barplot comparatif — top-3 features par dataset et par modèle |
| 5 | Markdown — Synthèse : quelles features sont universellement importantes ? Recommandation embarquée |

### Normalisation des noms de features pour la comparaison

| Monitoring | CWRU | Pronostia (horiz) | Famille |
|------------|------|-------------------|---------|
| `vibration` | `rms` | `rms_acc_horiz` | rms |
| `temperature` | `kurtosis` | `kurtosis_acc_horiz` | kurtosis |
| — | `crest` | `crest_factor_acc_horiz` | crest |
| — | `sd` | `std_acc_horiz` | std |

La comparaison se fait sur les familles statistiques communes (`rms`, `kurtosis`, `std`, `crest`),
pas sur les noms bruts.

---

## Résultats de sortie

```
notebooks/figures/cl_evaluation/cwru_feature_importance/
  ├── by_fault_type_global_comparison.png
  ├── by_fault_type_per_task_heatmap.png
  ├── by_severity_global_comparison.png
  └── by_severity_per_task_heatmap.png

notebooks/figures/cl_evaluation/pronostia_feature_importance/
  ├── by_condition_global_comparison.png
  ├── by_condition_channel_grouped.png
  ├── by_condition_per_task_heatmap.png
  └── temporal_position_importance.png

notebooks/figures/cl_evaluation/
  └── cross_dataset_feature_importance_heatmap.png
```

## Statut

✅ S11-18 — `cwru_feature_importance/by_fault_type.ipynb` (créé + figures produites)
✅ S11-18 — `cwru_feature_importance/by_severity.ipynb` (créé + figures produites)
✅ S11-19 — `pronostia_feature_importance/by_condition.ipynb` (créé + figures produites)
⬜ S11-20 — `cross_dataset_feature_importance.ipynb` (à créer)
