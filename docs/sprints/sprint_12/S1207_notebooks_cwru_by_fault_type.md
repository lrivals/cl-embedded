# S12-07 — Notebooks CL `cwru_by_fault_type/` (6 modèles + comparison)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-07 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S12-05 (exp_074–079 exécutées) |
| **Fichiers cibles** | `notebooks/cl_eval/cwru_by_fault_type/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan,comparison}.ipynb` |

---

## Objectif

Produire 7 notebooks d'analyse pour le scénario `by_fault_type` : un par modèle (6) et un notebook de comparaison croisée. Chaque notebook charge les résultats de l'expérience correspondante et génère les figures standardisées permettant de comparer les comportements CL entre modèles sur le dataset CWRU.

---

## Liste des notebooks

| Fichier | Expérience source | Priorité |
|---------|------------------|---------|
| `ewc.ipynb` | exp_074 | 🔴 |
| `hdc.ipynb` | exp_075 | 🔴 |
| `tinyol.ipynb` | exp_076 | 🔴 |
| `kmeans.ipynb` | exp_077 | 🔴 |
| `mahalanobis.ipynb` | exp_078 | 🔴 |
| `dbscan.ipynb` | exp_079 | 🔴 |
| `comparison.ipynb` | exp_074–079 | 🔴 |

---

## Structure commune par notebook modèle (8 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, chargement `metrics_cl.json` depuis `experiments/exp_07X/results/` |
| 2 | Résultats globaux : `acc_final`, `avg_forgetting`, `backward_transfer`, `n_params`, `ram_peak_bytes` |
| 3 | Courbe d'accuracy par tâche au fil de l'entraînement (accuracy matrix) |
| 4 | Barplot `per_task_acc` : Task 1 (Ball) / Task 2 (IR) / Task 3 (OR) en fin de séquence |
| 5 | Heatmap accuracy matrix `R[i,j]` — accuracy sur tâche i après entraînement sur tâche j |
| 6 | Courbe de forgetting : chute d'accuracy tâche 1 et 2 au fil des tâches |
| 7 | Profil mémoire : `ram_peak_bytes` vs budget 64 Ko |
| 8 | Tableau récapitulatif final (comparaison avec baseline single-task exp_068–073) |

---

## Structure du notebook `comparison.ipynb`

| Section | Contenu |
|---------|---------|
| 1 | Chargement de tous les `metrics_cl.json` (exp_074–079) |
| 2 | Tableau comparatif : AA / AF / BWT / RAM / latence pour les 6 modèles |
| 3 | Barplot groupé AA (6 modèles × 3 tâches) |
| 4 | Scatter AF vs RAM — trade-off oubli / contrainte embarquée |
| 5 | Ranking des modèles par `acc_final − avg_forgetting` |
| 6 | Comparaison avec baseline single-task (delta AF par rapport à exp_068–073) |

---

## Figures attendues

Toutes sauvegardées dans `notebooks/figures/cl_evaluation/{model}/cwru/by_fault_type/` :

```
accuracy_matrix.png
per_task_accuracy_bar.png
forgetting_curve.png
ram_vs_budget.png
summary_table.png
```

Et dans `notebooks/figures/cl_evaluation/comparison/cwru/by_fault_type/` :

```
comparison_aa_af_bwt.png
scatter_af_vs_ram.png
ranking_models.png
```

---

## Critères d'acceptation

- [ ] 7 notebooks présents dans `notebooks/cl_eval/cwru_by_fault_type/`
- [ ] Chaque notebook modèle s'exécute sans erreur (`Run All`)
- [ ] 5 figures sauvegardées par notebook modèle
- [ ] `comparison.ipynb` s'exécute sans erreur, 3 figures de comparaison générées
- [ ] Section 8 de chaque notebook : tableau comparatif baseline vs CL présent

## Statut

⬜ Non démarré
