# S12-08 — Notebooks CL `cwru_by_severity/` (6 modèles + comparison)

| Champ | Valeur |
|-------|--------|
| **ID** | S12-08 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 4h |
| **Dépendances** | S12-06 (exp_080–085 exécutées) |
| **Fichiers cibles** | `notebooks/cl_eval/cwru_by_severity/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan,comparison}.ipynb` |

---

## Objectif

Produire 7 notebooks d'analyse pour le scénario `by_severity` : un par modèle (6) et un notebook de comparaison croisée. La structure est identique à S12-07 (`cwru_by_fault_type`), avec les adaptations propres au scénario de dégradation progressive (tâches nommées 0.007" / 0.014" / 0.021" au lieu de Ball / IR / OR).

L'intérêt scientifique supplémentaire : comparer AF et BWT entre `by_fault_type` et `by_severity` — un drift de sévérité est plus doux qu'un changement de type de défaut, ce qui devrait se traduire par un oubli catastrophique moindre.

---

## Liste des notebooks

| Fichier | Expérience source | Priorité |
|---------|------------------|---------|
| `ewc.ipynb` | exp_080 | 🟡 |
| `hdc.ipynb` | exp_081 | 🟡 |
| `tinyol.ipynb` | exp_082 | 🟡 |
| `kmeans.ipynb` | exp_083 | 🟡 |
| `mahalanobis.ipynb` | exp_084 | 🟡 |
| `dbscan.ipynb` | exp_085 | 🟡 |
| `comparison.ipynb` | exp_080–085 + cross-scenario | 🟡 |

---

## Structure commune par notebook modèle (8 sections)

Identique à S12-07, avec les adaptations suivantes :

| Section | Adaptation by_severity |
|---------|----------------------|
| 4 | Barplot `per_task_acc` : Task 1 (0.007") / Task 2 (0.014") / Task 3 (0.021") |
| 6 | Courbe de forgetting : gradient de sévérité — chute plus progressive attendue vs by_fault_type |
| 8 | Tableau : comparaison baseline (exp_068–073) **et** by_fault_type (exp_074–079) |

---

## Structure du notebook `comparison.ipynb`

| Section | Contenu |
|---------|---------|
| 1 | Chargement de tous les `metrics_cl.json` (exp_080–085) |
| 2 | Tableau comparatif : AA / AF / BWT / RAM / latence pour les 6 modèles |
| 3 | Barplot groupé AA (6 modèles × 3 tâches sévérité) |
| 4 | Scatter AF vs RAM |
| 5 | **Cross-scenario** : AF(by_severity) vs AF(by_fault_type) par modèle — drift doux vs dur |
| 6 | Ranking des modèles par `acc_final − avg_forgetting` |

---

## Figures attendues

Toutes sauvegardées dans `notebooks/figures/cl_evaluation/{model}/cwru/by_severity/` :

```
accuracy_matrix.png
per_task_accuracy_bar.png
forgetting_curve.png
ram_vs_budget.png
summary_table.png
```

Et dans `notebooks/figures/cl_evaluation/comparison/cwru/by_severity/` :

```
comparison_aa_af_bwt.png
scatter_af_vs_ram.png
cross_scenario_af_comparison.png   # by_severity vs by_fault_type
```

---

## Critères d'acceptation

- [ ] 7 notebooks présents dans `notebooks/cl_eval/cwru_by_severity/`
- [ ] Chaque notebook modèle s'exécute sans erreur (`Run All`)
- [ ] 5 figures sauvegardées par notebook modèle
- [ ] `comparison.ipynb` inclut la section cross-scenario (AF by_severity vs by_fault_type)
- [ ] Section 8 : tableau comparatif baseline vs by_severity vs by_fault_type présent

## Statut

⬜ Non démarré
