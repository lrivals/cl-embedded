# S12-09 — Notebook baseline single-task CWRU

| Champ | Valeur |
|-------|--------|
| **ID** | S12-09 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S12-04 (exp_068–073 exécutées) |
| **Fichier cible** | `notebooks/cl_eval/baselines/cwru_single_task.ipynb` |

---

## Objectif

Produire le notebook de synthèse des baselines single-task CWRU (exp_068–073), homologue au notebook `pronostia_single_task.ipynb` déjà existant pour PRONOSTIA. Ce notebook résume les performances maximales atteignables par chaque modèle sur CWRU sans contrainte CL — il constitue le plafond de référence pour juger du coût en performance des scénarios `by_fault_type` et `by_severity`.

---

## Structure du notebook (6 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, chargement des 6 `metrics_single_task.json` (exp_068–073) |
| 2 | Tableau comparatif : accuracy / F1 / AUC-ROC / RAM / latence pour les 6 modèles |
| 3 | Barplot groupé accuracy + F1 (supervisés vs non supervisés) |
| 4 | Scatter RAM vs accuracy — trade-off performance / contrainte embarquée |
| 5 | Tableau de synthèse CWRU vs PRONOSTIA (mêmes 6 modèles, mêmes métriques) |
| 6 | Conclusion : modèle(s) à privilégier pour les scénarios CL (justification) |

---

## Figures attendues

Sauvegardées dans `notebooks/figures/cl_evaluation/baselines/cwru/` :

```
baseline_accuracy_f1_bar.png       # barplot groupé accuracy + F1
scatter_ram_vs_accuracy.png        # trade-off RAM / performance
cwru_vs_pronostia_comparison.png   # tableau visuel cross-dataset
```

---

## Critères d'acceptation

- [x] Notebook exécuté sans erreur (`Run All`)
- [x] 6 expériences chargées (exp_068–073), tableau comparatif complet
- [x] Section 5 : tableau CWRU vs PRONOSTIA présent (au moins accuracy, F1, RAM)
- [x] 3 figures sauvegardées dans `notebooks/figures/cl_evaluation/baselines/cwru/`

## Statut

✅ Terminé
