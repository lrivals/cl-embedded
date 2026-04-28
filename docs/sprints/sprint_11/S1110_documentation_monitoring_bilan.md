# S11-10 — Documentation Sprint 11 : bilan partie Monitoring

| Champ | Valeur |
|-------|--------|
| **ID** | S11-10 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟢 Nice-to-have |
| **Durée estimée** | 0.5h |
| **Dépendances** | S11-03, S11-04, S11-05, S11-07, S11-08, S11-09 (notebooks exécutés) |
| **Fichier cible** | `docs/sprints/sprint_11/` (ce fichier + mise à jour S1100) |

---

## Objectif

Documenter les résultats obtenus sur le dataset Equipment Monitoring (single-task online,
contribution des variables) une fois les notebooks S11-03 à S11-09 exécutés avec succès.

---

## Contenu à renseigner

### Tableau de synthèse à compléter

| Modèle | Exp | Accuracy (batch) | AUC prequential | Top feature | Statut notebook |
|--------|-----|:----------------:|:---------------:|-------------|:---------------:|
| EWC    | exp_030 | — | — | — | ⬜ |
| HDC    | exp_031 | — | — | — | ⬜ |
| TinyOL | exp_032 | — | — | — | ⬜ |
| KMeans | exp_033 | — | — | — | ⬜ |
| Mahalanobis | exp_034 | — | — | — | ⬜ |
| DBSCAN | exp_035 | — | — | — | ⬜ |

### JSONs à vérifier

```
experiments/exp_030_ewc_monitoring_single_task/results/feature_importance.json       ⬜
experiments/exp_031_hdc_monitoring_single_task/results/feature_importance.json       ⬜
experiments/exp_032_tinyol_monitoring_single_task/results/feature_importance.json    ⬜
experiments/exp_033_kmeans_monitoring_single_task/results/feature_importance.json    ⬜
experiments/exp_034_mahalanobis_monitoring_single_task/results/feature_importance.json ⬜
experiments/exp_035_dbscan_monitoring_single_task/results/feature_importance.json    ⬜
```

### Questions à répondre après exécution

- Convergence entre modèles supervisés et non supervisés sur le classement des variables ?
  (attendu : `vibration` > `temperature` > `pressure` > `humidity`)
- L'écart batch / prequential est-il faible (< 3% accuracy) ?
  Si oui → le mode online strict est viable sur Monitoring, argument pour le manuscrit.
- DBSCAN : le score proxy de distance est-il suffisamment discriminant (AUC > 0.7) ?
  Si non → le marquer "baseline exploratoire" dans la section résultats.

---

## Mise à jour à faire dans S1100

Une fois les notebooks exécutés, mettre à jour la section **Statut** de :
- [S1102_notebooks_single_task_online.md](S1102_notebooks_single_task_online.md) : passer les 3 entrées `⬜` en `✅`
- [S1103_notebooks_single_task_online_unsupervised.md](S1103_notebooks_single_task_online_unsupervised.md) : passer les 3 entrées `⬜` en `✅`

---

## Statut

⬜ À compléter après exécution des notebooks S11-03 à S11-09
