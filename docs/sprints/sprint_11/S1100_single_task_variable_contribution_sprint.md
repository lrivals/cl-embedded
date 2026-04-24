# Sprint 11 — Single-Task Online + Contribution des Variables (Dataset 2)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 11 |
| **Priorité globale** | 🟡 Important — enrichissement évaluation + interprétabilité modèles |
| **Durée estimée totale** | ~17h |
| **Dépendances** | Sprint 10 terminé (exp_030–035 existent), `src/evaluation/` opérationnel |

---

## Objectif

Ajouter des expériences **single-task en mode online strict** (un échantillon à la fois, évaluation prequential)
sur le dataset Equipment Monitoring pour les **six modèles** : trois supervisés CL (EWC, HDC, TinyOL) et
trois non supervisés (KMeans, Mahalanobis, DBSCAN), avec une
**analyse de contribution individuelle des variables d'entrée** (temperature, pressure, vibration, humidity).

Contexte scientifique : les expériences exp_030–032 (supervisés) et exp_033–035 (non supervisés) utilisent
le mode batch. Ce sprint ajoute le mode **online pur** (prequential predict-then-update), plus fidèle
au scénario MCU embarqué, et introduit l'interprétabilité par importance des variables.
Pour les modèles non supervisés, la stratégie **refit** (réentraînement sur buffer normal croissant)
est testée en priorité avant accumulate.

**Critère de succès** : 6 notebooks exécutables sans erreur, figures sauvegardées dans
`notebooks/figures/cl_evaluation/*/monitoring/single_task/`, JSON d'importance dans chaque `exp_030-035/results/`.

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:--------:|---------------|:----------:|-------------|
| S11-01 | Implémenter `feature_importance.py` (permutation, gradient, masking, plots) | 🔴 | `src/evaluation/feature_importance.py` | 3h | — |
| S11-02 | Exporter depuis `src/evaluation/__init__.py` | 🔴 | `src/evaluation/__init__.py` | 0.5h | S11-01 |
| S11-03 | Notebook EWC single-task online + contribution variables | 🔴 | `notebooks/cl_eval/monitoring_single_task/ewc.ipynb` | 2h | S11-01 |
| S11-04 | Notebook HDC single-task online + contribution variables | 🟡 | `notebooks/cl_eval/monitoring_single_task/hdc.ipynb` | 2h | S11-01 |
| S11-05 | Notebook TinyOL single-task online + MonitoringAutoencoder | 🟡 | `notebooks/cl_eval/monitoring_single_task/tinyol.ipynb` | 3h | S11-01 |
| S11-07 | Notebook KMeans single-task online + contribution variables (refit) | 🟡 | `notebooks/cl_eval/monitoring_single_task/kmeans.ipynb` | 2h | S11-01 |
| S11-08 | Notebook Mahalanobis single-task online + contribution variables (refit) | 🟡 | `notebooks/cl_eval/monitoring_single_task/mahalanobis.ipynb` | 2h | S11-01 |
| S11-09 | Notebook DBSCAN single-task online + contribution variables (refit) | 🟢 | `notebooks/cl_eval/monitoring_single_task/dbscan.ipynb` | 2h | S11-01 |
| S11-10 | Documentation Sprint 11 | 🟢 | `docs/sprints/sprint_11/` | 0.5h | S11-03..09 |

**Livrable** : module `feature_importance.py` + 6 notebooks complets (10 sections) + JSONs de résultats.

---

## Questions ouvertes

- `TODO(arnaud)` : La valeur prequential est-elle la bonne métrique à reporter dans le manuscrit,
  ou préférer l'accuracy post-training sur le test set ? Les deux sont produits.
- `TODO(arnaud)` : Comparer les classements de variables entre EWC (permutation+gradient),
  HDC (permutation+masking), TinyOL (permutation+norme encodeur) et les modèles non supervisés
  (permutation sur score d'anomalie) — convergence attendue entre supervisé et non supervisé ?
- `TODO(arnaud)` : Pour les modèles non supervisés en mode online, la stratégie **refit** (réentraîner
  sur buffer normal croissant) est-elle suffisante ou faut-il comparer avec **accumulate** dans ce sprint ?
- `TODO(fred)` : Valider avec Edge Spectrum que "temperature" ou "vibration" comme variable
  principale est cohérent avec l'expertise métier (maintenance prédictive industrielle).
- `TODO(arnaud)` : DBSCAN en mode online strict est heuristique (refit sur fenêtre glissante) —
  valider si ce protocole est publiable ou s'il faut le marquer comme baseline uniquement.
