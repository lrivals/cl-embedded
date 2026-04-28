# Sprint 11 — Single-Task Online + Contribution des Variables (Monitoring, CWRU, Pronostia)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 11 |
| **Priorité globale** | 🟡 Important — enrichissement évaluation + interprétabilité modèles |
| **Durée estimée totale** | ~35h |
| **Dépendances** | Sprint 10 terminé (exp_030–035 existent), `src/evaluation/` opérationnel, Sprint 12 terminé (exp_071–085 existent) |

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

**Extension (S11-11 → S11-22)** : étendre l'analyse de contributions aux datasets **CWRU** (9 features
statistiques de signal vibratoire) et **Pronostia** (13 features sur 2 canaux accélérométriques), avec
une analyse **per-task** pour mesurer si l'importance des features change selon le type/sévérité de défaut.

**Critère de succès** :

- 6 notebooks Monitoring exécutables sans erreur (S11-03–09), JSON d'importance dans exp_030-035
- 12 expériences CWRU/Pronostia avec `feature_importance.json` per-task (exp_100–111)
- 4 notebooks comparatifs CWRU/Pronostia + 1 notebook cross-dataset exécutables sans erreur

---

## Tâches

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:--------:|---------------|:----------:|-------------|
| S11-01 | Implémenter `feature_importance.py` (permutation, gradient, masking, plots) | ✅ | `src/evaluation/feature_importance.py` | 3h | — |
| S11-02 | Exporter depuis `src/evaluation/__init__.py` | ✅ | `src/evaluation/__init__.py` | 0.5h | S11-01 |
| S11-03 | Notebook EWC single-task online + contribution variables | ✅ | `notebooks/cl_eval/monitoring_single_task/ewc.ipynb` | 2h | S11-01 |
| S11-04 | Notebook HDC single-task online + contribution variables | ✅ | `notebooks/cl_eval/monitoring_single_task/hdc.ipynb` | 2h | S11-01 |
| S11-05 | Notebook TinyOL single-task online + MonitoringAutoencoder | ✅ | `notebooks/cl_eval/monitoring_single_task/tinyol.ipynb` | 3h | S11-01 |
| S11-07 | Notebook KMeans single-task online + contribution variables (refit) | ✅ | `notebooks/cl_eval/monitoring_single_task/kmeans.ipynb` | 2h | S11-01 |
| S11-08 | Notebook Mahalanobis single-task online + contribution variables (refit) | ✅ | `notebooks/cl_eval/monitoring_single_task/mahalanobis.ipynb` | 2h | S11-01 |
| S11-09 | Notebook DBSCAN single-task online + contribution variables (refit) | ✅ | `notebooks/cl_eval/monitoring_single_task/dbscan.ipynb` | 2h | S11-01 |
| S11-10 | Documentation Sprint 11 — partie Monitoring | ⬜ | `docs/sprints/sprint_11/` | 0.5h | S11-03..09 |
| S11-11 | Extension `feature_importance.py` — constantes CWRU/Pronostia + analyse per-task | ✅ | `src/evaluation/feature_importance.py` | 1h | S11-01 |
| S11-12 | Extension `train_kmeans.py` — export `feature_importance.json` per-task | ✅ | `scripts/train_kmeans.py` | 1.5h | S11-11 |
| S11-13 | Extension `train_mahalanobis.py` — export `feature_importance.json` per-task | ✅ | `scripts/train_mahalanobis.py` | 1.5h | S11-11 |
| S11-14 | Extension `train_ewc.py` — export importance per-task (permutation + gradient saliency) | ✅ | `scripts/train_ewc.py` | 1.5h | S11-11 |
| S11-15 | Extension `train_hdc.py` — export importance per-task (permutation + masking) | ✅ | `scripts/train_hdc.py` | 1.5h | S11-11 |
| S11-16 | Lancer exp_100–105 — KMeans + Mahalanobis CWRU/Pronostia feature importance | ✅ | `experiments/exp_100–105/` | 1h | S11-12, S11-13 |
| S11-17 | Lancer exp_106–111 — EWC + HDC CWRU/Pronostia feature importance | ✅ | `experiments/exp_106–111/` | 1h | S11-14, S11-15 |
| S11-18 | Notebooks feature importance CWRU (2 — by_fault_type + by_severity) | ✅ | `notebooks/cl_eval/cwru_feature_importance/` | 3h | S11-16, S11-17 |
| S11-19 | Notebook feature importance Pronostia (by_condition) | ✅ | `notebooks/cl_eval/pronostia_feature_importance/` | 2h | S11-16, S11-17 |
| S11-20 | Notebook cross-dataset comparison (Monitoring vs CWRU vs Pronostia) | ⬜ | `notebooks/cl_eval/cross_dataset_feature_importance.ipynb` | 2h | S11-18, S11-19 |
| S11-21 | Tests `test_feature_importance_cwru_pronostia.py` | ✅ | `tests/test_feature_importance_cwru_pronostia.py` | 1h | S11-11 |
| S11-22 | Documentation Sprint 11 — partie CWRU/Pronostia | ⬜ | `docs/sprints/sprint_11/` | 0.5h | S11-18..20 |
| S11-23 | Ablation study — retrait progressif de features (courbe nb_features vs AUC) | ⬜ | `src/evaluation/feature_importance.py` + `notebooks/cl_eval/ablation_feature_removal/` | 3h | S11-16, S11-17 |

**Livrable** :

- module `feature_importance.py` étendu + 6 notebooks Monitoring (10 sections) + JSONs exp_030-035
- 12 JSONs `feature_importance.json` per-task dans exp_100–111
- 4 notebooks comparatifs CWRU/Pronostia + 1 notebook cross-dataset
- 1 notebook ablation study (courbe nb_features vs AUC, argument Gap 2)

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
- `TODO(arnaud)` : `get_pronostia_dataloaders` n'expose pas de `test_loader` en mode CL — utiliser
  le `val_loader` pour l'importance (split stratifié) ou extraire `X_test` via `load_condition_features` ?
- `TODO(arnaud)` : L'importance de `kurtosis` est-elle stable entre by_fault_type et by_severity
  sur CWRU ? Si instable, cela invalide-t-il son utilisation comme feature MCU prioritaire ?
- `TODO(arnaud)` : `temporal_position` (Pronostia) — si son importance est élevée, il n'est pas
  disponible en déploiement MCU réel (hors supervision) → à exclure ou à traiter séparément ?
- `TODO(fred)` : Le classement CWRU (kurtosis/rms/crest dominant ?) est-il cohérent avec les
  pratiques de diagnostic vibratoire industriel chez Edge Spectrum ?
