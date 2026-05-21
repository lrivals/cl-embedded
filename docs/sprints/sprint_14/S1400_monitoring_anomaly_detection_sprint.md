# Sprint 14 — Anomaly Detection Monitoring : DBSCAN + EWC one-class + accumulate v2 + by_location

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 14 |
| **Semaine** | 5–9 mai 2026 |
| **Priorité globale** | 🔴 Critique — clôture couverture Monitoring (6 modèles × 2 stratégies × 2 scénarios) |
| **Durée estimée totale** | ~22h (budget 20–30h) |
| **Dépendances** | Sprint 13 terminé — exp_086–089 (HDC/TinyOL AE/KMeans/Mahalanobis refit monitoring), `run_anomaly_detection_scenario()` fonctionnel, `DBSCANDetector` implémenté |

---

## Objectif

Compléter la couverture du dataset Monitoring pour la phase Anomaly Detection en ajoutant les deux modèles manquants (**DBSCAN** et **EWC one-class**), en produisant les variantes **accumulate (v2)** pour tous les modèles, et en étendant le tout au scénario **by_location** (5 tâches). Le sprint produit un notebook de comparaison complet à 6 modèles.

**Critère de succès** : exp_123–136 enregistrées dans `experiments/` ; notebook 6 modèles exécutable sans erreur avec tableau AUROC par modèle × stratégie × scénario ; RAM profilée pour `EWCOneClassDetector` (input_dim=4).

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S14-01 | Implémenter `EWCOneClassDetector` (autoencoder MLP + régularisation EWC sur MSE reconstruction, API `fit_task` / `predict_score` / `on_task_end`) | 🔴 | `src/models/ewc/ewc_oneclass.py` | 4h | — |
| S14-02 | Config YAML EWC one-class (hidden_dim, latent_dim, lambda_ewc, threshold_percentile, n_epochs, lr) | 🔴 | `configs/ewc_oneclass_config.yaml` | 1h | S14-01 |
| S14-03 | Vérifier/adapter `DBSCANDetector` pour `run_anomaly_detection_scenario()` (fit_task + on_task_end + predict_score, retour AUROC) | 🔴 | `src/models/unsupervised/dbscan_detector.py` | 1.5h | — |
| S14-04 | exp_123–124 — DBSCAN Monitoring by_equipment refit + accumulate | 🔴 | `experiments/exp_123/`, `experiments/exp_124/` | 0.5h | S14-03 |
| S14-05 | exp_125–126 — EWC one-class Monitoring by_equipment refit + accumulate | 🔴 | `experiments/exp_125/`, `experiments/exp_126/` | 0.5h | S14-01, S14-02 |
| S14-06 | exp_127–130 — HDC / TinyOL AE / KMeans / Mahalanobis Monitoring by_equipment accumulate (v2 manquante) | 🔴 | `experiments/exp_127–130/` | 2h | — |
| S14-07 | RAM profiling `EWCOneClassDetector` (tracemalloc, input_dim=4) + annotations `# MEM:` dans `ewc_oneclass.py` | 🔴 | `src/models/ewc/ewc_oneclass.py`, `evaluation/memory_profiler.py` | 1h | S14-01 |
| S14-08 | Étendre `get_cl_dataloaders_anomaly_detection()` au scénario by_location (wrapper sur `get_cl_dataloaders_by_location()` existant) | 🟡 | `src/data/monitoring_dataset.py` | 1.5h | — |
| S14-09 | exp_131–136 — 6 modèles × Monitoring by_location refit | 🟡 | `experiments/exp_131–136/` | 2h | S14-03, S14-01, S14-08 |
| S14-10 | Notebook 6 modèles Monitoring — tableau AUROC by_equipment (refit + accumulate) + by_location, figures par modèle | 🟡 | `notebooks/cl_eval/monitoring_anomaly_detection/notebook_anomaly_detection_6models.ipynb` | 3h | S14-04 → S14-09 |
| S14-11 | Tests unitaires `EWCOneClassDetector` (fit_task, predict_score, threshold, EWC penalty) + `DBSCANDetector` (fit + infer) | 🟡 | `tests/test_ewc_oneclass.py`, `tests/test_dbscan_detector.py` | 2h | S14-01, S14-03 |
| S14-12 | accumulate by_location v2 — 6 modèles (si temps restant) | 🟢 | `experiments/exp_13Xb/` | 1.5h | S14-08, S14-09 |

> Détail : S1401_ewc_oneclass.md · S1402_ewc_oneclass_config.md · S1403_dbscan_integration.md · S1404_exp123_124_dbscan.md · S1405_exp125_126_ewc_oneclass.md · S1406_exp127_130_accumulate_v2.md · S1407_ram_profiling.md · S1408_by_location_loader.md · S1409_exp131_136_by_location.md · S1410_notebook_6models.md · S1411_tests.md

---

## Numérotation expériences

### Monitoring by_equipment — DBSCAN

| Exp | Modèle | Stratégie | Config | Statut | AUROC |
|-----|--------|-----------|--------|--------|-------|
| exp_123 | DBSCAN | by_equipment refit | `configs/unsupervised_config.yaml` | ✅ | 0.9871 |
| exp_124 | DBSCAN | by_equipment accumulate | `configs/unsupervised_config.yaml` | ✅ | 0.9873 |

### Monitoring by_equipment — EWC one-class

| Exp | Modèle | Stratégie | Config | Statut | AUROC |
|-----|--------|-----------|--------|--------|-------|
| exp_125 | EWC one-class | by_equipment refit | `configs/ewc_oneclass_config.yaml` | ✅ | 0.9630 |
| exp_126 | EWC one-class | by_equipment accumulate | `configs/ewc_oneclass_config.yaml` | ✅ | 0.9682 |

### Monitoring by_equipment — accumulate v2 (4 modèles existants)

| Exp | Modèle | Stratégie | Config | Statut | AUROC |
|-----|--------|-----------|--------|--------|-------|
| exp_127 | HDC | by_equipment accumulate | `configs/hdc_config.yaml` | ✅ | 0.9451 |
| exp_128 | TinyOL AE | by_equipment accumulate | `configs/tinyol_config.yaml` | ✅ | 0.9628 |
| exp_129 | KMeans | by_equipment accumulate | `configs/unsupervised_config.yaml` | ✅ | 0.9845 |
| exp_130 | Mahalanobis | by_equipment accumulate | `configs/unsupervised_config.yaml` | ✅ | 0.9877 |

### Monitoring by_location — 6 modèles refit

| Exp | Modèle | Stratégie | Config | Statut | AUROC |
|-----|--------|-----------|--------|--------|-------|
| exp_131 | HDC | by_location refit | `configs/hdc_config.yaml` | ✅ | 0.9470 |
| exp_132 | TinyOL AE | by_location refit | `configs/tinyol_config.yaml` | ✅ | 0.9329 |
| exp_133 | KMeans | by_location refit | `configs/unsupervised_config.yaml` | ✅ | 0.9851 |
| exp_134 | Mahalanobis | by_location refit | `configs/unsupervised_config.yaml` | ✅ | 0.9879 |
| exp_135 | DBSCAN | by_location refit | `configs/unsupervised_config.yaml` | ✅ | 0.9857 |
| exp_136 | EWC one-class | by_location refit | `configs/ewc_oneclass_config.yaml` | ✅ | 0.9552 |

---

## Critères d'acceptation

- [x] S14-01 : `EWCOneClassDetector` importable depuis `src.models.ewc.ewc_oneclass`, API `fit_task(X_normal)` / `predict_score(X)` / `on_task_end()` conforme à `run_anomaly_detection_scenario()` ✅
- [x] S14-02 : `configs/ewc_oneclass_config.yaml` charge sans erreur ; `HIDDEN_DIM`, `LATENT_DIM`, `LAMBDA_EWC`, `THRESHOLD_PERCENTILE` présents avec valeurs par défaut ≤ 64 Ko RAM ✅
- [x] S14-03 : `DBSCANDetector.fit_task()` + `predict_score()` intégrés dans `run_anomaly_detection_scenario()` sans modification du scénario générique ✅
- [x] S14-04 : exp_123–124 exécutées, `metrics_anomaly.json` + `config_snapshot.yaml` présents ✅
- [x] S14-05 : exp_125–126 exécutées, `metrics_anomaly.json` + `config_snapshot.yaml` présents ✅
- [x] S14-06 : exp_127–130 exécutées, `metrics_anomaly.json` présents ✅
- [x] S14-07 : `ram_peak_bytes` mesuré pour EWC one-class (input_dim=4) ; annotations `# MEM:` présentes sur chaque couche de `ewc_oneclass.py` ✅
- [x] S14-08 : `get_cl_dataloaders_anomaly_detection(scenario="by_location")` retourne 5 tâches avec split train_normal / test_all ✅
- [x] S14-09 : exp_131–136 exécutées, `metrics_anomaly.json` présents ✅
- [x] S14-10 : notebook `notebook_anomaly_detection_6models.ipynb` exécuté end-to-end, tableau AUROC 6×3 (modèle × by_equip_refit / by_equip_accum / by_loc_refit) présent ✅
- [x] S14-11 : `pytest tests/test_ewc_oneclass.py tests/test_dbscan_detector.py -v` → 100% pass (13 + 12 = 25 tests) ✅

---

## Livrable sprint 14

- **`src/models/ewc/ewc_oneclass.py`** — `EWCOneClassDetector` avec annotations RAM, lambda_ewc configurable
- **`configs/ewc_oneclass_config.yaml`** — paramètres par défaut conformes contrainte 64 Ko
- **14 expériences** (exp_123–136) — 2 DBSCAN + 2 EWC OC + 4 accumulate + 6 by_location
- **Notebook** `notebook_anomaly_detection_6models.ipynb` — comparaison complète Monitoring
- **2 fichiers de tests** — `test_ewc_oneclass.py`, `test_dbscan_detector.py`

---

## Questions ouvertes

- `TODO(arnaud)` : valider l'approche `EWCOneClassDetector` — autoencoder MLP avec régularisation EWC sur la loss MSE reconstruction. Seuil d'anomalie = percentile 95 du MSE de reconstruction calculé sur les données normales d'entraînement. Alternative : EWC sur un MLP one-class (output scalaire = score de normalité). Quelle approche est préférable pour la validité scientifique du manuscrit ?
- `TODO(arnaud)` : pour le scénario by_location accumulate, le dataset DBSCAN grossit à chaque tâche (toutes les données normales accumulées). Avec 5 locations, risque de lenteur à la tâche 5. Doit-on imposer une borne max (ex. 2000 échantillons) sur le buffer d'accumulation DBSCAN ?

---

> **Après ce sprint** : mettre à jour `docs/roadmap_phase1.md` (S14 ✅). Vérifier que exp_123–136 apparaissent dans `experiments_tracker.md`. Le sprint 15 (Pronostia) peut démarrer dès S14-01 + S14-02 terminés (dépendance EWC one-class).
