# Sprint 18 — Anomaly Detection Equipment Monitoring : loader + 6 modèles + mise à jour summary

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 18 |
| **Semaine** | 21–27 mai 2026 |
| **Priorité globale** | 🔴 Critique — Phase Anomaly Detection Extension (Sprints 18–19) |
| **Durée estimée totale** | ~9h (budget ~10h) |
| **Dépendances** | Sprint 17 terminé — exp_143–148 CWRU validées ; `EWCOneClassDetector` implémenté (S14-01) ; décision scénario Equipment Monitoring (S18-01, bloquante) |

---

## Objectif

Déployer les 6 modèles d'anomaly detection sur le dataset **Equipment Monitoring** (~50% normal) et produire le notebook d'analyse + mettre à jour le notebook de synthèse cross-dataset. Equipment Monitoring est le cas le plus favorable en ratio normal/faulty, ce qui permet de valider les modèles dans des conditions proches des hypothèses one-class.

**Critère de succès** : loader Equipment Monitoring anomaly detection fonctionnel ; exp_149–154 enregistrées ; notebook Equipment Monitoring livré ; `summary_anomaly_detection.ipynb` mis à jour avec les résultats Sprint 18.

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S18-01 | Décision et documentation scénario CL Equipment Monitoring anomaly detection : by_equipment_type (Pump → Turbine → Compressor) | 🔴 | `docs/context/datasets.md` | 0.5h | — |
| S18-02 | Implémenter `get_equipment_monitoring_dataloaders_anomaly_detection()` — train=normal, test=normal+faulty, scénario by_equipment_type | 🔴 | `src/data/equipment_monitoring_dataset.py` | 2h | S18-01 |
| S18-03 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc Equipment Monitoring (INPUT_DIM=4, split_strategy, equipment_types) | 🔴 | `configs/unsupervised_config.yaml` | 0.5h | S18-02 |
| S18-04 | exp_149–154 — 6 modèles Equipment Monitoring refit (batch) | 🔴 | `experiments/exp_149/` → `experiments/exp_154/` | 2h | S18-02, S18-03 |
| S18-05 | Notebook Equipment Monitoring — 6 modèles refit, AUROC par tâche, analyse impact ratio normal/faulty (~50%) | 🔴 | `notebooks/cl_eval/equipment_monitoring_anomaly_detection/notebook_equipment_monitoring_anomaly_detection.ipynb` | 2h | S18-04 |
| S18-06 | Mise à jour notebook récapitulatif cross-dataset — ajout résultats Equipment Monitoring (Sprints 17+18) | 🟡 | `notebooks/cl_eval/summary_anomaly_detection.ipynb` | 1h | S17-06, S18-05 |
| S18-07 | Tests `get_equipment_monitoring_dataloaders_anomaly_detection()` (shapes, ratio classes, nombre de tâches) | 🟡 | `tests/test_equipment_anomaly.py` | 1h | S18-02 |
| S18-08 | exp_149b–154b — 6 modèles Equipment Monitoring accumulate (si temps restant) | 🟢 | `experiments/exp_149b–154b/` | 2h | S18-04 |

> Détail : S1801_equipment_split_decision.md · S1802_equipment_anomaly_loader.md · S1803_equipment_config_update.md · S1804_exp149_154_equipment_refit.md · S1805_notebook_equipment_monitoring.md · S1806_notebook_summary_update.md · S1807_tests_equipment_anomaly.md · S1808_exp149b_154b_accumulate.md

---

## Numérotation expériences

### Equipment Monitoring — 6 modèles refit

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_149 | HDC | refit | `configs/hdc_config.yaml` | ⬜ |
| exp_150 | TinyOL AE | refit | `configs/tinyol_config.yaml` | ⬜ |
| exp_151 | KMeans | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_152 | Mahalanobis | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_153 | DBSCAN | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_154 | EWC one-class | refit | `configs/ewc_oneclass_config.yaml` | ⬜ |

### Equipment Monitoring — 6 modèles accumulate (🟢 si temps)

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_149b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_150b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_151b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_152b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_153b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_154b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Critères d'acceptation

- [ ] S18-01 : scénario CL Equipment Monitoring documenté dans `docs/context/datasets.md`
- [ ] S18-02 : `get_equipment_monitoring_dataloaders_anomaly_detection()` retourne 3 tâches (Pump / Turbine / Compressor) ; train_loader ne contient que des échantillons normaux
- [ ] S18-03 : bloc `equipment_monitoring:` dans `configs/unsupervised_config.yaml` avec `INPUT_DIM: 4`
- [ ] S18-04 : exp_149–154 exécutées, `metrics_anomaly.json` présents
- [ ] S18-05 : Notebook Equipment Monitoring exécutable, analyse du ratio 50% normal documentée
- [ ] S18-06 : `summary_anomaly_detection.ipynb` mis à jour avec les données Sprint 18
- [ ] S18-07 : `pytest tests/test_equipment_anomaly.py -v` → 100% pass

---

## Livrable sprint 18

- **`get_equipment_monitoring_dataloaders_anomaly_detection()`** dans `src/data/equipment_monitoring_dataset.py`
- **6 expériences** (exp_149–154)
- **Notebook Equipment Monitoring** `notebook_equipment_monitoring_anomaly_detection.ipynb`
- **`summary_anomaly_detection.ipynb`** mis à jour (Sprints 17+18)
- **`tests/test_equipment_anomaly.py`**

---

## Questions ouvertes

- `TODO(arnaud)` : Equipment Monitoring — le ratio ~50% normal est favorable. Faut-il tester un sous-échantillonnage des normaux (à 20% comme CWRU) pour étudier l'impact du ratio ? Ou garder le dataset tel quel pour maximiser la qualité des modèles one-class ?
- `TODO(fred)` : Le scénario by_equipment_type (Pump → Turbine → Compressor) représente-t-il un déploiement industriel réaliste pour Edge Spectrum ? Y a-t-il un ordre préférentiel ?

---

> **Après ce sprint** : mettre à jour `docs/roadmap.md` (S18 ✅). Commencer Sprint 19 (Pronostia). Le notebook `summary_anomaly_detection.ipynb` sera finalisé en S19-06 après intégration des résultats Pronostia.
