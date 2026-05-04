# Sprint 15 — Anomaly Detection Pronostia : loader + 6 modèles + notebook

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 15 |
| **Semaine** | 12–15 mai 2026 |
| **Priorité globale** | 🔴 Critique — premier dataset de roulements en anomaly detection |
| **Durée estimée totale** | ~13h (budget ~15h) |
| **Dépendances** | Sprint 14 terminé — `EWCOneClassDetector` implémenté (S14-01), `run_anomaly_detection_scenario()` validé sur Monitoring |

---

## Objectif

Déployer les 6 modèles d'anomaly detection sur le dataset **Pronostia** (FEMTO Bearing) en créant le loader anomaly detection dédié et en exécutant les expériences exp_137–142 (refit). Le dataset Pronostia présente ~90% de données normales — cas favorable pour le one-class learning. Le sprint produit un notebook d'analyse Pronostia avec AUROC par tâche et comparaison cross-modèle.

**Critère de succès** : `get_pronostia_dataloaders_anomaly_detection()` fonctionnel sur 3 tâches by_condition ; exp_137–142 enregistrées ; notebook Pronostia exécutable avec AUROC comparatif.

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S15-01 | Implémenter `get_pronostia_dataloaders_anomaly_detection()` — filtre train=normal (`FAILURE_RATIO` configurable sur fin de séquence), test=normal+faulty, 3 tâches by_condition | 🔴 | `src/data/pronostia_dataset.py` | 2.5h | — |
| S15-02 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc Pronostia (`INPUT_DIM=13`, `FAILURE_RATIO`, `condition_ids`) | 🔴 | `configs/unsupervised_config.yaml` | 0.5h | S15-01 |
| S15-03 | exp_137 — HDC Pronostia by_condition refit | 🔴 | `experiments/exp_137/` | 0.5h | S15-01 |
| S15-04 | exp_138 — TinyOL AE Pronostia by_condition refit | 🔴 | `experiments/exp_138/` | 0.5h | S15-01 |
| S15-05 | exp_139 — KMeans Pronostia by_condition refit | 🔴 | `experiments/exp_139/` | 0.5h | S15-01 |
| S15-06 | exp_140 — Mahalanobis Pronostia by_condition refit | 🔴 | `experiments/exp_140/` | 0.5h | S15-01 |
| S15-07 | exp_141 — DBSCAN Pronostia by_condition refit | 🔴 | `experiments/exp_141/` | 0.5h | S15-01 |
| S15-08 | exp_142 — EWC one-class Pronostia by_condition refit | 🔴 | `experiments/exp_142/` | 0.5h | S14-01, S15-01, S15-02 |
| S15-09 | RAM profiling sur Pronostia (input_dim=13 — EWC one-class et HDC changent de taille vs Monitoring) | 🔴 | `evaluation/memory_profiler.py` | 1h | S15-03 → S15-08 |
| S15-10 | Notebook Pronostia — 6 modèles refit, AUROC par tâche, analyse impact ratio normal/faulty (90%) | 🟡 | `notebooks/cl_eval/pronostia_anomaly_detection/notebook_pronostia_anomaly_detection.ipynb` | 3h | S15-03 → S15-08 |
| S15-11 | exp_137b–142b — 6 modèles Pronostia by_condition accumulate (v2) | 🟡 | `experiments/exp_137b–142b/` | 2h | S15-03 → S15-08 |
| S15-12 | Tests `get_pronostia_dataloaders_anomaly_detection()` (shapes, ratio normal/faulty, 3 tâches) | 🟡 | `tests/test_pronostia_anomaly.py` | 1.5h | S15-01 |
| S15-13 | Analyse sensibilité `FAILURE_RATIO` — AUROC avec 0.05 / 0.10 / 0.20 sur KMeans (1 modèle) | 🟢 | `notebooks/cl_eval/pronostia_anomaly_detection/` | 1h | S15-05 |

> Détail : S1501_pronostia_anomaly_loader.md · S1502_pronostia_config_update.md · S1503_exp137_142_pronostia_refit.md · S1504_ram_profiling_pronostia.md · S1505_exp137b_142b_accumulate.md · S1506_notebook_pronostia.md · S1507_tests_pronostia_anomaly.md

---

## Numérotation expériences

### Pronostia by_condition — 6 modèles refit

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_137 | HDC | by_condition refit | `configs/hdc_config.yaml` | ⬜ |
| exp_138 | TinyOL AE | by_condition refit | `configs/tinyol_config.yaml` | ⬜ |
| exp_139 | KMeans | by_condition refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_140 | Mahalanobis | by_condition refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_141 | DBSCAN | by_condition refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_142 | EWC one-class | by_condition refit | `configs/ewc_oneclass_config.yaml` | ⬜ |

### Pronostia by_condition — 6 modèles accumulate (🟡 si temps)

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_137b | HDC | by_condition accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_138b | TinyOL AE | by_condition accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_139b | KMeans | by_condition accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_140b | Mahalanobis | by_condition accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_141b | DBSCAN | by_condition accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_142b | EWC one-class | by_condition accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Critères d'acceptation

- [ ] S15-01 : `get_pronostia_dataloaders_anomaly_detection()` retourne 3 tâches ; `train_loader` ne contient que label=0 (données de début de vie)
- [ ] S15-02 : `configs/unsupervised_config.yaml` contient bloc `pronostia:` avec `INPUT_DIM: 13` et `FAILURE_RATIO: 0.10`
- [ ] S15-03 → S15-08 : exp_137–142 exécutées, `metrics_anomaly.json` présents
- [ ] S15-09 : RAM mesurée pour EWC one-class Pronostia (input_dim=13) ≤ 64 Ko
- [ ] S15-10 : Notebook exécutable, AUROC par tâche visible pour 6 modèles
- [ ] S15-12 : `pytest tests/test_pronostia_anomaly.py -v` → 100% pass

---

## Livrable sprint 15

- **`get_pronostia_dataloaders_anomaly_detection()`** dans `src/data/pronostia_dataset.py`
- **`configs/unsupervised_config.yaml`** mis à jour avec bloc Pronostia
- **6 expériences** (exp_137–142) refit — + 6 accumulate (137b–142b) si temps
- **Notebook** `notebook_pronostia_anomaly_detection.ipynb`
- **`tests/test_pronostia_anomaly.py`**

---

## Questions ouvertes

- `TODO(arnaud)` : `FAILURE_RATIO=0.10` — les 10 derniers pourcents de chaque séquence de roulement sont marqués "faulty". Est-ce cohérent avec la physique de dégradation Pronostia (accélération en fin de vie) ? Un seuil sur le RUL restant (ex. RUL < 20% de la durée totale) serait-il plus pertinent ?
- `TODO(arnaud)` : Pronostia a seulement 3 conditions opératoires → 3 tâches CL. Avec si peu de tâches, l'évaluation de l'oubli catastrophique (avg_forgetting, BWT) est-elle statistiquement significative pour le manuscrit ? Faut-il simuler des sous-tâches ou accepter cette limitation ?

---

> **Après ce sprint** : mettre à jour `docs/roadmap_phase1.md` (S15 ✅). Le sprint 16 (CWRU) peut démarrer dès S15-01 terminé — le loader CWRU est indépendant du loader Pronostia.
