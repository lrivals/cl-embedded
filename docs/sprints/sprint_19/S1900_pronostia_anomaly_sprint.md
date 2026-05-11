# Sprint 19 — Anomaly Detection Pronostia : 6 modèles + clôture summary cross-dataset

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 19 |
| **Semaine** | 28 mai – 3 juin 2026 |
| **Priorité globale** | 🔴 Critique — clôture Phase Anomaly Detection + summary final cross-dataset (Sprints 17–19) |
| **Durée estimée totale** | ~9h (budget ~10h) |
| **Dépendances** | Sprint 18 terminé — exp_149–154 Equipment Monitoring validées ; `get_pronostia_dataloaders_anomaly_detection()` déjà implémentée (Sprint 15) ; `EWCOneClassDetector` implémenté (S14-01) |

---

## Objectif

Déployer les 6 modèles d'anomaly detection sur **Pronostia FEMTO** (~90% normal, 13D) et produire le notebook de synthèse finale cross-dataset (Sprints 17+18+19). Pronostia est le cas le plus favorable en ratio normal mais le plus complexe en dimensionnalité (13 features spectrales). Ce sprint clôture définitivement la Phase Anomaly Detection.

**Critère de succès** : exp_155–160 enregistrées ; notebook Pronostia livré ; `summary_anomaly_detection.ipynb` finalisé avec les 3 datasets (CWRU + Equipment Monitoring + Pronostia).

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S19-01 | Documenter scénario CL Pronostia anomaly detection : by_bearing_condition (dégradation par roulement) | 🔴 | `docs/context/datasets.md` | 0.5h | — |
| S19-02 | Vérifier et documenter `get_pronostia_dataloaders_anomaly_detection()` — déjà implémentée en Sprint 15, adapter si nécessaire au mode anomaly detection | 🔴 | `src/data/pronostia_dataset.py` | 1h | S19-01 |
| S19-03 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc Pronostia (INPUT_DIM=13, by_bearing_condition, ratio ~90%) | 🔴 | `configs/unsupervised_config.yaml` | 0.5h | S19-02 |
| S19-04 | exp_155–160 — 6 modèles Pronostia refit (batch) | 🔴 | `experiments/exp_155/` → `experiments/exp_160/` | 2h | S19-02, S19-03 |
| S19-05 | Notebook Pronostia Anomaly Detection — 6 modèles refit, AUROC par tâche, analyse impact dimensionnalité 13D | 🔴 | `notebooks/cl_eval/pronostia_anomaly_detection/notebook_pronostia_anomaly_detection.ipynb` | 2h | S19-04 |
| S19-06 | Notebook summary final cross-dataset — AUROC × 6 modèles × 3 datasets (CWRU + Equipment + Pronostia), recommandations embarquées | 🔴 | `notebooks/cl_eval/summary_anomaly_detection.ipynb` | 2h | S18-06, S19-05 |
| S19-07 | Tests `get_pronostia_dataloaders_anomaly_detection()` en mode anomaly detection (si pas déjà couverts par Sprint 15) | 🟡 | `tests/test_pronostia_anomaly.py` | 1h | S19-02 |
| S19-08 | exp_155b–160b — 6 modèles Pronostia accumulate (si temps restant) | 🟢 | `experiments/exp_155b–160b/` | 2h | S19-04 |

> Détail : S1901_pronostia_split_decision.md · S1902_pronostia_anomaly_loader.md · S1903_pronostia_config_update.md · S1904_exp155_160_pronostia_refit.md · S1905_notebook_pronostia.md · S1906_notebook_summary_update.md · S1907_tests_pronostia_anomaly.md · S1908_exp155b_160b_accumulate.md

---

## Numérotation expériences

### Pronostia — 6 modèles refit

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_155 | HDC | refit | `configs/hdc_config.yaml` | ⬜ |
| exp_156 | TinyOL AE | refit | `configs/tinyol_config.yaml` | ⬜ |
| exp_157 | KMeans | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_158 | Mahalanobis | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_159 | DBSCAN | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_160 | EWC one-class | refit | `configs/ewc_oneclass_config.yaml` | ⬜ |

### Pronostia — 6 modèles accumulate (🟢 si temps)

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_155b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_156b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_157b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_158b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_159b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_160b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Critères d'acceptation

- [ ] S19-01 : scénario `by_bearing_condition` documenté dans `docs/context/datasets.md`
- [ ] S19-02 : `get_pronostia_dataloaders_anomaly_detection()` retourne 3 tâches ; train_loader ne contient que des échantillons normaux
- [ ] S19-03 : bloc `pronostia:` dans `configs/unsupervised_config.yaml` avec `INPUT_DIM: 13`
- [ ] S19-04 : exp_155–160 exécutées, `metrics_anomaly.json` présents
- [ ] S19-05 : Notebook Pronostia exécutable, analyse dimensionnalité 13D documentée
- [ ] S19-06 : Notebook summary final exécutable, tableau AUROC 6 modèles × 3 datasets complet, recommandations embarquées rédigées

---

## Livrable sprint 19

- **6 expériences** (exp_155–160) — clôture numérotation Phase Anomaly Detection
- **Notebook Pronostia** `notebook_pronostia_anomaly_detection.ipynb`
- **Notebook synthèse final** `summary_anomaly_detection.ipynb` — résultat central de la Phase Anomaly Detection (Sprints 13–19)
- **`tests/test_pronostia_anomaly.py`** (si non couvert par Sprint 15)

---

## Questions ouvertes

- `TODO(arnaud)` : Pronostia by_bearing_condition — la dégradation est temporelle (debut_de_vie → milieu → fin_de_vie). Avec ~90% de données normales (début de vie très représenté), les modèles ont beaucoup de normaux mais peu de données faulty pour le test. Faut-il sur-pondérer les données faulty dans l'évaluation AUROC ?
- `TODO(arnaud)` : Pronostia à 13D — Mahalanobis (cov 13×13 = 676 B @ FP32) reste sous 64 Ko mais la matrice de covariance est plus difficile à estimer avec peu de données. Faut-il augmenter `REG_COVAR` ?

---

> **Après ce sprint** : Phase Anomaly Detection clôturée. Mettre à jour `docs/roadmap.md` (Sprints 17–19 ✅). Commencer la rédaction de la section Anomaly Detection du manuscrit à partir de `summary_anomaly_detection.ipynb`.
