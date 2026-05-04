# Sprint 16 — Anomaly Detection CWRU : loader + 6 modèles + clôture Phase Anomaly Detection

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 16 |
| **Semaine** | 18–20 mai 2026 |
| **Priorité globale** | 🔴 Critique — clôture Phase Anomaly Detection (Sprints 13–16) |
| **Durée estimée totale** | ~9h (budget ~10h) |
| **Dépendances** | Sprint 15 terminé — `get_pronostia_dataloaders_anomaly_detection()` validé ; `EWCOneClassDetector` implémenté (S14-01) ; décision scénario CWRU (S16-01, bloquante, nécessite réponse `TODO(arnaud)`) |

---

## Objectif

Déployer les 6 modèles d'anomaly detection sur le dataset **CWRU** (~20% normal) et produire le notebook de clôture cross-dataset (Monitoring / Pronostia / CWRU). CWRU est le cas le plus difficile : peu de données normales d'entraînement, split potentiellement non-stationnaire. Le sprint conclut la Phase Anomaly Detection du projet.

**Critère de succès** : loader CWRU anomaly detection fonctionnel ; exp_143–148 enregistrées ; notebook CWRU livré ; tableau récapitulatif cross-dataset produit pour le manuscrit.

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S16-01 | Décision et documentation scénario CL CWRU anomaly detection : by_fault_type ou by_severity (nécessite réponse TODO(arnaud)) | 🔴 | `docs/context/datasets.md` | 0.5h | Réponse arnaud |
| S16-02 | Implémenter `get_cwru_dataloaders_anomaly_detection()` — train=normal (classe "Time_Normal"), test=normal+faulty, scénario retenu en S16-01 | 🔴 | `src/data/cwru_dataset.py` | 2h | S16-01 |
| S16-03 | Mettre à jour `configs/unsupervised_config.yaml` avec bloc CWRU (input_dim=9, split_strategy, fault_types ou severities) | 🔴 | `configs/unsupervised_config.yaml` | 0.5h | S16-02 |
| S16-04 | exp_143–148 — 6 modèles CWRU refit (batch) | 🔴 | `experiments/exp_143/` → `experiments/exp_148/` | 2h | S16-02, S16-03 |
| S16-05 | Notebook CWRU — 6 modèles refit, AUROC par tâche, analyse impact ratio normal/faulty (~20%) | 🔴 | `notebooks/cl_eval/cwru_anomaly_detection/notebook_cwru_anomaly_detection.ipynb` | 2h | S16-04 |
| S16-06 | Notebook récapitulatif cross-dataset — AUROC par modèle × dataset (Monitoring / Pronostia / CWRU), refit vs accumulate | 🟡 | `notebooks/cl_eval/summary_anomaly_detection.ipynb` | 2h | S14-10, S15-06, S16-05 |
| S16-07 | Tests `get_cwru_dataloaders_anomaly_detection()` (shapes, ratio classes, nombre de tâches) | 🟡 | `tests/test_cwru_anomaly.py` | 1h | S16-02 |
| S16-08 | exp_143b–148b — 6 modèles CWRU accumulate (si temps restant) | 🟢 | `experiments/exp_143b–148b/` | 2h | S16-04 |

> Détail : S1601_cwru_split_decision.md · S1602_cwru_anomaly_loader.md · S1603_cwru_config_update.md · S1604_exp143_148_cwru_refit.md · S1605_notebook_cwru.md · S1606_notebook_summary_crossdataset.md · S1607_tests_cwru_anomaly.md

---

## Numérotation expériences

### CWRU — 6 modèles refit (scénario à confirmer en S16-01)

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_143 | HDC | refit | `configs/hdc_config.yaml` | ⬜ |
| exp_144 | TinyOL AE | refit | `configs/tinyol_config.yaml` | ⬜ |
| exp_145 | KMeans | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_146 | Mahalanobis | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_147 | DBSCAN | refit | `configs/unsupervised_config.yaml` | ⬜ |
| exp_148 | EWC one-class | refit | `configs/ewc_oneclass_config.yaml` | ⬜ |

### CWRU — 6 modèles accumulate (🟢 si temps)

| Exp | Modèle | Stratégie | Config | Statut |
|-----|--------|-----------|--------|--------|
| exp_143b | HDC | accumulate | `configs/hdc_config.yaml` | ⬜ |
| exp_144b | TinyOL AE | accumulate | `configs/tinyol_config.yaml` | ⬜ |
| exp_145b | KMeans | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_146b | Mahalanobis | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_147b | DBSCAN | accumulate | `configs/unsupervised_config.yaml` | ⬜ |
| exp_148b | EWC one-class | accumulate | `configs/ewc_oneclass_config.yaml` | ⬜ |

---

## Critères d'acceptation

- [ ] S16-01 : scénario CL CWRU anomaly detection documenté dans `docs/context/datasets.md` avec justification
- [ ] S16-02 : `get_cwru_dataloaders_anomaly_detection()` retourne le bon nombre de tâches (3 si by_fault_type, 3 si by_severity) ; train_loader ne contient que la classe "Normal"
- [ ] S16-03 : bloc `cwru:` dans `configs/unsupervised_config.yaml` avec `INPUT_DIM: 9` et `SPLIT_STRATEGY`
- [ ] S16-04 : exp_143–148 exécutées, `metrics_anomaly.json` présents
- [ ] S16-05 : Notebook CWRU exécutable, analyse du ratio 20% normal documentée
- [ ] S16-06 : Notebook cross-dataset exécutable, tableau AUROC 6 modèles × 3 datasets présent
- [ ] S16-07 : `pytest tests/test_cwru_anomaly.py -v` → 100% pass

---

## Livrable sprint 16

- **`get_cwru_dataloaders_anomaly_detection()`** dans `src/data/cwru_dataset.py`
- **6 expériences** (exp_143–148) — clôture numérotation Phase Anomaly Detection
- **Notebook CWRU** `notebook_cwru_anomaly_detection.ipynb`
- **Notebook synthèse** `summary_anomaly_detection.ipynb` — résultat central Phase Anomaly Detection
- **`tests/test_cwru_anomaly.py`**

---

## Questions ouvertes

- `TODO(arnaud)` : CWRU anomaly detection — scénario **by_fault_type** (Ball → Inner Race → Outer Race) ou **by_severity** (0.007" → 0.014" → 0.021") ? Pour one-class learning, le scénario by_severity (drift progressif de sévérité) est plus naturel et modélise l'évolution d'un défaut dans le temps. By_fault_type est plus représentatif d'un déploiement industriel incrémental (nouveau type de défaut rencontré). **Décision bloquante pour S16-02 — répondre avant le 18 mai 2026.**
- `TODO(arnaud)` : CWRU contient ~20% de données normales (1 roulement Normal sur 10 fichiers). Avec si peu de données normales d'entraînement, le seuil de reconstruction (percentile 95 du MSE normal) risque d'être instable pour EWC one-class et TinyOL AE. Faut-il abaisser le percentile (ex. 80%) ou sur-échantillonner les normaux ?
- `TODO(fred)` : Le tableau cross-dataset (S16-06) constitue le résultat central de la Phase Anomaly Detection pour la validation industrielle Edge Spectrum. Quel critère AUROC est considéré comme acceptable pour envisager un portage sur STM32N6 (AUROC ≥ 0.90 ? ≥ 0.95 ?) ?

---

> **Après ce sprint** : mettre à jour `docs/roadmap_phase1.md` (S16 ✅, Phase Anomaly Detection clôturée). Mettre à jour `experiments_tracker.md` avec exp_143–148. Commencer la rédaction de la section Anomaly Detection du manuscrit à partir des notebooks de synthèse (S14-10, S15-06, S16-06).
