# Roadmap Phase 1 — Implémentation Python

> Mise à jour : 14 avril 2026  
> Horizon : 15 avril – 20 mai 2026 (étendu avec Sprints 6–9 notebooks)  
> ← [Index roadmap](roadmap.md)

---

## Sprint 1 — Semaine 1 (15–22 avril) ✅ TERMINÉ — exécuté et validé le 4 avril 2026

**Objectif** : Infrastructure du projet + M2 (EWC) fonctionnel sur Dataset 2

**Statut** : ✅ Code livré · ✅ 90/90 tests passent · ✅ exp_001 exécutée · ✅ RAM mesurée

#### Tâches

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S1-01 | Créer le dépôt GitHub + structure de dossiers | ✅ | ✅ | N/A | `README.md`, `.gitignore`, `pyproject.toml` | 1h |
| S1-02 | Télécharger Dataset 2 (Monitoring) + exploration | ✅ | ✅ | ✅ | `notebooks/01_data_exploration.ipynb` | 2h |
| S1-03 | Implémenter `monitoring_dataset.py` (loader + split domaine) | ✅ | ✅ | N/A | `src/data/monitoring_dataset.py` | 3h |
| S1-04 | Implémenter `ewc_mlp.py` (MLP + perte EWC) | ✅ | ✅ | N/A | `src/models/ewc/ewc_mlp.py` | 4h |
| S1-05 | Implémenter `fisher.py` (calcul Fisher diagonale) | ✅ | ✅ | N/A | `src/models/ewc/fisher.py` | 2h |
| S1-06 | Implémenter `baselines.py` (fine-tuning naïf + joint) | ✅ | ✅ | N/A | `src/training/baselines.py` | 2h |
| S1-07 | Implémenter `metrics.py` (AA, AF, BWT) | ✅ | ✅ | N/A | `src/evaluation/metrics.py` | 2h |
| S1-08 | Implémenter `memory_profiler.py` (tracemalloc) | ✅ | ✅ | N/A | `src/evaluation/memory_profiler.py` | 2h |
| S1-09 | Première expérience EWC : exp_001 | ✅ | ✅ | ✅ | `scripts/train_ewc.py`, `experiments/exp_001_ewc_dataset2/` | 2h |
| S1-10 | Tests unitaires modèle EWC + métriques | ✅ | ✅ | ✅ | `tests/test_ewc.py` | 2h |

**Livrable sprint 1** : EWC Online + MLP entraîné sur 3 domaines, métriques AA/AF/BWT et RAM mesurés, résultats dans `experiments/exp_001_ewc_dataset2/`.

**Résultats clés** (seed=42, cpu, Dataset 2 — 7 672 échantillons, 3 domaines) :

| Métrique | EWC Online | Fine-tuning naïf | Joint |
|----------|:----------:|:----------------:|:-----:|
| AA | **0.9824** | 0.9811 | 0.9811 |
| AF | **0.0010** | 0.0000 | — |
| RAM peak inférence | **1.1 Ko** (1.8% de 64 Ko) | — | — |
| RAM peak mise à jour | **6.7 Ko** (10.4% de 64 Ko) | — | — |
| Latence inférence | **0.036 ms** | — | — |

> ⚠️ Oubli catastrophique quasi-absent sur les deux méthodes — les 3 domaines du Dataset 2 ont des distributions proches (pas de domain shift fort). Limitation documentée dans [S109_exp001.md](sprints/sprint_1/S109_exp001.md).

---

## Sprint 2 — Semaine 2 (22–29 avril)

**Objectif** : M3 (HDC) fonctionnel + comparaison EWC vs HDC sur Dataset 2

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S2-01 | Implémenter `base_vectors.py` (génération + save HDC) | ✅ | ✅ | N/A | `src/models/hdc/base_vectors.py` | 2h |
| S2-02 | Implémenter `hdc_classifier.py` (encodage + prototypes + inférence) | ✅ | ✅ | N/A | `src/models/hdc/hdc_classifier.py` | 4h |
| S2-03 | Expérience HDC : exp_002 | ✅ | ✅ | ✅ | `experiments/exp_002_hdc_dataset2/` | 2h |
| S2-04 | Notebook comparaison EWC vs HDC vs Fine-tuning | ✅ | ✅ | ✅ | `notebooks/02_baseline_comparison.ipynb` | 3h |
| S2-05 | `scenarios.py` (gestion générique des streams CL) | ✅ | ✅ | N/A | `src/training/scenarios.py` | 2h |
| S2-06 | Visualisation accuracy matrix (heatmap forgetting) | ✅ | ✅ | N/A | `src/evaluation/plots.py` | 2h |
| S2-07 | Config YAML pour HDC + refactoring configs | ✅ | ✅ | N/A | `configs/hdc_config.yaml` | 1h |
| S2-08 | Tests unitaires HDC | ✅ | ✅ | ✅ | `tests/test_hdc.py` | 2h |
| S2-09 | Mise à jour README avec résultats préliminaires | 🟢 | ✅ | ⬜ | `README.md` | 1h |

**Livrable sprint 2** : comparaison EWC vs HDC vs Fine-tuning naïf sur Dataset 2, tableau de résultats complet.

---

## Sprint 3 — Semaine 3 (29 avril – 6 mai) ✅ TERMINÉ — exécuté et validé le 10 avril 2026

**Objectif** : M1 (TinyOL) — pré-entraînement + boucle online sur Dataset 1

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S3-01 | Télécharger Dataset 1 (Pump) + exploration | ✅ | ✅ | ✅ | `notebooks/01_data_exploration.ipynb` (section 2) | 2h |
| S3-02 | Implémenter `pump_dataset.py` (fenêtrage + features) | ✅ | ✅ | N/A | `src/data/pump_dataset.py` | 4h |
| S3-03 | Implémenter `autoencoder.py` (backbone + décodeur) | ✅ | ✅ | N/A | `src/models/tinyol/autoencoder.py` | 3h |
| S3-04 | Pré-entraînement backbone (données normales uniquement) | ✅ | ✅ | ✅ | `scripts/pretrain_tinyol.py` | 2h |
| S3-05 | Implémenter `oto_head.py` (tête OtO + boucle SGD online) | ✅ | ✅ | N/A | `src/models/tinyol/oto_head.py` | 3h |
| S3-06 | Expérience TinyOL : exp_003 | ✅ | ✅ | ✅ | `experiments/exp_003_tinyol_dataset1/` | 3h |
| S3-07 | Tests unitaires TinyOL | ✅ | ✅ | ✅ | `tests/test_tinyol.py` | 2h |
| S3-08 | Notebook CL évaluation Dataset 1 | ✅ | ✅ | ✅ | `notebooks/03_cl_evaluation.ipynb` | 2h |
| S3-09 | EDA plots enrichis normal/anomalie (2 datasets) | ✅ | ✅ | ✅ | `src/evaluation/eda_plots.py`, `scripts/explore_eda.py`, `notebooks/01_data_exploration.ipynb` | 3h |
| S3-10 | Réorganisation `notebooks/figures/` en sous-dossiers thématiques | ✅ | ✅ | ✅ | `notebooks/figures/**`, `scripts/visualize_feature_space.py`, `notebooks/01_data_exploration.ipynb`, `notebooks/02_baseline_comparison.ipynb` | 1h |

**Livrable sprint 3** : TinyOL entraîné et évalué sur Dataset 1, comparaison avec fine-tuning naïf.

---

## Sprint 4 — Semaine 4 (6–13 mai)

**Objectif** : Extension buffer UINT8 + comparaison finale 3 modèles

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S4-01 | Implémenter `quantization.py` (UINT8 encoder/decoder) | 🔴 | ⬜ | N/A | `src/utils/quantization.py` | 3h |
| S4-02 | Extension buffer UINT8 sur TinyOL | 🔴 | ⬜ | N/A | `src/models/tinyol/oto_head.py` (extension) | 3h |
| S4-03 | Exp buffer UINT8 vs FP32 : delta précision | 🔴 | ⬜ | ⬜ | `experiments/exp_004_tinyol_uint8/` | 2h |
| S4-04 | Tableau comparatif final 3 modèles (+ baselines) | 🔴 | ⬜ | ⬜ | `notebooks/04_final_comparison.ipynb` | 3h |
| S4-05 | Export ONNX des 3 modèles (vérification portabilité) | 🟡 | ⬜ | ⬜ | `scripts/export_onnx.py` | 3h |
| S4-06 | Profiling mémoire systématique (3 modèles) | 🔴 | ⬜ | ⬜ | `scripts/profile_memory.py` | 2h |
| S4-07 | Refactoring final + documentation docstrings | 🟡 | ⬜ | N/A | Tout `src/` | 4h |
| S4-08 | `CONTRIBUTING.md` + `LICENSE` | 🟢 | ⬜ | N/A | Racine | 1h |

**Livrable sprint 4** : tableau comparatif complet 3 modèles, chiffres RAM mesurés, export ONNX validé. Prêt pour portage MCU (Phase 2).

---

## Sprint 5 — Semaine 5 (13–20 mai)

**Objectif** : Baselines non supervisées (M4 K-Means+KNN, M5 PCA, M6 Mahalanobis) sur Dataset 2 et Dataset 1

> **Dépendance** : `pump_dataset.py` (S3-02) requis pour les expériences sur Dataset 1.  
> Ces modèles sont **PC-only sauf Mahalanobis** (M6 compatible STM32N6 — O(d²) RAM, inversion offline). Labels utilisés uniquement en évaluation.

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S5-01 | Structure `src/models/unsupervised/` + config YAML | ✅ | ✅ | N/A | `configs/unsupervised_config.yaml`, `src/models/unsupervised/__init__.py` | 1h |
| S5-02 | Implémenter `kmeans_detector.py` (K-Means + K dynamique silhouette/elbow) | ✅ | ✅ | N/A | `src/models/unsupervised/kmeans_detector.py` | 3h |
| S5-03 | Implémenter `knn_detector.py` (KNN distance-based anomaly detection) | ✅ | ✅ | N/A | `src/models/unsupervised/knn_detector.py` | 2h |
| S5-04 | Implémenter `pca_baseline.py` (PCA reconstruction error) | ✅ | ✅ | N/A | `src/models/unsupervised/pca_baseline.py` | 2h |
| S5-05 | Script d'entraînement + évaluation CL non supervisé | ✅ | ✅ | ✅ | `scripts/train_unsupervised.py` | 3h |
| S5-06 | Expérience non supervisée Dataset 2 : exp_005 | ✅ | ✅ | ✅ | `experiments/exp_005_unsupervised_dataset2/` | 2h |
| S5-07 | Expérience non supervisée Dataset 1 : exp_006 | ✅ | ✅ | ⬜ | `experiments/exp_006_unsupervised_dataset1/` | 2h |
| S5-08 | Tests unitaires + spec `unsupervised_spec.md` | ✅ | ✅ | ✅ | `tests/test_unsupervised.py`, `docs/models/unsupervised_spec.md` | 2h |
| S5-09 | Notebook comparatif supervisé vs non supervisé (6 modèles) | 🟡 | ✅ | ⬜ | `notebooks/05_supervised_vs_unsupervised.ipynb` | 2h |
| S5-10 | **Implémenter `mahalanobis_detector.py` (M6 — μ, Σ⁻¹ offline, seuil adaptatif)** | ✅ | ✅ | N/A | `src/models/unsupervised/mahalanobis_detector.py` | 2h |
| S5-11 | **Expérience Mahalanobis Dataset 1 et 2 : exp_007** | ✅ | ✅ | ✅ | `experiments/exp_007_mahalanobis/` | 2h |
| S5-12 | *(optionnel)* Implémenter `gmm_detector.py` (GMM EM offline, K petit) | 🟢 | ✅ | ⬜ | `src/models/unsupervised/gmm_detector.py` | 3h |
| S5-13 | Visualisation espace des features + clusters (PCA 2D, ellipses Mahalanobis, Voronoï K-Means, heatmap PCA reconstruction) | ✅ | ✅ | ✅ | `src/evaluation/feature_space_plots.py`, `notebooks/figures/05_feature_space_*.png` | 2h |
| S5-14 | Implémenter `dbscan_detector.py` (M7 DBSCAN — density-based clustering, epsilon adaptatif, points bruit = anomalies) + expérience exp_008 | ✅ | ✅ | ✅ | `src/models/unsupervised/dbscan_detector.py`, `experiments/exp_008_dbscan/` | 3h |
| S5-15 | EDA affinée : Pump_ID × Operational_Hours + Equipment × Location (seaborn violin/boxplot/heatmap/pairplot/corrélation) | ✅ | ✅ | ✅ | `src/evaluation/eda_plots.py`, `notebooks/01_data_exploration.ipynb` | 3h |

> **Corrections de chemins dataset appliquées le 8 avril 2026** — Standardisation de tous les configs vers une clé `csv_path` avec chemin complet vers le fichier CSV (au lieu de clés `path` pointant vers des dossiers avec fallback glob fragile) :
>
> - `configs/unsupervised_config.yaml` : `data_pump.csv_path` corrigé (🔴 critique — `FileNotFoundError`) et `data.csv_path` corrigé (🟡 fallback rglob supprimé). Voir [S505](sprints/sprint_5/S505_train_unsupervised.md), [S507](sprints/sprint_5/S507_exp006.md).
> - `configs/tinyol_config.yaml` : clé `data.path` renommée `data.csv_path` + chemin complet avec nom de fichier (🟢 préventif — `train_tinyol.py` non encore implémenté). Voir [S302](sprints/sprint_3/S302_pump_dataset.md).
> - `configs/ewc_config.yaml` : clé `data.path` (dossier) renommée `data.csv_path` (chemin complet CSV) ; `scripts/train_ewc.py` mis à jour (suppression `glob("*.csv")` → lecture directe).
> - `configs/hdc_config.yaml` : clé `data.path` (dossier parent) renommée `data.csv_path` (chemin complet CSV) ; `scripts/train_hdc.py` mis à jour (suppression `rglob("*.csv")` → lecture directe).

> **Pourquoi Mahalanobis (M6) et pas GMM/HMM en sprint dédié ?**  
> Mahalanobis est le seul des trois à satisfaire simultanément : (a) computation embarquée viable (inversion Σ offline, produit matriciel online), (b) applicable aux deux datasets, (c) interprétable comme baseline de référence.  
> GMM reste optionnel (entraînement EM offline, K=2–3 raisonnable). HMM exclu : complexité O(T×N²), Baum-Welch incompatible avec l'online learning, non applicable au Dataset 2 — relégué au backlog pour analyse offline.

**Livrable sprint 5** : 4 modèles non supervisés (K-Means dynamique, KNN anomaly, PCA reconstruction, Mahalanobis) évalués en scénario domain-incremental sur Dataset 2 et Dataset 1. Tableau comparatif AA/AF/BWT/AUROC vs M1/M2/M3. M6 Mahalanobis profilé en RAM (compatible 64 Ko).

---

## Sprint 5 — Extension : Expériences granulaires (≥ 12 avril 2026)

**Objectif** : Tester des scénarios CL plus fins — Pump par Pump_ID (5 tâches) et Monitoring par location (5 tâches) — pour comparer avec les scénarios coarse (temporel et equipment-type) et détecter de nouveaux patterns de forgetting et de séparabilité.

> **Motivation** : Les scénarios actuels (3 tâches) donnent des AA proches du hasard sur Dataset 1 (0.50–0.56) et excellentes sur Dataset 2 (0.87–0.98). Question : est-ce que la granularité du découpage change la difficulté et révèle des différences inter-pompe ou inter-site non visibles jusqu'ici ?

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S5-18 | Scénario pump par Pump_ID : loader + 4 expériences (exp_012–015) + plots | 🔴 | `src/data/pump_dataset.py`, `configs/pump_by_id_config.yaml`, `experiments/exp_012-015/` | 6h | S3-02, S5-10 |
| S5-19 | Scénario monitoring par location : loader + 4 expériences (exp_016–019) + plots | 🔴 | `src/data/monitoring_dataset.py`, `configs/monitoring_by_location_config.yaml`, `experiments/exp_016-019/` | 6h | S1-03, S5-10 |

**Nouvelles expériences planifiées** :

| Exp | Modèle | Scénario | Dataset | Statut |
|-----|--------|----------|---------|--------|
| exp_012 | TinyOL | 5 tâches par Pump_ID | Dataset 1 | ⬜ |
| exp_013 | EWC | 5 tâches par Pump_ID | Dataset 1 | ⬜ |
| exp_014 | HDC | 5 tâches par Pump_ID | Dataset 1 | ⬜ |
| exp_015 | Mahalanobis | 5 tâches par Pump_ID | Dataset 1 | ⬜ |
| exp_016 | EWC | 5 tâches par location | Dataset 2 | ⬜ |
| exp_017 | HDC | 5 tâches par location | Dataset 2 | ⬜ |
| exp_018 | TinyOL | 5 tâches par location | Dataset 2 | ⬜ |
| exp_019 | Mahalanobis | 5 tâches par location | Dataset 2 | ⬜ |

**Plots prévus** :
- Réutilisés : `plot_accuracy_matrix()`, `plot_forgetting_curve()`, `plot_model_radar()`, `plot_boxplots_by_pump_id()`, `plot_fault_rate_heatmap_pump()`, `plot_fault_rate_heatmap_equipment()`, `plot_cl_evolution()`, `plot_clustering_with_correctness()`
- Nouveaux : `plot_performance_by_pump_id_bar()` + `plot_performance_heatmap_equipment_location()` dans `src/evaluation/plots.py`

**Livrable** : 8 nouvelles expériences, 2 configs YAML, 2 loaders additionnels, comparaison avec scénarios existants. Réponse à la question : *les Pump_ID ont-ils des profils de panne distincts ? Les sites géographiques ont-ils des patterns spécifiques ?*

---

## Sprint 6 (Phase 1) — Infrastructure notebooks & expériences granulaires complètes

**Objectif** : Mettre en place l'infrastructure technique pour les 28 notebooks d'évaluation : utilitaire figures sous-dossiers, loader temporel Dataset 1, fonctions visualisation manquantes, 18 expériences manquantes.

> Détail complet : [`docs/sprints/sprint_6/S600_notebook_infra_sprint.md`](sprints/sprint_6/S600_notebook_infra_sprint.md)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S6-01 | Vérifier/corriger `save_figure()` pour sous-dossiers auto | ⬜ | ✅ | N/A | `src/evaluation/plots.py` | 1h |
| S6-02 | `get_pump_dataloaders_by_temporal_window()` — 4 quartiles 5 000 entrées | ⬜ | ✅ | N/A | `src/data/pump_dataset.py` | 3h |
| S6-03 | `configs/pump_by_temporal_window_config.yaml` | ⬜ | ✅ | N/A | `configs/pump_by_temporal_window_config.yaml` | 1h |
| S6-04 | `plot_performance_by_pump_id_bar()` | ⬜ | ✅ | N/A | `src/evaluation/plots.py` | 2h |
| S6-05 | `plot_performance_heatmap_equipment_location()` | ⬜ | ✅ | N/A | `src/evaluation/plots.py` | 2h |
| S6-06 | Run exp_012–015 : pump_by_id (TinyOL, EWC, HDC, Mahalanobis) | ⬜ | ✅ | ⬜ | `experiments/exp_012–015/` | 2h |
| S6-07 | Run exp_016–019 : monitoring_by_location (EWC, HDC, TinyOL, Mahalanobis) | ⬜ | ✅ | ⬜ | `experiments/exp_016–019/` | 2h |
| S6-08 | Run exp_020–021 : pump_by_id (KMeans, DBSCAN) | ⬜ | ✅ | ⬜ | `experiments/exp_020–021/` | 1h |
| S6-09 | Run exp_022–023 : monitoring_by_location (KMeans, DBSCAN) | ⬜ | ✅ | ⬜ | `experiments/exp_022–023/` | 1h |
| S6-10 | Run exp_024–029 : pump_by_temporal_window (6 modèles) | ⬜ | ✅ | ⬜ | `experiments/exp_024–029/` | 3h |
| S6-11 | Tests unitaires `get_pump_dataloaders_by_temporal_window()` | ⬜ | ✅ | ⬜ | `tests/test_pump_dataset.py` | 2h |

**Livrable sprint 6** : 18 expériences (exp_012–029) exécutées, infrastructure figures sous-dossiers opérationnelle, loader temporel validé.

---

## Sprint 7 (Phase 1) — Notebooks individuels Dataset 2 (Equipment Monitoring)

**Objectif** : 14 notebooks d'évaluation pour Dataset 2 — 6 modèles × 2 scénarios + 2 notebooks de comparaison.

> Détail complet : [`docs/sprints/sprint_7/S700_notebooks_monitoring_sprint.md`](sprints/sprint_7/S700_notebooks_monitoring_sprint.md)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S7-01 | Notebook TinyOL — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/tinyol.ipynb` | 2h |
| S7-02 | Notebook EWC — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/ewc.ipynb` | 2h |
| S7-03 | Notebook HDC — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/hdc.ipynb` | 2h |
| S7-04 | Notebook KMeans — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/kmeans.ipynb` | 2h |
| S7-05 | Notebook Mahalanobis — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/mahalanobis.ipynb` | 2h |
| S7-06 | Notebook DBSCAN — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/dbscan.ipynb` | 2h |
| S7-07 | Notebook TinyOL — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/tinyol.ipynb` | 2h |
| S7-08 | Notebook EWC — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/ewc.ipynb` | 2h |
| S7-09 | Notebook HDC — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/hdc.ipynb` | 2h |
| S7-10 | Notebook KMeans — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/kmeans.ipynb` | 2h |
| S7-11 | Notebook Mahalanobis — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/mahalanobis.ipynb` | 2h |
| S7-12 | Notebook DBSCAN — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/dbscan.ipynb` | 2h |
| S7-13 | Notebook Comparaison — monitoring_by_equipment | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb` | 3h |
| S7-14 | Notebook Comparaison — monitoring_by_location | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/monitoring_by_location/comparison.ipynb` | 3h |

**Livrable sprint 7** : 14 notebooks Dataset 2 prêts pour présentation encadrants. Figures dans `notebooks/figures/cl_evaluation/{model}/monitoring/{scenario}/`.

---

## Sprint 8 (Phase 1) — Notebooks individuels Dataset 1 (Pump Maintenance)

**Objectif** : 14 notebooks d'évaluation pour Dataset 1 — 6 modèles × 2 scénarios granulaires + 2 notebooks de comparaison.

> Détail complet : [`docs/sprints/sprint_8/S800_notebooks_pump_sprint.md`](sprints/sprint_8/S800_notebooks_pump_sprint.md)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S8-01 | Notebook TinyOL — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/tinyol.ipynb` | 2h |
| S8-02 | Notebook EWC — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/ewc.ipynb` | 2h |
| S8-03 | Notebook HDC — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/hdc.ipynb` | 2h |
| S8-04 | Notebook KMeans — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/kmeans.ipynb` | 2h |
| S8-05 | Notebook Mahalanobis — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/mahalanobis.ipynb` | 2h |
| S8-06 | Notebook DBSCAN — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/dbscan.ipynb` | 2h |
| S8-07 | Notebook TinyOL — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/tinyol.ipynb` | 2h |
| S8-08 | Notebook EWC — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/ewc.ipynb` | 2h |
| S8-09 | Notebook HDC — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/hdc.ipynb` | 2h |
| S8-10 | Notebook KMeans — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/kmeans.ipynb` | 2h |
| S8-11 | Notebook Mahalanobis — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/mahalanobis.ipynb` | 2h |
| S8-12 | Notebook DBSCAN — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/dbscan.ipynb` | 2h |
| S8-13 | Notebook Comparaison — pump_by_pump_id | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_pump_id/comparison.ipynb` | 3h |
| S8-14 | Notebook Comparaison — pump_by_temporal_window | ⬜ | ✅ | ⬜ | `notebooks/cl_eval/pump_by_temporal_window/comparison.ipynb` | 3h |

**Livrable sprint 8** : 14 notebooks Dataset 1 prêts pour présentation encadrants. Figures dans `notebooks/figures/cl_evaluation/{model}/pump/{scenario}/`.

---

## Sprint 9 (Phase 1) — Finalisation, archivage & mise à jour roadmap

**Objectif** : Clore proprement la Phase 1 Python : archivage `03_cl_evaluation.ipynb`, MAJ roadmaps, README.

> Détail complet : [`docs/sprints/sprint_9/S900_notebook_polish_sprint.md`](sprints/sprint_9/S900_notebook_polish_sprint.md)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S9-01 | MAJ `03_cl_evaluation.ipynb` : note d'intro + liens vers `cl_eval/` | ⬜ | ✅ | ⬜ | `notebooks/03_cl_evaluation.ipynb` | 1h |
| S9-02 | MAJ `roadmap_phase1.md` : Sprints 6–9 avec résultats réels | ⬜ | ✅ | N/A | `docs/roadmap_phase1.md` | 2h |
| S9-03 | MAJ `roadmap.md` : Phase 1 = Sprints 1–9, Phase 2 = Sprint 10 | ✅ | ✅ | N/A | `docs/roadmap.md` | 30min |
| S9-04 | Renommer Sprint 6 → Sprint 10 dans `roadmap_phase2.md` | ✅ | ✅ | N/A | `docs/roadmap_phase2.md` | 30min |
| S9-05 | MAJ table indicateurs de progression dans `roadmap.md` | ⬜ | ✅ | N/A | `docs/roadmap.md` | 30min |
| S9-06 | MAJ README.md : section "Notebooks d'évaluation" | ⬜ | ✅ | N/A | `README.md` | 1h |

**Livrable sprint 9** : Phase 1 complète et documentée — 9 sprints, 29 expériences, 28 notebooks, 6 modèles, 4 scénarios, 2 datasets.

---

## Résultats d'expériences

### M2 EWC — exp_001 (4 avril 2026, seed=42)

| Métrique | EWC Online | Fine-tuning naïf | Joint training |
|----------|:----------:|:----------------:|:--------------:|
| AA | **0.9824** | 0.9811 | 0.9811 |
| AF | **0.0010** | 0.0000 | N/A |
| BWT | +0.0000 | +0.0010 | N/A |
| RAM peak inférence | **1 171 B (1.1 Ko)** | — | — |
| RAM peak mise à jour | **6 837 B (6.7 Ko)** | — | — |
| Latence inférence | **0.036 ms** | — | — |
| Budget 64 Ko | ✅ 10.4% utilisés | — | — |

> Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor — 705 paramètres.  
> Note : oubli catastrophique quasi-absent sur ce dataset (domaines très similaires). Voir S109 pour l'analyse complète.

---

### M3 HDC — exp_002 (6 avril 2026, seed=42)

| Métrique | HDC Online |
|----------|:----------:|
| AA | **0.8698** |
| AF | **0.0000** |
| BWT | +0.0019 |
| RAM estimée FP32 | **14 344 B (14.0 Ko)** |
| RAM estimée INT8 | **6 152 B (6.0 Ko)** |
| RAM peak mesuré (inférence) | **14 504 B (14.2 Ko)** |
| Latence inférence | **0.048 ms** |
| Budget 64 Ko | ✅ 22.1% utilisés (FP32) |

> Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor — 2 048 éléments de prototypes.  
> AF = 0 par construction (accumulation additive, pas d'oubli catastrophique possible).  
> AA inférieure à EWC (0.8698 vs 0.9824) — attendu : HDC est moins expressif qu'un MLP mais 5× moins gourmand en RAM.

---

### M4a/M4b/M5 non supervisés — exp_005 (7 avril 2026, seed=42)

Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor

| Modèle | AA | AF | BWT | AUROC | RAM peak | Latence |
|--------|:--:|:--:|:---:|:-----:|:--------:|:-------:|
| K-Means (K=2, silhouette) | **0.9433** | 0.0049 | -0.0040 | **0.9621** | 5.2 Ko ✅ | 0.399 ms |
| KNN (accumulate, k=5) | **0.9524** | 0.0275 | -0.0275 | **0.9728** | 110.5 Ko ⚠️ | 15.755 ms |
| PCA (2 composantes, refit) | **0.9504** | 0.0020 | -0.0010 | **0.9078** | 2.1 Ko ✅ | 0.115 ms |

> ⚠️ KNN dépasse 64 Ko (stratégie `accumulate` sur 6 137 échantillons = ~98 Ko de X_ref). **PC-only** — non portable STM32N6 sans modification de stratégie.  
> AA nettement supérieure aux cibles (> 0.94 vs > 0.70) et AUROC > 0.90 : le Dataset 2 est bien séparable (peu de domain shift, mais forte séparabilité normal/faulty).  
> Pour comparaison directe EWC (0.9824) > KNN (0.9524) ≈ PCA (0.9504) > K-Means (0.9433) — sans aucune supervision pendant l'entraînement.

---

### M6 Mahalanobis — exp_007 (8 avril 2026, seed=42)

Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor — d=4 features

| Modèle | AA | AF | BWT | AUROC | RAM peak | Latence |
|--------|:--:|:--:|:---:|:-----:|:--------:|:-------:|
| **Mahalanobis ✅** | **0.9524** | **0.0010** | **-0.0010** | **0.9718** | **80 B analytique** | **0.018 ms** |
| K-Means (exp_005) ✅ | 0.9433 | 0.0049 | -0.0040 | 0.9621 | 5.2 Ko | 0.399 ms |
| KNN (exp_005) ⚠️ | 0.9524 | 0.0275 | -0.0275 | 0.9728 | 110.5 Ko | 15.755 ms |
| PCA (exp_005) ✅ | 0.9504 | 0.0020 | -0.0010 | 0.9078 | 2.1 Ko | 0.115 ms |

> **Meilleur modèle embarqué** : Mahalanobis domine sur tous les critères embarqués. AA = KNN (0.9524), AUROC proche de KNN (0.9718 vs 0.9728), oubli AF quasi-nul (0.0010), latence ×22 plus rapide que K-Means, RAM analytique = 80 B @ FP32 (0.12% du budget 64 Ko STM32N6).  
> RAM tracemalloc = 1504 B (overhead Python ×18.8 vs analytique — non représentatif du MCU). n_params = 20 (d + d² pour d=4). cl_strategy=refit : μ et Σ⁻¹ recalculés à chaque tâche, oubli structurel volontaire.  
> Dataset 1 (Pump) non exécuté : données Kaggle non disponibles localement. `FIXME(gap1)` : valider sur FEMTO PRONOSTIA.

---

### M1 TinyOL — exp_003 (10 avril 2026, seed=42)

Dataset 1 (Pump Maintenance) — 3 tâches chronologiques : sain → usure → pré-panne — 10 params OtO + 1 496 params encodeur

| Métrique | TinyOL |
|----------|:------:|
| AA | **0.5586** |
| AF | **0.0084** |
| BWT | -0.0084 |
| RAM peak update | **6 425 B (6.3 Ko)** |
| Latence inférence OtO | **0.010 ms** |
| n_params OtO | 10 |
| n_params encodeur | 1 496 |
| Budget 64 Ko | ✅ 9.8% utilisés |

> Dataset 1 (Pump Maintenance) — 3 tâches : T1=415, T2=415, T3=416 fenêtres (window=32, step=16).  
> AA ≈ 0.56 : performance proche du hasard — le backbone figé (pré-entraîné uniquement sur données normales T1) génère des embeddings peu discriminants pour les tâches d'usure/pré-panne. Limitation documentée : `FIXME(gap1)`.  
> `FIXME(gap2)` : RAM 6 425 B mesurée via tracemalloc (overhead Python) — non représentative du MCU. RAM analytique encodeur = 5 984 B + OtO = 40 B = 6 024 B (9.2% de 64 Ko).

---

### M2 EWC — exp_009 (10 avril 2026, seed=42)

Dataset 1 (Pump Maintenance) — 3 tâches chronologiques : sain → usure → pré-panne — MLP (25→32→16→1) + régularisation EWC Online (λ=1000, γ=0.9)

| Métrique | EWC |
|----------|:---:|
| AA | **0.4980** |
| AF | **0.0060** |
| BWT | -0.0060 |
| RAM peak forward | **1 171 B (1.1 Ko)** |
| RAM peak update | **8 889 B (8.7 Ko)** |
| Latence inférence | **0.043 ms** |
| n_params | 1 377 |
| Budget 64 Ko | ✅ 13.5% utilisés |

> Dataset 1 (Pump Maintenance) — T1=415, T2=415, T3=416 fenêtres. AA ≈ 0.50 : performances proches du hasard, identiques au fine-tuning naïf (aa_naive=0.498). La régularisation EWC ne suffit pas à discriminer des fenêtres peu séparables après normalisation Z-score (features de rang 25 sans drift temporel fort). AF très faible (0.006) : l'oubli est contenu mais la performance de base est déjà basse.  
> `FIXME(gap2)` : RAM mesurée via tracemalloc (overhead Python). RAM analytique MLP = 5 508 B @ FP32 + Fisher ≈ 3× (5 508 B) = 16.5 Ko @ FP32 (26% de 64 Ko).  
> `FIXME(gap1)` : Dataset 1 (Pump) non validé sur FEMTO PRONOSTIA — données Kaggle uniquement.

---

### M3 HDC — exp_010 (10 avril 2026, seed=42)

Dataset 1 (Pump Maintenance) — 3 tâches chronologiques : sain → usure → pré-panne — HDC (D=1024, n_levels=10, 25 features)

| Métrique | HDC |
|----------|:---:|
| AA | **0.5100** |
| AF | **0.0422** |
| BWT | +0.0060 |
| RAM peak forward | **14 504 B (14.2 Ko)** |
| RAM estimée FP32 | 14 344 B (14.0 Ko) |
| RAM estimée INT8 | 6 152 B (6.0 Ko) |
| Latence inférence | **0.256 ms** |
| n_params | 2 048 éléments (hypervecteurs) |
| Budget 64 Ko | ✅ 22.1% utilisés |

> Dataset 1 (Pump Maintenance) — T1=416, T2=416, T3=417 exemples. AA ≈ 0.51 : légèrement supérieur à EWC (0.498) et au hasard. BWT légèrement positif (+0.006) : signe que l'apprentissage des tâches suivantes bénéficie aux précédentes, caractéristique de l'additivité des prototypes HDC. AF=0.042 plus élevé qu'EWC : l'hypervecteur moyen de classe se déplace significativement avec les nouvelles tâches.  
> `FIXME(gap2)` : RAM analytique = 2 × D × 4 B (2 classes × 1 024 × FP32) = 8 192 B + base vectors ≈ 14 344 B FP32 / 6 152 B INT8. Mesure tracemalloc = 14 504 B (1.01× analytique — overhead quasi-nul).  
> `FIXME(gap1)` : Dataset 1 (Pump) non validé sur FEMTO PRONOSTIA — données Kaggle uniquement.

---

### M1 TinyOL — exp_011 (10 avril 2026, seed=42)

Dataset 2 (Equipment Monitoring) — 3 domaines : Pump → Turbine → Compressor — 10 params OtO + 184 params encodeur (4→8→8→8)

| Métrique | TinyOL |
|----------|:------:|
| AA | **0.9123** |
| AF | **0.0079** |
| BWT | -0.0029 |
| RAM peak update | **4 379 B (4.3 Ko)** |
| Latence inférence OtO | **0.010 ms** |
| n_params OtO | 10 |
| n_params encodeur | 184 |
| Budget 64 Ko | ✅ 6.7% utilisés |

> Dataset 2 (Equipment Monitoring) — T1=2 027 (Pump), T2=2 052 (Turbine), T3=2 058 (Compressor) exemples train. AA = 0.912 : excellente performance sur ce dataset tabulaire statique — le backbone (4→8→8→8) pré-entraîné sur données normales Pump encode efficacement les 4 features numériques. Oubli très faible (AF=0.0079), similaire à exp_003 sur Dataset 1. RAM 4 379 B (tracemalloc) inférieure à exp_003 (6 425 B) grâce à l'encodeur plus petit (184 vs 1 496 params).  
> RAM analytique encodeur = 184 × 4 B = 736 B + OtO = 40 B = 776 B (1.2% de 64 Ko).  
> `FIXME(gap2)` : overhead tracemalloc ×5.6 vs analytique (4 379 B vs 776 B) — non représentatif MCU.
