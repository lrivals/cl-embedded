# Sprint 12 (Phase 1) — Dataset 5 : CWRU Bearing Dataset (Fault Classification)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 12 — Phase 1 Extension |
| **Priorité globale** | 🟡 Important — deuxième dataset réel de roulements, croisement Gap 1 |
| **Durée estimée totale** | ~26h |
| **Dépendances** | Sprint 11 terminé (Battery RUL intégré, exp_056–067 exécutées) |

---

## Objectif

Intégrer le **CWRU Bearing Dataset** (Case Western Reserve University) comme cinquième source de validation du projet CL-Embedded. Ce sprint complète la couverture des données réelles de roulements en ajoutant un benchmark industriel de référence avec deux scénarios domain-incremental distincts : progression par **type de défaut** et progression par **sévérité de défaut**.

**Complémentarité avec PRONOSTIA** : PRONOSTIA modélise la dégradation progressive jusqu'à la panne (RUL / TTF binaire) ; CWRU modélise la classification de l'état courant du roulement (normal vs défaillant). Les deux ensemble couvrent les deux usages principaux de la maintenance prédictive embarquée.

**Critère de succès** : 18 expériences CWRU exécutées (exp_068–085), `cwru_dataset.py` validé par tests unitaires, EDA section 5 complète dans un notebook dédié, résultats cross-dataset comparables avec PRONOSTIA (exp_044–055) dans `roadmap_phase1.md`.

---

## Contexte dataset

**Dataset** : Case Western Reserve University (CWRU) Bearing Data Center
**Source** : CWRU Motor Performance Laboratory (données réelles d'accélérométrie de roulements)
**Chemin** : `data/raw/CWRU Bearing Dataset/`
**Format raw** : fichiers `.mat` (MATLAB binary), CSV pré-traité disponible

| Paramètre | Valeur |
|-----------|--------|
| Fichiers MAT | 10 (1 Normal + 3 Ball + 3 Inner Race + 3 Outer Race) |
| Fréquence échantillonnage | 48 kHz |
| Taille fenêtre | 2 048 points (0,04 s) |
| Charge opératoire | 1 HP (1 772 rpm) |
| Positions accéléromètre | Drive End (DE), Fan End (FE), Base (BA) |
| Features | 9 (max, min, mean, sd, rms, skewness, kurtosis, crest, form) |
| Label | Binaire — 0 = Normal, 1 = Défaillant |
| Échantillons CSV | 2 299 fenêtres pré-extraites |

### Fichiers MAT disponibles

| Fichier | Type défaut | Diamètre défaut | Accéléromètre |
|---------|-------------|-----------------|---------------|
| `Time_Normal_1_098.mat` | Normal | — | Drive End |
| `B007_1_123.mat` | Ball | 0.007" (0.178 mm) | Drive End |
| `B014_1_190.mat` | Ball | 0.014" (0.356 mm) | Drive End |
| `B021_1_227.mat` | Ball | 0.021" (0.533 mm) | Drive End |
| `IR007_1_110.mat` | Inner Race | 0.007" | Drive End |
| `IR014_1_175.mat` | Inner Race | 0.014" | Drive End |
| `IR021_1_214.mat` | Inner Race | 0.021" | Drive End |
| `OR007_6_1_136.mat` | Outer Race | 0.007" | Base |
| `OR014_6_1_202.mat` | Outer Race | 0.014" | Base |
| `OR021_6_1_239.mat` | Outer Race | 0.021" | Base |

### Scénario CL 1 — by_fault_type (domain-incremental, 3 tâches)

| Tâche | Domaine | Fichiers inclus | Fenêtres approx. |
|-------|---------|-----------------|-----------------|
| Task 1 | Ball Fault | B007 + B014 + B021 + Normal (subset) | ~920 |
| Task 2 | Inner Race Fault | IR007 + IR014 + IR021 + Normal (subset) | ~920 |
| Task 3 | Outer Race Fault | OR007 + OR014 + OR021 + Normal (subset) | ~460 |

> Le sous-ensemble Normal est réparti équitablement entre les 3 tâches pour maintenir l'équilibre de classe.

### Scénario CL 2 — by_severity (domain-incremental, 3 tâches)

| Tâche | Domaine | Fichiers inclus | Fenêtres approx. |
|-------|---------|-----------------|-----------------|
| Task 1 | Sévérité 0.007" | B007 + IR007 + OR007 + Normal (subset) | ~920 |
| Task 2 | Sévérité 0.014" | B014 + IR014 + OR014 + Normal (subset) | ~920 |
| Task 3 | Sévérité 0.021" | B021 + IR021 + OR021 + Normal (subset) | ~460 |

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S12-01 ✅ | `cwru_dataset.py` — loader CSV/MAT, 9 features, label binaire, streams by_fault_type + by_severity | 🔴 | `src/data/cwru_dataset.py` | 5h | — |
| S12-02 ✅ | Créer et valider 3 configs YAML CWRU | 🔴 | `configs/cwru_single_task_config.yaml`, `configs/cwru_by_fault_config.yaml`, `configs/cwru_by_severity_config.yaml` | 1h | S12-01 |
| S12-03 ✅ | EDA CWRU — notebook dédié `01D_data_exploration_cwru.ipynb` | 🟡 | `notebooks/01D_data_exploration_cwru.ipynb`, `notebooks/figures/eda/cwru/` | 2h | S12-01 |
| S12-04 ✅ | Run exp_068–073 — 6 modèles × scénario `no_split` (single-task baseline) | 🔴 | `experiments/exp_068–073/` | 2h | S12-01 + S12-02 |
| S12-05 ✅ | Run exp_074–079 — 6 modèles × by_fault_type (3 tâches) | 🔴 | `experiments/exp_074–079/` | 2h | S12-04 |
| S12-06 ✅ | Run exp_080–085 — 6 modèles × by_severity (3 tâches) | 🔴 | `experiments/exp_080–085/` | 2h | S12-04 |
| S12-07 | Notebooks individuels `cwru_by_fault_type/` — 6 modèles + comparison | 🔴 | `notebooks/cl_eval/cwru_by_fault_type/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan,comparison}.ipynb` | 4h | S12-05 |
| S12-08 | Notebooks individuels `cwru_by_severity/` — 6 modèles + comparison | 🟡 | `notebooks/cl_eval/cwru_by_severity/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan,comparison}.ipynb` | 4h | S12-06 |
| S12-09 | Notebook baseline single-task CWRU | 🟡 | `notebooks/cl_eval/baselines/cwru_single_task.ipynb` | 1h | S12-04 |
| S12-10 | Tests unitaires `cwru_dataset.py` | 🟡 | `tests/test_cwru_dataset.py` | 2h | S12-01 |
| S12-11 | MAJ roadmap : S12-01 à S12-11 ✅, table cross-dataset CWRU vs PRONOSTIA | 🟡 | `docs/roadmap_phase1.md` | 1h | S12-05 à S12-09 |

> Détail : S1201_cwru_dataset.md · S1202_configs_cwru.md · S1203_eda_cwru.md · S1204_exp_single_task_cwru.md · S1205_exp_cl_by_fault_type.md · S1206_exp_cl_by_severity.md · S1207_notebooks_cwru_by_fault_type.md · S1208_notebooks_cwru_by_severity.md · S1209_notebook_baseline_cwru.md · S1210_tests_cwru_dataset.md · S1211_maj_roadmap.md

---

## Numérotation expériences

### Scénario no_split (single-task baseline)

| Exp | Modèle | Scénario | Config | Statut |
|-----|--------|----------|--------|--------|
| exp_068 | EWC | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.978 AUC=0.996 RAM=1.1 Ko |
| exp_069 | HDC | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.887 AUC=0.937 RAM=7.7 Ko |
| exp_070 | TinyOL | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.900 AUC=0.877 RAM=0.9 Ko |
| exp_071 | KMeans | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.159 AUC=0.601 RAM=5.3 Ko |
| exp_072 | Mahalanobis | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.139 AUC=0.548 RAM=1.6 Ko |
| exp_073 | DBSCAN | CWRU no_split | `cwru_single_task_config.yaml` | ✅ acc=0.146 AUC=0.842 RAM=115 Ko ⚠️ |

### Scénario by_fault_type (3 tâches : Ball → IR → OR)

| Exp | Modèle | Scénario | Config | Statut |
|-----|--------|----------|--------|--------|
| exp_074 | EWC | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=1.000 AF=0.000 BWT=0.000 RAM=1.1 Ko |
| exp_075 | HDC | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=0.935 AF=0.045 BWT=-0.039 RAM=7.7 Ko |
| exp_076 | TinyOL | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=0.966 AF=0.002 BWT=+0.007 RAM=4.0 Ko |
| exp_077 | KMeans | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=0.152 AF=0.019 BWT=+0.039 RAM=5.3 Ko |
| exp_078 | Mahalanobis | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=0.316 AF=0.013 BWT=+0.286 RAM=1.6 Ko |
| exp_079 | DBSCAN | CWRU by_fault_type | `cwru_by_fault_config.yaml` | ✅ AA=0.126 AF=0.045 BWT=-0.045 RAM=16.4 Ko |

### Scénario by_severity (3 tâches : 0.007" → 0.014" → 0.021")

| Exp | Modèle | Scénario | Config | Statut |
|-----|--------|----------|--------|--------|
| exp_080 | EWC | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.952 AF=0.000 BWT=+0.007 RAM=1.1 Ko |
| exp_081 | HDC | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.892 AF=0.020 BWT=-0.007 RAM=7.7 Ko |
| exp_082 | TinyOL | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.900 AF=0.000 BWT=+0.013 RAM=4.0 Ko |
| exp_083 | KMeans | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.303 AF=0.065 BWT=+0.286 RAM=5.3 Ko |
| exp_084 | Mahalanobis | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.394 AF=0.091 BWT=+0.396 RAM=1.6 Ko |
| exp_085 | DBSCAN | CWRU by_severity | `cwru_by_severity_config.yaml` | ✅ AA=0.121 AF=0.292 BWT=-0.013 RAM=30.7 Ko |

---

## Critères d'acceptation

- [x] S12-01 : `CWRUDataset`, `CWRUFaultTypeStream` et `CWRUSeverityStream` importables depuis `src.data.cwru_dataset` ✅ 24 avril 2026
- [x] S12-02 : Les 3 configs YAML chargent sans erreur ; `WINDOW_SIZE=2048`, `N_FEATURES=9`, `N_TASKS_FAULT=3`, `N_TASKS_SEVERITY=3` présents ✅ 24 avril 2026
- [x] S12-03 : `01D_data_exploration_cwru.ipynb` exécuté sans erreur, 10 figures sauvegardées dans `notebooks/figures/eda/cwru/`, shape (2300, 9) confirmé ✅ 24 avril 2026
- [x] S12-04 : exp_068–073 exécutées, `metrics_single_task.json` présent dans chaque dossier `experiments/exp_06X/results/` ✅ 24 avril 2026
- [x] S12-05 : exp_074–079 exécutées, `metrics_cl.json` présent dans chaque dossier `experiments/exp_07X/results/` ✅ 26 avril 2026
- [x] S12-06 : exp_080–085 exécutées, `metrics_cl.json` présent dans chaque dossier `experiments/exp_08X/results/` ✅ 26 avril 2026
- [ ] S12-07 : 7 notebooks dans `notebooks/cl_eval/cwru_by_fault_type/`, 5 figures par modèle sauvegardées
- [ ] S12-08 : 7 notebooks dans `notebooks/cl_eval/cwru_by_severity/`, 5 figures par modèle sauvegardées
- [ ] S12-09 : `baselines/cwru_single_task.ipynb` exécuté sans erreur
- [ ] S12-10 : `pytest tests/test_cwru_dataset.py -v` → 100% pass (9 tests minimum)
- [ ] S12-11 : `roadmap_phase1.md` : S12-01 à S12-11 marqués ✅, table comparative CWRU / PRONOSTIA ajoutée

---

## Livrable sprint 12

- **`src/data/cwru_dataset.py`** — loader validé, 9 features, label binaire, 2 streams CL distincts
- **3 configs YAML** (`cwru_single_task_config.yaml`, `cwru_by_fault_config.yaml`, `cwru_by_severity_config.yaml`)
- **`tests/test_cwru_dataset.py`** — ≥ 9 tests unitaires sur données synthétiques (fixtures `tmp_path`)
- **18 expériences** (exp_068–085) — 6 single-task + 6 by_fault_type + 6 by_severity sur données réelles
- **EDA** dans `01D_data_exploration_cwru.ipynb` — spectre vibratoire, distribution défauts, comparaison types/sévérités
- **15 notebooks** `cl_eval/` — 7 by_fault_type + 7 by_severity + 1 baseline single-task

---

## Questions ouvertes

- `TODO(arnaud)` : Le split par type de défaut (Ball → IR → OR) est-il le scénario CL le plus pertinent scientifiquement, ou préférer un split par position temporelle (drift progressif) pour mieux simuler un déploiement embarqué ?
- `TODO(arnaud)` : Le CWRU a une seule condition de charge (1 HP). Comparer les métriques AA/AF/BWT CWRU vs PRONOSTIA (3 conditions) — l'absence de variabilité opératoire favorise-t-elle l'oubli catastrophique ou le rend-il négligeable ?
- `TODO(dorra)` : Sur STM32N6, le loader CSV (`feature_time_48k_2048_load_1.csv`) est préférable au loader MAT (scipy.io) pour l'embarqué — valider que les 9 features sont calculables in-situ depuis le signal brut 48 kHz avec la contrainte RAM 64 Ko.
- `TODO(fred)` : Dans un contexte industriel réel (Edge Spectrum), le type de défaut est-il connu a priori (supervision possible) ou doit-il être inféré en ligne ? Impact sur le scénario by_fault_type (supervision des tâches).
- `FIXME(gap1)` : Croiser les métriques AA/AF/BWT CWRU (exp_074–085) avec PRONOSTIA (exp_050–055) pour la table de synthèse Triple Gap du manuscrit — les deux datasets réels de roulements doivent donner des résultats cohérents.

---

> **⚠️ Après l'implémentation de ce sprint** : mettre à jour `docs/roadmap_phase1.md` en marquant S12-01 à S12-11 comme ✅. Ajouter une ligne CWRU dans la table comparative des datasets (`docs/context/datasets.md`). Vérifier que `FIXME(gap1)` dans les notebooks Phase 1 peut pointer vers exp_074–085 en complément de exp_050–055.
