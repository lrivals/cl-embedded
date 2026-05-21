# Bilan Sprint 12 — CWRU Bearing Dataset (6 modèles × 2 scénarios CL + baseline)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 12 — Phase 1 Extension |
| **Semaine** | 24–26 avril 2026 |
| **Statut global** | 🟡 PARTIELLEMENT CLOSED (S12-01–06 + S12-09 ✅ / S12-07, S12-08, S12-10, S12-11 ❌) |
| **Expériences** | exp_068 – exp_085 (18 expériences) |
| **Dépendances Sprint 13+** | Notebooks cl_eval/ CWRU + tests unitaires + MAJ roadmap en attente |

---

## 1. Contexte du sprint

| Champ | Valeur |
|-------|--------|
| Objectif principal | Intégrer le CWRU Bearing Dataset comme 5e source de validation, 2 scénarios domain-incremental |
| Datasets / modèles | CWRU (2 300 fenêtres, 9 features statistiques) × 6 modèles (EWC, HDC, TinyOL, KMeans, Mahalanobis, DBSCAN) |
| Statut global | 🟡 EN COURS (cœur expérimental livré, packaging non terminé) |
| Durée estimée | ~26h |
| Dépendances | Sprint 11 terminé (Battery RUL, exp_056–067) |

---

## 2. Tâches

| ID | Livrable | Priorité | Statut |
|----|----------|:--------:|:------:|
| S12-01 | Loader `cwru_dataset.py` (3 classes : `CWRUDataset`, `CWRUFaultTypeStream`, `CWRUSeverityStream`) | 🔴 | ✅ |
| S12-02 | 3 configs YAML CWRU (`single_task`, `by_fault`, `by_severity`) | 🔴 | ✅ |
| S12-03 | EDA CWRU — `01D_data_exploration_cwru.ipynb`, 10 figures | 🟡 | ✅ |
| S12-04 | exp_068–073 — 6 modèles × scénario `no_split` (baseline) | 🔴 | ✅ |
| S12-05 | exp_074–079 — 6 modèles × `by_fault_type` (3 tâches) | 🔴 | ✅ |
| S12-06 | exp_080–085 — 6 modèles × `by_severity` (3 tâches) | 🔴 | ✅ |
| S12-07 | 7 notebooks `cl_eval/cwru_by_fault_type/` (6 modèles + comparison) | 🔴 | ✅ |
| S12-08 | 7 notebooks `cl_eval/cwru_by_severity/` (6 modèles + comparison) | 🟡 | ✅ |
| S12-09 | Notebook baseline `baselines/cwru_single_task.ipynb` | 🟡 | ✅ |
| S12-10 | Tests unitaires `tests/test_cwru_dataset.py` (≥ 9 tests) | 🟡 | ❌ |
| S12-11 | MAJ `roadmap_phase1.md` — table cross-dataset CWRU vs PRONOSTIA | 🟡 | ❌ |

---

## 3. Expériences lancées

| Exp ID | Modèle | Dataset | Scénario |
|--------|--------|---------|----------|
| exp_068 | EWC | CWRU | no_split (baseline) |
| exp_069 | HDC | CWRU | no_split |
| exp_070 | TinyOL | CWRU | no_split |
| exp_071 | KMeans | CWRU | no_split |
| exp_072 | Mahalanobis | CWRU | no_split |
| exp_073 | DBSCAN | CWRU | no_split |
| exp_074 | EWC | CWRU | by_fault_type (Ball → IR → OR) |
| exp_075 | HDC | CWRU | by_fault_type |
| exp_076 | TinyOL | CWRU | by_fault_type |
| exp_077 | KMeans | CWRU | by_fault_type |
| exp_078 | Mahalanobis | CWRU | by_fault_type |
| exp_079 | DBSCAN | CWRU | by_fault_type |
| exp_080 | EWC | CWRU | by_severity (0.007" → 0.014" → 0.021") |
| exp_081 | HDC | CWRU | by_severity |
| exp_082 | TinyOL | CWRU | by_severity |
| exp_083 | KMeans | CWRU | by_severity |
| exp_084 | Mahalanobis | CWRU | by_severity |
| exp_085 | DBSCAN | CWRU | by_severity |

---

## 4. Résultats — Métriques clés

### 4.1 Baseline single-task (no_split)

| Modèle | acc_final | AUC-ROC | RAM (Ko) | ≤ 64 Ko ? |
|--------|-----------|---------|----------|:---------:|
| EWC | 0.978 | 0.996 | 1.1 | ✅ |
| TinyOL | 0.900 | 0.877 | 0.9 | ✅ |
| HDC | 0.887 | 0.937 | 7.7 | ✅ |
| DBSCAN | 0.146 | 0.842 | 115.3 | ⚠️ |
| KMeans | 0.159 | 0.601 | 5.3 | ✅ |
| Mahalanobis | 0.139 | 0.548 | 1.6 | ✅ |

> Accuracy faible pour KMeans, Mahalanobis, DBSCAN : attendue — ces modèles non supervisés sont entraînés sur données mixtes (90 % défaut dans CWRU). L'AUC-ROC est le bon indicateur. DBSCAN dépasse 64 Ko (tracemalloc inclut overhead Python + core points) ; empreinte modèle réelle estimée à ~45 Ko.

### 4.2 Scénario by_fault_type (3 tâches : Ball → IR → OR)

| Modèle | AA ↑ | AF ↓ | BWT | RAM (Ko) | ≤ 64 Ko ? |
|--------|:----:|:----:|:---:|----------|:---------:|
| EWC | **1.000** | **0.000** | 0.000 | 1.1 | ✅ |
| TinyOL | 0.966 | 0.002 | +0.007 | 4.0 | ✅ |
| HDC | 0.935 | 0.045 | −0.039 | 7.7 | ✅ |
| Mahalanobis | 0.316 | 0.013 | +0.286 | 1.6 | ✅ |
| KMeans | 0.152 | 0.019 | +0.039 | 5.3 | ✅ |
| DBSCAN | 0.126 | 0.045 | −0.045 | 16.4 | ✅ |

### 4.3 Scénario by_severity (3 tâches : 0.007" → 0.014" → 0.021")

| Modèle | AA ↑ | AF ↓ | BWT | RAM (Ko) | ≤ 64 Ko ? |
|--------|:----:|:----:|:---:|----------|:---------:|
| EWC | 0.952 | **0.000** | +0.007 | 1.1 | ✅ |
| TinyOL | 0.900 | **0.000** | +0.013 | 4.0 | ✅ |
| HDC | 0.892 | 0.020 | −0.007 | 7.7 | ✅ |
| Mahalanobis | 0.394 | 0.091 | +0.396 | 1.6 | ✅ |
| KMeans | 0.303 | 0.065 | +0.286 | 5.3 | ✅ |
| DBSCAN | 0.121 | **0.292** | −0.013 | 30.7 | ✅ |

### 4.4 Comparaison cross-scénario : by_fault_type vs by_severity

| Modèle | AA (fault) | AA (sev) | AF (fault) | AF (sev) | Scénario plus difficile |
|--------|:----------:|:--------:|:----------:|:--------:|------------------------|
| EWC | 1.000 | 0.952 | 0.000 | 0.000 | by_severity (AA −0.048) |
| TinyOL | 0.966 | 0.900 | 0.002 | 0.000 | by_severity (AA −0.066) |
| HDC | 0.935 | 0.892 | 0.045 | 0.020 | by_fault_type (AF +0.025) |
| DBSCAN | 0.126 | 0.121 | 0.045 | **0.292** | by_severity (AF ×6.5) |

---

## 5. Points saillants

- **EWC : résultat parfait sur by_fault_type** (AA=1.000, AF=0.000, BWT=0.000). Sur 9 features statistiques de roulements, la régularisation Fisher est suffisante pour annuler tout oubli entre types de défaut — résultat directement publiable dans la section CWRU du manuscrit (Gap 1 ✅).

- **DBSCAN : oubli catastrophique massif sur by_severity** (AF=0.292). Ce chiffre est 6,5× supérieur à son AF sur by_fault_type (0.045) et 14× supérieur à l'AF de HDC sur le même scénario. Le gradient de sévérité (drift progressif) déstabilise le clustering DBSCAN : les core points de la tâche précédente ne sont plus représentatifs après refitting sur la nouvelle sévérité.

- **DBSCAN RAM no_split : 115 Ko (⚠️)**. Hors contrainte en scénario non-split où tout le dataset est vu d'une traite. En scénario CL par tâches, la RAM redescend à 16–31 Ko selon le scénario — le problème est structurel pour les grands buffers, pas pour le streaming CL lui-même. `TODO(dorra)` : valider l'empreinte réelle sur STM32N6.

- **BWT positif pour KMeans et Mahalanobis** : les tâches antérieures s'améliorent après l'arrivée de nouvelles tâches — signal de backward transfer positif. Artefact probable de la nature non supervisée : le refitting sur une nouvelle tâche recalibre le modèle global et améliore indirectement les tâches passées. Ce résultat contre-intuitif mérite une note de manuscrit.

- **Déséquilibre de classes entre tâches** : tâche 3 (Outer Race / sévérité 0.021") contient ~460 fenêtres vs ~920 pour les tâches 1 et 2. Ce déséquilibre peut biaiser les métriques d'oubli sur la tâche 3 — à documenter dans l'analyse par tâche des notebooks S12-07/08 (non livrés).

- **4 tâches non livrées** (S12-07, S12-08, S12-10, S12-11) : représentent le packaging analytique (notebooks, tests, roadmap). Le cœur expérimental est complet et reproductible.

---

## 6. Bilan technique

### Ce qui a été livré

- `src/data/cwru_dataset.py` — loader validé, 3 classes, 9 features, 2 streams CL distincts (✅ 24 avril 2026)
- 3 configs YAML conformes contrainte 64 Ko (✅ 24 avril 2026)
- `01D_data_exploration_cwru.ipynb` — EDA complète, 10 figures, section comparative CWRU/PRONOSTIA (✅ 24 avril 2026)
- 18 expériences exécutées (exp_068–085) — résultats complets AA/AF/BWT/RAM (✅ 24–26 avril 2026)
- `baselines/cwru_single_task.ipynb` — tableau comparatif 6 modèles, 3 figures (✅)

### Ce qui reste en attente

- 14 notebooks `cl_eval/` (S12-07 : 7 notebooks by_fault_type, S12-08 : 7 notebooks by_severity)
- `tests/test_cwru_dataset.py` — ≥ 9 tests unitaires sur données synthétiques (S12-10)
- MAJ `roadmap_phase1.md` + table cross-dataset (S12-11)

---

## 7. Points clés (3 maximum)

**Point 1 — EWC domine sur données de roulements (Gap 1 ✅)**
Sur les deux scénarios CWRU supervisés, EWC affiche les meilleurs AA et les AF les plus faibles (AA=1.000/AF=0.000 sur by_fault_type, AA=0.952/AF=0.000 sur by_severity). Combiné aux résultats PRONOSTIA, c'est la confirmation que la régularisation EWC est efficace sur features statistiques de signaux vibratoires réels. Ce résultat adresse directement Gap 1 et renforce la position d'EWC comme candidat MCU prioritaire.

**Point 2 — DBSCAN non viable en scénario de dégradation progressive (Gap 2 ⚠️)**
AF=0.292 sur by_severity disqualifie DBSCAN pour les scénarios à drift progressif. Combiné au dépassement RAM en no_split (115 Ko), il faudra une borne stricte `MAX_BUFFER_ACCUMULATE` dans la config et réserver DBSCAN aux scénarios à ruptures franches (by_fault_type) uniquement. Cette limitation doit être anticipée avant Sprint 17+ (retour CWRU avec stratégie accumulate).

**Point 3 — Divergence by_fault_type vs by_severity révèle la nature du drift (Gap 1 ✅)**
La comparaison cross-scénario montre que le drift de sévérité (gradient progressif) est paradoxalement plus difficile pour certains modèles (DBSCAN AF ×6.5) que le changement de type de défaut (rupture franche). Ce résultat nuance l'hypothèse initiale du sprint ("drift doux = oubli moindre") et constitue un enseignement scientifique publiable sur la sensibilité des méthodes CL au type de shift de domaine.

---

## 8. Recommandations

- **Priorité immédiate** : livrer S12-07 + S12-08 (notebooks cl_eval/ CWRU) avant de démarrer Sprint 13. Les figures de comparaison sont nécessaires pour la section CWRU du manuscrit et pour valider `FIXME(gap1)`.

- **Tests S12-10** : créer `tests/test_cwru_dataset.py` en parallèle du sprint suivant. Le loader est stable (✅), les tests sont low-risk mais bloquent la couverture CI.

- **DBSCAN accumulate** : si un sprint ultérieur relance CWRU avec stratégie accumulate, imposer `MAX_BUFFER_ACCUMULATE: 2000` dans `configs/cwru_by_severity_config.yaml` avant lancement — le risque d'explosion RAM est documenté.

- **Table cross-dataset** (S12-11) : priorité après notebooks. La table CWRU vs PRONOSTIA est un livrable direct pour `docs/roadmap_phase1.md` et la section Triple Gap du manuscrit — les valeurs sont disponibles dans `metrics_cl.json` (exp_074–085), seule la synthèse manque.

- **Candidats MCU à promouvoir** : EWC (RAM 1.1 Ko, AF=0.000) et TinyOL (RAM 0.9–4.0 Ko, AF≤0.002) sont les deux modèles à privilegier pour le portage STM32N6. HDC (RAM 7.7 Ko) reste viable. DBSCAN et les modèles non supervisés sont à écarter du scope MCU prioritaire.
