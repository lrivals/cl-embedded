# Sprint 10 (Phase 1) — Dataset 3 : FEMTO PRONOSTIA (Bearing Degradation)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité globale** | 🔴 Critique — résout FIXME(gap1) dans toute la Phase 1 |
| **Durée estimée totale** | ~24h |
| **Dépendances** | Sprint 9 terminé (Phase 1 Python complète, roadmaps MAJ) |

---

## Objectif

Intégrer le dataset IEEE PHM 2012 **FEMTO PRONOSTIA** (Dataset 3) comme troisième source de validation du projet CL-Embedded. Ce sprint résout les `FIXME(gap1)` présents dans les expériences exp_003, exp_007, exp_009, exp_010 et dans tous les notebooks de comparaison de la Phase 1.

**Gap 1 comblé** : _Premier résultat CL publié sur données industrielles réelles de roulements_ — les datasets Kaggle (Pump Maintenance, Equipment Monitoring) sont des données semi-synthétiques; PRONOSTIA est un benchmark académique reconnu sur données réelles de dégradation de roulements à billes.

**Critère de succès** : 12 expériences Pronostia exécutées (exp_044–055), `pronostia_dataset.py` validé par tests unitaires, EDA section 3 complète dans `01_data_exploration.ipynb`, tous les `FIXME(gap1)` marqués ✅ dans `roadmap_phase1.md`.

---

## Contexte dataset

**Dataset** : FEMTO PRONOSTIA IEEE PHM 2012 — Learning Set  
**Source** : FEMTO-ST Institute / INSA Lyon (données réelles d'accélérométrie)  
**Chemin** : `data/raw/Pronostia dataset/binaries/`  
**Format** : fichiers `.npy` pré-convertis, shape `(N_epochs, 2, 2560)`

| Paramètre | Valeur |
|-----------|--------|
| Roulements | 6 (Bearing1_1/1_2, Bearing2_1/2_2, Bearing3_1/3_2) |
| Canaux | 2 (accélération horizontale + verticale) |
| Fréquence échantillonnage | 25,6 kHz |
| Durée epoch | 0,1 s → 2 560 points |
| WINDOW_SIZE | 2 560 (1 fenêtre = 1 epoch, sans overlap) |
| Features | 13 (6 stats × 2 canaux + temporal_position) |
| Label | Binaire — derniers 10% du signal = pré-défaillance (1) |

### Conditions opératoires (scénario CL domain-incremental)

| Tâche | Condition | Vitesse | Charge | Roulements | Fenêtres approx. |
|-------|-----------|---------|--------|------------|-----------------|
| Task 1 | Condition 1 | 1 800 rpm | 4 000 N | Bearing1_1 + Bearing1_2 | ~3 674 |
| Task 2 | Condition 2 | 1 650 rpm | 4 200 N | Bearing2_1 + Bearing2_2 | ~1 708 |
| Task 3 | Condition 3 | 1 500 rpm | 5 000 N | Bearing3_1 + Bearing3_2 | ~2 152 |

---

## Tâches

| ID | Tâche | Priorité | Fichier(s) cible(s) | Durée est. | Dépendances |
|----|-------|:---:|---------------------|:---:|-------------|
| S10-01 | `pronostia_dataset.py` — loader `.npy` + fenêtrage 2560 + label TTF binaire | 🔴 | `src/data/pronostia_dataset.py` | 4h | — |
| S10-02 | Valider configs YAML Pronostia (déjà créées) | 🔴 | `configs/pronostia_config.yaml`, `configs/pronostia_single_task_config.yaml` | 1h | S10-01 |
| S10-03 | EDA Pronostia — section 3 dans `01_data_exploration.ipynb` | 🟡 | `notebooks/01_data_exploration.ipynb` | 2h | S10-01 |
| S10-04 | Run exp_044–049 — 6 modèles × scénario `no_split` (single-task baseline) | 🟡 | `experiments/exp_044–049/` | 2h | S10-01 + S10-02 |
| S10-05 | Run exp_050–055 — 6 modèles × 3 conditions domain-incremental | 🔴 | `experiments/exp_050–055/` | 2h | S10-04 |
| S10-06 | Notebooks individuels `pronostia_by_condition/` — 6 modèles | 🔴 | `notebooks/cl_eval/pronostia_by_condition/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan}.ipynb` | 6h | S10-05 |
| S10-07 | Notebook comparaison + notebook baseline single-task | 🔴 | `notebooks/cl_eval/pronostia_by_condition/comparison.ipynb`, `notebooks/cl_eval/baselines/pronostia_single_task.ipynb` | 3h | S10-06 + S10-04 |
| S10-08 | Tests unitaires `pronostia_dataset.py` | 🟡 | `tests/test_pronostia_dataset.py` | 2h | S10-01 |
| S10-09 | MAJ roadmap : `FIXME(gap1)` → ✅ résolu dans tous les notebooks et `roadmap_phase1.md` | 🟡 | `docs/roadmap_phase1.md` | 1h | S10-05 à S10-07 |

> Détail : [S1001_pronostia_dataset.md](S1001_pronostia_dataset.md) · [S1002_configs_pronostia.md](S1002_configs_pronostia.md) · [S1003_eda_pronostia.md](S1003_eda_pronostia.md) · [S1004_exp_single_task_pronostia.md](S1004_exp_single_task_pronostia.md) · [S1005_exp_cl_by_condition_pronostia.md](S1005_exp_cl_by_condition_pronostia.md) · [S1006_notebooks_pronostia_by_condition.md](S1006_notebooks_pronostia_by_condition.md) · [S1007_notebook_comparison_pronostia.md](S1007_notebook_comparison_pronostia.md) · [S1008_tests_pronostia_dataset.md](S1008_tests_pronostia_dataset.md) · [S1009_maj_roadmap_gap1.md](S1009_maj_roadmap_gap1.md)

---

## Numérotation expériences

| Exp | Modèle | Scénario | Config | Statut |
|-----|--------|----------|--------|--------|
| exp_044 | EWC | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_045 | HDC | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_046 | TinyOL | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_047 | KMeans | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_048 | Mahalanobis | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_049 | DBSCAN | Pronostia no_split | `pronostia_single_task_config.yaml` | ⬜ |
| exp_050 | EWC | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |
| exp_051 | HDC | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |
| exp_052 | TinyOL | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |
| exp_053 | KMeans | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |
| exp_054 | Mahalanobis | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |
| exp_055 | DBSCAN | Pronostia by_condition (3 tâches) | `pronostia_config.yaml` | ⬜ |

---

## Critères d'acceptation

- [ ] S10-01 : `PronostiaDataset` et `PronostiaConditionStream` importables depuis `src.data.pronostia_dataset`
- [ ] S10-02 : `pronostia_config.yaml` et `pronostia_single_task_config.yaml` validés (chargement sans erreur)
- [ ] S10-03 : Section 3 dans `01_data_exploration.ipynb` exécutée sans erreur, figures sauvegardées dans `notebooks/figures/eda/pronostia/`
- [ ] S10-04 : exp_044–049 exécutées, `metrics_single_task.json` présent dans chaque dossier `experiments/exp_04X/results/`
- [ ] S10-05 : exp_050–055 exécutées, `metrics_cl.json` présent dans chaque dossier `experiments/exp_05X/results/`
- [ ] S10-06 : 6 notebooks dans `notebooks/cl_eval/pronostia_by_condition/`, 5 figures par modèle sauvegardées
- [ ] S10-07 : `comparison.ipynb` et `baselines/pronostia_single_task.ipynb` exécutés sans erreur
- [ ] S10-08 : `pytest tests/test_pronostia_dataset.py -v` → 100% pass (9 tests)
- [ ] S10-09 : `roadmap_phase1.md` : S10-01 à S10-09 marqués ✅, zéro `FIXME(gap1)` non résolu

---

## Livrable sprint 10

- **`src/data/pronostia_dataset.py`** — loader validé, 13 features, labels TTF binaires, stream par condition
- **`tests/test_pronostia_dataset.py`** — 9 tests unitaires sur données synthétiques (fixtures `tmp_path`)
- **12 expériences** (exp_044–055) — 6 single-task + 6 CL domain-incremental sur données industrielles réelles
- **EDA Section 3** dans `01_data_exploration.ipynb` — vibrométrie, distribution label, comparaison inter-conditions
- **8 notebooks** `cl_eval/` — 6 individuels by_condition + 1 comparaison + 1 baseline single-task
- **Gap 1 comblé** ✅ : premier résultat CL publié sur données industrielles réelles de roulements (FEMTO PRONOSTIA IEEE PHM 2012)

---

## Questions ouvertes

- `TODO(arnaud)` : La définition du label TTF binaire (derniers 10% = pré-défaillance) est-elle cohérente avec le protocole IEEE PHM 2012 Challenge ? Faut-il utiliser le temps-à-la-panne absolu fourni dans `Bearing_*_RUL.txt` pour calibrer ce seuil roulement par roulement ?
- `TODO(dorra)` : Le scénario domain-incremental par condition (3 tâches) est-il pertinent pour le STM32N6 ? Ou faut-il un scénario task-incremental par roulement individuel (6 tâches) pour mieux stresser la méthode CL ?
- `TODO(fred)` : Dans le contexte industriel Edge Spectrum, les conditions opératoires (vitesse + charge) sont-elles connues à l'avance (supervision possible) ou inconnues (détection de drift nécessaire) ?
- `FIXME(gap1)` : Comparer les métriques AA/AF/BWT obtenues sur PRONOSTIA avec les résultats Monitoring et Pump pour construire la table de synthèse Triple Gap du manuscrit.

---

> **⚠️ Après l'implémentation de ce sprint** : mettre à jour `docs/roadmap_phase1.md` en marquant S10-01 à S10-05 comme ✅. Vérifier que tous les `FIXME(gap1)` dans les notebooks de Phase 1 peuvent maintenant pointer vers exp_050–055.
