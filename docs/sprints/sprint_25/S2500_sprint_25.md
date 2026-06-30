# Sprint 25 — Tâches Natives des Datasets : RUL Régression + Classification Multi-classe (PC Python)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 25 |
| **Semaine** | 15 – 28 juillet 2026 |
| **Statut** | ✅ Terminé — reproduction validée 2026-06-12 |
| **Priorité globale** | 🔴 Critique — exploiter les tâches d'origine des datasets (RUL continu, multi-class fault) pour contribution manuscrit |
| **Durée estimée totale** | ~38h |
| **Dépendances** | Sprint 24 ✅ (profiling unifié, ONNX étendu, notebook comparatif) |

---

## Contexte et motivation

Tous les datasets du projet ont été uniformisés en **classification binaire normale/défaut** pour le framework CL. Cependant plusieurs datasets ont été conçus pour des tâches plus riches qui correspondent mieux aux besoins industriels réels :

| Dataset | Tâche officielle | Tâche actuelle projet | Gap |
|---------|-----------------|----------------------|-----|
| **CMAPSS** | Régression RUL continue (cycles restants jusqu'à défaillance) | Binaire RUL ≤ 30 | Signal prognostique perdu |
| **Pronostia** | Prognostique dégradation bearing (run-to-failure) | Binaire engineered (dernier 10%) | Label artificiel, pas de RUL réel |
| **Battery** | Régression RUL (cycles restants) | Binaire RUL < 200 | Idem CMAPSS |
| **CWRU** | Classification multi-classe (type + sévérité de défaut) | Binaire (normal vs. tout défaut) | 10 classes → 1 bit |
| **Paderborn** | Classification multi-état bearing (sain / outer-race / inner-race) | Binaire | 3 états → 1 bit |
| **Pump / Monitoring** | Détection d'anomalie | Binaire CL par domaine | ✅ Cohérent |

Sprint 25 corrige ce décalage en ajoutant un mode **natif** à chaque loader et en implémentant les têtes de modèle correspondantes dans le framework CL existant.

---

## Objectifs

1. **Loaders étendus** : ajouter mode `rul` (sortie float continue) et `multiclass` (labels 0..N) aux loaders CMAPSS, Pronostia, Battery, CWRU, Paderborn — sans casser le mode binaire existant
2. **Modèles adaptés** : têtes EWC régression (MSE) et EWC multi-class (softmax + cross-entropy), HDC régression (somme pondérée de prototypes)
3. **Métriques natives** : RMSE, MAE, Horizon Score PHM 2008 pour RUL ; F1-macro, matrice de confusion pour multi-class
4. **Expériences PC** : 5 expériences couvrant CMAPSS RUL, Pronostia RUL, CWRU multi-class, HDC régression, profiling RAM

```
Loaders (mode binaire existant conservé)
  cmapss_loader.py [mode=rul]      cwru_dataset.py [mode=multiclass]
  pronostia_dataset.py [mode=rul]  paderborn_loader.py [mode=multiclass]
  battery_dataset.py [mode=rul]
          ↓
  ewc_mlp_regression.py            ewc_mlp_multiclass.py
  (1 sortie, MSE + EWC)            (N sorties softmax + EWC)
          ↓                                ↓
  hdc_regressor.py
  (somme pondérée prototypes)
          ↓
  rul_metrics.py                   multiclass_metrics.py
  (RMSE, MAE, Horizon Score)       (F1-macro, confusion matrix)
          ↓
  configs YAML (cmapss_rul, pronostia_rul, cwru_multiclass, paderborn_multiclass)
          ↓
  scripts/train_ewc_rul.py         scripts/train_ewc_multiclass.py
          ↓
  exp_S25_01 EWC RUL CMAPSS        exp_S25_03 EWC multi-class CWRU
  exp_S25_02 EWC RUL Pronostia     exp_S25_04 HDC régression CMAPSS
  exp_S25_05 RAM profiling nouveaux modèles
```

**Critères de succès** :
1. `experiments/exp_S25_01/results.json` — RMSE par tâche + Average Forgetting en RMSE (CMAPSS FD001→FD004)
2. `experiments/exp_S25_03/results.json` — F1-macro par tâche + AF (CWRU 3 tâches by_fault_type)
3. `pytest tests/test_ewc_regression.py tests/test_ewc_multiclass.py -v` — tous verts
4. Mode binaire existant non cassé : `pytest tests/ -v` — 0 régression
5. `experiments/exp_S25_05/` — RAM profiling 3 nouveaux modèles (ewc_regression, ewc_multiclass, hdc_regressor)

---

## Tâches

### O1 — Extension des loaders (mode natif)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2501 | Étendre `cmapss_loader.py` : ajouter paramètre `mode: Literal["binary", "rul"]` — en mode `rul`, retourner RUL continu capé (clip 125) au lieu de la binarisation | 🔴 | ⬜ | `src/data/cmapss_loader.py` | 2h |
| S2502 | Étendre `pronostia_dataset.py` : calculer RUL réel = `(n_windows_total - window_idx) * WINDOW_DURATION_S` par bearing, exposer en mode `rul` | 🔴 | ⬜ | `src/data/pronostia_dataset.py` | 2h |
| S2503 | Étendre `battery_dataset.py` : mode `rul` — exposer colonne RUL brute (cycles restants) sans binarisation | 🟡 | ⬜ | `src/data/battery_dataset.py` | 1h |
| S2504 | Étendre `cwru_dataset.py` : mode `multiclass` — retourner labels 0–9 (Normal=0, Ball_007=1 … Outer_021=9) via mapping LabelEncoder | 🔴 | ⬜ | `src/data/cwru_dataset.py` | 2h |
| S2505 | Étendre `paderborn_loader.py` : mode `multiclass` — retourner labels 0/1/2 (K001=0, KA04=1, KI04=2) | 🟡 | ⬜ | `src/data/paderborn_loader.py` | 1h |

### O2 — Nouveaux modèles

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2506 | Implémenter `ewc_mlp_regression.py` : hérite du pattern `EWCMlpClassifier` — 1 neurone de sortie, perte MSE, EWC penalty inchangée, Fisher via gradient MSE | 🔴 | ✅ | `src/models/ewc/ewc_mlp_regression.py` | 3h |
| S2507 | Implémenter `ewc_mlp_multiclass.py` : N sorties softmax, cross-entropy, EWC penalty inchangée, Fisher sur poids softmax | 🔴 | ✅ | `src/models/ewc/ewc_mlp_multiclass.py` | 3h |
| S2508 | Implémenter `hdc_regressor.py` : pas de classes, prototype unique accumulateur + somme pondérée pour prédiction continue (HDC régression linéaire sur embeddings) | 🟡 | ✅ | `src/models/hdc/hdc_regressor.py` | 3h |

### O3 — Métriques natives

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2509 | Créer `rul_metrics.py` : `compute_rmse`, `compute_mae`, `compute_horizon_score` (pénalité asymétrique PHM 2008 : sur-estimation plus pénalisée qu'estimation précoce), `compute_avg_forgetting_rmse` | 🔴 | ✅ | `src/evaluation/rul_metrics.py` | 2h |
| S2510 | Créer `multiclass_metrics.py` : `compute_f1_macro`, `compute_confusion_matrix`, `compute_per_class_accuracy`, `compute_avg_forgetting_f1` | 🔴 | ✅ | `src/evaluation/multiclass_metrics.py` | 1h |

### O4 — Configs YAML

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2511 | Créer `configs/cmapss_rul_config.yaml` et `configs/pronostia_rul_config.yaml` — inclure `task_mode: rul`, `rul_cap: 125` (CMAPSS), hyperparamètres EWC régression | 🔴 | ⬜ | `configs/cmapss_rul_config.yaml`, `configs/pronostia_rul_config.yaml` | 1h |
| S2512 | Créer `configs/cwru_multiclass_config.yaml` et `configs/paderborn_multiclass_config.yaml` — inclure `task_mode: multiclass`, `n_classes: 10` / `3`, hyperparamètres EWC multi-class | 🔴 | ⬜ | `configs/cwru_multiclass_config.yaml`, `configs/paderborn_multiclass_config.yaml` | 1h |

### O5 — Scripts d'entraînement

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2513 | Créer `scripts/train_ewc_rul.py` : boucle CL sur CMAPSS FD001→FD004 (ou Pronostia C1→C3) en mode rul, logge RMSE par tâche + forgetting | 🔴 | ✅ | `scripts/train_ewc_rul.py` | 2h |
| S2514 | Créer `scripts/train_ewc_multiclass.py` : boucle CL par_fault_type CWRU (ou Paderborn) en mode multiclass, logge F1-macro par tâche + forgetting | 🔴 | ✅ | `scripts/train_ewc_multiclass.py` | 2h |

### O6 — Expériences PC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2515 | exp_S25_01 : EWC RUL / CMAPSS FD001→FD002→FD003→FD004 — RMSE par tâche, Average Forgetting en RMSE, BWT | 🔴 | ✅ | `experiments/exp_S25_01/` | 2h |
| S2516 | exp_S25_02 : EWC RUL / Pronostia Condition1→2→3 — RMSE par tâche, AF | 🟡 | ✅ | `experiments/exp_S25_02/` | 2h |
| S2517 | exp_S25_03 : EWC multi-class / CWRU by_fault_type (10 classes, 3 tâches) — F1-macro par tâche, matrice de confusion finale, AF | 🔴 | ✅ | `experiments/exp_S25_03/` | 2h |
| S2518 | exp_S25_04 : HDC régression / CMAPSS FD001→FD004 — RMSE par tâche (comparaison vs EWC RUL) | 🟡 | ✅ | `experiments/exp_S25_04/` | 2h |
| S2519 | exp_S25_05 : RAM profiling ewc_regression + ewc_multiclass + hdc_regressor sur CMAPSS et CWRU | 🟡 | ✅ | `experiments/exp_S25_05/` | 2h |

### O7 — Tests

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2520 | Tests unitaires `test_ewc_regression.py` : forward pass (output shape 1), MSE loss, consolidation Fisher, backward transfer | 🟡 | ⬜ | `tests/test_ewc_regression.py` | 1h |
| S2521 | Tests unitaires `test_ewc_multiclass.py` : forward pass (output shape N), softmax normalisé, EWC penalty, F1-macro non nul | 🟡 | ⬜ | `tests/test_ewc_multiclass.py` | 1h |
| S2522 | `pytest tests/ -v` — vérifier 0 régression sur tests binaires existants | 🟡 | ⬜ | `tests/` | 30 min |

### O8 — Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2523 | Mettre à jour `docs/roadmap_phase2.md` : Sprint 25 ajouté, motivation "tâches natives datasets" | 🟢 | ⬜ | `docs/roadmap_phase2.md` | 30 min |

---

## Ordre d'exécution recommandé

```
S2501 + S2504 (loaders CMAPSS + CWRU — prioritaires)
S2502 + S2503 + S2505 (loaders Pronostia + Battery + Paderborn)  [parallèle]
     ↓
S2509 + S2510 (métriques RUL + multiclass)  [parallèle]
     ↓
S2506 (EWC régression)
S2507 (EWC multiclass)   [parallèle]
S2508 (HDC regressor)
     ↓
S2511 + S2512 (configs YAML)
     ↓
S2513 + S2514 (scripts entraînement)
     ↓
S2515 → S2516 (expériences RUL)
S2517 → S2518 (expériences multiclass)  [parallèle avec RUL]
     ↓
S2519 (profiling RAM)
     ↓
S2520 + S2521 + S2522 (tests)
     ↓
S2523 (docs)
```

---

## Nomenclature des expériences

| Exp ID | Modèle | Dataset | Tâche native | Métriques |
|--------|--------|---------|-------------|-----------|
| exp_S25_01 | EWC Régression | CMAPSS | RUL continu | RMSE, MAE, Horizon Score, AF_rmse |
| exp_S25_02 | EWC Régression | Pronostia | RUL continu (par bearing) | RMSE, AF_rmse |
| exp_S25_03 | EWC Multi-class | CWRU | 10 classes fault type+severity | F1-macro, confusion matrix, AF_f1 |
| exp_S25_04 | HDC Régression | CMAPSS | RUL continu | RMSE vs EWC |
| exp_S25_05 | EWC-reg + EWC-mc + HDC-reg | CMAPSS + CWRU | — | ram_peak_bytes, n_params |

---

## Métriques attendues

| Expérience | Critère de validation |
|-----------|----------------------|
| exp_S25_01 (EWC RUL CMAPSS) | RMSE_task1 < 30 cycles (ordre de grandeur SOTA CMAPSS simple), AF_rmse documenté |
| exp_S25_03 (EWC multiclass CWRU) | F1-macro > 0.70 sur tâche 1 (CWRU binaire déjà ~0.98, 10 classes plus difficile) |
| exp_S25_05 (profiling) | ram_peak_bytes(ewc_regression) < ram_peak_bytes(ewc_binary) + 20% (1 sortie vs 1 sortie) |
| Tests | 0 régression sur tests binaires existants |

---

## Livrables

1. Loaders étendus sans régression binaire : `cmapss_loader.py`, `pronostia_dataset.py`, `battery_dataset.py`, `cwru_dataset.py`, `paderborn_loader.py`
2. Nouveaux modèles : `src/models/ewc/ewc_mlp_regression.py`, `src/models/ewc/ewc_mlp_multiclass.py`, `src/models/hdc/hdc_regressor.py`
3. Nouveaux modules évaluation : `src/evaluation/rul_metrics.py`, `src/evaluation/multiclass_metrics.py`
4. Configs YAML : `cmapss_rul_config.yaml`, `pronostia_rul_config.yaml`, `cwru_multiclass_config.yaml`, `paderborn_multiclass_config.yaml`
5. Scripts : `scripts/train_ewc_rul.py`, `scripts/train_ewc_multiclass.py`
6. 5 dossiers `experiments/exp_S25_01/` à `exp_S25_05/` avec `results.json` + `config_snapshot.yaml`
7. Tests : `tests/test_ewc_regression.py`, `tests/test_ewc_multiclass.py`

---

## Reproduction validée (2026-06-12)

Les 5 expériences PC et la suite de tests ont été **re-exécutées de bout en bout** (PC, seed=42, aucune carte requise — Sprint 25 = PC-only). Résultats déterministes **identiques** aux valeurs documentées :

| Exp | Métrique clé | Documenté | Reproduit |
|-----|--------------|-----------|-----------|
| exp_S25_01 — EWC RUL CMAPSS | RMSE_t1 / AF | 22.53 / 19.97 | ✅ 22.53 / 19.97 |
| exp_S25_02 — EWC RUL Pronostia | RMSE_t1 / AF | 83.68 / 9.67 | ✅ 83.68 / 9.67 |
| exp_S25_03 — EWC Multiclass CWRU | F1_t1 / AF | 0.955 / 0.848 | ✅ 0.955 / 0.848 |
| exp_S25_04 — HDC RUL CMAPSS | RMSE_t1 / AF | 23.43 / 20.17 | ✅ 23.43 / 20.17 |
| exp_S25_05 — RAM profiling | EWC_reg RAM | 27.9 Ko | ✅ 27.9 Ko |

- **Tests** : `test_ewc_regression` 7/7 + `test_ewc_multiclass` 8/8 PASSED ; suite complète **471 passed, 12 skipped, 2 failed** (les 2 échecs `test_board_recorder` ewc/monitoring sont pré-existants et hors périmètre S25).
- **Variation non-déterministe** : seuls les `ram_peak_bytes` tracemalloc PC de `ewc_multiclass` (~10–11 Ko) et `hdc_regressor` (~16–19 Ko) fluctuent d'un run à l'autre (overhead autograd PyTorch non représentatif MCU) ; tous restent < 256 Ko.

---

## Questions ouvertes

- `TODO(arnaud)` : La métrique Horizon Score (PHM 2008, pénalité asymétrique) est-elle requise pour le manuscrit ou suffisamment couverte par RMSE + MAE ?
- `TODO(arnaud)` : Le RMSE board (Sprint 26) sur CMAPSS constitue-t-il une contribution Gap 2 distincte, ou est-ce un complément du Gap 1 (données industrielles réelles) ?
- `TODO(dorra)` : Pour le portage board (Sprint 26), RUL en FP32 est-il suffisant ou faut-il un schéma de normalisation spécifique avant transmission UART ?
- `TODO(fred)` : Les cas d'usage Edge Spectrum visent-ils plutôt RUL continu (prognostic) ou détection de seuil (diagnostic) ? Cela conditionne la priorité exp_S25_01 vs exp_S25_03.
- `FIXME(gap1)` : CMAPSS est un dataset simulé (NASA) — vérifier si cela est compatible avec la revendication Gap 1 "données industrielles réelles" dans le manuscrit.
