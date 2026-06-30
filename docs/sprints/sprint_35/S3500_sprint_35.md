# Sprint 35 — Étude d'impact du nombre de features (fault detection) : 5-feat / all-feat / best-feat-par-modèle

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 35 |
| **Semaine** | 4 – 10 août 2026 |
| **Statut** | ✅ Implémenté (S3501–S3514) |
| **Priorité globale** | 🔴 Critique — Quantifier l'impact du **choix des features** sur la détection de panne, pour **5 datasets × 4 modèles**, sur **3 conditions** (`5feat` référence board / `all` dims natives / `best` meilleures features par modèle), **sur PC ET board ré-architecturé**. Régénérer les heatmaps **F1 + acc_final** par condition, les intégrer à la présentation, mettre à jour l'analyse, et **corriger l'artefact HDC×monitoring = 0.113** (re-run board correct). |
| **Durée estimée totale** | ~40h |
| **Dépendances** | `src/evaluation/feature_importance.py` ✅ (`permutation_importance`) · `configs/*_feature_subset.yaml` ✅ (top-5 mutual-info par dataset) · `scripts/generate_comparison_sprint23.py` ✅ (`comparison_sprint23.json`) · `notebooks/board_benchmark_all_datasets.ipynb` ✅ (`_heatmap_acc`) · `scripts/export_weights_c.py` ✅ · `scripts/sensor_stream.py` ✅ (protocole variable `[N:1B][features:f32×N]`) · firmware dims figées Sprint 32 · infra parité board↔PC Sprints 26-32 ✅ |

---

## Contexte et motivation

La heatmap « accuracy cross-dataset (Gap 1) » de la présentation
(`docs/presentation_seminaire_juin2026/presentation_plots.ipynb`, Slide 6, `gap1_gap2_heatmap_acc.png`)
expose `acc_final` **board** sur firmware **5 features** (Sprint 32), métrique **accuracy uniquement**.
Trois limites motivent ce sprint.

### Limite 1 — 5 features arbitraires (board figé)

Le firmware NUCLEO-F439ZI est câblé à 5 features pour la plupart des modèles :
`EWC_IN=5` (`inc/ewc_head.h:10`), `MAHA_DIM=5` (`inc/mahalanobis.h:5`), `TINYOL_IN=5` (`inc/tinyol.h:19`),
`HDC_N_FEATURES=5` (`inc/hdc.h:12`) ; seul `EWC_MC_IN=9` (`inc/ewc_head_multiclass.h:21`).
On ignore le **coût/gain** de ce choix vs toutes les features natives, ou vs un sous-ensemble **optimal par modèle**.

### Limite 2 — Accuracy trompeuse (manque F1)

Pour la détection de panne déséquilibrée, l'accuracy masque l'effondrement de la classe `faulty`
(cf. Sprint 26 : `F1_MC=0.243` alors que l'accuracy paraissait bonne). La heatmap **F1** manque.

### Limite 3 — Artefact HDC×monitoring = 0.113

`experiments/exp_S33_board_gap1/results_hdc_monitoring.json` porte
`note_feature: "monitoring zéro-paddé 4→5 feat (5ᵉ synthétique nulle)"` : monitoring est natif **4 features**,
zéro-paddé à 5, ce qui fait s'effondrer la projection HDC embarquée → `acc_final=0.1133` (dégénéré).
Valeur PC légitime = **0.8498** (`exp_S33_PC_hdc_monitoring`).

```
Limite 1 (5 features arbitraires)              Sprint 35 — Parties A+B+C
                                       ──▶  S3501 best features par modèle (perm. importance)
                                            S3502 configs condition `all`
                                            S3503 re-run PC × 3 conditions
                                            S3506-S3508 board ré-architecturé × 3 conditions

Limite 2 (accuracy trompeuse)                  Sprint 35 — Partie B+D
                                       ──▶  S3504 métrique F1 (classe faulty) PC+board
                                            S3510 heatmaps F1 + acc_final par condition

Limite 3 (artefact HDC×monitoring)             Sprint 35 — Partie C
                                       ──▶  S3509 re-run board HDC×monitoring corrigé
```

---

## Décisions validées (utilisateur)

- **Plateformes** : PC **+ board ré-architecturé** (dims d'entrée configurables au build, **1 reflash par condition**).
- **Datasets** : les **5** (cwru, monitoring, pronostia, paderborn + **cmapss binarisé au seuil de référence**, cohérent avec la heatmap existante).
- **Best features** : **permutation importance par modèle** (réutilise `src/evaluation/feature_importance.py`), top-k **optimisé sur F1 de validation**.
- **Fix HDC×monitoring** : **re-run board corrigé** (sans zéro-padding 4→5 dégénéré).

---

## Contraintes techniques à honorer

- **Protocole UART déjà variable** : `[N:1B][features:f32×N]`, mais `PROTO_MAX_N=16` (`firmware/stm32f4_blink/src/pipeline.c:36`). **CMAPSS natif = 21 sensors > 16** → la condition `all` impose soit de relever `PROTO_MAX_N` (≥21) avec `g_stream_storage`/`payload` qui en dépendent, soit de restreindre `all` au sous-ensemble exploitable. → tranché en S3506 (`TODO(dorra)` RAM vs PROTO_MAX_N).
- **Pas de hardcode** (règle CLAUDE.md) : dims modèles via `#define` dans `inc/` + flag de build ; jamais d'hyperparamètres dans le source (→ configs YAML).
- **Parité board↔PC** garantie seulement pour **EWC + Mahalanobis** (poids exportés via `export_weights_c.py`). **HDC** (projection embarquée) et **TinyOL** (init en ligne) restent **HW-only, parité N/A** par construction (Sprint 32) — inchangé par les conditions `all`/`best`.
- **F1** : à calculer/stocker en plus de `acc_final` (réutiliser `src/evaluation/anomaly_metrics.py` / `online_metrics.py`). Les `results.json` actuels ne l'exposent pas tous.
- **Aucun chiffre board inventé** : champs « à mesurer » tant que la NUCLEO n'a pas tourné.

---

## Critères de succès

1. `configs/best_features/{model}_{dataset}.yaml` produits pour les 4 modèles × 5 datasets (permutation importance, k optimisé sur F1 val).
2. Re-run PC complet : `experiments/exp_S35_PC_{condition}_{model}_{dataset}/results.json` avec **F1 ET acc_final** pour `condition ∈ {5feat, all, best}`.
3. Firmware ré-architecturé : dims modèles configurables au build (`#define`), **0 régression** sur la condition 5-feat existante ; décision `PROTO_MAX_N` documentée.
4. Re-run board par condition : `experiments/exp_S35_board_{condition}_{model}_{dataset}/`, parité board↔PC vérifiée (EWC+Maha), Gap 2 (< 100 ms) préservé.
5. **Artefact HDC×monitoring corrigé** : 0.113 remplacé par la vraie valeur dans `comparison_sprint23.json` et la heatmap.
6. **12 heatmaps** régénérées : `{F1, acc_final} × {5feat, all, best} × {board, pc}` → `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png`.
7. Plots intégrés à `presentation_plots.ipynb` (Slide 6 / nouvelle slide) ; analyse mise à jour.
8. `pytest tests/ -k "feature_selection or heatmap"` verts ; `make test` Unity verts (si dims firmware touchées).

---

## Tâches

### Partie A — Sélection de features par modèle (PC)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3501 | `scripts/select_best_features_per_model.py` : permutation importance par `(modèle, dataset)` (réutilise `permutation_importance`), top-k maximisant le **F1 de validation** → `configs/best_features/{model}_{dataset}.yaml`. | 🔴 | `scripts/select_best_features_per_model.py`, `configs/best_features/*.yaml` | 4h |
| S3502 | Configs condition `all` (dims natives) par dataset, alignées sur les loaders `src/data/`. | 🟡 | `configs/all_features/*.yaml` | 2h |

### Partie B — Re-run PC, 3 conditions (5feat / all / best)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3503 | Driver `scripts/run_feature_condition_sweep.py` : pour `condition × modèle × dataset`, train + eval **F1 + acc_final** → `experiments/exp_S35_PC_{condition}_{model}_{dataset}/`. | 🔴 | `scripts/run_feature_condition_sweep.py`, `experiments/exp_S35_PC_*` | 5h |
| S3504 | F1 (classe `faulty`) calculé/stocké en plus d'`acc_final`, PC **et** board (réutilise `anomaly_metrics.py`/`online_metrics.py`). | 🔴 | `src/evaluation/metrics.py`, `scripts/sensor_stream.py` | 3h |
| S3505 | RAM profiling des conditions (`scripts/profile_memory.py`) — exigence CLAUDE.md. | 🟡 | `experiments/exp_S35_PC_*/ram.json` | 2h |

### Partie C — Board ré-architecturé (dims configurables)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3506 | Paramétrer `EWC_IN`/`MAHA_DIM`/`TINYOL_IN`/`HDC_N_FEATURES` au build (`#define` + flag) ; statuer `PROTO_MAX_N≥21` vs restriction `all` ; **0 hardcode**, **0 régression** 5-feat. | 🔴 | `firmware/stm32f4_blink/inc/*.h`, `firmware/stm32f4_blink/src/pipeline.c` | 6h |
| S3507 | Regénérer poids/projections par condition (`export_weights_c.py`), matrice de build, **1 flash/condition**. | 🔴 | `scripts/export_weights_c.py`, `firmware/stm32f4_blink/Makefile` | 4h |
| S3508 | Streaming board par condition → `experiments/exp_S35_board_{condition}_{model}_{dataset}/` ; parité board↔PC (EWC+Maha) ; HDC/TinyOL HW-only. | 🔴 | `scripts/run_feature_condition_board.py`, `experiments/exp_S35_board_*` | 5h |
| S3509 | Re-run board **HDC×monitoring corrigé** (sans zéro-padding dégénéré) → remplace 0.113 par la vraie valeur dans `comparison_sprint23.json`. | 🔴 | `experiments/exp_S35_board_5feat_hdc_monitoring/`, `experiments/comparison_sprint23.json` | 3h |

### Partie D — Heatmaps, présentation, analyse

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3510 | Étendre `generate_comparison_sprint23.py` + le notebook board pour produire **F1 + acc_final × {5feat, all, best} × {board, pc}** → 12 PNG `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png`. | 🔴 | `scripts/generate_comparison_sprint23.py`, `notebooks/board_benchmark_all_datasets.ipynb`, `docs/figures/*` | 5h |
| S3511 | Ajouter les figures à `presentation_plots.ipynb` (Slide 6 / nouvelle slide) + maj `01_structure.md`/`02_script.md`. | 🟡 | `docs/presentation_seminaire_juin2026/*` | 2h |
| S3512 | Mettre à jour l'analyse (impact features × modèle, F1 vs acc, gain/coût du choix 5-feat, RAM/latence par condition). | 🟡 | `docs/datasets_analysis.md` ou `docs/sprints/sprint_35/S3512_analysis_update.md` | 3h |

### Partie E — Tests & docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3513 | Tests `tests/test_feature_selection.py`, `tests/test_heatmap_builders.py` (+ Unity firmware si dims configurables touchées). | 🟢 | `tests/test_feature_selection.py`, `tests/test_heatmap_builders.py` | 3h |
| S3514 | MAJ `CLAUDE.md` (statut Sprint 35), `docs/roadmap_phase2.md` ; invoquer `graphify_sprint_update`. | 🟢 | `CLAUDE.md`, `docs/roadmap_phase2.md` | 1h |

---

## Ordre d'exécution recommandé

```
Partie A          Partie B                 Partie C (board)            Partie D            Partie E
S3501 ─┐          S3503 ──┐                S3506 ──┐                                       
S3502 ─┴─▶ S3503  S3504 ──┼─▶ S3505        S3507 ──┼─▶ S3508 ─▶ S3509 ─▶ S3510 ─▶ S3511   S3513
                                                                          └─▶ S3512        S3514
```

---

## Nomenclature des expériences

| Exp ID | Sujet | Mesure |
|--------|-------|--------|
| exp_S35_PC_{condition}_{model}_{dataset}/results.json | re-run PC par condition | F1, acc_final, avg_forgetting, RAM |
| exp_S35_board_{condition}_{model}_{dataset}/ | re-run board par condition | F1, acc_final board, latence DWT, .bss, parité |
| exp_S35_board_5feat_hdc_monitoring/ | fix artefact HDC | vraie valeur HDC×monitoring board |

`condition ∈ {5feat, all, best}` · `model ∈ {mahalanobis, ewc, tinyol, hdc}` · `dataset ∈ {cwru, monitoring, pronostia, cmapss, paderborn}`

---

## Questions ouvertes

- `TODO(arnaud)` : seuil de binarisation CMAPSS retenu pour la condition fault-detection (cohérence avec `_S32_REFERENCE_THRESHOLD`) ?
- `TODO(dorra)` : relever `PROTO_MAX_N` à ≥21 (CMAPSS `all`) acceptable côté RAM, ou restreindre la condition `all` au sous-ensemble ≤16 ?
- `FIXME(gap2)` : confirmer latence board < 100 ms pour la condition `all` (dims plus grandes → forward plus coûteux, surtout HDC).
- `TODO(arnaud)` : pour HDC/TinyOL (HW-only), la condition `best/all` reste-t-elle « parité N/A » ou veut-on un protocole de comparaison HW-only dédié ?

---

## Livrables

1. `scripts/select_best_features_per_model.py` + `configs/best_features/*.yaml` + `configs/all_features/*.yaml`
2. `scripts/run_feature_condition_sweep.py` + `experiments/exp_S35_PC_*` (F1 + acc_final, 3 conditions)
3. Firmware dims configurables + `experiments/exp_S35_board_*` (parité, 3 conditions)
4. Artefact HDC×monitoring corrigé dans `comparison_sprint23.json`
5. 12 heatmaps `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png` + intégration présentation
6. Analyse mise à jour
7. Tests Python + Unity + MAJ `CLAUDE.md`/`roadmap_phase2.md` + `graphify_sprint_update`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3501 best features par modèle | ✅ | — | `scripts/select_best_features_per_model.py` + moteur `src/evaluation/feature_conditions.py` ; smoke EWC×CWRU → `configs/best_features/ewc_cwru.yaml` (k*=1, `sd`) |
| S3502 configs `all` | ✅ | — | 5 configs `configs/all_features/*.yaml` (dims natives 9/4/13/21/7, importées des loaders) |
| S3503 re-run PC 3 conditions | ✅ | — | `scripts/run_feature_condition_sweep.py` (`--dry-run` = 60 cellules) ; smoke best×ewc×cwru + all×maha×cwru produits ; sweep complet à lancer |
| S3504 métrique F1 PC+board | ✅ | — | `metrics.compute_fault_f1` (def. partagée PC↔board) ; `feature_conditions`/sweep + `sensor_stream._compute_stats` (hôte, **0 changement UART**) ; 3 tests `-k f1` PASS |
| S3505 RAM profiling | ✅ | — | `profile_memory.py --condition` (tracemalloc réutilisé de `train_and_evaluate`) → `ram.json` ; smoke `best_ewc_cwru`/`all_mahalanobis_cwru` ; 60 complets après sweep S3503 |
| S3506 firmware dims configurables | ✅ | — | `#ifndef` EWC_IN/MAHA_DIM/TINYOL_IN/HDC_N_FEATURES + `PROTO_MAX_N` ; **Option 1** retenue (`PROTO_MAX_N=21` → +100 B `.bss`) ; gardes `WEIGHTS_NATIVE_DIM` (poids réels = S3507) ; **`.bss` 5feat 104 956 B inchangé** ; `make test` 0 régression (2 TinyOL préexistants) |
| S3507 export + build matrix | ✅ | — | `export_weights_c.py` EWC dim variable (`fc1.weight.shape[1]`) + `#define EWC_HEAD_NATIVE_DIM`/`MAHA_NATIVE_DIM` ; gardes firmware `pipeline.c` par-modèle (fallback `WEIGHTS_NATIVE_DIM` → 0 régression 5feat) ; résolveur `--condition/--model/--dataset` ; builds 5/9/21 OK |
| S3508 streaming board 3 conditions | ✅ | — | Source unique `resolve_feature_indices`/`load_condition_arrays` (parité par construction) ; `sensor_stream --condition` (0 UART) ; driver `run_feature_condition_board.py` (train→export→build→flash→stream, sans `--update`) ; **parité EWC+Maha exacte vérifiée board réelle** sur `all`+`5feat` (k=4→21, ex. cmapss k=21 EWC 79µs/Maha 34µs ≪ 100ms) ; **paderborn débloqué** (bug normalizer data-layer corrigé : refit si cache ne couvre pas les features natives) — 7 cellules `(cond,ds)` reflashées (paderborn ×3 + best ×{cwru,monitoring,pronostia,cmapss}), **parité EWC+Maha 30/30** |
| S3509 fix HDC×monitoring | ✅ | — | Re-run board condition `all` (4-feat natif, sans padding) → `online_accuracy=0.8788` (≈ PC 0.85, vs artefact 0.1133) ; `_apply_s3509_override` corrige `comparison_sprint23.json` (jamais à la main) |
| S3510 heatmaps par condition | ✅ | — | `generate_comparison_sprint23.py` → `results_by_condition[cond][ds][model][platform]` (acc+F1) ; notebook `_heatmap(metric,condition,platform)` → **12 PNG** `gap1_heatmap_{metric}_{condition}_{platform}.png` ; **complétion 120/120 cellules (0 pending PC+board)** après correctifs Paderborn/TinyOL + sweep PC 60/60 + 7 reflashs board |
| S3511 présentation | ✅ | — | `presentation_plots.ipynb` : Slide 6 + `best` PC (acc+F1) ; nouvelle **Slide 6bis** panel board F1+acc × {5feat,all,best} (helper `show()`, 0 dup) ; nbconvert OK ; `01_structure.md`/`02_script.md` (message F1>accuracy, footnote board, fix HDC) |
| S3512 analyse | ✅ | — | Section « Analyse » rédigée dans `S3512_analysis_update.md` (impact features×modèle, F1 vs acc, coût board RAM/latence Gap 2, fix HDC 0.113→0.867, reco chiffrée) + § Gap 1 dans `triple_gap.md` ; chiffres ← `comparison_sprint23.json` |
| S3513 tests | ✅ | — | `test_feature_selection.py` +1 test déterminisme `permutation_importance` ; `test_heatmap_builders.py` (4 tests : structure 3×5×4×2, pending→None/NaN masqué, matrice 5×4, fix S3509≠0.113) ; **25 PASS** ; Unity `make test` **103/105** (2 TinyOL préexistants, 0 régression) |
| S3514 docs + graphify | ✅ | — | Statut Sprint 35 dans `CLAUDE.md` ; `roadmap_phase2.md` Sprint 35 → ✅ ; bilan complété ; `graphify_sprint_update` invoqué |
