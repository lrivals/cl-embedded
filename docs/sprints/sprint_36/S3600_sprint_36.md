# Sprint 36 — Comparaison précise PC ↔ board (EWC sur Pronostia + Monitoring)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 36 |
| **Semaine** | 11 – 17 août 2026 |
| **Statut** | ✅ Implémenté (S3601–S3609, board réelle NUCLEO-F439ZI) |
| **Priorité globale** | 🔴 Critique — Produire une **comparaison appariée et honnête PC ↔ NUCLEO-F439ZI** du modèle **EWC** sur **Pronostia (D4)** et **Monitoring (D2)**, dans des **conditions identiques** (mêmes données train/inférence des deux côtés), avec **tous les métriques** (acc/tâche, oubli, acc finale, F1, ROC-AUC, RAM, latence inférence **et** latence inférence+MAJ CL), une **comparaison prédiction-par-prédiction**, et un **notebook** rassemblant tous les plots. |
| **Durée estimée totale** | ~32h |
| **Dépendances** | `src/evaluation/feature_conditions.py` ✅ (`resolve_feature_indices`/`load_condition_arrays`, source unique board↔PC, Sprint 35) · `src/evaluation/metrics.py` ✅ (`compute_cl_metrics`, `compute_fault_f1`) · `src/evaluation/online_metrics.py` ✅ (acc/AUROC/forgetting) · `scripts/sensor_stream.py` ✅ (`--condition`/`--update`/`--dump-samples`/`--proto`) · `scripts/run_feature_condition_board.py` ✅ (driver train→export→build→flash→stream→parité) · `scripts/train_board_reference.py` ✅ · `scripts/export_weights_c.py --ewc-head` ✅ · `scripts/train_ewc.py` + `src/models/ewc/ewc_mlp.py` ✅ · firmware EWC + latence DWT + réponses v2/v3 (`firmware/stm32f4_blink/src/pipeline.c`, `inc/pipeline.h`) ✅ · `src/evaluation/plots.py` ✅ |

---

## Contexte et motivation

Les sprints précédents ont établi la **parité board↔PC** sur de nombreux axes (seuil RUL
Sprint 32, nombre de features Sprint 35), mais toujours de façon **transversale** (balayages
larges multi-modèles/multi-datasets). Il manque une **étude focalisée et appariée** qui
réponde précisément à la question : *« pour un même modèle et les mêmes données, les
chiffres de précision PC et board coïncident-ils, et que coûtent-ils en RAM/latence ? »*.

EWC est le modèle idéal pour cette étude :

- C'est (avec Mahalanobis) le **seul modèle à parité exacte board↔PC** (poids exportés via
  `export_weights_c.py`, forward C identique au PyTorch — HDC/TinyOL restent HW-only par
  construction, cf. Sprint 32). On peut donc **comparer les prédictions une à une**.
- Il porte l'enjeu scientifique central du projet : **oubli catastrophique** (cf. Sprint 26 —
  le « PC 0.981 » trompeur était la moyenne post-tâche ; F1 final tous-tâches = 0.240,
  `avg_forgetting_f1=0.847`). L'étude doit donc rapporter l'**oubli**, pas seulement l'acc finale.

Deux scénarios CL **contrastés** sont retenus :

- **Monitoring (D2)** — tabulaire, **domain-incremental par type d'équipement** (pump → turbine → compressor), ~5 features natives.
- **Pronostia (D4)** — accéléromètre + température, **class-incremental par condition**, 13 features natives.

### Tension parité ↔ MAJ CL (point de vigilance)

La **parité exacte** des prédictions n'existe **que poids gelés** (sans `--update`). Dès que
l'on active la MAJ CL en ligne (`--update`), PC et board peuvent **diverger** (ordre des
échantillons, float32 firmware vs float64 PyTorch). C'est pourquoi le sprint sépare
**deux passes** :

```
Passe GELÉE (sans --update)      → parité EXACTE prédictions PC↔board + métriques de base
                                   + latence INFÉRENCE seule (DWT)

Passe ONLINE (--update 2 côtés)  → latence INFÉRENCE + MAJ CL (DWT)
                                   + métriques online (acc/AUROC/forgetting) + parité APPROCHÉE
```

Les **deux conditions de features** sont mesurées : `5feat` (board figé historique) **et**
`all` (dims natives — réutilise l'infra Sprint 35).

---

## Décisions validées (utilisateur)

- **Modèle** : **EWC uniquement** (Mahalanobis peut servir de référence de parité si utile, mais n'est pas l'objet).
- **Datasets** : **Pronostia + Monitoring**.
- **Deux protocoles** documentés comme sous-tâches distinctes : passe **gelée** (parité exacte) **et** passe **online** (latence inf+MAJ).
- **Deux conditions** : `5feat` **et** `all` (dims natives).
- **Échantillons appariés** : streamer **l'intégralité du split test/inférence** des deux côtés (pas de N tronqué).

---

## Rework Sprint 36 (S3610–S3613) — cadrage apparié + axe INT8 vs FP32 board

Trois ajouts demandés après la première implémentation :

1. **Cadrage des comparaisons appariées** (à condition fixe, jamais croisées) :
   - **Comparaison A — `all` : board vs PC** ;
   - **Comparaison B — `5feat` : board vs PC**.
   Le balayage transverse `5feat` vs `all` devient une **étude secondaire** explicite
   (« comportement des modèles sous plus de contraintes / moins de features »). C'était déjà
   le comportement du notebook (indexation `summary[ds][cond][platform]`, jamais de croisement
   `5feat`-board × `all`-PC) — le rework rend le cadrage explicite côté notebook + doc.
2. **Axe INT8 vs FP32 sur board** pour les 2 conditions × 2 datasets, en **frozen + online**
   (détail : `S3610_int8_fp32_board.md`).
3. **Éléments de doc** ajoutés (ce document, `S3610`, roadmap, triple_gap, CLAUDE.md).

### Décisions rework (utilisateur)

- INT8 : **frozen + online** (les deux passes).
- INT8/FP32 : **board-only** — la référence PC reste FP32 (pas de `EWCMlpInt8Classifier` PC).
- Comparaisons appariées à condition fixe ; `5feat` vs `all` = étude secondaire.

### Découverte résolue (`TODO(dorra)`)

Le firmware initialisait la tête INT8 (`ewc_int8_init`) **sans charger les poids FP32 entraînés**
(`TODO(dorra)` `pipeline.c:507`, `ewc_int8_from_fp32` commenté) → le chemin `0x40` exécutait une
tête Xavier non entraînée. **S3610 résout ce TODO** : `ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head)`
est appelé après `ewc_head_load_or_init` (Xavier en fallback → **0 régression FP32**, chemin
FP32 inchangé). Le reste de la chaîne INT8 EWC existait déjà : `sensor_stream.py --model
ewc-int8` (flag `FRAME_FLAGS_INT8_MODE=0x40`) routé par `pipeline.c` vers
`ewc_int8_forward/update/consolidate` — **pas de nouveau flag protocole, aucune collision**.

---

## Contraintes techniques à honorer

- **Source unique des features** : board et PC consomment exactement les colonnes de
  `resolve_feature_indices(condition, "ewc", dataset)` (parité par construction, Sprint 35) — **ne pas dupliquer la sélection**.
- **Pas de hardcode** (règle CLAUDE.md) : dims EWC via `#define` au build (`-DEWC_IN=k`, `PROTO_MAX_N` si `k>16`) ; aucun hyperparamètre dans le source → `configs/sprint36_ewc_comparison.yaml`.
- **`model_weights*.h` jamais à la main** : régénérés via `export_weights_c.py --ewc-head`.
- **Protocole UART intouché** côté sémantique : F1/ROC-AUC dérivés **côté hôte** depuis le flux (cf. S3504). Toute évolution UART ⇒ synchroniser `sensor_stream.py` (règle CLAUDE.md).
- **Aucun chiffre board inventé** : tous les champs résultats portent **« à mesurer »** tant que la NUCLEO n'a pas tourné.
- **Gap 2** : latences (inférence ; inférence+MAJ) à confirmer **≪ 100 ms**.

---

## Critères de succès

1. `configs/sprint36_ewc_comparison.yaml` produit (datasets, conditions, protocoles, seed, split test complet, débit UART) — **aucun paramètre dans le code**.
2. Runs **PC** : `experiments/exp_S36_PC_{condition}_ewc_{dataset}/results.json` avec acc_matrix, AA/AF/BWT, acc_final, f1_faulty/macro, ROC-AUC, n_params, ram_peak_bytes, latence PC, + dump des prédictions par échantillon.
3. Runs **board gelés** : `experiments/exp_S36_board_frozen_{condition}_ewc_{dataset}/` avec online_accuracy, F1, ROC-AUC, **latence inférence** P50/P99, `.bss`, `parity_class=exact`, `parity_rate`.
4. Runs **board online** : `experiments/exp_S36_board_online_{condition}_ewc_{dataset}/` avec **latence inférence+MAJ CL** P50/P99 + métriques online + parité approchée documentée.
5. **Comparaison prédiction-par-prédiction** : `experiments/exp_S36_parity_{condition}_{protocol}_{dataset}.json` (table par échantillon, taux de concordance, désaccords listés).
6. **Tous les métriques** agrégés dans `experiments/exp_S36_summary.json`.
7. **Notebook** `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` exécuté (nbconvert), figures → `docs/figures/sprint36_pc_board_ewc/`.
8. **Gap 2** vérifié : toutes latences (inférence ; inférence+MAJ) ≪ 100 ms.
9. Tests `pytest tests/test_sprint36_comparison.py` verts ; Unity `make test` **0 régression** (EWC inchangé).

---

## Tâches

### Partie A — Données & configuration appariée

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3601 | `configs/sprint36_ewc_comparison.yaml` : datasets `[pronostia, monitoring]`, modèle `ewc`, conditions `[5feat, all]`, protocoles `[frozen, online]`, `seed: 42`, split test complet, débit UART (`rate_hz`, `proto: 3`). | 🔴 | `configs/sprint36_ewc_comparison.yaml` | 2h |

### Partie B — Runs PC (référence)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3602 | Train + éval PC EWC via `train_ewc.py` consommant `load_condition_arrays(...)` → `exp_S36_PC_{condition}_ewc_{dataset}/results.json` (acc_matrix, AA/AF/BWT, acc_final, F1, ROC-AUC, n_params, RAM, latence PC) + **dump prédictions par échantillon** pour la parité. | 🔴 | `scripts/train_ewc.py`, `experiments/exp_S36_PC_*` | 5h |

### Partie C — Runs board, passe GELÉE (parité exacte + latence inférence)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3603 | Réutiliser le pattern `run_feature_condition_board.py` (train réf `train_board_reference.py` → `export_weights_c.py --ewc-head` → build/flash `-DEWC_IN=k` (+`PROTO_MAX_N` si k>16) → `sensor_stream.py --condition ... --proto 3` **sans `--update`** → parité) → `exp_S36_board_frozen_{condition}_ewc_{dataset}/`. | 🔴 | `scripts/run_feature_condition_board.py` (réutilisé), `experiments/exp_S36_board_frozen_*` | 5h |

### Partie D — Runs board, passe ONLINE (latence inférence + MAJ CL)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3604 | Même build, `sensor_stream.py ... --update` (+ `--consolidate-on-task-change`) ⇒ latence **inférence+MAJ CL** (DWT) ; PC exécute la **même séquence online** ; métriques online (acc/AUROC/forgetting) ; parité approchée documentée → `exp_S36_board_online_{condition}_ewc_{dataset}/`. | 🔴 | `scripts/sensor_stream.py` (réutilisé), `experiments/exp_S36_board_online_*` | 5h |

### Partie E — Comparaison prédiction-par-prédiction

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3605 | `--dump-samples` des deux côtés ; alignement (réutilise logique `_parity()` de `run_feature_condition_board.py`) → `exp_S36_parity_{condition}_{protocol}_{dataset}.json` : table `[idx, true, pred_pc, pred_board, conf_pc, conf_board, match]`, taux de concordance, désaccords. | 🔴 | `scripts/board_pc_parity.py` (ou extension du driver), `experiments/exp_S36_parity_*` | 4h |

### Partie F — Agrégation & notebook

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3606 | Agréger PC/board × conditions × protocoles × datasets → `experiments/exp_S36_summary.json` (acc/tâche, AF, acc_final, F1, ROC-AUC, RAM PC/`.bss` board, latences inférence + inf+MAJ, parité). | 🟡 | `scripts/aggregate_sprint36.py`, `experiments/exp_S36_summary.json` | 3h |
| S3607 | Notebook `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` (tous les plots, cf. liste ci-dessous), figures → `docs/figures/sprint36_pc_board_ewc/`. | 🔴 | `notebooks/cl_eval/pc_board_ewc/comparison.ipynb`, `docs/figures/sprint36_pc_board_ewc/*` | 5h |

### Partie G — Tests & docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3608 | `tests/test_sprint36_comparison.py` (chargement summary, structure parité, cohérence métriques) + note Unity `make test` (0 régression, EWC inchangé). | 🟢 | `tests/test_sprint36_comparison.py` | 2h |
| S3609 | MAJ `S3600` (bilan), `docs/roadmap_phase2.md` (entrée Sprint 36), `docs/triple_gap.md` (§ Gap 2 latences mesurées) + `CLAUDE.md` (statut) ; invoquer `graphify_sprint_update`. | 🟢 | `docs/roadmap_phase2.md`, `docs/triple_gap.md`, `CLAUDE.md` | 1h |

### Partie H — Rework : cadrage apparié + axe INT8 vs FP32 board (S3610–S3613)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3610 | Firmware : résoudre `TODO(dorra)` → `ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head)` après `ewc_head_load_or_init` (chemin FP32 inchangé, 0 régression) + driver `--precision {fp32,int8}` dans `run_sprint36_board.py` (build identique, flag UART `FRAME_FLAGS_INT8_MODE` 0x40). | 🔴 | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/run_sprint36_board.py` | 4h |
| S3611 | Runs board INT8 **frozen + online** × 2 conditions × 2 datasets → `exp_S36_board_{frozen,online}_int8_*` : latence DWT, RAM poids INT8/FP32 (ratio ≈ 4×), métrique préservée, **accord INT8↔FP32 board**. Champs board = `"à mesurer"`/`null` tant que NUCLEO non branchée. | 🔴 | `scripts/run_sprint36_board.py`, `experiments/exp_S36_board_*_int8_*` | 4h |
| S3612 | Agrégation : clés additives `board_frozen_int8`/`board_online_int8` (ratios latence/RAM, `delta_metric_int8_vs_fp32`, `gap3_ram_ok`, `agreement_int8_vs_fp32`) + notebook : sections appariées A/B explicites, §10 relabel « étude secondaire », §12 plots INT8 vs FP32. | 🟡 | `scripts/aggregate_sprint36.py`, `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` | 4h |
| S3613 | Tests : `test_sprint36_comparison.py` (clés INT8, ratios, Gap 2/Gap 3) + Unity `make test` 0 régression FP32 ; docs : `S3610_int8_fp32_board.md`, roadmap/triple_gap §Gap 3/CLAUDE.md ; `graphify_sprint_update`. | 🟢 | `tests/test_sprint36_comparison.py`, `docs/sprints/sprint_36/S3610_int8_fp32_board.md` | 2h |

---

## Plots attendus (notebook S3607)

Inspirés de `notebooks/cl_eval/threshold_impact/comparison.ipynb` (gabarit PC↔board) et de `src/evaluation/plots.py`.

1. **Acc par tâche PC vs board** — barres groupées, par dataset.
2. **Matrices d'accuracy CL** — heatmap (`plot_accuracy_matrix`), PC et board côte à côte.
3. **Acc finale vs Oubli (AF)** — scatter PC vs board.
4. **Courbes d'oubli par tâche** — `plot_forgetting_curve`.
5. **Latence : inférence vs inférence+MAJ CL** — barres/scatter board, échelle log, ligne Gap 2 (100 ms).
6. **Latence PC vs board** — comparaison.
7. **Acc vs RAM** — scatter, zone budget STM32, PC (`ram_peak_bytes`) vs board (`.bss`).
8. **F1 et ROC-AUC PC vs board** — barres groupées.
9. **Parité prédictions** — table concordance + diagonale `pred_pc` vs `pred_board` + matrice de confusion des désaccords.
10. **Effet condition `5feat` vs `all`** — sur perf / parité / latence.
11. **Tableau récapitulatif final** — tous les métriques.

---

## Ordre d'exécution recommandé

```
Partie A     Partie B          Partie C            Partie D            Partie E       Partie F            Partie G
S3601 ─▶ S3602 (PC) ─┬─▶ S3603 (board gelé) ─┬─▶ S3605 (parité) ─▶ S3606 ─▶ S3607     S3608
                     └─▶ S3604 (board online)┘                                         S3609
```

---

## Nomenclature des expériences

| Exp ID | Sujet | Mesure |
|--------|-------|--------|
| `exp_S36_PC_{condition}_ewc_{dataset}/results.json` | référence PC | acc_matrix, AA/AF/BWT, acc_final, F1, ROC-AUC, RAM, latence PC, prédictions/échantillon |
| `exp_S36_board_frozen_{condition}_ewc_{dataset}/` | board poids gelés | online_accuracy, F1, ROC-AUC, latence **inférence** P50/P99, `.bss`, parité exacte |
| `exp_S36_board_online_{condition}_ewc_{dataset}/` | board MAJ CL | latence **inférence+MAJ** P50/P99, métriques online, parité approchée |
| `exp_S36_parity_{condition}_{protocol}_{dataset}.json` | comparaison prédiction-par-prédiction | table par échantillon, taux concordance, désaccords |
| `exp_S36_summary.json` | agrégat | tous métriques × plateformes × conditions × protocoles |

`condition ∈ {5feat, all}` · `protocol ∈ {frozen, online}` · `dataset ∈ {pronostia, monitoring}` · modèle = `ewc`

---

## Questions ouvertes

- `TODO(arnaud)` : cohérence de la métrique de référence — moyenne post-tâche (trompeuse, cf. Sprint 26) vs métrique du modèle final tous-tâches. Le sprint rapporte les deux et privilégie le modèle final.
- `TODO(arnaud)` : en passe **online**, faut-il imposer un ordre d'échantillons strictement identique PC↔board (déterminisme) pour rendre la parité online interprétable ?
- `FIXME(gap2)` : confirmer latence board < 100 ms pour `all` Pronostia (13 features → forward EWC plus coûteux) **et** pour la passe online (inférence+MAJ).
- `TODO(dorra)` : la passe online divergeant intrinsèquement (float32 vs float64), quel seuil de « taux de concordance acceptable » retenir comme critère ?

---

## Livrables

1. `configs/sprint36_ewc_comparison.yaml`
2. `experiments/exp_S36_PC_*` (référence PC, prédictions dumpées)
3. `experiments/exp_S36_board_frozen_*` (parité exacte + latence inférence)
4. `experiments/exp_S36_board_online_*` (latence inférence+MAJ + métriques online)
5. `experiments/exp_S36_parity_*` (comparaison prédiction-par-prédiction)
6. `experiments/exp_S36_summary.json`
7. `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` + figures `docs/figures/sprint36_pc_board_ewc/`
8. `tests/test_sprint36_comparison.py` + Unity 0 régression
9. MAJ `roadmap_phase2.md` / `triple_gap.md` / `CLAUDE.md` + `graphify_sprint_update`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3601 config appariée | ✅ | — | `sprint36_ewc_comparison.yaml` ; Monitoring `5feat ≡ all` (4 feat. natives) |
| S3602 runs PC | ✅ | — | 4 `results.json` + dump `samples` ; AA/AF/BWT/F1/ROC-AUC ; AF faible (CL court 3 tâches) |
| S3603 board gelé (parité) | ✅ | — | **parité 1.000** ; lat inf 48–65 µs ; `.bss` 100–145 Ko |
| S3604 board online (latence inf+MAJ) | ✅ | — | lat inf+MAJ 239–340 µs (Δ +191…+275 µs) ; parité~ 0.963–0.989 |
| S3605 parité prédictions | ✅ | — | 8 fichiers ; frozen 1.000 (hors-ligne), online **re-streamé board** (`board_samples.json`) |
| S3606 agrégation | ✅ | — | `exp_S36_summary.json` ; Δacc PC↔board ≤ 0.007 |
| S3607 notebook | ✅ | — | `comparison.ipynb` nbconvert OK → 10 PNG |
| S3608 tests | ✅ | — | `test_sprint36_comparison.py` 6/6 ; Unity 0 régression (2 TinyOL préexistants) |
| S3609 docs + graphify | ✅ | — | roadmap/triple_gap/CLAUDE.md MAJ + graphify_sprint_update |
| S3610 firmware INT8 + driver `--precision` | ✅ | — | `ewc_int8_from_fp32` après `ewc_head_load_or_init` (TODO(dorra) résolu) ; Unity `make test` **116 tests, 2 échecs TinyOL préexistants, 0 régression** ; `--precision {fp32,int8}` ajouté |
| S3611 runs board INT8 frozen+online | ✅ | — | **8 cellules mesurées board réelle NUCLEO-F439ZI, 0 CRC** : Gap 2 ✅ (frozen 51–68 µs, online 440–639 µs ≪ 100 ms), Gap 3 RAM ✅ (×4.0). **MAIS métrique NON préservée** : F1 INT8 **0.07–0.15** vs FP32 ≈ 0.92 ; accord INT8↔FP32 0.60–0.74 (frozen) / 0.85–0.88 (online) → forte dégradation de la quantif post-training firmware, **cohérente Sprint 29** (board INT8 EWC AUROC 0.25 vs 0.63), distincte du fake-quant QAT PC (Sprint 28, Δ≤0.006) |
| S3612 agrégation + notebook INT8 | ✅ | — | clés `board_{frozen,online}_int8` additives (summary rétro-compatible) ; notebook 28 cellules, §A/B appariées + §10 « étude secondaire » + §12 INT8/FP32 ; nbconvert OK → `int8_vs_fp32_board.png` |
| S3613 tests + docs | ✅ | — | `test_sprint36_comparison.py` étendu ; `S3610_int8_fp32_board.md` ; roadmap/triple_gap §Gap 3/CLAUDE.md |
