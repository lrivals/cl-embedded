# CLAUDE.md — Contexte du projet pour Claude Code

> Ce fichier est la source de vérité pour Claude Code dans ce dépôt.
> Lire entièrement avant toute intervention dans le code.

**Statut sprint (27 juin 2026)** : Sprint 26 ✅ implémenté — Board NUCLEO-F439ZI réel : RMSE_RUL=21.23 (ratio=0.94 ✅) · lat séparées 130µs (inférence) / 403µs (inférence+update) ✅ · **F1_MC=0.507 online / 0.243 inférence < 0.60 ❌** — **`FIXME(gap1)` RÉSOLU** : pas un bug de portage, parité numérique board↔PC exacte (0.243==0.243, 0.507==0.507) ; cause = **oubli catastrophique** du modèle EWC (le « PC 0.981 » était la moyenne des F1 post-tâche, trompeuse ; F1 modèle final tous-tâches = 0.240 ; avg_forgetting_f1=0.847). Amélioration modèle CL (λ/replay) hors périmètre portage. Diagnostic : `scripts/diagnose_multiclass_parity.py` · 75 tests dont 10/10 EWC reg+mc PASS + 2 échecs TinyOL pré-existants hors périmètre | Sprint 27 ✅ implémenté — **DUAL_MODE** (1 trame UART → EWC_REG + EWC_MC en séquence → réponse 25 B) validé board réelle : co-exécution OK, **latence combinée 637 µs << 100 ms (Gap 2 ✅)**, `.bss=66 748 B` (25.5 %), **RMSE_RUL=22.59 préservé** (≈ single-mode 21.15) car `g_ewc_reg` lit `raw[0:4]` CMAPSS purs, mais **F1_faute=0.043 ❌** — `FIXME(gap1)` : features mixtes (RUL 5 CMAPSS + faute 9 CWRU = 14 features pour 9 slots) → `g_ewc_mc` reçoit 5/9 slots hors-domaine ; limitation de construction, **pas un bug de portage** (parité board↔PC OK). Dataset unifié Pronostia → Sprint 28. 79 tests (4 dual PASS, 2 TinyOL préexistants). exp_S27_01 produit. | Sprint 28 🟡 S2807/S2808 ✅ implémentés — **benchmark INT8 vs FP32 PC, tableau 4×5 produit** (20 JSON : `experiments/exp_S28_PC_ewc_hdc/` + `exp_S28_PC_tinyol_maha/`). **Gap 3 RAM ✅ 18/18 cellules** (2.33× HDC int16-AM → 4.00× EWC/Maha int8) · **métrique préservée 12/16** : EWC (Δ≤0.006) + HDC (Δ=0) ✅ ; 2 TinyOL (CMAPSS/CWRU) `|Δ|>0.02` mais **amélioration** (fake-quant régularisante) ; 2 Mahalanobis (CWRU −0.236, Pronostia −0.238) dégradés → quantif INT8 `sigma_inv_` grande dynamique, **fallback Q15 recommandé** (confirme `TODO(arnaud)` S2805). Paderborn AUROC=N/A (tâches test mono-classe) ; HDC×Paderborn N/A (feature_bounds non calibrés). EWC binaire → AUROC détection de panne sur labels binarisés normal-vs-fault. 6/6 tests `test_int8_benchmark.py` PASS. Reste O6 (heatmaps S2810 + notebook S2811). | Sprint 29 ✅ O1–O8 implémentés — INT8 Firmware Board + Gap 3 multi-modèle. **O1–O7** : HDC/TinyOL INT8 firmware + pipeline + 5 couples board mesurés (`exp_S29_board_int8/`, RAM ×2.70–4.00, latence INT8 négative Cortex-M4 FPU, 0 CRC), notebook synthèse PC+board. **O8 ✅ (S2912–S2914, 28 juin 2026)** : extension board **5→20** (grille 4 modèles × 5 datasets comparable au PC). **S2912** portage **Mahalanobis INT8 firmware** (`mahalanobis_int8.c/.h` : mu+sigma_inv INT8 affine, déquant→distance FP32 FPU = parité Python ; **sélection par compilation `-DMAHA_INT8`** car nibble de flags protocole **saturé** 0x00–0xF0 ; `export_weights_c.py --maha-int8` + test vectors ; test Unity `test_mahalanobis_int8.c` **4/4 PASS**, `.bss` défaut 105 036 B invariant / +60 B sous `-DMAHA_INT8`, **0 régression** sur les 2 builds). **S2913** driver `run_s29_board_extend.py` (train→export→build→flash→stream par cellule ; `assemble_result` source unique ; N/A honnête `metric_value=null`+`na_reason`) → **15 cellules mesurées board réelle NUCLEO-F439ZI** : grille **20/20** = **18 streamées** (EWC 462 µs, HDC 2095 µs, TinyOL 71 µs, **Maha 6 µs** ; **0 erreur CRC** ; Gap 2 ✅ ≪ 100 ms) dont 2 métrique N/A mono-classe (hdc/maha × cwru) + **2 non mesurables** (encodeur TinyOL non exportable cmapss/paderborn). **Maha INT8 reproduit la dégradation connue** (`sigma_inv_` grande dynamique → Q15 = fallback Sprint 34). **S2914** notebook `sprint29_int8_board.ipynb` Section 2 généralisée 20 cellules + **heatmaps board 4×5 (2.5)** symétriques au PC (N/A en gris) + Section 4 mise à jour (Maha porté, ≠ « PC-only ») ; `board_gap3_heatmaps.png` ; nbconvert OK. Tests : `pytest -k "int8 or mahalanobis"` **47 PASS** + Unity `make test` (Maha INT8 +4, **2 TinyOL préexistants** hors périmètre, 0 régression). Docs : `S2912/S2913/S2914`. | Sprint 30 ✅ implémenté — **Paires de modèles parallèles** (Mahalanobis + supervisé : 3 paires M&HDC/M&EWC/M&TinyOL × 5 datasets). PC : 14/15 `exp_S30_PC_*` binarisés (indiv + ensemble 4 règles + désaccord rate/κ/origine), 1 N/A honnête (maha_hdc×paderborn, feature_bounds non calibrés) + Partie B native (RUL CMAPSS, CWRU multi-classe). **Portage board réelle (S3009/S3010)** : DUAL_MODE généralisé → **PAIR_MODE** `0x90/0xA0/0xB0` (nibble de mode libre, **aucune collision** vérifiée), réponse **22 B** `[pred_maha][score_maha][pred_sup][conf_sup][lat][auroc_maha][f1_sup]`, `sensor_stream.py` synchronisé. NUCLEO-F439ZI : maha_ewc **256 µs** combiné (5 Maha + 251 EWC), maha_hdc **651 µs** (5 + 647), **overhead ~0** (co-exécution séquentielle), **Gap 2 ✅ << 100 ms**, `.bss=104 576 B` (39.9 % de 256 Ko). 92 tests firmware (**+3 PAIR T80–T82 PASS**, 2 TinyOL préexistants) + 19 tests Python (model_pair/disagreement) PASS ; notebook `sprint30_pairs_disagreement.ipynb` (origines : score Maha plus élevé sur désaccords). `board_pair_recorder.py` produit `exp_S30_board_maha_{ewc,hdc}`. | Sprint 31 ✅ implémenté — **Méta-modèle de stacking** (logreg/MLP arbitrant les 2 sorties d'une paire Maha+supervisé). PC ✅ (S3101–S3104 : `src/ensemble/meta_learner.py`, features `[p_maha,p_sup,disagreement,conf_sup]` ∈ [0,1], méta ≥ ensemble 12/14 runs). **Portage board réelle (S3105–S3107)** : `meta_head.c/.h` (logreg + variante MLP) + `export_weights_c.py --meta` → `meta_weights.h` (généré, jamais à la main) ; `pipeline.c` PAIR généralisé → **TRIPLE_MODE** `0xD0` (maha+EWC+méta) / `0xE0` (maha+HDC+méta), **aucune collision** (nibble `0xF0`, `0xC0`=TINYOL_INT8), réponse **27 B** `[…PAIR 22 B…][pred_meta:u8][prob_meta:f32]` (slot `conf_sup`=`p_sup` pour reconstruction parité), `sensor_stream.py` synchronisé. NUCLEO-F439ZI : maha-ewc **258 µs** / maha-hdc **593 µs** combiné (méta logreg 4 features ~négligeable, latence dominée par le supervisé), **Gap 2 ✅ ≪ 100 ms**, `.bss=104 596 B` (39.9 %, +20 B `g_meta`). **Parité méta board↔PC = 1.000** sur 300 échantillons (verdict numpy reconstruit == board ; `board_pair_recorder.py --triple`, Δprob HDC=0.004 arrondi float32/64 sans impact verdict). 96 tests firmware (**+4 méta PASS**, 2 TinyOL préexistants hors périmètre) + 12 `test_meta_learner.py`. `exp_S31_board_maha_{ewc,hdc}` produits. | Sprint 32 ✅ implémenté — **Étude d'impact du seuil RUL→`faulty`** (réponse `TODO(arnaud)` `cmapss_config.yaml:50`). PC ✅ (S3201–S3204 : 3 loaders ← seuil config, 15 configs `configs/sweep/`, **60 runs** `exp_S32_*`, `positive_ratio` monotone). **Board réelle (S3205)** : découverte clé — firmware câblé **5 features** vs dims natives PC (CMAPSS=5, Battery=7, Pronostia=13) ; firmware agnostique au dataset à 5-feat → **modèles de référence board 5-feat** (`scripts/train_board_reference.py`, EWCMlpMulticlass 5→32→16→2 == `ewc_forward`). **Parité board↔PC exacte pour EWC + Mahalanobis** (export poids → header) ; **HDC** (projection embarquée, dim 1000≠1024, init en ligne) + **TinyOL** (archi board distincte, pas de checkpoint) = **HW-only, parité N/A par construction** (décision utilisateur). Firmware : `inc/model_weights_ewc.h` (vide par défaut → Xavier ; régénéré par export) + `pipeline.c` `ewc_head_load_or_init()` charge `g_ewc_head` si `EWC_HEAD_WEIGHTS_PROVIDED` (**fallback historique, 0 régression**) ; `export_weights_c.py --ewc-head` ; `sensor_stream.py --dump-samples` (features+pred/échantillon) + dataset **battery** (`sensor_sim._load_battery`, `configs/battery_feature_subset.yaml`) ; driver `run_board_threshold_sweep.py` (train→export→build→**1 flash/cellule**→stream 4 modèles→parité, sans `--update`, `--rate-hz 50 --proto 3`). NUCLEO-F439ZI : **`.bss=104 596 B` invariant au seuil** ; latences P50 Maha ≈ 5 µs / EWC ≈ 50 µs / TinyOL ≈ 5 µs / HDC ≈ 585 µs — **toutes ≪ 100 ms (Gap 2 ✅)** ; **parité CMAPSS 10/10** (EWC+Maha × 5 seuils), Pronostia/Battery → `exp_S32_board_sweep_summary.json`. Notebook `notebooks/cl_eval/threshold_impact/comparison.ipynb` (perf/HW vs seuil, heatmaps, invariance HW board, tables/parité PC↔board). `tests/test_threshold_sweep.py` **16/16 PASS** + Unity firmware 94/96 (2 TinyOL préexistants hors périmètre). | Sprint 33 ✅ implémenté — **Profilage énergétique & métriques de coût** (CR 19 mai / 9 juin 2026). O1–O3 préexistants (S3301–S3304 : `compute_cost.py` +FLOPs/**BOPs**/Params, `hw_cost_model.py` T-HW/FLOPS-W + `configs/hw_profile_f439zi.yaml`, `measure_macs.py`, marqueurs GPIO **PA8** `ENERGY_MARKERS`). **O4–O7 ajoutés** : `scripts/energy_capture.py` (driver PowerShield X-NUCLEO-LPM01A — `segment_by_phase`/`integrate_energy_uj` µJ + CLI `--csv`/`--campaign`) → `experiments/exp_S33_energy/` (8 JSON `{ewc,hdc,tinyol,maha}_{fp32,int8}` + `summary.json` delta_uj/ratio) ; `src/evaluation/autonomy.py` (I_moy=Σ(I·t)/T_cycle, `Autonomie_h`=Capacité/I_moy, capacités ← `hw_profile_f439zi.yaml:batterie`) + RAM profiling `profile_memory.py --model autonomy` (peak **208 B**) → `autonomy.json` ; notebook `notebooks/cl_eval/energy_cost/comparison.ipynb` exécuté nbconvert. **Décision clé** : NUCLEO branchée mais **PowerShield LPM01A non posé** → règle « aucun chiffre inventé » : tous les champs énergie/autonomie portent la valeur littérale **`"à mesurer"`** (le code est fonctionnel, prêt à re-remplir depuis un CSV LPM01A réel). **Coût calculé réellement** : `BOPs_fp32/BOPs_int8 = (32/8)² = 16` (gain INT8 quantitatif honnête), temps-HW proxy, throughput. Lien Gap 3 : l'INT8 réduit la RAM sans accélérer la latence FPU (Sprint 29) → question énergie ouverte (µJ réels requis), réponse différée. Tests : `test_hw_cost_model.py` (7) + `test_autonomy.py` (8) + `test_compute_cost.py` (16) = **31 PASS** ; Unity firmware 94/96 (2 TinyOL préexistants, 0 régression). `TODO(dorra)` fréq. échantillonnage/calibration LPM01A. **Complétion (non-matérielle, sans LPM01A)** : la chaîne de segmentation depuis un CSV LPM01A réel est **débloquée et testée bout-en-bout** (auparavant `_capture_one` levait `NotImplementedError` → jamais exercée) — les **fenêtres de phase sont déduites de la colonne de sync PA8** du CSV (`_load_csv` reconnaît `sync`/`pa8`/`gpio` ; `derive_phase_windows` convertit les fronts ; **limitation 1-bit assumée** : startup/acquisition/inference partagent le niveau haut → reportés `inference`, bas → `idle` ; granularité 4-phases = encodage multi-bit firmware futur). **Aucun chiffre écrit** dans `experiments/exp_S33_energy/` tant que la sonde n'a pas tourné (`--campaign` reste `"à mesurer"`). Nouveau `tests/test_energy_capture.py` **16 PASS** → **suite énergie 47 PASS**. | Sprint 34 ✅ implémenté — **Streaming/buffer (Partie A préexistante S3401–S3404) + Q15 Mahalanobis (Partie B S3405–S3409)**, réponse au `TODO(arnaud)` (dégradation INT8 `sigma_inv_` grande dynamique Sprint 28). **S3405** : `MahalanobisDetectorInt8` `quant ∈ {int8,q15}` (config `quantization:`) ; `q15` = `sigma_inv_` **int16 Q15** (scale par-tenseur `max|·|/32767`), `mu_` reste INT8 affine ; `calibrate_q15`/`anomaly_score_q15`/`predict_q15`/`get_memory_footprint_q15` (Σ⁻¹ d²×2 B = ÷2 vs FP32) ; **mode `int8` défaut strictement inchangé**. **S3406** : `run_s34_maha_q15.py` + 5 configs `mahalanobis_q15_*` → `exp_S34_maha_q15/`. **Métrique de recouvrement = corrélation de rang au FP32** (pilote seuil/AUROC) : **Q15 > INT8 sur les 5 datasets** (Pronostia 0.985 vs 0.649, Paderborn 0.921 vs 0.827, CWRU 0.536 vs 0.409) ; **AUROC recouvrée** sur ds non-dégénérés : **Pronostia ΔAUROC −0.113 (INT8) → +0.013 (Q15)** ✅ <0.02, CMAPSS +0.005, Monitoring ≈0 ; CWRU AUROC FP32=0.475 (sub-random, binarisation dégénérée) → AUROC non pertinente. **Nuance honnête** : sur très grande dynamique (Paderborn Σ⁻¹ ~6e5) l'erreur ABSOLUE de score Q15 > INT8 — non par perte de fidélité (Q15 reconstruit Σ⁻¹ **200× mieux**) mais parce que `mu_` reste INT8 et que son erreur est amplifiée par les grandes valeurs de Σ⁻¹ que Q15 préserve (INT8 les écrase → distances collapsées) ; piste future `mu_` Q15. **S3407** : firmware `mahalanobis_q15.c/.h` (`mu_q8` **uint8** affine + `sigma_inv_q15` int16, **déquant→distance FP32 sur FPU** = parité bit-à-bit Python, pas d'accumulation int32) ; FLAG `PROTO_FLAG_MAHA_Q15 = 0xF0` (**seul nibble libre** ; 0xD0/0xE0 pris par TRIPLE S31 — la doc citait 0xD0/0xE0/0xF0, corrigé), early-return par nibble **avant** la chaîne de bits (`0xF0 & 0x70 == 0x70` DUAL, `& 0x30` MULTICLASS) ; **réponse V3 (23 B) réutilisée** ; `sensor_stream.py --model maha-q15` + `FRAME_FLAGS_MAHA_Q15=0xF0` (synchro UART) ; `export_weights_c.py --maha-q15` → `inc/mahalanobis_q15_weights.h` (généré, `MAHA_Q15_WEIGHTS_PROVIDED`, vide par défaut → 0 régression) + `--maha-q15-test-vectors`. **S3408 board NUCLEO-F439ZI réelle** (`run_s34_board_maha_q15.py` train→export→build→flash→stream sans `--update`) : **parité board↔PC EXACTE** cmapss+pronostia (**300/300 prédictions**, `max_score_err` 9.6e-6 / 1.5e-3), **latence DWT P50=P99=5 µs ≪ 100 ms (Gap 2 ✅)**, **`.bss=105 036 B`** (53.7 %, +80 B `g_maha_q15`). **S3409** : `test_mahalanobis_q15.py` (7 PASS) + `test_mahalanobis_q15.c` (4 PASS, parité C↔Python) ; `make test` **109/111** (2 TinyOL préexistants hors périmètre, **0 régression**). | Sprint 35 ✅ implémenté (S3501–S3514) — **Étude d'impact du nombre de features** (`5feat`/`all`/`best-par-modèle`) × 5 datasets × 4 modèles, **PC + board ré-architecturé**. **S3507** : `export_weights_c.py` lit `EWC_IN=k` du checkpoint (+`#define EWC_HEAD_NATIVE_DIM`/`MAHA_NATIVE_DIM`) ; gardes `pipeline.c` passées de `WEIGHTS_NATIVE_DIM` (figé 5) à **dim native par modèle** (fallback → **0 régression 5feat**, `.bss=104 956 B`) ; résolveur `--condition/--model/--dataset`. **S3508** : source unique `resolve_feature_indices`/`load_condition_arrays` (`src/evaluation/feature_conditions.py`) ⇒ board et PC consomment les mêmes colonnes (**parité par construction**) ; `sensor_stream.py --condition` (sélection hôte, **0 changement UART**) ; driver `run_feature_condition_board.py` (train réf board Maha+`EWCMlpMulticlass(k)` → export → build/flash **1×/(cond,ds)** dims par modèle via `-D` (+`PROTO_MAX_N` si k>16) → stream 4 modèles **sans `--update`** → parité). **Board réelle NUCLEO-F439ZI** : 45 cellules (`all` 16 + `5feat` 16 + `best` 13 ; **paderborn initialement pending** = bug normalizer data-layer, **corrigé en complétion ↓**) ; **parité EWC+Maha exacte sur TOUTES dims k=1→21** (ex. cmapss `all` k=21 : EWC 79 µs / Maha 34 µs ; `best` k=1/2/4 ; **toutes ≪ 100 ms, Gap 2 ✅**) ; `best` per-modèle (fallback `all` documenté pour modèles sans config). **S3509** : artefact **HDC×monitoring 0.1133 corrigé** via condition `all` (4-feat natif, **sans zéro-padding**) → `online_accuracy≈0.87` (≈ PC 0.85) ; `generate_comparison_sprint23.py::_apply_s3509_override` (jamais à la main). **S3510** : `results_by_condition[cond][ds][model][platform]` (acc+F1) + notebook `_heatmap(metric,condition,platform)` → **12 PNG** `docs/figures/gap1_heatmap_{metric}_{condition}_{platform}.png` (pending masqué, 0 valeur en dur). PC sweep S3503 (`exp_S35_PC_*`) relancé. Tests : `pytest -k "feature_selection or f1 or threshold_sweep"` **34 PASS** ; Unity firmware 103/105 (2 TinyOL préexistants, **0 régression**). Best-features S3501 : 13 configs `configs/best_features/*.yaml` (paderborn KO bug normalizer). **Partie D/E finalisées** : **S3511** présentation — `presentation_plots.ipynb` Slide 6 (+ `best` PC acc+F1) + **Slide 6bis** panel board F1+acc × {5feat,all,best} (helper `show()`, 0 duplication ; nbconvert OK) + `01_structure.md`/`02_script.md` (message « accuracy trompeuse → F1 », footnote board, fix HDC) ; **S3512** analyse chiffrée `docs/sprints/sprint_35/S3512_analysis_update.md` (EWC×cmapss F1 board 0.38→0.62 en `5feat`→`all` ; accuracy trompeuse Maha×cmapss acc 0.745/F1 0.269 ; coût board `.bss` 104 956→183 936 B et latence ≤ 1.6 ms toutes conditions = **Gap 2 préservé** ; fix HDC×monitoring 0.113→0.867 ; reco : 5-feat par défaut, `best`/`all` ciblé cmapss+pronostia) + § Gap 1 `docs/triple_gap.md` ; **S3513** tests — `test_feature_selection.py` (+déterminisme `permutation_importance`) + nouveau `test_heatmap_builders.py` (structure 3×5×4×2, pending→None masqué, matrice 5×4, override S3509≠0.113) = **25 PASS** ; Unity `make test` **103/105** (2 TinyOL préexistants, **0 régression**) ; **S3514** statut CLAUDE.md + `roadmap_phase2.md` Sprint 35 → ✅ + bilan complété + `graphify_sprint_update`. **Complétion 12 heatmaps (120/120 cellules, 0 pending PC+board)** : 3 correctifs amont — (1) `paderborn_loader.py` refit du normalizer si le cache ne couvre pas les features natives (le cache top-5 cassait l'accès aux 7 features → `KeyError 'rms'`) ; (2) `feature_conditions._train_tinyol` `encoder_dims`/`decoder_dims` à **3 couches** (étaient 2 → `IndexError`, TinyOL PC ne tournait jamais) ; (3) `select_best_features_per_model.py --max-samples` (sous-échantillonnage **sélection seule**, sweep S3503 garde données complètes) pour rendre tractable CMAPSS (16k–49k éch./tâche). PC sweep **60/60 cellules 0 erreur** ; board **7 cellules reflashées** (paderborn ×3 + best ×{cwru,monitoring,pronostia,cmapss}), **parité EWC+Maha 30/30**, Gap 2 préservé (pire `best cmapss` HDC k=18 = 1 374 µs, `.bss` max 183 936 B = 70,2 %). Constats : **Paderborn = class-incremental mono-classe/tâche** → seul EWC tient (F1=0,80 PC+board), autres décrochent ; **HDC board F1=0 partout** = prédit la classe majoritaire (accuracy ok, F1_faulty nul → illustre « accuracy trompeuse → F1 »). 35 tests Python PASS (`test_feature_selection`+`test_heatmap_builders`). | Sprint 36 ✅ implémenté (S3601–S3609, board réelle NUCLEO-F439ZI) — **comparaison appariée et honnête PC ↔ board du modèle EWC** sur Pronostia (D4, class-incremental) + Monitoring (D2, domain-incremental), 2 conditions (`5feat`/`all`) × 2 protocoles (gelé/online). **S3601** `configs/sprint36_ewc_comparison.yaml` (Monitoring `5feat ≡ all` = 4 features natives, documenté). **S3602** réf PC `run_sprint36_pc.py` → `exp_S36_PC_*` (acc_matrix, AA/AF/BWT, F1, ROC-AUC, RAM, latence + dump `samples`) ; AA 0.98–0.99, AF≤0.01 (CL court 3 tâches, pas d'oubli catastrophique sur ces splits). **S3603 board gelé** `run_sprint36_board.py --pass frozen` : **parité EXACTE 1.000** × 4 cellules (checkpoint PC réutilisé → board flashé identique), lat inférence 48–65 µs (croît avec k), `.bss` 100–145 Ko. **S3604 board online** `--pass online` (`--update`+consolidate) : lat **inférence+MAJ CL 239–340 µs** (Δ MAJ +191…+275 µs, cohérent Sprint 26) **≪ 100 ms (Gap 2 ✅)** ; parité approchée 0.963–0.989 (float32 board ≠ float64 PC, miroir PC rejoue même séquence/ordre/seed). **S3605** `scripts/board_pc_parity.py` → **8 fichiers** `exp_S36_parity_{cond}_{proto}_{ds}.json` (table par échantillon `[idx,true,pred_pc,pred_board,conf,match]` + `mismatches`) : frozen reconstruit **hors-ligne** (exact 1.000, n=7534/7672) ; online **re-streamé sur la board** après ajout de la persistance `board_samples.json` dans `run_online` (additif rétro-compatible, **UART intact**) → parité~ 0.975/0.963/0.989, mismatch 87–282 concentrés sur les frontières de décision. **S3606** `aggregate_sprint36.py` → `exp_S36_summary.json` (lecture seule, indexé `[dataset][condition][platform]`, **Δacc_final PC↔board ≤ 0.007**). **S3607** `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` nbconvert → **10 PNG** `docs/figures/sprint36_pc_board_ewc/` (helpers `plots.py` réutilisés). **S3608** `test_sprint36_comparison.py` **6/6 PASS** (structure summary, parité frozen==1.0, forme table, Gap 2 latences<100 ms) ; Unity `make test` **112/114** (2 TinyOL préexistants, **0 régression** — EWC inchangé). § Gap 2 `triple_gap.md` enrichi (latences EWC inf vs inf+MAJ Pronostia/Monitoring). **Rework S3610–S3613** : (1) **cadrage apparié** — 2 comparaisons à condition fixe (A `all`:board↔PC, B `5feat`:board↔PC), balayage `5feat` vs `all` relabel **étude secondaire** (notebook §10). (2) **Axe INT8 vs FP32 board (frozen+online)** : firmware **résout `TODO(dorra)`** `ewc_int8_from_fp32(&g_ewc_int8,&g_ewc_head)` après `ewc_head_load_or_init` (le chemin 0x40 tournait sur Xavier non entraîné ; **0 régression FP32**, `make test` 116/2 TinyOL préexistants) ; `run_sprint36_board.py --precision {fp32,int8}` (build identique, flag UART `FRAME_FLAGS_INT8_MODE` 0x40, **pas de nouveau flag protocole**) → `exp_S36_board_{frozen,online}_int8_*` (RAM poids ÷4, latence/accord INT8↔FP32) ; `aggregate_sprint36.py` clés additives `board_*_int8` (`gap3_ram_ok`, `agreement_int8_vs_fp32`, `delta_metric_int8_vs_fp32`, summary rétro-compatible) ; notebook 28 cellules §12 INT8/FP32 (nbconvert OK → `int8_vs_fp32_board.png`). **8 cellules board réelle NUCLEO-F439ZI mesurées (0 CRC)** : **Gap 2 ✅** (frozen 51–68 µs ≈ FP32 ; online INT8 440–639 µs ~2× FP32 = MAJ non accélérée FPU, cohérent S29) + **Gap 3 RAM ✅** (×4.0) **MAIS métrique NON préservée ❌** — F1 INT8 **0.07–0.15 ≪ FP32 board ≈ 0.92**, accord INT8↔FP32 0.60–0.74 (frozen) / 0.85–0.88 (online) : forte dégradation de la **PTQ embarquée** de la tête EWC binaire, **cohérente Sprint 29** (board INT8 EWC AUROC 0.25 vs 0.63) et **distincte du fake-quant QAT PC** (Sprint 28, Δ≤0.006 préservé) ; `TODO(dorra)` résolu améliore vs Xavier (accord >0.5) mais la perte vient de la quantif, pas de l'init → piste QAT exporté / Q15. 9/9 tests `test_sprint36_comparison.py`. Doc `S3610_int8_fp32_board.md`. | Sprint 37 ✅ implémenté (S3701–S3709) — **Pipeline de publication GitLab (export sanitisé)** : transformation reproductible **« dépôt de travail → version GitLab »** pour le GitLab ISAE-SUPAERO, propre et professionnel (0 trace IA), couvrant **le code existant ET les ajouts futurs**. **Décisions utilisateur** : *dépôt exporté séparé* (dépôt git indépendant → aucun historique IA, donc aucun footer `Co-Authored-By`, n'atteint GitLab), *déclencheur local manuel* (`make gitlab-release` après validation feature), *exclure* `CLAUDE.md`/`skills/`/`graphify-out/`/`.claude/` + *docs neutres* générées. **Le dépôt de travail n'est JAMAIS poussé tel quel** — il y a toujours l'étape de transformation. **S3701** `configs/gitlab_release.yaml` (source de vérité : `exclude_paths`, `forbidden_patterns` à frontières de mot `\b…\b` → pas de faux positifs base64 notebooks, `rewrite_rules` replace→drop_blocks→drop_line en `fnmatch`, `allowlist`, `neutral_docs`, `release`). **S3702** `scripts/check_ai_traces.py` (scanner réutilisable, gate dur + scan manuel, exit 0/1, rapport `fichier:ligne[pattern]`, `--source` tolère zones internes connues). **S3703** `scripts/prepare_gitlab_release.py` (`git ls-files` → exclusions → réécritures → docs neutres → **gate dur** → dépôt séparé commit neutre sans footer ; `--dry-run`/`--check-only`/`--run-tests`/`--push` ; `.git` du dépôt séparé préservé, idempotent). **S3704** `docs/gitlab/{README_gitlab,CONTRIBUTING}.md` (onboarding pro, remplacent `CLAUDE.md`). **S3705** `Makefile` racine `gitlab-release`/`-dry`/`gitlab-check`. **S3706** `.github/workflows/ai-trace-guard.yml` garde-fou ajouts futurs = `--check-only` (l'export reste-t-il propre ? invariant correct vs scan brut toujours rouge). **S3707** runbook `docs/gitlab_publication.md`. **S3708** `tests/test_gitlab_release.py` **12/12 PASS** (652 tests collectés 0 erreur). Export réel vérifié : **3139 conservés / 1575 exclus / 2 docs neutres**, scan indépendant **0 trace**, commit `CL-Embedded <cl-embedded@isae-supaero.fr>` **sans footer**, idempotent. GitLab pas encore en place : push réel = action manuelle ultérieure (`git remote add gitlab …`). | Sprint 38 ✅ implémenté (S3800–S3809, board réelle NUCLEO-F439ZI) — **Mise à jour EWC autonome déclenchée par gate de nouveauté embarqué** : remplace le déclencheur humain (`PROTO_FLAG_UPDATE`) par un gate Mahalanobis + `SlidingWindowDriftDetector` qui décide à bord QUAND/AVEC QUEL label updater. 4 politiques (P0 `frozen`, P1 `always`, P2 `gated_truelabel`=active learning, P3 `gated_pseudolabel`=100 % autonome) × 2 datasets (Monitoring drift inter-équipements, Pronostia temporel/première faute) × 2 init_modes (pretrained/scratch). **S3801–S3804 préexistants** (config, réf PC 16 cellules, firmware gate `-DEWC_AUTO_UPDATE`/`-DGATE_PSEUDO_LABEL` + `drift_detector.c` Unity 6/6, board P0/P1). **S3805** : **verrou levé** — la réponse UART V3 ne transporte ni `n_updates` ni verdict ; sous `-DEWC_AUTO_UPDATE` **uniquement** `pipeline.c` **réinterprète 2 champs du snapshot** (`snap.auroc←(float)g_last_verdict` 0/1/2, `snap.forgetting←(float)g_n_updates`), **wire format V3 inchangé → `sensor_stream.py` intact** (champs déjà exposés, sémantique documentée) ; `.bss` défaut **105 036 B invariant**, gate **+300 B** (`g_drift`+`g_n_updates`+`g_last_verdict`), builds P2/P3 0 warning, 0 régression. Driver `run_sprint38_board.py` `--policy gated_*` : **Maha d'enrôlement welford = miroir exact PC** (`X[y==0][:500]`, ≠ `train_maha_board`), `export_weights_c.py --mahal --ewc-head --drift-thresholds`, stream **sans `--update`** (label transmis pour SGD P2 + scoring), `_pc_gate_replay` reconstruit le verdict PC sur l'ordre board (`maha.partial_fit` sur DRIFT en P3). **8 cellules gated board réelle (0 CRC)** : `update_rate` strictement **frozen=0 < gated≈0.025 (186–196 MAJ) < always=1** ; `mean_latency` gated **79–82 µs** ≪ always **238–251 µs** ≪ 100 ms (**Gap 2 ✅**) ; `gate_overhead≈27 µs` ; **parité verdict board↔PC = 1.000 sur les 8 cellules** (mêmes seuils exportés ⇒ décision d'update identique) ; F1 `pretrained` préservé (monitoring 0.919, pronostia gated_true 0.889) vs `scratch` où `always` (plafond appris en ligne) domine. Refs `scratch` frozen/always complétées. **S3806** `board_pc_parity38.py` → 16 JSON `exp_S38_parity_{policy}_{ds}_{init}.json` (table par échantillon pred+verdict, confusion verdict_pc×verdict_board ; frozen pred_parity=1.000, verdict_parity=1.000 partout). **S3807** `aggregate_sprint38.py` → `exp_S38_summary.json` indexé `[dataset][init_mode][policy][platform]` + **`economy_table`** (deltas vs `always`) : gated économisent **~97 % des MAJ** et ~159–169 µs/éch. au coût de **+300 B** RAM, F1 préservé (pretrained Δ≤0.02). **S3808** notebook `notebooks/cl_eval/autonomous_ewc/comparison.ipynb` (nbconvert OK, 4 PNG : économie vs précision, update_rate vs F1, confusion drift↔faute, parité board↔PC). **S3809** `test_sprint38_autonomous.py` **10/10 PASS** (calibration P95×{2.5,1.3}, logique 4 politiques, déterminisme gate, structure summary+economy_table, Gap 2, verdict_parity==1.0) + Unity `make test` 122 (drift 6/6, **2 TinyOL préexistants hors périmètre, 0 régression**). Docs : `S3805–S3809` ✅, roadmap, `triple_gap.md` (§ Gap 2 latence gate vs SGD permanent, § Gap 3 RAM gate +300 B).

---

## Graphe de connaissance — À consulter en priorité

**Au démarrage de chaque nouvelle conversation**, consulter en priorité :

1. [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) — rapport des nœuds architecturaux centraux, connexions clés, et rationale de design extrait du code et des docs
2. [`graphify-out/graph.html`](graphify-out/graph.html) — visualisation interactive (ouvrir dans le browser pour naviguer les dépendances)
3. [`graphify-out/graph.json`](graphify-out/graph.json) — données structurées pour requêtes programmatiques

Ce graphe reflète l'état réel du dépôt : code C firmware, modèles Python, specs docs, configs YAML, et leurs interdépendances. Il est plus fiable que la mémoire ou les suppositions pour répondre à "où est défini X ?" ou "qu'est-ce qui dépend de Y ?".

**Requêtes utiles en CLI :**

```bash
graphify query "what connects ewc_head.c to pipeline.c?"
graphify path "EWCMlpClassifier" "model_weights.h"
```

**Mise à jour du graphe** (après chaque sprint ou changement significatif) :

```bash
# Dans Claude Code IDE :
/graphify . --update
```

---

## Identité du projet

**Titre** : Apprentissage Incrémental pour Systèmes Embarqués à Ressources Limitées  
**Acronyme** : CL-Embedded  
**Type** : Recherche M2 + prototype industriel  
**Institution** : ISAE-SUPAERO (DISC), en collaboration avec ENAC (LII) et Edge Spectrum  
**Auteur** : Léonard Rivals  
**Période** : 16 mars – 6 août 2026  
**Deadline manuscrit préliminaire** : 15 avril 2026  

---

## Objectif principal (Objectif 1 du stage)

Implémenter et comparer trois méthodes d'apprentissage incrémental (continual learning, CL) sur PC en Python, puis les porter en C sur microcontrôleur **NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 256 Ko SRAM).

Application visée : **maintenance prédictive industrielle** (détection de panne, classification d'état).

> **Board de travail** : **NUCLEO-F439ZI** (Cortex-M4, 256 Ko SRAM, pas de NPU) — c'est la carte utilisée pour tout le développement firmware (Sprints 16–19 et au-delà). La STM32N6 (Cortex-M55, 64 Ko, NPU) était la cible originale du stage mais n'est pas disponible ; ne pas concevoir pour elle.

---

## Les modèles du projet

| ID | Modèle | Dataset | Méthode CL | Fichier spec |
|----|--------|---------|------------|--------------|
| M1 | TinyOL + tête OtO | D1 — Pump (temporel) | Architecture-based | `docs/models/tinyol_spec.md` |
| M2 | EWC Online + MLP | D2 — Monitoring (tabulaire) | Regularization-based | `docs/models/ewc_mlp_spec.md` |
| M2b | EWC INT8 | D5 CMAPSS, D6 Paderborn | Regularization-based + quantification | `src/models/ewc/ewc_mlp_int8.py` |
| M3 | HDC Hyperdimensional | D2 — Monitoring (tabulaire) | Architecture-based (non-neuronal) | `docs/models/hdc_spec.md` |
| M4 | Détecteurs non-supervisés | D3 CWRU, D4 Pronostia | Baseline anomaly detection | `src/models/unsupervised/` |

**M4 regroupe** : `KMeansDetector`, `DBSCANDetector`, `MahalanobisDetector`, `KNNDetector`, `PCABaseline` — utilisés comme baselines et pour la détection d'anomalies.

**Variantes anomaly** : `TinyOLAnomalyDetector` (`src/models/tinyol/tinyol_anomaly_detector.py`), `EWCOneClassSVM` (`src/models/ewc/ewc_oneclass.py`).

**Priorité d'implémentation** : M2 → M3 → M1 → M2b INT8 → M4 baselines. ✅ Toutes implémentées.

---

## Contraintes hardware (NUCLEO-F439ZI — board active)

1. **RAM ≤ 256 Ko** — budget réel de la NUCLEO-F439ZI (192 Ko SRAM + 64 Ko CCM)
2. **Pas de NPU** — forward pass et backpropagation s'exécutent tous les deux sur Cortex-M4 en FP32
3. **FP32 partout** — pas de contrainte INT8 sur la NUCLEO ; les annotations `# MEM: @ INT8` restent utiles pour référence future mais ne bloquent rien
4. **Latence ≤ 100 ms** par inférence + mise à jour (critère Gap 2, mesuré via DWT)
5. **Pas d'accès à un dataset complet en RAM** — online learning ou buffer borné

> **Règle de code** : tout paramètre de taille (couches, buffer, embeddings) doit avoir une constante nommée dans `configs/`. La valeur par défaut doit tenir dans 256 Ko (NUCLEO). Ne pas sur-contraindre à 64 Ko.

---

## Triple gap — Positionnement scientifique

Le projet cherche à être le **premier travail** à satisfaire simultanément :

| Gap | Description | Statut projet |
|-----|-------------|-------------- |
| **Gap 1** | Validation sur données industrielles de séries temporelles réelles | ✅ Comblé — CWRU, Pronostia, CMAPSS, Paderborn validés (Sprints 18–23) |
| **Gap 2** | Opération sous 100 Ko RAM avec chiffres précis mesurés | ✅ Comblé formellement Sprint 20 — RAM 1 000 B (.bss), latence 3.7 µs P50 / 4.0 µs P99 sur NUCLEO-F439ZI |
| **Gap 3** | Quantification INT8 pendant l'entraînement incrémental | ✅ Comblé Sprint 22 — EWC INT8 + HDC INT8 Python+C, validés board |

**Chaque décision d'architecture doit être justifiée par rapport à ces trois gaps.**

---

## Datasets

### D1 — Large Industrial Pump Maintenance Dataset (Kaggle)

- **Type** : Séries temporelles multivariées (température, vibration, pression, RPM)
- **Label** : `maintenance_required` (binaire)
- **Scénario CL** : Domain-incremental avec drift temporel
- **Chemin** : `data/raw/pump_maintenance/`
- **Spec** : `docs/context/datasets.md`

### D2 — Industrial Equipment Monitoring Dataset (Kaggle)

- **Type** : Tabulaire statique (température, pression, vibration, humidité, type équipement)
- **Label** : `faulty` (0/1 binaire)
- **Scénario CL** : Domain-incremental par type d'équipement (pump → turbine → compressor)
- **Chemin** : `data/raw/equipment_monitoring/`

### D3 — CWRU Bearing Fault Dataset

- **Type** : Vibration accéléromètre (time-series)
- **Label** : type de défaut (normal / inner / outer / ball) ou sévérité
- **Scénario CL** : Domain-incremental par type de défaut ou par sévérité
- **Chemin** : `data/raw/cwru/`
- **Loader** : `src/data/cwru_dataset.py` — configs : `cwru_by_fault_config.yaml`, `cwru_by_severity_config.yaml`

### D4 — PRONOSTIA (FEMTO Bearing)

- **Type** : Accéléromètre + température, RUL
- **Label** : condition (normal / fault)
- **Scénario CL** : Class-incremental par condition
- **Chemin** : `data/raw/pronostia/`
- **Loader** : `src/data/pronostia_dataset.py` — config : `pronostia_config.yaml`

### D5 — CMAPSS (NASA Turbofan)

- **Type** : Simulation capteurs turbofan (21 variables)
- **Label** : RUL (Remaining Useful Life)
- **Scénario CL** : Domain-incremental par FD set (FD001–FD004)
- **Chemin** : `data/raw/cmapss/`
- **Loader** : `src/data/cmapss_loader.py` — configs : `cmapss_config.yaml`, `cmapss_feature_subset.yaml`

### D6 — Paderborn Bearing Dataset

- **Type** : Vibration (bearing damage evolution)
- **Label** : niveau de dommage (health / damaged levels)
- **Scénario CL** : Domain-incremental par degré de dommage
- **Chemin** : `data/raw/paderborn/`
- **Loader** : `src/data/paderborn_loader.py` — configs : `board_paderborn.yaml`, `paderborn_feature_subset.yaml`

---

## Structure du dépôt

```
cl-embedded/
├── CLAUDE.md                   ← CE FICHIER
├── README.md
├── pyproject.toml
├── configs/                    ← ≥ 49 fichiers YAML organisés en :
│   ├── {model}_config.yaml     ←   base (ewc, hdc, tinyol)
│   ├── board_{model}.yaml      ←   board-specific (board_ewc, board_hdc, board_tinyol, board_mahalanobis)
│   ├── {dataset}_config.yaml   ←   dataset-specific (cwru_by_fault, pronostia, cmapss, paderborn…)
│   ├── *_anomaly_detection_config.yaml  ← anomaly detection variants
│   ├── *_single_task_config.yaml        ← baselines single-task
│   └── *_feature_subset.yaml   ←   feature selection (cmapss, pronostia, paderborn)
├── data/
│   ├── raw/                    ← données brutes (gitignore)
│   └── processed/              ← features extraites (gitignore)
├── docs/
│   ├── models/                 ← specs détaillées M1-M4
│   ├── context/                ← contexte projet, hardware, datasets
│   ├── sprints/                ← 28 répertoires sprint avec tâches détaillées
│   ├── roadmap_phase1.md
│   ├── roadmap_phase2.md
│   └── triple_gap.md
├── firmware/
│   └── stm32f4_blink/          ← firmware NUCLEO-F439ZI (Cortex-M4)
│       ├── inc/                ← 11 headers (ewc_head.h, hdc.h, mahalanobis.h, tinyol.h, pipeline.h, profiling.h, metrics.h, hw_info.h, model_weights.h, ewc_head_int8.h, stm32f4xx.h)
│       ├── src/                ← 11 sources (main.c, pipeline.c, ewc_head.c, ewc_head_int8.c, hdc.c, mahalanobis.c, tinyol.c, profiling.c, metrics.c, hw_info.c, syscalls.c)
│       ├── tests/              ← 8 fichiers test Unity (test_ewc_head.c, test_hdc.c, test_mahalanobis.c, test_tinyol.c, test_pipeline.c, test_profiling.c, test_ewc_int8.c, test_runner.c)
│       ├── startup/            ← startup_stm32f439xx.s
│       ├── build/              ← artefacts compilés (.elf, .bin, .map)
│       └── Makefile            ← arm-none-eabi-gcc, cibles : all / test / flash / size
├── skills/                     ← prompts Claude spécialisés
├── src/
│   ├── data/                   ← 8 loaders (pump, monitoring, cwru, pronostia, cmapss, paderborn, battery, + utils)
│   ├── models/
│   │   ├── ewc/                ← ewc_mlp.py, ewc_mlp_int8.py, ewc_oneclass.py, fisher.py
│   │   ├── hdc/                ← hdc_classifier.py, base_vectors.py
│   │   ├── tinyol/             ← autoencoder.py, oto_head.py, tinyol_anomaly_detector.py
│   │   └── unsupervised/       ← kmeans, dbscan, mahalanobis, knn, pca detectors
│   ├── training/               ← boucles CL, scénarios
│   ├── evaluation/             ← metrics.py, memory_profiler.py, anomaly_metrics.py, online_metrics.py, feature_importance.py, drift_detector.py, compute_cost.py, plots.py, eda_plots.py, feature_space_plots.py
│   └── utils/                  ← quantization helpers, misc
├── experiments/                ← 160+ expériences (exp_001–exp_160, exp_S18–exp_S23)
├── notebooks/                  ← exploration, visualisation
├── tests/
└── scripts/                    ← 48 scripts CLI (train, eval, export, board, visualize)
```

---

## Conventions de code

### Style
- **Python ≥ 3.10**
- Type hints obligatoires sur toutes les fonctions publiques
- Docstrings format NumPy
- `black` pour le formatage, `ruff` pour le linting
- Pas de dépendances lourdes non justifiées (pas de HuggingFace transformers, etc.)

### Nommage
- Classes : `CamelCase` — ex. `EWCMlpClassifier`, `TinyOLAutoencoder`
- Fonctions/variables : `snake_case`
- Constantes de config : `UPPER_SNAKE_CASE` dans les configs YAML
- Fichiers de résultats : `{exp_id}_{model}_{dataset}_{date}.json`

### Reproductibilité
- Seed fixé via `utils/reproducibility.py` : `set_seed(42)` par défaut
- Chaque expérience génère un `config_snapshot.yaml` dans `experiments/exp_XXX/`
- Pas de résultats hardcodés — tout sort d'une exécution de script

### Contrainte embarquée dans le code
```python
# Toujours annoter les tenseurs avec leur empreinte mémoire estimée
# Format : # MEM: {taille en octets} @ FP32 / {taille en octets} @ INT8
hidden = torch.relu(self.fc1(x))  # MEM: 256 B @ FP32 / 64 B @ INT8
```

---

## Métriques d'évaluation obligatoires

Pour chaque expérience CL, reporter systématiquement :

| Métrique | Formule | Module |
|---------|---------|--------|
| `acc_final` | Accuracy sur toutes les tâches vues après entraînement complet | `evaluation/metrics.py` |
| `avg_forgetting` (AF) | Chute moyenne d'accuracy entre pic et fin par tâche | `evaluation/metrics.py` |
| `backward_transfer` (BWT) | Impact de l'apprentissage futur sur les tâches passées | `evaluation/metrics.py` |
| `ram_peak_bytes` | RAM maximale mesurée à l'exécution (tracemalloc) | `evaluation/memory_profiler.py` |
| `inference_latency_ms` | Latence forward pass (moyenne sur 100 runs) | `evaluation/memory_profiler.py` |
| `n_params` | Nombre total de paramètres entraînables | automatique via `model.parameters()` |

**Modules d'évaluation additionnels** (ne pas réimplémenter — ils existent dans `src/evaluation/`) :

- `anomaly_metrics.py` — AUROC, F1, précision/rappel pour détection d'anomalies
- `online_metrics.py` — métriques temps-réel par batch (forgetting tracker incrémental)
- `feature_importance.py` — attribution par tâche (shapley-like, per-task)
- `drift_detector.py` — détection de drift de distribution entre domaines
- `compute_cost.py` — MACs, budget mémoire analytique (coût computationnel)
- `plots.py` / `eda_plots.py` / `feature_space_plots.py` — visualisations standard du projet

---

## Références bibliographiques clés

Utiliser ces clés BibTeX exactes (issues de `references.bib` du projet manuscrit) :

- `Ren2021TinyOL` — TinyOL (architecture-based, MCU-validated)
- `Kirkpatrick2017EWC` — EWC original
- `Benatti2019HDC` — HDC online learning sur MCU
- `Ravaglia2021QLRCL` — QLR-CL, rejeu latent UINT8
- `Kwon2023LifeLearner` — LifeLearner Tiny (STM32H747)
- `Capogrosso2023TinyML` — Survey TinyML de référence
- `DeLange2021Survey` — Taxonomie CL de référence
- `Hurtado2023CLPdM` — CL × maintenance prédictive

---

## Ce que Claude Code NE DOIT PAS faire

- ❌ Inventer des résultats de benchmark ou des chiffres RAM non mesurés
- ❌ Introduire des dépendances qui ne seraient pas portables sur MCU (ex. bibliothèques de visualisation dans le code de modèle)
- ❌ Modifier les hyperparamètres dans les fichiers source — toujours passer par les configs YAML
- ❌ Supprimer les annotations `# MEM:` dans les couches de modèle
- ❌ Créer des notebooks sans les déplacer dans `notebooks/`
- ❌ Committer des données brutes (les répertoires `data/` sont dans `.gitignore`)
- ❌ Modifier `model_weights.h` à la main — toujours générer via `scripts/export_weights_c.py` ou `scripts/export_weights_tinyol.py`
- ❌ Hardcoder des tailles de buffer ou de couche dans le code C — utiliser les `#define` dans les headers `inc/`
- ❌ Bypasser ou modifier le protocole UART (`pipeline.c` v3) sans mettre à jour `sensor_stream.py` en parallèle

---

## Interlocuteurs et rôles (pour les commentaires / TODO)

- `TODO(arnaud)` — question pour Arnaud Dion (superviseur ISAE-SUPAERO)
- `TODO(dorra)` — question technique pour Dorra Ben Khalifa (quantification, hardware)
- `TODO(fred)` — point pour Frédéric Zbierski (Edge Spectrum, contexte industriel)
- `FIXME(gap1/2/3)` — point bloquant lié à l'un des trois gaps

---

## Environnement de développement — Outils installés

| Outil | Version | Chemin | Usage dans le projet |
| ----- | ------- | ------ | -------------------- |
| **STM32CubeMX** | 6.17.0 | `~/STM32CubeMX/STM32CubeMX` | Génération de code d'initialisation MCU, configuration périphériques NUCLEO-F439ZI, pinout |
| **CMake** | 4.3.2 | `/usr/local/bin/cmake` | Build system pour le code C embarqué, intégration VSCode via CMake Tools |
| **OpenOCD** | — | — | Flash + debug NUCLEO-F439ZI via ST-LINK v2 embarqué |
| **arm-none-eabi-gcc** | — | — | Compilateur croisé pour le firmware C (Cortex-M4) |

> CMake est le build system de référence (`firmware/stm32f4_blink/` + `firmware/stm32f4_cubemx/`). STM32CubeMX génère le `.ioc` et les fichiers HAL pour la NUCLEO-F439ZI.

---

## Commandes rapides

```bash
# Setup
pip install -e ".[dev]"

# Entraînement modèles core
python scripts/train_ewc.py --config configs/ewc_config.yaml
python scripts/train_hdc.py --config configs/hdc_config.yaml
python scripts/train_tinyol.py --config configs/tinyol_config.yaml
python scripts/train_mahalanobis.py --config configs/board_mahalanobis.yaml

# Évaluation complète
python scripts/evaluate_all.py --exp_dir experiments/

# Tests Python
pytest tests/ -v

# Profiling mémoire
python scripts/profile_memory.py --model ewc --dataset monitoring

# Firmware — build & flash (depuis firmware/stm32f4_blink/)
make all          # Compile pour Cortex-M4 (-Os, FP hard)
make flash        # Flash via OpenOCD ST-LINK
make test         # Build + run tests Unity sur host (TEST_MODE=1)
make size         # Taille Flash/RAM du binaire

# Board — streaming & expériences
python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset cwru
python scripts/board_dataset_builder.py --dry-run
python scripts/board_experiment_recorder.py --exp exp_S23_01

# Export poids modèles → header C
python scripts/export_weights_c.py --model ewc --config configs/board_ewc.yaml
python scripts/export_weights_tinyol.py --config configs/board_tinyol.yaml
```

## Fin d'une implementation

A la fin d'implementation d'une tache d'un sprint :

1. Mettre à jour le fichier doc de la tâche du sprint et le roadmap de sprint.
2. Invoquer le skill **`graphify_sprint_update`** (`skills/graphify_sprint_update.md`) — il évalue si un update du graphe de connaissance est pertinent avant de le lancer.
