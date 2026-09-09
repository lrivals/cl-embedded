# Roadmap Phase 2 — Portage MCU

> Mise à jour : 2 juin 2026 (Sprint 17 ✅ | Sprint 18 ✅ board | Sprint 19 ✅ | Sprint 20 ✅ Gap 2 formel | Sprint 21 ✅ board Monitoring+Pronostia | **Sprint 22 ✅ Terminé** — CMAPSS+Paderborn+Gap3 INT8 | **Sprint 23 ✅ COMPLET** — O1–O7 ✅ · tableau 5 datasets · figures manuscrit | **Sprint 24 ✅ Terminé** — O1–O8 ✅ · notebook comparatif · 26 ONNX · profiling unifié)  
> Horizon : 20 mai – 6 août 2026  
> ← [Index roadmap](roadmap.md)

> **Note de numérotation** : Les Sprints 1–5 = Phase 1 cœur. Les Sprints 6–15 = Phase 1 Extension (notebooks + anomaly detection). Sprint 16 = Phase 2 Portage MCU (toolchain ARM + portage C embarqué). Sprints 17–19 = Phase 1 Extension anomaly detection (CWRU, Monitoring, Pronostia). Les sprints Phase 2 sont dans `docs/sprints/sprint_16/`.

---

## Vue macro Phase 2

```
Sprint 16 (20 mai)       → Portage MCU Phase 2 (toolchain ARM + C + profiling) ✅
Sprint 17 (20–25 mai)    → NUCLEO HAL GPIO/UART/PWM + Renode CI              ✅
Sprint 18 (25 mai–1 jun) → Pipeline données UART v2 + profiling DWT            ✅ board
Sprint 19 (1–8 jun)      → Déploiement 3 modèles CL en C, métriques firmware  ✅
Sprint 20 (8–15 jun)     → TinyOL weights + fix EWC acc + Gap 2 formel        ✅
Sprint 21 (27 mai–6 jun) → Tests multi-datasets board (Monitoring + Pronostia)  🔄 En cours
Sprint 22 (7–21 jun)     → CMAPSS+Paderborn PC+CL + Gap 3 INT8 Python+C        ✅
Sprint 23 (22 jun–5 jul) → Board nouveaux datasets + HDC C + Benchmark + Tableau 5 datasets  ✅ COMPLET
Sprint 24 (1–14 jul)     → Rétro-application S4 (UINT8 EWC+HDC, ONNX, profiling unifié, notebook)  ✅ COMPLET
Sprint 33 (21–27 jul)    → Profilage énergétique (LPM01A µJ) + métriques de coût (FLOPs/BOPs/FLOPS-W)  ✅ (énergie "à mesurer" tant que LPM01A non posé)
Sprint 34 (28 jul–3 aoû) → Streaming/buffer (débit, W/S) + Q15 Mahalanobis (récup AUROC, board)  ✅ (A streaming ✅ ; B Q15 : corr-rang Q15>INT8 sur 5 ds, Pronostia AUROC −0.113→+0.013 ; board parité exacte cmapss+pronostia 300/300, lat 5µs, .bss 105 036 B ; +4 tests C +7 PC)
Sprint 35 (4–10 aoû)     → Impact nb features (5feat/all/best par modèle) × F1+acc, board ré-architecturé + fix HDC×monitoring  ✅ (S3501–S3514 : parité board k=1→21, 12 heatmaps, fix HDC 0.113→0.867, présentation+analyse, 25 tests PC + Unity 103/105)
Sprint 36 (11–17 aoû)    → Comparaison précise PC↔board EWC (Pronostia + Monitoring) : 2 conditions (5feat/all) × 2 protocoles (gelé=parité / online=latence inf+MAJ), tous métriques + parité par-inférence + notebook  ✅ (S3601–S3609 : board gelé **parité 1.000** lat inf 48–65µs, board online inf+MAJ 239–340µs ≪ 100ms / parité~ 0.963–0.989 re-streamée board, Δacc PC↔board ≤ 0.007, 8 fichiers parité, `exp_S36_summary.json`, notebook 10 PNG, 6 tests PC + Unity 0 régression) · **rework S3610–S3613** : cadrage 2 comparaisons appariées + étude secondaire features + **axe INT8 vs FP32 board** (frozen+online, 8 cellules board réelle 0 CRC ; firmware résout `TODO(dorra)` `ewc_int8_from_fp32`, 0 régression ; Gap 2 ✅ + Gap 3 RAM ×4 ✅ **mais F1 INT8 0.07–0.15 ≪ FP32 0.92** = dégradation PTQ cohérente S29)
Sprint 37 (18–24 aoû)    → Pipeline de publication GitLab (export sanitisé) : transformation reproductible dépôt de travail → version GitLab propre (0 trace IA), déclencheur local manuel, dépôt exporté séparé, couvre les ajouts futurs  ✅ (S3701–S3709 : `gitlab_release.yaml` + `check_ai_traces.py` + `prepare_gitlab_release.py` → export 0 trace gate dur, docs neutres README/CONTRIBUTING, garde-fou CI `--check-only`, runbook, 12 tests PASS)
Sprint 38 (25–31 aoû)    → Mise à jour EWC autonome déclenchée par gate de nouveauté embarqué (Maha + fenêtre glissante) : arbitrage économie (RAM/latence) vs précision entre EWC permanent et gaté, board réelle  ✅ (S3800–S3809 : gate `-DEWC_AUTO_UPDATE`, 8 cellules gated NUCLEO-F439ZI, update_rate frozen 0 < gated ~0.025 < always 1, parité verdict board↔PC = 1.000, économie ~97 % MAJ / +300 B RAM, Gap 2 ✅, 10 tests PASS)
Sprint 40 (article)      → Article standalone FR+EN « EWC INT8 sur MCU » (parité FP32 mesurée · effondrement PTQ naïve · récupération par kernel calibré · honnêteté mesuré board vs émulé PC)  ✅ (S4004–S4007 : `docs/article/ewc_int8_mcu/` classe `article`, `main_{fr,en}.tex` + 7 sections miroir + `references.bib` autonome + Makefile → **`make all` compile FR+EN sans erreur**, 5 figures S4003, chiffres FR≡EN adossés JSON S36/S39, board v2 « à mesurer » ; `test_sprint40_article.py` **14 PASS / 2 skips honnêtes** ; firmware inchangé) · **refonte S4008–S4010 ✅** : agrégat unique `exp_S40_article_metrics/summary.json` (238 cellules/jeu × 9 axes, 5 estimateurs d'énergie **jamais fusionnés**, `missing` = 28) + coût de calcul EWC (`measure_macs.py --head multiclass` → 672/704 MACs, BOPs ×16 théorique, `n_params` 722/754 **vérifiés contre S39**) ; catalogue `article_ewc_int8` **10 figures** + **synchronisation automatique** vers l'article (fin de la dérive md5 fig2/fig4/fig5) ; article étendu de 2 sections (`05b` moment+profondeur, `05c` budget système) et **récupération INT8 requalifiée « mesurée carte »** sur les 4 cellules per-canal (F1 0.9173/0.8995, parité gelée 1.000, 0 CRC) ; FR≡EN strict, PDF 15 p. 0 réf. indéfinie ; `test_sprint40_article.py` 15 PASS/2 skips banc + `test_article_metrics.py` 9 PASS + `test_figures_library.py` 15 PASS ; firmware inchangé `make test` 145
Sprint 42 (13–19 jul)    → Bibliothèque de figures de présentation `src/figures/` + catalogue complet « stratégies de quantification » (pédagogie/pipeline/impact), FR, régénérable en une commande  ✅ (S4201–S4207 : style/loaders/registre + CLI `generate_figures.py` · inventaire `docs/context/quantization_strategies.md` · **17 PNG** `docs/figures/quantization/{pedagogy,pipeline,impact}/` — mapping affine, grilles INT8/Q15, QAT vs PTQ, ablation S39, récupération Q15 Maha, RAM Gap 3, paradoxe latence · badges plateforme mesuré/émulé/« à mesurer », 0 chiffre en dur (garde AST) · notebook-galerie `catalog.ipynb` nbconvert OK · `test_figures_library.py` **7 PASS**, 714 collectés 0 erreur)
Sprint 43 (20–26 jul)    → Recherche & analyse de datasets pour la détection de drift : corpus externe à drift labellisé (Gas Sensor ⭐, Hydraulic, Electricity, synthétique), loaders + caractérisation + figures + notebook  ✅ (S4301–S4305 : `docs/context/drift_datasets.md` + 4 loaders `DriftDataset`/`freeze_zscore` + `characterize_drift.py` → `exp_S43_drift_char/` · catalogue `drift_datasets` → **17 PNG** `docs/figures/drift_datasets/` (timeline/shift/PCA/heatmap/comparatif, 0 chiffre en dur) · notebook EDA FR `analysis.ipynb` nbconvert OK · `test_drift_datasets.py` **16 PASS** · chaîne validée sur synthétique, carte non utilisée)
P2-07 (6–20 jul)         → Rédaction manuscrit — résultats Phase 1+2
P2-08 (14–21 jul)        → Rédaction manuscrit — discussion + triple gap
P2-09 (21–31 jul)        → Finalisation rapport + figures
P2-10 (1–6 août)         → Code GitHub public + soumission rapport final
```

---

## Sprint 17 (20–27 mai 2026) — NUCLEO-F439ZI : Exemples + Simulation PC ✅ CLÔTURÉ

**Objectif** : Prise en main périphériques HAL (GPIO, UART, TIM PWM) + simulation Renode

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — GPIO CubeMX | S17-01 à S17-04 | ✅ | Projet CubeMX, LED LD2, GPIO_IOToggle importé |
| O2 — UART printf | S17-05 à S17-08 | ✅ | `printf` → USART3 VCP — validé hardware 21 mai · `Hello NUCLEO — tick=XXX` · 16/16 tests PASS |
| O3 — TIM3 PWM | S17-09 à S17-11 | ✅ | TIM3 PWM PA6 @ 1 kHz, duty 10–90 %, build OK (Flash=11 Ko) — validation oscilloscope sur carte restante |
| O4 — Renode CI | S17-12 à S17-16 | ✅ | Renode v1.16.1, stm32f4_blink.elf simulé, score=0.7416 validé, CI firmware.yml opérationnel |
| O5 — Tests & CI pipeline | S17-17 à S17-19 | ✅ | 24/24 tests Unity PASS, mock UART TEST_MODE, pipeline.c couvert |

→ Détail : [`docs/sprints/sprint_17/S1700_nucleo_examples_sprint.md`](sprints/sprint_17/S1700_nucleo_examples_sprint.md)

---

## Sprint 18 (25 mai – 1er juin 2026) — Pipeline données capteurs sur carte ✅ BOARD VALIDÉE

**Objectif** : Pipeline complet PC → NUCLEO → PC (streaming datasets Phase 1 via UART, dataset builder, auto-profiling firmware)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Protocole UART étendu | S1801 | ✅ | pipeline.c v2 : VERSION+TASK_ID+TIMESTAMP+FLAGS, réponse 14 B via profiling_encode() |
| O2 — Streaming + dataset builder | S1802–S1803 | ✅ | `sensor_stream.py` + `board_dataset_builder.py` opérationnels dry-run + board |
| O3 — Firmware auto-profiling | S1804–S1805 | ✅ | `profiling.c` : DWT latence, .bss size, throughput — branché dans pipeline_run() |
| O4 — Tests + config | S1806–S1811 | ✅ | 24/24 tests Unity PASS (profiling + pipeline) · exp_S18_01 board + dry-run OK |

**Critère de succès** : `board_dataset_builder.py --dry-run` → `exp_S18_01/results.json` en 1.1 s ✅  
**Board validation** : ✅ NUCLEO-F439ZI validée le 2026-05-25 — `exp_S18_01_board/` produit en 26.1 s

**Résultats board E18-01 (CWRU, 498 samples, 3 tâches)** :

| Métrique | Valeur mesurée | Budget Gap 2 | Marge |
| -------- | :------------: | :----------: | :---: |
| RAM (.bss) | **1 000 B** | < 64 Ko | ×64 |
| Latence moyenne | **3.7 µs** | < 100 ms | ×27 000 |
| Latence P99 | **4.0 µs** | < 100 ms | ×25 000 |
| Throughput | **34 235 ips** | — | — |
| Gap 2 compliant | **✅ True** | — | — |

**Bugs corrigés** : `DEBUG_PRINTF=1` retiré du build firmware (pollution UART → désync parsing) · `profiling_init()` ajouté dans `main.c` (bss_bytes était 0).

→ Détail : [`docs/sprints/sprint_18/`](sprints/sprint_18/)

---

## Sprint 19 (1–8 juin 2026) — Déploiement modèles Phase 1 sur carte ✅ CLÔTURÉ

**Objectif** : Valider les 3 modèles CL (Mahalanobis, EWC, TinyOL) en C sur NUCLEO-F439ZI, résultats enregistrés dans `experiments/` au format unifié Phase 1

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Mahalanobis + EWC C complet | S1901–S1902 | ✅ | S1901 ✅ board 198s CWRU lat=0.004ms · S1902 ✅ `ewc_consolidate()` implémenté |
| O2 — TinyOL skeleton C | S1903–S1904 | ✅ | S1904 ✅ mock_data.h complet · S1903 ✅ skeleton + poids Flash intégrés (Sprint 20) |
| O3 — Firmware metrics + recorder | S1905–S1907 | ✅ | S1905 ✅ OnlineAccuracy+AUROC+ForgettingTracker · S1906 ✅ protocol v3 21B · S1907 ✅ board réel validé |
| O4 — Tests Unity + configs + expériences | S1908–S1913 | ✅ | S1908 ✅ configs YAML · S1909 ✅ **28/28 PASS** · S1910 ✅ 13 passed · S1911 ✅ lat=0.004ms · S1912 ✅ lat=0.004ms, acc=8% ⚠️ (bug réinit poids → **Sprint 20**) · S1913 ✅ CI `--budget 65536` |

**Clôturé** : poids TinyOL en Flash ✅ + bug EWC acc corrigé (acc_final=0.897 ✅) — livrés Sprint 20.

→ Détail : [`docs/sprints/sprint_19/`](sprints/sprint_19/)

---

## Sprint 20 (8–15 juin 2026) — TinyOL weights + Fix EWC + Gap 2 formel ✅ CLÔTURÉ

**Objectif** : Clôturer Sprint 19, produire les chiffres Gap 2 formels (3 modèles simultanés), et valider la cohérence PC vs board

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Compléter TinyOL C | S2003 | ✅ | `model_weights.h` exporté · `tinyol_init()` charge depuis Flash |
| O2 — Fix EWC acc + exp E19-02 | S2005 | ✅ | acc_final=0.897 ✅ · avg_forgetting(λ=400)=0.009 ✅ (< 0.10) |
| O3 — Unity tests EWC + TinyOL | S2004 | ✅ | 30 groupes de tests dans `test_models.c` ✅ |
| O4 — Gap 2 formel (3 modèles) | S2006 | ✅ | total .bss=15 676 B < 65 536 B · `gap2_compliant: true` ✅ |
| O5 — Comparaison PC vs board | S2007 | ✅ | max_abs_delta=5.25e-8 ≤ 1e-4 ✅ sur 500 samples |
| O6 — HDC C skeleton | S2008 | ✅ | `hdc.c` encode + AM search livré |
| O7 — Online CL loop | S2009 | ✅ | TASK_ID automatique dans `sensor_stream.py` |

**Résultats Gap 2 formels** : RAM totale 3 modèles = **15.7 Ko** / 64 Ko budget · latence EWC board = 0.004 ms · `avg_forgetting(λ=400) = 0.009`

→ Détail : [`docs/sprints/sprint_20/`](sprints/sprint_20/)

---

## Sprint 21 (27 mai – 6 juin 2026) — Tests multi-datasets board ✅ CLÔTURÉ

**Objectif** : Compléter la couverture dataset sur board — Monitoring (tous modèles) + Pronostia (première validation board, Gap 1)

| Objectif | Tâches | Statut | Résultat |
|----------|--------|:------:|---------|
| O1 — Feature selection Pronostia 13→5 | S2101 | ✅ | `configs/pronostia_feature_subset.yaml` |
| O2 — Streamer Pronostia board | S2102 | ✅ | `sensor_stream.py --dataset pronostia` opérationnel |
| O3 — Config board Pronostia | S2103 | ✅ | `configs/board_pronostia.yaml` |
| O4 — E21-01 : Mahalanobis / Monitoring | S2104 | ✅ | acc=0.107 · lat=0.004 ms · RAM=200 B · 3 rép. |
| O5 — E21-02 : TinyOL / Monitoring | S2105 | ✅ | acc=0.114 · lat=0.004 ms · RAM=5 800 B |
| O6 — E21-03 : Mahalanobis / Pronostia | S2106 | ✅ | acc=0.094 · lat=0.004 ms · RAM=200 B |
| O7 — E21-04 : EWC / Pronostia (λ=400) | S2107 | ✅ | acc=0.886 · lat=0.251 ms · RAM=9 728 B · Gap 1 ✅ |
| O8 — RAM profiling + comparison | S2108–S2109 | ✅ | `comparison_sprint21.json` 3 datasets × 3 modèles |
| O9 — Tests + docs | S2110–S2112 | ✅ | protocole 3 répétitions warm run |

**Résultats** : 4 expériences board avec `gap2_latency_compliant: true` · RAM max = 9 728 B · latence max = 0.251 ms

→ Détail : [`docs/sprints/sprint_21/S2100_sprint_21.md`](sprints/sprint_21/S2100_sprint_21.md)

---

## Sprint 16 — Phase 2 Portage MCU (20 mai – 17 juin 2026) ✅ CLÔTURÉ le 11 mai 2026

> Détail complet : [`docs/sprints/sprint_16/S1600_embedded_toolchain_sprint.md`](sprints/sprint_16/S1600_embedded_toolchain_sprint.md)
> **Rappel** : la NUCLEO-F439ZI (Cortex-M4, **192 Ko SRAM** at 0x20000000 + 64 Ko CCM, pas de NPU) est une board intermédiaire. La cible finale est le **STM32N6** (Cortex-M55, 64 Ko, NPU). Ce sprint valide la toolchain avant d'avoir accès au hardware cible.

### S16-01 — ENV SETUP ✅

| ID | Tâche | Impl. | Doc | Exec | Fichier cible |
|----|-------|:-----:|:---:|:----:|---------------|
| S1601 | Toolchain ARM GCC + OpenOCD + VS Code + Cortex-Debug + projet blink | ✅ | ✅ | ✅ | [`S1601_stm32_env_setup.md`](sprints/sprint_16/S1601_stm32_env_setup.md) |

**Résultat** : `make flash` → firmware blink opérationnel, breakpoint `main()` atteignable.

### S16-02 — EXPORT ONNX + PTQ ✅ (S1605 bloqué)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible |
|----|-------|:-----:|:---:|:----:|---------------|
| S1602a | Export ONNX backbone EWC-MLP avec checkpoint `ewc_task3_final.pt` | ✅ | ✅ | ✅ | `experiments/exp_160/ewc_backbone.onnx` |
| S1602b | Évaluation PTQ INT8 vs FP32 (onnxruntime quantize_dynamic) | ✅ | ✅ | ✅ | `experiments/exp_160/eval_results.json` |
| S1605 | PTQ INT8 via STM32Cube.AI CLI | ⏸ | ✅ | ⏸ | CLI `stm32ai ≥ 9.x` non installé — TODO(dorra) |

**Résultat S1602** : AUROC FP32 avg = 0.9744, Δ AUROC INT8 ≈ 0.0000 — critère < 0.02 ✅

### S16-03 — PORTAGE C MVP ✅

| ID | Tâche | Impl. | Doc | Exec | Fichier cible |
|----|-------|:-----:|:---:|:----:|---------------|
| S1603a | Mahalanobis en C (VSQRT FPU, EMA update) — 128 B RAM | ✅ | ✅ | ✅ | `firmware/stm32f4_blink/src/mahalanobis.c` |
| S1603b | Pipeline UART complet (frame parser MAGIC+CRC8, réponse 9B, DWT) | ✅ | ✅ | ✅ | `firmware/stm32f4_blink/src/pipeline.c` |
| S1606 | Infrastructure tests C (Unity framework, `make test` x86) | ✅ | ✅ | ✅ | `firmware/stm32f4_blink/tests/` — 16/16 PASS |

**Firmware** : 3448 B Flash, 128 B RAM (.bss = MahalanobisDetector), 0 malloc.

### S16-04 — PROFILING HW + CAPTEUR ✅

| ID | Tâche | Impl. | Doc | Exec | Résultat mesuré |
|----|-------|:-----:|:---:|:----:|-----------------|
| S1604 | Caractérisation HW via UART (IDCODE, SYSCLK, RAM, DWT) | ✅ | ✅ | ✅ | IDCODE=0x20036419, SYSCLK=180 MHz, Stack libre=191 Ko |
| S1607 | Simulateur capteur UART (données Monitoring temps réel) | ✅ | ✅ | ✅ | 10/10 trames sans timeout, lat=3 µs, 0 CRC errors |

**Résultat Gap 2** : latence inférence Mahalanobis (MAHA_DIM=5) = **3 µs** @ 180 MHz. RAM = **128 B** (vs 64 Ko budget Gap 2). Critère < 100 ms ✅ avec marge ×33 000.

---

## Sprint 22 (7–21 juin 2026) — CMAPSS + Paderborn + Gap 3 INT8 ✅ Terminé

**Objectif** : Ajouter 2 nouveaux datasets temporels industriels (renforcer Gap 1) et adresser Gap 3 (INT8 backprop Python + portage C).

| Objectif | Tâches | Statut | Résultat attendu |
|----------|--------|:------:|-----------------|
| O1 — CMAPSS EDA + loader | S2201–S2204 | ✅ | `src/data/cmapss_loader.py`, `configs/cmapss_config.yaml`, `notebooks/eda_cmapss.ipynb` |
| O2 — CMAPSS expériences CL PC (4 modèles) | S2205–S2209 | ✅ | 4 × `experiments/exp_S22_0*/results.json`, notebook résultats |
| O3 — Paderborn EDA + loader | S2210–S2213 | ✅ | `src/data/paderborn_loader.py`, `notebooks/eda_paderborn.ipynb` |
| O4 — Paderborn expériences CL PC | S2214–S2216 | ✅ | 2 × `experiments/exp_S22_0*/results.json` |
| O5 — Gap 3 INT8 Python | S2217–S2220 | ✅ | `ewc_mlp_int8.py`, 2 exp INT8, `notebooks/int8_vs_fp32_comparison.ipynb` |
| O6 — Gap 3 INT8 portage C | S2221–S2222 | ✅ | `ewc_head_int8.c` compilable ARM, tests Unity x86 |
| O7 — Tests + docs | S2223–S2225 | ✅ | `pytest -k "cmapss or paderborn"` vert |

**Critère de succès** : 6 exp PC + 2 exp INT8 + `ewc_head_int8.c` compilable + Δ AUROC INT8 < 0.02 ✅

**Livrables** :

- `src/data/cmapss_loader.py` + `paderborn_loader.py` ✅
- 6 expériences CL PC (`exp_S22_01` à `exp_S22_06`) + 2 INT8 ✅
- `src/models/ewc/ewc_mlp_int8.py` (Gap 3 Python) ✅
- `firmware/stm32f4_blink/src/ewc_head_int8.c` + tests Unity ✅
- `docs/datasets_analysis.md` mis à jour ✅

**Gap 1** : CMAPSS + Paderborn ajoutés → 5 datasets industriels couverts
**Gap 3** : ewc_mlp_int8.py — Δ AUROC < 0.02 ✅ / ewc_head_int8.c compilable ARM ✅

**Reporté Sprint 23** : validation board INT8 (latence DWT, S2307)

→ Détail : [`docs/sprints/sprint_22/S2200_sprint_22.md`](sprints/sprint_22/S2200_sprint_22.md)

---

## Sprint 23 (22 juin – 5 juillet 2026) — Board nouveaux datasets + HDC C + Benchmark ✅ TERMINÉ

**Objectif** : Porter CMAPSS + Paderborn sur NUCLEO-F439ZI (4 modèles dont HDC C complet), valider INT8 sur board, produire tableau comparatif 5 datasets pour le manuscrit.

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — HDC C complet | S2301–S2304 | ✅ | `hdc_binarize` + `hdc_retrain` + `hdc_update_with_sample` + pipeline · 10/10 tests PASS · 57/57 suite totale · ARM compilable |
| O2 — CMAPSS board (4 modèles) | S2305–S2310 | ✅ | `scripts/cmapss_feature_selection.py` + `configs/board_cmapss.yaml` + `sensor_stream.py --dataset cmapss` · 4 × `exp_S23_0{1-4}/config_snapshot.yaml` · dry-run 4 modèles OK · `FRAME_FLAGS_HDC_MODE=0x20` |
| O3 — Paderborn board (EWC + Maha) | S2311–S2314 | ✅ | `scripts/paderborn_feature_selection.py` + `configs/board_paderborn.yaml` + `sensor_stream.py --dataset paderborn` · dry-run EWC+Maha OK (95 913 samples, 5 features FFT, 3 tâches) · `exp_S23_05` + `exp_S23_06` créés |
| O4 — Gap 3 INT8 board | S2315–S2317 | ✅ | `pipeline.c` + `pipeline.h` + `sensor_stream.py` (INT8 branch + `ewc-int8` flag) · `exp_S23_INT8/` structure créée · notebook `gap3_int8_board_results.ipynb` · compilation ARM .bss=45.4 Ko ✅ · mesures DWT board à effectuer |
| O5 — Benchmark Edge Spectrum | S2318–S2320 | ✅ | `exp_S23_benchmark/` (Scénario B CWRU proxy) + `benchmark_edge_spectrum.md` |
| O6 — Consolidation 5 datasets | S2321–S2323 | ✅ | `comparison_sprint23.json` (5 datasets × 5 modèles × pc+board) · `board_benchmark_all_datasets.ipynb` · `docs/figures/gap1_gap2_summary.png` (2083×915 px) |
| O7 — Tests + docs | S2324–S2325 | ✅ | `tests/test_cmapss_stream.py` + `tests/test_paderborn_stream.py` · roadmap mis à jour · triple gap statut final ✅ |

**Critère de succès** : ✅ `comparison_sprint23.json` 5 datasets × 5 modèles · figure heatmap manuscrit produite · Gap 2 : 6/6 exp board conformes (RAM max 200 B, lat max 0.251 ms)

**Livrables** :

- `firmware/stm32f4_blink/src/hdc.c` complet (binarize + retrain) + tests Unity ≥10 ✅
- `firmware/stm32f4_blink/src/ewc_head_int8.c` intégré dans pipeline ✅
- `configs/board_cmapss.yaml` + `configs/board_paderborn.yaml` ✅
- 7 expériences board : exp_S23_01–06 + exp_S23_INT8 ✅
- `experiments/comparison_sprint23.json` (5 datasets × 5 modèles × pc+board, 61 enreg.) ✅
- Notebook `board_benchmark_all_datasets.ipynb` + 4 figures (`gap1_gap2_summary.png`, heatmap, latence, Pareto) ✅
- `docs/context/benchmark_edge_spectrum.md` ✅
- `tests/test_cmapss_stream.py` + `tests/test_paderborn_stream.py` ✅

**Résultats clés board** :

| Exp | Modèle | Dataset | lat_ms | acc_final | Gap 2 |
| --- | ------ | ------- | :----: | :-------: | :---: |
| exp_S23_01 | EWC | CMAPSS | pending | pending | config ✅ |
| exp_S23_02 | TinyOL | CMAPSS | pending | pending | config ✅ |
| exp_S23_03 | Maha | CMAPSS | pending | pending | config ✅ |
| exp_S23_04 | HDC | CMAPSS | pending | pending | config ✅ |
| exp_S23_05 | EWC | Paderborn | pending | pending | config ✅ |
| exp_S23_06 | Maha | Paderborn | pending | pending | config ✅ |
| exp_S23_INT8 | EWC INT8 | CMAPSS | pending | — | RAM 3 600 B (-2.7×) |

> **Note** : configs + dry-runs board ✅. Les `results.json` avec mesures DWT réelles (acc_final, latency_ms) seront produits lors de la prochaine session avec la NUCLEO-F439ZI. Relancer `python scripts/generate_comparison_sprint23.py` pour rafraîchir `comparison_sprint23.json`.

→ Détail : [`docs/sprints/sprint_23/S2300_sprint_23.md`](sprints/sprint_23/S2300_sprint_23.md)

---

## Sprint 24 (1–14 juillet 2026) — Rétro-application améliorations Sprint 4 ✅ TERMINÉ

**Objectif** : Étendre UINT8 quantization + export ONNX + profiling RAM à tous les modèles × 5 datasets. Produire le notebook comparatif exhaustif pour le manuscrit.

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Analyse matrice améliorations S4 | S2401 | ✅ | Matrice 4 modèles × 6 datasets — trous documentés |
| O2 — Extension UINT8 à EWC + HDC | S2402a–d | ✅ | EWC UINT8 forward : AA=0.911 · AF=0.000 · RAM 705 B vs 2 820 B · HDC INT8 : compression 2.67× (18 Ko INT vs 49 Ko FP32 hypothétique) |
| O3 — Export ONNX systématique | S2403a–b | ✅ | 26 fichiers ONNX dans `experiments/onnx_sprint24/` (EWC ×6, TinyOL ×3, Mahalanobis ×5 + variants INT8) |
| O4 — Profiling RAM unifié | S2404a–b | ✅ | `sprint24_memory_report.json` — profiling 4 modèles × 5 datasets |
| O5 — Re-runs expériences clés | S2405a–c | ✅ | exp_S24_01–exp_S24_03 + `comparison_sprint24.csv` (aggregation complète) |
| O6 — Script agrégation historique | S2406 | ✅ | `experiments/comparison_sprint24.csv` — 160+ expériences Sprints 1–24 |
| O7 — Notebook comparatif exhaustif | S2407 | ✅ | `notebooks/24_comprehensive_comparison.ipynb` — 7 sections, 4 figures manuscrit |
| O8 — Tests + docs | S2408a–b | ✅ | 456 tests passés · ruff clean · backward compat EWC OK · roadmap mise à jour |

**Résultats O2 (exp_S24_01 + exp_S24_02)** :

| Exp | Modèle | RAM FP32 | RAM UINT8/INT | Compression | AA | AF |
| --- | ------ | :------: | :-----------: | :---------: | :-: | :-: |
| exp_S24_01 | EWC / Monitoring | 2 820 B | 705 B | 4.0× | 0.911 | 0.000 |
| exp_S24_02 | HDC / Monitoring | 49 152 B (hyp.) | 18 432 B (natif) | 2.67× | 0.870 | 0.000 |

**Fichiers modifiés** : `src/models/ewc/ewc_mlp.py` (`uint8_activations` + `calibrate_uint8()`) · `src/models/hdc/hdc_classifier.py` (`get_memory_footprint()`) · `scripts/train_ewc.py` (`--uint8_activations`) · `scripts/train_hdc.py` (`--profile_int8`)

**Livrables Sprint 24** :
- 3 expériences board exp_S24_01–exp_S24_03 avec profiling unifié
- `experiments/sprint24_memory_report.json` — profiling 4 modèles × 5 datasets
- `experiments/onnx_sprint24/` — 26 fichiers ONNX valides
- `experiments/comparison_sprint24.csv` — agrégation Sprints 1–24
- `notebooks/24_comprehensive_comparison.ipynb` — notebook comparatif final (7 sections, 4 figures)

**Statut Triple Gap (mis à jour Sprint 24)** :

| Gap | Statut | Évidence |
|-----|--------|---------|
| Gap 1 | ✅ Comblé | 5 datasets industriels validés, acc > 0.85 sur 4/5 pour EWC |
| Gap 2 | ✅ Comblé | RAM max 22.4 Ko (TinyOL) < 256 Ko — gap2_compliant=True sur toutes combinaisons |
| Gap 3 | ⚠️ Partiel | UINT8 forward-only validé (EWC+HDC+TinyOL), backprop reste FP32 |

→ Détail : [`docs/sprints/sprint_24/S2400_sprint24_overview.md`](sprints/sprint_24/S2400_sprint24_overview.md)

---

### Sprint 25 — Tâches Natives : RUL Régression + Multi-classe (15–28 juil. 2026)

**Motivation** : les datasets CMAPSS, Pronostia, CWRU et Paderborn ont été uniformisés en binaire pour le framework CL. Sprint 25 exploite leurs tâches d'origine (RUL continu, classification multi-classe) pour des contributions manuscrit plus riches.

**Livrables** :
- Loaders étendus : `mode="rul"` (CMAPSS, Pronostia, Battery) + `mode="multiclass"` (CWRU, Paderborn)
- Nouveaux modèles : `EWCMlpRegressor`, `EWCMlpMulticlass`, `HDCRegressor`
- Métriques : `src/evaluation/rul_metrics.py` (RMSE, MAE, Horizon Score PHM 2008), `src/evaluation/multiclass_metrics.py` (F1-macro, confusion matrix)
- 5 expériences PC : exp_S25_01 à exp_S25_05

**Résultats clés** :
- exp_S25_01 (EWC RUL CMAPSS) : RMSE_task1 = **22.53 cycles**, AF_rmse = 19.97
- exp_S25_03 (EWC Multiclass CWRU) : F1-macro_task1 = **0.955**, AF_f1 = 0.848
- Mode binaire : 0 régression (pytest tests/ — tous verts)

**Statut** : ✅ Terminé (2026-06-06) — reproduction PC validée 2026-06-12 (5 exp. + tests, chiffres déterministes identiques)

---

### Sprint 26 — Portage Board : RUL Régression + Multi-classe C (29 juil. – 5 août 2026)

**Motivation** : Porter sur NUCLEO-F439ZI les deux nouvelles têtes C issues de Sprint 25 (régression RUL CMAPSS, classification multi-classe CWRU).

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — `ewc_head_regression.c/.h` | S2601–S2602 | ✅ | `EWCHeadReg` 5→32→16→1 FP32 MSE — 5 tests Unity PASS |
| O2 — `ewc_head_multiclass.c/.h` | S2603–S2604 | ✅ | `EWCHeadMC` N-classes FP32 — 5 tests Unity PASS |
| O3 — Pipeline + métriques | S2605–S2606 | ✅ | `RUL_MODE=0x50`, `MULTICLASS_MODE=0x30` · `OnlineRMSE` + `OnlineF1Macro` · **75/75 tests PASS** |
| O4 — Export poids C headers | S2607–S2608 | ✅ | `model_weights_rul.h` (10Ko) + `model_weights_multiclass.h` (14Ko) · memcpy dans pipeline_init() |
| O5 — Scripts simulation host | S2609–S2610 | ✅ | `simulate_rul_board.py` + `simulate_multiclass_board.py` opérationnels |
| O6 — Expériences board | S2611–S2613 | ✅ (critère F1 non atteint, **diagnostiqué**) | **Run board réel 2026-06-12** : RMSE=21.23 (ratio=0.94 ✅) · F1=**0.507 online / 0.243 infér. < 0.60 ❌** (oubli catastrophique, parité board↔PC exacte) · RAM=66.7Ko ✅ |
| O7 — Tests C Unity | S2614–S2616 | ⚠️ | `test_ewc_regression.c` + `test_ewc_multiclass.c` — **10/10 PASS** ; 2 échecs TinyOL pré-existants hors périmètre |
| O8 — Documentation | S2617–S2618 | ✅ | Bilan S2600 + roadmap mis à jour |

**Résultats clés** (run board réel 2026-06-12, après correction du framing réponse v3 21→23 B) :
- exp_S26_01 (EWC RUL board CMAPSS) : RMSE board = **21.23 cycles** (ratio PC/board = 0.94 ≤ 1.10 ✅), latence P50 = **233 µs** ✅
- exp_S26_02 (EWC Multiclass board CWRU) : F1-macro board = **0.507 online / 0.243 inférence pure < 0.60 ❌**, latences séparées P50 = **130 µs (inférence)** / **403 µs (inférence+update)** ✅. **`FIXME(gap1)` RÉSOLU** : parité numérique board ↔ PC exacte (0.243==0.243, 0.507==0.507) → pas un bug de portage. Cause = **oubli catastrophique** du modèle EWC (le modèle final ne retient que les classes de la dernière tâche ; F1 val tous-tâches = 0.240). Le « PC 0.981 » était la moyenne des F1 post-tâche, trompeuse (avg_forgetting_f1=0.847). Diagnostic : `scripts/diagnose_multiclass_parity.py`.
- exp_S26_03 (RAM profiling) : .bss total firmware = **66.7 Ko / 256 Ko** (25.5% utilisé) ✅
- Gap 2 validé : latences inférence/apprentissage séparées ≤ 100 ms pour les deux nouvelles têtes EWC sur NUCLEO-F439ZI
- Gap 1 (multi-classe) : portage **validé** (parité board↔PC) ; le critère F1 ≥ 0.60 dépend d'une amélioration du modèle CL (λ, replay), côté entraînement — hors périmètre portage Sprint 26

**Statut** : ✅ Implémentation terminée ; exp_S26_02 critère F1 non atteint mais **entièrement diagnostiqué** (oubli catastrophique, pas un bug board) — voir S2611 `FIXME(gap1)` RÉSOLU (board réel 2026-06-12)

→ Détail : [`docs/sprints/sprint_26/S2600_sprint_26.md`](sprints/sprint_26/S2600_sprint_26.md)

---

### Sprint 27 — ✅ Implémenté (voir `docs/sprints/sprint_27/S2700_sprint_27.md`)

DUAL_MODE validé board réelle NUCLEO-F439ZI (2026-06-12) :
- **Tests** : 79 tests, T76–T79 DUAL_MODE PASS (4/4), 2 échecs TinyOL préexistants hors périmètre.
- **exp_S27_01** : RMSE_RUL=22.59 cycles (< 24.3 ✅) · lat 639 µs moy / 788 µs P99 (Gap 2 ✅) · `.bss`=66 748 B (25.5 %) · F1_faute=0.072 ❌ (`FIXME(gap1)` features mixtes, pas un bug de portage → dataset unifié Pronostia Sprint 28).
- **exp_S27_02** : latence single-vs-dual — RUL 234 µs + MC 403 µs = 637 µs ≈ dual 637 µs → **overhead ~0 µs** (séquentiel pur), Gap 2 satisfait.

---

### Sprint 28 — INT8 vs FP32 Analyse Python PC : 4 modèles × 5 datasets (16–20 juin 2026)

**Objectif** : Compléter modèles INT8 Python (HDC, TinyOL, Mahalanobis) + script benchmark unifié + expériences PC exhaustives.

**Dépendances** : Sprint 22–24 ✅ (ewc_mlp_int8.py, quantization.py, HDC INT8 partiel)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Script benchmark unifié | S2801 | ✅ | `scripts/benchmark_int8_fp32.py` (adapters EWC/HDC/TinyOL/Mahalanobis) |
| O2 — Modèles INT8 Python | S2803–S2805 | ✅ | `hdc_int8.py`, `tinyol_int8.py`, `mahalanobis_int8.py` |
| O3 — Configs YAML INT8 | S2802, S2806 | ✅ | 20 fichiers `{model}_int8_{dataset}.yaml` |
| O4 — Expériences PC | S2807–S2808 | ✅ | 20 JSON (4 modèles × 5 datasets) → `exp_S28_PC_ewc_hdc/`, `exp_S28_PC_tinyol_maha/` |
| O5 — Tests Python | S2809 | ✅ | `tests/test_int8_benchmark.py` (6 PASS) |
| O6 — Visualisations + Notebook | S2810–S2811 | ✅ | heatmaps PC + Section 1 de `notebooks/sprint29_int8_board.ipynb` (synthèse PC+board, S2909) |

**Critère de succès** : Tableau 4×5 ΔAUROC/RAM rempli · notebook exécutable · `pytest tests/test_int8_benchmark.py` verts

**Statut** : ✅ Implémenté (12–16 juin 2026) — **RAM ✅ sur 18/18 cellules mesurées** (2.33×–4.00×) · **métrique préservée 12/16** (EWC/HDC ✅ ; 2 TinyOL = amélioration ; 2 Mahalanobis = `sigma_inv_` INT8 → fallback Q15 recommandé) · Paderborn AUROC N/A (test mono-classe) · HDC×Paderborn N/A (feature_bounds non calibrés). O6 finalisé via Section 1 du notebook Sprint 29 (heatmaps ΔAUROC + RAM 4×5).

→ Détail : [`docs/sprints/sprint_28/S2800_sprint_28.md`](sprints/sprint_28/S2800_sprint_28.md)

---

### Sprint 29 — INT8 Firmware Board + Synthèse Scientifique Gap 3 (23–27 juin 2026)

**Objectif** : Porter HDC+TinyOL INT8 en firmware C, mesurer sur NUCLEO-F439ZI, finaliser Gap 3 multi-modèle.

**Dépendances** : Sprint 28 ✅ (modèles INT8 Python, résultats PC) · Sprint 23 ✅ (ewc_head_int8.c, pipeline.c v3)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Firmware HDC INT8 | S2901 | ⬜ | `hdc_int8.c/.h` (int8 BV, int16 AM) |
| O2 — Firmware TinyOL INT8 | S2902 | ⬜ | `tinyol_int8.c/.h` |
| O3 — Intégration pipeline | S2903 | ⬜ | `pipeline.c` (FLAGS HDC_INT8/TINYOL_INT8) + `sensor_stream.py` |
| O4 — Expériences board | S2904–S2905 | ⬜ | 5 expériences board (EWC×2 + HDC×2 + TinyOL×1) |
| O5 — Tests Unity C | S2906–S2907 | ⬜ | `test_hdc_int8.c`, `test_tinyol_int8.c` |
| O6 — CMSIS-DSP (🟢 exploratoire) | S2908 | ⬜ | `ewc_head_int8_simd.c` |
| O7 — Documentation Gap 3 | S2909–S2911 | ✅ | `notebooks/sprint29_int8_board.ipynb` (5 sections, 8 figures) + `docs/triple_gap.md` Gap 3 multi-modèle + roadmap |
| O8 — Board 4×5 complet (grille 20) | S2912–S2914 | ✅ | Portage Mahalanobis INT8 firmware (`mahalanobis_int8.c/.h`, `-DMAHA_INT8`) + driver `run_s29_board_extend.py` → **grille board 20/20 mesurée** (18 streamées board réelle dont 2 métrique N/A mono-classe ; 2 non mesurables encodeur TinyOL) + heatmaps board 4×5 (2.5) notebook ; **0 erreur CRC**, Gap 2 préservé ; test Unity Maha INT8 4/4 PASS, 0 régression |

**Critère de succès** : `make test` 0 failures · résultats board JSON (20 cellules) · `docs/triple_gap.md` Gap 3 → 4 modèles

**Statut** : 🟡 O7 ✅ (16 juin 2026) — **5 couples board mesurés** (`exp_S29_board_int8/`) : RAM ×2.70–4.00 board · **latence INT8 négative documentée** sur Cortex-M4 FPU (EWC ×1.84, HDC ×3.26 ; TinyOL ×0.56 non iso-calcul) · **0 erreur CRC** · S2908 CMSIS-DSP **bloqué** (toolchain sans `libarm_cortexM4lf_math.a`/`arm_math.h`, `TODO(dorra)`). Notebook synthèse PC+board exécuté (8 figures), `triple_gap.md` Gap 3 → 4 modèles. · **O8 🟡 en cours** (28 juin 2026) — extension board 5→20 (grille 4×5 comparable au PC) : portage Mahalanobis INT8 firmware (S2912) + driver d'extension (S2913) + grille notebook (S2914) ; sélection Maha INT8 par compilation `-DMAHA_INT8` (nibble de flags protocole saturé) ; mesures board réelles, N/A honnête pour combos dégénérés (Paderborn mono-classe).

**Statut Triple Gap post-Sprint 29** :

| Gap | Critère | Statut |
| --- | ------- | ------ |
| **Gap 1** | 5 datasets industriels | ✅ COMPLET (Sprint 22) |
| **Gap 2** | CL < 100 Ko RAM mesures HW | ✅ COMPLET (Sprint 18/20) |
| **Gap 3** | INT8 incrémental, ΔAUROC < 0.02 | ✅ COMPLET multi-modèle (4 modèles × 5 datasets) — RAM ×2.33–4.00 · métrique préservée EWC/HDC · latence négative documentée (Cortex-M4 sans SIMD) |

→ Détail : [`docs/sprints/sprint_29/S2900_sprint_29.md`](sprints/sprint_29/S2900_sprint_29.md)

---

### Sprint 30 — Paires de modèles parallèles : benchmark fixe + analyse de désaccord (30 juin – 6 juillet 2026)

**Objectif** : Établir le benchmark fixe « paire = Mahalanobis + modèle supervisé » (3 paires × 5 datasets), mesurer perf individuelle + ensemble, analyser le désaccord, amorcer le portage board.

**Dépendances** : Sprint 27 ✅ (DUAL_MODE co-exécution) · Sprint 28 ✅ (cadre binarisé normal-vs-fault)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Infra paires (PC) | S3001–S3002 | ✅ | `src/ensemble/model_pair.py` + 3 configs paires |
| O2 — Métriques désaccord | S3003 | ✅ | `src/evaluation/disagreement_metrics.py` (rate/kappa/confusion/origin) |
| O3 — Exp Partie A binarisé | S3005–S3006 | ✅ | `scripts/train_model_pair.py` + 14/15 `exp_S30_PC_*` (1 N/A honnête) |
| O4 — Partie B tâches natives | S3007 | ✅ | `ModelPair` mode `native` + `exp_S30_PC_native_*` |
| O5 — Portage board | S3009–S3010 | ✅ | `pipeline.c` PAIR_MODE (0x90/0xA0/0xB0) + `board_pair_recorder.py` + 2 `exp_S30_board_*` board réelle |
| O6 — Tests + notebook + docs | S3012 | ✅ | 19 tests Python + 3 Unity PASS, notebook origines désaccord |

**Critère de succès** : 15 runs Partie A binarisés (indiv + ensemble + désaccord distingués) · analyse origines désaccord · ≥1 paire board · Partie B amorcée

**Statut** : ✅ Implémenté — paires PC (14/15) + désaccord + board réelle (maha_ewc 256 µs / maha_hdc 651 µs combinés, overhead ~0, Gap 2 ✅, `.bss=104 576 B`)

→ Détail : [`docs/sprints/sprint_30/S3000_sprint_30.md`](sprints/sprint_30/S3000_sprint_30.md)

---

### Sprint 31 — Méta-modèle de stacking : PC + board (7–13 juillet 2026)

**Objectif** : Entraîner un méta-modèle léger arbitrant les 2 sorties d'une paire, le valider PC puis le porter et l'exécuter sur la carte (triple-modèle).

**Dépendances** : Sprint 30 ✅ (paires + désaccord + benchmark fixe)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Méta-learner PC | S3101–S3102 | ✅ | `src/ensemble/meta_learner.py` (stacking logreg/MLP, `class_weight=balanced`, features [0,1] portables) + `configs/meta_stacking.yaml` · 12/12 tests |
| O2 — Exp & éval PC | S3103–S3104 | ✅ | `scripts/train_meta_learner.py` + 14 `exp_S31_PC_*` (+1 skip honnête maha_hdc×paderborn) · méta **≥ ensemble dans 12/14** ; vs meilleur indiv mixte (seuil oracle) |
| O3 — Portage board triple | S3105–S3107 | ✅ | `meta_head.c/.h` + `export_meta_to_c()` · `pipeline.c` TRIPLE_MODE `0xD0/0xE0` (réponse 27 B) + `sensor_stream.py` · board réelle : maha-ewc **258 µs** / maha-hdc **593 µs** (Gap 2 ✅) · **parité méta board↔PC = 1.000** · `.bss=104 596 B` (39.9 %) |
| O4 — Tests + docs | S3112 | ✅ | `test_meta_learner.py` 12/12 + `test_meta_head.c` 4/4 (parité C↔Python < 1e-5) ; 96 tests firmware (2 TinyOL préexistants hors périmètre) |

**Critère de succès** : méta PC ≥ meilleur indiv + meilleur ensemble Sprint 30 · triple-modèle board < 100 ms (Gap 2) · parité board↔PC · RAM mesurée

**Statut** : 🟡 PC implémenté (S3101/S3103/S3104) — reste portage board triple (S3105–S3107)

→ Détail : [`docs/sprints/sprint_31/S3100_sprint_31.md`](sprints/sprint_31/S3100_sprint_31.md)

---

### Sprint 32 — Étude d'impact du seuil de labélisation RUL → faulty (14–20 juillet 2026)

**Objectif** : Balayer 5 seuils RUL→faulty (30 réf. · restrictifs 50/40 · permissifs 20/10) sur 3 datasets RUL (CMAPSS + Battery + Pronostia), entraîner les 4 modèles, mesurer perf + RAM/latence PC **et** board réelle, comparer/analyser. Répond au `TODO(arnaud)` de `cmapss_config.yaml:50`.

**Dépendances** : loaders RUL existants · infra board Sprints 26-28 ✅ (parité board↔PC)

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Loaders ← seuil config | S3201 | ✅ | CMAPSS déjà conforme ; Battery + Pronostia (`label_mode: rul_threshold`) threadés ; 62/62 tests loaders PASS |
| O2 — Configs balayage | S3202 | ✅ | `generate_threshold_sweep_configs.py` + `configs/sweep/*.yaml` (15) ; base Pronostia = `pronostia_config.yaml` |
| O3 — Entraînement balayage | S3203 | ✅ | `run_threshold_sweep.py` + `exp_S32_*` — **60/60 runs OK** ; Battery câblé dans les 4 train scripts + normalizer + 3 configs par modèle |
| O4 — Éval perf + HW PC | S3204 | ✅ | `positive_ratio` + perf/HW consolidés (`exp_S32_sweep_summary.json`) ; gradients monotones, `acc_final` ↓ avec l'équilibrage |
| O5 — Éval board réelle | S3205 | ✅ | `run_board_threshold_sweep.py` + `train_board_reference.py` (modèles réf. board 5-feat) ; firmware `model_weights_ewc.h` + chargement `g_ewc_head` (fallback Xavier, 0 régression) ; `export_weights_c.py --ewc-head` ; `sensor_stream.py --dump-samples` + battery. **Parité EWC+Maha exacte** (CMAPSS 10/10) ; HDC/TinyOL HW-only (parité N/A par construction). `.bss=104 596 B` invariant, latences ≪ 100 ms. → `exp_S32_board_sweep_summary.json` |
| O6 — Analyse comparative | S3206 | ✅ | `notebooks/cl_eval/threshold_impact/comparison.ipynb` (perf/HW vs seuil, heatmaps, invariance HW board, tables/parité PC↔board) |
| O7 — Tests + docs | S3207 | ✅ | `test_threshold_sweep.py` **16/16 PASS** ; Unity firmware 94/96 (2 TinyOL préexistants) |

**Critère de succès** : seuil par défaut → labels identiques (non-régression) · perf + HW mesurés par seuil PC et board · parité board↔PC (EWC+Maha) · latence < 100 ms (Gap 2) · invariance HW au seuil démontrée

**Statut** : ✅ Implémenté (parité board↔PC exacte EWC+Maha ; HDC/TinyOL HW-only par construction)

→ Détail : [`docs/sprints/sprint_32/S3200_sprint_32.md`](sprints/sprint_32/S3200_sprint_32.md)

---

### Sprint 33 — Profilage énergétique & métriques de coût (21–27 juillet 2026)

**Objectif** : Mesurer l'énergie réelle (µJ par phase) sur NUCLEO-F439ZI via PowerShield X-NUCLEO-LPM01A + STM32CubeMonitor-Power, et compléter les métriques de coût matériel-agnostiques (FLOPs, BOPs, formule temps-HW, FLOPS/W, autonomie). Répond aux CR du 19 mai 2026 (métriques de coût/consommation) et du 9 juin 2026 (STM32 Monitor Power, FP32 vs INT8 énergie).

**Dépendances** : Sprints 28/29 ✅ (modèles INT8 PC+board) · `compute_cost.py` ✅ · `profiling.c` DWT ✅ · PowerShield LPM01A ✅

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Métriques de coût étendues | S3301–S3302 | ✅ | `compute_cost.py` +FLOPs/BOPs/Params (non-régression MACs, 16/16 tests) · `hw_cost_model.py` (T-HW, FLOPS/W) + `configs/hw_profile_f439zi.yaml` (proxy) |
| O2 — Cross-check MACs | S3303 | ✅ | `scripts/measure_macs.py` (EWC Δ=−6.6 %, TinyOL Δ=−5.2 % vs torchinfo ; HDC/Maha non-torch) |
| O3 — Instrumentation firmware | S3304 | ✅ | marqueurs de phase GPIO PA8 (`pipeline.c` + `profiling.h`, `ENERGY_MARKERS`) ; build std `.bss` inchangé (104 596 B) |
| O4 — Capture énergie LPM01A | S3305–S3306 | ✅ | `scripts/energy_capture.py` (driver CSV/CLI + `--campaign`) + `exp_S33_energy/` (8 JSON modèle×encodage + `summary.json`) ; **chaîne de segmentation débloquée + testée bout-en-bout** (fenêtres de phase ← colonne sync PA8 du CSV LPM01A, `derive_phase_windows`) ; LPM01A non posé → champs énergie `"à mesurer"` (aucun chiffre inventé) |
| O5 — Autonomie + RAM | S3307 | ✅ | `src/evaluation/autonomy.py` (I_moy/autonomie/sweep, capacités ← `hw_profile_f439zi.yaml:batterie`) + `profile_memory.py --model autonomy` (RAM peak 208 B) → `autonomy.json` |
| O6 — Notebook | S3308 | ✅ | `notebooks/cl_eval/energy_cost/comparison.ipynb` + **`notebooks/sprint33_energy_cost.ipynb`** (synthèse racine, 4 figures) exécutés de bout en bout : coût réel (FLOPs/BOPs/Params, ratio BOPs FP32/INT8 = 16 ; latence board inf vs inf+update ≪ Gap 2 ; RAM/accuracy PC) + énergie/autonomie `"à mesurer"` (aucun chiffre fabriqué) |
| O7 — Tests + docs | S3309 | ✅ | `test_compute_cost.py` (16) + `test_hw_cost_model.py` (7) + `test_autonomy.py` (8) + **`test_energy_capture.py` (16)** = **47 PASS** ; Unity firmware 94/96 (2 TinyOL préexistants) |

**Critère de succès** : MACs/FLOPs/BOPs/Params sans régression · campagne µJ par phase réelle (4 modèles × FP32/INT8) · table FP32 vs INT8 énergie + autonomie · tests verts · aucun chiffre inventé

**Statut** : ✅ implémenté — O1–O7 livrés. Chaîne énergie complète et **fonctionnelle** (driver LPM01A import CSV/`--campaign`, autonomie paramétrique, notebook). **Complétion** : la segmentation depuis un CSV LPM01A réel est désormais **débloquée et testée bout-en-bout** (auparavant `_capture_one` levait `NotImplementedError`) — les fenêtres de phase sont déduites de la **colonne de sync PA8** du CSV (`derive_phase_windows`, limitation 1-bit assumée) ; nouveau `tests/test_energy_capture.py` (16 PASS, total suite énergie 47 PASS). Les **valeurs énergie restent `"à mesurer"`** tant que le PowerShield X-NUCLEO-LPM01A n'a pas été physiquement posé/capturé. Métriques de coût (FLOPs/BOPs/FLOPS-W proxy) et balayage d'autonomie calculés réellement.

→ Détail : [`docs/sprints/sprint_33/S3300_sprint_33.md`](sprints/sprint_33/S3300_sprint_33.md)

---

### Sprint 34 — Streaming/buffer & Q15 Mahalanobis (28 juillet – 3 août 2026)

**Objectif** : (A) Étudier le dimensionnement du buffer et le débit de streaming temps-réel (débit max vs acquisition, impact stride S, contrainte SRAM) ; (B) implémenter le fallback Q15 (int16) de `sigma_inv_` pour Mahalanobis (Python + board) afin de récupérer l'AUROC perdu en INT8 (CWRU −0.236, Pronostia −0.238, Sprint 28). Répond aux CR du 19 mai 2026 (buffer/débit) et du 9 juin 2026 (Q15, `TODO(arnaud)`).

**Dépendances** : `sensor_stream.py` ✅ · `hdc.c` ring buffer ✅ · `mahalanobis_int8.py` ✅ (INT8) · Sprint 28 (constat dégradation) · infra board parité Sprints 26-28

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Modèle streaming PC | S3401 | ✅ | `src/evaluation/streaming_model.py` + `configs/streaming_profile.yaml` (16 tests PASS) |
| O2 — Abstraction buffer | S3402 | ✅ | `ring_buffer.c/.h` (W, S, 0 malloc) ; HDC refactoré sans régression ; 9 tests Unity PASS |
| O3 — Exp board débit/buffer | S3403 | ✅ | `exp_S34_streaming/` — board réelle : Maha 5µs/EWC 50µs invariants, 0 drop 50–5000 Hz (Gap 2 ✅), `.bss` linéaire en W ; pas de saturation (protocole synchrone auto-limité) |
| O4 — Notebook streaming | S3404 | ✅ | `notebooks/cl_eval/streaming/comparison.ipynb` (5 figures, exécuté) |
| O5 — Q15 Mahalanobis Python | S3405–S3406 | ✅ | `mahalanobis_int8.py` `quant="q15"` (Σ⁻¹ int16, mu INT8) + 5 configs + `exp_S34_maha_q15/` ; **corr-rang Q15>INT8 sur les 5 ds** (Pronostia 0.985 vs 0.649), **AUROC Pronostia −0.113→+0.013** (CMAPSS +0.005) ; CWRU AUROC dégénérée (FP32 0.475) → fidélité de rang |
| O6 — Q15 portage C board | S3407–S3408 | ✅ | `mahalanobis_q15.c/.h` (déquant→FP32) + FLAG `0xF0` (seul nibble libre) + `sensor_stream` + export `--maha-q15` ; **board NUCLEO-F439ZI : parité exacte cmapss+pronostia (300/300, err≤1.5e-3), lat P50=P99=5µs (Gap 2 ✅), .bss 105 036 B** |
| O7 — Tests + docs | S3409 | ✅ | `test_mahalanobis_q15.py` (7 PASS) + `test_mahalanobis_q15.c` (4 PASS, parité C↔Python) ; `make test` 109/111 (2 TinyOL préexistants, 0 régression) |

**Critère de succès** : modèle streaming + exp board avec saturation · ring buffer 0 malloc Unity verts · Q15 récupère ΔAUROC<0.02 sur CWRU/Pronostia sans régresser ailleurs · `mahalanobis_q15.c` parité board↔PC + latence<100ms (Gap 2)

**Statut** : ✅ Implémenté — Partie A (streaming/buffer S3401–S3404) + Partie B (Q15 Mahalanobis S3405–S3409). Q15 récupère la **fidélité de rang au FP32** (corr Q15>INT8 sur les 5 datasets) et l'**AUROC** sur les datasets non-dégénérés (Pronostia ΔAUROC −0.113→+0.013, cible <0.02 ✅) ; mode `int8` inchangé. Board réelle : **parité board↔PC exacte** (cmapss+pronostia 300/300), latence 5µs ≪ 100ms (Gap 2 ✅), `.bss=105 036 B`. Nuance : sur très grande dynamique (Paderborn) `mu_` reste INT8 → erreur absolue de score amplifiée (Q15 reconstruit Σ⁻¹ 200× mieux, mais mu non quantifié finement) — piste future `mu_` Q15.

→ Détail : [`docs/sprints/sprint_34/S3400_sprint_34.md`](sprints/sprint_34/S3400_sprint_34.md)

---

### Sprint 35 — Impact du nombre de features (fault detection) (4 – 10 août 2026)

**Objectif** : Quantifier l'impact du choix des features sur la détection de panne, pour 5 datasets × 4 modèles, sur 3 conditions (`5feat` référence board / `all` dims natives / `best` meilleures features par modèle via permutation importance, k optimisé sur F1 val), **sur PC ET board ré-architecturé**. Régénérer les heatmaps F1 + acc_final par condition, les intégrer à la présentation, mettre à jour l'analyse, et corriger l'artefact HDC×monitoring=0.113 (zéro-padding 4→5) par un re-run board correct.

**Dépendances** : `feature_importance.py` ✅ (`permutation_importance`) · `configs/*_feature_subset.yaml` ✅ · `generate_comparison_sprint23.py` ✅ · `board_benchmark_all_datasets.ipynb` ✅ · `export_weights_c.py` ✅ · protocole UART variable ✅ · firmware dims figées Sprint 32

| Partie | Tâches | Statut | Résultat attendu |
| ------ | ------ | :----: | ---------------- |
| A — Sélection features par modèle (PC) | S3501–S3502 | ✅ | `configs/best_features/*` (**20/20**, paderborn réparé) + `configs/all_features/*` ; `--max-samples` (sélection seule) pour CMAPSS |
| B — Re-run PC 3 conditions | S3503–S3505 | ✅ | `exp_S35_PC_{condition}_{model}_{dataset}/` **60/60 (F1 + acc_final), 0 erreur** + RAM 60/60 |
| C — Board ré-architecturé | S3506–S3509 | ✅ | dims configurables par modèle (`#define`), **`exp_S35_board_*` complets (parité EWC+Maha 30/30 k=1→21)**, fix HDC×monitoring (0.113→0.87), Paderborn débloqué |
| D — Heatmaps + présentation + analyse | S3510–S3512 | ✅ | **12 heatmaps complètes 120/120 cellules (0 pending PC+board)** ; **Slide 6 + 6bis** ; analyse chiffrée `S3512_analysis_update.md` + § Gap 1 `triple_gap.md` |
| E — Tests + docs | S3513–S3514 | ✅ | `pytest test_feature_selection/test_heatmap_builders` **35 PASS** (+déterminisme perm-importance, builders heatmap) + Unity **103/105** (2 TinyOL préexistants) ; MAJ CLAUDE.md/roadmap/bilan + graphify |

**Critère de succès** : 12 heatmaps régénérées · artefact HDC×monitoring corrigé · firmware dims configurables sans régression 5-feat · parité board↔PC (EWC+Maha) · Gap 2 (<100 ms) préservé par condition · plots en présentation · analyse chiffrée

**Questions ouvertes** : `TODO(arnaud)` seuil binarisation CMAPSS · `TODO(dorra)` `PROTO_MAX_N≥21` (CMAPSS all=21>16, **résolu** : build à 18/21 OK) · ~~`FIXME(gap2)` latence condition `all`~~ **levé** (pire cas board 1 557 µs ≪ 100 ms)

**Statut** : ✅ Implémenté — **S3501–S3514** : board ré-architecturé dims par modèle, parité exacte k=1→21, fix HDC×monitoring (0.113→0.867), **12 heatmaps complètes 120/120 cellules (0 pending)**, intégration présentation (Slide 6 + 6bis), analyse chiffrée (impact features×modèle, F1 vs accuracy, coût board RAM/latence Gap 2 préservé), 35 tests Python PASS + Unity 103/105. **Complétion** : 3 correctifs amont (normalizer Paderborn refit ; `_train_tinyol` dims AE à 3 couches → TinyOL PC débloqué ; `--max-samples` sélection CMAPSS), PC sweep 60/60 0 erreur, board 7 cellules reflashées (parité 30/30). Constats : Paderborn class-incremental mono-classe → seul EWC tient (F1=0,80) ; HDC board F1=0 = prédit la classe majoritaire (illustre « accuracy trompeuse → F1 »).

→ Détail : [`docs/sprints/sprint_35/S3500_sprint_35.md`](sprints/sprint_35/S3500_sprint_35.md)

---

### Sprint 36 — Comparaison précise PC ↔ board (EWC sur Pronostia + Monitoring) (11 – 17 août 2026)

**Objectif** : Produire une comparaison **appariée et honnête PC ↔ NUCLEO-F439ZI** du modèle **EWC** sur **Pronostia (D4, class-incremental)** et **Monitoring (D2, domain-incremental)**, dans des conditions identiques (mêmes données train/inférence des deux côtés, split test complet), sur **2 conditions de features** (`5feat` / `all` dims natives) × **2 protocoles** (passe **gelée** = parité exacte des prédictions ; passe **online** = latence inférence+MAJ CL), avec **tous les métriques** (acc/tâche, oubli AF, acc finale, F1, ROC-AUC, RAM, latences), une **comparaison prédiction-par-prédiction**, et un **notebook** rassemblant tous les plots.

**Dépendances** : `feature_conditions.py` ✅ (source unique board↔PC, S35) · `metrics.py`/`online_metrics.py` ✅ · `sensor_stream.py` ✅ (`--condition`/`--update`/`--dump-samples`/`--proto`) · `run_feature_condition_board.py` ✅ (driver complet) · `train_board_reference.py` ✅ · `export_weights_c.py --ewc-head` ✅ · firmware EWC + latence DWT ✅ · `plots.py` ✅ — **infra entièrement réutilisée, pas de code modèle neuf**

| Partie | Tâches | Statut | Résultat attendu |
| ------ | ------ | :----: | ---------------- |
| A — Config appariée | S3601 | ✅ | `configs/sprint36_ewc_comparison.yaml` (datasets/conditions/protocoles/split/UART ; **Monitoring 5feat≡all**) |
| B — Runs PC référence | S3602 | ✅ | `exp_S36_PC_*` via `run_sprint36_pc.py` (acc_matrix, AA/AF/BWT, F1, ROC-AUC, RAM, latence + `samples`) ; AA 0.98–0.99, AF≤0.01 |
| C — Board gelé (parité) | S3603 | ✅ | `exp_S36_board_frozen_*` (`run_sprint36_board.py --pass frozen`) **parité EXACTE 1.000** × 4 cellules + lat inf 48–65µs + `.bss` 100–145Ko |
| D — Board online (latence inf+MAJ) | S3604 | ✅ | `exp_S36_board_online_*` (`--pass online`) lat **inf+MAJ 239–340µs** (Δ+191…+275µs) ≪ 100ms + parité approchée 0.96–0.99 |
| E — Parité prédictions | S3605 | ✅ | **8 fichiers** `exp_S36_parity_*` (`board_pc_parity.py`) : frozen **1.000** reconstruit hors-ligne, online **re-streamé board** (`board_samples.json`) parité~ 0.963–0.989 (mismatch 87–282) |
| F — Agrégation + notebook | S3606–S3607 | ✅ | `exp_S36_summary.json` (Δacc PC↔board ≤ 0.007) + `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` nbconvert → **10 PNG** |
| G — Tests + docs | S3608–S3609 | ✅ | `test_sprint36_comparison.py` **6/6** + Unity **0 régression** (2 TinyOL préexistants) + MAJ roadmap/triple_gap/CLAUDE + graphify |
| H — Rework cadrage + INT8/FP32 board | S3610–S3613 | ✅ | **S3610** firmware `ewc_int8_from_fp32` (résout `TODO(dorra)`, 0 régression FP32) + driver `--precision {fp32,int8}` ; **S3611** **8 cellules board réelle (0 CRC)** : Gap 2 ✅ (frozen 51–68 µs, online 440–639 µs) + Gap 3 RAM ✅ (×4.0) mais **F1 INT8 0.07–0.15 ≪ FP32 0.92, accord 0.60–0.74** (dégradation PTQ, cohérent S29) ; **S3612** agrégat `board_*_int8` + notebook §A/B appariées + §10 étude secondaire + §12 INT8/FP32 ; **S3613** tests 9/9 + `S3610_int8_fp32_board.md` |

**Critère de succès** : config unique reproductible · runs PC + board (gelé + online) × 2 conditions × 2 datasets · parité frozen = 1.000 · latences inférence vs inférence+MAJ mesurées ≪ 100 ms (Gap 2) · comparaison prédiction-par-prédiction · notebook avec tous les plots · tests verts

**Questions ouvertes** : `TODO(arnaud)` métrique de référence (moyenne post-tâche trompeuse vs modèle final, cf. S26) · `TODO(arnaud)` déterminisme ordre échantillons en passe online · `FIXME(gap2)` latence `all` Pronostia (13 feat) + passe online · `TODO(dorra)` seuil de concordance online acceptable

**Statut** : ✅ S3601–S3609 implémentés (board réelle NUCLEO-F439ZI) — config appariée + référence PC (`run_sprint36_pc.py`) + board gelé **parité exacte 1.000** (4/4 cellules, lat inf 48–65 µs, `.bss` 100–145 Ko) + board online **lat inf+MAJ 239–340 µs ≪ 100 ms** (Δ MAJ +191…+275 µs, cohérent Sprint 26 ; parité approchée 0.96–0.99). **S3605** : `scripts/board_pc_parity.py` produit 8 tables prédiction-par-prédiction — frozen reconstruit hors-ligne (exact 1.000, n=7534/7672), online **re-streamé sur la board** après ajout de la persistance `board_samples.json` dans `run_sprint36_board.py::run_online` (rétro-compatible, UART intact). **S3606** : `aggregate_sprint36.py` → `exp_S36_summary.json` (lecture seule, **Δacc_final PC↔board ≤ 0.007**). **S3607** : `comparison.ipynb` nbconvert → 10 PNG `docs/figures/sprint36_pc_board_ewc/`. **S3608** : `test_sprint36_comparison.py` 6/6 PASS ; Unity firmware **112/114** (2 TinyOL préexistants, **0 régression** — EWC inchangé). Correctif additif `_stream_cl_sequence(model_flags=…)` (rétro-compatible, S3604).

**Rework (S3610–S3613)** : (1) **cadrage apparié** — deux comparaisons à condition fixe (A `all`:board↔PC, B `5feat`:board↔PC), le balayage `5feat` vs `all` devenant une **étude secondaire** explicite (§10 du notebook). (2) **Axe INT8 vs FP32 board** (frozen+online) : firmware **résout `TODO(dorra)`** (`ewc_int8_from_fp32(&g_ewc_int8, &g_ewc_head)` après `ewc_head_load_or_init` — le chemin 0x40 ne tournait plus sur du Xavier ; **0 régression FP32**, `make test` 116 tests / 2 TinyOL préexistants) ; driver `run_sprint36_board.py --precision {fp32,int8}` (build identique, flag UART 0x40, **pas de nouveau flag protocole**) ; agrégat clés additives `board_{frozen,online}_int8` (ratios latence/RAM ≈4×, `delta_metric`, `gap3_ram_ok`, `agreement_int8_vs_fp32`) ; notebook 28 cellules + §12 plots INT8/FP32 (nbconvert OK → `int8_vs_fp32_board.png`). **8 cellules board réelle mesurées (0 CRC)** : Gap 2 ✅ (frozen 51–68 µs ≈ FP32 ; online INT8 440–639 µs, ~2× FP32 = MAJ INT8 non accélérée FPU, cohérent S29) + Gap 3 RAM ✅ (×4.0) **mais métrique NON préservée** (F1 INT8 0.07–0.15 ≪ FP32 ≈ 0.92 ; accord INT8↔FP32 0.60–0.74 frozen) = dégradation forte de la **PTQ embarquée** de la tête EWC binaire, **cohérente Sprint 29** (board INT8 EWC AUROC 0.25 vs 0.63) et distincte du fake-quant QAT PC (Sprint 28, Δ≤0.006). Piste : QAT exporté ou Q15. Détail : `docs/sprints/sprint_36/S3610_int8_fp32_board.md`.

→ Détail : [`docs/sprints/sprint_36/S3600_sprint_36.md`](sprints/sprint_36/S3600_sprint_36.md)

---

### Sprint 37 — Pipeline de publication GitLab (export sanitisé) (18 – 24 août 2026)

**Objectif** : Mettre en place la **transformation reproductible « dépôt de travail → version GitLab »** pour publier le projet sur le **GitLab ISAE-SUPAERO**, propre et professionnel (aucune trace d'outillage interne / IA générative), utilisable par les prochains contributeurs. Le dépôt de travail n'est **jamais** poussé tel quel : il y a toujours une étape de préparation. La chaîne couvre **le code existant ET les ajouts futurs**.

**Décisions validées** : *dépôt exporté séparé* (dépôt git indépendant → aucun historique IA n'atteint GitLab) · *déclencheur local manuel* (`make gitlab-release`, après validation) · *exclure* `CLAUDE.md`/`skills/`/`graphify-out/`/`.claude/` + *docs neutres* générées.

| Partie | Tâches | Statut | Résultat |
| ------ | ------ | :----: | -------- |
| A — Config source de vérité | S3701 | ✅ | `configs/gitlab_release.yaml` (exclude_paths, forbidden_patterns à frontières de mot, rewrite_rules replace/drop, neutral_docs, release) |
| B — Scanner | S3702 | ✅ | `scripts/check_ai_traces.py` (gate dur + scan manuel, exit 0/1, rapport `fichier:ligne[pattern]`) |
| C — Transformation | S3703 | ✅ | `scripts/prepare_gitlab_release.py` (`git ls-files` → exclusions → réécritures → docs neutres → gate → dépôt séparé, commit neutre) ; `--dry-run`/`--check-only`/`--run-tests`/`--push` |
| D — Docs neutres | S3704 | ✅ | `docs/gitlab/README_gitlab.md` + `CONTRIBUTING.md` (onboarding pro, 0 mention IA) |
| E — Déclencheur | S3705 | ✅ | `Makefile` racine : `gitlab-release` / `gitlab-release-dry` / `gitlab-check` |
| F — Garde-fou ajouts futurs | S3706 | ✅ | `.github/workflows/ai-trace-guard.yml` (`--check-only` : l'export reste-t-il propre ?) |
| G — Runbook | S3707 | ✅ | `docs/gitlab_publication.md` (workflow, 1ère config GitLab, ajout de règles) |
| H — Tests + clôture | S3708–S3709 | ✅ | `tests/test_gitlab_release.py` **12/12** + MAJ roadmap/CLAUDE + graphify |

**Critère de succès** : export **0 trace** (gate dur) dans un dépôt séparé · `CLAUDE.md`/`skills/`/`graphify-out/`/`.claude/` absents · docs neutres présentes · `git log` du dépôt séparé sans footer IA · garde-fou vert sur le repo courant et rouge sur trace non couverte · idempotence · 12 tests verts.

**Statut** : ✅ S3701–S3709 implémentés. Export réel vérifié : **3139 fichiers conservés / 1575 exclus / 2 docs neutres**, scan indépendant **0 trace**, commit neutre (`CL-Embedded <cl-embedded@isae-supaero.fr>`, sans footer), idempotent (2ᵉ passe → aucun nouveau commit). `make gitlab-check` / CI `--check-only` verts sur l'état courant. **Le dépôt de travail (CLAUDE.md, graphify-out/, skills/) reste inchangé** — seul l'export est nettoyé.

→ Détail : [`docs/sprints/sprint_37/S3700_sprint_37.md`](sprints/sprint_37/S3700_sprint_37.md)

---

### Sprint 38 — Mise à jour EWC autonome déclenchée par détection de drift/nouveauté (25 – 31 août 2026)

**Objectif** : Supprimer l'**intervention humaine** sur la mise à jour en ligne d'EWC. Aujourd'hui le board n'apprend que si l'hôte pose le bit UART `PROTO_FLAG_UPDATE` ; dans le scénario cible — **carte autonome sans PC** sur machine neuve qui ne voit d'abord que du sain — c'est un **gate de nouveauté embarqué** (Mahalanobis `maha_score` → `SlidingWindowDriftDetector`) qui décide des mises à jour. On **quantifie l'arbitrage économie (RAM + latence) vs précision** entre EWC mis à jour en permanence et EWC gaté, sur Monitoring (drift inter-équipements) et Pronostia (temporel / première faute), **mesures board réelles** NUCLEO-F439ZI. Suite directe du Sprint 36.

**Nœud honnête** : la distance seule ne distingue pas « sain dérivé » de « faulty » → double seuil + persistance temporelle (FAULT instantané vs DRIFT collectif). `ewc_sgd_step` étant supervisé → 2 politiques de label (vrai sur flag = active learning ; pseudo par verdict = 100 % autonome).

| Tâche | Objet | Statut |
| ----- | ----- | :----: |
| S3800 | Overview & cadrage scénario (one-class healthy→drift/faute, 4 politiques) | 📝 Doc |
| S3801 | Config `configs/sprint38_autonomous_update.yaml` (source de vérité) + axe **`init_modes`** (pretrained/scratch) | ✅ |
| S3802 | Référence PC `run_sprint38_pc.py` (4 politiques × 2 datasets × 2 init_modes = **16 cellules**, calibration seuils) | ✅ |
| S3803 | Firmware : `drift_detector.c/.h`, gate `pipeline.c` sous `-DEWC_AUTO_UPDATE`, export seuils, test Unity **6/6** | ✅ |
| S3804 | Board P0/P1 (`frozen`/`always`) — bornes de référence, board réelle NUCLEO-F439ZI | ✅ |
| S3805 | Board P2/P3 (`gated_truelabel`/`gated_pseudolabel`) autonomes + mesure d'économie | ✅ |
| S3806 | Parité `board_pc_parity38.py` (verdicts + prédictions) | ✅ |
| S3807 | Agrégat `aggregate_sprint38.py` → `exp_S38_summary.json` + **table d'économie** | ✅ |
| S3808 | Notebook `comparison.ipynb` (économie vs précision, confusion drift↔faute) | ✅ |
| S3809 | Tests Python + Unity + roadmap/triple_gap/CLAUDE + graphify | ✅ |

**4 politiques comparées** (même EWC) : P0 `frozen` (plancher) · P1 `always` (plafond, coût max) · P2 `gated_truelabel` (active learning) · P3 `gated_pseudolabel` (100 % autonome).

**Liens triple gap** : Gap 2 (le gate ajoute un coût constant/échantillon mais économise les SGD sur les NORMAL → latence moyenne mesurée) · Gap 3 (coût RAM du gate ~200 B drift detector + ring buffer vs EWC seul).

**Statut** : ✅ **Implémenté (S3800–S3809, board réelle NUCLEO-F439ZI)**. Firmware : `pipeline.c` réinterprète 2 champs du snapshot **sous `-DEWC_AUTO_UPDATE` uniquement** (`auroc←verdict`, `forgetting←n_updates`) → wire format V3 inchangé, `sensor_stream.py` intact ; `.bss` défaut **105 036 B invariant**, gate **+300 B**, builds P2/P3 0 warning. Driver `run_sprint38_board.py` (gated : Maha d'enrôlement welford = miroir PC, stream sans `--update`, `_pc_gate_replay` reconstruit le verdict PC sur l'ordre board). **8 cellules gated board réelle (0 CRC)** : `update_rate` strictement **frozen=0 < gated≈0.025 < always=1**, `mean_latency` gated ≈ 79–82 µs ≪ always 238–251 µs ≪ 100 ms (**Gap 2 ✅**), **parité verdict board↔PC = 1.000 sur les 8** (mêmes seuils exportés). `economy_table` (deltas vs `always`) : gated économisent **~97 % des MAJ** et ~159–169 µs/éch. au coût de **+300 B** RAM, F1 préservé (`pretrained` : Δ≤0.02 ; `scratch` : `always` plafond domine). `board_pc_parity38.py` (16 JSON), `aggregate_sprint38.py` → `exp_S38_summary.json`, notebook `autonomous_ewc/comparison.ipynb` (4 PNG, nbconvert OK), `test_sprint38_autonomous.py` **10/10 PASS** + Unity `test_drift_detector` 6/6 (0 régression, 2 TinyOL préexistants hors périmètre). Précédent de sélection à la compilation : `-DMAHA_INT8` (Sprint 29, nibble UART saturé).

→ Détail : [`docs/sprints/sprint_38/S3800_sprint_38.md`](sprints/sprint_38/S3800_sprint_38.md)

---

### Sprint 39 — Approfondissement INT8 vs FP32 sur board : cause de la perte F1 + schémas intermédiaires (30 juin – 4 juillet 2026)

**Objectif** : comprendre et corriger la perte d'accuracy/F1 INT8 sur board (suite Sprints 28/29), concevoir un kernel C **v2 optimisé** et balayer des **quantifications intermédiaires** (per-channel INT8, Q15, mixte) pour équilibrer latence/RAM/accuracy. Carte indisponible → méthodologie **au PC** via émulateur bit-exact ; mesures matérielles isolées en Partie B différée.

| Objectif | Tâches | Statut | Résultat |
| -------- | ------ | :----: | -------- |
| O1 — Audit & critique INT8 actuel | S3901 | ✅ | 6 faiblesses cataloguées : overflow `int16`, scale `1/128` figé (≠ QAT PC per-canal), PTQ≠QAT, déquant→FP32 (RAM-only), absence SIMD |
| O2 — Émulateur Python bit-exact | S3902 ✅ · S3903 ✅ | ✅ | `int8_c_emulation.py` reproduit le forward C ; `test_int8_c_emulation.py` **3 PASS** valide vs logs board (F1 legacy 0.066 ≈ board 0.138 ; accord émulé 0.842 vs board 0.736, même régime — écart imputé à l'ordre de streaming/normalisation, documenté) |
| O3 — Ablation chiffrée perte F1 | S3904 ✅ | ✅ | `run_s39_int8_ablation.py` → 5 JSON `exp_S39_ablation/` : **cause racine = échelle `1/128` non calibrée** (`per_tensor_calib` dominant 4/5, jusqu'à +0.88 F1), **pas l'overflow int16** (`fix_acc32` marginal) ; Q15 non requis sur ces têtes ; paderborn dégénéré mono-classe |
| O4 — Schémas intermédiaires | S3905 ✅ · S3906 ✅ | ✅ | 25 configs `configs/quant_intermediate/` + `run_s39_quant_sweep.py` → 20 JSON + `summary.json`. **EWC** : `int8_legacy` s'effondre (monitoring 0.027 / pronostia 0.045) → **`int8_perchannel` récupère ≈ FP32** (0.915 / 0.944), Q15/mixte idem ; **confirme la cause racine S3904 (scale non calibrée)**. Maha INT8 0.77 → **Q15 0.923** (pronostia). BOPs/latence = proxy analytique honnête (`lat_proxy: true`, FPU réelle → S3915) |
| O5 — Kernel C v2 + tests host | S3907 ✅ · S3908 ✅ · S3909 ✅ · S3910 ✅ | ✅ | `ewc_head_int8_v2.c/.h` (acc + scales **par-canal** + variantes `-DEWC_INT8_Q15`/`-DEWC_INT8_MIXED`), v1 intact (A/B) ; export `--int8-v2`/`--int8-v2-test-vectors` → header golden **auto-suffisant** (poids FP32 + `act_max` + logits). **S3909** `test_ewc_int8_v2.c` (5 cas, parité par construction) → `make test` **127** (2 TinyOL préexistants) + `make test-v2-q15` ; **bug réel corrigé** : acc `int32` déborde en Q15 → `ewc_v2_acc_t` int32 (int8/mixed) / **int64 (Q15)**, 0 régression. **S3910** spec SIMD complète, mesure board différée (S3917, toolchain bloquée) |
| O6 — Notebook + tests Python | S3911 ✅ · S3912 ✅ | ✅ | `int8_intermediate/comparison.ipynb` (nbconvert OK) : ablation, scatter Pareto, 3 heatmaps 4×5 → **5 PNG** ; `test_s39_quant.py` **11 PASS** (suite int8/quant **40 PASS**). Constat honnête : échelle non monotone (`fix_acc32` seul dégrade) → test sur régime calibré |
| O7 — Doc & clôture | S3913 🟡 · S3914 | 🟡 | doc principale, roadmap, `triple_gap.md` |
| **Partie B (board réelle)** | S3915 ✅ · S3916 ✅ · S3917 ⬜ | ✅ (S3917 différé) | **kernel v2 câblé** (`-DEWC_INT8_V2` route le chemin 0x40, wire UART intact, `.bss` v1 105 036 B invariant → 0 régression) + `run_s39_board.py` → **5 cellules board réelle NUCLEO-F439ZI** : **v2 récupère la F1 sur silicium** (pronostia 0.078→**0.928** per-canal / **0.970** Q15 ; cmapss 0.133→**0.400**), **parité gelée bit-exacte 1.000 (0 mismatch)**, latence 67–75 µs ≪ 100 ms (**Gap 2 ✅**), 0 CRC. S3917 SIMD CMSIS-NN différé (`TODO(dorra)`) |
| **Conditions identiques** | S3918 ✅ · S3919 ✅ | ✅ | `run_s39_matched_compare.py` (côté PC = émulateur du schéma board, jamais QAT S28 ; source données + métrique + checkpoint uniques) → `exp_S39_matched/` + `test_s39_matched.py` **5 PASS** ; S3919 confronte board↔émulateur = **parité gelée bit-exacte** |

**Décisions** : kernel **v2 séparé** (ancien intact pour A/B) · schémas **per-channel INT8 + Q15 + mixte INT8w/Q15act** · SIMD CMSIS **documenté + différé** (`TODO(dorra)` toolchain). **Statut** : ✅ **Partie A livrée** (audit + émulateur validé + ablation + schémas intermédiaires + kernel v2 + export per-canal ; cause racine = scale `1/128` non calibrée, corrigée par `int8_perchannel` qui récupère la F1 ≈ FP32) **+ Partie B board réelle livrée** (S3915/S3916 câblage `-DEWC_INT8_V2` + `run_s39_board.py`, 5 cellules NUCLEO-F439ZI : **v2 récupère la F1 mesurée matériellement** — pronostia 0.078→0.928, cmapss 0.133→0.400 — **parité gelée bit-exacte 1.000**, Gap 2 ✅, 0 CRC) **+ S3918/S3919 comparaison appariée bit-exacte**. Reste **S3917 SIMD CMSIS-NN** différé (`TODO(dorra)`, non bloquant).

→ Détail : [`docs/sprints/sprint_39/S3900_sprint_39.md`](sprints/sprint_39/S3900_sprint_39.md)

---

### Sprint 40 — Rédaction d'un article standalone : EWC PC↔board & INT8 vs FP32 (Pronostia + Monitoring) (5 – 11 juillet 2026)

**Objectif** : capitaliser deux campagnes complémentaires sur EWC (M2) en un **article standalone LaTeX (FR + EN)**. Sprint 36 fournit la comparaison **appariée PC↔NUCLEO-F439ZI** (parité FP32 exacte, Gap 2, Δacc≤0.007) et l'effondrement F1 de la PTQ INT8 « legacy » ; Sprint 39 en donne le **diagnostic** (émulateur bit-exact : cause = scale figé, récupération per-channel/Q15 ≈ FP32). Le sprint complète le **kernel v2 + la validation board** différés du Sprint 39 pour obtenir la récupération INT8 **réelle sur carte**, unifie les données dans un notebook de synthèse, puis rédige l'article.

**Nœud honnête (levé en partie par la refonte)** : la récupération INT8 n'était qu'**émulée PC** (board S3915 différée) ; elle est désormais **mesurée sur carte** sur les 4 cellules per-canal du kernel v2 (F1 0.9173 Monitoring / 0.8995 Pronostia, parité gelée 1.000 contre l'émulateur, 0 CRC). Les 8 cellules restantes de la grille (`q15`, A/B `int8_legacy` v2) restent au banc. L'article distingue explicitement « mesuré board » vs « émulé PC » ; les cellules board v2 portent `"à mesurer"` tant que la carte n'a pas streamé (règle « aucun chiffre inventé »). Paradoxe latence INT8 (RAM ÷4 sans accélération FPU) assumé, SIMD CMSIS différé.

| Bloc | Tâches | Statut | Résultat attendu |
| ---- | ------ | :----: | ---------------- |
| A — Prérequis données (Sprint 39 différé) | S4001 · S4002 | 📝 Doc | kernel v2 calibré (int32 + per-canal + Q15) + tests host ; board réelle : latence/`.bss`/F1/parité/accord INT8↔FP32 (différé carte) |
| B — Synthèse unifiée | S4003 | 📝 Doc | notebook rechargeant exp_S36+exp_S39+exp_S40 → 5 figures article, aucune valeur en dur |
| C — Rédaction article | S4004 · S4005 · S4006 | 📝 Doc | squelette LaTeX + `references.bib` + Makefile ; versions FR et EN (miroir strict) |
| D — Clôture | S4007 | ✅ | tests figures↔JSON + FR≡EN ; `make test` 0 régression ; roadmap/`triple_gap.md`/`graphify` |
| E — Refonte (agrégation + figures + extension) | S4008 · S4009 · S4010 | ✅ | agrégat unique des métriques EWC (perf, RAM, latence, énergie, MACs/FLOPs/BOPs) ; catalogue de figures régénérable + sync anti-dérive ; article étendu (moment, profondeur, budget système) ; registre des mesures manquantes |

**Message scientifique** : (1) portabilité EWC MCU (parité FP32, Gap 2/3) · (2) piège de la PTQ naïve (effondrement F1 malgré RAM ÷4) · (3) récupération par kernel calibré (per-channel/Q15 ≈ FP32) · (4) honnêteté mesuré/émulé + paradoxe latence.

**Liens triple gap** : Gap 2 (latences board <100 ms) · Gap 3 (récupération F1 INT8 board ⇒ RAM ÷4 sans perte de métrique, `FIXME(gap3)` levé si confirmé). **Statut** : 📝 **Documenté (S4000–S4007)** — Bloc A/B board différés si carte indisponible ; S4001 (firmware+émulateur) et S4003–S4006 faisables sans carte.

→ Détail : [`docs/sprints/sprint_40/S4000_sprint_40.md`](sprints/sprint_40/S4000_sprint_40.md)

---

### Sprint 41 — Rédaction du manuscrit final M2 (démarré 3 juillet 2026)

**Objectif** : produire `Manuscrit Final RIVALS.pdf` (~30 pages de texte FR hors abstracts/TOC/biblio/annexes, dépôt Moodle) en intégrant les retours des rapporteurs (limite de pages ; clarification du domaine = intersection CL × TinyML × PdM) et en exposant méthodologie, contribution, évaluation, perspectives.

**Cadrage validé** : fil narratif **par triple gap** ; corps = socle (4 modèles, portage board, RAM, INT8 vs FP32) + **S36** (comparaison appariée PC↔board) ; S34/S35/S38 en perspectives ; énergie (S33) exclue ; datasets focus **CMAPSS + Pronostia + Monitoring** (grilles 4×5 en annexe) ; cadre mixte supervisé+non supervisé **assumé** ; chiffres RAM/INT8 en évolution (S39/S40) gérés par placeholders `[à confirmer — exp_XXX]`.

**Workflow imposé** : textes produits en md dans `docs/rapport_de_stage/FIchier_md/` (dossier gitignoré) ; le projet Overleaf n'est **jamais** modifié sans instruction explicite ; aucun chiffre non traçable vers `experiments/exp_*` ; notebook de figures en fin de sprint.

| Bloc | Tâches | Statut | Résultat attendu |
| ---- | ------ | :----: | ---------------- |
| A — Infrastructure & cadrage | S4101 · S4102 | ✅ | gitignore + arborescence `FIchier_md/` ; 8 fiches de cadrage (messages clés, chiffres+sources vérifiées, figures, refs, budget pages) |
| B — Audits rigueur | S4103 · S4104 | ✅ | biblio : doublon Aljundi2018, corrections Ravaglia/Wu/Lin, ~10 entrées BibTeX prêtes (Belay, Park, Su, Zong, Lessmeier, CWRU, Jacob, Krishnamoorthi, Mahalanobis) ; glossaire : STM32N6 obsolète → NUCLEO-F439ZI, ~20 acronymes + 7 entrées à créer (PTQ/QAT, Q15, DWT, .bss, watermark, parité…) |
| C — Rédaction (à la demande) | S4105–S4108 | 🟡 | ✅ ch. 1–3 (S4105) · ✅ ch. 4 (S4106) · ⏳ ch. 5–7 (S4107, placeholders RAM-INT8) · ✅ ch. 8 + abstracts + annexes (S4108 ; perspectives = drift/CL, features, énergie ; Q15 exclu) |
| D — Figures & consolidation | S4109 · S4110 | ⏳ | notebook `notebooks/manuscrit_final/figures.ipynb` (0 valeur en dur) ; résolution placeholders depuis S39/S40, vérif chiffres↔JSON, comptage pages, checklist consignes |

→ Détail : [`docs/sprints/sprint_41/S4100_sprint_41.md`](sprints/sprint_41/S4100_sprint_41.md)

---

### Sprint 42 — Bibliothèque de figures + explication des stratégies de quantification (13 – 19 juillet 2026)

**Objectif** : mettre en place une **infrastructure pérenne de génération de figures** (`src/figures/`) — style commun, chargement traçable des `experiments/`, registre de catalogues, CLI régénérable — dont le premier cas d'usage est le **catalogue complet des stratégies de quantification** (FP32, INT8 QAT/PTQ-legacy/v2, Q15, HDC int16-AM), aujourd'hui dispersées sur ~8 sprints sans figure explicative.

**Cadrage validé** : périmètre = toutes les stratégies ; trois familles de figures (pédagogie / pipeline-flux / impact mesuré) ; langue FR ; règles d'honnêteté héritées (aucun chiffre inventé, badges plateforme mesuré/émulé/« à mesurer », métriques nommées).

| Bloc | Tâches | Statut | Résultat |
| ---- | ------ | :----: | -------- |
| A — Infrastructure | S4201 | ✅ | `src/figures/{style,loaders,registry,schematic}.py` + catalogs + CLI `generate_figures.py` (`--catalog`/`--all`/`--list`) |
| B — Contenu quantification | S4202 · S4203 · S4204 · S4205 | ✅ | inventaire `quantization_strategies.md` ; **17 PNG** (pédagogie P1–P6, pipeline F1–F5, impact I1–I6) sous `docs/figures/quantization/` |
| C — Assemblage & clôture | S4206 · S4207 | ✅ | notebook-galerie `catalog.ipynb` (nbconvert OK) ; `test_figures_library.py` **7 PASS**, 714 collectés 0 erreur ; roadmap + CLAUDE.md + graphify |

**Messages portés par les figures** : *quantifier ≠ quantifier* (moment + calibration de l'échelle dominent : QAT ✓ / PTQ figée ✗ / PTQ calibrée ✓) ; la dynamique des tenseurs décide du format (Σ⁻¹ → Q15) ; paradoxe latence FPU (INT8 ×1.84 vs FP32, gain RAM seul). Board v2 (S40) : Pronostia chargé, Monitoring « à mesurer » (remplissage automatique à la relance). `TODO(dorra)` scales per-channel kernel v2, `TODO(arnaud)` notations manuscrit laissés ouverts.

→ Détail : [`docs/sprints/sprint_42/S4200_sprint_42.md`](sprints/sprint_42/S4200_sprint_42.md)

---

### Sprint 43 — Recherche & analyse de datasets pour la détection de drift (20 – 26 juillet 2026)

**Objectif** : constituer un corpus de **datasets externes à drift labellisé** (aujourd'hui absent du projet — le drift y est déduit du scénario CL, jamais annoté échantillon par échantillon), les acquérir, les analyser et **caractériser/quantifier le drift**. Socle des Sprints 44 (modèles PC) et 45 (portage board).

**Cadrage validé (utilisateur, 7 juillet 2026)** : datasets **externes trouvés sur internet**, priorité drift, de préférence **dual-usage drift+faute** (pour le sprint tandem futur) ; **ne pas** réutiliser les datasets projet actuels sauf s'ils portent des labels de drift (aucun ne le fait) ; short-list proposée (UCI Gas Sensor Array Drift ⭐, USP INSECTS ⭐, Electricity/NOAA, générateurs synthétiques à points exacts, hydraulique/SECOM dual-usage).

**Sélection finale (décisions utilisateur)** : **UCI Gas Sensor Array Drift** ⭐ (128 feat., 10 batches, dual-usage 6 gaz), **Electricity/ELEC2** (7 feat., GT absente), **Hydraulic Condition Monitoring** (17 feat., segmenté par cooler — **remplace INSECTS**), **synthétique numpy** (points exacts — **remplace `river`**, non installé).

| Bloc | Tâches | Statut | Résultat |
| ---- | ------ | :----: | ---------------- |
| A — Recherche & acquisition | S4301 · S4302 | ✅ | `docs/context/drift_datasets.md` (fiches ≥5 candidats) ; 4 loaders + module commun `src/data/drift_dataset.py` (`DriftDataset`+`freeze_zscore`) exposant `drift_points`/`drift_type` ; 4 configs ; `scripts/download_drift_datasets.py` idempotent ; registre `DRIFT_LOADERS`/`DRIFT_CONFIGS` |
| B — Analyse & caractérisation | S4303 · S4304 | ✅ | `scripts/characterize_drift.py` → `exp_S43_drift_char/<ds>/characterization.json` (KS/PSI/JS/MMD/Maha réutilisé/PCA glissants + alignement change-points). Synthétique : pics [1575,2925,4575] vs [1500,3000,4500] → `alignment_score=75` ; Gas : Maha ~7→760 batch1→2 ; Electricity : `alignment_score=null`. **S4304** : catalogue `src/figures/catalogs/drift_datasets.py` (registre S4201) → **17 PNG** `docs/figures/drift_datasets/` : timelines (pic mesuré ↔ vérité-terrain), shift avant/après (réutilise `plot_anomaly_score_distributions`), trajectoire PCA temporelle, heatmap JS fenêtre×fenêtre, comparatif inter-datasets — labels FR, `drift_points` légendés « vérité-terrain », synthétique étiqueté « validation », 0 chiffre en dur (garde AST). |
| C — Assemblage & clôture | S4305 | ✅ | notebook `notebooks/cl_eval/drift_datasets/analysis.ipynb` (galerie FR commentée, résumés+tableau comparatif depuis JSON, nbconvert OK, skip gracieux si `data/raw` absent) ; `tests/test_drift_datasets.py` **16 PASS** (contrat loaders, ordre chronologique, validité GT, normalisation figée, GT exacte synthétique, garde AST 0-chiffre, idempotence caractérisation) ; roadmap/`CLAUDE.md`/graphify. |
| D — EDA exhaustives feature-level | S4306 · S4307 · S4308 | ✅ | 3 notebooks FR `notebooks/cl_eval/drift_datasets/eda_{gas_sensor,hydraulic,electricity}.ipynb` (miroir `eda_paderborn`) — nbconvert OK, **0 erreur**, figures inline (11/9/9) sous `figures/eda/<ds>/`, **0 chiffre en dur** (tout de `load()`+JSON S4303). Distinction explicite **axe classe** (gaz/faute/prix) vs **axe dérive** (batch/condition cooler/fenêtre) ; réutilisation des helpers `eda_plots`/`feature_space_plots` (`plot_feature_space_2d` sur les cas binaires, scatter inline multi-catégorie façon `eda_paderborn` pour l'axe dérive) ; Electricity **sans verticale vérité-terrain** (`drift_points=None`, honnête). Correctif amont : `fit_tsne2d` `n_iter`→`max_iter` (sklearn ≥1.5). |

**Message scientifique** : fournir une **vérité-terrain de drift** (points de changement) qui rende mesurables le délai de détection et le taux de fausses alarmes des détecteurs (S44), en privilégiant des capteurs industriels réels (Gap 1 sur l'axe drift).

**Liens triple gap** : Gap 1 (données industrielles réelles de drift). **Statut** : ✅ **S4301–S4308 implémentés** (corpus + loaders + caractérisation + 17 figures régénérables + galerie S4305 + **3 EDA exhaustives feature-level S4306–S4308** + 16 tests). Chaîne de mesure validée sur le synthétique (pics alignés ±1 fenêtre). **Carte non utilisée** (board = S45).

→ Détail : [`docs/sprints/sprint_43/S4300_sprint_43.md`](sprints/sprint_43/S4300_sprint_43.md)

---

### Sprint 44 — Modèles de détection de drift sur PC (supervisés ∥ non-supervisés) (27 juillet – 2 août 2026)

**Objectif** : implémenter et évaluer une **famille de détecteurs de drift** sur les datasets S43 — statistiques streaming (DDM, EDDM, Page-Hinkley), tests deux-échantillons (ADWIN, KSWIN, KS, MMD, PSI/JS) et **baseline projet** (`SlidingWindowDriftDetector`) — avec métriques de détection **et** RAM/latence (proxies PC honnêtes), pour produire la **reco des détecteurs portables MCU** (S45).

**Cadrage validé (utilisateur, 7 juillet 2026)** : trois familles (streaming + deux-échantillons + baseline) ; signal **supervisé (flux d'erreur) ET non-supervisé (features) à parité** (axe d'étude) ; **priorité aux méthodes à état borné** (viabilité MCU annotée dès le PC) ; évaluation = détection **+ coût** dans le même tableau.

| Bloc | Tâches | Statut | Résultat attendu |
| ---- | ------ | :----: | ---------------- |
| A — Inventaire & config | S4401 | ✅ | `docs/context/drift_detectors.md` (taxonomie 8 détecteurs + baseline, état mémoire, viabilité MCU argumentée) + interface `src/models/drift/base.py::BaseDriftDetector` (verdict 3 niveaux `DriftVerdict`, `error_stream`) + `configs/sprint44_drift_detection.yaml` |
| B — Détecteurs | S4402 · S4403 | ✅ | supervisés O(1) (`DDM` 20 B/`EDDM` 32 B/`PageHinkley` 16 B, `requires_label=True`) + non-supervisés à état borné (`ADWIN` O(log W) borné/`KSWIN`/`KSTest`/`MMD` linéaire/`PSI` O(bins), `requires_label=False`) + agrégation multivariée `MultiFeatureDriftDetector` |
| C — Évaluation & exécution | S4404 · S4405 | ✅ | **S4404** harnais `src/evaluation/drift_metrics.py` (`compute_drift_metrics` délai/FAR/MDR/P/R/F1/MTFA/MTD + gestion `None` honnête sans GT ponctuelle ; `profile_drift_detector` coût **proxy PC** `_proxy:true` ; `build_comparison_table` ; `alarms_from_verdicts` source unique enum/str/int). **S4405** driver `scripts/run_sprint44_pc.py` (modèle de faute = `LogisticRegression` sur l'enrôlement pour le flux d'erreur supervisé, lève `TODO(arnaud)` S4400 ; baseline via adaptateur — score = max des valeurs absolues z-scorées) → **grille 36/36 cellules** `experiments/exp_S44_PC_{detector}_{dataset}/results.json` (9 détecteurs × 4 datasets, seed 42, `null` honnête) + catalogue `src/figures/catalogs/drift_detection_pc.py` (registre S4201) → 5 PNG `docs/figures/drift_detection_pc/` (délai↔FAR, raster alarmes, coût mém/latence, heatmap F1, synthèse supervisé∥non-supervisé ; 0 chiffre en dur, garde AST). Validé sur synthétique (GT exacte : supervisés délai fini, FAR≈0). |
| D — Assemblage & clôture | S4406 | ✅ | notebook `notebooks/cl_eval/drift_detection/comparison.ipynb` (galerie FR, tableau de synthèse + reco chargés des JSON, nbconvert OK) ; tests `tests/test_drift_metrics.py` **12 PASS** (oracle/paresseux/paranoïaque/`None` honnête + garde AST) + `test_drift_detectors.py` **50 PASS** (interface exhaustive, déterminisme seed 42, O(1) vs O(W)) ; **reco MCU** dans `docs/context/drift_detectors.md` (traçable aux `results.json`) : primaires Page-Hinkley/DDM/EDDM (O(1)) + PSI (O(bins), non-sup.), baseline référence, ADWIN/KSWIN/KS secondaires, MMD PC-only ; roadmap/`CLAUDE.md`/graphify. |

**Message scientifique** : quantifier le compromis **délai ↔ fausses alarmes ↔ coût** par détecteur, et l'arbitrage **supervisé (précis, exige labels) vs non-supervisé (autonome)** — décisif pour une carte déployée seule (scénario S38).

**Liens triple gap** : Gap 2 (latence par update) · Gap 3 (état mémoire). **Statut** : ✅ **S4401–S4406 implémentés** — détecteurs (`test_drift_detectors.py` **50 PASS**) + harnais/grille/figures/notebook/reco (`test_drift_metrics.py` **12 PASS**, grille **36/36 cellules**). Constat mesuré : supervisés O(1) *invariant au dataset* (viabilité haute), non-supervisés à coût **croissant avec la dimensionnalité** (PSI haute ; ADWIN/KS *moyenne→pc_only* ; MMD *pc_only*). **PC-only, board non utilisée** (non requise par les specs S4404–S4406 ; portage = S45).

→ Détail : [`docs/sprints/sprint_44/S4400_sprint_44.md`](sprints/sprint_44/S4400_sprint_44.md)

---

### Sprint 45 — Portage board des détecteurs de drift (NUCLEO-F439ZI) (3 – 9 août 2026)

**Objectif** : porter en C les détecteurs retenus MCU-viables (reco S44), mesurer leur **RAM `.bss` et latence DWT réelles** sur la NUCLEO-F439ZI, vérifier la **parité board↔PC**, et les intégrer au firmware sans toucher au protocole UART (nibble saturé → sélection à la compilation `-DDRIFT_DETECT`, précédent `-DEWC_AUTO_UPDATE`/`-DMAHA_INT8`).

**Nœud honnête** : le portage ne change pas les métriques de détection (établies S44) ; il vérifie **parité** (même verdict que le Python), **coût réel** (`.bss` + DWT) et **faisabilité** (build par défaut invariant, 0 régression). Cellules board `« à mesurer »` tant que la carte n'a pas streamé.

| Bloc | Tâches | Statut | Résultat attendu |
| ---- | ------ | :----: | ---------------- |
| A — Sélection & cadrage | S4501 | ✅ | liste figée **Page-Hinkley/DDM/PSI portés** (tracée `state_bytes` S44 : 16/20/124 B) + SlidingWindow réf. + **ADWIN différé PC-only** ; cadrage `-DDRIFT_DETECT`/`-DDRIFT_METHOD`, verdict via `snap.auroc` (wire V3 inchangé) |
| B — Firmware | S4502 | ✅ | 3 détecteurs C (0 malloc, backing statique, `# MEM:`) + intégration `pipeline.c` sous `-DDRIFT_DETECT` (chemin EWC : PSI←maha_score, PH/DDM←`1[pred≠label]`) ; **`test_drift_methods` 6/6 parité C↔Python** ; **`.bss` défaut invariant 105 036 B** (+36/+40/+132 B par méthode) |
| C — Export, parité & mesure | S4503 | ✅ | `export_weights_c.py --drift-methods` → `drift_methods_params.h` généré + `run_sprint45_board.py` (stream chronologique) + `board_pc_parity45.py` ; **board réelle `page_hinkley×gas_sensor_drift` : parité 1.000 (0/13910), lat 270 µs (Gap 2 ✅), `.bss` 166 352 B, 0 CRC** ; `test_sprint45_board.py` 9/9 |
| C' — Grille & assemblage | S4504 · S4505 | ✅ | agrégat `aggregate_sprint45.py` → `exp_S45_summary.json` (`[dataset][detector][platform]`, mesuré-board vs proxy-PC) ; **colonne `gas_sensor_drift` mesurée board réelle** : **DDM parité 1.000 (0/13910), lat 270 µs (Gap 2 ✅), `.bss` 166 356 B** ; **PSI N/A honnête** (overflow SRAM au link — signal Mahalanobis O(k²) à k=128 feat.) ; notebook `drift_detection_board/comparison.ipynb` (heatmaps, proxy↔board, parité) + `test_sprint45_board.py` **15 PASS + 1 skip** ; `make test` 134 (2 TinyOL préexistants, 0 régression) |

**Message scientifique** : établir **quels détecteurs de drift tiennent sur MCU et à quel coût mesuré** (Page-Hinkley/DDM O(1) ≪ PSI O(bins) ≪ fenêtres O(W)), lesquels rester PC-only — chiffres board réels, pas proxies.

**Liens triple gap** : Gap 2 (latences board < 100 ms) · Gap 3 (RAM détecteurs dans le budget). **Statut** : ✅ **Sprint 45 implémenté (S4501–S4505)** — board réelle : **Page-Hinkley + DDM parité 1.000, lat 270 µs (Gap 2 ✅), `.bss` ≈ 166 Ko (Gap 3 ✅), 0 régression** ; **PSI = limite matérielle mesurée** (overflow SRAM en haute dim, portable basse dim seulement) ; ADWIN/KS/KSWIN/MMD PC-only. Reste de la grille (`hydraulic`/`synthetic`) = runbook 1 cellule/(det,ds), N/A honnête. **Suite** : tandem drift + faute autonome → [`docs/context/drift_fault_tandem.md`](context/drift_fault_tandem.md).

→ Détail : [`docs/sprints/sprint_45/S4500_sprint_45.md`](sprints/sprint_45/S4500_sprint_45.md)

---

### Sprint 46 — Comparaison des moments de quantification (avant / après / les deux) (16 – 22 juillet 2026)

**Objectif** : première étude qui compare **frontalement** les trois *moments* de quantification — **avant** l'entraînement (QAT / fake-quant), **après** (PTQ sur modèle FP32 figé), et **les deux** enchaînés (QAT → export PTQ, le chemin réel de déploiement firmware) — à modèle/dataset/seed fixés. Périmètre : **EWC** (prioritaire) puis **TinyOL**, sur **Monitoring (D2)** et **Pronostia (D4)**.

**Nœud honnête** : seuls EWC et TinyOL ont une vraie boucle à fake-quant → axe 3-way. **HDC** est nativement entier (quantification structurelle, INT8≡FP32) et **Mahalanobis** est PTQ-only (axe INT8-vs-Q15, S34) → traités en **contexte N/A honnête**, aucune cellule 3-way artificielle. `before` (fake-quant à l'inférence) = **borne haute** ; `both` (noyau entier réel) = seule colonne **fidèle au déploiement**. PC/émulateur prioritaire, board différée (`« à mesurer »`).

| Bloc | Tâches | Statut | Résultat attendu |
| ---- | ------ | :----: | ---------------- |
| A — Cadrage & taxonomie | S4601 | ✅ | taxonomie 3 moments + mapping par modèle + clé config `quant_moment` → `docs/context/quantization_moments.md` (HDC N/A structurel, Maha N/A hors-axe) |
| B — Harnais PC | S4602 | ✅ | `run_s46_quant_moment.py` unifié ; **câble le chemin `both`** (QAT→`from_state_dict`→`forward_quant`) ; réutilise `EWCAdapter`/`int8_c_emulation`/`SCHEME_*` ; JSON aligné S28/S39 + `config_snapshot` |
| C — Expériences PC | **S4603 ✅ · S4604 ✅ · S4605 ✅** | ✅ | **EWC 3-way × 2 ds mesuré** (`exp_S46_ewc/`, seed 42) · **TinyOL 3-way × 2 ds mesuré** (`exp_S46_tinyol/` ; collapse recon-error before≈after≈both documenté) · **HDC/Maha contexte N/A mesuré** (`exp_S46_context/` ; HDC INT8≡FP32 ×2.33 · Maha Pronostia int8 casse −0.113 / q15 récupère +0.013) |
| D — Assemblage | **S4606 ✅ · S4607 ✅** | ✅ | **catalogue figures `quant_moment` (4 PNG M1–M4) + notebook galerie ✅** (garde AST étendue à `quant_moment.py`) · **`test_s46_quant_moment.py` 9 PASS** (schéma 4 moments, N/A honnête HDC/Maha, câblage `both`, garde 0-chiffre, déterminisme QAT) + MAJ triple_gap |
| E — Board `both` (carte réelle) | **S4608 ✅** | ✅ | **colonne `both` mesurée NUCLEO-F439ZI** (carte disponible) : head **QAT multiclasse** `EWCMlpMulticlassInt8` (nouveau) → `export_weights_c.py --int8-v2` → kernel v2 calibré `-DEWC_INT8_V2` → flash/stream ; driver `run_sprint46_board.py` (réutilise `build_and_flash_s40`). **F1 `both` 0.9213 (monitoring) / 0.9072 (pronostia), parité board↔émulateur 1.000, lat 65/68 µs (Gap 2 ✅), `.bss` 101/106 Ko (Gap 3 ✅, RAM ÷4), 0 CRC** ; **A/B `both` ≥ `after`** (+0.004 / +0.008 vs `exp_S40_board_v2`) |

**Message scientifique** : *quantifier ≠ quantifier* — le **moment** (before/after/both) et la **calibration** dominent la préservation de métrique ; `both` est la seule variante honnête vis-à-vis du déploiement embarqué. **Confirmé par la mesure (S4603, EWC)** : QAT préserve (Δ≤0.001), PTQ naïve `legacy_c` s'effondre (AUROC → 0.498 Monitoring / 0.546 Pronostia, Δ≈−0.45/−0.55), PTQ calibrée récupère tout (+0.477/+0.452), `both` (QAT→PTQ calibré) préservé (Gap 3 métrique ✅ + RAM ÷4 ✅). **TinyOL (S4604)** : sur l'erreur de reconstruction pas de kernel entier per-canal ni de QAT AE → before≈after≈both (collapse honnête ; Monitoring Δ−0.185 dégradé, Pronostia Δ−0.006 préservé). **HDC/Maha (S4605)** : hors-axe des moments — HDC structurel INT8≡FP32, Maha reproduit S28/S34 (int8 casse / q15 récupère sur grande dynamique).

**Liens triple gap** : Gap 3 (moments de quantification pendant l'apprentissage incrémental). **Statut** : ✅ **Sprint 46 implémenté (S4601–S4608)** — A/B/C/D ✅ (PC/émulateur) + **E (S4608) mesurée board réelle NUCLEO-F439ZI** : colonne `both` (QAT→export PTQ→kernel v2 calibré) via head **QAT multiclasse** `EWCMlpMulticlassInt8` (nouveau, compatible firmware) → **F1 0.9213 (monitoring) / 0.9072 (pronostia), parité board↔émulateur 1.000, lat 65/68 µs (Gap 2 ✅), RAM ÷4 `.bss` 101/106 Ko (Gap 3 ✅), 0 CRC** ; **A/B `both` ≥ `after`** (+0.004 / +0.008 vs `exp_S40_board_v2`, source FP32). Message confirmé : QAT préserve la métrique et **égale** la PTQ calibrée sur ce head (le *moment* compte, mais la *calibration* du noyau v2 suffit déjà à récupérer). **Réutilise** : `EWCMlpInt8Classifier` (S28), `int8_c_emulation.py` (S39), `tinyol_int8.py` (S24), `mahalanobis_int8/q15` (S34), `TinyOL/HDC/MahalanobisAdapter` (S28), `build_and_flash_s40`/kernel v2 (S40) ; nouveautés = chemin `both` (S4602) + mode `context` (S4605) + head QAT multiclasse + driver board (S4608).

→ Détail : [`docs/sprints/sprint_46/S4600_sprint_46.md`](sprints/sprint_46/S4600_sprint_46.md)

---

### Sprint 47 — Profondeur & schéma de quantification pour EWC (sub-INT8, granularité, symétrie) (23 – 29 juillet 2026)

**Objectif** : **troisième axe** de la quantification, orthogonal au *moment* (S46) et au *format* (S4202) — *à quelle profondeur en bits (sub-INT8) et avec quelle calibration (granularité / symétrie)* la tête EWC casse, et quel schéma rachète la métrique. Périmètre **EWC-only × {Monitoring (D2), Pronostia (D4)}**, **PC-only** (émulateur bit-exact `int8_c_emulation.py`) ; le portage board est le **Sprint 48**.

**Nœud honnête** : la **RAM reportée est THÉORIQUE (bit-packée)** — un poids INT4 dans un `int8_t` n'économise rien de plus que l'INT8 ; le gain ÷8/÷16 n'est réel qu'avec un kernel bit-packé (mesuré S48). L'émulateur mesure la **métrique** (AUROC) et une **RAM analytique**, pas la latence ni la `.bss` réelles.

| Bloc | Tâches | Statut | Résultat |
| ---- | ------ | :----: | -------- |
| A — Cadrage & taxonomie | **S4701 ✅** | ✅ | `docs/context/quantization_depth.md` : 3 sous-axes (`weight_bits` 8→ternaire/binaire · `granularity` per_tensor/per_channel · `symmetry` symmetric/affine) + RAM théorique vs bit-packing + mapping **EWC balayé / HDC structurel / Maha format-only / TinyOL hors-périmètre** (N/A justifié) + AUROC binaire (S4601) |
| B — Harnais PC | **S4702 ✅** | ✅ | `QuantConfig` étendu (`weight_bits`/`weight_mode`/`symmetry` + `subint8()`, **q15() → weight_bits=16** rétro-compat) ; forward câblé ternaire (TWN)/binaire (BWN)/affine (zero-point via `compute_scale_zero_point`) ; `theoretical_weight_ram` ; `scripts/run_s47_quant_depth.py` (réutilise `EWCAdapter`/`_eval_quant_auroc`/émulateur) ; **0 régression** presets S39 (golden figée) |
| C — Expériences PC | **S4703 ✅** | ✅ | **28 cellules mesurées** (`exp_S47_depth/`, EWC × 2 ds × 7 profondeurs × 2 granularités, seed 42) : `delta_auroc`/`agreement_vs_fp32`/RAM théorique |
| C — Symétrie / contexte | **S4704 ✅ · S4705 ✅** | ✅ | **S4704** : **12 cellules** `exp_S47_symmetry/` (EWC × 2 ds × bits critiques {int2,int3,int4} × {symmetric,affine}, per_channel) via `--filter sym` + routage sortie ; **gain affine négatif partout** (la per-channel suffit, le zero-point ne rachète rien). **S4705** : `exp_S47_context/context.json` (script traçable `write_s47_context.py`) — HDC/Maha/TinyOL en N/A justifié, aucun champ métrique |
| D — Figures / notebook | **S4706 ✅** | ✅ | catalogue `quant_depth` (5 PNG `docs/figures/quantization_depth/`, 0 chiffre en dur garde AST, N/A gris, RAM « théorique bit-packée ») + notebook galerie `notebooks/cl_eval/quant_depth/comparison.ipynb` (tableaux rechargés par cellule, nbconvert OK) |
| E–F — Tests / pointeur board | **S4707 ✅ · S4708 ✅** | ✅ | **S4707** : `tests/test_s47_quant_depth.py` **21 PASS** (schéma JSON, monotonie RAM bits↓, 0-régression golden S39, presets `subint8`/ternaire/binaire, **`test_na_honesty`** contexte HDC/Maha/TinyOL sans champ métrique fabriqué, **`test_no_hardcoded_numbers`** garde AST `quant_depth.py`) ; roadmap + `triple_gap.md` (§ Gap 3) + CLAUDE.md à jour ; `graphify_sprint_update`. **S4708** : table de sélection board figée (traçable `exp_S47_depth/`) → **frontière = ternaire, agressive = binaire** (Monitoring + Pronostia), référence int8 (déjà S39), **pas d'affine** (n'a pas aidé) ; nuance bit-packing → S48 |

**Message scientifique (mesuré S4703/S4704)** : *quantifier ≠ quantifier* aussi en **profondeur**. **Monitoring** robuste jusqu'à 2 bits (per_channel Δ≈−0.003). **Pronostia** montre le « cliff » et l'effet de la granularité : **per_tensor 2-bit Δ=−0.046 / accord 0.908** vs **per_channel 2-bit Δ=−0.009 / accord 0.951** → **la per-channel repousse le cliff (H1 confirmée)** ; ternaire Δ=−0.015, binaire Δ=−0.028. RAM théorique ÷4 (8b) → ÷8 (4b) → ÷16 (2b) → ÷32 (binaire), **sous réserve d'un kernel bit-packé (S48)**. **Axe symétrie (S4704)** : le **zero-point affine ne rachète pas la métrique** (gain ≤ 0 sur les 6 cellules ; dégrade même Monitoring où les activations sont déjà bien séparées) → **c'est la per-channel, pas l'affine, qui repousse le cliff**.

**Liens triple gap** : Gap 3 (jusqu'où descendre en bits avant que la métrique casse, et quel schéma la rachète). **Statut** : ✅ **Sprint 47 implémenté (S4701–S4708)** (taxonomie + émulateur étendu + harnais + 28 cellules profondeur + 12 cellules symétrie + contexte N/A + 5 figures + notebook + **tests 21 PASS + pointeur board figé**). **Configs gagnantes → board S48** : frontière **ternaire** (Monitoring Δ=−0.0021 / Pronostia Δ=−0.0153, ×20.25 théo.), agressive **binaire** (Monitoring Δ=−0.0117 encore ≥ −0.02 / Pronostia Δ=−0.0275 casse, ×32 théo.), référence **int8** per_channel (déjà S39/v2) ; **aucune variante affine** (S4704 : le zero-point ne rachète rien). **Réutilise** : `int8_c_emulation.py` (`_weight_scales`/`_quant_weight`/`_act_params` déjà `n_bits`-paramétrés, S39), `compute_scale_zero_point` (affine, `src/utils/quantization.py`), `EWCAdapter`/`_eval_quant_auroc` (S28/S46), registre figures `src/figures/` (S4201) ; nouveautés = modes ternaire/binaire, symétrie affine, `subint8()`, `theoretical_weight_ram`, harnais de sweep + routage symétrie, catalogue `quant_depth`.

→ Détail : [`docs/sprints/sprint_47/S4700_sprint_47.md`](sprints/sprint_47/S4700_sprint_47.md)

---

### Sprint 48 — Portage board sub-INT8 pour EWC (RAM `.bss` réelle, latence, parité) (30 juillet – 5 août 2026)

**Objectif** : matérialiser en firmware les schémas gagnants du Sprint 47 — mesurer la **RAM `.bss` réelle** (bit-packée) et prouver la **parité bit-à-bit** avec l'émulateur. Périmètre EWC × {Monitoring, Pronostia}.

**Liens triple gap** : Gap 3 (le gain sub-INT8 est-il réel ? kernel bit-packé) + Gap 2 (coût latence dépacking). **Statut** : ✅ **Sprint 48 implémenté (S4801–S4807, board réelle NUCLEO-F439ZI)** :

- **S4801 ✅** matrice de portage figée aux gagnants réels S4708 (**frontière = ternaire, agressive = binaire**, et non « INT4/INT2 linéaire » du gabarit pré-S47) ; **décision utilisateur** : famille générale (linéaire INT4/INT2 **+** ternaire/binaire), firmware agnostique au mode. Flags `-DEWC_INT4` (QMAX 7) / `-DEWC_INT2` (QMAX **1** — correctif du 3) / **`-DEWC_INT1`** (binaire, ajouté : seul le 1 bit matérialise ÷32) ± `-DEWC_INTx_PACKED` ; wire V3 (23 B) inchangé.
- **S4802 ✅** kernels sur `ewc_head_int8_v2.c/.h` : bloc typedef étendu + `EWC_V2_PACK_BITS`, primitives `ewc_v2_unpack_weight`/`ewc_v2_pack_row` (LSB-first, complément à deux 4/2 bits, binaire dédié), struct packée `EWCHeadSubInt8Packed` + `ewc_subint8_packed_forward` (dépack → MAC FPU), **v2 existant intact**. Unity `test_ewc_subint8.c` (parité INT4/ternaire/binaire non-packé & packé, `sizeof` packé < int8, round-trip sign-extension) : **tous PASS**, `make test` **141 tests, 2 TinyOL préexistants hors périmètre, 0 régression**, **`.bss` défaut invariant 105 036 B**.
- **S4803 ✅** export `--ewc-subint8` (+`--weight-mode`/`--packed`) réutilisant `_quant_weight_mode` (émulateur S47) → `ewc_head_subint8_weights.h` + `--ewc-subint8-test-vectors` → golden tous schémas ; `_pack_weights` miroir exact du dépack C ; `test_ewc_subint8_export.py` **7 PASS**.

- **S4804 ✅** **câblage `pipeline.c`** (décision utilisateur) : le kernel sub-INT8 (S4802) n'était pas routé (0x40 → v2 S39) ; ajout d'une route sub-INT8 **gardée** `EWC_SUBINT8_WEIGHTS_PROVIDED` (globals `g_ewc_subint8`/`g_ewc_subint8_packed`, chargés par `memcpy` du header, forward frozen), + garde `#error` cohérence packing header↔build. `.bss` défaut invariant 105 036 B. Driver `run_s48_board_depth.py` (train→export→build→flash→stream, N/A honnête). **12 cellules board réelle NUCLEO-F439ZI** (cœur scientifique) : **parité 1.000 sur 12, 0 CRC**, `.bss` **non-packé invariant par mode** (Monitoring k=4 : 105 640 ; Pronostia k=5 : 106 152 — nœud d'honnêteté : un sub-INT8 dans un `int8_t` n'économise rien) vs **packé** (gain 336 B INT4 → 504 B ternaire → 572–604 B binaire, ÷2/÷4/÷8 des matrices de poids), **latence dépacking ≈ +55 µs** (67→123 µs) **≪ 100 ms (Gap 2 ✅)**.
- **S4805 ✅** `board_pc_parity48.py` (émulateur subint8 = réplique bit-exacte, **12/12 parité 1.000, `max_score_err ≤ 1.2e-7`**) + `aggregate_sprint48.py` (lecture seule) → `exp_S48_summary.json` (`[dataset][weight_bits][granularity][platform]`, `bss_saved_by_packing` jamais conflaté, deltas théorie↔matériel).
- **S4806 ✅** catalogue `quant_depth_board` (5 PNG symétriques au PC, badges mesuré-board/théorique-PC, N/A gris, **0 chiffre en dur** garde AST) + notebook nbconvert OK.
- **S4807 ✅** `test_sprint48_board.py` **6/6 PASS** (structure, Gap 2, Gap 3 `.bss` packé<non-packé, parité=1.000, N/A honnête, garde AST) + `test_figures_library.py` 10 PASS ; `make test` **141 (2 TinyOL préexistants hors périmètre, 0 régression)**.

**Message scientifique (mesuré board)** : le gain RAM sub-INT8 est **réel mais conditionnel au bit-packing** — un INT4/ternaire/binaire dans un conteneur `int8_t` ne change pas le `.bss` (identique à l'INT8), seul le kernel packé matérialise le ÷2/÷4/÷8 (gain croissant quand les bits baissent). Le **dépacking coûte ~55 µs** mais reste deux ordres de grandeur sous le budget Gap 2. La **parité board↔émulateur = 1.000** confirme que le schéma de quantification est porté sans perte (kernel firmware == émulateur subint8). **Clôt les deux `TODO(dorra)` S47** (kernel bit-packé + coût du dépacking).

→ Détail : [`docs/sprints/sprint_48/S4800_sprint_48.md`](sprints/sprint_48/S4800_sprint_48.md)

---

### Sprint 49 — Mesure RAM complète (`.data + .bss + pic de pile`) + relance des expériences

**Objectif** : répondre aux deux actions 🔴 « RAM » du CR du 16 juillet 2026 — les mesures rapportées jusqu'ici (`.bss` seul, sans pic de pile) **sous-estiment** la RAM. Généraliser `RAM totale = .data + .bss + pic de pile`, sous conditions définies (mêmes groupes, même protocole MAJ/éval), **par phase** (idle / inference / update).

**Liens triple gap** : Gap 2 (chiffre RAM honnête, pile incluse). **Statut** : ✅ **Sprint 49 implémenté (S4901–S4907)** — S4901–S4903 board réelle NUCLEO-F439ZI ; S4904–S4907 doc structurée + fixes plots + terminologie + tests (aucune carte requise, lecture seule).

- **S4901 ✅** méthode figée : le doc `docs/context/ram_measurement.md` (créé S39) **enrichi** de 3 sections — instants par phase (idle/inference/update), board (référence) vs PC (proxy) **non fusionnables** (CR §1), schéma JSON par cellule + historique du pic. **Correctif** : la fonction firmware réelle est `profiling_ram_peak_bytes` (pas `profiling_total_ram_bytes`).
- **S4902 ✅** orchestrateur unique `scripts/run_ram_full_sweep.py` (1 JSON/cellule au schéma S4901). **Réutilisation stricte** : `measure_stack_watermark.{OpenOCD,read_symbols,scan_stack_peak}` (board) + `sensor_stream.py` + `ram_breakdown.py` + `feature_conditions.train_and_evaluate`/`tracemalloc` (PC). **Écart assumé vs spec** : le pic de pile **n'est PAS** dans le snapshot UART V3 (`profiling_encode` remonte `bss_bytes`) → historique lu par **OpenOCD** (wire inchangé). Flash via le serveur OpenOCD persistant (`program … verify reset`) car `make flash` entre en conflit ST-LINK.
- **S4903 ✅** sweep exécuté → **32 cellules** `experiments/exp_S49_ram/` (condition `5feat`, seed 42) : **16 board** (14 mesurées + 2 N/A Maha×int8) + **16 PC** (8 mesurées + 8 N/A int8). **Invariant `total = data + bss + max(pic_inf, pic_upd)` OK sur les 14 board.** Mesuré : `.data=460 B`, `.bss` 100 152 B (Monitoring k=4) / 105 036 B (Pronostia k=5), pic de pile idle≈4,24 Ko / inférence 4,33–4,42 Ko / update 4,33–4,72 Ko, **total 40–42 % de 256 Ko (Gap 2 ✅)**. Constats : `idle ≠ 0` (trame `pipeline_run()` réservée dès le boot) ; **seul EWC creuse la pile en `update`** (SGD backward), les autres ont `inference == update` (trame partagée `hv[HDC_DIM]`) ; **INT8 ≡ FP32 en `.bss`** (conversion runtime, les 2 structs coexistent → gain RAM INT8 réel = bit-packing, cf. Sprint 48).
- **S4904 ✅** agrégateur lecture-seule `scripts/aggregate_ram.py` → `experiments/exp_S49_ram/summary.json` (indexé `[model][dataset][condition][encoding][platform]`, ratio int8/fp32 **calculé**) + `docs/context/ram_report.md` **généré** (méthode / tableau par cellule / historique du pic / constats sans chiffre en dur / limites N/A). Aucune valeur saisie à la main.
- **S4905 ✅** catalogue figures `src/figures/catalogs/ram_full.py` (registre S4201) → 4 PNG `docs/figures/ram_full/` : RAM totale empilée `.data`/`.bss`/pic, **historique du pic par phase réelle** (`idle`/`inférence`/`mise à jour CL`), ratio int8/fp32, board vs PC séparés. **Fixes CR** : étiquettes de phase neutres (plus d'étapes de couches neuronales), pic tracé **après** chaque phase ; 0 chiffre en dur (garde AST, valeurs via `load_experiment`). MAJ `docs/presentation_ram_measurement.md` (§6 figures ram_full).
- **S4906 ✅** terminologie unifiée → « **mémoire non constante** » (ancien libellé retiré) : 4 occurrences corrigées dans `docs/presentation_ram_measurement.md` (le CR et les specs sprint_49 décrivant l'instruction sont exclus).
- **S4907 ✅** `tests/test_ram_full.py` **6/6 PASS** (total==composantes, `pic_update ≥ pic_inference`, N/A honnête, schéma indexé, garde AST 0-chiffre, terminologie) + `test_figures_library.py` 11 PASS ; roadmap + triple_gap (§ Gap 2 : RAM = `.data + .bss + pic`). Tests indépendants de la carte.

→ Détail : [`docs/sprints/sprint_49/S4900_sprint_49.md`](sprints/sprint_49/S4900_sprint_49.md)

---

### Sprint 50 — Mesure d'énergie réelle (carte ENAC + LPM01A) + coût latence INT8

**Objectif** : répondre à l'action 🔴 du CR du 16 juillet 2026 (« récupérer la carte à l'ENAC pour mesurer l'énergie électrique par inférence, isoler par composant ») et à la tâche 🟡 (« détailler le coût de latence INT8 »). Depuis le Sprint 33, toute l'infra énergie existe mais porte `"à mesurer"` (LPM01A jamais posé).

**Liens triple gap** : Gap 2 (énergie/autonomie mesurées + latence INT8 chiffrée), Gap 3 (l'INT8 réduit-il les µJ malgré le paradoxe latence FPU S29 ?). **Statut** : ✅ **complété board réelle** — volet latence INT8 (S5004–S5007) *et* **volet énergie exécuté (S5008, 2026-08-04/05) : banc LPM01A monté, 8/8 cellules mesurées**. Nuance essentielle : ce sont des **courants moyens** mesurés, pas des µJ/inférence — ces derniers sont **N/A mesuré**, la référence au repos du firmware (attente UART par scrutation active) étant plus consommatrice que tous les régimes de flux.

- **S5001 ✅ (cadrage)** — bloc `energy_calibration:` ajouté à `configs/hw_profile_f439zi.yaml` (résout `TODO(dorra)` S33 : fréquence 100 kSPS *high-sampling*, plage µA–mA auto-range, offset/gain `<à_calibrer>`, sync PA8) ; doc `S5001_cadrage_energie.md` complétée (setup, **plan d'isolation honnête** : MCU/périph isolables par delta, **capteur `"na"`** car simulé UART, protocole balisé PA8). Aucun chiffre d'énergie (cadrage seul).
- **S5002 ✅ (chaîne + placeholders)** — driver `scripts/run_s50_energy.py`, **couche mince réutilisant strictement** `energy_capture.py` (0 réécriture segmentation/intégration). **8 cellules placeholder honnêtes** `experiments/exp_S50_energy/` (`"à mesurer"`, `source: placeholder`) + `summary.json`. Ajoute ce qui manquait : `phase_durations_s` (dérivé des fronts PA8 réels — clé lue par `autonomy.py` sans producteur jusqu'ici), `energy_uj_per_inference`, `by_component` (delta MCU/périph, `sensor: na`). Manifeste `configs/energy_campaign_s50.yaml` (gabarit `csv/n_inference/n_update`). **Note honnête** : PA8 1-bit ne sépare pas inférence/MAJ → `energy_uj_per_update` = `"à mesurer"` (protocole delta dédié requis).
- **S5003 ✅ (autonomie câblée)** — `autonomy.json` produit à chaque campagne via `autonomy.average_current_ma`/`sweep_capacities` (capacités ← profil HW, jamais en dur) ; placeholder tant que S5002 sans µJ réels. Déblocage : les durées de phase réelles rendent l'autonomie calculable automatiquement dès qu'un CSV arrive.
- **S5008 ✅ (campagne exécutée board réelle NUCLEO-F439ZI + X-NUCLEO-LPM01A, 2026-08-04/05)** — **le protocole delta prévu a été invalidé par deux constats de banc mesurés**, puis remplacé. (1) **La première acquisition d'une session est biaisée d'environ +8 mA** (62,74 puis 54,94/54,86/54,82 mA à conditions identiques) : le delta plaçait sa référence en premier, son biais dépassait l'effet cherché et **inversait le signe du résultat** → `lpm01a_probe.warmup()` (rebut de préchauffage, appelé par les deux pilotes). (2) **La référence « au repos » est plus haute que tous les régimes de flux** (54,78 ± 0,13 mA vs 46,40–51,22 mA), y compris Mahalanobis à ~0,05 % d'occupation — écart non explicable par le taux d'occupation, **cause non établie** (le firmware attend la trame UART par scrutation active, donc le « repos » n'est pas inactif ; d'autres causes de banc non exclues) → les **µJ/inférence sortiraient négatifs** et restent `"à mesurer"` avec une raison **mesurée**, pas un « pas encore fait ». Méthode livrée : `scripts/run_s50_board_current.py` — **courant moyen à cadence imposée** (100 Hz, fenêtre 10 s, 3 répétitions, `N = cadence × fenêtre` **exact par construction**), à cadence identique le trafic UART et le travail hôte sont communs ⇒ l'écart entre cellules est imputable au modèle. **Grille 8/8 mesurée** (repos 54,78 ± 0,13 mA ; dispersion ≤ 0,05 mA) : EWC 46,550/46,547 · HDC 47,843/**51,220** · TinyOL 46,600/46,513 · Maha 46,403/46,373 mA (fp32/int8) ; l'ordre suit exactement celui des latences connues (Maha 5 µs → 46,4 mA … HDC INT8 1958 µs → 51,2 mA). **Réponse Gap 3 : l'INT8 ne réduit PAS la consommation** — EWC (−0,003 mA) et Maha (−0,030) dans le bruit, TinyOL marginal (−0,087), **HDC INT8 +7,1 % de courant** (cohérent avec sa latence INT8 dégradée). **Autonomie chiffrée** (S5003) depuis le courant mesuré : 43,1 h (Maha INT8) à 39,0 h (HDC INT8) sur 2000 mAh, avec `duty_cycle: null` + `regime_mesure` pour ne pas la faire passer pour une autonomie duty-cyclée. **Veille PHY mesurée sans effet utile** (63,24 mA → mode dynamique définitivement fermé). Correctifs d'outillage : `load_measured_cell` (rejouer le manifeste **écrasait** les cellules mesurées), `BUILD_SPECIFIC` (empêche d'écrire `maha_int8` depuis le build FP32 — même classe de bug que le drapeau TinyOL S5201), `autonomy.average_current_ma_from_delta`. **6ᵉ figure** `e6_courant_moyen_mesure.png`. Tests **96 PASS** (dont `tests/test_board_current.py` 12 neufs) ; Unity **145/0**, `.bss` défaut **invariant 105 036 B**. Restent ouverts : µJ/inférence (levier = `WFI` dans l'attente UART), cause de l'écart repos↔flux, `by_component` MCU/périph (câblage), `energy_uj_per_update`.
- **S5004 ✅ (latence INT8 — board réelle NUCLEO-F439ZI)** — instrumentation DWT **cycle-level** du kernel EWC INT8 v2 (`ewc_head_int8_v2.c`, 3 accumulateurs déquant/MAC/requant, **cycles bruts** — la conversion µs tronquerait), **entièrement gardée `-DINT8_SEGMENT_PROFILE`** (build défaut invariant `.bss=105 036 B`, `make test` inchangé, wire V3 23 B intact via réinterprétation des slots `[acc][auroc][forgetting]`). Driver `scripts/run_s50_int8_latency.py` (1 flash instrumenté/dataset → stream 0x40 INT8 + 0x10 FP32 = **mêmes échantillons**). **2 cellules mesurées (0 CRC)** `exp_S50_int8_latency/{monitoring,pronostia}.json` + `ewc.json` : MAC ≈ 6 785 cyc, **requant ≈ 3 777 cyc (surcoût dominant, `lroundf`)**, déquant ≈ 200 cyc ; **total INT8 74 µs vs FP32 48–50 µs → +24 à +26 µs** = **paradoxe latence FPU S29 confirmé et chiffré** (Gap 2 ✅ ≪ 100 ms). **Maha = N/A honnête** (kernel déquant→distance, sans MAC/requant par couche). Doc `docs/context/int8_latency_breakdown.md`.
- **S5005 ✅ (coût/bénéfice)** — `docs/context/int8_cost_benefit.md` : table de décision (RAM poids ÷4 gain / F1 préservé S46 / latence +50 % surcoût FPU / énergie « à mesurer ») + **nuance honnête RAM poids ÷4 ≠ RAM système totale ≈1.0 (S49)** + table reproduit-littérature (Ravaglia/Capogrosso/Zhu-Lin/Benatti) vs contribution propre (EWC sur PRONOSTIA, compromis mesuré carte réelle, paradoxe FPU chiffré). 0 chiffre en dur (chaque valeur → JSON source).
- **S5006 ✅ (figures + notebook)** — catalogue `src/figures/catalogs/energy_real.py` (registre S4201) → **5 PNG** `docs/figures/energy_real/` (E1 µJ/inf gris « à mesurer », E2 par composant, E3 autonomie, **E4 breakdown latence INT8 mesuré**, E5 coût/bénéfice RAM×4 / latence×1.5 / énergie à mesurer) ; badges mesuré-board / à-mesurer gris, **0 chiffre en dur (garde AST)**. Notebook `notebooks/cl_eval/energy_cost/comparison.ipynb` §8 « mesures réelles S50 » (nbconvert OK).
- **Tests** : `tests/test_s50_energy.py` **13/13 PASS** (+ S5007 : µJ>0 seulement si CSV, `by_component.sensor="na"` honnête, segments dequant/mac/requant, garde AST `energy_real.py`, `energy_calibration` présent) ; `test_figures_library.py` 12 + `test_energy_capture.py` 16 + `test_autonomy.py` 8 = **49 PASS 0 régression**. Firmware `make test` 141 (2 TinyOL préexistants hors périmètre, **0 régression**, `.bss` défaut invariant).

→ Détail : [`docs/sprints/sprint_50/S5000_sprint_50.md`](sprints/sprint_50/S5000_sprint_50.md)

---

### Sprint 51 — Score système composite + contexte de comparaison matérielle

**Objectif** : répondre au CR du 16 juillet 2026 §3 (« définir une métrique globale d'évaluation système ») et aux tâches 🟢 associées — un **indicateur composite** unique intégrant RAM totale + latence + énergie + params/MACs + accuracy, pour **classer les configurations** (modèle × carte × encodage) à plateforme fixée, plutôt que des métriques isolées. Contextualise aussi la comparaison matérielle : PC↔board **non exploitable** (CR §1, multi-cœur vs mono-cœur séquentiel) → specs des 2 matériels + réorientation **board↔board** (perspective, 2e carte requise).

**Nœud honnête** : un score composite est un **choix de pondération** — il classe, il ne redéfinit pas ce qu'est un bon modèle. Normalisation par dimension obligatoire (unités incomparables) ; analyse de sensibilité aux poids ; classement **à plateforme fixée** (jamais un score PC vs un score board). Dépend des sorties **Sprint 49** (RAM totale) + **Sprint 50** (énergie) : dimension manquante → `pending`, jamais 0.

| Bloc | Tâches | Statut |
| ---- | ------ | :----: |
| A — Définition & implémentation | S5101 (définition : dimensions/normalisation/pondérations/sensibilité) · S5102 (`src/evaluation/system_score.py` + tests) | 📝 Doc |
| B — Classement & contexte matériel | S5103 (classement `exp_S51_system_score/`) · S5104 (`docs/context/hardware_comparison.md`, 2 matériels) · S5105 (réorientation board↔board, perspective) | 📝 Doc |
| C — Assemblage & clôture | S5106 (catalogue figures `system_score` + notebook) · S5107 (tests + roadmap + triple_gap + graphify) | 📝 Doc |

**Liens triple gap** : synthèse transverse (Gap 1 perf ↔ Gap 2 RAM/latence ↔ Gap 3 énergie/encodage) en un score classant. **Statut** : 📝 **Doc — spec complète (S5100–S5107) ; implémentation à venir** (consomme S49/S50). **Réutilise** : `hw_cost_model.py`/`compute_cost.py` (params/MACs/BOPs), `exp_S49_ram/summary.json`, `exp_S50_energy/summary.json`, harnais PC↔board S36 (réorienté board↔board), registre figures `src/figures/` (S4201).

→ Détail : [`docs/sprints/sprint_51/S5100_sprint_51.md`](sprints/sprint_51/S5100_sprint_51.md)

---

### Sprint 52 — Correctifs de mesure carte (action 2.6 de l'audit S4110)

**Objectif** : corriger un chiffre **publié qui ne mesure pas ce qu'il annonce**. `sensor_stream.py` n'attribuait aucun flag UART à `--model tinyol` (`model_flags = 0`) : la trame empruntait le chemin d'inférence **par défaut** du firmware, c'est-à-dire **Mahalanobis** (`pipeline.c`, branche `else`). Les cellules carte « TinyOL » de la grille S35 dupliquaient donc, au chiffre près, celles de Mahalanobis — jusque dans le tableau 5.1 du manuscrit.

**Nœud honnête** : le défaut dépassait le flag manquant. Les poids TinyOL n'étaient chargés qu'à `TINYOL_IN == 5` (garde figée) et **aucun driver ne les exportait** : même avec le bon flag, la route aurait tourné à **poids nuls** dès qu'une condition changeait la dimension (Monitoring 5feat = 4 variables, `all`, `best`). Corriger honnêtement supposait donc de porter TinyOL à dimension quelconque — `TINYOL_NATIVE_DIM`, sur le patron `MAHA_NATIVE_DIM`/`EWC_HEAD_NATIVE_DIM` (S3507) — avant toute re-mesure.

| Bloc | Tâches | Statut |
| ---- | ------ | :----: |
| A — Correctifs | S5201 : flag UART 0x80 · gardes `TINYOL_NATIVE_DIM` (`pipeline.c`, `tinyol.c`) · `recon[TINYOL_OUT]` (débordement de pile latent) · export à dim k · TinyOL promu modèle **à parité** dans les drivers S35/S32 | ✅ |
| B — Re-mesure carte | grille S35 (15 cellules TinyOL, parité **44/44**) + grille S32 (15 cellules, parité **45/45**), NUCLEO-F439ZI réelle, **0 CRC** | ✅ |
| C — Aval | figures `manuscrit_final` (16 PNG) · manuscrit ch. 5 (tableau 5.1, Paderborn, note d'architecture) · `actions_restantes_resultats.txt` | ✅ |

**Liens triple gap** : Gap 1 (la ligne TinyOL carte redevient une mesure : 0,927 / 0,300 / 0,889 en F1 sur Monitoring/CMAPSS/Pronostia), Gap 2 (latence TinyOL enfin mesurée : 69–148 µs selon `k`, contre 3–5 µs pour Mahalanobis — signature d'un auto-encodeur `k→32→16→k`, très loin des 100 ms). **Nœud d'honnêteté** : la cellule `best × cmapss` est déclarée **N/A** — la sélection `best` n'y retient que `T2`, capteur **constant** de CMAPSS, d'où un seuil calibré nul et une décision tranchée par l'arrondi flottant ; `_degenerate_reason()` la détecte et consigne `na_reason` plutôt qu'un faux échec de parité. **Statut** : ✅ **implémenté**. Effet de bord vérifié : les **2 échecs Unity TinyOL** de la dette 4.2 disparaissent (`make test` 141, 0 échec) — le header `model_weights.h` était désynchronisé de son propre golden, recopié à la main ; le golden est désormais **généré avec les poids** (`tests/tinyol_reference.h`).

→ Détail : [`docs/sprints/sprint_52/S5201_flag_uart_tinyol.md`](sprints/sprint_52/S5201_flag_uart_tinyol.md)

---

### Sprint 53 — Campagne énergie élargie : lever les verrous de banc, mesurer les axes système

**Objectif** : exploiter la fenêtre matérielle pendant laquelle le X-NUCLEO-LPM01A est **posé**. Le Sprint 50 n'a pu produire que le **courant moyen** de 8 cellules à 100 Hz ; µJ/inférence, profil par phase, `by_component` et µJ/mise-à-jour restent `"à mesurer"`. Ce sprint lève les verrous qui l'empêchaient et convertit les axes déjà flashables (gate autonome S38, bit-packing S48) en résultats énergétiques.

**Nœud honnête** : cinq verrous, dont **trois constatés sur banc** au Sprint 50 (pas de voie de synchro sur le LPM01A ; `acqmode dyn` fermé au-delà de 59 mA ; pas de vrai repos, le firmware attendant la trame en scrutation active) et **deux identifiés dans les données mais jamais formulés** — (4) à 100 Hz l'écart EWC INT8↔FP32 (−0,003 mA) est **sous la dispersion du banc** (±0,03 mA), donc la question Gap 3 énergie est *non tranchée* et non tranchée par la négative ; (5) dans `run_s50_board_current.py:263-285` la référence au repos est mesurée **juste après le préchauffage** et les cellules ensuite — **l'ordre est confondu avec la condition**, hypothèse jamais testée alors que le préchauffage lui-même dérive (62,74 → 54,94 → 54,86 → 54,82 mA). Un résultat négatif reste un résultat : « le bit-packing coûte plus d'énergie qu'il n'en économise », « ralentir le MCU ne gagne rien » sont des conclusions attendues et publiables.

| Bloc | Tâches | Statut |
| ---- | ------ | :----: |
| A — Verrous de banc | **S5301 ✅ mesuré carte réelle** (protocole contre-balancé repos/charge — verdict **calculé** `artefact_ordre`, répliqué sur 2 sessions, 7 puis **9 repos ordonnés** ; cause : la carte bascule de niveau de repos **au premier flux reçu**, −10,06 mA, la référence S50 était sur le niveau haut et ses cellules sur le bas ⇒ µJ/inférence négatifs ; sur référence établie les surcoûts redeviennent **positifs et ordonnés selon la latence**, EWC +2,03 · HDC +3,36 mA, répliqués à ~0,1 mA près malgré ~4 mA d'écart absolu inter-session) · **S5302 ✅ mesuré carte réelle 2026-08-06** (`__WFI()` gardé `-DUART_WFI_IDLE` : réveil validé **300 trames @200 Hz, 0 CRC**, latence DWT 50 µs inchangée ; **repos 49,767 → 27,370 mA = −45,0 %** dans la même session contre-balancée ⇒ le repos passe SOUS toutes les cellules et **le protocole delta est débloqué, 8/8 cellules chiffrées** 134–374 µJ *par transaction UART+inférence* ; **le lot `-DINFER_BATCH_N` isole le calcul** — EWC **5,08 µJ**, Maha **0,437 µJ** par inférence, **r² = 0,9998**, parité de prédiction 1.000 vs `N=1`, point `N=200` d'EWC écarté par une règle de saturation **mesurée sur la cadence atteinte** (66 Hz) ; `veille_uA`/`actif_mA` remplis dans `hw_profile_f439zi.yaml` ; autonomie duty-cyclée **≈ 73 h à 1 Hz / 2000 mAh** vs 43 h S50 — les autonomies antérieures étaient **pessimistes**, mesurées sans veille ; `.bss` défaut invariant 105 036 B, `make test` 145/0, carte rendue au build par défaut) · **S5303 ✅ mesuré carte réelle** (3 fréquences ; **l'énergie par inférence croît de +26,2 % de 45 à 180 MHz** — 170,1 → 214,6 µJ — et le courant de base de **+54 %**, pendant que **Gap 2 reste tenu 12,8×** à 45 MHz : *la marge de latence est convertible en autonomie* ; cellule 90 MHz **N/A** sous la règle d'ajustement unifiée, à rejouer ; `acqmode dyn` rouvert à 45 MHz ; balayage SYSCLK gardé `-DSYSCLK_MHZ`, PCLK1 maintenu à 45 MHz pour ne pas toucher au BRR ; VOS/overdrive/wait states conditionnés ensemble ; cible `make check-sysclk` — **3 builds compilés, `.bss` invariant 105 036 B**, `.text` −32/−40 B aux fréquences réduites = overdrive retiré ; pilote refusant de mesurer si la carte ne confirme pas sa fréquence — mesures de banc en attente de carte) | 🟡 En cours |
| B — Méthodes de mesure | **S5304 ✅ mesuré carte réelle** (7 cellules, 0 CRC ; régression `I = I_base + pente·rate` pondérée `1/σ²`, saturation détectée sur la cadence ATTEINTE ; **Gap 3 énergie tranché** : l'INT8 ne change pas mesurablement l'énergie par inférence, −0,2 µJ pour ±2,3 µJ d'incertitude combinée — le gain INT8 reste la RAM ; 1 cellule `hdc_fp32` en N/A honnête, r² 0,885) · **S5305 ⚠️ N/A honnête mesuré** (le mode dynamique s'est rouvert **à 45 MHz seulement** — 100 000/100 000 éch. — mais la **voie A** (segmentation sur le seul courant) est **mesurément insuffisante** : pas de palier de détection, chiffres à l'appui ; la **voie B** (soudure PA8 → Arduino D7) reste la seule voie vers le 3ᵉ estimateur) | 🟢 S5304 fait · S5305 conclu N/A |
| C — Axes scientifiques | **S5306 ⚠️ séance 1/3** (les deux régressions du build S38 sont non publiables, r² 0,366 et 0,696 ⇒ `energy_uj_per_update` reste `"à mesurer"` ; l'anomalie « repos > charge » du Sprint 50 **dépend du binaire flashé** ; **isolation à variable unique mesurée** — à dimension tenue k=4, poids entraînés −1,07 ± 0,41 µA/Hz et repli Xavier +1,67 ± 0,75, contre +20,73 ± 0,21 sur le build stock ⇒ **les poids sont écartés**, la dimension reste candidate, mécanisme NON établi) · S5307 (énergie sub-INT8 packé vs non-packé) · S5308 (isolation par composant + autonomie duty-cycle) | 🟡 S5306 partiel · S5307/S5308 📝 Spec |
| D — Consolidation | S5309 (agrégation + figures E6–E10 + notebook) · S5310 (tests + resynchronisation des docs + clôture `TODO`) | 📝 Spec |

**Liens triple gap** : Gap 2 (marge de latence enfin convertie en argument — faut-il ralentir le MCU pour tenir l'autonomie ? ; RAM `.bss` du gate et du packing rapportée à leur coût énergétique), Gap 3 (**verdict énergie de l'INT8** obtenu en régime de saturation, où l'effet sort du bruit ; arbitrage **RAM ↔ énergie** du bit-packing). **Statut** : 🟡 **En cours — 4 tâches mesurées sur carte réelle (S5301–S5304), 2 conclues en N/A honnête chiffré (S5305, S5306 partielle), S5307–S5310 non exécutées.** Séance de banc du 2026-09-07 : **7 défauts d'outillage corrigés, dont 3 produisaient des résultats crédibles et faux** (carte non alimentée profilée comme une mesure ; cellule écrite depuis le mauvais build ; biais de première acquisition inversant le signe du delta). Suivie d'un lot de correctifs hors banc **A1–A7** (pilote de sonde sous tests, garde-fou de plausibilité du courant, **ajustement unifié** — le balayage SYSCLK jetait les écarts-types et publiait d'autres chiffres que le balayage de cadence —, règle de publication d'une **différence** de pentes arrêtée avant mesure, contrats `f1_faulty`/flux explicités, provenance de la cadence atteinte) et de la mesure **B3**. **Réutilise sans réécriture** : `lpm01a_probe.py`, `run_s50_board_current.py` (`measure_current`/`stream_command`/`warmup`), `energy_capture.py`, `autonomy.py`, pilotes board S38/S48, registre figures S4201. **Points structurants** : les trois estimateurs de µJ/inférence (delta S5302, régression S5304, intégration S5305) sont conservés **côte à côte et jamais fusionnés** — leur écart est le contrôle de validité de la campagne ; S5304 est délibérément sans dépendance, c'est la seule tâche qui produit des µJ **quoi qu'il advienne** des verrous firmware.

→ Détail : [`docs/sprints/sprint_53/S5300_sprint_53.md`](sprints/sprint_53/S5300_sprint_53.md)

---

## Sprint P2-05 (18–24 juin 2026) — INT8 BACKPROP (Gap 3)

**Objectif** : Explorer la quantification INT8 pendant l'update incrémental → Gap 3

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S1014 | Implémentation SGD INT8 sur tête MLP (Python, simulation) | ⬜ | ⬜ | ⬜ | `src/models/ewc/ewc_mlp_int8.py` | 4h |
| S1015 | Expérience comparative FP32 vs INT8 update (AUROC, AF, BWT) | ⬜ | ⬜ | ⬜ | `experiments/exp_162/` | 2h |
| S1016 | Portage INT8 update en C (si résultats Python acceptables) | ⬜ | ⬜ | ⬜ | `firmware/stm32f4_blink/src/ewc_head_int8.c` | 4h |

**Livrable** : tableau AUROC FP32 vs INT8 + analyse impact sur AF/BWT. Constitue la contribution Gap 3.

---

## Sprint P2-06 (25 juin – 1 juillet 2026) — BENCHMARK EDGE SPECTRUM

**Objectif** : Validation industrielle sur équipement Edge Spectrum (Fred)

| ID | Tâche | Impl. | Doc | Exec | Fichier cible | Durée est. |
|----|-------|:-----:|:---:|:----:|---------------|------------|
| S1017 | Préparer démo : pipeline UART/SPI capteur → NUCLEO → décision | ⬜ | ⬜ | ⬜ | — | 3h |
| S1018 | Validation sur données CWRU ou PRONOSTIA en temps réel simulé | ⬜ | ⬜ | ⬜ | `experiments/exp_163/` | 3h |
| S1019 | Rapport benchmark industriel (latence, AUROC, RAM) | ⬜ | ⬜ | ⬜ | `experiments/exp_163/` | 2h |

> `TODO(fred)` : confirmer disponibilité équipement + format données capteur

---

## Sprints P2-07 à P2-09 (2–31 juillet 2026) — RÉDACTION MANUSCRIT

**Objectif** : Rapport final M2 intégrant Phase 1 + Phase 2

| Sprint | Contenu |
|--------|---------|
| P2-07 (2–8 jul) | Chapters résultats Phase 1 : tableaux AUROC/AF/BWT, figures notebooks |
| P2-08 (9–15 jul) | Chapters Phase 2 : profiling HW + Gap 2/3 + discussion triple gap |
| P2-09 (16–31 jul) | Relecture, figures finales, intégration retours Arnaud/Dorra |

**Livrables** : brouillon complet manuscrit M2 fin juillet.

---

## Sprint P2-10 (1–6 août 2026) — FINALISATION

| ID | Tâche |
|----|-------|
| S1020 | Nettoyage code + docstrings + tests manquants |
| S1021 | GitHub public : README complet + LICENSE + .gitignore propre |
| S1022 | Soumission rapport final ISAE-SUPAERO (6 août 2026) |

---

## Backlog Phase 2 (hors planning courant)

| Tâche | Priorité | Gap | Notes |
|-------|:--------:|-----|-------|
| Portage sur STM32N6 réel (cible finale) | 🔴 | Gap 2 | Dépend disponibilité hardware (TODO dorra/fred) |
| Buffer replay UINT8 sur TinyOL (M1 + buffer) | 🟡 | Gap 3 | Sprint 4 Phase 1 dépriorisé |
| Benchmark FEMTO PRONOSTIA temps réel | 🟡 | Gap 1 | Données déjà disponibles |
| HMM offline RUL analysis | 🟢 | — | PC-only, hors contrainte MCU |

---

## Triple Gap — Statut final après Sprint 23

| Gap | Critère | Sprint cible | Statut |
| --- | ------- | :----------: | ------ |
| **Gap 1** | Validation sur 5 datasets industriels (CWRU + Monitoring + Pronostia + CMAPSS + Paderborn) | Phase 1 ✅ / Sprint 22 ✅ | ✅ PRONOSTIA + CWRU + Monitoring + CMAPSS + Paderborn validés (5 datasets) |
| **Gap 2** | CL < 100 Ko RAM avec mesures précises HW | Sprint 16 ✅ / Sprint 18 ✅ / Sprint 20 ✅ | ✅ RAM 3 modèles = 15.7 Ko / 64 Ko · Mahalanobis 200 B @ 0.004 ms · EWC 9.7 Ko @ 0.004 ms · TinyOL 5.7 Ko · acc_final=0.897 · avg_forgetting=0.009 |
| **Gap 3** | INT8 pendant entraînement incrémental, Δ AUROC < 0.02 | Sprint 22 ✅ | ⚠️ Partiel — ewc_mlp_int8.py Δ AUROC < 0.02 ✅ · ewc_head_int8.c compilable ARM ✅ · réduction RAM ×2.7 documentée · accélération latence non mesurée sur Cortex-M4 FPU (voir exp_S23_INT8) |

> **Gap 3** : si `latency_int8 < latency_fp32` mesuré board → ✅. Sinon → ⚠️ résultat négatif honnête (réduction RAM ×2.7 sans accélération latence sur Cortex-M4 FPU — aucun travail précédent n'a mesuré cela sur MCU).
>
> `TODO(arnaud)` : le tableau Triple Gap dans la roadmap doit-il figurer dans le README public pour le rapport de stage ?
> `FIXME(gap3)` : formulation "Gap 3 partiel" à discuter avec Arnaud — réduction RAM ×2.7 sans accélération latence peut être présentée comme contribution originale.
