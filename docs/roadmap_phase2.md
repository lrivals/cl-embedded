# Roadmap Phase 2 — Portage MCU

> Mise à jour : 27 mai 2026 (Sprint 17 ✅ | Sprint 18 ✅ board | Sprint 19 ✅ | Sprint 20 ✅ Gap 2 formel | **Sprint 21 🔄 En cours** — tests Monitoring complet + portage Pronostia board)  
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
P2-05 (15–22 jun)        → INT8 backprop incrémental (Gap 3)
P2-06 (22–29 jun)        → Benchmark Edge Spectrum (Fred) + HDC C port
P2-07 (30 jun–14 jul)    → Rédaction manuscrit — résultats Phase 1+2
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

## Sprint 21 (27 mai – 6 juin 2026) — Tests multi-datasets board 🔄 EN COURS

**Objectif** : Compléter la couverture dataset sur board — Monitoring (tous modèles) + Pronostia (première validation board, Gap 1)

| Objectif | Tâches | Statut | Résultat attendu |
|----------|--------|:------:|-----------------|
| O1 — Feature selection Pronostia 13→5 | S2101 | ⬜ | `configs/pronostia_feature_subset.yaml` — top-5 mutual info |
| O2 — Streamer Pronostia board | S2102 | ⬜ | `sensor_stream.py --dataset pronostia` opérationnel |
| O3 — Config board Pronostia | S2103 | ⬜ | `configs/board_pronostia.yaml` |
| O4 — E21-01 : Mahalanobis / Monitoring | S2104 | ⬜ | acc_final, forgetting, latency_ms < 100 |
| O5 — E21-02 : TinyOL / Monitoring | S2105 | ⬜ | comparaison avec EWC Monitoring (E19-02) |
| O6 — E21-03 : Mahalanobis / Pronostia | S2106 | ⬜ | première mesure Pronostia board |
| O7 — E21-04 : EWC / Pronostia (λ=400) | S2107 | ⬜ | Gap 1 formel — CL board sur données industrielles réelles |
| O8 — RAM profiling + comparison | S2108–S2109 | ⬜ | `comparison_sprint21.json` 3 datasets × 3 modèles |
| O9 — Tests + docs | S2110–S2112 | ⬜ | `pytest -k pronostia` vert |

**Critère de succès** : 4 `results.json` dans `experiments/`, `gap2_latency_compliant: true` partout, `--dataset pronostia --dry-run` sans erreur

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

## Statut triple gap (Phase 2)

| Gap | Critère | Sprint cible | Statut |
|-----|---------|:------------:|--------|
| **Gap 1** | Validation données industrielles réelles | Phase 1 ✅ | ✅ PRONOSTIA + CWRU + Monitoring validés |
| **Gap 2** | CL < 100 Ko RAM avec mesures précises HW | Sprint 16 ✅ / Sprint 18 ✅ / Sprint 20 ✅ | ✅ RAM 3 modèles = 15.7 Ko / 64 Ko · Mahalanobis 200 B @ 0.004 ms · EWC 9.7 Ko @ 0.004 ms · TinyOL 5.7 Ko · acc_final=0.897 · avg_forgetting=0.009 |
| **Gap 3** | INT8 pendant entraînement incrémental | P2-05 | ⬜ Non adressé |
