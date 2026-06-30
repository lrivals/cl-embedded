# Sprint 23 — Board CMAPSS+Paderborn + Gap 3 board + Benchmark Edge Spectrum

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 23 |
| **Semaine** | 22 juin – 5 juillet 2026 |
| **Statut** | ✅ Terminé — O1–O7 ✅ complet (2026-06-02) |
| **Priorité globale** | 🔴 Critique — Gap 1 board (2 nouveaux datasets MCU) + Gap 2 formel étendu + Gap 3 board + Benchmark industriel |
| **Durée estimée totale** | ~32h |
| **Dépendances** | Sprint 22 ✅ (CMAPSS+Paderborn loaders, ewc_head_int8.c livré, tests Unity INT8 verts) |

---

## Objectifs

Sprint 23 ferme toutes les contributions manquantes avant la rédaction manuscrit :

| Contribution | Gap | Statut après Sprint 23 |
|---|---|---|
| CMAPSS sur board (4 modèles) | Gap 1 + Gap 2 | ✅ |
| Paderborn sur board (EWC + Maha) | Gap 1 + Gap 2 | ✅ |
| HDC C complet (skeleton → full) | Gap 2 | ✅ |
| INT8 latence mesurée board | Gap 3 | ✅ |
| Benchmark Edge Spectrum (Fred) | — | ✅ |
| Tableau comparatif 5 datasets × 5 modèles | manuscrit | ✅ |

```
Sprint 22 ✅                                        Sprint 23
─────────────────────────────────     ─────────────────────────────────────────
ewc_head_int8.c (compilable)    ──▶   S2307: board INT8 latence mesurée DWT
cmapss_loader.py ✅             ──▶   S2301: feature select 21→5 + board_cmapss.yaml
paderborn_loader.py ✅          ──▶   S2308: feature select ≤5 + board_paderborn.yaml
hdc.c skeleton (S2008 Sprint20) ──▶   S2303: HDC C full + AM update + Unity tests
                                       ↓
                                  E23-01 EWC / CMAPSS board
                                  E23-02 TinyOL / CMAPSS board
                                  E23-03 Mahalanobis / CMAPSS board
                                  E23-04 HDC / CMAPSS board  ← premier HDC réel MCU
                                  E23-05 EWC / Paderborn board
                                  E23-06 Mahalanobis / Paderborn board
                                  E23-07 INT8 vs FP32 board (CMAPSS)
                                       ↓
                                  Benchmark Edge Spectrum (Fred)
                                       ↓
                              comparison_sprint23.json
                         (4 datasets × 4 modèles × PC+board)
                                       ↓
                          notebooks/board_benchmark_all_datasets.ipynb
                             (figures finales manuscrit)
```

**Critères de succès** :
1. 7 dossiers `experiments/exp_S23_*/results.json` avec `gap2_latency_compliant: true`
2. `hdc.c` : AM update opérationnel, tests Unity ≥ 10/10 PASS
3. `exp_S23_INT8` : `latency_int8_ms < latency_fp32_ms` + `auroc_delta < 0.02`
4. `comparison_sprint23.json` : 4 datasets × 4 modèles en une entrée JSON
5. Benchmark Fred : `exp_S23_benchmark/results.json` produit

---

## Tâches

### O1 — HDC C complet (skeleton → implémentation fonctionnelle)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2301 | `hdc.c` : compléter AM update (accumulate + binarize) + retrain incrémental | 🔴 | ✅ | `firmware/stm32f4_blink/src/hdc.c` | 4h |
| S2302 | `hdc.h` : `hdc_update_with_sample()`, `hdc_binarize()`, `hdc_retrain()` — API complète | 🔴 | ✅ | `firmware/stm32f4_blink/inc/hdc.h` | 1h |
| S2303 | Tests Unity `test_hdc.c` : 10/10 PASS, 57/57 suite totale | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_hdc.c` | 2h |
| S2304 | Intégration `pipeline.c` : `PROTO_FLAG_HDC_MODE 0x20U` + branche HDC dans `pipeline_run()` | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 1h |

### O2 — CMAPSS sur board (feature selection + configs + 4 expériences)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2305 | Feature selection CMAPSS 21→5 (mutual info sur FD001) + `configs/board_cmapss.yaml` | 🔴 | ✅ | `scripts/cmapss_feature_selection.py`, `configs/board_cmapss.yaml` | 1h |
| S2306 | `sensor_stream.py --dataset cmapss` : loader CMAPSS adapté streaming UART | 🔴 | ✅ | `scripts/sensor_stream.py` | 1h |
| S2307 | exp_S23_01 : EWC / CMAPSS board (dry-run + live, λ=400) | 🔴 | ✅ | `experiments/exp_S23_01/` | 1h |
| S2308 | exp_S23_02 : TinyOL / CMAPSS board | 🟡 | ✅ | `experiments/exp_S23_02/` | 1h |
| S2309 | exp_S23_03 : Mahalanobis / CMAPSS board (baseline) | 🟡 | ✅ | `experiments/exp_S23_03/` | 30 min |
| S2310 | exp_S23_04 : HDC / CMAPSS board — **premier test HDC C réel sur MCU** | 🔴 | ✅ | `experiments/exp_S23_04/` | 1h |

### O3 — Paderborn sur board

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2311 | Feature selection Paderborn ≤5 (FFT features top-5) + `configs/board_paderborn.yaml` | 🔴 | ✅ | `scripts/paderborn_feature_selection.py`, `configs/board_paderborn.yaml` | 1h |
| S2312 | `sensor_stream.py --dataset paderborn` : streaming features FFT pré-calculées | 🔴 | ✅ | `scripts/sensor_stream.py` | 1h |
| S2313 | exp_S23_05 : EWC / Paderborn board (sain → OR → IR) | 🔴 | ✅ | `experiments/exp_S23_05/` | 1h |
| S2314 | exp_S23_06 : Mahalanobis / Paderborn board | 🟡 | ✅ | `experiments/exp_S23_06/` | 30 min |

### O4 — Gap 3 : INT8 validation board (latence + précision)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2315 | Intégration `ewc_head_int8.c` dans pipeline firmware : `PROTO_FLAG_INT8_MODE 0x40U` | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c`, `inc/pipeline.h`, `scripts/sensor_stream.py` | 2h |
| S2316 | exp_S23_INT8 : EWC FP32 vs INT8 sur board CMAPSS — structure config + results placeholders | 🔴 | ✅ | `experiments/exp_S23_INT8/` | 2h |
| S2317 | Notebook `notebooks/gap3_int8_board_results.ipynb` : tableau FP32 vs INT8 (latence, AUROC, AF, RAM) | 🔴 | ✅ | `notebooks/gap3_int8_board_results.ipynb` | 2h |

### O5 — Benchmark Edge Spectrum `TODO(fred)`

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2318 | Préparer démo pipeline : données capteur Edge Spectrum → NUCLEO → décision temps réel | 🟡 | ✅ | `scripts/edge_spectrum_demo.py` | 3h |
| S2319 | exp_S23_benchmark : validation sur données Fred (CWRU proxy Scénario B activé) | 🟡 | ✅ | `experiments/exp_S23_benchmark/` | 3h |
| S2320 | Rapport benchmark `docs/context/benchmark_edge_spectrum.md` : latence + AUROC + RAM | 🟡 | ✅ | `docs/context/benchmark_edge_spectrum.md` | 2h |

### O6 — Consolidation : tableau comparatif 5 datasets + figures manuscrit

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2321 | `scripts/generate_comparison_sprint23.py` : agréger CWRU + Monitoring + Pronostia + CMAPSS + Paderborn | 🔴 | ✅ | `experiments/comparison_sprint23.json` | 2h |
| S2322 | Notebook `notebooks/board_benchmark_all_datasets.ipynb` : figures finales 5 datasets × 5 modèles PC+board | 🔴 | ✅ | `notebooks/board_benchmark_all_datasets.ipynb` | 3h |
| S2323 | Figure `docs/figures/gap1_gap2_summary.png` : heatmap acc_final + barplot latence (pour manuscrit) | 🔴 | ✅ | `docs/figures/gap1_gap2_summary.png` | 2h |

### O7 — Tests + Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2324 | `pytest tests/ -k "cmapss or paderborn"` vert (tests stream + recorder) | 🟡 | ✅ | `tests/test_cmapss_stream.py`, `tests/test_paderborn_stream.py` | 1h |
| S2325 | Roadmap update (Sprint 23 clôturé, Triple Gap statut final) | 🟡 | ✅ | `docs/roadmap_phase2.md` | 30 min |

---

## Résultats réels board NUCLEO-F439ZI (2026-06-02)

| Expérience | Modèle | Dataset | lat P50 (ms) | acc_final | AF | Gap 2 | Critère atteint |
|---|---|---|---:|---:|---:|---|---|
| exp_S23_01 | EWC | CMAPSS | 0.251 | **0.840** | 0.000 | ✅ | ✅ acc≥0.75 |
| exp_S23_02 | TinyOL | CMAPSS | 0.126 | 0.148 | 0.045 | ✅ | ❌ poids inadaptés |
| exp_S23_03 | Mahalanobis | CMAPSS | 0.004 | 0.575 | 0.000 | ✅ | ✅ acc≥0.60 |
| exp_S23_04 | **HDC** | CMAPSS | 0.646 | **0.853** | 0.000 | ✅ | ✅ acc≥0.65 🎯 |
| exp_S23_05 | EWC | Paderborn | 0.251 | **0.931** | 0.077 | ✅ | ✅ acc≥0.80 |
| exp_S23_06 | Mahalanobis | Paderborn | 0.004 | 0.380 | 0.690 | ✅ | ❌ oubli catastrophique |
| exp_S23_INT8 | EWC INT8 | CMAPSS | 0.461 | 0.853 | 0.000 | ✅ | ⚠️ plus lent que FP32 |
| exp_S23_benchmark | EWC | CWRU proxy | 0.251 | **0.883** | 0.175 | ✅ | ✅ Scénario B |

**Critères de succès sprint** :
- `gap2_latency_compliant: true` pour 8/8 expériences ✅
- HDC C AM update opérationnel, 10/10 Unity PASS ✅
- exp_S23_INT8 : latence INT8 > FP32 ❌ (Cortex-M4 FPU), ΔAUROC=0.013 < 0.02 ✅, RAM -2.7× ✅
- exp_S23_benchmark produit (Scénario B) ✅

**Notes** :
- TinyOL CMAPSS : poids embarqués entraînés sur Monitoring → inadaptés CMAPSS. Sprint 24 : exporter poids CMAPSS.
- Mahalanobis Paderborn : oubli catastrophique 3 tâches (AF=0.69) — attendu pour modèle sans CL.
- INT8 résultat négatif documenté : contribution Gap 3 via RAM -2.7×, latence non améliorée sur Cortex-M4 FPU.

---

## Livrables

1. `firmware/stm32f4_blink/src/hdc.c` complet + `hdc.h` + tests Unity
2. `firmware/stm32f4_blink/src/ewc_head_int8.c` intégré firmware
3. `configs/board_cmapss.yaml` + `configs/board_paderborn.yaml`
4. 7 dossiers `experiments/exp_S23_*/results.json`
5. `experiments/comparison_sprint23.json` (4+ datasets)
6. 3 notebooks : `gap3_int8_board_results`, `board_benchmark_all_datasets`, figure heatmap
7. `docs/context/benchmark_edge_spectrum.md` (si Fred disponible)

---

## Notes et risques

- **HDC C** : le skeleton `hdc.c` (S2008, Sprint 20) n'implémente que l'encodage + AM search. Il manque `hdc_update()` (accumulation en ligne). Compter 4h pour la complétion.
- **Paderborn features board** : le streaming de features FFT pré-calculées (non signaux bruts) simplifie l'adaptation UART. Les features sont déjà extraites par `paderborn_loader.py` (Sprint 22).
- **INT8 latence** : si l'accélération INT8 est négligeable sur Cortex-M4 FPU (qui préfère FP32), documenter ce résultat explicitement pour le manuscrit — c'est une contribution négative valide.
- `TODO(fred)` : S2318–S2320 dépendent de la disponibilité d'Edge Spectrum. Si non disponible avant fin juin, reporter en P2-06 et utiliser CWRU comme proxy.
- `TODO(arnaud)` : valider que 5 datasets (CWRU, Monitoring, Pronostia, CMAPSS, Paderborn) suffisent pour le chapitre résultats Gap 1.
