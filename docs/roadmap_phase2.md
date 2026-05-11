# Roadmap Phase 2 — Portage MCU

> Mise à jour : 11 mai 2026 → 11 mai 2026 (Sprint 16 clôturé)  
> Horizon : 20 mai – 6 août 2026  
> ← [Index roadmap](roadmap.md)

> **Note de numérotation** : Les Sprints 1–5 = Phase 1 cœur. Les Sprints 6–15 = Phase 1 Extension (notebooks + anomaly detection). Sprint 16 = Phase 2 Portage MCU (toolchain ARM + portage C embarqué). Sprints 17–19 = Phase 1 Extension anomaly detection (CWRU, Monitoring, Pronostia). Les sprints Phase 2 sont dans `docs/sprints/sprint_16/`.

---

## Vue macro Phase 2

```
Sprint 17 (18–20 mai)    → Anomaly detection CWRU (clôture Phase AD)
Sprint 18 (21–27 mai)    → Anomaly detection Equipment Monitoring
Sprint 19 (28 mai–3 jun) → Anomaly detection Pronostia
Sprint 16 (20 mai–17 jun)→ Portage MCU Phase 2 (toolchain ARM + C + profiling)
P2-05 (18–24 jun)        → INT8 backprop incrémental (Gap 3)
P2-06 (25 jun–1 jul)     → Benchmark Edge Spectrum (Fred)
P2-07 (2–8 jul)          → Rédaction manuscrit — résultats Phase 1+2
P2-08 (9–15 jul)         → Rédaction manuscrit — discussion + triple gap
P2-09 (16–31 jul)        → Finalisation rapport + figures
P2-10 (1–6 août)         → Code GitHub public + soumission rapport final
```

---

## Sprint 17 (18–20 mai 2026) — CWRU Anomaly Detection

**Objectif** : Anomaly detection CWRU — 6 modèles, clôture Phase Anomaly Detection (Sprints 13–17)
→ Détail : [`docs/sprints/sprint_17/`](sprints/sprint_17/)

---

## Sprint 18 (21–27 mai 2026) — Equipment Monitoring Anomaly Detection

**Objectif** : Anomaly detection Equipment Monitoring (4D, ~50% normal, by_equipment_type)
→ Détail : [`docs/sprints/sprint_18/`](sprints/sprint_18/)

---

## Sprint 19 (28 mai–3 juin 2026) — Pronostia Anomaly Detection

**Objectif** : Anomaly detection Pronostia (13D, ~90% normal, by_bearing_condition)
→ Détail : [`docs/sprints/sprint_19/`](sprints/sprint_19/)

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
| **Gap 2** | CL < 100 Ko RAM avec mesures précises HW | P2-04 | ⬜ En attente profiling HW |
| **Gap 3** | INT8 pendant entraînement incrémental | P2-05 | ⬜ Non adressé |
