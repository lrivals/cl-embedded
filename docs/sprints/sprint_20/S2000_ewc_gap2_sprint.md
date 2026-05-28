# Sprint 20 — Finalisation EWC + Gap 2 formel + Validation PC vs board

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 20 |
| **Semaine** | 8–15 juin 2026 |
| **Statut** | ✅ Clôturé |
| **Priorité globale** | 🔴 Critique — clôture Sprint 19 + Gap 2 mesurable |
| **Durée estimée totale** | ~40h |
| **Dépendances** | Sprint 19 (métriques firmware ✅, recorder ✅, dry-run ✅), Sprint 18 (pipeline UART v2 ✅) |

---

## Objectif

Finaliser les éléments bloquants de Sprint 19 et obtenir les **chiffres Gap 2 formels** (RAM mesurée + latence mesurée, 3 modèles simultanés) exploitables dans le manuscrit.

```
ewc_consolidate() + poids TinyOL  →  Unity tests N/N PASS
                                            ↓
              Protocol v3 firmware (21B response avec métriques)
                                            ↓
        board_experiment_recorder.py --exp ewc --lambda 400 vs 0
                                            ↓
           parse_map_file.py --budget 64Ko → tableau Gap 2
                                            ↓
         compare_mahalanobis_pc_vs_board.py → delta < 1e-4
```

**Critère de succès** :
1. `make test` : N/N Unity PASS (host x86) incluant EWC + TinyOL
2. `python scripts/board_experiment_recorder.py --dry-run --model ewc` : `results.json` avec `avg_forgetting` ≤ 0.10 (λ=400) vs ≥ 0.25 (λ=0)
3. `parse_map_file.py` confirme RAM totale 3 modèles < 64 Ko
4. `compare_mahalanobis_pc_vs_board.py` : delta PC/C ≤ 1e-4 sur 500 samples

---

## Tâches

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Dépendances |
|----|-------|:--------:|:------:|--------------------|-------------|
| S2001 | `ewc_consolidate()` : Fisher EMA + θ* snapshot, déclaration .h | 🔴 | ✅ | `src/ewc_head.c`, `inc/ewc_head.h` | S19 EWC existant |
| S2002 | Protocol v3 firmware : réponse 21B + métriques snapshot | 🔴 | ✅ | `src/pipeline.c` | S2001, S1905 |
| S2003 | Export poids TinyOL → `model_weights.h` + validation forward pass | 🔴 | ✅ | `inc/model_weights.h`, `src/tinyol.c` | S1903 skeleton |
| S2004 | Unity tests EWC + TinyOL (8 groupes) sur `mock_data.h` | 🟡 | ✅ | `tests/test_models.c` | S2001, S2003 |
| S2005 | Expérience E19-02 EWC : λ=400 vs λ=0, 3 tâches Monitoring, forgetting mesuré | 🔴 | ✅ | `experiments/exp_S19_02/` | S2001, S2002 |
| S2006 | RAM profiling 3 modèles simultanés : `parse_map_file.py --budget 64Ko` | 🟡 | ✅ | `scripts/parse_map_file.py` | S2001–S2003 |
| S2007 | Comparaison PC vs board : Mahalanobis + EWC delta ≤ 1e-4 | 🟡 | ✅ | `scripts/compare_mahalanobis_pc_vs_board.py` | S2002, S2005 |
| S2008 | *(optionnel)* HDC C skeleton : hypervecteur encode + recherche AM | 🟢 | ✅ | `src/hdc.c`, `inc/hdc.h` | S2004 |
| S2009 | *(optionnel)* Online CL loop : changement TASK_ID automatique dans `sensor_stream.py` | 🟢 | ✅ | `scripts/sensor_stream.py` | S2005 |
| S2010 | Document de présentation exhaustif + 12 figures visuelles (Sprints 16–20) | 🟡 | ✅ | `docs/presentation_board_sprint16_20.md`, `scripts/generate_presentation_plots.py`, `docs/figures/presentation_board/` | S2005, S2006 |
| S2011 | Génération des figures PNG + notebook Jupyter de présentation | 🟡 | ✅ | `docs/figures/presentation_board/*.png`, `notebooks/presentation_board_sprint16_20.ipynb` | S2010 |

> Détail : [S2001](S2001_ewc_consolidate.md) · [S2002](S2002_protocol_v3.md) · [S2003](S2003_tinyol_weights.md) · [S2004](S2004_unity_tests.md) · [S2005](S2005_exp_ewc.md) · [S2006](S2006_ram_profiling.md) · [S2007](S2007_pc_vs_board.md) · [S2008](S2008_hdc_skeleton.md) · [S2009](S2009_cl_loop.md) · [S2010](S2010_presentation_summary_plots.md) · [S2011](S2011_figures_notebook.md)

---

## Budget RAM cible (3 modèles simultanés)

| Composant | RAM .bss | Stack peak | Total |
|-----------|:--------:|:----------:|:-----:|
| `MahalanobisDetector` | 200 B | 40 B | **~240 B** |
| `EWCHead` (poids + Fisher + θ*) | 9.5 Ko | 200 B | **~9.7 Ko** |
| `TinyOLEncoder` (poids .bss) | 5.6 Ko | 512 B | **~6.1 Ko** |
| Métriques (acc + AUROC + forgetting) | 314 B | — | **~314 B** |
| Profiling DWT + pipeline | 60 B | — | **~60 B** |
| **Total estimé** | **~16 Ko** | **~1 Ko** | **~17 Ko** |
| **Marge / 64 Ko** | | | **✅ ~47 Ko free** |

> `FIXME(gap2)` : validé sur NUCLEO-F439ZI (192 Ko SRAM). Validation formelle sur STM32N6 (64 Ko) bloquée par accès hardware.

---

## Questions ouvertes

- `TODO(arnaud)` : Tolérance PC vs C : 1e-4 (FP32 strict) ou 1% (acceptable pour manuscrit) ?
- `TODO(arnaud)` : Inclure E19-02 EWC (λ sweep) dans le tableau comparatif du chapitre 4 ?
- `TODO(dorra)` : Proxy Fisher `grad² ≈ w²` acceptable pour la publication, ou faut-il accumuler les vrais gradients ?
- `TODO(dorra)` : Protocol v3 : envoyer `task_id` dans la réponse pour segmentation PC-side par tâche ?
- `FIXME(gap2)` : Linker .map NUCLEO indicatif — STM32N6 Cortex-M55 requis pour la table formelle du manuscrit
