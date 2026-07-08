# Sprint 39 — Approfondissement INT8 vs FP32 sur board (suite Sprints 28/29)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 39 |
| **Semaine** | 30 juin – 4 juillet 2026 |
| **Statut** | 🟡 En cours — Partie A (PC) prioritaire ; Partie B (board) différée (carte indisponible) |
| **Priorité globale** | 🔴 Critique — comprendre et corriger la perte d'accuracy/F1 INT8 (Gap 3) |
| **Durée estimée totale** | ~40h (Partie A ~30h, Partie B ~10h différée) |
| **Dépendances** | Sprint 28 ✅ (QAT PC) · Sprint 29 ✅ (board INT8) · Sprint 34 ✅ (Q15 Maha) · Sprint 36 ✅ (PTQ board F1) |

## Contexte et motivation

Le Gap 3 (quantification INT8 pendant l'entraînement incrémental) est **comblé sur la RAM** (×2.33–4.0)
mais deux problèmes restent ouverts et mal compris :

1. **Perte d'accuracy/F1 en INT8 sur board.** Le QAT fake-quant *PC* préserve la métrique (Sprint 28,
   Δ≤0.006), mais la PTQ *embarquée* dégrade catastrophiquement : F1 EWC **0.07–0.15** vs FP32 ≈0.92
   (Sprint 36), AUROC 0.25 vs 0.63 (Sprint 29). On ignorait *à quoi* et *dans quelle proportion*.

2. **Pas de gain de latence INT8.** Le kernel C déquantifie vers FP32 dans la boucle interne → latence
   **~1.84× pire** que FP32 sur Cortex-M4 FPU (Sprint 23/29). Aucun chemin entier optimisé (SIMD/CMSIS-NN
   bloqué S2908).

**Diagnostic (formalisé en S3901, audit).** Trois causes candidates dans `ewc_head_int8.c` :
accumulateur `int16_t` (overflow latent), échelle **fixe `1/128`** non calibrée (≠ scales par-canal du QAT
PC, clampe les activations), et PTQ one-shot sans recalibration (`ewc_int8_from_fp32`).

**Stratégie « maison » (carte indisponible).** Un **émulateur Python bit-exact** du chemin C (S3902)
reproduit la dégradation board **sans flasher**, permettant l'ablation chiffrée (S3904) et le balayage de
schémas intermédiaires (per-channel INT8, Q15, mixte — S3906) au PC. Les mesures qui exigent le matériel
(latence DWT, `.bss` cible, parité streaming, SIMD CMSIS) sont **isolées en Partie B différée**.

**Résultats baseline connus (à reproduire/expliquer)** :

| Source | Modèle | FP32 | INT8 board | Écart |
|--------|--------|:----:|:----------:|:-----:|
| Sprint 36 | EWC F1 (pronostia 5feat, frozen) | ≈0.916 | **0.138** | −0.78 |
| Sprint 29 | EWC AUROC (cwru) | 0.453 | **0.401** | −0.05 |
| Sprint 28 | EWC AUROC PC QAT (cmapss) | 0.768 | 0.773 | +0.006 ✅ |
| Sprint 34 | Maha AUROC (pronostia) INT8→Q15 | 0.860 | 0.747→**0.873** | Q15 récupère ✅ |

**Critères de succès (Partie A)** :
1. L'émulateur reproduit qualitativement la dégradation board (legacy_c) et la récupération des variantes.
2. Tableau d'ablation attribuant la perte F1 à chaque facteur (int32, scale calibré, par-canal, Q15).
3. ≥1 schéma intermédiaire récupère la F1 vers le FP32 dans l'émulateur ET passe `make test` host (kernel v2).
4. Trade-off latence-proxy / RAM / accuracy documenté pour 4 modèles × 5 datasets.

## Tâches

### Partie A — PC (réalisable sans board)

#### O1 — Audit & critique de l'INT8 actuel

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3901 | Doc d'audit : faiblesses (overflow int16, scale 1/128 fixe, déquant→FP32, PTQ≠QAT, absence SIMD) | 🔴 | `docs/sprints/sprint_39/S3901_audit_int8_actuel.md` | ✅ |

#### O2 — Émulateur Python bit-exact du chemin C INT8

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3902 | `int8_c_emulation.py` — forward C bit-exact (acc int16, >>7, Q7 1/128) + variantes paramétrables | 🔴 | `src/utils/int8_c_emulation.py` | ✅ |
| S3903 | Validation émulateur vs logs board (`exp_S36` / `exp_S29_board_int8`) sans flasher | 🔴 | `tests/test_int8_c_emulation.py` | ⬜ |

#### O3 — Diagnostic chiffré de la perte (ablation PC)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3904 | Ablation : attribuer la chute F1 par facteur isolé (int32, scale calibré, par-canal, Q15) | 🔴 | `scripts/run_s39_int8_ablation.py` → `experiments/exp_S39_ablation/` | ⬜ |

#### O4 — Schémas de quantification intermédiaires (PC)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3905 | Configs des variantes (per-channel INT8, Q15, mixte INT8w/Q15act) via `configs/` | 🔴 | `configs/quant_intermediate/*.yaml` | ✅ |
| S3906 | Campagne PC trade-off : 4 modèles × 5 datasets × 5 schémas (métrique + RAM + proxy latence) | 🔴 | `scripts/run_s39_quant_sweep.py` → `experiments/exp_S39_quant_sweep/` | ✅ |

#### O5 — Kernel C v2 optimisé + tests host (sans flash — `make test` x86)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3907 | `ewc_head_int8_v2.c/.h` — acc int32, scales par-canal importés, variantes Q15/mixte ; ancien intact (A/B) | 🔴 | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c`, `inc/ewc_head_int8_v2.h` | ✅ |
| S3908 | Export scales calibrés par-canal → header C (`export_weights_c.py --int8-v2`) | 🔴 | `scripts/export_weights_c.py` → `inc/ewc_head_int8_v2_weights.h` | ✅ |
| S3909 | Tests Unity host parité C v2 ↔ émulateur Python (`make test` x86, aucun board) | 🟡 | `firmware/stm32f4_blink/tests/test_ewc_int8_v2.c` | ✅ |
| S3910 | Spec SIMD CMSIS-NN/DSP différée (code prêt + build conditionnel ; mesure → Partie B) | 🟢 | `docs/sprints/sprint_39/S3910_simd_cmsis_spec.md` | ✅ |

#### O5b — Comparaison PC↔board à conditions strictement identiques

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3918 | Harnais apparié PC : côté PC = émulateur exécutant le **schéma exact du board** (jamais le QAT S28), source unique données/normalisation/métrique → référence bit-comparable | 🔴 | `scripts/run_s39_matched_compare.py` → `experiments/exp_S39_matched/`, `tests/test_s39_matched.py` | ⬜ |

> **Ordre de priorité des schémas (consigne)** : tester **INT8 en premier** (`legacy_c` puis `per_channel_int8`).
> Ne passer aux schémas 16-bit (`q15`, `mixed_int8w_q15act`) **que si l'INT8 échoue** au critère
> (métrique non préservée / parité cassée) — l'INT8 reste le défaut visé (RAM ×4), Q15 est le repli ciblé (RAM ×2).

#### O6 — Expériences + notebook + tests Python

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3911 | Notebook trade-off (RAM vs accuracy vs latence-proxy) + heatmaps ablation | 🟡 | `notebooks/cl_eval/int8_intermediate/comparison.ipynb` | ✅ |
| S3912 | Tests Python (émulateur, ablation, schémas) | 🟡 | `tests/test_s39_quant.py` | ✅ |

#### O7 — Doc & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3913 | Doc principale + entrée roadmap + section Gap 3 `triple_gap.md` | 🟡 | ce fichier, `docs/roadmap_phase2.md`, `docs/triple_gap.md` | 🟡 |
| S3914 | `graphify_sprint_update` (nouveaux .c/.py/.md → update graphe) | 🟢 | — | ⬜ |

### Partie B — Board NUCLEO-F439ZI (DIFFÉRÉ — nécessite la carte)

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S3915 | **Board** : flasher v2, mesurer latence DWT + `.bss` + F1 + parité board↔PC par schéma | 🔴 | `scripts/run_s39_board.py` → `experiments/exp_S39_board/` | ⬜ différé |
| S3916 | **Board** : confirmer récupération F1 v2 vs ancien (A/B board réelle) | 🔴 | `experiments/exp_S39_board/` | ⬜ différé |
| S3917 | **Board** : débloquer toolchain CMSIS + bench SIMD `arm_dot_prod_q7` (lève S2908) | 🟢 | `firmware/.../ewc_head_int8_v2_simd.c`, `experiments/exp_S39_board/results_simd.json` | ⬜ différé |
| S3919 | **Board** : confronter board ↔ `exp_S39_matched` (S3918) **bit-à-bit** (frozen = 1.000 exact ; online = accord ≈ documenté) | 🔴 | `scripts/board_pc_parity.py --against exp_S39_matched` | ⬜ différé |

## Ordre d'exécution recommandé

```
S3901 ✅ → S3902 ✅ → S3903 → S3904 ─┐
                                      ├→ S3906 → S3911
S3905 ───────────────────────────────┘
S3904 → S3907 → S3908 → S3909 → S3910
S3903/S3907 → S3918 (harnais apparié, INT8 d'abord) → S3906 → S3912 → S3913 → S3914
(Partie B : S3915 → S3916 → S3919, S3917 — quand carte disponible)
```

## Nomenclature des expériences

| Exp ID | Contenu | Plateforme |
|--------|---------|:----------:|
| `exp_S39_ablation/` | Ablation F1 par facteur (échelle ABLATION_LADDER) | PC |
| `exp_S39_quant_sweep/` | 4 modèles × 5 datasets × {FP32, INT8 legacy, per-channel, Q15, mixte} | PC |
| `exp_S39_matched/` | Côté PC apparié (émulateur = schéma board exact) prêt pour parité bit-exacte | PC |
| `exp_S39_board/` | Latence DWT, `.bss`, F1, parité, SIMD | Board (différé) |

## Budget mémoire estimé (tête EWC, NUCLEO-F439ZI)

| Schéma | Poids | Précision interm. | RAM vs FP32 | Accuracy attendue |
|--------|:-----:|:-----------------:|:-----------:|:-----------------:|
| FP32 | float32 | FP32 | ×1 | référence |
| INT8 legacy (actuel) | Q7 fixe | Q7 + acc int16 | ×4 | ❌ dégradée |
| INT8 per-channel | int8 calibré | int8 + acc int32 | ×4 | ✅ cible ≈ FP32 |
| Q15 | int16 calibré | int16 + acc int32 | ×2 | ✅ fidélité 256× |
| Mixte INT8w/Q15act | int8 calibré | act int16 | ×4 poids | ✅ compromis |

## Questions ouvertes

- `TODO(dorra)` : compléter la toolchain CMSIS-NN/DSP (`libarm_cortexM4lf_math.a` + `arm_math.h`) pour
  débloquer le bench SIMD INT8 (S3910/S3917).
- `TODO(arnaud)` : pour le manuscrit, présenter l'INT8 « RAM-only » (latence FPU problème ouvert) ou
  attendre le résultat SIMD board avant de conclure sur le Gap 3 latence ?
- `FIXME(gap3)` : si un schéma intermédiaire (Q15) récupère la F1 ET réduit la RAM (×2), est-ce le défaut
  recommandé pour le board (au lieu d'INT8 ×4 dégradé) ?

## Livrables

1. `docs/sprints/sprint_39/` (ce dossier) — audit, specs, tâches.
2. `src/utils/int8_c_emulation.py` — émulateur bit-exact (✅ livré).
3. `experiments/exp_S39_ablation/`, `experiments/exp_S39_quant_sweep/` — résultats PC.
4. `firmware/.../ewc_head_int8_v2.c/.h` + tests Unity host.
5. Notebook trade-off + section Gap 3 enrichie de `triple_gap.md`.

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3901 | ✅ | — | Audit livré : 6 faiblesses cataloguées |
| S3902 | ✅ | — | Émulateur livré + smoke-test : legacy_c dégrade, variantes récupèrent |
| S3905 | ✅ | — | 25 configs `configs/quant_intermediate/*.yaml` (5 schémas × 5 datasets), `yaml.safe_load` OK |
| S3906 | ✅ | — | `run_s39_quant_sweep.py` → 20 JSON + `summary.json`. **EWC** : `int8_legacy` s'effondre (monitoring 0.027, pronostia 0.045) → **`int8_perchannel` récupère ≈ FP32** (0.915, 0.944), Q15/mixed idem. **Maha** : INT8 0.77 → **Q15 0.923** (pronostia). BOPs/latence = proxy analytique honnête (`lat_proxy: true`, FPU réelle → S3915) |
| S3907 | ✅ | — | `ewc_head_int8_v2.c/.h` (acc int32, scales par-canal, variantes `-DEWC_INT8_Q15`/`-DEWC_INT8_MIXED`) ; v1 intact (A/B). Câblé C_SOURCES+TEST_SRC. `make test` **122** (2 TinyOL préexistants, **0 régression**) |
| S3908 | ✅ | — | `export_weights_c.py --int8-v2` → `ewc_head_int8_v2_weights.h` (int8 par-canal + scales calibrés, header vide par défaut = 0 régression) + `--int8-v2-test-vectors` (golden vectors émulateur fp32/per_channel/q15). Réutilise `_weight_scales`/`calibrate_activations` (parité par construction) |
| S3909 | ✅ | — | `test_ewc_int8_v2.c` (5 cas) : header golden **auto-suffisant** (poids FP32 + `act_max` + logits) via `export_weights_c.py --int8-v2-test-vectors` étendu → le test reconstruit la tête et prouve la parité par construction. `make test` **127** (v2 4 PASS + 1 IGNORE q15, 2 TinyOL préexistants). **Bug réel trouvé & corrigé** : l'acc `int32` du kernel v2 **déborde en Q15** (int16×int16 sommé > 2³¹) → `ewc_v2_acc_t` = int32 (int8/mixed) / **int64 (Q15)** ; `make test-v2-q15` **PASS** (parité q15 1e-3). 0 régression (int8/mixed inchangés) |
| S3910 | ✅ | — | Spec SIMD complète (déjà rédigée) ; mesure board = Partie B S3917 (toolchain CMSIS bloquée, `TODO(dorra)`). Aucun code produit (différé assumé) |
| S3911 | ✅ | — | `notebooks/cl_eval/int8_intermediate/comparison.ipynb` (nbconvert OK) : ablation par facteur, scatter Pareto RAM↔métrique (taille ∝ latence-proxy), 3 heatmaps 4×5 (N/A gris), récap reco. **5 PNG** `docs/figures/sprint39_int8_intermediate/` (0 valeur en dur) |
| S3912 | ✅ | — | `tests/test_s39_quant.py` **11 PASS** : ablation (tail calibré stable, endpoints récupèrent FP32, dominant=scale calib), sweep (structure, RAM ×4/×2, HDC Δ=0, `lat_proxy` marqué, legacy s'effondre, q15 récupère) + 2 tests émulateur live. **Honnêteté** : l'échelle n'est **pas** monotone bout-en-bout (`fix_acc32` seul dégrade monitoring) → test reformulé sur le régime calibré. Suite `-k "int8 or quant or emulation"` **40 PASS** |
| S3903, S3913–S3919 | ⬜ | — | Documentés ; Partie B board S3915+ différée (carte non requise ici) |
