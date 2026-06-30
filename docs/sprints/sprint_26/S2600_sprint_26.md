# Sprint 26 — Portage Board NUCLEO-F439ZI : RUL Régression + Multi-class (C Firmware)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 26 |
| **Semaine** | 29 juillet – 5 août 2026 |
| **Statut** | 🔄 En cours — O2 ✅ |
| **Priorité globale** | 🔴 Critique — porter RUL régression et multi-class sur board avant deadline manuscrit (6 août) |
| **Durée estimée totale** | ~24h |
| **Dépendances** | Sprint 25 ✅ (ewc_mlp_regression.py, ewc_mlp_multiclass.py, résultats PC exp_S25_01 et exp_S25_03) |

---

## Contexte et motivation

Sprint 25 a validé sur PC les tâches natives des datasets (RUL continu sur CMAPSS, multi-class fault sur CWRU). Sprint 26 porte ces capacités sur la **NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 256 Ko SRAM) pour démontrer la faisabilité embarquée et mesurer les performances réelles (latence DWT, footprint RAM .bss).

Le firmware existant supporte déjà :
- `ewc_head.c` — EWC FP32 binaire (2 sorties, BCE)
- `ewc_head_int8.c` — EWC INT8 quantifié (2 sorties)
- `pipeline.c` — protocole UART v3 binaire (14 octets de réponse)
- `metrics.c` — OnlineAccuracy, OnlineAUROC, ForgettingTracker

Sprint 26 ajoute les têtes C pour régression (1 sortie, MSE) et multi-class (N sorties, softmax) en suivant exactement la même architecture statique (pas de malloc).

```
Sprint 25 ✅                              Sprint 26
──────────────────────────────    ──────────────────────────────────────────
ewc_mlp_regression.py         ──▶  S2601 ewc_head_regression.c + .h
ewc_mlp_multiclass.py         ──▶  S2603 ewc_head_multiclass.c + .h
exp_S25_01 (CMAPSS RUL PC)    ──▶  S2607 export model_weights_rul.h
exp_S25_03 (CWRU multiclass)  ──▶  S2608 export model_weights_multiclass.h
                                          ↓
                               S2605 pipeline.c (FLAGS RUL_MODE, MULTICLASS_MODE)
                               S2606 metrics.c (OnlineRMSE, OnlineF1Macro)
                                          ↓
                               S2609 simulate_rul_board.py  (host UART → CMAPSS FD001)
                               S2610 simulate_multiclass_board.py (host UART → CWRU)
                                          ↓
                               exp_S26_01 EWC RUL board / CMAPSS FD001
                               exp_S26_02 EWC multi-class board / CWRU
                               exp_S26_03 RAM profiling board (ewc_reg vs ewc_fp32 vs ewc_int8)
```

**Critères de succès** :
1. `make -j4` — firmware compile sans warning sur arm-none-eabi-gcc
2. `make test` — `test_ewc_regression.c` + `test_ewc_multiclass.c` verts (compilation host, Unity)
3. `experiments/exp_S26_01/results.json` — RMSE board sur CMAPSS FD001 dans ±10% du RMSE PC
4. `experiments/exp_S26_02/results.json` — F1-macro board sur CWRU ≥ 0.60
5. Latence mesurée DWT ≤ 100 ms pour les deux têtes (critère Gap 2)
6. `arm-none-eabi-size build/stm32f4_blink.elf` — usage SRAM total < 256 Ko

---

## Tâches

### O1 — Tête C EWC Régression

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2601 | Implémenter `ewc_head_regression.c` : copie de `ewc_head.c` avec output_dim=1, perte MSE (gradient = `ŷ - y`), pas de softmax — statically allocated, no malloc | 🔴 | ⬜ | `firmware/stm32f4_blink/src/ewc_head_regression.c` | 3h |
| S2602 | Créer `ewc_head_regression.h` : prototypes `ewc_reg_init()`, `ewc_reg_forward()`, `ewc_reg_update()`, `ewc_reg_consolidate()` | 🔴 | ⬜ | `firmware/stm32f4_blink/inc/ewc_head_regression.h` | 30 min |

### O2 — Tête C EWC Multi-class

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2603 | Implémenter `ewc_head_multiclass.c` : sortie N classes (N configurable via `#define EWC_MC_N_CLASSES`), softmax en FP32, cross-entropy, EWC penalty identique à `ewc_head.c` | 🔴 | ✅ | `firmware/stm32f4_blink/src/ewc_head_multiclass.c` | 3h |
| S2604 | Créer `ewc_head_multiclass.h` : prototypes `ewc_mc_init()`, `ewc_mc_forward()`, `ewc_mc_predict()`, `ewc_mc_update()`, `ewc_mc_consolidate()` | 🔴 | ✅ | `firmware/stm32f4_blink/inc/ewc_head_multiclass.h` | 30 min |

### O3 — Pipeline UART et métriques

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2605 | Étendre `pipeline.c` : décoder `FLAGS & PROTO_FLAG_RUL_MODE` → router vers `ewc_reg_forward()` ; `FLAGS & PROTO_FLAG_MULTICLASS_MODE` → `ewc_mc_forward()` ; encoder RUL float dans réponse UART | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 2h |
| S2606 | Étendre `metrics.c` : ajouter `OnlineRMSE` (Welford en ligne, 4 B SRAM) et `OnlineF1Macro` (matrice de confusion compacte, `n_classes × n_classes × 2B` SRAM) | 🔴 | ✅ | `firmware/stm32f4_blink/src/metrics.c`, `firmware/stm32f4_blink/inc/metrics.h` | 2h |

### O4 — Export poids entraînés vers C headers

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2607 | Créer `scripts/export_weights_ewc_rul.py` : charge le modèle `ewc_mlp_regression` entraîné sur CMAPSS FD001, exporte les poids en tableaux C `const float` dans `model_weights_rul.h` | 🔴 | ✅ | `scripts/export_weights_ewc_rul.py`, `firmware/stm32f4_blink/inc/model_weights_rul.h` | 2h |
| S2608 | Créer `scripts/export_weights_ewc_multiclass.py` : charge le modèle `ewc_mlp_multiclass` entraîné sur CWRU, exporte vers `model_weights_multiclass.h` | 🔴 | ✅ | `scripts/export_weights_ewc_multiclass.py`, `firmware/stm32f4_blink/inc/model_weights_multiclass.h` | 1h |

### O5 — Scripts de simulation host

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2609 | Créer `scripts/simulate_rul_board.py` : envoie séquence CMAPSS FD001 via UART (protocole binaire v3, flag RUL_MODE), collecte prédictions RUL board, calcule RMSE vs labels réels, logge latence DWT | 🔴 | ✅ | `scripts/simulate_rul_board.py` | 2h |
| S2610 | Créer `scripts/simulate_multiclass_board.py` : envoie échantillons CWRU (9 features, flag MULTICLASS_MODE), collecte classes prédites, calcule F1-macro on-board | 🟡 | ✅ | `scripts/simulate_multiclass_board.py` | 1h30 |

### O6 — Expériences board

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2611 | exp_S26_01 : EWC RUL board / CMAPSS FD001 → RMSE board + latence DWT (100 inférences) + SRAM .bss | 🔴 | ✅ | `experiments/exp_S26_01/` | 2h |
| S2612 | exp_S26_02 : EWC multi-class board / CWRU 3 tâches → F1-macro board + latence DWT + SRAM | 🟡 | ⚠️ | `experiments/exp_S26_02/` | 2h |
| S2613 | exp_S26_03 : RAM profiling board — comparer `ewc_head_regression.c` vs `ewc_head.c` FP32 vs `ewc_head_int8.c` (tailles .bss via linker map) | 🟡 | ✅ | `experiments/exp_S26_03/` | 1h |

### O7 — Tests C (compilation host)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2614 | Créer `test_ewc_regression.c` (Unity) : forward pass shape, MSE loss valeur, EWC penalty non nul après consolidation, gradient SGD signe correct | 🟡 | ✅ | `firmware/stm32f4_blink/tests/test_ewc_regression.c` | 1h30 |
| S2615 | Créer `test_ewc_multiclass.c` (Unity) : softmax sums to 1, argmax correct sur données mock, cross-entropy > 0, consolidation Fisher non nul | 🟡 | ✅ | `firmware/stm32f4_blink/tests/test_ewc_multiclass.c` | 1h30 |
| S2616 | `make test` — vérifier 0 régression sur tests existants après modifications pipeline.c + metrics.c | 🟡 | ⚠️ | `firmware/stm32f4_blink/` | 30 min |

### O8 — Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2617 | Tableau comparatif PC vs board : RMSE PC / RMSE board sur CMAPSS, F1-macro PC / F1-macro board sur CWRU, latences, RAM — insérer dans ce fichier section Bilan | 🟢 | ✅ | `docs/sprints/sprint_26/S2600_sprint_26.md` | 1h |
| S2618 | Mettre à jour `docs/roadmap_phase2.md` : Sprint 26 clôturé, contributions Gap 1 (CMAPSS RUL board) + Gap 2 (latence mesurée nouvelles têtes) | 🟢 | ✅ | `docs/roadmap_phase2.md` | 30 min |

---

## Ordre d'exécution recommandé

```
S2607 + S2608 (export poids PC → C headers)  ← prerequis sur résultats Sprint 25
     ↓
S2601 + S2602 (ewc_head_regression.c)
S2603 + S2604 (ewc_head_multiclass.c)    [parallèle]
     ↓
S2606 (metrics.c — OnlineRMSE + OnlineF1Macro)
     ↓
S2605 (pipeline.c — routing FLAGS)
     ↓
S2614 + S2615 (tests C host — sans board)
S2616 (make test — 0 régression)
     ↓
S2609 + S2610 (scripts simulation UART)
     ↓
make flash  → board connectée
S2611 → S2612 → S2613 (expériences board)
     ↓
S2617 + S2618 (docs)
```

---

## Nomenclature des expériences

| Exp ID | Modèle | Dataset | Tâche | Métriques board |
|--------|--------|---------|-------|----------------|
| exp_S26_01 | EWC Régression C | CMAPSS FD001 | RUL continu | RMSE, latence DWT µs, SRAM .bss Ko |
| exp_S26_02 | EWC Multi-class C | CWRU 3 tâches | 10 classes | F1-macro, latence DWT µs, SRAM .bss Ko |
| exp_S26_03 | EWC-reg vs EWC-fp32 vs EWC-int8 | — | — | Δ SRAM .bss (comparaison taille têtes) |

---

## Budget mémoire estimé (NUCLEO-F439ZI — 256 Ko SRAM)

| Composant | Taille FP32 | Notes |
|-----------|------------|-------|
| `ewc_head_regression.c` | ~10 Ko | 5→32→16→1, identique à ewc_head.c (même MLP) |
| `ewc_head_multiclass.c` (N=10) | ~11 Ko | 9→32→16→10, +10 neurones output |
| `ewc_head_multiclass.c` (N=3) | ~10.2 Ko | version Paderborn |
| `OnlineRMSE` struct | 12 B | Welford : n, mean, M2, rmse |
| `OnlineF1Macro` (N=10) | ~400 B | matrice confusion 10×10 int16 |
| Firmware existant (pipeline + hdc + maha + tinyol) | ~45 Ko | inchangé |
| **Total estimé (EWC-reg + EWC-mc-10)** | **~66 Ko** | << 256 Ko ✅ |

---

## Métriques attendues

| Expérience | Critère de validation |
|-----------|----------------------|
| exp_S26_01 (EWC RUL board) | RMSE board dans ±10% du RMSE PC exp_S25_01 ; latence ≤ 100 ms |
| exp_S26_02 (EWC multi-class board) | F1-macro board ≥ 0.60 ; latence ≤ 100 ms |
| exp_S26_03 (RAM) | ewc_regression ≈ ewc_fp32 en SRAM (même architecture, 1 sortie au lieu de 2) |
| Tests C | `make test` : `test_ewc_regression` et `test_ewc_multiclass` tous verts |

---

## Livrables

1. Firmware C : `ewc_head_regression.c/.h`, `ewc_head_multiclass.c/.h`
2. Pipeline étendu : `pipeline.c` (FLAGS RUL_MODE + MULTICLASS_MODE), `metrics.c` (OnlineRMSE + OnlineF1Macro)
3. C headers poids : `model_weights_rul.h`, `model_weights_multiclass.h`
4. Scripts Python host : `export_weights_ewc_rul.py`, `export_weights_ewc_multiclass.py`, `simulate_rul_board.py`, `simulate_multiclass_board.py`
5. Tests C : `test_ewc_regression.c`, `test_ewc_multiclass.c`
6. 3 dossiers `experiments/exp_S26_01/` à `exp_S26_03/` avec `results.json`
7. Tableau comparatif PC vs board (section Bilan de ce fichier)

---

## Décisions d'architecture prises pendant le sprint

### FLAGS byte — résolution conflit bit 7 (S2605, 6 juin 2026)

La spec initiale de S2605 assignait `PROTO_FLAG_RUL_MODE = 0x80U` (bit 7). Or ce bit est déjà occupé par `PROTO_FLAG_TINYOL_MODE = 0x80U` ajouté lors d'un sprint précédent.

**Décision** : garder `TINYOL_MODE = 0x80` intact, et utiliser des combinaisons de bits existants pour les nouveaux modes :

| Flag | Valeur | Justification |
|------|:------:|---------------|
| `PROTO_FLAG_RUL_MODE` | `0x50` (EWC\|INT8) | Combinaison libre — RUL n'utilise pas INT8 seul |
| `PROTO_FLAG_MULTICLASS_MODE` | `0x30` (EWC\|HDC) | Combinaison libre — multi-class n'utilise pas HDC seul |

Le routing `if/else` vérifie les combinaisons avant les flags simples pour garantir la priorité correcte. Zéro collision avec les modes existants.

> `TODO(dorra)` : Le byte FLAGS est maintenant totalement saturé. Tout nouveau mode post-Sprint 26 nécessite un protocole V4 avec FLAGS sur 2 octets.

---

## Questions ouvertes

- `TODO(arnaud)` : La démonstration board RUL (CMAPSS FD001) est-elle suffisante comme contribution Gap 2, ou faut-il aussi Pronostia board pour la revendication "données industrielles" ?
- `TODO(dorra)` : Pour la matrice de confusion 10×10 on-board (CWRU 10 classes), 400 B SRAM est acceptable — mais faut-il réduire à 3×3 (Paderborn) pour économiser RAM si d'autres modules sont ajoutés ?
- `TODO(fred)` : Les résultats RMSE board sur CMAPSS peuvent-ils être intégrés dans le benchmark Edge Spectrum initié au Sprint 23 ?
- `FIXME(gap2)` : La tête régression reste FP32 en Sprint 26 — annoter dans le manuscrit comme limite : pas d'INT8 pour régression continue (gradient non borné contrairement à classification).
- `FIXME(gap2)` : Si latence EWC multi-class (N=10 sorties) dépasse 100 ms, envisager réduire à N=3 (Paderborn) pour valider le critère.

---

## Bilan (complété 2026-06-06)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S2601 ewc_head_regression.c | ✅ | ~3h | `ewc_head_regression.c` + `.h` — 206 lignes, SGD+EWC MSE, pas de softmax |
| S2603 ewc_head_multiclass.c | ✅ | ~1h | `ewc_head_multiclass.c` + `.h` — softmax N classes, cross-entropy, EWC |
| S2605 pipeline.c étendu | ✅ | ~2h | `RUL_MODE=0x50`, `MULTICLASS_MODE=0x30`, buffer `PROTO_MAX_N=16` (fix MAHA_DIM=5) |
| S2606 metrics.c OnlineRMSE+F1Macro | ✅ | ~1h | Welford O(1) RMSE · cm[10][10] int16 F1-macro · 8 tests · **75/75 PASS** |
| S2607 export_weights_ewc_rul.py + model_weights_rul.h | ✅ | ~1h | Checkpoint model_ewc_reg.pt → 10Ko header C, memcpy dans pipeline_init() |
| S2608 export_weights_ewc_multiclass.py + model_weights_multiclass.h | ✅ | ~1h | Checkpoint model_ewc_mc.pt → 14Ko header C, memcpy dans pipeline_init() |
| S2614 test_ewc_regression.c | ✅ | ~30min | 5 tests Unity : forward, SGD, EWC penalty×2, consolidation |
| S2615 test_ewc_multiclass.c | ✅ | ~30min | 5 tests Unity : forward, softmax, argmax, EWC penalty×2 |
| S2616 make test 0 régression | ⚠️ | ~5min | **75 Tests, 2 Failures** (board réel 2026-06-12). 10/10 EWC reg+mc PASS ; 2 échecs **pré-existants hors périmètre** (`test_tinyol_*` — réf. hardcodées ≠ `model_weights.h` régénéré) |
| S2611 exp_S26_01 board RUL | ✅ | ~1h | **Run board réel 2026-06-12** : RMSE=21.23 (ratio=0.94 ≤ 1.10 ✅), lat=233µs ✅ |
| S2612 exp_S26_02 board multiclass | ⚠️ | ~1h | **Run board réel 2026-06-12** : F1-macro=**0.507 < 0.60 ❌** (critère non atteint), lat=403µs ✅. L'ancien 0.729 ✅ était un artefact du framing v3 cassé (21 vs 23 B) |
| S2613 exp_S26_03 RAM profiling | ✅ | ~15min | .bss=66.7Ko / 256Ko (74.5% disponible) |

### Tableau comparatif PC vs board (Sprint 26)

> Chiffres mesurés sur **board réel le 2026-06-12** (run authentique, framing v3 corrigé).

| Modèle | Dataset | RMSE PC | RMSE Board | Ratio | Latence DWT µs | SRAM .bss tête |
|--------|---------|---------|-----------|-------|---------------|-------------|
| EWC Régression | CMAPSS FD001 | 22.53 cycles | **21.23 cycles** | 0.94 ✅ | **233 µs** ✅ | 8.46 Ko |

| Modèle | Dataset | F1-macro PC | F1-macro Board | Critère | Latence DWT µs | SRAM .bss tête |
|--------|---------|-------------|----------------|---------|----------------|-------------|
| EWC Multi-class (N=10) | CWRU 3 tâches | 0.981 | **0.507** | ≥ 0.60 ❌ | **403 µs** ✅ | 11.76 Ko |

**Total firmware .bss : 66.7 Ko / 256 Ko (25.5% utilisé) ✅**

⚠️ **exp_S26_02 — critère F1 non atteint (0.507 < 0.60).** Voir `FIXME(gap1)` dans
`S2611_experiences_board.md` : diagnostic en cours (désalignement label/index ou dérive SGD online).

### Bugs corrigés pendant le sprint
- **Version UART** : `uart_receive_sample` filtrait `PROTO_VERSION_V2` uniquement → accepte V2 et V3
- **Buffer features** : `float raw[MAHA_DIM=5]` tronquait les 9 features CWRU → `float raw[PROTO_MAX_N=16]`
- **Échelle label RUL** : label encodé [0,255] mais modèle entraîné sur [0,1] → division par 255 côté board
- **Flush UART** : premier sample garbage après reset MCU → `reset_input_buffer()` + `sleep(1.0)` côté host
- **Framing réponse v3 (2026-06-12)** : `simulate_rul_board.py` et `simulate_multiclass_board.py`
  lisaient `RESPONSE_SIZE=21` alors que `uart_send_response_v3` émet **23 B** (champ `ram_b` u16
  oublié) → désynchronisation série progressive. Corrigé : 23 B + offsets `acc@11/auroc@15/fgt@19`.
