# Sprint 29 — INT8 Firmware Board + Synthèse Scientifique Gap 3

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 29 |
| **Semaine** | 23–27 juin 2026 |
| **Statut** | 🟡 En cours — S2901/S2902/S2903 ✅ (firmware INT8) · **S2904/S2905 ✅ (15 juin, 5 expériences board)** · **S2906/S2907 ✅ (15 juin, 10 tests Unity INT8 PASS)** · **S2908 ⛔ bloqué (CMSIS-DSP absent de la toolchain)** · reste S2909–S2911 |
| **Priorité globale** | 🔴 Critique — porter HDC+TinyOL INT8 en firmware C, mesurer sur NUCLEO-F439ZI, finaliser Gap 3 multi-modèle pour le manuscrit |
| **Durée estimée totale** | ~24h |
| **Dépendances** | Sprint 28 ✅ (modèles INT8 Python, résultats PC) · Sprint 23 ✅ (`ewc_head_int8.c`, `pipeline.c` v3) · Sprint 26 ✅ (FLAGS protocol, `metrics.c`) |

---

## Contexte et motivation

Sprint 28 a produit les modèles INT8 Python pour HDC, TinyOL et Mahalanobis et les résultats PC. Sprint 29 porte cette analyse sur le **firmware NUCLEO-F439ZI** (Cortex-M4 @ 180 MHz, 256 Ko SRAM) pour :

1. Mesurer la latence DWT réelle INT8 vs FP32 sur hardware (Cortex-M4 FPU sans SIMD INT8 natif)
2. Documenter honnêtement le résultat négatif latence ET le résultat positif RAM ×2.7–4.0×
3. Explorer CMSIS-DSP SIMD INT8 pour une potentielle accélération (`arm_dot_prod_q7`)
4. Produire le notebook de synthèse PC+board et la mise à jour `docs/triple_gap.md` pour le manuscrit

**Résultat latence Sprint 23 connu (EWC INT8 CMAPSS board)** :

| Métrique | EWC FP32 | EWC INT8 | Attendu (Cortex-M4 FPU) |
|---------|:--------:|:--------:|:------------------------:|
| Latence DWT | 0.251 ms | 0.461 ms | INT8 **plus lent** — FPU optimisé FP32, pas de SIMD INT8 |
| RAM poids | 9 728 B | 3 600 B | ×2.7 réduction ✅ |

**Note architecturale** : Le Cortex-M4 FPU exécute les opérations FP32 en 1 cycle. Les opérations INT8 nécessitent des instructions LDRSH + multiplication entière sans parallélisme SIMD → INT8 firmware est plus lent. Ce résultat négatif est une contribution scientifique honnête : aucun travail précédent ne l'a mesuré sur MCU avec CL.

```
Sprint 28 ✅                         Sprint 29
──────────────────────────────    ──────────────────────────────────────────
hdc_int8.py (Python)         ──▶  S2901 hdc_int8.c + hdc_int8.h
tinyol_int8.py (Python)      ──▶  S2902 tinyol_int8.c + tinyol_int8.h
exp_S28_PC results           ──▶  contexte pour validation board
pipeline.c v3 (Sprint 23)   ──▶  S2903 pipeline.c (FLAGS HDC_INT8, TINYOL_INT8)
                                          ↓
                               S2904 exp_S29_board_int8/ (EWC×2 + HDC×2 datasets)
                               S2905 exp_S29_board_int8/ (TinyOL×1 dataset)
                                          ↓
                               S2906 tests/test_hdc_int8.c (Unity)
                               S2907 tests/test_tinyol_int8.c (Unity)
                                          ↓ (optionnel)
                               S2908 ewc_head_int8_simd.c (CMSIS-DSP exploratoire)
                                          ↓
                               S2909 notebooks/sprint29_int8_board.ipynb
                               S2910 docs/triple_gap.md (Gap 3 multi-modèle)
                               S2911 docs/roadmap_phase2.md
```

**Critères de succès** :
1. `make -j4` — firmware compile avec `hdc_int8.c` + `tinyol_int8.c` intégrés (0 warning)
2. `make test` — 0 failures Unity (incluant `test_hdc_int8.c` + `test_tinyol_int8.c`)
3. `experiments/exp_S29_board_int8/results_*.json` — latence DWT mesurée pour ≥3 modèles INT8
4. `arm-none-eabi-size build/stm32f4_blink.elf` — .bss total < 128 Ko (marge ×2 du budget Sprint 26 66.7 Ko)
5. `docs/triple_gap.md` — section Gap 3 mise à jour avec tableau 4 modèles
6. Notebook `sprint29_int8_board.ipynb` exécutable end-to-end

---

## Tâches

### O1 — Firmware C INT8 HDC

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2901 | Implémenter `firmware/stm32f4_blink/src/hdc_int8.c` + `inc/hdc_int8.h` : encode hypervecteurs base `int8_t` (±1), AM `int16_t`, forward query accumulateur `int32_t`, pattern identique à `ewc_head_int8.c`. Allocation statique, pas de malloc. | 🔴 | ✅ | `firmware/stm32f4_blink/src/hdc_int8.c`, `firmware/stm32f4_blink/inc/hdc_int8.h` | 3h |

### O2 — Firmware C INT8 TinyOL

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2902 | Implémenter `firmware/stm32f4_blink/src/tinyol_int8.c` + `inc/tinyol_int8.h` : autoencoder activations `uint8_t`, OtOHead poids `int8_t`, update delta-weight INT8 simulé (dequant → gradient FP32 → requant). Allocation statique. | 🟡 | ✅ | `firmware/stm32f4_blink/src/tinyol_int8.c`, `firmware/stm32f4_blink/inc/tinyol_int8.h` | 3h |

### O3 — Intégration pipeline

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2903 | Étendre `pipeline.c` : ajouter `PROTO_FLAG_HDC_INT8 0x60U` (HDC\|INT8) et `PROTO_FLAG_TINYOL_INT8 0xC0U` (TINYOL\|INT8 — **pas** `0x22`/`0x81` : collision PROFILING/UPDATE), brancher HDC INT8 (encode/predict/update) et TinyOL INT8 (encode + OtO predict/update) AVANT le check INT8_MODE. Mettre à jour `sensor_stream.py` (`--model hdc-int8\|tinyol-int8`). | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py` | 2h |

### O4 — Expériences board

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2904 | exp_S29_board_int8 — EWC INT8 sur CWRU + Pronostia (CMAPSS déjà fait S23) ; HDC INT8 sur CMAPSS + Monitoring. Mesurer latence DWT + RAM .bss + metric (AUROC/acc). Sauver `results_*.json`. | 🔴 | ✅ | `experiments/exp_S29_board_int8/` + `scripts/run_s29_board_int8.py` | 4h |
| S2905 | exp_S29_board_int8 — TinyOL INT8 sur CWRU (anomaly detection). Mesurer latence DWT + RAM .bss + AUROC. Comparer au TinyOL FP32 Sprint 20. | 🟡 | ✅ | `experiments/exp_S29_board_int8/` | 2h |

### O5 — Tests Unity C

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2906 | `firmware/stm32f4_blink/tests/test_hdc_int8.c` (Unity) : encode produit valeurs dans `[-128, 127]` ; query retourne bonne classe après 10 updates ; budget RAM .bss cumulé < 128 Ko avec HDC INT8 ajouté | 🟡 | ⬜ | `firmware/stm32f4_blink/tests/test_hdc_int8.c` | 2h |
| S2907 | `firmware/stm32f4_blink/tests/test_tinyol_int8.c` (Unity) : forward UINT8 valeurs dans `[0, 255]` ; erreur reconstruction INT8 ≈ FP32 (delta < 5%) ; RAM .bss global < 128 Ko | 🟡 | ⬜ | `firmware/stm32f4_blink/tests/test_tinyol_int8.c` | 2h |

### O6 — Exploratoire CMSIS-DSP (si temps disponible)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2908 | Prototype `firmware/stm32f4_blink/src/ewc_head_int8_simd.c` : utiliser `arm_dot_prod_q7()` de CMSIS-DSP pour le produit scalaire INT8 dans le forward EWC. Mesurer latence DWT vs version scalaire. Documenter gain/perte. | 🟢 | ⬜ | `firmware/stm32f4_blink/src/ewc_head_int8_simd.c` | 2h |

### O7 — Documentation scientifique

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2909 | Notebook `notebooks/sprint29_int8_board.ipynb` : synthèse PC (Sprint 28) + board (Sprint 29), tableau comparatif 4 modèles × {latence_fp32, latence_int8, ram_fp32, ram_int8, delta_metric} PC + board, conclusion Gap 3 multi-modèle formelle | 🔴 | ⬜ | `notebooks/sprint29_int8_board.ipynb` | 3h |
| S2910 | Mise à jour `docs/triple_gap.md` — section Gap 3 : étendre EWC-only (S23) → 4 modèles, tableau `gap3_{ram,latency,metric}_met` par modèle×dataset, note résultat négatif latence honnête, note CMSIS-DSP exploratoire S2908 | 🔴 | ⬜ | `docs/triple_gap.md` | 2h |
| S2911 | Mise à jour `docs/roadmap_phase2.md` — marquer Sprint 28 ✅ + Sprint 29 ✅, mettre à jour statut Gap 3 → `✅ COMPLET multi-modèle (4 modèles × 5 datasets)` | 🟡 | ⬜ | `docs/roadmap_phase2.md` | 1h |

---

## Ordre d'exécution recommandé

```
S2901 (hdc_int8.c)         S2902 (tinyol_int8.c)    [parallèle]
     ↓                              ↓
S2903 (pipeline.c FLAGS + sensor_stream.py)
     ↓
S2906 (test_hdc_int8.c)    S2907 (test_tinyol_int8.c)   [parallèle]
make test → 0 failures
     ↓
make flash → board connectée
S2904 (exp board EWC+HDC)  S2905 (exp board TinyOL)   [si board dispo]
     ↓
S2908 (CMSIS-DSP, 🟢 optionnel)
     ↓
S2909 (notebook synthèse PC+board)
     ↓
S2910 (triple_gap.md)
S2911 (roadmap)
```

---

## Nomenclature des expériences

| Exp ID | Modèles | Datasets | Métriques board |
|--------|---------|----------|----------------|
| exp_S29_board_ewc_int8_cwru | EWC INT8 | CWRU | AUROC, latence DWT µs, RAM .bss B |
| exp_S29_board_ewc_int8_pronostia | EWC INT8 | Pronostia | AUROC, latence DWT µs, RAM .bss B |
| exp_S29_board_hdc_int8_cmapss | HDC INT8 | CMAPSS | acc_final, latence DWT µs, RAM .bss B |
| exp_S29_board_hdc_int8_monitoring | HDC INT8 | Monitoring | acc_final, latence DWT µs, RAM .bss B |
| exp_S29_board_tinyol_int8_cwru | TinyOL INT8 | CWRU | AUROC, latence DWT µs, RAM .bss B |

---

## Budget mémoire firmware estimé (post-Sprint 29)

| Composant | RAM .bss estimé | Notes |
|-----------|:--------------:|-------|
| Firmware Sprint 26 existant | 66.7 Ko | ewc_head×3 + pipeline + metrics + HDC FP32 + Maha + TinyOL |
| `hdc_int8.c` (D=2048 `int8_t` AM) | ~4.5 Ko | AM `int16_t` D=2048 = 4 Ko + base vectors `int8_t` D×n |
| `tinyol_int8.c` (enc 9→32→16 UINT8) | ~1.5 Ko | activation buffers UINT8 + poids INT8 |
| **Total estimé** | **~72.7 Ko** | << 256 Ko ✅ (28.4% SRAM) |

---

## Notes d'implémentation

**S2901 `hdc_int8.c`** : Pattern à suivre — `firmware/stm32f4_blink/src/ewc_head_int8.c`. Différences clés : les hypervecteurs de base (`int8_t base_vecs[HDC_N_FEATURES][HDC_D]`) sont déjà binarisés, l'AM (`int16_t am[HDC_N_CLASSES][HDC_D]`) accumule des bundles entiers. Attention : si HDC_D=2048 et HDC_N_CLASSES=4, AM = 4×2048×2 = 16 Ko .bss.

**S2903 pipeline.c** : Le byte FLAGS est saturé (documenté S2600 — TODO dorra). Utiliser des combinaisons de bits existants sans collision : `HDC_INT8 = HDC_MODE | INT8_MODE = 0x20 | 0x02 = 0x22` (si bits disponibles). Vérifier l'absence de collision avec `RUL_MODE=0x50` et `MULTICLASS_MODE=0x30` de Sprint 26.

**S2908 CMSIS-DSP** : Vérifier d'abord `arm-none-eabi-gcc --print-file-name=libarm_cortexM4lf_math.a` — si absent, la tâche est bloquée. Sinon : lier avec `-larm_cortexM4lf_math -lm` dans le Makefile. La fonction `arm_dot_prod_q7(pSrcA, pSrcB, blockSize, result)` accumule un produit scalaire q7×q7 en q31.

---

## Questions ouvertes

- `TODO(dorra)` : Confirmer présence `libarm_cortexM4lf_math.a` (CMSIS-DSP) dans la toolchain actuelle avant de lancer S2908.
- `TODO(arnaud/fred)` : Résultat négatif latence INT8 confirmé pour EWC (S23) → attendu pour HDC et TinyOL aussi sur Cortex-M4 FPU. Si confirmé, recentrer message Gap 3 exclusivement sur réduction RAM pour le manuscrit ?
- `FIXME(gap3)` : Les opérations HDC sont nativement entières — HDC "INT8 board" mesure la RAM, pas la latence. Distinguer clairement dans le manuscrit "quantification des poids" vs "opérations INT8 natives".

---

## Livrables

1. Firmware C : `hdc_int8.c/.h`, `tinyol_int8.c/.h`
2. `pipeline.c` étendu : FLAGS `HDC_INT8` + `TINYOL_INT8` + `sensor_stream.py` mis à jour
3. Tests Unity : `test_hdc_int8.c`, `test_tinyol_int8.c` — `make test` 0 failures
4. 5 répertoires `experiments/exp_S29_board_int8_*/` avec `results.json`
5. (Optionnel) `ewc_head_int8_simd.c` — prototype CMSIS-DSP
6. `notebooks/sprint29_int8_board.ipynb` — notebook synthèse exécutable
7. `docs/triple_gap.md` — Gap 3 mis à jour multi-modèle
8. `docs/roadmap_phase2.md` — S28+S29 ✅

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S2901 hdc_int8.c + .h | ✅ | — | HDCInt8 bv int8[9][2048] + am int16[4][2048], encode signé, predict int32 argmax, update saturé ±32767. LCG seed fixe. |
| S2902 tinyol_int8.c + .h | ✅ | — | Encodeur poids INT8 (fake-quant Q7) + activations UINT8 ; tête OtO INT8 SGD BCE (maîtres FP32). Init depuis model_weights.h. |
| S2903 pipeline.c + sensor_stream.py | ✅ | — | FLAGS `0x60` (HDC_INT8) + `0xC0` (TINYOL_INT8), branches avant check INT8_MODE. sensor_stream `--model hdc-int8\|tinyol-int8`. Makefile +2 .c. `make all` OK, .bss=104 576 B (102 Ko) < 128 Ko ✅. `make test` 79/2 pré-existants. Board flashée (Verified OK). |
| S2904 exp board EWC+HDC | ✅ | 15/06 | EWC ×1.84 lat / ×2.70 RAM (=S23) · HDC ×3.26 lat / ×3.06 RAM · 4 JSON |
| S2905 exp board TinyOL | ✅ | 15/06 | TinyOL CWRU export+flash · AUROC 0.992 · ×4.0 RAM · caveat chemins FP32≠INT8 |
| S2906 test_hdc_int8.c | ⬜ | — | — |
| S2907 test_tinyol_int8.c | ⬜ | — | — |
| S2908 CMSIS-DSP exploratoire | ⬜ | — | — |
| S2909 notebook sprint29 | ⬜ | — | — |
| S2910 triple_gap.md | ⬜ | — | — |
| S2911 roadmap_phase2.md | ⬜ | — | — |
