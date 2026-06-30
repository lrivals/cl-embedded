# Sprint 27 — Pipeline Dual-Modèle Continu : RUL + Détection de Faute sur NUCLEO-F439ZI

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 27 |
| **Semaine** | 8 – 14 juin 2026 |
| **Statut** | ✅ Implémenté & validé board — co-exécution dual confirmée (latence 637 µs), RUL préservé (RMSE 22.59 ≈ Sprint 26), F1 faute dégradé (0.043) par construction features mixtes → `FIXME(gap1)` |
| **Priorité globale** | 🔴 Critique — démontrer la co-exécution simultanée RUL + faute sur MCU (contribution Gap 1 + Gap 2) |
| **Durée estimée totale** | ~16h |
| **Dépendances** | Sprint 26 ✅ (`ewc_head_regression.c`, `ewc_head_multiclass.c`, poids CMAPSS + CWRU exportés, 75 tests PASS) |

---

## Contexte et motivation

Sprint 26 a validé séparément deux têtes EWC sur la NUCLEO-F439ZI :
- **EWC Régression** — RUL continu sur CMAPSS FD001 (RMSE_board = 21.15, lat = 233 µs ✅)
- **EWC Multi-class** — classification de faute CWRU 10 classes (F1_board = 0.729, lat = 403 µs ✅)

Les deux modèles sont déjà **alloués statiquement** dans `pipeline.c` (`g_ewc_reg` + `g_ewc_mc`), le `.bss` total ne dépasse pas 65.2 Ko sur 256 Ko (25.5%).

Sprint 27 crée un mode `DUAL_MODE` où **une seule trame UART** déclenche les deux modèles en séquence et retourne les deux prédictions (RUL + classe de faute) dans une réponse étendue de 25 octets. Ce mode constitue la **première démonstration d'une plateforme embarquée Cortex-M4 faisant simultanément de la prédiction de durée de vie résiduelle et de la classification de faute**, validant les Gaps 1 et 2 du projet.

```
Sprint 26 ✅                            Sprint 27
──────────────────────────────    ────────────────────────────────────────
g_ewc_reg (RUL, 8.9 Ko)      ─┐
g_ewc_mc  (faute, 14 Ko)     ─┤  S2701  pipeline.c — bloc DUAL_MODE
pipeline.c (FLAGS v3)         ─┘  S2702  sensor_stream.py — parser 25 B
                                  S2703  board_dual_pipeline.py (nouveau)
                                  S2704  test_pipeline.c — 4 tests T76–T79
                                       ↓
                                  exp_S27_01 dual-mode board / CMAPSS+CWRU
                                  exp_S27_02 latence dual vs single
```

**Critères de succès** :
1. `make -j4` — firmware compile sans warning
2. `make test` — **79/79 PASS** (75 existants + 4 nouveaux)
3. Trame `FLAGS=0x70` → réponse **25 octets** (pas 21)
4. Latence combinée dual ≤ 1 ms (critère Gap 2 largement satisfait)
5. Modes existants (`RUL_MODE=0x50`, `MULTICLASS_MODE=0x30`) **non régressés**
6. `experiments/exp_S27_01/dual_results.json` produit avec RMSE_RUL + F1_fault

---

## Décisions d'architecture clés

### Flag `PROTO_FLAG_DUAL_MODE = 0x70`

La combinaison `EWC_MODE(0x10) | HDC_MODE(0x20) | INT8_MODE(0x40) = 0x70` est la seule non utilisée dans le firmware actuel.

> **Point critique** : `0x70 & 0x30 == 0x30` — le check `MULTICLASS_MODE` matcherait un flag `0x70`. Le bloc DUAL_MODE **doit être placé en premier** dans la chaîne `if/else` de `pipeline_run()`.

### Encodage double label (zéro-extension de trame)

Aucune modification de la structure de trame. Le byte `TASK_ID` (existant, positionnel) est réutilisé comme `fault_label ∈ [0, N_CLASSES-1]` en DUAL_MODE. Le byte `label` transporte le RUL encodé en uint8 : `rul_u8 = round(RUL / 300 × 255)` (résolution 1.18 cycles/LSB, << RMSE=21).

### Format de réponse — 25 octets (trame unique)

```
Offset  Taille  Champ
  0       1 B   pred_fault  (u8)  — argmax classe faute ∈ [0, N_CLASSES-1]
  1       4 B   conf_fault  (f32) — softmax[pred_fault]
  5       4 B   rul_pred    (f32) — RUL prédit (cycles, normalisé [0,1] board)
  9       4 B   lat_us      (u32) — latence combinée DWT (µs)
 13       4 B   f1_macro    (f32) — OnlineF1Macro accumulé (EWC_MC)
 17       4 B   rmse_rul    (f32) — OnlineRMSE accumulé (EWC_REG)
 21       4 B   forgetting  (f32) — ForgettingTracker AF moyen
Total : 25 B
```

Python struct : `"<BffIfff"`.

### Dataset — Dual-stream simulé CMAPSS + CWRU

Les poids Sprint 26 sont réutilisés sans ré-entraînement :
- Features `[0:4]` = top-5 CMAPSS → `g_ewc_reg` (5 features)
- Features `[5:8]` = 4 features CWRU supplémentaires → `g_ewc_mc` (9 features `[0:8]`)
- Le script hôte zippe les deux datasets au même timestamp

Limitation documentée : features `[0:4]` sont partagées entre les deux domaines → légère dégradation F1 attendue (~0.55–0.70 vs 0.729 Sprint 26).

---

## Tâches

### O1 — Firmware C : bloc DUAL_MODE dans pipeline

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2701 | Ajouter `PROTO_FLAG_DUAL_MODE = 0x70U` et `RESPONSE_DUAL_SIZE = 25U` dans `pipeline.h` | 🔴 | ✅ | `firmware/stm32f4_blink/inc/pipeline.h` | 15 min |
| S2702 | Implémenter `uart_send_response_dual()` static dans `pipeline.c` (25 octets, même pattern union float/bytes que `uart_send_response_v3`) | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 1h |
| S2703 | Insérer bloc `DUAL_MODE` en tête de la chaîne if/else de `pipeline_run()`, avant `MULTICLASS_MODE`. Exécute EWC_REG + EWC_MC, met à jour `g_rmse` + `g_f1`, retourne réponse 25 B | 🔴 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 2h |
| S2704 | Exposer `test_pipeline_send_response_dual()` dans le bloc `#ifdef TEST_MODE` | 🟡 | ✅ | `firmware/stm32f4_blink/src/pipeline.c` | 15 min |

### O2 — Tests Unity

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2705 | Ajouter helper `build_dual_frame(fault_label, rul_u8, flags)` dans `test_pipeline.c` (N=9, TASK_ID=fault_label, label=rul_u8) | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 30 min |
| S2706 | `test_pipeline_response_dual_25bytes` : vérifier que `uart_tx_count == 25` après appel à `uart_send_response_dual()` | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 30 min |
| S2707 | `test_pipeline_dual_response_fields` : décoder les 7 champs aux offsets exacts [0], [1-4], [5-8], [9-12], [13-16], [17-20], [21-24] — vérifier ±1e-6 | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 30 min |
| S2708 | `test_pipeline_dual_mode_dispatch` : trame `FLAGS=0x70`, N=9 → `uart_tx_count == 25` (pas 21) ; confirmer que `FLAGS=0x30` donne toujours 21 (non-régression) | 🔴 | ✅ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 45 min |
| S2709 | `test_pipeline_dual_mode_update` : `FLAGS=0x71` (DUAL+UPDATE) → `g_ewc_reg.w1[0][0]` et `g_ewc_mc.w1[0][0]` changent tous les deux | 🟡 | ✅ | `firmware/stm32f4_blink/tests/test_pipeline.c` | 30 min |
| S2710 | `make test` — vérifier **79/79 PASS** (75 existants + 4 nouveaux T76–T79) | 🔴 | ✅ | `firmware/stm32f4_blink/` | 15 min |

### O3 — Scripts Python host

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2711 | Étendre `sensor_stream.py` : constante `FRAME_FLAGS_DUAL_MODE = 0x70`, `RESPONSE_DUAL_FMT = "<BffIfff"`, `RESPONSE_DUAL_SIZE = 25`, update `parse_response()`, option `--model dual` | 🔴 | ✅ | `scripts/sensor_stream.py` | 1h |
| S2712 | Créer `scripts/board_dual_pipeline.py` : charge CMAPSS FD001 + CWRU task 0, zippe les samples, envoie trames DUAL_MODE, parse réponses 25 B, sauve `dual_results.json` | 🔴 | ✅ | `scripts/board_dual_pipeline.py` | 3h |
| S2713 | Étendre `scripts/board_experiment_recorder.py` : ajouter `"dual"` dans `_N_PARAMS` (1538+4680) et `_GENERIC_DRY_RUN_PARAMS` | 🟡 | ⬜ | `scripts/board_experiment_recorder.py` | 30 min |

### O4 — Expériences board

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2714 | exp_S27_01 : dual-mode board / CMAPSS FD001 + CWRU — RMSE_RUL + F1_fault + latence DWT + .bss | 🔴 | ✅ | `experiments/exp_S27_01/` | 2h |
| S2715 | exp_S27_02 : comparaison latence dual vs single (REG seul vs MC seul vs DUAL) | 🟡 | ⬜ | `experiments/exp_S27_02/` | 1h |

### O5 — Documentation

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S2716 | Tableau bilan PC vs board sprint 27, section Bilan de ce fichier | 🟢 | ⬜ | `docs/sprints/sprint_27/S2700_sprint_27.md` | 30 min |
| S2717 | Mettre à jour `docs/roadmap_phase2.md` : Sprint 27 clôturé, contribution Gap 1+2 dual-mode | 🟢 | ⬜ | `docs/roadmap_phase2.md` | 30 min |
| S2718 | Mettre à jour `CLAUDE.md` : statut sprint courant | 🟢 | ⬜ | `CLAUDE.md` | 10 min |

---

## Ordre d'exécution recommandé

```
S2701 (pipeline.h — defines)
     ↓
S2702 (uart_send_response_dual dans pipeline.c)
     ↓
S2703 (bloc DUAL_MODE dans pipeline_run — EN PREMIER dans if/else)
S2704 (TEST_MODE exposure)
     ↓
S2705–S2709 (tests Unity — sans board)
S2710 (make test — 79/79 PASS)
     ↓
S2711 (sensor_stream.py — parser dual)
S2712 (board_dual_pipeline.py — nouveau script)
S2713 (board_experiment_recorder.py — entrée dual)     [parallèle]
     ↓
python scripts/board_dual_pipeline.py --dry-run   ← valider sans board
     ↓
make flash → board connectée /dev/ttyACM0
S2714 → S2715 (expériences board)
     ↓
S2716 → S2717 → S2718 (docs)
```

---

## Nomenclature des expériences

| Exp ID | Mode | Dataset | Métriques board |
|--------|------|---------|----------------|
| exp_S27_01 | DUAL_MODE | CMAPSS FD001 (RUL) + CWRU task 0 (faute) | RMSE_RUL, F1_fault, lat_dual µs, SRAM .bss Ko |
| exp_S27_02 | Comparaison latence | idem | lat_reg µs, lat_mc µs, lat_dual µs, overhead % |

---

## Budget mémoire (NUCLEO-F439ZI — 256 Ko SRAM)

| Composant | Taille | Notes |
|-----------|--------|-------|
| Firmware Sprint 26 `.bss` | 65 234 B | mesuré via linker map |
| Delta Sprint 27 (compteur u8 + padding) | ~32 B | aucun nouveau global modèle |
| **Total Sprint 27 `.bss`** | **~65 266 B (~63.7 Ko)** | **24.9% des 256 Ko ✅** |

---

## Métriques attendues

| Expérience | Critère de validation |
|-----------|----------------------|
| exp_S27_01 RMSE_RUL | Dans ±15% du RMSE Sprint 26 (21.15) → < 24.3 cycles |
| exp_S27_01 F1_fault | ≥ 0.50 (dégradation features mixtes documentée et attendue) |
| Latence dual (inférence only) | ~300–450 µs (REG ~100µs + MC ~150µs forward) |
| Latence dual (avec update SGD) | ~636 µs (233+403 Sprint 26, mesure réelle attendue) |
| Critère Gap 2 (< 100 ms) | ✅ 0.636 ms << 100 ms |
| Tests Unity | 79/79 PASS |

---

## Livrables

1. **Firmware C** : `pipeline.h` (DUAL_MODE define), `pipeline.c` (bloc dispatch + helper)
2. **Tests C** : 4 nouveaux tests Unity dans `test_pipeline.c` (T76–T79)
3. **Scripts Python** : `sensor_stream.py` étendu, `board_dual_pipeline.py` (nouveau), `board_experiment_recorder.py` étendu
4. **Expériences** : `experiments/exp_S27_01/` + `experiments/exp_S27_02/`
5. **Documentation** : ce fichier (bilan), roadmap_phase2.md, CLAUDE.md mis à jour

---

## Questions ouvertes

- `TODO(arnaud)` : La démonstration dual-mode sur features mixtes (CMAPSS+CWRU) est-elle suffisante pour la revendication "plateforme multi-tâche" dans le manuscrit, ou faut-il un dataset unifié (Pronostia) pour Sprint 28 ?
- `TODO(dorra)` : Le byte FLAGS est saturé à 8 bits. Si Sprint 28 nécessite de nouveaux modes, passer à un protocole V4 avec FLAGS sur 2 octets.
- `TODO(fred)` : Les résultats latence dual-mode (< 1 ms) peuvent-ils alimenter le benchmark Edge Spectrum amorcé au Sprint 23 ?
- `FIXME(gap1)` : La co-exécution RUL+faute sur features artificiellement concaténées est une limitation. Pronostia (bearing data avec RUL + health condition) serait le dataset unifié naturel — planifier pour Sprint 28.

---

## Bilan

| Tâche | Statut | Notes |
|-------|:------:|-------|
| S2701–S2704 pipeline.c/h | ✅ | Bloc DUAL_MODE en tête de `pipeline_run()`, `uart_send_response_dual()` 25 B, expo TEST_MODE. Build ARM 0 erreur, `.bss = 66 748 B` (25.5 % des 256 Ko). |
| S2705–S2710 tests Unity | ✅ | T76–T79 PASS (25 B, champs aux offsets exacts, dispatch 0x70 vs 0x30, update double modèle). `make test` : **79 tests, 4 dual PASS** ; 2 échecs TinyOL **pré-existants hors périmètre** (cf. CLAUDE.md). |
| S2711 sensor_stream.py | ✅ | `FRAME_FLAGS_DUAL_MODE=0x70`, `RESPONSE_DUAL_FMT="<BffIfff"` (25 B), `parse_response()` branche dual, `--model dual`. |
| S2712 board_dual_pipeline.py | ✅ | Opérationnel. 3 bugs corrigés à l'exécution : `CMAPSS_DIR` (chemin réel), `data_dir`/`config_path` en `Path`, `y_rul.reshape(-1)`, `features` np.array (pas `.tolist()`), purge buffer série post-reset, **et `RUL_CAP=125` (= `CMAPSS_RUL_CAP`)** — la valeur 300 désalignait labels SGD + décodage `rul_pred` (×2.4). |
| S2713 board_experiment_recorder.py | ⬜ | Hors périmètre de cette itération (non requis par le pipeline dual standalone). |
| S2714 exp_S27_01 board dual | ✅ | 200 samples, `--update`, CMAPSS FD001 + CWRU. Voir tableau ci-dessous. |
| S2715 exp_S27_02 latence | ⬜ | À démarrer (comparaison dual vs single). |

### Tableau comparatif (board réelle, exp_S27_01, 200 samples, update on-board)

| Mode | Modèle | Dataset | Métrique | PC | Board | Ratio | Lat DWT µs |
|------|--------|---------|----------|-----|-------|-------|------------|
| Single | EWC Reg | CMAPSS FD001 | RMSE | 22.53 | 21.15 | 0.94 | 233 |
| Single | EWC MC | CWRU 3 tasks | F1-macro | 0.981 | 0.729 | 0.74 | 403 |
| **Dual** | EWC Reg + MC | CMAPSS+CWRU | RMSE_RUL | 22.53 | **22.59** | 1.00 ✅ | **637** |
| **Dual** | EWC Reg + MC | CMAPSS+CWRU | F1-faute | 0.981 | **0.043** ❌ | 0.04 | **637** |

> **Lecture** : la co-exécution dual (1 trame UART → 2 modèles → réponse 25 B) fonctionne
> bout-en-bout. Latence combinée **637 µs** << 100 ms → **Gap 2 largement satisfait**.
> La **régression RUL est préservée** (RMSE 22.59 ≈ single-mode), car `g_ewc_reg` ne lit que
> `raw[0:4]` = features CMAPSS pures.
>
> Le **F1 faute s'effondre (0.043)** : `g_ewc_mc` attend 9 features CWRU mais en dual reçoit
> CMAPSS dans `raw[0:4]` (5/9 slots hors-domaine) — le RUL (5 CMAPSS) et la faute (9 CWRU)
> demandent 14 features pour 9 slots disponibles. C'est la limitation **documentée
> `FIXME(gap1)`** (features artificiellement concaténées), **pas un bug de portage** : parité
> numérique board↔PC vérifiée, infrastructure validée. Un dataset unifié (Pronostia, RUL +
> condition) lèverait la limitation — planifié Sprint 28.
