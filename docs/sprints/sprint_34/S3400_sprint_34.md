# Sprint 34 — Streaming/buffer & Q15 Mahalanobis

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 34 |
| **Semaine** | 28 juillet – 3 août 2026 |
| **Statut** | ⬜ À démarrer |
| **Priorité globale** | 🔴 Critique — (A) étudier le dimensionnement du buffer et le débit de streaming temps-réel ; (B) implémenter le fallback **Q15 (int16)** de `sigma_inv_` pour Mahalanobis (Python + board) afin de récupérer l'AUROC perdu en INT8. Répond aux CR du 19 mai 2026 (buffer/débit) et du 9 juin 2026 (Q15 Mahalanobis, `TODO(arnaud)`). |
| **Durée estimée totale** | ~25h |
| **Dépendances** | `scripts/sensor_stream.py` ✅ (`rate_hz`) · `firmware/stm32f4_blink/src/hdc.c` ✅ (ring buffer existant) · `src/models/unsupervised/mahalanobis_int8.py` ✅ (INT8) · Sprint 28 (constat dégradation Maha INT8 : CWRU −0.236, Pronostia −0.238) · infra board parité Sprints 26-28 ✅ |

---

## Contexte et motivation

Deux pistes des CR restent non implémentées et non testées :

### Partie A — Streaming & buffer (CR 19 mai 2026)

Le CR demande : « évaluer la latence pour estimer combien de données on peut streamer sur la carte en même temps, par modèle », « étudier le remplissage du buffer », « impact du stride S sur latence perçue et charge CPU à étudier ». État vérifié : seul un **ring buffer spécifique à HDC** existe (`hdc.c`, `buf_head % HDC_RETRAIN_BUF`) ; `sensor_stream.py` expose `rate_hz` mais **aucune étude de saturation** débit/latence ni d'abstraction buffer générique `(W, S)`. Les formules du CR (`Débit_max = 1/Latence_inf`, `Débit_streaming = f_acq × S/W`, contrainte `W × sizeof(sample)` ≤ SRAM) ne sont pas instrumentées.

### Partie B — Q15 Mahalanobis (CR 9 juin 2026)

Le benchmark FP32 vs INT8 (Sprint 28) a montré que la quantification **INT8 de `sigma_inv_`** (matrice de précision, grande dynamique) **casse la distance de Mahalanobis** : AUROC −0.236 sur CWRU, −0.238 sur Pronostia. Le CR et le code recommandent un **fallback Q15 (int16)** : `TODO(arnaud)` dans `src/models/unsupervised/mahalanobis_int8.py:96-97`. État : non implémenté ; pas de Mahalanobis INT8/Q15 en C.

Décisions validées (utilisateur) :
- **Découpage** : ce sprint = streaming/buffer + Q15 Mahalanobis (l'énergie/coût est au **Sprint 33**).
- **Q15 jusqu'au board** : Python **et** portage C (`mahalanobis_q15.c`), avec parité board↔PC (pattern des autres modèles INT8).

```
CR 19 mai (débit streaming, buffer W/S, SRAM)         Sprint 34 — Partie A
                                              ──▶  S3401 streaming_model.py (Débit_max/streaming)
                                                   S3402 ring_buffer.c/.h (abstraction W,S)
                                                   S3403 exp board débit/buffer (saturation)
                                                   S3404 notebook streaming

CR 9 juin (Maha INT8 cassé → Q15, TODO arnaud)        Sprint 34 — Partie B
                                              ──▶  S3405 mahalanobis_int8.py quant="q15"
                                                   S3406 exp PC Q15 vs INT8 vs FP32 (récup AUROC)
                                                   S3407 mahalanobis_q15.c + pipeline (FLAG)
                                                   S3408 exp board Q15 (parité board↔PC)
                                                   S3409 tests + docs
```

---

## Critères de succès

1. `streaming_model.py` calcule `Débit_max`, `Débit_streaming`, la marge temps-réel et le budget buffer `W × sizeof(sample)` vs SRAM, paramétrés par config.
2. Abstraction `ring_buffer.c/.h` réutilisable `(W, S)`, **0 malloc**, tailles en `#define`, tests Unity verts ; HDC peut s'appuyer dessus sans régression.
3. Expérience board débit/buffer avec **point de saturation** identifié (drop/timeout) et latence DWT par config de stride/rate.
4. Q15 Mahalanobis : **ΔAUROC < 0.02** récupéré sur CWRU et Pronostia (vs −0.236 / −0.238 en INT8), **sans régresser** FP32 ni INT8 sur les autres datasets.
5. `mahalanobis_q15.c` : **parité board↔PC** des distances/prédictions, latence < 100 ms (Gap 2), `.bss` mesuré.
6. `pytest tests/ -k "streaming or mahalanobis_q15"` verts ; `make test` Unity verts ; aucun chiffre board inventé (champs « à mesurer » tant que non exécuté).

---

## Tâches

### O1 — Modèle de streaming/buffer (PC)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3401 | `src/evaluation/streaming_model.py` : `debit_max = 1/latence_inf`, `debit_streaming = f_acq × S/W`, vérification `debit_streaming ≤ debit_max` (marge temps-réel), budget buffer `W × sizeof(sample)` vs SRAM. Paramètres (W, S, f_acq, sizeof, SRAM) dans `configs/streaming_profile.yaml`. | 🔴 | `src/evaluation/streaming_model.py`, `configs/streaming_profile.yaml` | 3h |

### O2 — Abstraction buffer circulaire firmware

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3402 | Généraliser le buffer circulaire en abstraction réutilisable `(W, S)` : `ring_buffer.c/.h` (extrait/généralisé du buffer HDC), API push/window/stride, tailles en `#define` dans le header, **0 malloc**. HDC peut le réutiliser sans changer de comportement. | 🟡 | `firmware/stm32f4_blink/src/ring_buffer.c`, `firmware/stm32f4_blink/inc/ring_buffer.h` | 3h |

### O3 — Expérience board débit/buffer

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3403 | **Expérience board** : balayage `rate_hz` / stride via `sensor_stream.py`, mesure latence DWT, remplissage buffer et **point de saturation** (drop/timeout/CRC errors), `.bss` par configuration W. Enregistrement `experiments/exp_S34_streaming/`. RAM profiling par config de buffer. | 🔴 | `scripts/sensor_stream.py`, `experiments/exp_S34_streaming/` | 4h |

### O4 — Notebook streaming

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3404 | `notebooks/cl_eval/streaming/comparison.ipynb` : débit max vs débit acquisition par modèle, courbes latence vs stride, occupation buffer vs W, frontière temps-réel, note sur le multi-stream concurrent (analytique). Synthèse écrite. | 🟡 | `notebooks/cl_eval/streaming/comparison.ipynb` | 2h |

### O5 — Q15 Mahalanobis (Python)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3405 | `mahalanobis_int8.py` : mode `quant="q15"` quantifiant `sigma_inv_` en **int16 (Q15)** avec scale par-tenseur adapté à la grande dynamique ; `mu_` reste INT8. Lève le `TODO(arnaud):96-97`. Met à jour `get_memory_footprint()` (Q15 = 2× FP32 économie au lieu de 4×, mais distance préservée). | 🔴 | `src/models/unsupervised/mahalanobis_int8.py` | 3h |
| S3406 | **Expérience PC** Q15 vs INT8 vs FP32 sur CWRU + Pronostia (cibles) + 3 autres datasets (non-régression) : AUROC, ΔAUROC, RAM. Cible : ΔAUROC < 0.02 sur CWRU/Pronostia. Configs `configs/mahalanobis_q15_{dataset}.yaml`. RAM profiling. Enregistrement `experiments/exp_S34_maha_q15/`. | 🔴 | `experiments/exp_S34_maha_q15/`, `configs/mahalanobis_q15_*.yaml` | 3h |

### O6 — Q15 Mahalanobis (portage C board)

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3407 | Portage C : `mahalanobis_q15.c/.h` (`sigma_inv` int16 Q15, accumulation distance en int32/FP32), intégration `pipeline.c` (FLAG `MAHA_Q15_MODE`), branche Q15 dans `sensor_stream.py`. Poids exportés via `export_weights_c.py` (ne pas éditer le header à la main). | 🔴 | `firmware/stm32f4_blink/src/mahalanobis_q15.c`, `firmware/stm32f4_blink/inc/mahalanobis_q15.h`, `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py` | 4h |
| S3408 | **Expérience board** Q15 : parité board↔PC des distances/prédictions (protocole Sprints 26-28), latence DWT, `.bss`. Enregistrement `experiments/exp_S34_board_maha_q15/`. | 🔴 | `experiments/exp_S34_board_maha_q15/` | 2h |

### O7 — Tests + docs

| ID | Tâche | Priorité | Statut | Fichier(s) cible(s) | Durée est. |
|----|-------|:--------:|:------:|--------------------|------------|
| S3409 | `tests/test_streaming_model.py` (débit_max/streaming, contrainte SRAM), `tests/test_mahalanobis_q15.py` (récup AUROC, non-régression FP32/INT8). Unity : `test_ring_buffer.c` (push/window/stride, wrap-around) + `test_mahalanobis_q15.c` (parité, Q15) — `make test`. MAJ `docs/roadmap_phase2.md` + `CLAUDE.md`. Invoquer `graphify_sprint_update`. | 🟢 | `tests/test_streaming_model.py`, `tests/test_mahalanobis_q15.py`, `firmware/stm32f4_blink/tests/test_ring_buffer.c`, `firmware/stm32f4_blink/tests/test_mahalanobis_q15.c`, `docs/roadmap_phase2.md`, `CLAUDE.md` | 3h |

---

## Ordre d'exécution recommandé

```
Partie A                              Partie B
S3401 (streaming_model.py)            S3405 (mahalanobis_int8.py quant="q15")
        ↓                                     ↓
S3402 (ring_buffer.c/.h)              S3406 (exp PC Q15 vs INT8 vs FP32)
        ↓                                     ↓
S3403 (exp board débit/buffer)        S3407 (mahalanobis_q15.c + pipeline FLAG)
        ↓                                     ↓
S3404 (notebook streaming)            S3408 (exp board Q15 — parité)
                       ↘             ↙
                        S3409 (tests + docs + graphify)
```

---

## Nomenclature des expériences

| Exp ID | Sujet | Mesure |
|--------|-------|--------|
| exp_S34_streaming/{rateXX_strideYY}.json | balayage débit/stride board | latence DWT, remplissage buffer, saturation, .bss |
| exp_S34_maha_q15/{dataset}_{fp32,int8,q15}.json | Q15 vs INT8 vs FP32 PC | AUROC, ΔAUROC, RAM |
| exp_S34_maha_q15/summary.json | agrégat Q15 PC | récup AUROC CWRU/Pronostia |
| exp_S34_board_maha_q15/{dataset}.json | Q15 board | parité board↔PC, latence DWT, .bss |

---

## Budget mémoire firmware estimé

| Composant | RAM .bss | Notes |
|-----------|:--------:|-------|
| `ring_buffer` (W × sizeof(sample)) | dépend de W (config) | borné, 0 malloc ; S3403 mesure par W |
| Mahalanobis Q15 (`sigma_inv` int16) | ≈ 2× plus petit que FP32 | vs INT8 (÷4) mais distance préservée — compromis assumé |
| **Attendu** | **≤ budget Gap 2 (64 Ko)** | S3403/S3408 le confirment par mesure |

---

## Notes d'implémentation

**S3401 / règle CLAUDE.md** : W, S, f_acq, sizeof(sample), taille SRAM → `configs/streaming_profile.yaml`, jamais en dur. Réutiliser la latence d'inférence mesurée (Sprints 18-29, `profiling.c` DWT) comme entrée de `debit_max`.

**S3402** : extraire l'abstraction du buffer HDC existant sans changer son comportement (non-régression HDC). Pas de malloc (règle MCU), tailles en `#define` dans `inc/ring_buffer.h`.

**S3405 Q15** : la cause racine (Sprint 28) est la **grande dynamique** de `sigma_inv_` qui sature en INT8. Q15 (int16) offre une résolution 256× plus fine → distance préservée. `mu_` reste INT8 (faible dynamique). Documenter le compromis RAM (Q15 ÷2 au lieu de ÷4) vs fidélité.

**S3407 protocole UART** : tout nouveau FLAG (`MAHA_Q15_MODE`) ajouté **en parallèle** dans `pipeline.c` v3 et `sensor_stream.py` (règle CLAUDE.md). Accumulation de distance en int32 pour éviter l'overflow, conversion finale FP32 (FPU dispo sur Cortex-M4).

**S3408 parité** : suivre le protocole de parité board↔PC des Sprints 26-28 (prédictions/distances identiques sur mêmes entrées). Aucun chiffre board inventé tant que la NUCLEO n'a pas tourné.

---

## Questions ouvertes

- `TODO(arnaud)` **résolu par ce sprint** : le fallback Q15 suffit-il à récupérer l'AUROC Mahalanobis sur CWRU/Pronostia ? → quantifié par S3406.
- `TODO(fred)` : débit d'acquisition réel (f_acq) et taille de fenêtre (W) côté capteur Edge Spectrum, pour ancrer le modèle de streaming ?
- `TODO(arnaud)` : le multi-stream concurrent doit-il rester une étude analytique (S3404) ou faire l'objet d'un prototype firmware (hors périmètre actuel) ?
- `FIXME(gap2)` : confirmer que la latence board reste < 100 ms pour toutes les configs de buffer/stride testées.

---

## Livrables

1. `src/evaluation/streaming_model.py` + `configs/streaming_profile.yaml`
2. `firmware/stm32f4_blink/src/ring_buffer.c` + `inc/ring_buffer.h` (abstraction W/S, 0 malloc)
3. `experiments/exp_S34_streaming/` (balayage débit/buffer, point de saturation)
4. `notebooks/cl_eval/streaming/comparison.ipynb`
5. `src/models/unsupervised/mahalanobis_int8.py` (mode `quant="q15"`) + `configs/mahalanobis_q15_*.yaml`
6. `experiments/exp_S34_maha_q15/` (Q15 vs INT8 vs FP32 PC, récup AUROC)
7. `firmware/stm32f4_blink/src/mahalanobis_q15.c` + `inc/mahalanobis_q15.h` + intégration pipeline + `experiments/exp_S34_board_maha_q15/`
8. `tests/test_streaming_model.py`, `tests/test_mahalanobis_q15.py`, `test_ring_buffer.c`, `test_mahalanobis_q15.c` + MAJ `docs/roadmap_phase2.md` + `CLAUDE.md`

---

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S3401 streaming_model.py | ✅ | — | `src/evaluation/streaming_model.py` + `configs/streaming_profile.yaml` + 16 tests PASS |
| S3402 ring_buffer.c/.h | ✅ | — | Abstraction `(W,S)` 0-malloc ; HDC refactoré (mêmes résultats, `test_hdc.c` inchangé) ; 9 tests `test_ring_buffer.c` PASS |
| S3403 exp board débit/buffer | ✅ | — | Board NUCLEO-F439ZI : Maha 5µs / EWC 50µs invariants, **0 drop** 50–5000 Hz, **Gap 2 ✅** ; `.bss` linéaire en W (320→3200 B) ; pas de saturation (protocole synchrone auto-limité). `exp_S34_streaming/` |
| S3404 notebook streaming | ✅ | — | `notebooks/cl_eval/streaming/comparison.ipynb` exécuté (5 figures, 0 erreur) |
| S3405 mahalanobis_int8.py Q15 | ⬜ | — | — |
| S3406 exp PC Q15 vs INT8 vs FP32 | ⬜ | — | — |
| S3407 mahalanobis_q15.c + pipeline | ⬜ | — | — |
| S3408 exp board Q15 (parité) | ⬜ | — | — |
| S3409 tests + docs | ⬜ | — | — |
