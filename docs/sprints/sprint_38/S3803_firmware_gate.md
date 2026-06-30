# S3803 — Firmware : gate de mise à jour autonome (sélection à la compilation)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🔴 Critique — c'est le cœur technique : remplacer le déclencheur humain par un gate embarqué. |
| **Statut** | ✅ Implémenté — gate `pipeline.c` sous `-DEWC_AUTO_UPDATE`/`-DGATE_PSEUDO_LABEL`, `drift_thresholds.h` généré, `export_weights_c.py --drift-thresholds`, `test_drift_detector.c` **6/6 PASS** (0 régression). |
| **Durée estimée** | 7h |
| **Dépendances** | `firmware/.../inc/ring_buffer.h` ✅ (S3402) · `firmware/.../src/mahalanobis.c` ✅ (`maha_score`, `maha_update`) · `firmware/.../src/ewc_head.c` ✅ (`ewc_sgd_step`) · `src/evaluation/drift_detector.py` ✅ (référence de parité) · `scripts/export_weights_c.py` ✅ |
| **Fichiers cibles** | `firmware/.../inc/drift_detector.h`, `src/drift_detector.c`, `inc/drift_thresholds.h` (généré), `src/pipeline.c` (gate), `Makefile`, `scripts/export_weights_c.py` (`--drift-thresholds`), `firmware/.../tests/test_drift_detector.c` |
| **Références** | Sprint 29 S2912 (`-DMAHA_INT8` — précédent de sélection à la compilation, nibble UART saturé) |

---

## Contexte

Le nibble de flags du protocole UART est **saturé** (0x10..0xF0 attribués) → on **ne touche pas au
protocole**. On suit le précédent `-DMAHA_INT8` (Sprint 29) : la mise à jour autonome est activée par un
**flag de compilation** `-DEWC_AUTO_UPDATE`. Le build par défaut reste strictement inchangé (0 régression).

## Spec

### 1. Port C du détecteur — `drift_detector.c/.h`
Port de `SlidingWindowDriftDetector` : `drift_init(window_size, fault_threshold, drift_threshold, drift_ratio)`,
`drift_update(score) → {DRIFT_NORMAL, DRIFT_FAULT, DRIFT_DRIFT}`, `drift_reset()`. Fenêtre glissante via
**`RingBuffer`** (S3402, 0 malloc), backing statique dans la struct. **Parité bit-à-bit** avec le Python :
même ordre (push → FAULT instantané → DRIFT sur fraction de fenêtre) et même dénominateur (count courant).
État ~200 B @ FP32 pour W=50.

### 2. Seuils générés — `inc/drift_thresholds.h`
`export_weights_c.py --drift-thresholds` lit `drift_thresholds.json` (S3802) et émet
`DRIFT_FAULT_THRESHOLD`, `DRIFT_DRIFT_THRESHOLD`, `DRIFT_WINDOW_SIZE`, `DRIFT_RATIO` + garde
`DRIFT_THRESHOLDS_PROVIDED`. **Jamais édité à la main** (règle CLAUDE.md). Header vide par défaut →
fallback neutre (seuils n'activant rien) → 0 régression.

### 3. Gate dans `pipeline.c` (chemin EWC 0x10)
Sous `-DEWC_AUTO_UPDATE`, le chemin EWC ignore `PROTO_FLAG_UPDATE` et décide à bord :
```
s = maha_score(&g_detector, raw);
v = drift_update(&g_drift, s);
#ifdef GATE_PSEUDO_LABEL                 /* P3 : 100 % autonome */
    if (v == DRIFT_FAULT)  ewc_sgd_step(&g_ewc_head, raw, 1);   /* pseudo-label faulty */
    else if (v == DRIFT_DRIFT) maha_update(&g_detector, raw);   /* adapte le normal */
#else                                    /* P2 : vrai label sur flag (active learning) */
    if (v != DRIFT_NORMAL) ewc_sgd_step(&g_ewc_head, raw, (int)g_recv_label);
#endif
```
Le compteur `n_updates` (SGD réellement effectués) est renvoyé dans la réponse pour mesure board.
Build par défaut (sans `-DEWC_AUTO_UPDATE`) : chemin historique `PROTO_FLAG_UPDATE` inchangé.

### 4. Makefile
Ajouter `src/drift_detector.c` aux sources firmware + à `TEST_SRC`. `-DEWC_AUTO_UPDATE` /
`-DGATE_PSEUDO_LABEL` passés via `EXTRA_CFLAGS` (pas de clobber). `DRIFT_WINDOW_MAX` surchargeable.

### 5. Test Unity — `test_drift_detector.c`
Parité C↔Python sur une **séquence de scores** connue : verdicts identiques (NORMAL/DRIFT/FAULT),
priorité FAULT>DRIFT, déclenchement DRIFT au franchissement de `drift_ratio`, `drift_reset` vide la fenêtre.
Déclarer les tests dans `test_runner.c`.

## Vérification

```bash
cd firmware/stm32f4_blink
make test          # test_drift_detector PASS + 0 régression (build défaut inchangé)
make               # build défaut compile (drift_detector.c lié, gate inactif)
make EXTRA_CFLAGS="-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL" EWC_IN=5 MAHA_DIM=5   # build P3 compile
make size          # .bss défaut invariant ; +~256 B sous -DEWC_AUTO_UPDATE (g_drift)
```
- Parité `test_drift_detector` : verdicts == sortie `SlidingWindowDriftDetector.update_batch` sur la même séquence.

## Résultats d'implémentation

- **`make test` : 122 tests, 2 échecs TinyOL préexistants (hors périmètre), 0 régression** ; les
  **6 tests drift PASS** (NORMAL/FAULT/DRIFT, priorité FAULT>DRIFT, seuil `drift_ratio`, `drift_reset`,
  séquence-parité vérifiée bit-à-bit contre le Python `[1,6,6,11,6,2] → N,N,D,F,D,D`).
- **`.bss` build par défaut = 105 036 B (invariant, 0 régression)** ; build gate
  `-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL` = 105 332 B (**+296 B** : `g_drift` ~256 B + `g_n_updates`
  + alignement). Les deux builds compilent sans warning lié au gate.
- **Footgun documenté** : `DriftDetector` n'est **pas copiable** (`window.storage` pointe vers son
  propre `storage[]` interne). On l'initialise toujours **en place** via pointeur (comme le global
  statique `g_drift`) — jamais retourné/copié par valeur. Le test l'initialise via `make_det(&d)`.
- `inc/drift_thresholds.h` par défaut = **seuils neutres** (≈FLT_MAX → 0 déclenchement) ;
  `export_weights_c.py --drift-thresholds <json>` pose `DRIFT_THRESHOLDS_PROVIDED` + les valeurs
  calibrées (vérifié : fault=7.6486 / drift=3.9773 / window=50 / ratio=0.6 → build gate OK).
