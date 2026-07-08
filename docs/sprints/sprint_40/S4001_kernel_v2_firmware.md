# S4001 — Kernel C INT8 v2 (calibré) + export + tests host

| Champ | Valeur |
|-------|--------|
| **Sprint** | 40 |
| **Priorité** | 🔴 Critique — prérequis données (récupération F1 INT8) |
| **Statut** | ✅ Implémenté — kernel v2 + export + 5 tests host PASS ; branche update online v2 ajoutée à `pipeline.c` (S4002) ; 0 régression v1 |
| **Durée estimée** | ~6h (PC + `make test` host, aucun board) |
| **Dépendances** | S3901 ✅ (audit) · S3902 ✅ (émulateur) · S3904 ✅ (ablation) · `src/utils/int8_c_emulation.py` |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c`, `inc/ewc_head_int8_v2.h`, `scripts/export_weights_c.py`, `firmware/stm32f4_blink/tests/test_ewc_int8_v2.c` |
| **Références** | S3907/S3908/S3909 · `firmware/.../src/ewc_head_int8.c` (v1, à laisser intact pour A/B) |

## Contexte

L'ablation Sprint 39 (`exp_S39_ablation/`) a établi que la cause racine de l'effondrement F1 INT8 board est
le **scale figé `1/128`** du kernel v1 (`ewc_head_int8.c`), **pas** l'overflow int16 (`fix_acc32` marginal).
Ce ticket implémente le **kernel v2 calibré** afin que la récupération démontrée à l'émulateur soit
**exécutable sur carte réelle** (S4002) et donc citable dans l'article comme résultat matériel.

> **Artefact à réconcilier** : `firmware/stm32f4_blink/inc/ewc_head_int8_v2.h` existe déjà (untracked) —
> vérifier sa cohérence avec la spec ci-dessous avant d'écrire le `.c` ; ne pas dupliquer.

## Spec

### Kernel v2 (`ewc_head_int8_v2.c/.h`)
- **Accumulateur int32** (corrige F1 : plus d'overflow int16).
- **Scales calibrés par-canal** importés depuis le header généré (corrige F2/F3 : ≠ `1/128` figé).
- **Option Q15 par compilation** `-DEWC_INT8_Q15` (poids/activations int16, repli fidélité 256×).
- **v1 strictement inchangé** : le v2 est un fichier séparé sélectionné par build → permet l'A/B board
  (S4002) et garantit **0 régression** sur le chemin FP32 et INT8-legacy existant.
- Déquant→distance en FP32 sur FPU (parité bit-à-bit avec l'émulateur Python) — le paradoxe latence est
  assumé et documenté (pas d'objectif d'accélération ici ; SIMD = S3910/S3917 différé).

### Export (`export_weights_c.py --int8-v2`)
- Étend l'exporteur pour émettre `inc/ewc_head_int8_v2_weights.h` avec les **scales par-canal calibrés**
  (garde `EWC_INT8_V2_WEIGHTS_PROVIDED` → vide par défaut = fallback, **0 régression**).
- Réutilise la logique de calibration de `int8_c_emulation.py` (`per_channel_int8` / `q15`) comme **source
  unique** : le header C et l'émulateur PC consomment les mêmes scales ⇒ parité par construction.

### Tests Unity host (`test_ewc_int8_v2.c`, `make test` x86, sans board)
| Test | Vérifie |
|------|---------|
| `test_v2_no_overflow` | accumulateur int32, pas de wrap sur activations larges |
| `test_v2_parity_emulator` | logits C v2 == `int8_c_emulation.forward_quant(per_channel)` |
| `test_v2_q15_parity` | build `-DEWC_INT8_Q15` == émulateur `q15` |
| `test_v2_recovers_f1` | F1 v2 ≫ F1 v1 (legacy) sur vecteurs de test étiquetés |
| `test_v1_unchanged` | kernel v1 inchangé (A/B) → **0 régression** |

## Vérification

```bash
python scripts/export_weights_c.py --model ewc --config configs/board_ewc.yaml --int8-v2
cd firmware/stm32f4_blink && make test           # Unity host : v2 + v1 inchangé, 0 régression
pytest tests/test_int8_c_emulation.py -q          # parité émulateur (référence PC)
```

> Les 2 tests TinyOL préexistants restent hors périmètre (échecs connus). Critère : v2 ajoute des tests
> PASS, v1 inchangé, aucune nouvelle régression.
