# S2912 — Portage Mahalanobis INT8 firmware (NUCLEO-F439ZI)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 29 (extension O8 — board 4×5 complet) |
| **Priorité** | 🔴 (bloque la 4ᵉ ligne board Mahalanobis) |
| **Statut** | ✅ Implémenté (28 juin 2026) — `mahalanobis_int8.c/.h` + export + test Unity (4/4 PASS), parité C↔Python validée |
| **Durée estimée** | 4h |
| **Dépendances** | `src/models/unsupervised/mahalanobis_int8.py` ✅ (quantifieurs PC) · pattern Q15 S3407 ✅ · NUCLEO-F439ZI connectée |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/mahalanobis_int8.c` + `inc/mahalanobis_int8.h` + `inc/mahalanobis_int8_weights.h` (généré) · `scripts/export_weights_c.py` (`--maha-int8`) · `firmware/stm32f4_blink/tests/test_mahalanobis_int8.c` |

---

## Contexte

Le firmware porte aujourd'hui Mahalanobis en **FP32** (`mahalanobis.c`) et **Q15**
(`mahalanobis_q15.c`, Sprint 34), mais **pas en INT8**. Or la grille PC (Sprint 28) compare
Mahalanobis FP32 vs INT8 sur les 5 datasets ; pour obtenir une grille board 4×5 = 20
symétrique au PC (demande utilisateur), il faut une variante **Mahalanobis INT8** mesurable
sur la carte.

Le résultat attendu est **honnête et déjà connu** : l'INT8 dégrade Mahalanobis
(`sigma_inv_` à grande dynamique, cf. Sprint 28 cwru −0.236 / pronostia −0.238 ; Sprint 34
a établi Q15 comme fallback recommandé). Ce portage ne vise pas à « réparer » l'INT8 mais à
**mesurer fidèlement** son comportement sur board, en parité avec le PC.

---

## Contrainte de conception : espace de flags protocole saturé

Le nibble haut du flag protocole (`pipeline.h:24-74`) est **entièrement assigné**
(0x00 = Maha FP32 défaut … 0xF0 = MAHA_Q15). **Aucun flag runtime n'est libre** pour
MAHA_INT8.

→ **Sélection par compilation** : `-DMAHA_INT8`. Dans le chemin Mahalanobis par défaut de
`pipeline.c`, `#ifdef MAHA_INT8` appelle `mahalanobis_int8_score()` au lieu de
`mahalanobis_score()`. Cohérent avec le driver d'extension (S2913) qui re-flashe par cellule
(comme TinyOL). **0 régression** : sans `-DMAHA_INT8`, le chemin FP32 est strictement inchangé.

---

## Spécification

### 1. `inc/mahalanobis_int8.h` + `src/mahalanobis_int8.c`

Miroir exact de `MahalanobisDetectorInt8` (mode `int8`) de `mahalanobis_int8.py` —
**parité board↔PC par construction**, comme le chemin Q15 (S3407) :
- `mu_` quantifié **int8 affine** (`mu_q`, `mu_scale`, `mu_zp`).
- `sigma_inv_` quantifié **int8** (scale par-tenseur).
- Forward : **déquantification → distance de Mahalanobis en FP32 sur la FPU** (pas
  d'accumulation int32), pour reproduire bit-à-bit le chemin Python `anomaly_score`/`predict`.
- API : `mahalanobis_int8_score(const float *x, int d) -> float`,
  `mahalanobis_int8_predict(...) -> int` (seuil), `mahalanobis_int8_load_weights()`.
- Tailles via `#define` (pas de magie) ; dimension native par `MAHA_INT8_NATIVE_DIM`.

### 2. `inc/mahalanobis_int8_weights.h` (généré)

Vide par défaut (garde `MAHA_INT8_WEIGHTS_PROVIDED` non défini → fallback sans régression),
régénéré par l'export. **Ne jamais éditer à la main** (règle CLAUDE.md).

### 3. `scripts/export_weights_c.py --maha-int8`

Sur le modèle de `export_maha_q15_to_c` (ligne 473) : réutilise **exactement** les
quantifieurs de `mahalanobis_int8.py` → écrit `inc/mahalanobis_int8_weights.h`
(`mu_q` uint8/int8 + scales + `sigma_inv` int8 + scale + seuil). Ajouter aussi
`--maha-int8-test-vectors` → `tests/test_vectors_maha_int8.h` (entrée + distance attendue
calculée par le chemin Python, pour le test C).

### 4. `tests/test_mahalanobis_int8.c` (Unity)

Parité **C↔Python** : charge les vecteurs de test exportés, vérifie que
`mahalanobis_int8_score()` reproduit la distance Python à la tolérance flottante près.
Brancher dans `test_runner.c` + `Makefile` (host `TEST_MODE=1`).

---

## Critères d'acceptation

- `make test` → nouveau `test_mahalanobis_int8` PASS, **0 régression** sur les tests existants.
- Build `-DMAHA_INT8` compile sans warning ; build par défaut inchangé (FP32).
- Parité board↔PC vérifiée à l'exécution (S2913) : prédictions identiques, `max_score_err`
  au niveau du bruit flottant (comme Q15 S3408).

---

## Notes

- Pas de SIMD : la distance reste FP32 sur FPU → la latence INT8 ne sera pas plus rapide
  (cohérent avec le résultat négatif latence des autres modèles).
- La RAM « poids » analytique Mahalanobis (mu + sigma_inv) FP32 vs INT8 sera renseignée dans
  `RAM_WEIGHTS["mahalanobis"]` de l'orchestrateur (S2913).
