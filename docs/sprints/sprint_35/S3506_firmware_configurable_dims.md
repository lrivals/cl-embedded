# S3506 — Firmware : dimensions d'entrée configurables au build

| Champ | Valeur |
|-------|--------|
| **Sprint** | 35 |
| **Priorité** | 🔴 Critique — débloque le board ré-architecturé (S3507, S3508) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 6h |
| **Dépendances** | firmware Sprint 32 (dims figées) ✅, `firmware/stm32f4_blink/Makefile` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/inc/{ewc_head,mahalanobis,tinyol,hdc}.h`, `firmware/stm32f4_blink/src/pipeline.c` |
| **Références** | `inc/ewc_head.h:10` (`EWC_IN=5`), `inc/mahalanobis.h:5` (`MAHA_DIM=5`), `inc/tinyol.h:19` (`TINYOL_IN=5`), `inc/hdc.h:12` (`HDC_N_FEATURES=5`), `src/pipeline.c:36` (`PROTO_MAX_N=16`) |

---

## Contexte

Le firmware est figé à 5 features par modèle (Sprint 32). Pour les conditions `all`/`best`,
les dims d'entrée doivent devenir **configurables au build** sans hardcode, avec **0 régression**
sur la condition 5-feat existante.

## Spec

- Rendre `EWC_IN`, `MAHA_DIM`, `TINYOL_IN`, `HDC_N_FEATURES` **surchageables au build**
  (`#define` par défaut dans `inc/`, override via `-D` du Makefile / flag de condition).
  Aucune valeur en dur ailleurs que dans les headers `inc/` (règle CLAUDE.md).
- **Décision `PROTO_MAX_N`** (`TODO(dorra)`) : CMAPSS `all`=21 > 16 actuel. Deux options :
  1. Relever `PROTO_MAX_N` à ≥21 → impacte `g_stream_storage[STREAM_BUF_W*PROTO_MAX_N]`
     (`pipeline.c:73`) et `payload[]` (`pipeline.c:347`) → coût `.bss`/stack à mesurer.
  2. Restreindre la condition `all` board au sous-ensemble ≤16 (CMAPSS top-16) et documenter.
  → Trancher et **documenter le choix + le coût RAM** dans ce fichier au moment de l'implémentation.
- **Non-régression** : la condition `5feat` doit produire des résultats **identiques** au firmware
  actuel (parité bit-à-bit EWC/Maha avec les exports existants).

## Vérification

```bash
cd firmware/stm32f4_blink
make EWC_IN=9 MAHA_DIM=9 size      # build condition all-9-feat compile, RAM affichée
make size                          # build par défaut (5feat) inchangé
make test                          # Unity vert, 0 régression (test_pipeline, test_ewc_head...)
```

**Critère** : compile pour chaque dim cible ; `.bss` 5-feat inchangé ; décision `PROTO_MAX_N` écrite.

---

## Décision `PROTO_MAX_N` (`TODO(dorra)` → tranché : **Option 1**)

`PROTO_MAX_N` devient **configurable au build** (`#ifndef`, défaut **16**), pas restreint ni
forcé globalement à 21. La condition `all`×cmapss (21 capteurs) se construit explicitement avec
`-DPROTO_MAX_N=21` ; toutes les autres conditions (`5feat`/`best` ≤ 16) gardent 16.

**Coût RAM mesuré** (`make PROTO_MAX_N=21 size`) : `.bss` passe de **104 956 B → 105 056 B**,
soit **+100 B** = `STREAM_BUF_W·(21−16)·4 = 5·5·4` (croissance de `g_stream_storage`). `raw[]` et
`payload[]` sont sur la **pile** (pas de coût `.bss`). À défaut (16), la ligne de commande
compilateur est inchangée ⇒ **`.bss` 5-feat bit-identique**. Justification : restreindre `all`
(Option 2) perdrait 5 capteurs CMAPSS pour économiser 100 B — non rentable vs budget 256 Ko.

## Implémentation (✅)

- **Headers** `#ifndef` : `EWC_IN` (`ewc_head.h`), `MAHA_DIM` (`mahalanobis.h`),
  `TINYOL_IN` (`tinyol.h`, + `TINYOL_OUT` lié à `TINYOL_IN`), `HDC_N_FEATURES` (`hdc.h`).
- **`pipeline.c`** : `PROTO_MAX_N` en `#ifndef` ; nouveau `WEIGHTS_NATIVE_DIM` (=5, aussi dans
  `tinyol.h` pour `tinyol.c`/`tinyol_int8.c`). Les copies de poids placeholder de
  `model_weights.h` (header généré, **jamais édité à la main**) sont gardées par
  `#if (DIM == WEIGHTS_NATIVE_DIM)` : Maha (loop), TinyOL (memcpy), EWC
  (`#if defined(EWC_HEAD_WEIGHTS_PROVIDED) && (EWC_IN == WEIGHTS_NATIVE_DIM)`). Hors dim native →
  init neutre (`maha_init`=identité, TinyOL=0, EWC=Xavier) ; **poids réels regénérés en S3507**.
- **`Makefile`** : nouvelle cible `size` ; les vars `EWC_IN`/`MAHA_DIM`/`TINYOL_IN`/
  `HDC_N_FEATURES`/`PROTO_MAX_N` sont traduites en `-D…` via `ifdef` (rien passé par défaut ⇒
  ligne compilateur identique).

## Résultats build (`arm-none-eabi-size`)

| Build | `.bss` (B) | Δ vs défaut | Statut |
|-------|-----------:|------------:|--------|
| défaut (5feat) | 104 956 | — | ✅ inchangé, 0 warning overread |
| `EWC_IN=9 MAHA_DIM=9` | 107 116 | +2 160 | ✅ compile, init neutre |
| `PROTO_MAX_N=21` | 105 056 | +100 | ✅ compile |
| `TINYOL_IN=9 HDC_N_FEATURES=9` | 122 324 | +17 368 | ✅ compile |
| `…=21` (toutes dims) | 183 936 | +78 980 | ✅ compile, < 192 Ko RAM |

`make test` (dims défaut) : **105 tests, 103 PASS, 2 FAIL TinyOL pré-existants** (hors périmètre,
`test_tinyol_predict_normal_zero_weights` / `test_tinyol_forward_delta`) ⇒ **0 régression**.
