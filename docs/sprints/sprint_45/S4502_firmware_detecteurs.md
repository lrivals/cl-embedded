# S4502 — Firmware : port C des détecteurs de drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 45 |
| **Priorité** | 🔴 Critique — cœur du sprint : rendre les détecteurs exécutables sur Cortex-M4, à parité avec le Python. |
| **Statut** | ✅ Implémenté — 3 détecteurs portés (parité C↔Python), intégrés sous `-DDRIFT_DETECT`, `.bss` défaut invariant. |
| **Durée estimée** | 10h |
| **Dépendances** | S4501 ✅ (liste + cadrage) · `firmware/.../inc/ring_buffer.h` ✅ (fenêtre 0 malloc) · `firmware/.../src/drift_detector.c` ✅ (précédent de port bit-à-bit, S3803) · `firmware/.../src/profiling.c` ✅ · S4402/S4403 ✅ (réf. Python) |
| **Fichiers cibles** | `firmware/stm32f4_blink/inc/drift/{page_hinkley,ddm,psi}.h`, `src/drift/{page_hinkley,ddm,psi}.c`, `src/pipeline.c`, `Makefile`, `tests/test_drift_methods.c`, `tests/test_runner.c` |
| **Références** | S3803 (`drift_detector.c`, gabarit de port + footgun non-copiable) · CLAUDE.md § « pas de hardcode, `#define` dans headers » |

---

## Contexte

Porter en C les détecteurs retenus (S4501), en suivant le gabarit de `drift_detector.c` (S3803) :
**état statique, 0 malloc, parité bit-à-bit** avec le Python. Les détecteurs à fenêtre réutilisent
`ring_buffer.h`. Intégration dans `pipeline.c` sous `-DDRIFT_DETECT` (build par défaut inchangé).

## Spec

### 1. Interface commune C — `inc/drift/drift_method.h`

Miroir de `BaseDriftDetector` (S4401) :
```
typedef enum { DM_NORMAL = 0, DM_WARNING = 1, DM_DRIFT = 2 } DriftMethodVerdict;
```
Chaque détecteur expose `*_init(...)`, `*_update(d, value) -> DriftMethodVerdict`, `*_reset(d)`.
État en **struct à backing statique** (comme `DriftDetector`).

### 2. Détecteurs portés

- **Page-Hinkley** (`page_hinkley.c/.h`) : `float cum_sum, min_sum, mean; uint32_t n;` — état O(1),
  `# MEM: ~20 B @ FP32`. `update` : maj moyenne courante, `cum_sum += x − mean − delta`, `DM_DRIFT` si
  `cum_sum − min_sum > lambda`. Parité bit-à-bit (mêmes opérations/ordre float que le Python).
- **DDM** (`ddm.c/.h`) : `float p, s, p_min, s_min; uint32_t n;` — O(1). Seuils 2σ (WARNING) / 3σ (DRIFT).
- **PSI** (`psi.c/.h`) : histogramme à **bacs fixes** (bornes calibrées à l'enrôlement, fournies par
  header généré S4503) ; `uint16_t counts[PSI_BINS]` (référence) + comptage courant ; `# MEM: 2·PSI_BINS
  ·2 B`. `PSI_BINS` surchargeable. Non-supervisé → branché sur `maha_score` (S4501).
- (Conditionnel) **ADWIN** (`adwin.c/.h`) si S4501 le retient, avec **borne de buckets** statique.

### 3. Intégration `pipeline.c` (sous `-DDRIFT_DETECT`)

Global statique `g_drift_method` (type selon `-DDRIFT_METHOD`), initialisé depuis
`inc/drift_methods_params.h` (S4503). Dans le chemin d'inférence : calculer le signal (score Maha pour
non-supervisé, `1[pred≠label]` pour supervisé), appeler `*_update`, **réinterpréter un champ du snapshot**
pour remonter le verdict (précédent S3805) — **wire format V3 inchangé**. Build par défaut (sans
`-DDRIFT_DETECT`) : chemin historique intact.

### 4. Makefile

Ajouter `src/drift/*.c` aux sources firmware **et** à `TEST_SRC`. `-DDRIFT_DETECT` / `-DDRIFT_METHOD=…` /
`-DPSI_BINS=…` passés via `EXTRA_CFLAGS` (pas de clobber). Compilation par méthode.

### 5. Test Unity — `test_drift_methods.c`

Parité C↔Python sur des **séquences de scores connues** (réutiliser les vecteurs S44) : verdicts
identiques (NORMAL/WARNING/DRIFT), Page-Hinkley déclenche au bon indice, DDM franchit 2σ/3σ, PSI dépasse
le seuil au bon bac. `*_reset` remet l'état à zéro. Déclarer dans `test_runner.c`. **Footgun S3803** :
structs à fenêtre non copiables (le `RingBuffer.storage` pointe vers le backing interne) → init en place.

## Contraintes

- **0 malloc**, tout en backing statique ; tailles en `#define` surchargeables.
- **Parité bit-à-bit** : mêmes opérations flottantes, même ordre que le Python (pas d'accumulation int
  cachée) — comme `mahalanobis_q15.c` (déquant→FP32 FPU).
- Build par défaut **invariant** (`.bss` inchangé, 0 régression) — vérifié `make size` + `make test`.

## Vérification

```bash
cd firmware/stm32f4_blink
make test                                                        # test_drift_methods PASS + 0 régression
make                                                             # build défaut compile (détecteurs liés, inactifs)
make EXTRA_CFLAGS="-DDRIFT_DETECT -DDRIFT_METHOD=page_hinkley"    # build actif compile
make size                                                        # .bss défaut invariant ; +delta sous -DDRIFT_DETECT
```
- `test_drift_methods` : verdicts == sortie Python des détecteurs S4402/S4403 sur la même séquence.
- `.bss` build par défaut identique à l'actuel (105 036 B) ; delta documenté par méthode.

---

## Résolution (implémentée)

**Fichiers créés** : `inc/drift/{drift_method,page_hinkley,ddm,psi}.h` + `src/drift/{page_hinkley,
ddm,psi}.c` + `tests/test_drift_methods.c`. Chaque détecteur = struct à backing statique, 0 malloc,
init en place (footgun S3803), `# MEM:` annotés, tailles en `#define` surchargeables.

**Interface commune** `drift_method.h` : `DriftMethodVerdict {DM_NORMAL,DM_WARNING,DM_DRIFT}` +
IDs `DRIFT_PAGE_HINKLEY/DDM/PSI` (défaut neutre). Chaque détecteur : `*_init / *_update(value) ->
verdict / *_reset`.

**Parité bit-à-bit** vérifiée sur séquences dont les verdicts attendus sont produits par le Python
lui-même (`src/models/drift/*.py`) : Page-Hinkley DRIFT au bon indice sur saut de moyenne ; DDM
franchit 2σ (WARNING) puis 3σ (DRIFT) ; PSI DRIFT à la fin du bloc effondré (statistique 14.99 ± tol
FP32). `*_reset` remet l'état à zéro. **`make test` : 6/6 nouveaux PASS** (134 tests, 2 TinyOL
préexistants hors périmètre, **0 régression**).

**Intégration `pipeline.c`** sous `#ifdef DRIFT_DETECT` : global `g_drift_method` (type selon
`#if DRIFT_METHOD`), init depuis `inc/drift_methods_params.h` (généré S4503, neutre par défaut) ;
dans le chemin EWC — signal = `maha_score` (PSI) ou `1[pred != g_recv_label]` (PH/DDM) → `*_update`
→ `g_drift_method_verdict` remonté via `snap.auroc` (+ `snap.forgetting` = nb DRIFT cumulés). **Wire
format V3 (23 B) inchangé**, `sensor_stream.py` intact.

**Makefile** : `src/drift/*.c` ajoutés à `C_SOURCES` **et** `TEST_SRC` ; règle de motif
`$(BUILD_DIR)/%.o: src/drift/%.c` + `-Iinc/drift` ; `PSI_BINS`/`PSI_BLOCK_SIZE`/`DRIFT_METHOD` via
`EXTRA_CFLAGS` (pas de clobber).

**Vérification build (NUCLEO-F439ZI, arm-none-eabi-gcc)** :

| Build | `.bss` | Δ vs défaut |
|-------|-------:|------------:|
| défaut (sans `-DDRIFT_DETECT`) | **105 036 B** | 0 (invariant, 0 régression) |
| `-DDRIFT_METHOD=DRIFT_PAGE_HINKLEY` | 105 072 B | **+36 B** |
| `-DDRIFT_METHOD=DRIFT_DDM` | 105 076 B | **+40 B** |
| `-DDRIFT_METHOD=DRIFT_PSI` | 105 168 B | **+132 B** (histogramme (3·PSI_BINS+1)·4) |

Les 3 builds actifs compilent sans warning issu des fichiers `drift/` (warnings restants
pré-existants : `model_weights_*.h`, `uart_send_response_v2`, `pipeline.c:435`).
