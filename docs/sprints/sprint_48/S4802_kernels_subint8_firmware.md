# S4802 — Kernels sub-INT8 firmware (+ bit-packing) + Unity parité

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🔴 Critique — le kernel embarqué est le cœur du sprint (RAM `.bss` réelle + latence). |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 9h |
| **Dépendances** | S4801 (flags de build) · Sprint 39 ✅ (`ewc_head_int8_v2.c`) |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/ewc_head_int8_v2.c`, `firmware/stm32f4_blink/inc/ewc_head_int8_v2.h`, `firmware/stm32f4_blink/tests/test_ewc_subint8.c`, `firmware/stm32f4_blink/tests/test_runner.c` |
| **Références** | typedef `ewc_v2_w_t` + `EWC_V2_W_QMAX` ; émulateur `int8_c_emulation.py` (parité) |

---

## Contexte

Le kernel `ewc_head_int8_v2.c` est déjà **générique** (typedef + QMAX, variantes de build). Cette tâche ajoute
les profondeurs **4 et 2 bits** et un **chemin bit-packé** qui matérialise le gain RAM `.bss`, en gardant la
**parité bit-à-bit avec l'émulateur PC** (S47) comme contrat.

## Spec

### 1. Variantes de profondeur (non-packé)

Étendre le bloc de sélection de type dans `ewc_head_int8_v2.h` :

```c
#if defined(EWC_INT4)
typedef int8_t ewc_v2_w_t;   /* poids 4-bit dans conteneur int8 (non-packé) */
#define EWC_V2_W_QMAX 7
#elif defined(EWC_INT2)
typedef int8_t ewc_v2_w_t;
#define EWC_V2_W_QMAX 3
#endif
```

Le forward calibré existant (`round`, acc int32, déquant `acc·s_w[j]·s_a`) fonctionne tel quel — seul `QMAX`
change. **Ceci démontre le point d'honnêteté** : `.bss` non-packé ≈ INT8 (conteneur int8 identique).

### 2. Chemin bit-packé (`-DEWC_INTx_PACKED`)

- **Stockage** : `uint8_t w_packed[...]` — 2 poids/octet (INT4, nibbles) ou 4 poids/octet (INT2). Le `.bss`
  des matrices est **÷2 (INT4) / ÷4 (INT2)** vs INT8 → matérialise le ÷8/÷16 vs FP32.
- **Dépacking au forward** : extraire le poids signé (`(int8_t)((nib << 4) >> 4)` pour l'extension de signe INT4)
  juste avant le MAC FPU. Fonction `ewc_v2_unpack_weight(idx)` isolée et testée.
- **Latence** : le dépacking ajoute des opérations entières par MAC → mesuré DWT (S4804), `TODO(dorra)`.

### 3. Parité émulateur (contrat)

Le schéma de quantification (round, scales per-channel, saturation `[−QMAX,QMAX]`, déquant) doit être
**identique** au chemin `subint8` de l'émulateur S47. Test Unity `test_ewc_subint8.c` : golden vectors générés
par l'émulateur (S4803) → logits C == logits Python (aux arrondis float32 près, comme `test_ewc_int8_v2.c`).

### 4. Tests Unity

`firmware/stm32f4_blink/tests/test_ewc_subint8.c` (enregistré dans `test_runner.c`) :
- `test_int4_quant_parity` : logits INT4 non-packé == émulateur `subint8(4, per_channel)`.
- `test_int4_packed_parity` : logits INT4 packé == INT4 non-packé (le packing ne change **que** le stockage).
- `test_int2_quant_parity`, `test_int2_packed_parity`.
- `test_unpack_sign_extension` : extraction signée correcte des nibbles/2-bits.

## Contraintes

- **`.bss` défaut invariant** (build standard sans flag) — les sub-INT8 sont des variantes de build (précédent S2912).
- **0 régression** : `make test` passe sur le build par défaut et les builds sub-INT8 ; les 2 échecs TinyOL
  préexistants restent hors périmètre.
- `#define` de tailles dans les headers `inc/` (pas de tailles en dur dans le `.c` — conforme CLAUDE.md).
- Annotations `# MEM:`/`/* MEM */` mises à jour pour le stockage packé.

## Vérification

```bash
cd firmware/stm32f4_blink
make test                                   # défaut : 0 régression
make test CFLAGS_EXTRA="-DEWC_INT4"          # parité INT4 non-packé
make test CFLAGS_EXTRA="-DEWC_INT4 -DEWC_INTx_PACKED"   # parité INT4 packé
make size CFLAGS_EXTRA="-DEWC_INT4 -DEWC_INTx_PACKED"   # .bss réduit vs INT8
```

---

## Résolution (implémentée)

**Header `inc/ewc_head_int8_v2.h`** — bloc typedef étendu (`-DEWC_INT4` QMAX 7 / `-DEWC_INT2` QMAX **1** [correctif du 3 du gabarit] / `-DEWC_INT1` binaire), chacun définissant `EWC_V2_PACK_BITS` (4/2/1). Ajout des primitives de packing gardées `#if defined(EWC_V2_PACK_BITS)` : `EWC_V2_PACK_STRIDE(n)`, `ewc_v2_unpack_weight(row,i)` (LSB-first ; complément à deux + extension de signe pour 4/2 bits ; code dédié binaire `{−1,+1}↔{0,1}`) et `ewc_v2_pack_row(dst,q,n)` (miroir exact). Struct packée `EWCHeadSubInt8Packed` (poids `uint8_t` de stride bit-packé) + décl. `ewc_subint8_packed_forward`, gardées `#if defined(EWC_INTx_PACKED)`.

**Kernel `src/ewc_head_int8_v2.c`** — ajout de `ewc_subint8_packed_forward` (gardé), identique à `ewc_int8_v2_forward` mais dépacke chaque poids via `ewc_v2_unpack_weight` avant le MAC FPU (le packing ne change **que** le stockage). **`ewc_int8_v2_forward` / `ewc_int8_v2_from_fp32_calib` inchangés** → le forward non-packé (linéaire on-board, ou ternaire/binaire chargé depuis golden) réutilise le v2 tel quel.

**Choix d'architecture** : les sub-INT8 chargent des **poids pré-quantifiés PC** (le firmware ne reproduit pas TWN/BWN on-board) ; le non-packé prouve le nœud d'honnêteté (`.bss` ≈ int8, conteneur identique), le packé matérialise ÷2/÷4/÷8.

**Tests Unity `tests/test_ewc_subint8.c`** (enregistrés `test_runner.c`, ajoutés à `TEST_SRC`, cibles Makefile `test-sub-int4/-int2/-binary` ± `-packed`, `test-sub-all`) :
- `test_int4_quant_parity` (linéaire on-board QMAX 7 == golden), `test_ternary_parity`, `test_binary_parity` (poids quantifiés chargés == golden) ;
- `test_int4_packed_parity`, `test_int2_packed_parity` (packé == golden) ;
- `test_packed_storage_smaller` (sizeof packé < int8 == stride théorique — matérialise le gain, host) ;
- `test_unpack_sign_extension` (round-trip pack/unpack par profondeur).

**Résultats** (host, aucune carte) : **tous PASS** sur chaque build ; `make test` par défaut **141 tests, 2 échecs TinyOL préexistants hors périmètre, 0 régression** ; **`.bss` défaut invariant 105 036 B** ; builds ARM `-DEWC_INT2 -DEWC_INTx_PACKED` compilent sans erreur. Le `test_v2_parity_emulator` (S39) a vu sa garde étendue (TEST_IGNORE sous sub-INT8, comme Q15). `TODO(dorra)` coût latence dépacking → mesure DWT S4804.
