# S4803 — Export `--ewc-subint8` + test vectors golden

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🔴 Critique — chaîne d'export = garantie de parité par construction. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Durée estimée** | 5h |
| **Dépendances** | S4802 (kernel) · S4702 (émulateur étendu S47) |
| **Fichiers cibles** | `scripts/export_weights_c.py`, `firmware/stm32f4_blink/inc/ewc_head_subint8_weights.h` (généré), `firmware/stm32f4_blink/tests/test_vectors_subint8.h` (généré) |
| **Références** | patron `--int8-v2`/`--int8-v2-test-vectors` (S3908/S3909), primitives émulateur `_weight_scales`/`_quant_weight` |

---

## Contexte

Comme `--int8-v2` réutilise **exactement** les primitives de l'émulateur (parité par construction), l'export
sub-INT8 doit faire de même : quantifier avec le **même** `subint8(bits, granularity, symmetry)` du Sprint 47,
puis émettre le header C (packé ou non) et les golden vectors pour le test Unity S4802.

## Spec

### 1. Option `--ewc-subint8`

```
python scripts/export_weights_c.py --ewc-subint8 <checkpoint.pt> \
    --weight-bits 4 --granularity per_channel [--symmetry symmetric] [--packed] [--out inc/]
```

Étapes (miroir de `_export_int8_v2`) :
1. Charger la tête EWC FP32, extraire `EWCHeadWeights.from_state_dict`.
2. `scales = _weight_scales(W, granularity, weight_bits)` ; `q = _quant_weight(W, scales, weight_bits)` — **mêmes
   primitives que l'émulateur** → parité.
3. Si `--packed` : empaqueter `q` (2 poids/octet INT4, 4/octet INT2) dans `uint8_t[]`.
4. Émettre `inc/ewc_head_subint8_weights.h` : matrices (packées ou non), `scale_w*[]`, `scale_act_*`, biais FP32,
   garde `EWC_SUBINT8_WEIGHTS_PROVIDED`, en-tête « GÉNÉRÉ — ne pas modifier à la main ».

### 2. Option `--ewc-subint8-test-vectors`

Génère `tests/test_vectors_subint8.h` : entrées `TV_SUBINT8_INPUT` + logits golden (`forward_quant` avec le
`subint8` correspondant), pour `test_ewc_subint8.c` (S4802).

### 3. Header généré (jamais édité à la main)

Respecte l'interdiction CLAUDE.md (« ne pas modifier `model_weights.h` à la main »). La garde
`EWC_SUBINT8_WEIGHTS_PROVIDED` (vide par défaut → fallback, 0 régression) suit le patron
`EWC_HEAD_INT8_V2_*`/`MAHA_Q15_WEIGHTS_PROVIDED`.

## Contraintes

- **Réutiliser les primitives de l'émulateur** (S47) — ne pas ré-implémenter la quantification (parité par construction).
- Header **généré** uniquement via le script ; défaut vide → 0 régression.
- `--packed` produit un stockage `uint8_t` cohérent avec le dépacking C (S4802).

## Vérification

```bash
python scripts/export_weights_c.py --ewc-subint8 <ckpt> --weight-bits 4 --granularity per_channel --packed
test -f firmware/stm32f4_blink/inc/ewc_head_subint8_weights.h
python scripts/export_weights_c.py --ewc-subint8 <ckpt> --weight-bits 4 --granularity per_channel --ewc-subint8-test-vectors
cd firmware/stm32f4_blink && make test CFLAGS_EXTRA="-DEWC_INT4 -DEWC_SUBINT8_WEIGHTS_PROVIDED"   # golden parité
```

---

## Résolution (implémentée)

**`scripts/export_weights_c.py`** — options `--ewc-subint8` (+ `--weight-bits {4,2}`, `--weight-mode {linear,ternary,binary}`, `--granularity`, `--symmetry`, `--packed`) et `--ewc-subint8-test-vectors`.

- `_ewc_subint8_quantize` : `EWCHeadWeights.from_state_dict` → **`_quant_weight_mode`** (émulateur S47, dispatch linéaire/ternaire/binaire) → poids quantifiés + scales par-canal ; `calibrate_activations` → scales d'activation. **Parité par construction** (mêmes primitives).
- `_pack_weights(q, pack_bits)` : empaquette les entiers signés en `uint8_t[]` (LSB-first, complément à deux 4/2 bits, binaire `q>0→1`), **miroir exact** de `ewc_v2_pack_row`/`ewc_v2_unpack_weight` (firmware).
- `export_ewc_subint8_to_c` → `inc/ewc_head_subint8_weights.h` : poids packés (`uint8_t`) ou conteneurs `int8_t`, `EWC_SUB_SCALE_W*`, biais FP32, scales activation, gardes `EWC_SUBINT8_WEIGHTS_PROVIDED`/`EWC_SUBINT8_PACK_BITS`/`EWC_SUBINT8_PACKED`, bannière « GÉNÉRÉ — ne pas éditer ». Vide par défaut → fallback → 0 régression.
- `export_ewc_subint8_test_vectors_h` → `tests/test_vectors_subint8.h` : golden **auto-suffisant tous schémas** (poids FP32 + act_max + entrées + poids quantifiés/scales INT4/ternaire/binaire + logits `forward_quant(subint8(...))`).

**Header généré, jamais édité à la main** (règle CLAUDE.md), patron `--int8-v2` (S3908/S3909).

**Tests `tests/test_ewc_subint8_export.py`** (PC-only) : **7 PASS** — round-trip `_pack_weights`↔dépack C (4/2/1 bits), parité export ↔ `_quant_weight_mode` (linéaire/ternaire/binaire), garde du header généré. Aucun chiffre golden en dur (tout dérive de l'émulateur).

**Vérifié bout-en-bout** avec `experiments/exp_S39_matched/checkpoints/ewc_pronostia_5feat.pt` (k=5) : header packé binaire → `uint8_t EWC_SUB_W1[32][1]` (8 poids/octet), et les golden alimentent `test_ewc_subint8.c` (S4802, tous PASS).
