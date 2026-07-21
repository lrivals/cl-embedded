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

_À compléter lors de l'implémentation._
