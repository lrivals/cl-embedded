# S3908 — Export des scales calibrés par-canal → header C

| Champ | Valeur |
|-------|--------|
| **Sprint** | 39 |
| **Priorité** | 🔴 Critique — alimente le kernel v2 (jamais éditer le header à la main) |
| **Statut** | ✅ Implémenté (1er juillet 2026) — `--int8-v2` + `--int8-v2-test-vectors`, header généré vérifié |
| **Durée estimée** | 2h |
| **Dépendances** | S3907 (struct `EWCHeadInt8V2`) · `scripts/export_weights_c.py` (à étendre) |
| **Fichier cible** | `scripts/export_weights_c.py` (`--int8-v2`) → `firmware/.../inc/ewc_head_int8_v2_weights.h` |
| **Références** | `export_weights_c.py --ewc-head` (patron) · `src/utils/int8_c_emulation.py` (`_weight_scales`, `calibrate_activations`) |

---

## Contexte

Le kernel v2 (S3907) a besoin des **scales par-canal** (un par neurone de sortie) et des **scales
d'activation calibrés** — exactement ceux calculés par l'émulateur (`_weight_scales` mode `per_channel`,
`calibrate_activations`). Pour garantir la **parité board↔PC par construction**, ces scales sont calculés
côté PC et **exportés** dans un header C généré, jamais saisi à la main (règle CLAUDE.md).

## Extension de `export_weights_c.py`

Nouveau flag `--int8-v2` :

1. Charge le checkpoint EWC board (`ewc_head.pt`, EWCMlpMulticlass).
2. Calcule via `int8_c_emulation` :
   - `scale_w1/2/3[j] = max|W[j,:]| / 127` (per-canal),
   - `scale_act_in/h1/h2` via `calibrate_activations` sur un lot représentatif.
3. Quantifie les poids `round(W / scale[:,None])` clip [-127,127].
4. Émet `ewc_head_int8_v2_weights.h` avec les tableaux `w*` (int8) + `scale_w*` (float) + `scale_act_*`.

```c
/* GÉNÉRÉ par export_weights_c.py --int8-v2 — NE PAS ÉDITER */
#define EWC_INT8_V2_WEIGHTS_PROVIDED 1
static const int8_t  EWC_V2_W1[EWC_H1][EWC_IN] = { ... };
static const float   EWC_V2_SCALE_W1[EWC_H1]   = { ... };
static const float   EWC_V2_SCALE_ACT_IN       = 0.0123f;
/* … w2/w3, scale_w2/3, scale_act_h1/h2, biais … */
```

> Header **vide par défaut** (`EWC_INT8_V2_WEIGHTS_PROVIDED 0`) → fallback init, **0 régression** sur les
> builds existants. Symétrique au pattern `EWC_HEAD_WEIGHTS_PROVIDED` (Sprint 32).

## Vérification

```bash
python scripts/export_weights_c.py --int8-v2 --config configs/board_ewc.yaml \
    --checkpoint experiments/.../ewc_head.pt
grep EWC_INT8_V2_WEIGHTS_PROVIDED firmware/stm32f4_blink/inc/ewc_head_int8_v2_weights.h
cd firmware/stm32f4_blink && make test          # parité C v2 ↔ Python (S3909)
```
