# S3106 / S3107 — Pipeline triple-modèle + expériences board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 31 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (board réelle NUCLEO-F439ZI) |
| **Durée estimée** | 3h (S3106) + 3h (S3107) |
| **Dépendances** | S3105 (`meta_head.c`) · Sprint 30 S3009 ✅ (DUAL_MODE généralisé) · `scripts/board_experiment_recorder.py` |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/pipeline.c`, `inc/pipeline.h`, `scripts/sensor_stream.py`, `experiments/exp_S31_board_*/` |
| **Références** | `firmware/stm32f4_blink/src/pipeline.c` (DUAL_MODE l.430-478) |

---

## Contexte

Étend le DUAL_MODE généralisé du Sprint 30 (Maha + supervisé) en **triple-modèle** : les 2 sorties de base alimentent `meta_head` qui produit le verdict final, le tout en une trame UART.

## S3106 — pipeline.c triple + sensor_stream.py

- Après les forwards Mahalanobis + supervisé, construire le vecteur `feats` (`[score_maha, prob_sup, disagreement, conf_sup]`) et appeler `meta_predict`.
- Nouveau FLAG mode triple (vérifier collisions de bits — byte saturé, TODO dorra).
- Réponse étendue : sorties des 2 bases + verdict méta + latence.
- **MAJ `sensor_stream.py` en parallèle** (`parse_response`, format trame) — règle CLAUDE.md.

## S3107 — Expériences board (RAM profiling obligatoire)

- Via `board_experiment_recorder.py` : latence **triple-modèle** (Maha + supervisé + méta), vérifier **< 100 ms (Gap 2)**.
- `.bss` total mesuré (`arm-none-eabi-size`).
- **Parité board↔PC** : verdict méta board == verdict Python sur mêmes entrées.

---

## Vérification

```bash
# maha-ewc : générer meta_weights.h, builder, flasher, mesurer
python scripts/export_weights_c.py --meta experiments/exp_S31_PC_maha_ewc_cwru/meta_weights.json --dump-test-vectors
cd firmware/stm32f4_blink && make all && make flash && arm-none-eabi-size build/stm32f4_blink.elf
cd - && python scripts/sensor_stream.py --port /dev/ttyACM0 --dataset cwru --model triple-maha-ewc
python scripts/board_pair_recorder.py --pair maha-ewc --dataset cwru --triple --update \
    --n-samples 300 --output experiments/exp_S31_board_maha_ewc
# maha-hdc : ré-générer meta_weights.h (HDC) puis re-flasher avant de relancer
python scripts/export_weights_c.py --meta experiments/exp_S31_PC_maha_hdc_cwru/meta_weights.json --dump-test-vectors
```

---

## Bilan d'implémentation ✅ (board réelle NUCLEO-F439ZI)

**S3106 — pipeline triple :**
- `pipeline.h` : flags `PROTO_FLAG_TRIPLE_MAHA_EWC=0xD0`, `PROTO_FLAG_TRIPLE_MAHA_HDC=0xE0` (nibble haut libre, **aucune collision** : `0xC0`=TINYOL_INT8), `RESPONSE_TRIPLE_SIZE=27`.
- `pipeline.c` : bloc TRIPLE **avant** PAIR. Features `[p_maha, p_sup, disagreement, conf_sup]` (ordre `DEFAULT_FEATURES`) → `meta_forward(&g_meta)`. `p_maha = sigmoid(score − seuil)` (parité `ModelPair._maha_proba`) ; `p_sup` = softmax classe 1 (EWC) ou label (HDC, sans `predict_proba` côté PC). Réponse 27 B = PAIR 22 B + `[pred_meta:u8][prob_meta:f32]` (le slot `conf_sup` transporte `p_sup` pour reconstruction de parité). `g_meta` initialisé via `meta_init` dans `pipeline_init`.
- `sensor_stream.py` : `FRAME_FLAGS_TRIPLE_*`, `RESPONSE_TRIPLE_FMT="<BfBfIffBf"`, `parse_response` branche 27 B, modèles `triple-maha-ewc/hdc`.

**S3107 — expériences board (300 samples, `--update`) :**

| Paire | lat. supervisé | lat. paire | **lat. triple** | overhead méta | parité board↔PC | `.bss` |
|-------|---------------|-----------|-----------------|---------------|------------------|--------|
| maha-ewc (0xD0) | 252 µs | 256 µs | **258 µs** | ~2 µs | **1.000** (Δprob=0.0) | 104 596 B |
| maha-hdc (0xE0) | 653 µs | 657 µs | **593 µs** | négligeable* | **1.000** (Δprob=0.004) | 104 596 B |

- **Gap 2 ✅** : latence triple ≪ 100 ms (258 / 593 µs). Le méta-modèle (logreg 4 features) ajoute un coût négligeable ; la latence est dominée par le modèle supervisé (EWC ~250 µs, HDC ~650 µs).
- **Parité méta board↔PC = 1.000** sur les 300 échantillons (verdict board == verdict numpy reconstruit). `Δprob` HDC = 0.004 (arrondi float32↔float64 du sigmoïde) sans impact sur le verdict.
- `.bss` = **104 596 B** (39.9 % de 256 Ko), +20 B vs Sprint 30 (struct `g_meta`).
- Expériences : `experiments/exp_S31_board_maha_ewc/`, `experiments/exp_S31_board_maha_hdc/` (results.json + config_snapshot.yaml).

\* variation run-à-run (état d'apprentissage en ligne distinct par stream) — chiffres board réels, non lissés.
