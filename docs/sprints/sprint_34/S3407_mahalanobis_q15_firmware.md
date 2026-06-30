# S3407 / S3408 — Q15 Mahalanobis (firmware) + expérience board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 34 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h (S3407) + 2h (S3408) |
| **Dépendances** | S3405 (`mahalanobis_int8.py` quant="q15") · `scripts/export_weights_c.py` ✅ |
| **Fichiers cibles** | `firmware/stm32f4_blink/src/mahalanobis_q15.c`, `firmware/stm32f4_blink/inc/mahalanobis_q15.h`, `firmware/stm32f4_blink/src/pipeline.c`, `scripts/sensor_stream.py`, `experiments/exp_S34_board_maha_q15/` |
| **Références** | `pipeline.c` (dispatch FLAGS confirmé : DUAL_MODE `0x70`, PAIR_MODE `0x90/0xA0/0xB0`), `sensor_stream.py` (FLAGS confirmées : `0x90-0xBF` nibble-haut libres, `0xD0/0xE0/0xF0` disponibles), `export_weights_c.py:97-104` (pattern export FP32 `_array1d_to_c`/`_array2d_to_c`) |

---

## Contexte

Porte le mode Q15 Python (S3405) en C, suivant le pattern de parité board↔PC des Sprints
26-28. Le byte FLAGS est confirmé **sans collision** : DUAL_MODE occupe `0x70`, PAIR_MODE
`0x90/0xA0/0xB0` — les nibbles hauts `0xD0`, `0xE0`, `0xF0` sont libres.

---

## S3407 — `mahalanobis_q15.c/.h` + intégration pipeline

```c
// inc/mahalanobis_q15.h
#pragma once
#include <stdint.h>

#define MAHA_Q15_N_FEATURES <N>   /* identique à mahalanobis.h existant */

typedef struct {
    int16_t mu_q8[MAHA_Q15_N_FEATURES];                          /* MEM: inchangé, INT8 promu */
    int16_t sigma_inv_q15[MAHA_Q15_N_FEATURES][MAHA_Q15_N_FEATURES]; /* MEM: d^2 x 2 B */
    float   sigma_inv_scale;   /* scale par-tenseur, exporté avec les poids */
} MahalanobisQ15;

void  maha_q15_init(MahalanobisQ15 *m);
float maha_q15_distance(const MahalanobisQ15 *m, const float *x);  /* accumulation int32 */
```

- Accumulation de la forme quadratique en **int32** (éviter overflow sur produits int16 x
  int16 sommés sur `d^2` termes), conversion finale en FP32 via la FPU disponible sur
  Cortex-M4 (pas de contrainte INT8 stricte sur la NUCLEO, règle CLAUDE.md).
- Nouveau flag `MAHA_Q15_MODE = 0xD0` (ou `0xE0`/`0xF0`) dans `pipeline.h` — vérifier
  l'absence de collision avec les masks existants avant de fixer la valeur définitive.
- **MAJ `sensor_stream.py` en parallèle** (`parse_response`, nouvelle constante de flag) —
  règle CLAUDE.md : ne jamais désynchroniser protocole UART ↔ host script.
- Poids exportés via `scripts/export_weights_c.py` (étendre le pattern FP32 existant,
  lignes 97-104, avec une fonction de quantification Q15 — **ne jamais éditer
  `model_weights.h` à la main**, règle CLAUDE.md).

## S3408 — Expérience board

`experiments/exp_S34_board_maha_q15/{dataset}.json` : parité board↔PC des
distances/prédictions (protocole Sprints 26-28 : mêmes entrées → mêmes sorties), latence
DWT, `.bss`. Aucun chiffre inventé tant que la board n'a pas tourné.

---

## Vérification

```bash
cd firmware/stm32f4_blink && make all && arm-none-eabi-size build/stm32f4_blink.elf
make test   # test_mahalanobis_q15.c (S3409) : parité forward C ↔ Python

python scripts/sensor_stream.py --port /dev/ttyACM0 --model maha-q15 --dataset cmapss
```

---

## Réalisé (S3407/S3408)

- **Flag** : nibble `0xF0` (PROTO_FLAG_MAHA_Q15) — **seul libre** (0x10–0xE0 tous pris depuis
  TRIPLE_MODE S31 ; la doc citait 0xD0/0xE0/0xF0, corrigé). Dispatch en early-return par nibble
  **avant** la chaîne de bits (`0xF0 & 0x70 == 0x70` DUAL, `& 0x30 == 0x30` MULTICLASS).
- **Calcul** : `maha_q15.c/.h` stocke `mu_q8` (uint8 affine, **pas** int8 — Python quantifie en
  [0,255]) + `sigma_inv_q15` (int16) ; **déquant → distance FP32 sur la FPU** (décision : parité
  bit-à-bit avec Python plutôt qu'accumulation int32). RAM ÷2 sur Σ⁻¹ préservée (storage int16).
- **Réponse V3 (23 B)** réutilisée — aucun nouveau format. `sensor_stream.py` : `--model maha-q15`,
  `FRAME_FLAGS_MAHA_Q15=0xF0`, résolution taille réponse → V3 (synchro UART respectée).
- **Export** : `export_weights_c.py --maha-q15 <pkl>` → `inc/mahalanobis_q15_weights.h` (généré,
  `MAHA_Q15_WEIGHTS_PROVIDED`), + `--maha-q15-test-vectors` → `tests/test_vectors_q15.h`. Vide par
  défaut → init identité, **0 régression**.
- **S3408 board NUCLEO-F439ZI réelle** (`scripts/run_s34_board_maha_q15.py`, train→export→build→
  flash→stream sans `--update`) : **parité board↔PC EXACTE** sur CMAPSS + Pronostia (300/300
  prédictions, `max_score_err` 9.6e-6 / 1.5e-3) ; **latence DWT P50=P99=5 µs ≪ 100 ms (Gap 2 ✅)** ;
  `.bss=105 036 B` (53.7 % de 192 Ko, +80 B `g_maha_q15`). `exp_S34_board_maha_q15/{cmapss,
  pronostia}.json` + `summary.json`.
