# S4804 — Driver board + mesures réelles (DWT, `.bss`, parité, CRC)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 48 |
| **Priorité** | 🔴 Critique — produit les mesures que l'émulateur ne peut pas donner. |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir (board supposée disponible) |
| **Durée estimée** | 6h |
| **Dépendances** | S4802 (kernel) · S4803 (export) · S4801 (matrice de portage) |
| **Fichiers cibles** | `scripts/run_s48_board_depth.py`, `experiments/exp_S48_board/` |
| **Références** | patrons `run_sprint36_board.py`, `run_s34_board_maha_q15.py` (train→export→build→flash→stream) ; DWT profiling (S20/S29) |

---

## Contexte

Driver bout-en-bout par cellule de la matrice S4801, sur **NUCLEO-F439ZI réelle** (supposée disponible). Il
mesure ce que le Sprint 47 ne peut pas : **latence DWT**, **`.bss` réel** (packé vs non-packé), **parité
board↔PC**, **intégrité CRC**.

## Spec

### 1. Boucle du driver `run_s48_board_depth.py`

```
Pour chaque cellule (dataset, weight_bits, granularité, symétrie, packing) de S4801 :
  1. Train réf EWC (checkpoint FP32, voie AUROC S28) sur le dataset.
  2. export_weights_c.py --ewc-subint8 --weight-bits N --granularity G [--packed]  (S4803)
  3. make clean && make CFLAGS_EXTRA="-DEWC_INT<N> [-DEWC_INTx_PACKED] -DEWC_SUBINT8_WEIGHTS_PROVIDED"
  4. make size  → relève .bss (packé / non-packé)
  5. make flash
  6. sensor_stream.py --port /dev/ttyACM0 --dataset <ds> --proto 3   (sans --update)
     → collecte prédictions + latence DWT P50/P99 + compteur CRC
  7. AUROC board vs labels ; parité board↔PC (émulateur S47 rejoué sur le même ordre)
  8. Écrire experiments/exp_S48_board/exp_S48_<ds>_<bits>_<gran>[_packed].json
```

### 2. Schéma JSON (cellule)

```json
{
  "dataset": "monitoring", "weight_bits": 4, "granularity": "per_channel",
  "symmetry": "symmetric", "packed": true,
  "auroc_board": null, "auroc_pc_emulator": null, "parity_pred": null,
  "latency_dwt_p50_us": null, "latency_dwt_p99_us": null,
  "bss_bytes": null, "bss_bytes_int8_ref": null, "ram_ratio_measured_vs_int8": null,
  "crc_errors": null, "gap2_ok": null, "na_reason": null,
  "config_snapshot": { ... }
}
```

Tous `null` avant streaming (**aucun chiffre inventé**). Si un schéma déborde la SRAM ou casse l'AUROC →
`na_reason` renseigné (N/A honnête, précédent PSI×gas_sensor S45).

### 3. Mesures attendues (gabarit — `pending`)

| Cellule | `.bss` non-packé | `.bss` packé | latence P50 (µs) | AUROC board | parité |
|---------|:---:|:---:|:---:|:---:|:---:|
| Monitoring INT4 pc | pending | pending | pending | pending | pending |
| Pronostia INT4 pc | pending | pending | pending | pending | pending |
| Monitoring INT2 pc | pending | pending | pending | pending | pending |
| Pronostia INT2 pc | pending | pending | pending | pending | pending |

**Attendus qualitatifs** (à confirmer par la mesure, non écrits comme chiffres) : `.bss` non-packé ≈ INT8 ;
`.bss` packé réduit ; latence ≪ 100 ms (Gap 2) même avec dépacking ; parité pred = 1.000 (schéma identique
émulateur/board).

## Contraintes

- Stream **sans `--update`** (frozen — on isole le schéma de quantification, pas l'apprentissage online).
- **0 CRC** attendu (intégrité UART) ; toute erreur consignée.
- `pending`/`null` avant exécution ; N/A honnête si débordement/dégénérescence.

## Vérification

```bash
python scripts/run_s48_board_depth.py --port /dev/ttyACM0 --cell monitoring_int4_perchannel_packed
python -c "import json,glob; d=json.load(open(sorted(glob.glob('experiments/exp_S48_board/*.json'))[-1])); assert {'latency_dwt_p50_us','bss_bytes','parity_pred'} <= d.keys()"
```

---

## Résolution (implémentée)

**Prérequis firmware (câblage `pipeline.c`, décision utilisateur)** : le kernel sub-INT8 (S4802) n'était **pas routé** dans `pipeline.c` — la route `0x40` exécutait le kernel v2 (S39). Ajout d'une route sub-INT8 **gardée** `#if defined(EWC_SUBINT8_WEIGHTS_PROVIDED)`, prioritaire sur le v2 : globals `g_ewc_subint8` (non-packé, réutilise `ewc_int8_v2_forward`) / `g_ewc_subint8_packed` (`ewc_subint8_packed_forward`), chargés par `memcpy` du header généré à l'init, forward frozen (pas d'`UPDATE` : S48 isole le schéma). **Garde de cohérence** `#error` si `EWC_INTx_PACKED` ≠ `EWC_SUBINT8_PACKED` (empêche le débordement `memcpy` int8→packé). **`.bss` défaut invariant 105 036 B**, `make test` 141 (2 TinyOL préexistants hors périmètre, 0 régression).

**Driver `run_s48_board_depth.py`** : boucle par cellule (train réf EWC FP32 1×/dataset via `train_ewc_head` → `export_weights_c.py --ewc-subint8` → `make EXTRA_CFLAGS="-D<FLAG> [-DEWC_INTx_PACKED] -DEWC_SUBINT8_WEIGHTS_PROVIDED" EWC_IN=k` → `.bss` → flash → `ss._stream_uart` flag 0x40 frozen → parité émulateur + AUROC). Robustesse `try/except` par cellule, N/A honnête (`_na`/`_pending`), `board_samples.json` persisté. `--all`/`--dataset`/`--mode`/`--cell`/`--packed`/`--no-flash`/`--no-stream`.

**12 cellules mesurées board réelle NUCLEO-F439ZI (cœur scientifique, choix utilisateur)** — packé+non-packé × {int4, ternaire, binaire} × {monitoring (k=4), pronostia (k=5)} :

| Cellule | `.bss` non-packé | `.bss` packé | gain (B) | lat P50 non-pk / pk (µs) | parité | CRC |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Monitoring INT4 pc | 105 640 | 105 304 | 336 | 67 / 123 | 1.000 | 0 |
| Monitoring ternaire pc | 105 640 | 105 136 | 504 | 67 / 123 | 1.000 | 0 |
| Monitoring binaire pc | 105 640 | 105 068 | 572 | 67 / 125 | 1.000 | 0 |
| Pronostia INT4 pc | 106 152 | 105 816 | 336 | 70 / 127 | 1.000 | 0 |
| Pronostia ternaire pc | 106 152 | 105 648 | 504 | 70 / 127 | 1.000 | 0 |
| Pronostia binaire pc | 106 152 | 105 548 | 604 | 70 / 130 | 1.000 | 0 |

**Résultats clés** : (1) **`.bss` non-packé invariant par mode** (105 640 / 106 152) = nœud d'honnêteté confirmé (un sub-INT8 dans un `int8_t` n'économise rien) ; (2) **le packing matérialise le gain**, croissant quand les bits baissent (÷8 binaire > ÷4 ternaire > ÷2 INT4) ; (3) **latence dépacking ≈ +55 µs** (67→123 µs) mais **≪ 100 ms — Gap 2 ✅** ; (4) **parité board↔émulateur = 1.000 sur les 12** (0 mismatch, `max_score_err ≤ 1.2e-7`) ; (5) **0 CRC**. Aucun débordement SRAM à k≤5 (pas de N/A).
