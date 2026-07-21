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

_À compléter lors de l'implémentation (dès accès NUCLEO)._
