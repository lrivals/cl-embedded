# S2609–S2610 — Scripts de simulation host UART board

| Champ | Valeur |
|-------|--------|
| **Sprint** | 26 |
| **Priorité** | 🔴 Critique (S2609) / 🟡 Important (S2610) |
| **Statut** | ✅ TERMINÉ |
| **Durée estimée** | S2609 : 2h / S2610 : 1h30 = 3h30 total |
| **Dépendances** | S2605 ✅ (pipeline.c étendu, firmware compilé et flashé), S2607/S2608 ✅ (poids exportés), board NUCLEO-F439ZI connectée sur `/dev/ttyACM0` |
| **Fichiers cibles** | `scripts/simulate_rul_board.py`, `scripts/simulate_multiclass_board.py` |
| **Référence** | `scripts/sensor_stream.py` (pattern complet UART v3 côté host), `firmware/stm32f4_blink/inc/pipeline.h` (constants FLAGS, MAGIC, CRC8) |

---

## Contexte

Les scripts suivent le même pattern que `sensor_stream.py` : construction de trames UART binaires (protocole v3), envoi vers board, lecture de la réponse 21 B, calcul de métriques côté host, log JSON.

**Différences par rapport à `sensor_stream.py`** :

| Aspect | `sensor_stream.py` | `simulate_rul_board.py` | `simulate_multiclass_board.py` |
|--------|---------------------|------------------------|-------------------------------|
| Dataset | Monitoring / CWRU | CMAPSS FD001 | CWRU |
| Flag mode | `EWC_MODE (0x10)` | `RUL_MODE (0x80)` | `MULTICLASS_MODE (0x90)` |
| Label | 0/1 binaire (`uint8`) | RUL float → `uint8` clampé | classe 0–9 (`uint8`) |
| Réponse `conf:f32` | probabilité ∈ [0,1] | RUL prédit (float) | probabilité classe prédite |
| Métrique host | AUROC / Accuracy | RMSE | F1-macro |

---

## S2609 — `scripts/simulate_rul_board.py`

### Spec complète

```python
#!/usr/bin/env python3
"""
simulate_rul_board.py — Envoie séquences CMAPSS FD001 via UART à la NUCLEO-F439ZI,
collecte les prédictions RUL, calcule RMSE vs labels réels et log latence DWT.

Usage :
    python scripts/simulate_rul_board.py \\
        --port /dev/ttyACM0 \\
        --config configs/cmapss_feature_subset.yaml \\
        --n-samples 200 \\
        --output experiments/exp_S26_01/board_rul_results.json \\
        [--baud 115200] [--task-id 0] [--update] [--consolidate-at 100]

Protocole : v3 binaire, flag RUL_MODE=0x80, réponse 21 B.
"""

from __future__ import annotations
import argparse
import json
import math
import struct
import time
from pathlib import Path

import serial
import numpy as np

from src.data.cmapss_loader import get_cl_dataloaders

# ── Constantes protocole (miroir de pipeline.h) ────────────────────────────
PROTO_MAGIC      = b'\xCD\xAB'
PROTO_VERSION_V3 = 0x03
PROTO_FLAG_UPDATE      = 0x01
PROTO_FLAG_CONSOLIDATE = 0x04
PROTO_FLAG_EWC_MODE    = 0x10
PROTO_FLAG_RUL_MODE    = 0x80   # Sprint 26

RESPONSE_SIZE = 21   # [pred:u8][conf:f32][lat_us:u32][acc:f32][auroc:f32][forgetting:f32]


def crc8(data: bytes) -> int:
    """Polynomial 0x07 — identique à pipeline.c::proto_crc8."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_frame(features: np.ndarray, label_rul: float, task_id: int,
                flags: int, timestamp_ms: int) -> bytes:
    """
    Construit une trame UART v3 binaire.

    label : RUL clampé en uint8 (0–255). En mode RUL, le board lit g_recv_label
    comme uint8 et le cast en float. Pour des RUL > 255, normaliser avant l'envoi
    (ex. RUL / 125 × 255).
    """
    n = len(features)
    label_u8 = min(255, max(0, int(round(label_rul))))

    # Header
    payload = PROTO_MAGIC
    payload += struct.pack('<B', PROTO_VERSION_V3)
    payload += struct.pack('<B', task_id)
    payload += struct.pack('<I', timestamp_ms & 0xFFFFFFFF)
    payload += struct.pack('<B', n)

    # Features (float32 little-endian)
    for f in features:
        payload += struct.pack('<f', float(f))

    payload += struct.pack('<B', label_u8)
    payload += struct.pack('<B', flags)

    # CRC8 sur tout le payload avant le CRC
    payload += struct.pack('<B', crc8(payload))
    return payload


def parse_response(data: bytes) -> dict:
    """Décode une réponse v3 de 21 B."""
    pred, = struct.unpack_from('<B', data, 0)
    conf, = struct.unpack_from('<f', data, 1)    # = RUL prédit en mode RUL_MODE
    lat,  = struct.unpack_from('<I', data, 5)
    acc,  = struct.unpack_from('<f', data, 9)
    aur,  = struct.unpack_from('<f', data, 13)
    fgt,  = struct.unpack_from('<f', data, 17)
    return {
        "pred": pred,
        "rul_pred": conf,   # champ conf réutilisé pour RUL
        "lat_us": lat,
        "acc": acc,
        "auroc": aur,
        "forgetting": fgt,
    }


def run_simulation(
    port: str,
    config_path: Path,
    n_samples: int,
    output_path: Path,
    baud: int = 115200,
    task_id: int = 0,
    do_update: bool = True,
    consolidate_at: int | None = None,
) -> None:
    tasks = get_cl_dataloaders(config_path=config_path, mode="rul")
    task = tasks[task_id]   # DataLoader CMAPSS FD001

    ser = serial.Serial(port, baud, timeout=2.0)
    time.sleep(0.5)   # attendre reset MCU

    rul_preds: list[float] = []
    rul_trues: list[float] = []
    latencies: list[int]   = []

    flags_base = PROTO_FLAG_RUL_MODE
    if do_update:
        flags_base |= PROTO_FLAG_UPDATE

    for i, (x_batch, y_batch) in enumerate(task["train"]):
        if i >= n_samples:
            break

        x = x_batch[0].numpy()   # 1 sample
        y = y_batch[0].item()

        flags = flags_base
        if consolidate_at is not None and i == consolidate_at:
            flags |= PROTO_FLAG_CONSOLIDATE

        frame = build_frame(x, y, task_id=task_id, flags=flags,
                            timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF)
        ser.write(frame)

        resp_bytes = ser.read(RESPONSE_SIZE)
        if len(resp_bytes) < RESPONSE_SIZE:
            print(f"[WARN] Sample {i}: réponse tronquée ({len(resp_bytes)} B)")
            continue

        r = parse_response(resp_bytes)
        rul_preds.append(r["rul_pred"])
        rul_trues.append(y)
        latencies.append(r["lat_us"])

        if i % 20 == 0:
            rmse = math.sqrt(sum((p-t)**2 for p, t in zip(rul_preds, rul_trues)) / len(rul_preds))
            print(f"[{i:4d}] RUL_pred={r['rul_pred']:.1f} true={y:.1f} "
                  f"RMSE={rmse:.2f} lat={r['lat_us']}µs")

    ser.close()

    rmse_final = math.sqrt(
        sum((p-t)**2 for p, t in zip(rul_preds, rul_trues)) / len(rul_preds)
    )
    lat_arr = sorted(latencies)
    n = len(lat_arr)

    results = {
        "n_samples": len(rul_preds),
        "rmse_board": rmse_final,
        "latency_p50_us": lat_arr[n // 2],
        "latency_p99_us": lat_arr[int(n * 0.99)],
        "latency_max_us": lat_arr[-1],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ RMSE board = {rmse_final:.2f} | lat P50={results['latency_p50_us']}µs")
    print(f"Résultats → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default="/dev/ttyACM0")
    parser.add_argument("--config",  type=Path,
                        default=Path("configs/cmapss_feature_subset.yaml"))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output",    type=Path,
                        default=Path("experiments/exp_S26_01/board_rul_results.json"))
    parser.add_argument("--baud",      type=int, default=115200)
    parser.add_argument("--task-id",   type=int, default=0)
    parser.add_argument("--update",    action="store_true", default=True)
    parser.add_argument("--consolidate-at", type=int, default=None)
    args = parser.parse_args()

    run_simulation(
        port=args.port,
        config_path=args.config,
        n_samples=args.n_samples,
        output_path=args.output,
        baud=args.baud,
        task_id=args.task_id,
        do_update=args.update,
        consolidate_at=args.consolidate_at,
    )


if __name__ == "__main__":
    main()
```

### Exécution

```bash
# Board connectée, firmware flashé (S2601 + S2605 implémentés)
python scripts/simulate_rul_board.py \
    --port /dev/ttyACM0 \
    --n-samples 200 \
    --output experiments/exp_S26_01/board_rul_results.json

# Résultat attendu :
#   RMSE board ≤ RMSE_PC × 1.10 (critère exp_S26_01)
#   lat P50 ≤ 100 000 µs (critère Gap 2)
```

---

## S2610 — `scripts/simulate_multiclass_board.py`

### Spec complète

```python
#!/usr/bin/env python3
"""
simulate_multiclass_board.py — Envoie échantillons CWRU (9 features) via UART,
collecte classes prédites, calcule F1-macro on-board.

Usage :
    python scripts/simulate_multiclass_board.py \\
        --port /dev/ttyACM0 \\
        --config configs/cwru_by_fault_config.yaml \\
        --n-samples-per-task 100 \\
        --output experiments/exp_S26_02/board_mc_results.json \\
        [--n-classes 10] [--baud 115200]
"""

from __future__ import annotations
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import serial
from sklearn.metrics import f1_score

from src.data.cwru_dataset import get_cl_dataloaders

PROTO_MAGIC           = b'\xCD\xAB'
PROTO_VERSION_V3      = 0x03
PROTO_FLAG_UPDATE     = 0x01
PROTO_FLAG_CONSOLIDATE = 0x04
PROTO_FLAG_EWC_MODE   = 0x10
PROTO_FLAG_RUL_MODE   = 0x80
PROTO_FLAG_MULTICLASS = PROTO_FLAG_EWC_MODE | PROTO_FLAG_RUL_MODE  # 0x90

RESPONSE_SIZE = 21


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_frame(features: np.ndarray, label: int, task_id: int,
                flags: int, timestamp_ms: int) -> bytes:
    n = len(features)
    payload = PROTO_MAGIC
    payload += struct.pack('<B', PROTO_VERSION_V3)
    payload += struct.pack('<B', task_id)
    payload += struct.pack('<I', timestamp_ms & 0xFFFFFFFF)
    payload += struct.pack('<B', n)
    for f in features:
        payload += struct.pack('<f', float(f))
    payload += struct.pack('<B', label & 0xFF)
    payload += struct.pack('<B', flags)
    payload += struct.pack('<B', crc8(payload))
    return payload


def parse_response(data: bytes) -> dict:
    pred, = struct.unpack_from('<B', data, 0)
    conf, = struct.unpack_from('<f', data, 1)
    lat,  = struct.unpack_from('<I', data, 5)
    return {"pred": pred, "conf": conf, "lat_us": lat}


def run_simulation(
    port: str,
    config_path: Path,
    n_samples_per_task: int,
    output_path: Path,
    n_classes: int = 10,
    baud: int = 115200,
) -> None:
    tasks = get_cl_dataloaders(config_path=config_path, mode="multiclass")

    ser = serial.Serial(port, baud, timeout=2.0)
    time.sleep(0.5)

    all_preds: list[int] = []
    all_trues: list[int] = []
    latencies: list[int] = []

    for task_id, task in enumerate(tasks):
        print(f"\n── Tâche {task_id} ──")
        flags = PROTO_FLAG_MULTICLASS | PROTO_FLAG_UPDATE
        last_idx = n_samples_per_task - 1

        for i, (x_batch, y_batch) in enumerate(task["train"]):
            if i >= n_samples_per_task:
                break

            x = x_batch[0].numpy()
            y = int(y_batch[0].item())
            f = flags | (PROTO_FLAG_CONSOLIDATE if i == last_idx else 0)

            frame = build_frame(x, y, task_id=task_id, flags=f,
                                timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF)
            ser.write(frame)
            resp = ser.read(RESPONSE_SIZE)
            if len(resp) < RESPONSE_SIZE:
                continue

            r = parse_response(resp)
            all_preds.append(r["pred"])
            all_trues.append(y)
            latencies.append(r["lat_us"])

        # F1-macro partiel après chaque tâche
        f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)
        print(f"  F1-macro après task{task_id} : {f1:.3f} ({len(all_preds)} samples)")

    ser.close()

    f1_final = f1_score(all_trues, all_preds, average="macro", zero_division=0)
    lat_s = sorted(latencies)
    n = len(lat_s)

    results = {
        "n_classes": n_classes,
        "n_tasks": len(tasks),
        "n_samples": len(all_preds),
        "f1_macro_board": f1_final,
        "latency_p50_us": lat_s[n // 2],
        "latency_p99_us": lat_s[int(n * 0.99)],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ F1-macro board = {f1_final:.3f} (critère ≥ 0.60)")
    print(f"Résultats → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default="/dev/ttyACM0")
    parser.add_argument("--config",  type=Path,
                        default=Path("configs/cwru_by_fault_config.yaml"))
    parser.add_argument("--n-samples-per-task", type=int, default=100)
    parser.add_argument("--output",  type=Path,
                        default=Path("experiments/exp_S26_02/board_mc_results.json"))
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--baud",    type=int, default=115200)
    args = parser.parse_args()

    run_simulation(
        port=args.port,
        config_path=args.config,
        n_samples_per_task=args.n_samples_per_task,
        output_path=args.output,
        n_classes=args.n_classes,
        baud=args.baud,
    )


if __name__ == "__main__":
    main()
```

### Exécution

```bash
python scripts/simulate_multiclass_board.py \
    --port /dev/ttyACM0 \
    --n-samples-per-task 100 \
    --output experiments/exp_S26_02/board_mc_results.json

# Résultat attendu :
#   F1-macro board ≥ 0.60 (critère exp_S26_02)
#   lat P50 ≤ 100 000 µs (critère Gap 2)
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2609 — `simulate_rul_board.py` | ✅ | Flags corrigés (0x50), loader `get_cl_dataloaders(data_dir, config_path, mode="rul")`, RUL normalisé [0,255] via CMAPSS_RUL_CAP |
| S2610 — `simulate_multiclass_board.py` | ✅ | Flags corrigés (0x30), loader remplacé par `get_cl_splits(scenario="by_fault_type", mode="multiclass")`, --csv-path au lieu de --config |

---

## Questions ouvertes

- `FIXME(gap2)` : Le label RUL est encodé en `uint8` (0–255) dans le protocole UART. CMAPSS FD001 a des RUL jusqu'à 362 cycles. Deux options : (1) clipper à 125 (RUL cap du dataset, `CMAPSS_RUL_CAP`) puis mapper sur [0, 255] — **option retenue** ; (2) encoder RUL comme float dans les 4 derniers octets de features. Option 1 est plus simple mais perd de la résolution. Documenter dans le manuscrit.
- `TODO(dorra)` : Le flag `PROTO_FLAG_MULTICLASS = 0x90` correspond à `EWC_MODE | RUL_MODE`. Vérifier dans `pipeline.c` que le test `(flags & 0x90) == 0x90` est évalué **avant** le test `flags & EWC_MODE (0x10)` pour éviter une mauvaise branche.
- `TODO(fred)` : Les résultats `board_rul_results.json` peuvent être intégrés dans le benchmark Edge Spectrum initié au Sprint 23. Confirmer le format JSON attendu.
