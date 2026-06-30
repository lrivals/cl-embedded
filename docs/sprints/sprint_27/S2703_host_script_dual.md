# S2712 — Script hôte `board_dual_pipeline.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 27 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté — `board_dual_pipeline.py` opérationnel, exp_S27_01 produit sur board réelle |
| **Durée estimée** | 3h |
| **Dépendances** | S2711 ✅ (`sensor_stream.py` étendu), CMAPSS FD001 + CWRU datasets disponibles dans `data/raw/`, loaders `src/data/cmapss_loader.py` + `src/data/cwru_dataset.py` |
| **Fichiers cibles** | `scripts/board_dual_pipeline.py` (nouveau) |
| **Référence** | `scripts/simulate_rul_board.py` (pattern CMAPSS), `scripts/simulate_multiclass_board.py` (pattern CWRU), `scripts/board_experiment_recorder.py` (pattern output JSON) |

---

## Contexte

`board_dual_pipeline.py` est le driver hôte pour le mode DUAL_MODE. Il :
1. Charge les features CMAPSS FD001 (5 features, labels RUL) et CWRU task 0 (9 features, labels faute) indépendamment
2. Construit des trames DUAL_MODE en zippant les deux streams
3. Envoie chaque trame à la board via UART
4. Parse les réponses 25 octets
5. Calcule RMSE_RUL + F1_fault offline (vérification vs métriques on-board)
6. Sauvegarde les résultats dans `experiments/exp_S27_01/dual_results.json`

---

## Structure du script

```python
#!/usr/bin/env python3
"""board_dual_pipeline.py — Driver hôte DUAL_MODE : RUL + Faute simultanés.

Usage:
    python scripts/board_dual_pipeline.py --port /dev/ttyACM0 --n-samples 200 --update
    python scripts/board_dual_pipeline.py --dry-run --n-samples 200 --output experiments/exp_S27_01

Encodage dual :
    - features[0:5] = top-5 CMAPSS FD001 (normalisées z-score)
    - features[5:9] = 4 features CWRU supplémentaires (slots [5:8] du loader CWRU)
    - TASK_ID = fault_label ∈ [0, N_CLASSES-1]  (TASK_ID réutilisé en DUAL_MODE)
    - label   = rul_u8 = round(RUL / RUL_CAP × 255)
"""

import argparse
import json
import time
import struct
from pathlib import Path
import numpy as np

from sensor_stream import (
    build_frame_v2,
    parse_response,
    FRAME_FLAGS_DUAL_MODE,
    FRAME_FLAGS_UPDATE,
    FRAME_FLAGS_PROFILING,
    FRAME_FLAGS_CONSOLIDATE,
    RESPONSE_DUAL_SIZE,
)
from src.data.cmapss_loader import CMAPSSLoader
from src.data.cwru_dataset import CWRUDataset

RUL_CAP      = 300    # cycles CMAPSS FD001 — RUL max pour normalisation
N_CMAPSS_FEAT = 5     # top-5 features CMAPSS (cmapss_feature_subset.yaml)
N_CWRU_EXTRA  = 4     # features CWRU slots [5:8] (index dans vecteur 9-dim)
N_FEATURES    = 9     # features totales envoyées dans la trame DUAL_MODE


def load_cmapss_fd001(n_samples: int):
    """Retourne (X[N,5], y_rul[N]) normalisés z-score, troncation à RUL_CAP."""
    loader = CMAPSSLoader(fd_set="FD001", config_path="configs/cmapss_feature_subset.yaml")
    X, y = loader.get_top5_features(split="test")
    y = np.clip(y, 0, RUL_CAP)
    return X[:n_samples], y[:n_samples]


def load_cwru_task0(n_samples: int):
    """Retourne (X[N,9], y_fault[N]) — task 0 = normal + inner fault."""
    ds = CWRUDataset(config_path="configs/cwru_by_fault_config.yaml")
    X, y = ds.get_task(task_id=0)
    return X[:n_samples], y[:n_samples]


def run_dual_experiment(ser, X_rul, y_rul, X_cwru, y_fault,
                        update: bool, consolidate_at: int, verbose: bool):
    """Stream les samples duaux vers la board, retourne les résultats."""
    n = min(len(X_rul), len(X_cwru))
    results = []

    for i in range(n):
        # ── Construire vecteur 9 features ────────────────────────────────
        features = np.zeros(N_FEATURES, dtype=np.float32)
        features[:N_CMAPSS_FEAT] = X_rul[i]            # slots [0:4]
        features[N_CMAPSS_FEAT:] = X_cwru[i][5:9]      # slots [5:8] CWRU extra

        # ── Encoder les deux labels ───────────────────────────────────────
        rul_u8      = int(round(min(float(y_rul[i]), RUL_CAP) / RUL_CAP * 255))
        fault_label = int(y_fault[i])

        # ── Flags ─────────────────────────────────────────────────────────
        flags = FRAME_FLAGS_DUAL_MODE | FRAME_FLAGS_PROFILING
        if update:
            flags |= FRAME_FLAGS_UPDATE
        if consolidate_at > 0 and i > 0 and i % consolidate_at == 0:
            flags |= FRAME_FLAGS_CONSOLIDATE

        # ── Construire et envoyer la trame ────────────────────────────────
        frame = build_frame_v2(
            features=features.tolist(),
            label=rul_u8,
            task_id=fault_label,    # TASK_ID réutilisé comme fault_label
            ts_ms=int(time.time() * 1000) & 0xFFFFFFFF,
            flags=flags,
        )
        ser.write(frame)

        # ── Lire et parser la réponse 25 B ────────────────────────────────
        raw = ser.read(RESPONSE_DUAL_SIZE)
        if len(raw) != RESPONSE_DUAL_SIZE:
            print(f"[WARN] sample {i} : réponse {len(raw)} B (attendu {RESPONSE_DUAL_SIZE})")
            continue
        resp = parse_response(raw)

        rul_pred_cycles = resp["rul_pred"] * RUL_CAP   # dénormaliser → cycles
        results.append({
            "i":            i,
            "rul_true":     float(y_rul[i]),
            "rul_pred":     rul_pred_cycles,
            "fault_true":   fault_label,
            "fault_pred":   resp["pred"],
            "conf_fault":   resp["confidence"],
            "latency_us":   resp["latency_us"],
            "f1_board":     resp["f1_macro"],
            "rmse_board":   resp["rmse_rul"],
            "forgetting":   resp["forgetting"],
        })

        if verbose and i % 50 == 0:
            print(f"  [{i:3d}] RUL true={y_rul[i]:.0f} pred={rul_pred_cycles:.1f} | "
                  f"fault true={fault_label} pred={resp['pred']} | lat={resp['latency_us']} µs")

    return results


def compute_offline_metrics(results):
    """RMSE RUL + F1-macro faute calculés hors board pour vérification."""
    rul_true  = np.array([r["rul_true"]   for r in results])
    rul_pred  = np.array([r["rul_pred"]   for r in results])
    f_true    = np.array([r["fault_true"] for r in results])
    f_pred    = np.array([r["fault_pred"] for r in results])

    rmse = float(np.sqrt(np.mean((rul_true - rul_pred) ** 2)))

    classes = np.unique(f_true)
    f1_per_class = []
    for c in classes:
        tp = np.sum((f_true == c) & (f_pred == c))
        fp = np.sum((f_true != c) & (f_pred == c))
        fn = np.sum((f_true == c) & (f_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_class.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    f1_macro = float(np.mean(f1_per_class))

    return {"rmse_rul_offline": rmse, "f1_fault_offline": f1_macro}
```

---

## Arguments CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--port` | `/dev/ttyACM0` | Port série NUCLEO-F439ZI |
| `--baud` | `115200` | Baudrate UART |
| `--n-samples` | `200` | Nombre de samples à streamer |
| `--update` | `False` | Activer le flag UPDATE (SGD on-board) |
| `--consolidate-at` | `0` | Période de consolidation (0 = désactivé) |
| `--dry-run` | `False` | Simuler sans board (données synthétiques) |
| `--output` | `experiments/exp_S27_01` | Répertoire de sortie |
| `--verbose` | `False` | Afficher chaque sample |

---

## Mode `--dry-run`

En dry-run, le script génère des prédictions synthétiques plausibles basées sur les performances Sprint 26 (RMSE_RUL ~23, F1_fault ~0.60) avec bruit gaussien. Permet de valider le pipeline de parsing et de sérialisation JSON sans board.

```python
def dry_run_response(rul_true, fault_label):
    """Simule une réponse board plausible."""
    rul_pred    = rul_true + np.random.normal(0, 22)   # RMSE ~22
    fault_pred  = fault_label if np.random.rand() > 0.4 else (fault_label + 1) % 10
    return {
        "pred":        fault_pred,
        "confidence":  np.random.uniform(0.4, 0.9),
        "rul_pred":    max(0, rul_pred) / RUL_CAP,     # normalisé [0,1]
        "latency_us":  np.random.randint(550, 720),    # ~636 µs ±10%
        "f1_macro":    np.random.uniform(0.50, 0.70),
        "rmse_rul":    np.random.uniform(20, 26) / RUL_CAP,
        "forgetting":  np.random.uniform(0.0, 0.05),
    }
```

---

## Format du fichier de sortie `dual_results.json`

```json
{
  "experiment": "exp_S27_01",
  "mode":       "dual",
  "dataset_rul": "CMAPSS_FD001",
  "dataset_fault": "CWRU_task0",
  "n_samples":  200,
  "update":     true,
  "metrics_board": {
    "rmse_rul":   21.8,
    "f1_fault":   0.61,
    "lat_mean_us": 638,
    "lat_p99_us":  712,
    "forgetting":  0.02
  },
  "metrics_offline": {
    "rmse_rul_offline": 22.3,
    "f1_fault_offline": 0.59
  },
  "bss_bytes": 65266,
  "samples": [
    {"i": 0, "rul_true": 191, "rul_pred": 188.4, "fault_true": 0, "fault_pred": 0, ...},
    ...
  ]
}
```

---

## Vérification

```bash
# 1. Dry-run (sans board)
python scripts/board_dual_pipeline.py --dry-run --n-samples 200 \
    --output experiments/exp_S27_01
# → Produit dual_results.json sans erreur

# 2. Board live
make flash   # depuis firmware/stm32f4_blink/
python scripts/board_dual_pipeline.py \
    --port /dev/ttyACM0 --n-samples 200 --update \
    --output experiments/exp_S27_01 --verbose
# → Réponses 25 B, pas de timeout, JSON produit

# 3. Vérification métriques
python -c "
import json
r = json.load(open('experiments/exp_S27_01/dual_results.json'))
print('RMSE_RUL:', r['metrics_board']['rmse_rul'])   # < 24.3
print('F1_fault:', r['metrics_board']['f1_fault'])   # > 0.50
print('Lat mean:', r['metrics_board']['lat_mean_us'])  # < 1000 µs
"
```
