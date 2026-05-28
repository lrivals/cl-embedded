# S2102–S2103 — Streamer Pronostia board + config board_pronostia.yaml

| Champ | Valeur |
|-------|--------|
| **Sprint** | 21 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Fait |
| **Durée estimée** | 2h + 1h |
| **Dépendances** | S2101 (`configs/pronostia_feature_subset.yaml` ✅) |
| **Fichiers cibles** | `scripts/sensor_stream.py`, `scripts/sensor_sim.py`, `configs/board_pronostia.yaml` |

---

## S2102 — Ajout `--dataset pronostia` dans sensor_stream.py

### Contexte

`sensor_stream.py` supporte actuellement `--dataset cwru` et `--dataset monitoring`.  
`sensor_sim.py` (legacy) a le même gap.  
`pronostia_dataset.py` expose `load_condition_features(npy_dir, condition)` → `(X[N,13], y[N])`.

### Ce qu'il faut ajouter

#### Dans `scripts/sensor_sim.py`

Ajouter `_load_pronostia()` en suivant le pattern `_load_monitoring()` :

```python
_PRONOSTIA_BINARIES = Path("data/raw/Pronostia dataset/binaries")
_PRONOSTIA_SUBSET_CFG = Path("configs/pronostia_feature_subset.yaml")

def _load_pronostia() -> tuple[np.ndarray, np.ndarray]:
    """Charge Pronostia (3 conditions concaténées) et applique la sélection 13→5 features."""
    import yaml
    from src.data.pronostia_dataset import load_condition_features, N_CONDITIONS

    subset = yaml.safe_load(_PRONOSTIA_SUBSET_CFG.read_text())
    indices = subset["feature_indices"]  # [2, 3, 8, 9, 12]

    all_X, all_y = [], []
    for cond in range(1, N_CONDITIONS + 1):
        X_cond, y_cond = load_condition_features(_PRONOSTIA_BINARIES, condition=cond)
        all_X.append(X_cond[:, indices].astype(np.float32))
        all_y.append(y_cond.astype(np.int64))
    return np.concatenate(all_X), np.concatenate(all_y)
```

Ajouter `"pronostia": _load_pronostia` dans `loaders` de `load_dataset()`.

#### Dans `scripts/sensor_stream.py`

Même logique : ajouter `_load_pronostia()` dans `_load_dataset()` en important depuis `sensor_sim`.  
Modifier `--dataset choices` : `["cwru", "monitoring", "pronostia"]`.

### Scénario CL Pronostia

Le découpage CL est par condition opératoire (3 tâches) :

```bash
# Dry-run CL séquence Pronostia
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:200,cond2:200,cond3:200 \
    --update --consolidate-on-task-change \
    --protocol-version 3 --dry-run --verbose \
    --output experiments/exp_S21_04

# Board live
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:200,cond2:200,cond3:200 \
    --update --consolidate-on-task-change \
    --protocol-version 3 --port /dev/ttyACM0 \
    --output experiments/exp_S21_04
```

### Vérification

```bash
# Test dry-run sans board (CI-compatible)
python scripts/sensor_stream.py --dataset pronostia --dry-run --n-samples 30 --verbose

# Test CL séquence dry-run
python scripts/sensor_stream.py --dataset pronostia \
    --cl-sequence cond1:10,cond2:10,cond3:10 \
    --dry-run --protocol-version 3 --verbose
```

---

## S2103 — `configs/board_pronostia.yaml`

### Format (calqué sur `board_ewc.yaml` + `board_mahalanobis.yaml`)

```yaml
# board_pronostia.yaml — Config multi-modèles Pronostia board (NUCLEO-F439ZI)
# Pronostia : 3 conditions opératoires (CL domain-incremental)
# Features : top-5 sélectionnées par mutual_info (voir pronostia_feature_subset.yaml)

MODEL: multi
PLATFORM: nucleo_f439zi
DATASET: pronostia
FEATURE_SUBSET: "configs/pronostia_feature_subset.yaml"

# Mahalanobis (anomaly detection — même config que board_mahalanobis.yaml)
MAHA_DIM: 5
THRESHOLD: 2.5
EMA_ALPHA: 0.05
RAM_MAHA_BYTES: 200

# EWC head (classification — même architecture que board_ewc.yaml)
EWC_IN: 5
EWC_H1: 32
EWC_H2: 16
EWC_OUT: 2
EWC_LR: 0.01
EWC_LAMBDA: 400.0
FISHER_EMA_DECAY: 0.99
RAM_EWC_BYTES: 9728

# CL scénario Pronostia
n_tasks: 3
n_samples: 600        # 200 par condition
cl_sequence: "cond1:200,cond2:200,cond3:200"

# Tâches CL
tasks:
  - id: 0
    name: cond1
    description: "1 800 rpm / 4 000 N — Bearing1_1 + Bearing1_2"
    n_samples: 200
  - id: 1
    name: cond2
    description: "1 650 rpm / 4 200 N — Bearing2_1 + Bearing2_2"
    n_samples: 200
  - id: 2
    name: cond3
    description: "1 500 rpm / 5 000 N — Bearing3_1 + Bearing3_2"
    n_samples: 200

# Gap 2 — budget mémoire (identique Monitoring, N_FEATURES=5 inchangé)
ram_budget_bytes: 65536
ram_total_estimate_bytes: 15728   # = maha(200) + ewc(9728) + tinyol(5800)
gap2_compliant: true

# Export poids C (mêmes poids que board_ewc.yaml — architecture identique)
WEIGHTS_HEADER: "firmware/stm32f4_blink/inc/model_weights.h"
WEIGHTS_CHECKPOINT: "experiments/exp_S21_03/mahalanobis_weights.pkl"
```

### Vérification

```bash
python -c "import yaml; d=yaml.safe_load(open('configs/board_pronostia.yaml')); \
    assert d['MAHA_DIM'] == 5; assert d['EWC_IN'] == 5; \
    assert d['ram_total_estimate_bytes'] < d['ram_budget_bytes']; print('config OK')"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il un `board_pronostia_mahalanobis.yaml` et un `board_pronostia_ewc.yaml` séparés (cohérence avec `board_mahalanobis.yaml` et `board_ewc.yaml`) ou le fichier multi-modèles suffit ?
- `FIXME(gap1)` : `temporal_position` (feature idx 12) est-elle disponible en temps réel ? Si non, la remplacer par `crest_factor_acc_horiz` (idx 5) après discussion Arnaud.
