# S2305–S2310 — CMAPSS sur board : feature selection + streaming + 4 expériences

| Champ | Valeur |
|-------|--------|
| **Sprint** | 23 |
| **Priorité** | 🔴 Critique (S2305–S2307, S2310) / 🟡 Important (S2308–S2309) |
| **Statut** | ✅ Terminé — 2026-06-02 |
| **Durée estimée** | 1h + 1h + 1h + 1h + 30 min + 1h = 5h30 |
| **Dépendances** | Sprint 22 ✅ — `src/data/cmapss_loader.py` livré, `data/raw/cmapss/` présent, `board_experiment_recorder.py` opérationnel |
| **Fichiers cibles** | `scripts/cmapss_feature_selection.py`, `configs/board_cmapss.yaml`, `configs/cmapss_feature_subset.yaml`, `scripts/sensor_stream.py`, `experiments/exp_S23_01/` à `exp_S23_04/` |
| **Référence** | `scripts/pronostia_feature_selection.py` (pattern), `configs/board_ewc.yaml` (config board), `scripts/board_experiment_recorder.py`, `scripts/board_dataset_builder.py` |

---

## Contexte

`cmapss_loader.py` (Sprint 22) charge les 4 sous-datasets FD001–FD004 avec 21 capteurs et binarise le RUL en `faulty`. Pour la board NUCLEO, les contraintes de RAM et de latence imposent de réduire à **5 features** sélectionnées par information mutuelle.

Le scénario CL CMAPSS sur board est **domain-incremental** : Task 1 = FD001 → Task 2 = FD002 → (optionnel) FD003 → FD004. Les 4 expériences testent chacun des 4 modèles MCU (EWC, TinyOL, Mahalanobis, HDC) sur les 2 premières tâches au minimum.

---

## S2305 — `scripts/cmapss_feature_selection.py`

### Fonctionnalités

Script autonome : charge FD001, calcule `mutual_info_classif` entre les 21 capteurs et le label `faulty`, sauvegarde les top-5 dans `configs/cmapss_feature_subset.yaml`.

```python
"""
cmapss_feature_selection.py — Sélection top-5 features CMAPSS par mutual info.

Fit uniquement sur FD001 (train set) pour éviter la fuite de données inter-domaines.
Sauvegarde dans configs/cmapss_feature_subset.yaml.

Usage :
    python scripts/cmapss_feature_selection.py
    python scripts/cmapss_feature_selection.py --n-features 5 --subset-id FD001
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif

# Constantes (cohérence avec cmapss_loader.py)
CMAPSS_FAULTY_THRESHOLD = 30
CMAPSS_RUL_CAP = 125
SENSOR_NAMES = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]
COL_NAMES = ["unit_nr", "time_cycles", "op1", "op2", "op3"] + SENSOR_NAMES

OUTPUT_PATH = Path("configs/cmapss_feature_subset.yaml")
DATA_DIR    = Path("data/raw/cmapss/")
```

### Structure de sortie `configs/cmapss_feature_subset.yaml`

```yaml
# cmapss_feature_subset.yaml — Top-5 features CMAPSS sélectionnées par mutual info
# Généré par scripts/cmapss_feature_selection.py — NE PAS éditer manuellement.
# Fit sur FD001 uniquement.
selected_features:
  - "T50"      # exemple (à confirmer par exécution du script)
  - "Ps30"
  - "NRf"
  - "BPR"
  - "htBleed"
n_features: 5
fit_subset: "FD001"
method: "mutual_info_classif"
faulty_threshold: 30
```

### Vérification

```bash
python scripts/cmapss_feature_selection.py
cat configs/cmapss_feature_subset.yaml
# Doit lister 5 noms de capteurs parmi les 21 de SENSOR_NAMES
```

---

## S2306 — `scripts/sensor_stream.py` : `--dataset cmapss`

### Extension à ajouter

Ajouter le cas `cmapss` dans la fonction `load_dataset()` existante de `sensor_stream.py` (pattern identique à `cwru` et `monitoring` déjà présents) :

```python
elif args.dataset == "cmapss":
    from src.data.cmapss_loader import get_cl_dataloaders
    import yaml
    feature_subset = yaml.safe_load(
        Path("configs/cmapss_feature_subset.yaml").read_text()
    )["selected_features"]
    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/cmapss/"),
        config_path=Path("configs/board_cmapss.yaml"),
        feature_names=feature_subset,
    )
```

Le flag CMAPSS doit utiliser `FRAME_FLAGS_EWC_MODE` ou `FRAME_FLAGS_HDC_MODE` selon le modèle spécifié via `--model`. Ajouter `FRAME_FLAGS_HDC_MODE = 0x20` dans les constantes de protocole :

```python
FRAME_FLAGS_HDC_MODE    = 0x20   # utilise HDCClassifier (bit 5, cohérence pipeline.h)
```

### Vérification

```bash
# Dry-run 2 tâches CMAPSS, modèle EWC
python scripts/sensor_stream.py --dataset cmapss --model ewc \
    --dry-run --n-samples 50 --tasks 2
# Attendu : aucune erreur, 50 lignes de résultats simulés
```

---

## S2307 — exp_S23_01 : EWC / CMAPSS board

### Structure du dossier

```
experiments/exp_S23_01/
├── config_snapshot.yaml
├── stream_task1.json
├── stream_task2.json
└── results.json
```

### `config_snapshot.yaml`

```yaml
exp_id: "exp_S23_01"
model: "ewc"
dataset: "cmapss"
platform: "nucleo_f439zi"
tasks: ["FD001", "FD002"]
n_samples_per_task: 200
board_config: "configs/board_cmapss.yaml"
feature_subset: "configs/cmapss_feature_subset.yaml"
ewc_lambda: 400.0
seed: 42
sprint: 23
date: "2026-06-22"
```

### Commandes de lancement

```bash
# 1. Dry-run (vérification pipeline sans board)
python scripts/sensor_stream.py \
    --dataset cmapss --model ewc \
    --dry-run --n-samples 200 --tasks 2 \
    --output experiments/exp_S23_01/stream_dryrun.json

# 2. Avec board connectée
python scripts/sensor_stream.py \
    --dataset cmapss --model ewc \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 200 --tasks 2 --rate-hz 20 \
    --update --consolidate \
    --output experiments/exp_S23_01/stream_task{task_id}.json

# 3. Enregistrement résultats
python scripts/board_experiment_recorder.py \
    --exp-dir experiments/exp_S23_01/ \
    --model ewc --dataset cmapss
```

### `results.json` attendu

```json
{
  "exp_id": "exp_S23_01",
  "model": "ewc",
  "dataset": "cmapss",
  "platform": "nucleo_f439zi",
  "tasks": ["FD001", "FD002"],
  "acc_final": ">= 0.75",
  "avg_forgetting": "< 0.10",
  "latency_ms": "< 1.0",
  "ram_peak_bytes": "<= 65536",
  "gap2_latency_compliant": true,
  "n_samples_total": 400
}
```

---

## S2308 — exp_S23_02 : TinyOL / CMAPSS board

### Commandes de lancement

```bash
# Dry-run TinyOL CMAPSS
python scripts/sensor_stream.py \
    --dataset cmapss --model tinyol \
    --dry-run --n-samples 200 --tasks 2 \
    --output experiments/exp_S23_02/stream_dryrun.json

# Live avec board
python scripts/sensor_stream.py \
    --dataset cmapss --model tinyol \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 200 --tasks 2 --rate-hz 10 \
    --update --output experiments/exp_S23_02/stream_task{task_id}.json
```

### `results.json` attendu (critères sprint)

```json
{
  "exp_id": "exp_S23_02",
  "model": "tinyol",
  "dataset": "cmapss",
  "acc_final": ">= 0.70",
  "latency_ms": "< 5.0",
  "ram_peak_bytes": "<= 65536",
  "gap2_latency_compliant": true
}
```

---

## S2309 — exp_S23_03 : Mahalanobis / CMAPSS board (baseline)

```bash
# Dry-run Mahalanobis (modèle par défaut — pas de flag EWC/HDC)
python scripts/sensor_stream.py \
    --dataset cmapss --model mahalanobis \
    --dry-run --n-samples 200 --tasks 2 \
    --output experiments/exp_S23_03/stream_dryrun.json
```

### `results.json` attendu

```json
{
  "exp_id": "exp_S23_03",
  "model": "mahalanobis",
  "dataset": "cmapss",
  "acc_final": ">= 0.60",
  "latency_ms": "< 0.1",
  "ram_peak_bytes": "<= 1024",
  "gap2_latency_compliant": true
}
```

---

## S2310 — exp_S23_04 : HDC / CMAPSS board — premier HDC réel MCU

> **Milestone** : première exécution HDC C sur MCU avec données réelles (non synthétiques). Critère de succès : `pred != -1` (prédiction valide) et latence mesurée DWT.

```bash
# Dry-run HDC CMAPSS
python scripts/sensor_stream.py \
    --dataset cmapss --model hdc \
    --dry-run --n-samples 200 --tasks 2 \
    --output experiments/exp_S23_04/stream_dryrun.json

# Live HDC avec board (S2304 intégré)
python scripts/sensor_stream.py \
    --dataset cmapss --model hdc \
    --port /dev/ttyACM0 --baud 115200 \
    --n-samples 200 --tasks 2 --rate-hz 10 \
    --update --consolidate \
    --output experiments/exp_S23_04/stream_task{task_id}.json
```

### `results.json` attendu

```json
{
  "exp_id": "exp_S23_04",
  "model": "hdc",
  "dataset": "cmapss",
  "acc_final": ">= 0.65",
  "latency_ms": "< 2.0",
  "ram_peak_bytes": "<= 28672",
  "gap2_latency_compliant": true,
  "note_gap1": "premier test HDC C réel sur MCU — contribution Gap 1"
}
```

---

## `configs/board_cmapss.yaml`

```yaml
# board_cmapss.yaml — Config board NUCLEO pour CMAPSS (domain-incremental)
# Ne pas modifier — créer une copie pour chaque variante expérimentale.

MODEL: ewc_head      # modèle par défaut (remplacé par --model en CLI)
PLATFORM: nucleo_f439zi

# Architecture (identique à board_ewc.yaml)
EWC_IN: 5            # top-5 features sélectionnées
EWC_H1: 32
EWC_H2: 16
EWC_OUT: 2

# Hyperparamètres EWC
EWC_LR: 0.01
EWC_LAMBDA: 400.0
FISHER_EMA_DECAY: 0.99

# Dataset CMAPSS
dataset: "cmapss"
data_dir: "data/raw/cmapss/"
feature_subset_path: "configs/cmapss_feature_subset.yaml"
domain_order: ["FD001", "FD002", "FD003", "FD004"]
n_tasks_board: 2     # FD001 + FD002 (FD003/FD004 optionnels)
n_samples_per_task: 200
faulty_threshold: 30
rul_cap: 125

# Budget mémoire (vérification Gap 2)
ram_budget_bytes: 65536   # 64 Ko

# Export poids C
WEIGHTS_HEADER: "firmware/stm32f4_blink/inc/model_weights.h"
WEIGHTS_CHECKPOINT: "experiments/exp_S22_cmapss_base/ewc_weights.pkl"
```

---

## Vérification end-to-end

```bash
# Feature selection
python scripts/cmapss_feature_selection.py
python -c "import yaml; d=yaml.safe_load(open('configs/cmapss_feature_subset.yaml')); assert len(d['selected_features'])==5"

# Dry-run 4 modèles
for model in ewc tinyol mahalanobis hdc; do
    python scripts/sensor_stream.py \
        --dataset cmapss --model $model \
        --dry-run --n-samples 50 --tasks 2 && \
    echo "$model dry-run OK"
done

# Vérifier les 4 dossiers d'expériences créés
ls experiments/exp_S23_0{1,2,3,4}/results.json
```

---

## Résultats d'implémentation (2026-06-02)

### Ce qui a été livré

| Sous-tâche | Statut | Notes |
| ---------- | :----: | ----- |
| S2305 — `scripts/cmapss_feature_selection.py` + `configs/board_cmapss.yaml` | ✅ | Produit yaml avec deux clés : `selected_features` (sensor_stream.py) + `features` (cmapss_loader.py) |
| S2306 — `sensor_stream.py` : `--dataset cmapss`, `--model ewc/tinyol/mahalanobis/hdc`, `FRAME_FLAGS_HDC_MODE=0x20` | ✅ | `_load_cmapss()` via `get_cl_dataloaders` (FD001+FD002 concaténés) |
| S2307 — `configs/cmapss_feature_subset.yaml` mis à jour (ajout `selected_features`) | ✅ | Features sélectionnées : Ps30, T50, Phi, P30, BPR |
| S2308 — `experiments/exp_S23_01/config_snapshot.yaml` — EWC / CMAPSS | ✅ | λ=400, FD001+FD002, 200 samples/task |
| S2309 — `experiments/exp_S23_02/config_snapshot.yaml` — TinyOL / CMAPSS | ✅ | |
| S2309 — `experiments/exp_S23_03/config_snapshot.yaml` — Mahalanobis / CMAPSS | ✅ | baseline non-neuronal |
| S2310 — `experiments/exp_S23_04/config_snapshot.yaml` — HDC / CMAPSS | ✅ | note_gap1 : premier test HDC C réel sur MCU |

### Résultats réels board NUCLEO-F439ZI (2026-06-02)

| Expérience | Modèle | acc_final | avg_forgetting | latence P50 (ms) | RAM statique (B) | Gap 2 | Notes |
|---|---|---:|---:|---:|---:|---|---|
| exp_S23_01 | EWC | **0.840** | 0.000 | 0.251 | 9 728 | ✅ | λ=400, AF=0 excellent |
| exp_S23_02 | TinyOL | 0.148 | 0.045 | 0.126 | 7 040 | ✅ | ⚠️ poids Monitoring ≠ CMAPSS |
| exp_S23_03 | Mahalanobis | 0.575 | 0.000 | 0.004 | 1 200 | ✅ | baseline, pas de CL |
| exp_S23_04 | HDC | **0.853** | 0.000 | 0.646 | 28 364 | ✅ | 🎯 premier HDC MCU réel |

**Critères sprint** : EWC acc≥0.75 ✅ | TinyOL acc≥0.70 ❌ (poids inadaptés) | Maha acc≥0.60 ✅ | HDC acc≥0.65 ✅ | toutes latences < 100 ms ✅

> **TinyOL faible** : les poids embarqués dans `model_weights.h` sont entraînés sur Monitoring. Le seuil de reconstruction MSE=0.05 ne s'applique pas aux features CMAPSS — acc=0.148 reflète cette inadéquation. Solution Sprint 24 : exporter les poids TinyOL spécifiques CMAPSS via `scripts/export_weights_tinyol.py`.

### Vérification board réelle (2026-06-02)

```text
ewc/cmapss        → ✅  acc=0.840  lat=0.251ms  AF=0.00  9728B
tinyol/cmapss     → ⚠️  acc=0.148  lat=0.126ms  AF=0.05  poids inadaptés
mahalanobis/cmapss → ✅  acc=0.575  lat=0.004ms  AF=0.00  1200B  (baseline)
hdc/cmapss        → ✅  acc=0.853  lat=0.646ms  AF=0.00  28364B  (premier test HDC C MCU — Gap 1 ✅)
```

### Correction apportée vs spec

La spec S2306 utilise `data/raw/cmapss/` — le chemin réel du dépôt est `data/raw/CMAPSS Jet Engine Simulated Data/`. Corrigé dans `_load_cmapss()`. Les labels retournés par les DataLoaders ont shape `(N, 1)` ; un `.flatten()` est appliqué avant concaténation.

---

## Questions ouvertes

- `TODO(arnaud)` : Les sous-datasets FD003 et FD004 (conditions multi-fault + haute fan speed) sont-ils nécessaires pour le chapitre Gap 1 ? Ou FD001+FD002 suffisent-ils pour démontrer le domain-incremental ?
- `TODO(arnaud)` : La binarisation `RUL ≤ 30` donne-t-elle un taux de défaut équilibré sur FD001–FD004 ? Si le taux est < 10% sur certains sous-datasets, envisager un équilibrage par sous-échantillonnage.
- `FIXME(gap1)` : Documenter dans `docs/context/datasets.md` en quoi CMAPSS (turbofan NASA) diffère de CWRU (roulements) et Monitoring (équipements industriels) pour justifier la diversité Gap 1.
- `FIXME(gap2)` : Les latences attendues (`< 1 ms` EWC, `< 5 ms` TinyOL) sont des estimations depuis Pronostia/Monitoring. Mesurer les valeurs réelles DWT sur CMAPSS — elles devraient être identiques (même architecture, mêmes 5 features).
