# S2501–S2505 — Extension des loaders : mode natif RUL et multi-classe

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🔴 Critique (S2501, S2504) / 🟡 Important (S2502, S2503, S2505) |
| **Statut** | ✅ Terminé |
| **Durée estimée** | S2501 : 2h / S2502 : 2h / S2503 : 1h / S2504 : 2h / S2505 : 1h = 8h total |
| **Dépendances** | Sprint 24 ✅ — `src/data/cmapss_loader.py`, `src/data/cwru_dataset.py`, `src/data/pronostia_dataset.py`, `src/data/battery_dataset.py` tous présents et validés |
| **Fichiers cibles** | `src/data/cmapss_loader.py`, `src/data/pronostia_dataset.py`, `src/data/battery_dataset.py`, `src/data/cwru_dataset.py`, `src/data/paderborn_loader.py` |
| **Référence** | Pattern binaire existant dans chaque loader (constantes `CMAPSS_FAULTY_THRESHOLD`, `FAILURE_RATIO`, etc.), `src/training/scenarios.py` (boucle CL) |

---

## Contexte

Tous les loaders du projet retournent actuellement des labels **binaires** (0 = normal, 1 = défaut), uniformisés pour le framework CL commun. Cependant :

- **CMAPSS** et **Pronostia** : conçus pour la **régression RUL** (Remaining Useful Life), la binarisation `RUL ≤ 30` jette le signal prognostique continu.
- **CWRU** : contient **10 classes** distinctes (Normal + 3 types de défaut × 3 sévérités) ; la binarisation `normal vs tout défaut` perd la discrimination inter-classe.
- **Paderborn** : 3 états bearing distincts (sain / outer-race / inner-race) réduits à 1 bit.

Ce groupe de tâches ajoute un paramètre `mode: Literal["binary", "rul", "multiclass"]` à chaque loader, **sans casser le mode binaire existant** (valeur par défaut = `"binary"`).

**Invariant** : la signature `get_cl_dataloaders()` reste compatible — seul le type des labels change selon le mode.

---

## S2501 — `src/data/cmapss_loader.py` : mode `rul`

### Objectif

En mode `rul`, retourner le RUL continu capé (clip 0–125) au lieu de `(rul <= 30).astype(int)`. Labels : `float32`, shape `(N,)`.

### Modification de signature

```python
from typing import Literal

def get_cl_dataloaders(
    data_dir: Path = DATA_DIR_DEFAULT,
    config_path: Path = Path("configs/cmapss_config.yaml"),
    feature_names: list[str] | None = None,
    mode: Literal["binary", "rul"] = "binary",   # NOUVEAU paramètre
) -> list[dict]:
    ...
```

### Logique de labellisation à modifier

Dans la fonction interne `_make_labels()` (ou équivalent) :

```python
def _make_labels(rul_series: pd.Series, mode: str) -> np.ndarray:
    """Produit les labels selon le mode choisi."""
    rul_capped = np.clip(rul_series.values, 0, CMAPSS_RUL_CAP).astype(np.float32)
    if mode == "rul":
        return rul_capped                                     # float32, shape (N,)
    # mode == "binary" (défaut)
    return (rul_series.values <= CMAPSS_FAULTY_THRESHOLD).astype(np.float32)  # 0/1
```

### Constantes à ajouter (si absentes)

```python
CMAPSS_RUL_CAP: int = 125   # déjà présent — vérifier
```

### Vérification

```bash
python -c "
from pathlib import Path
from src.data.cmapss_loader import get_cl_dataloaders
tasks = get_cl_dataloaders(
    data_dir=Path('data/raw/CMAPSS Jet Engine Simulated Data/'),
    config_path=Path('configs/cmapss_config.yaml'),
    mode='rul',
)
for t in tasks:
    y = t['train_loader'].dataset.tensors[1]
    assert y.dtype == __import__('torch').float32, 'Labels doivent être float32'
    assert y.min() >= 0 and y.max() <= 125, f'RUL hors plage : [{y.min():.0f}, {y.max():.0f}]'
    print(f'Task {t[\"task_id\"]} — RUL range [{y.min():.0f}, {y.max():.0f}], dtype={y.dtype}')
# Mode binaire non cassé :
tasks_bin = get_cl_dataloaders(
    data_dir=Path('data/raw/CMAPSS Jet Engine Simulated Data/'),
    config_path=Path('configs/cmapss_config.yaml'),
    mode='binary',
)
y_bin = tasks_bin[0]['train_loader'].dataset.tensors[1]
assert set(y_bin.unique().tolist()).issubset({0.0, 1.0}), 'Mode binaire cassé'
print('Mode binaire OK')
"
```

---

## S2502 — `src/data/pronostia_dataset.py` : mode `rul`

### Objectif

Calculer le RUL réel de chaque fenêtre = `(n_windows_total - window_idx) * WINDOW_DURATION_S`. En mode `rul`, retourner ce RUL en secondes (float32) au lieu du label binaire dernier-10%.

### Constantes à ajouter

```python
WINDOW_DURATION_S: float = 0.1   # 2560 points @ 25 600 Hz = 0.1 s par fenêtre
```

### Logique à ajouter dans le loader

```python
def _compute_rul_labels(n_windows: int) -> np.ndarray:
    """RUL décroissant en secondes pour chaque fenêtre."""
    return np.array(
        [(n_windows - i) * WINDOW_DURATION_S for i in range(n_windows)],
        dtype=np.float32,
    )
```

### Modification de `get_pronostia_dataloaders()`

```python
def get_pronostia_dataloaders(
    npy_dir: Path = Path("data/raw/Pronostia dataset/binaries"),
    normalizer_path: Path = Path("configs/pronostia_normalizer.yaml"),
    mode: Literal["binary", "rul"] = "binary",   # NOUVEAU
) -> list[dict]:
    ...
    # Dans la boucle de traitement par bearing :
    if mode == "rul":
        y = _compute_rul_labels(n_windows)
    else:
        y = _make_binary_labels(n_windows, failure_ratio=FAILURE_RATIO)
```

### Vérification

```bash
python -c "
from pathlib import Path
from src.data.pronostia_dataset import get_pronostia_dataloaders
tasks = get_pronostia_dataloaders(
    npy_dir=Path('data/raw/Pronostia dataset/binaries'),
    mode='rul',
)
for t in tasks:
    y = t['train_loader'].dataset.tensors[1]
    assert y.dtype == __import__('torch').float32
    assert y.min() >= 0, 'RUL négatif détecté'
    print(f'Task {t[\"task_id\"]} — RUL range [{y.min():.1f}s, {y.max():.1f}s]')
"
```

---

## S2503 — `src/data/battery_dataset.py` : mode `rul`

### Objectif

Exposer la colonne RUL brute (cycles restants) sans binarisation. Labels : `float32`, shape `(N,)`.

### Modification de signature

```python
def get_cl_dataloaders(
    data_dir: Path = ...,
    config_path: Path = ...,
    mode: Literal["binary", "rul"] = "binary",   # NOUVEAU
) -> list[dict]:
```

### Logique de labellisation

```python
if mode == "rul":
    y = df["rul_remaining"].values.astype(np.float32)
else:
    y = (df["rul_remaining"] < BATTERY_RUL_THRESHOLD).astype(np.float32)
```

### Vérification

```bash
python -c "
from src.data.battery_dataset import get_cl_dataloaders
tasks = get_cl_dataloaders(mode='rul')
print(f'N tâches : {len(tasks)}')
y = tasks[0]['train_loader'].dataset.tensors[1]
print(f'RUL dtype={y.dtype}, range=[{y.min():.0f}, {y.max():.0f}]')
"
```

---

## S2504 — `src/data/cwru_dataset.py` : mode `multiclass`

### Objectif

Retourner des labels 0–9 correspondant aux 10 classes CWRU (Normal=0, Ball_007=1, Ball_014=2, Ball_021=3, IR_007=4, IR_014=5, IR_021=6, OR_007=7, OR_014=8, OR_021=9) via `LabelEncoder`.

**Note** : le scénario CL multi-classe utilise le découpage `by_fault_type` (3 tâches), avec les 10 classes présentes au total mais distribuées par tâche (chaque tâche voit Normal + son type de défaut).

### Mapping de classes à ajouter

```python
# Mapping global stable (ordre alphabétique via LabelEncoder ou fixé ici)
MULTICLASS_LABEL_MAP: dict[str, int] = {
    "Normal_1":   0,
    "Ball_007_1": 1,
    "Ball_014_1": 2,
    "Ball_021_1": 3,
    "IR_007_1":   4,
    "IR_014_1":   5,
    "IR_021_1":   6,
    "OR_007_6_1": 7,
    "OR_014_6_1": 8,
    "OR_021_6_1": 9,
}
N_CLASSES_MULTICLASS: int = 10
```

### Modification de `get_cl_splits()` ou équivalent

```python
def get_cl_splits(
    scenario: Literal["by_fault_type", "by_severity"] = "by_fault_type",
    mode: Literal["binary", "multiclass"] = "binary",   # NOUVEAU
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict]:
    ...
    # Dans la boucle de tâches :
    if mode == "multiclass":
        y = df[FAULT_COL].map(MULTICLASS_LABEL_MAP).values.astype(np.int64)
    else:
        y = (df[FAULT_COL] != NORMAL_LABEL).astype(np.float32)
```

### Vérification

```bash
python -c "
from src.data.cwru_dataset import get_cl_splits
tasks = get_cl_splits(scenario='by_fault_type', mode='multiclass')
import numpy as np
all_classes = set()
for t in tasks:
    classes = set(np.unique(t['y_train']))
    all_classes |= classes
    print(f'Task {t[\"task_id\"]} — classes : {sorted(classes)}, shape : {t[\"X_train\"].shape}')
print(f'Classes totales observées : {sorted(all_classes)} (attendu : sous-ensemble de 0-9)')
# Mode binaire non cassé
tasks_bin = get_cl_splits(mode='binary')
y_bin = tasks_bin[0]['y_train']
assert set(y_bin).issubset({0.0, 1.0}), 'Mode binaire CWRU cassé'
print('Mode binaire OK')
"
```

---

## S2505 — `src/data/paderborn_loader.py` : mode `multiclass`

### Objectif

Retourner des labels 0/1/2 correspondant aux 3 états bearing Paderborn : `K001` (sain) = 0, `KA04` (outer-race damage) = 1, `KI04` (inner-race damage) = 2.

### Mapping à ajouter

```python
PADERBORN_MULTICLASS_MAP: dict[str, int] = {
    "K001": 0,   # sain (healthy)
    "KA04": 1,   # outer-race fault
    "KI04": 2,   # inner-race fault
}
N_CLASSES_MULTICLASS: int = 3
```

### Modification de signature

```python
def get_cl_dataloaders(
    data_dir: Path = ...,
    config_path: Path = ...,
    mode: Literal["binary", "multiclass"] = "binary",   # NOUVEAU
) -> list[dict]:
    ...
    if mode == "multiclass":
        y = df["bearing_id"].map(PADERBORN_MULTICLASS_MAP).values.astype(np.int64)
    else:
        y = (df["bearing_id"] != "K001").astype(np.float32)
```

### Vérification

```bash
python -c "
from src.data.paderborn_loader import get_cl_dataloaders
tasks = get_cl_dataloaders(mode='multiclass')
for t in tasks:
    y = t['train_loader'].dataset.tensors[1]
    assert y.dtype == __import__('torch').int64
    print(f'Task {t[\"task_id\"]} — classes : {y.unique().tolist()}')
"
```

---

## Vérification end-to-end

```bash
# Tous les loaders en mode natif
python -c "
from pathlib import Path
from src.data.cmapss_loader import get_cl_dataloaders as cmapss_load
from src.data.pronostia_dataset import get_pronostia_dataloaders as pronostia_load
from src.data.cwru_dataset import get_cl_splits as cwru_load
from src.data.paderborn_loader import get_cl_dataloaders as paderborn_load
import torch

# CMAPSS RUL
t = cmapss_load(data_dir=Path('data/raw/CMAPSS Jet Engine Simulated Data/'),
                config_path=Path('configs/cmapss_config.yaml'), mode='rul')
assert t[0]['train_loader'].dataset.tensors[1].dtype == torch.float32

# CWRU multiclass
t = cwru_load(mode='multiclass')
assert t[0]['y_train'].dtype.kind == 'i'

# Régressions non cassées
t = cmapss_load(data_dir=Path('data/raw/CMAPSS Jet Engine Simulated Data/'),
                config_path=Path('configs/cmapss_config.yaml'), mode='binary')
y = t[0]['train_loader'].dataset.tensors[1]
assert set(y.unique().tolist()).issubset({0.0, 1.0})

print('Tous les loaders OK ✅')
"

# Zéro régression pytest
pytest tests/ -v -k "not ewc_regression and not ewc_multiclass and not hdc_regressor"
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2501 — `cmapss_loader.py` mode `rul` | ✅ | `_make_labels()` helper, no-stratify en mode rul, `class_weights=None` |
| S2502 — `pronostia_dataset.py` mode `rul` | ✅ | `WINDOW_DURATION_S`, `_compute_rul_labels()`, `continuous_labels` flag dans `_features_to_dataloader` |
| S2503 — `battery_dataset.py` mode `rul` | ✅ | RUL depuis `RUL_COL`, no-stratify en mode rul |
| S2504 — `cwru_dataset.py` mode `multiclass` | ✅ | `MULTICLASS_LABEL_MAP`, nouvelle fonction `get_cl_splits()` retournant numpy arrays |
| S2505 — `paderborn_loader.py` mode `multiclass` | ✅ | `PADERBORN_MULTICLASS_MAP`, labels int64 shape `[N]` en multiclass |

---

## Questions ouvertes

- `TODO(arnaud)` : Pour CWRU multi-classe, chaque tâche CL doit-elle voir uniquement les classes de son type de défaut (Ball / Inner / Outer), ou toutes les 10 classes dès la tâche 1 ?
- `TODO(fred)` : Les cas d'usage Edge Spectrum visent-ils plutôt RUL continu (prognostic) ou détection de seuil (diagnostic) ? Cela conditionne la priorité exp_S25_01 (RUL CMAPSS) vs exp_S25_03 (multiclass CWRU).
- `FIXME(gap1)` : CMAPSS est un dataset simulé (NASA turbofan) — vérifier si cela reste compatible avec la revendication Gap 1 "données industrielles réelles" ou s'il faut préciser "industriellement représentatif" dans le manuscrit.
- `TODO(dorra)` : La normalisation MinMax fixée sur FD001 (mode binaire) est-elle correcte en mode `rul` ? Le cap à 125 cycles suffit-il à stabiliser la distribution ?
