# S1-03 — Implémenter `monitoring_dataset.py` (loader + split domaine)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-03 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S1-02 (dataset téléchargé, stats normalisation disponibles) |
| **Fichier cible** | `src/data/monitoring_dataset.py` |

---

## Objectif

Implémenter le loader Python du Dataset 2 (Industrial Equipment Monitoring) qui :
1. Lit le CSV brut depuis `data/raw/equipment_monitoring/`
2. Applique la normalisation Z-score (statistiques de `configs/monitoring_normalizer.yaml`)
3. Encode les variables catégorielles
4. Découpe les données en tâches CL ordonnées (Task 1 = Pump → Task 2 = Turbine → Task 3 = Compressor)
5. Retourne des `DataLoader` PyTorch prêts à l'emploi pour l'entraînement CL

**Critère de succès** : `get_cl_dataloaders()` retourne trois paires `(train_loader, val_loader)` sans fuite d'information inter-tâches, et le test `tests/test_monitoring_dataset.py` passe.

---

## Sous-tâches

### 1. Constantes de configuration

Définir en tête de fichier toutes les constantes (pas de valeurs hardcodées dans le code) :

```python
# src/data/monitoring_dataset.py

# Ordre des domaines CL — conforme à CLAUDE.md et docs/models/ewc_mlp_spec.md
DOMAIN_ORDER: list[str] = ["Pump", "Turbine", "Compressor"]

# Features numériques à normaliser
NUMERIC_FEATURES: list[str] = ["temperature", "pressure", "vibration", "humidity"]

# Feature catégorielle définissant les domaines
DOMAIN_FEATURE: str = "equipment"

# Label binaire de défaut
LABEL_COL: str = "faulty"

# Split train/val par tâche (pas de test set — évaluation CL sur toutes les tâches vues)
VAL_RATIO: float = 0.2

# Tailles mesurées (N=7672 total, source : notebooks/01_data_exploration.ipynb)
# Pump: 2534, Turbine: 2565, Compressor: 2573
```

### 2. Chargement et validation du CSV

```python
import pandas as pd
from pathlib import Path

def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Charge le CSV brut et valide les colonnes et labels attendus.

    Parameters
    ----------
    csv_path : Path
        Chemin vers equipment_anomaly_data.csv

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError : si colonnes manquantes ou labels inattendus
    """
```

Validations obligatoires :
- Présence de toutes les colonnes `NUMERIC_FEATURES + [DOMAIN_FEATURE, LABEL_COL]`
- `faulty` ∈ {0, 1} uniquement
- Les trois domaines `["Pump", "Turbine", "Compressor"]` présents dans `equipment`

### 3. Normalisation Z-score

```python
import yaml

def load_normalizer(config_path: Path) -> dict:
    """
    Charge les statistiques mean/std depuis configs/monitoring_normalizer.yaml.
    Fit sur Task 1 (Pump) uniquement — voir S1-02.
    """

def normalize_features(df: pd.DataFrame, normalizer: dict) -> pd.DataFrame:
    """
    Applique la normalisation Z-score sur NUMERIC_FEATURES.
    Les statistiques sont celles du Task 1 — appliquées à tous les domaines
    sans re-fit pour éviter la fuite d'information.
    """
```

> **Règle** : ne jamais appeler `fit` ou recalculer mean/std dans ce module. Les stats viennent
> exclusivement de `configs/monitoring_normalizer.yaml` (calculées sur T1 dans S1-02).

### 4. Encodage des features catégorielles

```python
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode la colonne `equipment` en label encoding ordinal.

    Mapping fixe (ordinal, pas one-hot) :
      Pump        → 0
      Turbine     → 1
      Compressor  → 2

    Note : on n'utilise pas one-hot ici car la colonne `equipment`
    sert AUSSI à splitter les tâches — elle est retirée des features
    d'entrée du modèle dans get_task_tensors(). L'input_dim final = 4.

    TODO(arnaud) : confirmer si `location` doit être incluse comme feature.
    """
```

### 5. Split par tâche CL

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

def get_task_split(
    df: pd.DataFrame,
    domain: str,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne les sous-DataFrames train/val pour un domaine donné.

    Le split est stratifié sur `faulty` pour conserver le taux ~10% de défaut.
    Seed fixe pour reproductibilité.
    """

def df_to_tensors(
    df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit un DataFrame en tenseurs (X, y).

    X : features NUMERIC_FEATURES uniquement, shape [N, 4], dtype float32
    y : label LABEL_COL, shape [N, 1], dtype float32

    Note : `equipment` est exclu de X (variable de split, pas feature d'entrée).
    # MEM: batch de 32 → 32×4×4 = 512 B @ FP32 / 128 B @ INT8
    """
```

### 6. Interface principale

```python
def get_cl_dataloaders(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL.

    Retourne une liste ordonnée de dicts par tâche :
    [
      {
        "task_id": 1,
        "domain": "Pump",
        "train_loader": DataLoader,
        "val_loader": DataLoader,
        "n_train": int,
        "n_val": int,
      },
      ...  # Task 2 = Turbine, Task 3 = Compressor
    ]

    L'ordre est toujours DOMAIN_ORDER = ["Pump", "Turbine", "Compressor"].
    """
```

### 7. Écrire le test

Créer `tests/test_monitoring_dataset.py` :

```python
def test_cl_dataloaders_shape():
    """Vérifie que les loaders ont les bonnes dimensions."""
    # X shape: [batch, 4], y shape: [batch, 1]

def test_no_data_leakage():
    """Vérifie que les domaines sont bien séparés entre tâches."""
    # domain i ne doit pas apparaître dans les données de task j ≠ i

def test_normalizer_applied():
    """Vérifie que les features numériques sont normalisées (mean ≈ 0 sur T1)."""

def test_label_binary():
    """Vérifie que y ∈ {0.0, 1.0} uniquement."""
```

---

## Critères d'acceptation

- [ ] `src/data/monitoring_dataset.py` importable sans erreur
- [ ] `get_cl_dataloaders()` retourne exactement 3 tâches dans l'ordre Pump → Turbine → Compressor
- [ ] `X.shape == [*, 4]` et `y.shape == [*, 1]` pour tous les loaders
- [ ] Pas de fuite d'information : aucune donnée Turbine ou Compressor dans T1
- [ ] Normalisation issue de `monitoring_normalizer.yaml` — aucun recalcul en ligne
- [ ] `tests/test_monitoring_dataset.py` — tous les tests passent via `pytest tests/ -v`
- [ ] Annotations `# MEM:` présentes sur les tenseurs principaux
- [ ] `ruff check src/data/monitoring_dataset.py` et `black --check` passent

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter |
|---------|-------------|
| `input_dim = 4` confirmé | `configs/ewc_config.yaml` → `model.input_dim` |
| Nb échantillons par domaine commenté | En-tête de `monitoring_dataset.py` |
| Interface `get_cl_dataloaders` validée | Utilisée directement dans S1-06 (`cl_trainer.py`) |

---

## Questions ouvertes

- `TODO(arnaud)` : inclure `location` comme feature d'entrée ? (augmenterait `input_dim` à 5 ou plus selon la cardinalité)
- `TODO(arnaud)` : stratifier le val split sur `faulty` (taux ~10%) — confirmer que c'est le bon choix vs split temporel
