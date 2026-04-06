# S3-02 — Implémenter `pump_dataset.py` (fenêtrage + features)

| Champ | Valeur |
|-------|--------|
| **ID** | S3-02 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S3-01 (dataset téléchargé, stats normalisation calculées) |
| **Fichier cible** | `src/data/pump_dataset.py` |
| **Complété le** | — |

---

## Objectif

Implémenter le loader Python du Dataset 1 (Large Industrial Pump Maintenance) qui :
1. Lit le CSV brut depuis `data/raw/pump_maintenance/`, parse le timestamp, valide les colonnes
2. Applique un fenêtrage glissant (WINDOW_SIZE=32, STEP_SIZE=16) et extrait 25 features statistiques par fenêtre
3. Découpe les données en 3 tâches CL selon l'ordre chronologique (task-free, sans mélange)
4. Normalise les features par Z-score (statistiques fit sur Task 1 uniquement, depuis `configs/pump_normalizer.yaml`)
5. Retourne les tenseurs PyTorch prêts à l'emploi pour l'entraînement TinyOL

**Critère de succès** : `from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter` passe, l'interface décrite dans `docs/context/datasets.md` est respectée, et `pytest tests/test_pump_dataset.py -v` passe intégralement.

---

## Sous-tâches

### 1. Constantes de configuration

Définir en tête de fichier toutes les constantes (aucune valeur hardcodée dans le code) :

```python
# src/data/pump_dataset.py

# Conforme à configs/tinyol_config.yaml et docs/context/datasets.md
WINDOW_SIZE: int = 32        # points temporels par fenêtre
STEP_SIZE: int = 16          # chevauchement 50%
N_FEATURES: int = 25         # 6 features × 4 canaux + 1 feature globale (label temporel)
N_TASKS: int = 3             # découpage chronologique

FEATURE_COLUMNS: list[str] = ["temperature", "vibration", "pressure", "rpm"]
LABEL_COLUMN: str = "maintenance_required"
TIMESTAMP_COLUMN: str = "timestamp"

FEATURES_PER_CHANNEL: list[str] = ["mean", "std", "rms", "kurtosis", "peak", "crest_factor"]
# 6 features × 4 canaux = 24 + 1 (label temporel normalisé) = 25 — conforme tinyol_spec.md §3.2

# Tailles mesurées — à mettre à jour après S3-01
# DATASET_TOTAL_N: int = ???  # TODO(arnaud) après exploration
```

### 2. Classe `PumpMaintenanceDataset`

```python
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis


class PumpMaintenanceDataset:
    """
    Loader et feature extractor pour le Dataset 1 — Large Industrial Pump Maintenance.

    Charge le CSV brut, valide les colonnes, et extrait les features statistiques
    par fenêtre glissante selon la spécification tinyol_spec.md §3.

    Parameters
    ----------
    data_dir : str | Path
        Chemin vers data/raw/pump_maintenance/ contenant le CSV brut.

    Notes
    -----
    Empreinte mémoire fenêtre brute : WINDOW_SIZE × N_CHANNELS × 4 B = 32×4×4 = 512 B @ FP32
    Empreinte mémoire features : N_FEATURES × 4 B = 25 × 4 = 100 B @ FP32
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        Charge et valide le CSV brut.

        Returns
        -------
        pd.DataFrame trié chronologiquement.

        Raises
        ------
        FileNotFoundError
            Si aucun CSV n'est trouvé dans data_dir.
        ValueError
            Si les colonnes attendues sont absentes ou le label est invalide.
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Aucun CSV dans {self.data_dir}")

        df = pd.read_csv(csv_files[0], parse_dates=[TIMESTAMP_COLUMN])

        # Validation des colonnes
        expected = FEATURE_COLUMNS + [LABEL_COLUMN, TIMESTAMP_COLUMN]
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        # Validation du label
        if not df[LABEL_COLUMN].isin([0, 1]).all():
            raise ValueError(f"Label '{LABEL_COLUMN}' doit être 0/1 uniquement")

        # Tri chronologique obligatoire (scénario Domain-Incremental temporel)
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        self._df = df
        return df

    def extract_features(
        self,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrait les features statistiques par fenêtre glissante.

        Parameters
        ----------
        window_size : int
            Nombre de points temporels par fenêtre (défaut : 32).
        step_size : int
            Pas de la fenêtre glissante (défaut : 16, chevauchement 50%).

        Returns
        -------
        features : np.ndarray, shape [N_windows, N_FEATURES], dtype float32
            # MEM: N_windows × 25 × 4 B @ FP32
        labels : np.ndarray, shape [N_windows], dtype float32
            Label majoritaire de la fenêtre (vote majoritaire sur maintenance_required).
            # MEM: N_windows × 4 B @ FP32
        """
        if self._df is None:
            self.load()
        return _extract_window_features(self._df, window_size, step_size)
```

### 3. Extraction des features par fenêtre

```python
def _extract_window_features(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fenêtrage glissant + extraction des 25 features statistiques.

    Features extraites (conforme tinyol_spec.md §3.2) :
    Pour chaque canal (temperature, vibration, pressure, rpm) :
      → mean, std, rms, kurtosis, peak, crest_factor  (6 features × 4 canaux = 24)
    + 1 feature globale : label temporel normalisé (position de la fenêtre dans [0, 1])
    = 25 features total
    """
    raw = df[FEATURE_COLUMNS].values.astype(np.float32)  # [N, 4]
    labels_raw = df[LABEL_COLUMN].values.astype(np.float32)
    n_total = len(raw)

    all_features = []
    all_labels = []

    for start in range(0, n_total - window_size + 1, step_size):
        window = raw[start : start + window_size]  # [window_size, 4]
        # MEM: fenêtre brute = 32 × 4 × 4 B = 512 B @ FP32 / 32 × 4 B = 128 B @ INT8

        feats = _compute_channel_features(window)  # 24 features

        # Feature globale : position temporelle normalisée ∈ [0, 1]
        temporal_pos = np.float32(start / max(n_total - window_size, 1))
        feats = np.append(feats, temporal_pos)  # → 25 features
        # MEM: vecteur features = 25 × 4 B = 100 B @ FP32

        # Label de la fenêtre : vote majoritaire
        window_label = np.float32(labels_raw[start : start + window_size].mean() >= 0.5)

        all_features.append(feats)
        all_labels.append(window_label)

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.float32)


def _compute_channel_features(window: np.ndarray) -> np.ndarray:
    """
    Calcule les 6 features statistiques pour chaque canal.

    Parameters
    ----------
    window : np.ndarray, shape [window_size, n_channels]

    Returns
    -------
    np.ndarray, shape [n_channels × 6] = [24]

    Features (conforme FEATURES_PER_CHANNEL) :
      mean, std, rms, kurtosis, peak, crest_factor
    """
    n_channels = window.shape[1]
    feats = []
    for c in range(n_channels):
        x = window[:, c]
        mean = np.mean(x)
        std = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        kurt = float(scipy_kurtosis(x, fisher=True))   # kurtosis excess (Fisher)
        peak = float(np.max(np.abs(x)))
        crest = peak / (rms + 1e-8)                    # éviter division par zéro
        feats.extend([mean, std, rms, kurt, peak, crest])
    return np.array(feats, dtype=np.float32)
```

### 4. Classe `CLStreamSplitter`

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
import yaml


class CLStreamSplitter:
    """
    Découpe le stream de features en N tâches CL selon l'ordre chronologique.

    Référence : docs/context/datasets.md — interface API PumpMaintenanceDataset.

    Parameters
    ----------
    features : np.ndarray, shape [N_windows, N_FEATURES]
    labels : np.ndarray, shape [N_windows]
    n_tasks : int
        Nombre de tâches CL (défaut : 3, découpage 33%/33%/33%).
    strategy : str
        Stratégie de découpage. Seul "chronological" est supporté (obligatoire MCU).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_tasks: int = N_TASKS,
        strategy: str = "chronological",
    ) -> None:
        if strategy != "chronological":
            raise ValueError(
                f"Seul strategy='chronological' est supporté (reçu : '{strategy}'). "
                "Le mélange inter-tâche invaliderait le scénario Domain-Incremental temporel."
            )
        self.features = features
        self.labels = labels
        self.n_tasks = n_tasks
        self._normalizer: dict | None = None
        self._task_slices = self._compute_slices()

    def _compute_slices(self) -> list[tuple[int, int]]:
        """Découpage chronologique égal en n_tasks tranches."""
        n = len(self.features)
        size = n // self.n_tasks
        slices = []
        for i in range(self.n_tasks):
            start = i * size
            end = (i + 1) * size if i < self.n_tasks - 1 else n
            slices.append((start, end))
        return slices

    def fit_normalizer(self, task_id: int = 0) -> dict:
        """
        Calcule les statistiques de normalisation Z-score sur la tâche task_id.

        IMPORTANT : toujours fit sur task_id=0 (T1 = état sain) pour éviter
        la fuite d'information des tâches futures. Conforme tinyol_spec.md §3.3.

        Parameters
        ----------
        task_id : int
            Index de la tâche de référence (0 = T1).

        Returns
        -------
        dict avec les clés 'mean' et 'std', shape [N_FEATURES].
        """
        start, end = self._task_slices[task_id]
        task_features = self.features[start:end]
        normalizer = {
            "mean": task_features.mean(axis=0).tolist(),
            "std": task_features.std(axis=0).tolist(),
            "fit_on": f"task_{task_id}",
            "n_samples": int(end - start),
        }
        self._normalizer = normalizer
        return normalizer

    def apply_normalizer(self, normalizer: dict) -> None:
        """
        Applique la normalisation Z-score sur l'ensemble du stream.

        Les statistiques viennent exclusivement du normalizer fourni — aucun recalcul.
        """
        mean = np.array(normalizer["mean"], dtype=np.float32)
        std = np.array(normalizer["std"], dtype=np.float32)
        self.features = (self.features - mean) / (std + 1e-8)
        self._normalizer = normalizer

    def save_normalizer(self, path: str | Path) -> None:
        """Sauvegarde le normalizer dans un fichier YAML (committable, portage MCU)."""
        if self._normalizer is None:
            raise RuntimeError("fit_normalizer() doit être appelé avant save_normalizer()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._normalizer, f, default_flow_style=False)
        print(f"[PumpDataset] Normalizer sauvegardé → {path}")

    def get_task_tensors(
        self,
        task_id: int,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """
        Retourne les DataLoaders PyTorch pour une tâche donnée.

        Parameters
        ----------
        task_id : int
            Index de la tâche (0-based).
        batch_size : int
        val_ratio : float
            Fraction de validation (sur la tâche courante uniquement).
        seed : int

        Returns
        -------
        dict avec les clés :
          'task_id', 'train_loader', 'val_loader', 'n_train', 'n_val'
        """
        start, end = self._task_slices[task_id]
        X = torch.tensor(self.features[start:end], dtype=torch.float32)
        y = torch.tensor(self.labels[start:end], dtype=torch.float32).unsqueeze(1)
        # MEM: X shape [N_task, 25], y shape [N_task, 1] — taille dépend du split

        n = len(X)
        n_val = max(1, int(n * val_ratio))
        # Pas de mélange — on conserve l'ordre chronologique pour le val set
        X_train, X_val = X[: n - n_val], X[n - n_val :]
        y_train, y_val = y[: n - n_val], y[n - n_val :]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False,
        )

        return {
            "task_id": task_id,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }
```

### 5. Interface principale (conforme à `docs/context/datasets.md`)

```python
# Usage recommandé — conforme à docs/context/datasets.md

from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter

# 1. Chargement brut
dataset = PumpMaintenanceDataset("data/raw/pump_maintenance/")

# 2. Feature engineering (fenêtres + features statistiques)
features, labels = dataset.extract_features(window_size=32, step_size=16)

# 3. Split en stream CL (ordre chronologique obligatoire)
cl_stream = CLStreamSplitter(features, labels, n_tasks=3, strategy="chronological")

# 4. Normalisation (calculée sur Task 1 uniquement)
normalizer = cl_stream.fit_normalizer(task_id=0)  # IMPORTANT : fit sur T1 seulement
cl_stream.apply_normalizer(normalizer)
cl_stream.save_normalizer("configs/pump_normalizer.yaml")  # committable

# 5. Récupérer les DataLoaders par tâche
for task_id in range(3):
    task_data = cl_stream.get_task_tensors(task_id=task_id, batch_size=32)
    print(f"Task {task_id} : {task_data['n_train']} train, {task_data['n_val']} val")
```

### 6. Écrire les tests

Créer `tests/test_pump_dataset.py` :

```python
import numpy as np
import pytest
from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter, N_FEATURES, N_TASKS

# Ces tests nécessitent des données en data/raw/pump_maintenance/
# Pour les tests unitaires sans données, utiliser des fixtures synthétiques

@pytest.fixture
def synthetic_features():
    """Génère des features synthétiques pour les tests sans données réelles."""
    rng = np.random.default_rng(42)
    n = 300
    features = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    labels = (rng.random(n) > 0.9).astype(np.float32)
    return features, labels


def test_window_shape(synthetic_features):
    """Vérifie que les features ont la bonne dimensionnalité."""
    features, labels = synthetic_features
    assert features.shape[1] == N_FEATURES, f"Attendu {N_FEATURES} features, obtenu {features.shape[1]}"
    assert labels.shape[0] == features.shape[0]


def test_chronological_split(synthetic_features):
    """Vérifie que le split chronologique ne mélange pas les tâches."""
    features, labels = synthetic_features
    splitter = CLStreamSplitter(features, labels, n_tasks=N_TASKS)
    slices = splitter._task_slices

    # Les tâches doivent être contiguës et non-chevauchantes
    for i in range(len(slices) - 1):
        assert slices[i][1] == slices[i + 1][0], \
            f"Tâche {i} et {i+1} ne sont pas contiguës"

    # Couverture complète
    assert slices[0][0] == 0
    assert slices[-1][1] == len(features)


def test_no_temporal_leakage(synthetic_features):
    """Vérifie l'absence d'overlap entre tâches (scénario CL correct)."""
    features, labels = synthetic_features
    splitter = CLStreamSplitter(features, labels, n_tasks=N_TASKS)
    slices = splitter._task_slices

    indices = [set(range(s, e)) for s, e in slices]
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            assert indices[i].isdisjoint(indices[j]), \
                f"Overlap entre tâche {i} et tâche {j}"


def test_normalizer_fit_task1(synthetic_features):
    """Après normalisation, mean ≈ 0 sur Task 1 (tolérance 1e-5)."""
    features, labels = synthetic_features
    splitter = CLStreamSplitter(features, labels, n_tasks=N_TASKS)
    normalizer = splitter.fit_normalizer(task_id=0)
    splitter.apply_normalizer(normalizer)

    start, end = splitter._task_slices[0]
    task1_features = splitter.features[start:end]
    assert abs(task1_features.mean()) < 1e-5, \
        f"Mean Task 1 après normalisation : {task1_features.mean():.6f} (attendu ≈ 0)"


def test_strategy_not_chronological_raises(synthetic_features):
    """Vérifie que strategy != 'chronological' lève une erreur."""
    features, labels = synthetic_features
    with pytest.raises(ValueError, match="chronological"):
        CLStreamSplitter(features, labels, strategy="random")


def test_task_tensors_shape(synthetic_features):
    """Vérifie les shapes des DataLoaders."""
    features, labels = synthetic_features
    splitter = CLStreamSplitter(features, labels, n_tasks=N_TASKS)
    _ = splitter.fit_normalizer(task_id=0)
    splitter.apply_normalizer(_)

    task_data = splitter.get_task_tensors(task_id=0, batch_size=16)
    for X_batch, y_batch in task_data["train_loader"]:
        assert X_batch.shape[1] == N_FEATURES, f"X shape incorrect : {X_batch.shape}"
        assert y_batch.shape[1] == 1, f"y shape incorrect : {y_batch.shape}"
        break  # Un seul batch suffit
```

---

## Critères d'acceptation

- [ ] `from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter` — aucune erreur d'import
- [ ] `features.shape == [N_windows, 25]` pour tout dataset valide
- [ ] `labels.shape == [N_windows]`, valeurs ∈ {0.0, 1.0}
- [ ] Découpage chronologique strict : T1 < T2 < T3 (pas de mélange)
- [ ] Absence d'overlap entre tâches : `set(T1) ∩ set(T2) == ∅`
- [ ] `strategy="random"` lève une `ValueError` explicite
- [ ] `fit_normalizer(task_id=0)` → mean ≈ 0 sur Task 1 après `apply_normalizer()`
- [ ] `save_normalizer("configs/pump_normalizer.yaml")` génère un YAML valide committable
- [ ] Annotations `# MEM:` présentes sur les tenseurs de fenêtres et features
- [ ] `pytest tests/test_pump_dataset.py -v` — tous les tests passent
- [ ] `ruff check src/data/pump_dataset.py` + `black --check` passent

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter | Statut |
|---------|-------------|--------|
| `N_FEATURES = 25` confirmé | `configs/tinyol_config.yaml` → `backbone.input_dim` | ⬜ déjà dans le YAML |
| Nb fenêtres total + par tâche | En-tête de `pump_dataset.py` | ⬜ après S3-01 |
| Interface `get_task_tensors` | Utilisée dans S3-04 (`pretrain_tinyol.py`) et S3-06 (exp_003) | ⬜ |
| `configs/pump_normalizer.yaml` | Committable — utilisé par S3-04 et MCU | ⬜ S3-04 |

---

## Interface attendue par `pretrain_tinyol.py` (S3-04) et `train_tinyol.py` (S3-06)

```python
from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter

# Chargement et feature engineering
dataset = PumpMaintenanceDataset(config["data"]["path"])
features, labels = dataset.extract_features(
    window_size=config["data"]["window_size"],
    step_size=config["data"]["step_size"],
)

# Stream CL
cl_stream = CLStreamSplitter(features, labels, n_tasks=config["data"]["n_tasks"])
normalizer = cl_stream.fit_normalizer(task_id=0)
cl_stream.apply_normalizer(normalizer)
cl_stream.save_normalizer(config["data"]["normalizer_path"])

# Task 0 (pré-entraînement backbone : données saines uniquement)
task0 = cl_stream.get_task_tensors(task_id=0, batch_size=config["pretrain"]["batch_size"])
```

---

## Questions ouvertes

- `TODO(arnaud)` : stratégie de découpage en 3 tâches — égal (33%/33%/33%) ou basé sur des seuils de taux de panne observés dans S3-01 ?
- `TODO(arnaud)` : inclure `rpm` comme 4ème canal ? La spec dit 4 canaux mais mentionne aussi que le dataset peut varier — confirmer les colonnes disponibles.
- `TODO(arnaud)` : pour le val split de chaque tâche, conserver l'ordre chronologique (dernière fraction) ou split aléatoire stratifié ? Impact sur l'évaluation CL.
- `FIXME(gap2)` : mesurer l'empreinte RAM réelle du pipeline de fenêtrage avec `memory_profiler.py` lors de S3-06 (exp_003).
