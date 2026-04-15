"""
pump_dataset.py — Loader PyTorch pour le Dataset 1 (Large Industrial Pump Maintenance Dataset).

Scénario CL : Domain-Incremental avec drift temporel
    Task 1 = données saines (début)  →  Task 2 = usure progressive  →  Task 3 = pré-défaillance

Source : data/raw/pump_maintenance/Large_Industrial_Pump_Maintenance_Dataset.csv
N = 20 000 échantillons, 4 canaux (temperature, vibration, pressure, rpm).

Pipeline :
    1. Chargement et tri chronologique (Operational_Hours comme axe temporel)
    2. Fenêtrage glissant : WINDOW_SIZE=32, STEP_SIZE=16 → 25 features statistiques
    3. Split chronologique en N_TASKS=3 tranches égales et contiguës
    4. Normalisation Z-score fixée sur Task 1 uniquement
    5. Retourne des DataLoaders PyTorch prêts pour TinyOL

Usage :
    from pathlib import Path
    from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter, get_pump_dataloaders

    ds = PumpMaintenanceDataset(Path("data/raw/pump_maintenance/.../pump.csv"))
    df = ds.load()
    features, labels = ds.extract_features()

    splitter = CLStreamSplitter(features, labels)
    normalizer = splitter.fit_normalizer(task_id=0)
    splitter.apply_normalizer(normalizer)
    splitter.save_normalizer(Path("configs/pump_normalizer.yaml"), normalizer)

    tasks = [splitter.get_task_tensors(i) for i in range(3)]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import kurtosis as scipy_kurtosis
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes de configuration — toutes les valeurs ici, jamais en dur
# ---------------------------------------------------------------------------

# Taille de la fenêtre glissante (points temporels)
WINDOW_SIZE: int = 32

# Pas entre deux fenêtres consécutives (50% overlap)
STEP_SIZE: int = 16

# Nombre total de features par fenêtre : 6 stats × 4 canaux + 1 temporel
N_FEATURES: int = 25

# Nombre de tâches CL (split chronologique égal)
N_TASKS: int = 3

# Nombre de tâches CL — scénario temporel (4 quartiles de 5 000 entrées)
N_TEMPORAL_TASKS: int = 4

# Entrées par tâche dans le scénario temporel (20 000 / 4)
ENTRIES_PER_TEMPORAL_TASK: int = 5000

# Canaux retenus pour le fenêtrage (conforme tinyol_config.yaml)
FEATURE_COLUMNS: list[str] = ["temperature", "vibration", "pressure", "rpm"]

# Colonne label binaire après renommage
LABEL_COLUMN: str = "maintenance_required"

# Colonne temporelle (reconstituée depuis Operational_Hours — pas de timestamp dans le CSV)
TEMPORAL_COLUMN: str = "operational_hours"

# Stats extraites par canal
FEATURES_PER_CHANNEL: list[str] = ["mean", "std", "rms", "kurtosis", "peak", "crest_factor"]

# Noms des 25 features dans l'ordre : 24 stats (6×4 canaux) + 1 temporal_position
FEATURE_NAMES: list[str] = [
    f"{stat}_{col}" for col in FEATURE_COLUMNS for stat in FEATURES_PER_CHANNEL
] + ["temporal_position"]

# Mapping renommage colonnes brutes CSV → noms pipeline
_COLUMN_RENAME: dict[str, str] = {
    "Temperature": "temperature",
    "Vibration": "vibration",
    "Pressure": "pressure",
    "RPM": "rpm",
    "Flow_Rate": "flow_rate",
    "Operational_Hours": "operational_hours",
    "Maintenance_Flag": "maintenance_required",
    "Pump_ID": "pump_id",
}

# Ratio validation par tâche (split temporel : pas de mélange)
VAL_RATIO: float = 0.2


# ---------------------------------------------------------------------------
# 1. Chargement et validation du CSV brut
# ---------------------------------------------------------------------------


class PumpMaintenanceDataset:
    """
    Loader pour le Dataset 1 — Large Industrial Pump Maintenance Dataset.

    Gère le chargement, la validation, le renommage des colonnes,
    le tri chronologique, et l'extraction des 25 features statistiques.

    Parameters
    ----------
    csv_path : Path
        Chemin vers le CSV brut (Large_Industrial_Pump_Maintenance_Dataset.csv).
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        Charge le CSV, renomme les colonnes, trie chronologiquement et valide.

        Returns
        -------
        pd.DataFrame
            DataFrame trié par operational_hours, colonnes normalisées.

        Raises
        ------
        FileNotFoundError
            Si le CSV n'existe pas.
        ValueError
            Si colonnes manquantes ou labels hors {0, 1}.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Renommage colonnes brutes → noms pipeline
        df = df.rename(columns=_COLUMN_RENAME)

        # Validation des colonnes obligatoires
        required_cols = FEATURE_COLUMNS + [LABEL_COLUMN, TEMPORAL_COLUMN]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes après renommage : {missing}\n"
                f"Colonnes présentes : {list(df.columns)}"
            )

        # Validation des labels binaires
        unexpected_labels = set(df[LABEL_COLUMN].unique()) - {0, 1}
        if unexpected_labels:
            raise ValueError(
                f"Labels inattendus dans '{LABEL_COLUMN}' : {unexpected_labels}. "
                f"Attendu : {{0, 1}} uniquement."
            )

        # Tri chronologique par axe temporel (Operational_Hours croissant)
        df = df.sort_values(TEMPORAL_COLUMN).reset_index(drop=True)

        self._df = df
        return df

    def extract_features(
        self,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique le fenêtrage glissant et extrait 25 features statistiques.

        Doit être appelé après :meth:`load`.

        Parameters
        ----------
        window_size : int
            Nombre de points temporels par fenêtre. Default : WINDOW_SIZE (32).
        step_size : int
            Pas entre fenêtres consécutives. Default : STEP_SIZE (16).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - features : shape [N_windows, 25], dtype float32
            - labels   : shape [N_windows], dtype float32 — majority vote par fenêtre

        Notes
        -----
        # MEM: fenêtre brute 32×4×4 = 512 B @ FP32 / 128 B @ INT8
        # MEM: vecteur features 25×4 = 100 B @ FP32 / 25 B @ INT8
        """
        if self._df is None:
            raise RuntimeError("Appeler load() avant extract_features().")

        df = self._df
        n_total = len(df)

        signal_data: np.ndarray = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        label_data: np.ndarray = df[LABEL_COLUMN].to_numpy(dtype=np.float32)

        starts = list(range(0, n_total - window_size + 1, step_size))
        n_windows = len(starts)

        features = np.empty((n_windows, N_FEATURES), dtype=np.float32)
        labels = np.empty(n_windows, dtype=np.float32)

        denom = max(n_total - window_size, 1)

        for i, start in enumerate(starts):
            end = start + window_size
            # MEM: fenêtre brute 32×4×4 = 512 B @ FP32 / 128 B @ INT8
            window = signal_data[start:end]  # [window_size, 4]

            # 24 features statistiques (6 stats × 4 canaux)
            channel_feats = self._compute_channel_features(window)  # [24]

            # Feature 25 : position temporelle normalisée ∈ [0, 1]
            temporal_pos = np.float32(start / denom)

            # MEM: vecteur features 25×4 = 100 B @ FP32 / 25 B @ INT8
            features[i] = np.append(channel_feats, temporal_pos)

            # Label : vote majoritaire sur la fenêtre
            labels[i] = np.float32(1.0 if label_data[start:end].mean() >= 0.5 else 0.0)

        return features, labels

    @staticmethod
    def _compute_channel_features(window: np.ndarray) -> np.ndarray:
        """
        Calcule les 6 features statistiques pour chacun des 4 canaux.

        Parameters
        ----------
        window : np.ndarray
            Shape [window_size, n_channels] (n_channels = 4).

        Returns
        -------
        np.ndarray
            Shape [24] — ordre : [mean_T, std_T, rms_T, kurt_T, peak_T, crest_T,
                                   mean_V, ..., crest_R].
        """
        n_channels = window.shape[1]
        feats = np.empty(n_channels * len(FEATURES_PER_CHANNEL), dtype=np.float32)

        for ch in range(n_channels):
            x = window[:, ch]
            mean = x.mean()
            std = x.std()
            rms = np.sqrt(np.mean(x**2))
            kurt = float(scipy_kurtosis(x, fisher=True))  # excess kurtosis
            peak = float(np.max(np.abs(x)))
            crest = float(peak / rms) if rms > 1e-12 else 0.0

            base = ch * len(FEATURES_PER_CHANNEL)
            feats[base + 0] = mean
            feats[base + 1] = std
            feats[base + 2] = rms
            feats[base + 3] = kurt
            feats[base + 4] = peak
            feats[base + 5] = crest

        return feats


# ---------------------------------------------------------------------------
# 2. Découpage CL chronologique + normalisation
# ---------------------------------------------------------------------------


class CLStreamSplitter:
    """
    Découpe les features/labels en tâches CL chronologiques et gère la normalisation.

    Le seul découpage autorisé est ``"chronological"`` (égal et contigu).
    Aucun mélange temporel pour éviter la fuite d'information inter-tâches.

    Parameters
    ----------
    features : np.ndarray
        Shape [N_windows, N_FEATURES] — sortie de PumpMaintenanceDataset.extract_features().
    labels : np.ndarray
        Shape [N_windows] — labels par fenêtre.
    n_tasks : int
        Nombre de tâches CL. Default : N_TASKS (3).
    strategy : str
        Stratégie de découpage. Seule ``"chronological"`` est autorisée.

    Raises
    ------
    ValueError
        Si ``strategy`` n'est pas ``"chronological"``.
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
                f"Stratégie '{strategy}' non supportée. "
                f"Seule 'chronological' est autorisée (contrainte MCU : pas de mélange)."
            )

        self.strategy = strategy
        self.n_tasks = n_tasks
        self._features = features.copy()
        self._labels = labels.copy()
        self._slices: list[tuple[np.ndarray, np.ndarray]] = []
        self._compute_slices()

    def _compute_slices(self) -> None:
        """Découpe features/labels en n_tasks tranches égales et contiguës."""
        n = len(self._features)
        slice_size = n // self.n_tasks

        self._slices = []
        for i in range(self.n_tasks):
            start = i * slice_size
            # La dernière tâche prend les fenêtres restantes
            end = (i + 1) * slice_size if i < self.n_tasks - 1 else n
            self._slices.append((self._features[start:end], self._labels[start:end]))

    def fit_normalizer(self, task_id: int = 0) -> dict:
        """
        Calcule les statistiques Z-score sur la tâche ``task_id`` (Task 1 = données saines).

        Les stats ne doivent jamais être recalculées en ligne sur MCU.

        Parameters
        ----------
        task_id : int
            Indice de la tâche source (0-indexé). Default : 0 (Task 1).

        Returns
        -------
        dict
            ``{"mean": np.ndarray[N_FEATURES], "std": np.ndarray[N_FEATURES]}``
        """
        feats, _ = self._slices[task_id]
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        # Éviter la division par zéro pour les features constantes
        std = np.where(std < 1e-8, 1.0, std)
        return {"mean": mean, "std": std}

    def apply_normalizer(self, normalizer: dict) -> None:
        """
        Applique la normalisation Z-score sur toutes les tâches.

        Modifie ``self._slices`` en place. Stats fixes — pas de recalcul.

        Parameters
        ----------
        normalizer : dict
            ``{"mean": np.ndarray[N_FEATURES], "std": np.ndarray[N_FEATURES]}``
        """
        mean: np.ndarray = np.asarray(normalizer["mean"], dtype=np.float32)
        std: np.ndarray = np.asarray(normalizer["std"], dtype=np.float32)

        self._slices = [((feats - mean) / std, lbls) for feats, lbls in self._slices]

    def save_normalizer(self, path: Path, normalizer: dict) -> None:
        """
        Sérialise les statistiques de normalisation en YAML.

        Le format est compatible avec ``load_config()`` et committable pour MCU.

        Parameters
        ----------
        path : Path
            Chemin de sortie (ex. ``configs/pump_normalizer.yaml``).
        normalizer : dict
            ``{"mean": np.ndarray[N_FEATURES], "std": np.ndarray[N_FEATURES]}``
        """
        mean_vals = normalizer["mean"]
        std_vals = normalizer["std"]

        data = {
            "fit_task": "task_1_chronological",
            "normalization": "zscore",
            "feature_names": FEATURE_NAMES,
            "mean": {name: float(val) for name, val in zip(FEATURE_NAMES, mean_vals)},
            "std": {name: float(val) for name, val in zip(FEATURE_NAMES, std_vals)},
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def get_task_tensors(
        self,
        task_id: int,
        batch_size: int = 32,
        val_ratio: float = VAL_RATIO,
        seed: int = 42,
    ) -> dict:
        """
        Retourne les DataLoaders train/val pour une tâche donnée.

        Le split préserve l'ordre temporel : les ``val_ratio`` dernières fenêtres
        constituent la validation (pas de mélange aléatoire pour éviter la fuite).

        Parameters
        ----------
        task_id : int
            Indice de la tâche (0-indexé).
        batch_size : int
            Taille des mini-batches. Default : 32.
        val_ratio : float
            Fraction réservée à la validation. Default : VAL_RATIO (0.2).
        seed : int
            Seed pour la reproductibilité. Default : 42.

        Returns
        -------
        dict
            ``{"task_id": int, "train_loader": DataLoader, "val_loader": DataLoader,
               "n_train": int, "n_val": int}``
        """
        set_seed(seed)

        feats, lbls = self._slices[task_id]
        n = len(feats)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val

        # Split temporel strict : train = début, val = fin
        x_train = torch.from_numpy(feats[:n_train].astype(np.float32))  # MEM: n_train×25×4 B @ FP32
        y_train = torch.from_numpy(lbls[:n_train].astype(np.float32)).unsqueeze(
            1
        )  # MEM: n_train×1×4 B @ FP32
        x_val = torch.from_numpy(feats[n_train:].astype(np.float32))
        y_val = torch.from_numpy(lbls[n_train:].astype(np.float32)).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        return {
            "task_id": task_id + 1,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "n_train": n_train,
            "n_val": n_val,
        }


# ---------------------------------------------------------------------------
# 3. Chargement du normaliseur depuis YAML
# ---------------------------------------------------------------------------


def load_pump_normalizer(config_path: Path) -> dict:
    """
    Charge les statistiques mean/std depuis configs/pump_normalizer.yaml.

    Ne recalcule jamais les stats — elles sont fixées depuis S3-02.

    Parameters
    ----------
    config_path : Path
        Chemin vers pump_normalizer.yaml.

    Returns
    -------
    dict
        ``{"mean": np.ndarray[N_FEATURES], "std": np.ndarray[N_FEATURES]}``

    Raises
    ------
    ValueError
        Si les clés ``mean`` ou ``std`` sont absentes.
    """
    cfg = load_config(str(config_path))

    if "mean" not in cfg or "std" not in cfg:
        raise ValueError(f"Le fichier normalizer ({config_path}) doit contenir 'mean' et 'std'.")

    mean = np.array([cfg["mean"][name] for name in FEATURE_NAMES], dtype=np.float32)
    std = np.array([cfg["std"][name] for name in FEATURE_NAMES], dtype=np.float32)

    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# 4. Interface principale
# ---------------------------------------------------------------------------


def get_pump_dataloaders(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL sur le Dataset 1.

    Pipeline :
        load → extract_features → CLStreamSplitter
        → load_pump_normalizer → apply_normalizer → get_task_tensors × 3

    Parameters
    ----------
    csv_path : Path
        Chemin vers le CSV brut.
    normalizer_path : Path
        Chemin vers configs/pump_normalizer.yaml (doit exister).
    batch_size : int
        Taille des mini-batches. Default : 32.
    val_ratio : float
        Fraction de validation par tâche. Default : VAL_RATIO (0.2).
    seed : int
        Seed global. Default : 42.
    window_size : int
        Taille de la fenêtre glissante. Default : WINDOW_SIZE (32).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (16).

    Returns
    -------
    list[dict]
        Liste ordonnée de 3 dicts (Task 1 → Task 2 → Task 3) :

        .. code-block:: python

            {
                "task_id": int,            # 1, 2 ou 3
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    ds = PumpMaintenanceDataset(csv_path)
    ds.load()
    features, labels = ds.extract_features(window_size=window_size, step_size=step_size)

    splitter = CLStreamSplitter(features, labels)

    normalizer = load_pump_normalizer(normalizer_path)
    splitter.apply_normalizer(normalizer)

    return [
        splitter.get_task_tensors(i, batch_size=batch_size, val_ratio=val_ratio, seed=seed)
        for i in range(N_TASKS)
    ]


def get_pump_dataloaders_by_id(
    csv_path: str,
    normalizer_path: str,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> list[dict]:
    """
    Crée un scénario CL domain-incremental où chaque tâche correspond à un Pump_ID distinct.

    Les données sont découpées par identifiant de pompe (Pump_ID ∈ {1, 2, 3, 4, 5}), donnant
    5 tâches ordonnées croissamment. Contrairement au scénario chronologique, le drift est
    inter-pompe (caractéristiques mécaniques distinctes) plutôt qu'intra-pompe (vieillissement).

    Pipeline par tâche :
        load → filter(pump_id) → extract_features → apply_normalizer → stratified_split

    Parameters
    ----------
    csv_path : str
        Chemin complet vers le CSV pump maintenance.
    normalizer_path : str
        Chemin vers pump_normalizer.yaml (normaliseur ajusté sur Task 1 chronologique).
    batch_size : int
        Taille de batch pour les DataLoaders. Default : VAL_RATIO (32).
    val_ratio : float
        Fraction validation (split stratifié sur label). Default : VAL_RATIO (0.2).
    seed : int
        Seed reproductibilité. Default : 42.
    window_size : int
        Taille fenêtre glissante. Default : WINDOW_SIZE (32).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (16).

    Returns
    -------
    list[dict]
        Liste de 5 dicts (un par Pump_ID croissant) :

        .. code-block:: python

            {
                "task_id": int,            # 1..5
                "pump_id": int,            # identifiant pompe (1..5)
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    # 1. Chargement complet du dataset
    ds = PumpMaintenanceDataset(Path(csv_path))
    full_df = ds.load()

    # 2. Colonne pump_id renommée par _COLUMN_RENAME ("Pump_ID" → "pump_id")
    if "pump_id" not in full_df.columns:
        raise ValueError(
            f"Colonne 'pump_id' introuvable après renommage. "
            f"Colonnes présentes : {list(full_df.columns)}"
        )
    pump_ids = sorted(full_df["pump_id"].unique())

    # 3. Normalisation Z-score (stats fixes, calculées sur Task 1 chronologique)
    normalizer = load_pump_normalizer(Path(normalizer_path))
    mean_vec = normalizer["mean"]  # [N_FEATURES]
    std_vec = normalizer["std"]    # [N_FEATURES]

    result: list[dict] = []

    for task_idx, pid in enumerate(pump_ids):
        # 4a. Filtrer les lignes du pump_id courant (ordre chronologique préservé)
        pump_df = full_df[full_df["pump_id"] == pid].copy().reset_index(drop=True)

        # 4b. Extraction features via la méthode existante (fenêtrage glissant)
        original_df = ds._df
        ds._df = pump_df
        X, y = ds.extract_features(window_size=window_size, step_size=step_size)
        ds._df = original_df  # restaurer l'état original

        # 4c. Normalisation Z-score avec stats fixes
        # MEM: X [N_windows, 25] × 4B = N_windows × 100 B @ FP32
        X = (X - mean_vec) / std_vec

        # 4d. Split stratifié par label (évite un split purement temporel
        #     qui pourrait concentrer les défauts dans val uniquement)
        n = len(X)
        unique_classes = np.unique(y)

        if len(unique_classes) >= 2 and int(n * val_ratio) >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            train_idx, val_idx = next(sss.split(X, y))
        else:
            # Fallback temporel si une seule classe ou dataset trop petit
            n_val = max(1, int(n * val_ratio))
            train_idx = np.arange(n - n_val)
            val_idx = np.arange(n - n_val, n)

        x_train = torch.from_numpy(X[train_idx].astype(np.float32))  # MEM: n_train×25×4 B @ FP32
        y_train = torch.from_numpy(y[train_idx].astype(np.float32)).unsqueeze(1)
        x_val = torch.from_numpy(X[val_idx].astype(np.float32))
        y_val = torch.from_numpy(y[val_idx].astype(np.float32)).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        result.append(
            {
                "task_id": task_idx + 1,
                "pump_id": int(pid),
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

    return result


def get_pump_dataloaders_by_temporal_window(
    csv_path: str | Path,
    normalizer_path: str | Path,
    n_tasks: int = N_TEMPORAL_TASKS,
    entries_per_task: int = ENTRIES_PER_TEMPORAL_TASK,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> list[dict]:
    """
    Crée un scénario CL domain-incremental par fenêtres temporelles.

    Découpe les 20 000 entrées en ``n_tasks`` tranches de ``entries_per_task`` lignes
    chacune, triées par Operational_Hours (ordre chronologique global, tous Pump_ID mélangés).

      T1 : lignes 0–4 999    (Operational_Hours les plus basses)
      T2 : lignes 5 000–9 999
      T3 : lignes 10 000–14 999
      T4 : lignes 15 000–19 999

    Applique le même feature engineering que ``get_pump_dataloaders()`` :
    fenêtrage WINDOW_SIZE=32, STEP_SIZE=16, 6 stats × 4 canaux + temporal_position.

    Normalisation Z-score ajustée sur T1 uniquement (``pump_normalizer.yaml`` préajusté).

    Split train/val **temporel** (chronologique, pas stratifié) pour respecter l'ordre
    causal : les ``val_ratio`` dernières fenêtres de chaque tâche constituent le set de
    validation.

    Parameters
    ----------
    csv_path : str | Path
        Chemin complet vers le CSV pump maintenance.
    normalizer_path : str | Path
        Chemin vers ``pump_normalizer.yaml`` (normaliseur ajusté sur Task 1 chronologique).
    n_tasks : int
        Nombre de tâches (quartiles). Default : N_TEMPORAL_TASKS (4).
    entries_per_task : int
        Nombre de lignes CSV par tâche. Default : ENTRIES_PER_TEMPORAL_TASK (5 000).
    batch_size : int
        Taille de batch pour les DataLoaders. Default : 32.
    val_ratio : float
        Fraction validation (split temporel, pas stratifié). Default : VAL_RATIO (0.2).
    seed : int
        Seed reproductibilité. Default : 42.
    window_size : int
        Taille fenêtre glissante. Default : WINDOW_SIZE (32).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (16).

    Returns
    -------
    list[dict]
        Liste de ``n_tasks`` dicts (un par quartile temporel) :

        .. code-block:: python

            {
                "task_id": int,            # 1..n_tasks
                "temporal_window": int,    # alias sémantique de task_id
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    # 1. Chargement + tri chronologique global (load() trie par operational_hours)
    ds = PumpMaintenanceDataset(Path(csv_path))
    full_df = ds.load()  # déjà trié par operational_hours croissant

    # 2. Chargement du normaliseur fixe (ajusté sur T1 chronologique)
    normalizer = load_pump_normalizer(Path(normalizer_path))
    mean_vec = normalizer["mean"]  # [N_FEATURES]
    std_vec = normalizer["std"]    # [N_FEATURES]

    result: list[dict] = []

    for task_idx in range(n_tasks):
        # 3. Découper la tranche de entries_per_task lignes (indices continus)
        start = task_idx * entries_per_task
        end = start + entries_per_task
        slice_df = full_df.iloc[start:end].copy().reset_index(drop=True)

        # 4. Extraction features via fenêtrage glissant (même pipeline que get_pump_dataloaders)
        original_df = ds._df
        ds._df = slice_df
        X, y = ds.extract_features(window_size=window_size, step_size=step_size)
        ds._df = original_df  # restaurer l'état original

        # 5. Normalisation Z-score avec stats fixes (pas de recalcul par tâche)
        # MEM: X [N_windows, 25] × 4B = N_windows × 100 B @ FP32
        X = (X - mean_vec) / std_vec

        # 6. Split train/val temporel (chronologique — pas de shuffle pour respecter causalité)
        n = len(X)
        n_val = max(1, int(n * val_ratio))
        train_idx = np.arange(n - n_val)
        val_idx = np.arange(n - n_val, n)

        x_train = torch.from_numpy(X[train_idx].astype(np.float32))  # MEM: n_train×25×4 B @ FP32
        y_train = torch.from_numpy(y[train_idx].astype(np.float32)).unsqueeze(1)
        x_val = torch.from_numpy(X[val_idx].astype(np.float32))
        y_val = torch.from_numpy(y[val_idx].astype(np.float32)).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        result.append(
            {
                "task_id": task_idx + 1,
                "temporal_window": task_idx + 1,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

    return result


# ---------------------------------------------------------------------------
# 6. Loader single-task (baseline hors-CL)
# ---------------------------------------------------------------------------


def get_pump_dataloaders_single_task(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> dict:
    """
    Retourne un dict unique (pas une liste) avec toutes les données pump fusionnées.

    Pas de découpage temporel ou par pump_id — baseline hors-CL.
    Tous les Pump_ID et toutes les périodes opérationnelles sont réunis.
    La normalisation Z-score est fittée sur le train split uniquement.

    Parameters
    ----------
    csv_path : Path
        Chemin vers le CSV brut (Large_Industrial_Pump_Maintenance_Dataset.csv).
    normalizer_path : Path
        Accepté pour compatibilité avec les autres loaders, mais ignoré.
        La normalisation est calculée sur le train split en interne.
    batch_size : int
        Taille des mini-batches. Default : 32.
    test_ratio : float
        Fraction réservée au test (sur le dataset total). Default : 0.2.
    val_ratio : float
        Fraction réservée à la validation (sur le train uniquement). Default : 0.1.
    seed : int
        Seed pour la reproductibilité. Default : 42.
    window_size : int
        Taille de la fenêtre glissante. Default : WINDOW_SIZE (32).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (16).

    Returns
    -------
    dict
        ``{"train_loader": DataLoader, "val_loader": DataLoader,
           "test_loader": DataLoader, "n_train": int, "n_val": int, "n_test": int}``
    """
    set_seed(seed)

    # 1. Chargement du CSV complet (tous Pump_ID, toutes périodes)
    ds = PumpMaintenanceDataset(Path(csv_path))
    ds.load()

    # 2. Extraction des 25 features via fenêtrage glissant (identique aux autres loaders)
    # MEM: X [N_windows, 25] × 4 B = N_windows × 100 B @ FP32
    X, y = ds.extract_features(window_size=window_size, step_size=step_size)

    n_total = len(X)

    # 3. Split stratifié train+val / test
    n_test = max(1, int(n_total * test_ratio))
    try:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        trainval_idx, test_idx = next(sss_test.split(X, y))
    except ValueError:
        # Fallback si classe unique ou données insuffisantes
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        test_idx = perm[:n_test]
        trainval_idx = perm[n_test:]

    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 4. Split stratifié train / val (sur le train+val uniquement)
    n_val = max(1, int(len(X_trainval) * val_ratio))
    try:
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
        train_idx_inner, val_idx_inner = next(sss_val.split(X_trainval, y_trainval))
    except ValueError:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(len(X_trainval))
        val_idx_inner = perm[:n_val]
        train_idx_inner = perm[n_val:]

    X_train = X_trainval[train_idx_inner]
    y_train = y_trainval[train_idx_inner]
    X_val = X_trainval[val_idx_inner]
    y_val = y_trainval[val_idx_inner]

    # 5. Normalisation Z-score fittée sur X_train uniquement (pas de fuite vers val/test)
    # MEM: mean/std [25] × 4 B = 100 B @ FP32 chacun
    mean_vec = X_train.mean(axis=0)
    std_vec = X_train.std(axis=0)
    std_vec[std_vec == 0] = 1.0  # éviter la division par zéro

    X_train = (X_train - mean_vec) / std_vec
    X_val = (X_val - mean_vec) / std_vec
    X_test = (X_test - mean_vec) / std_vec

    # 6. Conversion en tenseurs et DataLoaders
    x_tr = torch.from_numpy(X_train.astype(np.float32))   # MEM: n_train×25×4 B @ FP32
    y_tr = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    x_va = torch.from_numpy(X_val.astype(np.float32))
    y_va = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)
    x_te = torch.from_numpy(X_test.astype(np.float32))
    y_te = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_va, y_va), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
