"""
battery_dataset.py — Loader PyTorch pour le Dataset 4 (Battery Remaining Useful Life).

Scénario CL : Domain-Incremental par fenêtres temporelles de cycles (3 tâches)
    Task 1 = cycles   1–378  (début de vie — batterie neuve)
    Task 2 = cycles 379–756  (mi-vie — vieillissement progressif)
    Task 3 = cycles 757–1134 (fin de vie — dégradation accélérée)

Source : data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv
N = 15 064 cycles, 7 features électrochimiques, label binaire RUL < 200.

Pipeline :
    1. Chargement et validation du CSV
    2. Binarisation du label : RUL < RUL_FAILURE_THRESHOLD → dégradé (1)
    3. Split chronologique en 3 tâches égales par Cycle_Index
    4. Normalisation Z-score fixée sur Task 1 uniquement
    5. Retourne des DataLoaders PyTorch compatibles avec les scripts existants

Usage :
    from pathlib import Path
    from src.data.battery_dataset import get_battery_dataloaders

    tasks = get_battery_dataloaders(
        csv_path=Path("data/raw/Battery.../Battery_RUL.csv"),
        normalizer_path=Path("configs/battery_normalizer.yaml"),
    )
    # tasks[0] = {"task_id": 1, "temporal_window": 1, "train_loader": ..., "val_loader": ..., ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

# Colonnes features électrochimiques (Cycle_Index exclu — variable de split ; RUL exclu — cible)
FEATURE_COLUMNS: list[str] = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]

# Nombre de features (7 features électrochimiques)
N_FEATURES: int = len(FEATURE_COLUMNS)  # 7

# Colonne temporelle (axe de découpage CL)
CYCLE_COL: str = "Cycle_Index"

# Colonne cible (RUL en cycles restants)
RUL_COL: str = "RUL"

# Seuil de binarisation : RUL < seuil → dégradé (label=1)
# 200 cycles ≈ 17.6% de la durée de vie totale (~1 134 cycles) — zone de dégradation accélérée
RUL_FAILURE_THRESHOLD: int = 200

# Nombre de tâches CL (fenêtres de cycles égales)
N_TASKS: int = 3

# Ratio validation par tâche
VAL_RATIO: float = 0.2


# ---------------------------------------------------------------------------
# 1. Chargement et validation du CSV
# ---------------------------------------------------------------------------


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Charge le CSV Battery RUL, valide les colonnes et binarise le label.

    Parameters
    ----------
    csv_path : Path
        Chemin vers Battery_RUL.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame trié par Cycle_Index avec colonne ``faulty`` binaire ajoutée.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas.
    ValueError
        Si des colonnes obligatoires sont manquantes.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = FEATURE_COLUMNS + [CYCLE_COL, RUL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    # Binarisation du label : RUL < seuil = dégradé (1), sinon normal (0)
    # MEM: colonne faulty [N] × 1 B @ UINT8
    df["faulty"] = (df[RUL_COL] < RUL_FAILURE_THRESHOLD).astype(np.float32)

    # Tri chronologique par Cycle_Index
    df = df.sort_values(CYCLE_COL).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 2. Normalisation Z-score
# ---------------------------------------------------------------------------


def fit_normalizer(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques Z-score sur un DataFrame (Task 1 uniquement).

    Returns
    -------
    dict
        ``{"mean": {feat: float}, "std": {feat: float}}``
    """
    mean_series = df[FEATURE_COLUMNS].mean()
    std_series = df[FEATURE_COLUMNS].std().replace(0, 1.0)
    return {"mean": mean_series.to_dict(), "std": std_series.to_dict()}


def save_normalizer(path: Path, normalizer: dict) -> None:
    """Sérialise les statistiques de normalisation en YAML."""
    data = {
        "fit_task": "task_1_cycles_1_378",
        "normalization": "zscore",
        "rul_failure_threshold": RUL_FAILURE_THRESHOLD,
        "feature_names": FEATURE_COLUMNS,
        "mean": normalizer["mean"],
        "std": normalizer["std"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_battery_normalizer(config_path: Path) -> dict:
    """
    Charge les statistiques mean/std depuis configs/battery_normalizer.yaml.

    Returns
    -------
    dict
        ``{"mean": {feat: float}, "std": {feat: float}}``
    """
    cfg = load_config(str(config_path))
    if "mean" not in cfg or "std" not in cfg:
        raise ValueError(f"Le fichier normalizer ({config_path}) doit contenir 'mean' et 'std'.")
    return {"mean": cfg["mean"], "std": cfg["std"]}


def normalize_features(df: pd.DataFrame, normalizer: dict) -> pd.DataFrame:
    """
    Applique la normalisation Z-score sur FEATURE_COLUMNS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes FEATURE_COLUMNS.
    normalizer : dict
        ``{"mean": {feat: float}, "std": {feat: float}}``

    Returns
    -------
    pd.DataFrame
        Copie avec les features normalisées.
    """
    df_out = df.copy()
    for feat in FEATURE_COLUMNS:
        df_out[feat] = (df_out[feat] - normalizer["mean"][feat]) / normalizer["std"][feat]
    return df_out


# ---------------------------------------------------------------------------
# 3. Conversion DataFrame → tenseurs
# ---------------------------------------------------------------------------


def df_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit un DataFrame en tenseurs (X, y).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - X : shape [N, 7], dtype float32
        - y : shape [N, 1], dtype float32 — label binaire ``faulty``

    Notes
    -----
    # MEM: batch de 32 → 32×7×4 = 896 B @ FP32 / 224 B @ INT8
    """
    # MEM: X tenseur — N×7×4 B @ FP32 / N×7×1 B @ INT8
    x_np: np.ndarray = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32).copy()
    y_np: np.ndarray = df["faulty"].to_numpy(dtype=np.float32).reshape(-1, 1).copy()
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


# ---------------------------------------------------------------------------
# 4. Interface principale CL (scénario by_temporal_window)
# ---------------------------------------------------------------------------


def get_battery_dataloaders(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    n_tasks: int = N_TASKS,
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL sur le Dataset 4 (Battery RUL).

    Découpe les 15 064 cycles en ``n_tasks`` tranches égales par Cycle_Index :
      - Task 1 : cycles   1–378  → ~5 021 lignes (début de vie)
      - Task 2 : cycles 379–756  → ~5 021 lignes (mi-vie)
      - Task 3 : cycles 757–1134 → ~5 022 lignes (fin de vie)

    Pipeline :
        load_raw_dataset → load_battery_normalizer → normalize_features
        → split chronologique → [df_to_tensors + DataLoader] × 3

    Parameters
    ----------
    csv_path : Path
        Chemin vers Battery_RUL.csv.
    normalizer_path : Path
        Chemin vers configs/battery_normalizer.yaml (doit exister).
    batch_size : int
        Taille des mini-batches. Default : 32.
    val_ratio : float
        Fraction validation par tâche (split stratifié). Default : VAL_RATIO (0.2).
    seed : int
        Seed global. Default : 42.
    n_tasks : int
        Nombre de tâches CL. Default : N_TASKS (3).

    Returns
    -------
    list[dict]
        Liste ordonnée de 3 dicts (Task 1 → 2 → 3) :

        .. code-block:: python

            {
                "task_id": int,           # 1, 2 ou 3
                "temporal_window": int,   # alias sémantique de task_id
                "domain": str,            # alias "window_{N}" pour compatibilité
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    df = load_raw_dataset(csv_path)
    normalizer = load_battery_normalizer(normalizer_path)
    df = normalize_features(df, normalizer)

    n_total = len(df)
    slice_size = n_total // n_tasks

    tasks: list[dict] = []

    for task_idx in range(n_tasks):
        start = task_idx * slice_size
        end = start + slice_size if task_idx < n_tasks - 1 else n_total
        df_task = df.iloc[start:end].reset_index(drop=True)

        # Split stratifié sur faulty pour conserver le taux de défauts
        try:
            df_train, df_val = train_test_split(
                df_task,
                test_size=val_ratio,
                stratify=df_task["faulty"],
                random_state=seed,
            )
        except ValueError:
            # Fallback chronologique si une seule classe
            n_val = max(1, int(len(df_task) * val_ratio))
            df_train = df_task.iloc[:-n_val]
            df_val = df_task.iloc[-n_val:]

        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        x_train, y_train = df_to_tensors(df_train)
        x_val, y_val = df_to_tensors(df_val)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
        )

        tasks.append(
            {
                "task_id": task_idx + 1,
                "temporal_window": task_idx + 1,
                "domain": f"window_{task_idx + 1}",
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_train": len(x_train),
                "n_val": len(x_val),
            }
        )

    return tasks


# ---------------------------------------------------------------------------
# 5. Interface single-task (baseline hors-CL)
# ---------------------------------------------------------------------------


def get_battery_dataloaders_single_task(
    csv_path: Path,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Baseline hors-CL : tous les cycles fusionnés, split global stratifié.

    La normalisation Z-score est fittée sur le train split uniquement.

    Parameters
    ----------
    csv_path : Path
        Chemin vers Battery_RUL.csv.
    batch_size : int
        Taille des mini-batches. Default : 32.
    test_ratio : float
        Fraction test (sur total). Default : 0.2.
    val_ratio : float
        Fraction val (sur train uniquement). Default : 0.1.
    seed : int
        Seed. Default : 42.

    Returns
    -------
    dict
        ``{"train_loader": DataLoader, "val_loader": DataLoader,
           "test_loader": DataLoader, "n_train": int, "n_val": int, "n_test": int,
           "normalizer": dict}``

    Notes
    -----
    # MEM: dataset complet ~15064 × 7 × 4 B = 421 792 B @ FP32
    """
    set_seed(seed)

    df = load_raw_dataset(csv_path)

    # Split stratifié train+val / test
    df_trainval, df_test = train_test_split(
        df, test_size=test_ratio, stratify=df["faulty"], random_state=seed
    )

    # Split stratifié train / val
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_ratio, stratify=df_trainval["faulty"], random_state=seed
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Normalisation Z-score fittée exclusivement sur le train
    # MEM: mean/std [7] × 4 B = 28 B @ FP32 chacun
    normalizer = fit_normalizer(df_train)

    df_train = normalize_features(df_train, normalizer)
    df_val = normalize_features(df_val, normalizer)
    df_test = normalize_features(df_test, normalizer)

    x_train, y_train = df_to_tensors(df_train)
    x_val, y_val = df_to_tensors(df_val)
    x_test, y_test = df_to_tensors(df_test)

    def _make_loader(X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": _make_loader(x_train, y_train, shuffle=True),
        "val_loader": _make_loader(x_val, y_val, shuffle=False),
        "test_loader": _make_loader(x_test, y_test, shuffle=False),
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "normalizer": normalizer,
    }
