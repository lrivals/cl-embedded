"""
cwru_dataset.py — Loader numpy/pandas pour le CWRU Bearing Dataset.

Scénario CL : Domain-Incremental
    by_fault_type : Task 0 = Ball  →  Task 1 = Inner Race  →  Task 2 = Outer Race
    by_severity   : Task 0 = 0.007"  →  Task 1 = 0.014"  →  Task 2 = 0.021"

Source : data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv
N = 2300 fenêtres × 9 features statistiques. Label : 0 = Normal, 1 = Défaut.

Pas de dépendance PyTorch — numpy + pandas uniquement (portabilité MCU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = ["max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form"]
N_FEATURES: int = len(FEATURE_COLS)  # 9

FAULT_COL: str = "fault"
NORMAL_LABEL: str = "Normal_1"
N_TASKS: int = 3

# Ordre des tâches — by_fault_type
FAULT_TYPE_ORDER: list[str] = ["ball", "inner_race", "outer_race"]

FAULT_TYPE_LABELS: dict[str, list[str]] = {
    "ball":        ["Ball_007_1", "Ball_014_1", "Ball_021_1"],
    "inner_race":  ["IR_007_1",   "IR_014_1",   "IR_021_1"],
    "outer_race":  ["OR_007_6_1", "OR_014_6_1", "OR_021_6_1"],
}

# Ordre des tâches — by_severity
SEVERITY_ORDER: list[str] = ["007", "014", "021"]

SEVERITY_LABELS: dict[str, list[str]] = {
    "007": ["Ball_007_1", "IR_007_1",  "OR_007_6_1"],
    "014": ["Ball_014_1", "IR_014_1",  "OR_014_6_1"],
    "021": ["Ball_021_1", "IR_021_1",  "OR_021_6_1"],
}

# Toutes les étiquettes de défaut attendues dans le CSV
_ALL_FAULT_LABELS: set[str] = {
    label
    for labels in FAULT_TYPE_LABELS.values()
    for label in labels
} | {NORMAL_LABEL}


# ---------------------------------------------------------------------------
# CWRUDataset
# ---------------------------------------------------------------------------


class CWRUDataset:
    """
    Charge feature_time_48k_2048_load_1.csv et expose X / y en numpy.

    Note : le CSV contient 2300 lignes (10 classes × 230 fenêtres chacune).
    La spec S1201 mentionne 2299 — écart d'une ligne, la valeur réelle est 2300.

    Parameters
    ----------
    csv_path : str | Path
        Chemin vers le CSV pré-calculé.
    mat_dir : str | Path | None
        Répertoire des fichiers MAT bruts (optionnel, réservé pour usage futur).
    random_state : int
        Seed pour le shuffle des lignes Normal avant répartition entre tâches.

    Attributes
    ----------
    X : np.ndarray
        Shape (N, 9), dtype float32.
    y : np.ndarray
        Shape (N,), dtype int8. 0 = Normal, 1 = Défaut.
    fault_labels : np.ndarray
        Shape (N,), dtype object. Étiquettes brutes (ex. "Ball_007_1").
    """

    def __init__(
        self,
        csv_path: str | Path,
        mat_dir: str | Path | None = None,
        random_state: int = 42,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._mat_dir = Path(mat_dir) if mat_dir is not None else None
        self.random_state = random_state

        self._df = self._load_and_validate()
        self._shuffle_normal()
        self.X, self.y, self.fault_labels = self._extract_arrays()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_validate(self) -> pd.DataFrame:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {self._csv_path}")

        df = pd.read_csv(self._csv_path)

        required = set(FEATURE_COLS + [FAULT_COL])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

        if df[FEATURE_COLS].isna().any(axis=None):
            raise ValueError("NaN détectés dans les features — CSV corrompu.")

        unknown = set(df[FAULT_COL].unique()) - _ALL_FAULT_LABELS
        if unknown:
            raise ValueError(f"Étiquettes de défaut inconnues : {unknown}")

        return df

    def _shuffle_normal(self) -> None:
        # Shuffle les lignes Normal avec le seed fixé avant la répartition par tâche.
        # Les deux streams verront le même ordre shufflé.
        normal_idx = self._df.index[self._df[FAULT_COL] == NORMAL_LABEL].to_numpy()
        rng = np.random.default_rng(self.random_state)
        shuffled_idx = rng.permutation(normal_idx)
        # Réassigner les lignes Normal dans le DataFrame dans l'ordre shufflé
        normal_rows = self._df.loc[normal_idx].copy()
        normal_rows.index = shuffled_idx
        self._df.loc[shuffled_idx] = normal_rows.values
        self._df = self._df.sort_index()

    def _extract_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = (self._df[FAULT_COL] != NORMAL_LABEL).to_numpy(dtype=np.int8)
        fault_labels = self._df[FAULT_COL].to_numpy()
        return X, y, fault_labels


# ---------------------------------------------------------------------------
# CWRUFaultTypeStream
# ---------------------------------------------------------------------------


class CWRUFaultTypeStream:
    """
    Stream domain-incremental par type de défaut.

    Tâches (dans l'ordre) :
        0 "ball"        : Ball_007/014/021 + Normal[0:77]
        1 "inner_race"  : IR_007/014/021   + Normal[77:154]
        2 "outer_race"  : OR_007/014/021   + Normal[154:]

    Parameters
    ----------
    dataset : CWRUDataset

    Yields
    ------
    task_id : int
    task_name : str
    X_task : np.ndarray  shape (N_task, 9), float32
    y_task : np.ndarray  shape (N_task,),   int8
    """

    def __init__(self, dataset: CWRUDataset) -> None:
        self._dataset = dataset

    def iter_tasks(
        self,
    ) -> Generator[tuple[int, str, np.ndarray, np.ndarray], None, None]:
        ds = self._dataset
        normal_mask = ds.fault_labels == NORMAL_LABEL
        normal_splits_X = np.array_split(ds.X[normal_mask], N_TASKS)   # [77, 77, 76]
        normal_splits_y = np.array_split(ds.y[normal_mask], N_TASKS)

        for task_id, task_name in enumerate(FAULT_TYPE_ORDER):
            fault_mask = np.isin(ds.fault_labels, FAULT_TYPE_LABELS[task_name])
            X_task = np.concatenate([ds.X[fault_mask], normal_splits_X[task_id]])
            y_task = np.concatenate([ds.y[fault_mask], normal_splits_y[task_id]])
            yield task_id, task_name, X_task, y_task


# ---------------------------------------------------------------------------
# CWRUSeverityStream
# ---------------------------------------------------------------------------


class CWRUSeverityStream:
    """
    Stream domain-incremental par sévérité du défaut.

    Tâches (dans l'ordre) :
        0 "007" : Ball/IR/OR_007 + Normal[0:77]
        1 "014" : Ball/IR/OR_014 + Normal[77:154]
        2 "021" : Ball/IR/OR_021 + Normal[154:]

    Parameters
    ----------
    dataset : CWRUDataset

    Yields
    ------
    task_id : int
    task_name : str
    X_task : np.ndarray  shape (N_task, 9), float32
    y_task : np.ndarray  shape (N_task,),   int8
    """

    def __init__(self, dataset: CWRUDataset) -> None:
        self._dataset = dataset

    def iter_tasks(
        self,
    ) -> Generator[tuple[int, str, np.ndarray, np.ndarray], None, None]:
        ds = self._dataset
        normal_mask = ds.fault_labels == NORMAL_LABEL
        normal_splits_X = np.array_split(ds.X[normal_mask], N_TASKS)   # [77, 77, 76]
        normal_splits_y = np.array_split(ds.y[normal_mask], N_TASKS)

        for task_id, task_name in enumerate(SEVERITY_ORDER):
            fault_mask = np.isin(ds.fault_labels, SEVERITY_LABELS[task_name])
            X_task = np.concatenate([ds.X[fault_mask], normal_splits_X[task_id]])
            y_task = np.concatenate([ds.y[fault_mask], normal_splits_y[task_id]])
            yield task_id, task_name, X_task, y_task


# ---------------------------------------------------------------------------
# Single-task DataLoaders (scénario no_split — baseline hors-CL)
# ---------------------------------------------------------------------------


def get_cwru_cl_dataloaders_by_fault_type(
    csv_path: str | Path,
    batch_size: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Retourne 3 dicts de tâches pour le scénario CL domain-incremental by_fault_type.

    Ordre : Ball → Inner Race → Outer Race.
    StandardScaler fit uniquement sur le train set de la Tâche 0 (pas de data leakage).
    Compatible avec les boucles CL de train_ewc.py / train_hdc.py / train_tinyol.py.

    Parameters
    ----------
    csv_path : str | Path
    batch_size : int
    test_ratio : float
    val_ratio : float
    seed : int

    Returns
    -------
    list[dict]  — 3 éléments, chacun avec :
        task_id, task_name, domain, train_loader, val_loader, test_loader,
        n_train, n_val, n_test
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = CWRUDataset(csv_path, random_state=seed)
    stream = CWRUFaultTypeStream(ds)

    tasks: list[dict] = []
    scaler: StandardScaler | None = None

    for task_id, task_name, X_task, y_task in stream.iter_tasks():
        y_float = y_task.astype(np.float32)

        # Stratified split : test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_task, y_float, test_size=test_ratio, random_state=seed, stratify=y_float
        )

        # Stratified split : val (depuis trainval)
        val_ratio_adj = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio_adj, random_state=seed, stratify=y_trainval
        )

        # Normalisation — fit uniquement sur Task 0 (no data leakage)
        if task_id == 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
        else:
            X_train = scaler.transform(X_train).astype(np.float32)  # type: ignore[union-attr]
        X_val = scaler.transform(X_val).astype(np.float32)  # type: ignore[union-attr]
        X_test = scaler.transform(X_test).astype(np.float32)  # type: ignore[union-attr]

        def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
            Xt = torch.from_numpy(X_arr)
            yt = torch.from_numpy(y_arr).unsqueeze(1)
            return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

        tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "domain": task_name,
            "train_loader": _make_loader(X_train, y_train, shuffle=True),
            "val_loader": _make_loader(X_val, y_val, shuffle=False),
            "test_loader": _make_loader(X_test, y_test, shuffle=False),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

    return tasks


def get_cwru_cl_dataloaders_by_severity(
    csv_path: str | Path,
    batch_size: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Retourne 3 dicts de tâches pour le scénario CL domain-incremental by_severity.

    Ordre : 0.007" → 0.014" → 0.021" (dégradation progressive).
    StandardScaler fit uniquement sur le train set de la Tâche 0 (pas de data leakage).
    Compatible avec les boucles CL de train_ewc.py / train_hdc.py / train_tinyol.py.

    Parameters
    ----------
    csv_path : str | Path
    batch_size : int
    test_ratio : float
    val_ratio : float
    seed : int

    Returns
    -------
    list[dict]  — 3 éléments, chacun avec :
        task_id, task_name, domain, train_loader, val_loader, test_loader,
        n_train, n_val, n_test
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = CWRUDataset(csv_path, random_state=seed)
    stream = CWRUSeverityStream(ds)

    tasks: list[dict] = []
    scaler: StandardScaler | None = None

    for task_id, task_name, X_task, y_task in stream.iter_tasks():
        y_float = y_task.astype(np.float32)

        # Stratified split : test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_task, y_float, test_size=test_ratio, random_state=seed, stratify=y_float
        )

        # Stratified split : val (depuis trainval)
        val_ratio_adj = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio_adj, random_state=seed, stratify=y_trainval
        )

        # Normalisation — fit uniquement sur Task 0 (no data leakage)
        if task_id == 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
        else:
            X_train = scaler.transform(X_train).astype(np.float32)  # type: ignore[union-attr]
        X_val = scaler.transform(X_val).astype(np.float32)  # type: ignore[union-attr]
        X_test = scaler.transform(X_test).astype(np.float32)  # type: ignore[union-attr]

        def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
            Xt = torch.from_numpy(X_arr)
            yt = torch.from_numpy(y_arr).unsqueeze(1)
            return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

        tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "domain": task_name,
            "train_loader": _make_loader(X_train, y_train, shuffle=True),
            "val_loader": _make_loader(X_val, y_val, shuffle=False),
            "test_loader": _make_loader(X_test, y_test, shuffle=False),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

    return tasks


def get_cwru_dataloaders_single_task(
    csv_path: str | Path,
    batch_size: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Charge le CWRU dataset complet en un seul split train/val/test.

    Normalisation StandardScaler fit sur le train set uniquement.
    Compatible avec les boucles single-task de train_ewc.py / train_hdc.py / train_tinyol.py.

    Parameters
    ----------
    csv_path : str | Path
        Chemin vers feature_time_48k_2048_load_1.csv.
    batch_size : int
        Taille de batch (1 pour online / MCU).
    test_ratio : float
        Fraction du dataset pour le test (0.2 = 20%).
    val_ratio : float
        Fraction du dataset pour la validation (0.1 = 10% du total).
    seed : int
        Seed pour les splits aléatoires.

    Returns
    -------
    dict avec : train_loader, val_loader, test_loader, n_train, n_val, n_test,
                scaler (StandardScaler fit sur train).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = CWRUDataset(csv_path, random_state=seed)
    X, y = ds.X, ds.y.astype(np.float32)

    # Stratified split : test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    # Stratified split : val (depuis trainval)
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adj, random_state=seed, stratify=y_trainval
    )

    # Normalisation StandardScaler (fit uniquement sur train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        Xt = torch.from_numpy(X_arr)
        yt = torch.from_numpy(y_arr).unsqueeze(1)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": _make_loader(X_train, y_train, shuffle=True),
        "val_loader": _make_loader(X_val, y_val, shuffle=False),
        "test_loader": _make_loader(X_test, y_test, shuffle=False),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scaler": scaler,
    }
