"""
pronostia_dataset.py — Loader PyTorch pour le Dataset 3 (FEMTO PRONOSTIA IEEE PHM 2012).

Scénario CL : Domain-Incremental par condition opératoire
    Task 1 = Condition 1 (1 800 rpm, 4 000 N) → Bearing1_1 + Bearing1_2
    Task 2 = Condition 2 (1 650 rpm, 4 200 N) → Bearing2_1 + Bearing2_2
    Task 3 = Condition 3 (1 500 rpm, 5 000 N) → Bearing3_1 + Bearing3_2

Source : data/raw/Pronostia dataset/binaries/{0..5}.npy
Format binaire : shape [N_samples, 6] — colonnes [hour, min, sec, microsec, acc_horiz, acc_vert]
Résout : FIXME(gap1) — premier résultat CL sur données industrielles réelles de roulements.

Pipeline :
    1. Chargement des 2 fichiers .npy de la condition (2 roulements)
    2. Fenêtrage sans overlap : WINDOW_SIZE=2560 (1 epoch = 0.1 s @ 25.6 kHz)
    3. Extraction de 12 features statistiques (6 stats × 2 canaux) + position temporelle = 13
    4. Label binaire : derniers FAILURE_RATIO=10% des fenêtres = pré-défaillance (1)
    5. Split stratifié train/val par défaut, chronologique en fallback
    6. Normalisation Z-score fixée sur Task 1 (Condition 1)

Usage :
    from pathlib import Path
    from src.data.pronostia_dataset import get_pronostia_dataloaders

    tasks = get_pronostia_dataloaders(
        npy_dir=Path("data/raw/Pronostia dataset/binaries"),
        normalizer_path=Path("configs/pronostia_normalizer.yaml"),
    )
    # tasks[0] = {"task_id": 1, "condition": 1, "train_loader": ..., "val_loader": ..., ...}

Citation : Nectoux et al., IEEE PHM 2012.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

# Taille d'une fenêtre = 1 epoch de mesure PRONOSTIA (2 560 points @ 25.6 kHz = 0.1 s)
WINDOW_SIZE: int = 2560

# Pas entre fenêtres — sans overlap (fenêtres disjointes = epochs indépendants)
STEP_SIZE: int = 2560

# Nombre de canaux d'accélérométrie (horizontal + vertical)
N_CHANNELS: int = 2

# Statistiques extraites par canal — identiques à pump_dataset.py
FEATURES_PER_CHANNEL: list[str] = ["mean", "std", "rms", "kurtosis", "peak", "crest_factor"]

# Features totales = 6 stats × 2 canaux + position temporelle
N_FEATURES: int = N_CHANNELS * len(FEATURES_PER_CHANNEL) + 1  # 13

# Noms des 13 features dans l'ordre
CHANNEL_NAMES: list[str] = ["acc_horiz", "acc_vert"]
FEATURE_NAMES: list[str] = [
    f"{stat}_{ch}" for ch in CHANNEL_NAMES for stat in FEATURES_PER_CHANNEL
] + ["temporal_position"]

# Fraction terminale du signal marquée pré-défaillance (label=1)
# Choix : 10% ≈ protocole PRONOSTIA standard (dégradation accélérée phase finale)
FAILURE_RATIO: float = 0.10

# Nombre de conditions opératoires (tâches CL)
N_CONDITIONS: int = 3

# Mapping condition → indices fichiers .npy (0=Bearing1_1, 1=Bearing1_2, ...)
CONDITION_BEARING_MAP: dict[int, list[int]] = {
    1: [0, 1],  # Condition 1 : 1 800 rpm, 4 000 N
    2: [2, 3],  # Condition 2 : 1 650 rpm, 4 200 N
    3: [4, 5],  # Condition 3 : 1 500 rpm, 5 000 N
}

# Indices colonnes dans les .npy
COL_ACC_HORIZ: int = 4
COL_ACC_VERT: int = 5

# Ratio validation par tâche
VAL_RATIO: float = 0.2


# ---------------------------------------------------------------------------
# 1. Chargement et extraction de features depuis un fichier .npy
# ---------------------------------------------------------------------------


def load_bearing_features(
    npy_path: Path,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    failure_ratio: float = FAILURE_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge un fichier .npy et extrait les features statistiques par fenêtre.

    Chaque fichier correspond à un roulement complet depuis l'état sain jusqu'à la défaillance.
    Le label est dérivé de la position temporelle relative dans le signal.

    Parameters
    ----------
    npy_path : Path
        Chemin vers le fichier binaire (ex. ``data/raw/Pronostia dataset/binaries/0.npy``).
    window_size : int
        Taille de la fenêtre en points. Default : WINDOW_SIZE (2 560).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (2 560) — sans overlap.
    failure_ratio : float
        Fraction terminale du signal étiquetée pré-défaillance. Default : FAILURE_RATIO (0.10).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - features : shape [N_windows, 13], dtype float32
        - labels   : shape [N_windows], dtype float32 — 1.0 = pré-défaillance

    Notes
    -----
    # MEM: fenêtre brute 2560×2×4 = 20 480 B @ FP32 / 5 120 B @ INT8
    # MEM: vecteur features 13×4 = 52 B @ FP32 / 13 B @ INT8
    """
    if not npy_path.exists():
        raise FileNotFoundError(f"Fichier .npy introuvable : {npy_path}")

    data = np.load(npy_path)  # [N_samples, 6]

    # Extraction des 2 canaux d'accélérométrie
    # MEM: signal brut 2 canaux × N_samples × 4B @ FP32
    acc = data[:, [COL_ACC_HORIZ, COL_ACC_VERT]].astype(np.float32)

    n_total = len(acc)
    starts = list(range(0, n_total - window_size + 1, step_size))
    n_windows = len(starts)

    # Seuil de label : dernières failure_ratio fenêtres = pré-défaillance
    failure_start_idx = int(n_windows * (1.0 - failure_ratio))

    features = np.empty((n_windows, N_FEATURES), dtype=np.float32)
    labels = np.empty(n_windows, dtype=np.float32)

    denom = max(n_total - window_size, 1)

    for i, start in enumerate(starts):
        end = start + window_size
        # MEM: fenêtre brute 2560×2×4 = 20 480 B @ FP32 / 5 120 B @ INT8
        window = acc[start:end]  # [2560, 2]

        channel_feats = _compute_channel_features(window)  # [12]
        temporal_pos = np.float32(start / denom)

        # MEM: vecteur features 13×4 = 52 B @ FP32 / 13 B @ INT8
        features[i] = np.append(channel_feats, temporal_pos)
        labels[i] = np.float32(1.0 if i >= failure_start_idx else 0.0)

    return features, labels


def _compute_channel_features(window: np.ndarray) -> np.ndarray:
    """
    Calcule les 6 features statistiques pour chacun des 2 canaux.

    Parameters
    ----------
    window : np.ndarray
        Shape [window_size, 2] — colonnes : acc_horiz, acc_vert.

    Returns
    -------
    np.ndarray
        Shape [12] — ordre : [mean_h, std_h, rms_h, kurt_h, peak_h, crest_h,
                               mean_v, std_v, rms_v, kurt_v, peak_v, crest_v].
    """
    feats = np.empty(N_CHANNELS * len(FEATURES_PER_CHANNEL), dtype=np.float32)

    for ch in range(N_CHANNELS):
        x = window[:, ch]
        mean = x.mean()
        std = x.std()
        rms = np.sqrt(np.mean(x**2))
        kurt_val = scipy_kurtosis(x, fisher=True)
        kurt = float(kurt_val) if np.isfinite(kurt_val) else 0.0
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
# 2. Agrégation par condition (2 roulements → 1 tâche CL)
# ---------------------------------------------------------------------------


def load_condition_features(
    npy_dir: Path,
    condition: int,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    failure_ratio: float = FAILURE_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge et concatène les features des 2 roulements d'une condition opératoire.

    Parameters
    ----------
    npy_dir : Path
        Répertoire contenant les 6 fichiers .npy (0.npy à 5.npy).
    condition : int
        Numéro de condition opératoire (1, 2 ou 3).
    window_size : int
        Taille de fenêtre. Default : WINDOW_SIZE (2 560).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (2 560).
    failure_ratio : float
        Fraction terminale pré-défaillance. Default : FAILURE_RATIO (0.10).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features et labels concaténés des 2 roulements.
    """
    if condition not in CONDITION_BEARING_MAP:
        raise ValueError(
            f"Condition {condition} invalide. Valeurs autorisées : {list(CONDITION_BEARING_MAP)}"
        )

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for bearing_idx in CONDITION_BEARING_MAP[condition]:
        npy_path = npy_dir / f"{bearing_idx}.npy"
        feats, lbls = load_bearing_features(npy_path, window_size, step_size, failure_ratio)
        all_features.append(feats)
        all_labels.append(lbls)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# 3. Normalisation Z-score
# ---------------------------------------------------------------------------


def fit_normalizer(features: np.ndarray) -> dict:
    """
    Calcule les statistiques Z-score sur un ensemble de features.

    Toujours fitter sur la Task 1 (Condition 1) uniquement pour éviter la fuite d'info.

    Returns
    -------
    dict
        ``{"mean": np.ndarray[13], "std": np.ndarray[13]}``
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def save_normalizer(path: Path, normalizer: dict) -> None:
    """Sérialise les statistiques de normalisation en YAML."""
    data = {
        "fit_condition": "condition_1_1800rpm_4000N",
        "normalization": "zscore",
        "feature_names": FEATURE_NAMES,
        "mean": {name: float(val) for name, val in zip(FEATURE_NAMES, normalizer["mean"])},
        "std": {name: float(val) for name, val in zip(FEATURE_NAMES, normalizer["std"])},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_pronostia_normalizer(config_path: Path) -> dict:
    """
    Charge les statistiques mean/std depuis configs/pronostia_normalizer.yaml.

    Parameters
    ----------
    config_path : Path
        Chemin vers pronostia_normalizer.yaml.

    Returns
    -------
    dict
        ``{"mean": np.ndarray[13], "std": np.ndarray[13]}``
    """
    cfg = load_config(str(config_path))
    if "mean" not in cfg or "std" not in cfg:
        raise ValueError(f"Le fichier normalizer ({config_path}) doit contenir 'mean' et 'std'.")

    mean = np.array([cfg["mean"][name] for name in FEATURE_NAMES], dtype=np.float32)
    std = np.array([cfg["std"][name] for name in FEATURE_NAMES], dtype=np.float32)
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# 4. Conversion en DataLoaders
# ---------------------------------------------------------------------------


def _features_to_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    val_ratio: float = VAL_RATIO,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Split stratifié (ou chronologique) et conversion en DataLoaders."""
    n = len(features)
    unique_classes = np.unique(labels)

    if len(unique_classes) >= 2 and int(n * val_ratio) >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(features, labels))
    else:
        n_val = max(1, int(n * val_ratio))
        train_idx = np.arange(n - n_val)
        val_idx = np.arange(n - n_val, n)

    # MEM: tenseurs train — n_train×13×4 B @ FP32
    x_train = torch.from_numpy(features[train_idx].astype(np.float32))
    y_train = torch.from_numpy(labels[train_idx].astype(np.float32)).unsqueeze(1)
    x_val = torch.from_numpy(features[val_idx].astype(np.float32))
    y_val = torch.from_numpy(labels[val_idx].astype(np.float32)).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=shuffle
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, len(train_idx), len(val_idx)


# ---------------------------------------------------------------------------
# 5. Interface principale CL (scénario by_condition)
# ---------------------------------------------------------------------------


def get_pronostia_dataloaders(
    npy_dir: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    failure_ratio: float = FAILURE_RATIO,
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL sur le Dataset 3 (PRONOSTIA).

    Pipeline :
        load_condition_features × 3 → load_pronostia_normalizer
        → apply Z-score → split stratifié → DataLoader

    Parameters
    ----------
    npy_dir : Path
        Répertoire contenant les fichiers 0.npy à 5.npy.
    normalizer_path : Path
        Chemin vers configs/pronostia_normalizer.yaml (doit exister).
    batch_size : int
        Taille des mini-batches. Default : 32.
    val_ratio : float
        Fraction validation par tâche. Default : VAL_RATIO (0.2).
    seed : int
        Seed global. Default : 42.
    window_size : int
        Taille de la fenêtre en points. Default : WINDOW_SIZE (2 560).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (2 560).
    failure_ratio : float
        Fraction terminale pré-défaillance. Default : FAILURE_RATIO (0.10).

    Returns
    -------
    list[dict]
        Liste ordonnée de 3 dicts (Condition 1 → 2 → 3) :

        .. code-block:: python

            {
                "task_id": int,          # 1, 2 ou 3
                "condition": int,        # condition opératoire (1, 2 ou 3)
                "domain": str,           # alias "condition_{N}" pour compatibilité
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    normalizer = load_pronostia_normalizer(normalizer_path)
    mean_vec = normalizer["mean"]
    std_vec = normalizer["std"]

    tasks: list[dict] = []

    for condition in range(1, N_CONDITIONS + 1):
        feats, lbls = load_condition_features(
            npy_dir, condition, window_size, step_size, failure_ratio
        )

        # Normalisation Z-score avec stats fixes (Condition 1)
        # MEM: X [N_windows, 13] × 4B = N_windows × 52 B @ FP32
        feats = (feats - mean_vec) / std_vec

        train_loader, val_loader, n_train, n_val = _features_to_dataloader(
            feats, lbls, batch_size, shuffle=True, seed=seed, val_ratio=val_ratio
        )

        tasks.append(
            {
                "task_id": condition,
                "condition": condition,
                "domain": f"condition_{condition}",
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_train": n_train,
                "n_val": n_val,
            }
        )

    return tasks


# ---------------------------------------------------------------------------
# 6. Interface single-task (baseline hors-CL)
# ---------------------------------------------------------------------------


def get_pronostia_dataloaders_single_task(
    npy_dir: Path,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    failure_ratio: float = FAILURE_RATIO,
) -> dict:
    """
    Baseline hors-CL : toutes les conditions fusionnées, split global stratifié.

    La normalisation Z-score est fittée sur le train split uniquement.
    Pas de découpage par condition — mesure la performance maximale atteignable.

    Parameters
    ----------
    npy_dir : Path
        Répertoire contenant les fichiers 0.npy à 5.npy.
    batch_size : int
        Taille des mini-batches. Default : 32.
    test_ratio : float
        Fraction test (sur total). Default : 0.2.
    val_ratio : float
        Fraction val (sur train uniquement). Default : 0.1.
    seed : int
        Seed. Default : 42.
    window_size : int
        Taille de la fenêtre. Default : WINDOW_SIZE (2 560).
    step_size : int
        Pas entre fenêtres. Default : STEP_SIZE (2 560).
    failure_ratio : float
        Fraction terminale pré-défaillance. Default : FAILURE_RATIO (0.10).

    Returns
    -------
    dict
        ``{"train_loader": DataLoader, "val_loader": DataLoader,
           "test_loader": DataLoader, "n_train": int, "n_val": int, "n_test": int}``
    """
    set_seed(seed)

    all_feats: list[np.ndarray] = []
    all_lbls: list[np.ndarray] = []

    for condition in range(1, N_CONDITIONS + 1):
        feats, lbls = load_condition_features(
            npy_dir, condition, window_size, step_size, failure_ratio
        )
        all_feats.append(feats)
        all_lbls.append(lbls)

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_lbls, axis=0)
    n_total = len(X)

    # Split stratifié train+val / test
    n_test = max(1, int(n_total * test_ratio))
    try:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        trainval_idx, test_idx = next(sss_test.split(X, y))
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        test_idx = perm[:n_test]
        trainval_idx = perm[n_test:]

    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Split stratifié train / val
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

    # Normalisation Z-score fittée sur X_train uniquement
    # MEM: mean/std [13] × 4 B = 52 B @ FP32 chacun
    mean_vec = X_train.mean(axis=0)
    std_vec = X_train.std(axis=0)
    std_vec[std_vec == 0] = 1.0

    X_train = (X_train - mean_vec) / std_vec
    X_val = (X_val - mean_vec) / std_vec
    X_test = (X_test - mean_vec) / std_vec

    def _make_loader(Xd: np.ndarray, yd: np.ndarray, shuffle: bool) -> DataLoader:
        xt = torch.from_numpy(Xd.astype(np.float32))  # MEM: N×13×4 B @ FP32
        yt = torch.from_numpy(yd.astype(np.float32)).unsqueeze(1)
        return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": _make_loader(X_train, y_train, shuffle=True),
        "val_loader": _make_loader(X_val, y_val, shuffle=False),
        "test_loader": _make_loader(X_test, y_test, shuffle=False),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
