"""
paderborn_loader.py — Loader PyTorch pour Paderborn Bearing Dataset (KAt-DataCenter).

Scénario CL : Domain-Incremental
    Task 1 = K001 (healthy)        → label 0
    Task 2 = KA04 (outer-race fault) → label 1
    Task 3 = KI04 (inner-race fault)  → label 1

Feature engineering : FFT window-based (rms, kurtosis, crest_factor, energy_band_1..4)
Feature selection : top-5 mutual info (fit sur K001+KA04 uniquement).

Structure des .mat :
    mat[key][0,0]['X'][0, channel_idx]['Data'][0]  → signal 1D
    Channel 0 : Mech_4kHz   (16 007 samples, mécanique)
    Channel 1 : HostService  (256 608 samples, vibration @ ~64 kHz)  ← utilisé
    Channel 2 : Temp_1Hz     (5 samples, température)

Usage :
    from pathlib import Path
    from src.data.paderborn_loader import get_cl_dataloaders
    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/"),
        config_path=Path("configs/paderborn_config.yaml"),
    )
    # tasks[0] = {"task_id": 1, "domain": "K001", "train_loader": ..., "val_loader": ..., ...}

Référence : Lessmeier et al., 2016 — KAt-DataCenter Paderborn University.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io
import torch
import yaml
from numpy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PADERBORN_WINDOW_SIZE: int = 1024
PADERBORN_OVERLAP: float = 0.5
PADERBORN_SAMPLING_RATE: int = 64_000  # Hz
PADERBORN_N_FEATURES_RAW: int = 7  # rms + kurtosis + crest + 4 bandes
PADERBORN_N_FEATURES_SELECTED: int = 5

# Bandes fréquentielles (Hz)
FREQ_BANDS: list[tuple[float, float]] = [
    (0, 1_000),
    (1_000, 2_000),
    (2_000, 5_000),
    (5_000, 10_000),
]

FEATURE_NAMES_RAW: list[str] = [
    "rms",
    "kurtosis",
    "crest_factor",
    "energy_band_1",
    "energy_band_2",
    "energy_band_3",
    "energy_band_4",
]

# Scénario CL domain-incremental
DOMAIN_ORDER: list[str] = ["K001", "KA04", "KI04"]
DOMAIN_LABELS: dict[str, int] = {"K001": 0, "KA04": 1, "KI04": 1}  # mode binary

# Mapping multiclass : 3 états bearing distincts
PADERBORN_MULTICLASS_MAP: dict[str, int] = {
    "K001": 0,  # sain (healthy)
    "KA04": 1,  # outer-race fault
    "KI04": 2,  # inner-race fault
}
N_CLASSES_MULTICLASS: int = 3

VAL_RATIO: float = 0.2

# Index du canal vibration dans les .mat Paderborn
_VIBRATION_CHANNEL_IDX: int = 1


# ---------------------------------------------------------------------------
# 1. Lecture d'un fichier .mat
# ---------------------------------------------------------------------------


def _load_mat_vibration(mat_path: Path) -> np.ndarray:
    """
    Charge un fichier .mat Paderborn et extrait le signal de vibration.

    Parameters
    ----------
    mat_path : Path
        Chemin vers le fichier .mat.

    Returns
    -------
    np.ndarray
        Signal 1D float32, shape (N_samples,). Typiquement ~256 608 points @ 64 kHz.

    Notes
    -----
    # MEM: signal brut 256608 × 4 B @ FP32 ≈ 1 026 Ko
    """
    mat = scipy.io.loadmat(str(mat_path))
    # La clé de données est le nom du fichier sans extension
    key = [k for k in mat.keys() if not k.startswith("_")][0]
    data = mat[key][0, 0]
    # Channel 1 = HostService = vibration @ ~64 kHz
    signal = data["X"][0, _VIBRATION_CHANNEL_IDX]["Data"][0].astype(np.float32)
    return signal


# ---------------------------------------------------------------------------
# 2. Fenêtrage
# ---------------------------------------------------------------------------


def _extract_windows(
    signal: np.ndarray,
    window_size: int = PADERBORN_WINDOW_SIZE,
    overlap: float = PADERBORN_OVERLAP,
) -> np.ndarray:
    """
    Découpe un signal 1D en fenêtres chevauchantes.

    Parameters
    ----------
    signal : np.ndarray
        Signal 1D, shape (N_samples,).
    window_size : int
        Nombre de points par fenêtre.
    overlap : float
        Fraction de recouvrement (0.5 → 50 %).

    Returns
    -------
    np.ndarray
        Shape [N_windows, window_size], dtype float32.

    Notes
    -----
    # MEM: fenêtre 1024 × 4 B @ FP32 = 4 096 B
    """
    step = int(window_size * (1.0 - overlap))
    n_total = len(signal)
    starts = range(0, n_total - window_size + 1, step)
    # MEM: toutes fenêtres N×1024×4 B @ FP32
    windows = np.stack([signal[s : s + window_size] for s in starts], axis=0)
    return windows


# ---------------------------------------------------------------------------
# 3. Extraction de features
# ---------------------------------------------------------------------------


def _compute_features(windows: np.ndarray, fs: int = PADERBORN_SAMPLING_RATE) -> np.ndarray:
    """
    Calcule les 7 features time-freq par fenêtre.

    Parameters
    ----------
    windows : np.ndarray
        Shape [N_windows, window_size], float32.
    fs : int
        Fréquence d'échantillonnage en Hz.

    Returns
    -------
    np.ndarray
        Shape [N_windows, 7] : [rms, kurtosis, crest_factor, eb1, eb2, eb3, eb4].

    Notes
    -----
    # MEM: features N×7×4 B @ FP32 / N×7×1 B @ INT8
    """
    n_windows, window_size = windows.shape
    # MEM: features N×7×4 B @ FP32
    features = np.empty((n_windows, PADERBORN_N_FEATURES_RAW), dtype=np.float32)

    freqs = rfftfreq(window_size, d=1.0 / fs)  # shape (window_size//2 + 1,)

    for i, w in enumerate(windows):
        # --- time-domain ---
        rms = np.sqrt(np.mean(w**2))
        kurt = float(scipy_kurtosis(w, fisher=True))
        peak = np.max(np.abs(w))
        crest = peak / (rms + 1e-12)

        # --- frequency-domain energy bands ---
        # MEM: spectrum (window_size/2+1)×8 B @ complex128
        spectrum = np.abs(rfft(w)) ** 2  # puissance spectrale
        energy_bands = []
        for f_lo, f_hi in FREQ_BANDS:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            energy_bands.append(float(spectrum[mask].sum()))

        features[i] = [rms, kurt, crest, *energy_bands]

    return features


# ---------------------------------------------------------------------------
# 4. Chargement d'une condition complète
# ---------------------------------------------------------------------------


def _load_condition(
    data_dir: Path,
    condition: str,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge tous les .mat d'une condition, extrait features et labels.

    Parameters
    ----------
    data_dir : Path
        Répertoire racine contenant K001/, KA04/, KI04/.
    condition : str
        Nom de la condition (ex. "K001").
    max_files : int | None
        Limite le nombre de fichiers .mat chargés. None = tous.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - X : shape [N_total, 7], float32
        - y : shape [N_total], float32 — label binaire

    Notes
    -----
    # MEM: cumul features toutes fenêtres toute condition
    """
    condition_dir = data_dir / condition
    mat_files = sorted(condition_dir.glob("*.mat"))
    if max_files is not None:
        mat_files = mat_files[:max_files]

    if not mat_files:
        raise FileNotFoundError(f"Aucun .mat trouvé dans {condition_dir}")

    label = float(DOMAIN_LABELS[condition])
    all_features: list[np.ndarray] = []

    for mat_path in mat_files:
        signal = _load_mat_vibration(mat_path)
        windows = _extract_windows(signal)
        features = _compute_features(windows)
        all_features.append(features)

    X = np.concatenate(all_features, axis=0)
    y = np.full(len(X), label, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# 5. Sélection de features par mutual information
# ---------------------------------------------------------------------------


def compute_feature_selection(
    data_dir: Path,
    n_features: int = PADERBORN_N_FEATURES_SELECTED,
    max_files: int | None = 5,
    output_path: Path | None = None,
) -> list[str]:
    """
    Sélectionne les top-N features par mutual info (fit sur K001 + KA04).

    Sauvegarde le résultat dans ``configs/paderborn_feature_subset.yaml``.

    Parameters
    ----------
    data_dir : Path
        Répertoire racine du dataset Paderborn.
    n_features : int
        Nombre de features à sélectionner.
    max_files : int | None
        Nombre de .mat par condition pour le fit (réduit le temps).
    output_path : Path | None
        Chemin de sauvegarde du YAML. Défaut : configs/paderborn_feature_subset.yaml.

    Returns
    -------
    list[str]
        Noms des top-N features (dans l'ordre decroissant de MI).
    """
    X_k001, y_k001 = _load_condition(data_dir, "K001", max_files=max_files)
    X_ka04, y_ka04 = _load_condition(data_dir, "KA04", max_files=max_files)

    X_fit = np.concatenate([X_k001, X_ka04], axis=0)
    y_fit = np.concatenate([y_k001, y_ka04], axis=0).astype(int)

    mi_scores = mutual_info_classif(X_fit, y_fit, random_state=42)
    ranked_indices = np.argsort(mi_scores)[::-1]

    selected = [FEATURE_NAMES_RAW[i] for i in ranked_indices[:n_features]]
    all_ranked = [FEATURE_NAMES_RAW[i] for i in ranked_indices]
    scores_ranked = [float(mi_scores[i]) for i in ranked_indices]

    if output_path is None:
        output_path = Path("configs/paderborn_feature_subset.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            {
                "selected_features": selected,
                "all_features_ranked": all_ranked,
                "mi_scores": dict(zip(all_ranked, scores_ranked)),
            },
            f,
            default_flow_style=False,
        )

    return selected


# ---------------------------------------------------------------------------
# 6. Normalisation MinMax
# ---------------------------------------------------------------------------


def _fit_normalizer(
    X_train: np.ndarray,
    feature_names: list[str],
    output_path: Path | None = None,
) -> dict:
    """
    Fit un MinMaxScaler sur X_train (Task 1 = K001 uniquement) et sauvegarde en YAML.

    Parameters
    ----------
    X_train : np.ndarray
        Shape [N, n_features], float32.
    feature_names : list[str]
        Noms des features (dans l'ordre des colonnes de X_train).
    output_path : Path | None
        Chemin de sauvegarde. Défaut : configs/paderborn_normalizer.yaml.

    Returns
    -------
    dict
        Dictionnaire {feature: {min, max}}.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    normalizer = {
        name: {"min": float(scaler.data_min_[i]), "max": float(scaler.data_max_[i])}
        for i, name in enumerate(feature_names)
    }

    if output_path is None:
        output_path = Path("configs/paderborn_normalizer.yaml")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(normalizer, f, default_flow_style=False)

    return normalizer


def _apply_normalizer(X: np.ndarray, normalizer: dict, feature_names: list[str]) -> np.ndarray:
    """Applique la normalisation MinMax définie par le dictionnaire normalizer."""
    X_norm = X.copy()
    for i, name in enumerate(feature_names):
        lo = normalizer[name]["min"]
        hi = normalizer[name]["max"]
        denom = hi - lo if (hi - lo) > 1e-12 else 1.0
        X_norm[:, i] = (X[:, i] - lo) / denom
    return X_norm


# ---------------------------------------------------------------------------
# 7. Interface principale
# ---------------------------------------------------------------------------


def get_cl_dataloaders(
    data_dir: Path,
    config_path: Path,
    feature_names: list[str] | None = None,
    mode: Literal["binary", "multiclass"] = "binary",
) -> list[dict]:
    """
    Point d'entrée principal — même interface que monitoring_dataset.get_cl_dataloaders().

    Pipeline :
        _load_condition × 3 → feature selection → normalizer → split → DataLoader

    Parameters
    ----------
    data_dir : Path
        Répertoire racine contenant K001/, KA04/, KI04/.
    config_path : Path
        Chemin vers configs/paderborn_config.yaml.
    feature_names : list[str] | None
        Surcharge la sélection de features. None = lire depuis paderborn_feature_subset.yaml
        ou recalculer.
    mode : {"binary", "multiclass"}
        "binary" (défaut) : labels 0/1 (DOMAIN_LABELS).
        "multiclass" : labels 0/1/2 (PADERBORN_MULTICLASS_MAP), dtype int64.

    Returns
    -------
    list[dict]
        Liste de 3 dicts (K001 → KA04 → KI04) :

        .. code-block:: python

            {
                "task_id": int,
                "domain": str,
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed: int = cfg.get("training", {}).get("seed", 42)
    batch_size: int = cfg.get("training", {}).get("batch_size", 32)
    test_split: float = cfg.get("data", {}).get("test_split", VAL_RATIO)
    max_files: int | None = cfg.get("data", {}).get("max_files_per_condition", None)
    n_features_sel: int = cfg.get("data", {}).get(
        "n_features_selected", PADERBORN_N_FEATURES_SELECTED
    )
    feature_subset_path = Path(
        cfg.get("data", {}).get("feature_subset_path", "configs/paderborn_feature_subset.yaml")
    )
    normalizer_path = Path(
        cfg.get("data", {}).get("normalizer_path", "configs/paderborn_normalizer.yaml")
    )

    set_seed(seed)

    # --- Feature selection ---
    if feature_names is None:
        if feature_subset_path.exists():
            with open(feature_subset_path) as f:
                subset_cfg = yaml.safe_load(f)
            feature_names = subset_cfg["selected_features"]
        else:
            feature_names = compute_feature_selection(
                data_dir,
                n_features=n_features_sel,
                max_files=max_files if max_files is not None else 5,
                output_path=feature_subset_path,
            )

    feat_indices = [FEATURE_NAMES_RAW.index(fn) for fn in feature_names]

    # --- Charger toutes les conditions ---
    condition_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in DOMAIN_ORDER:
        X_raw, y = _load_condition(data_dir, cond, max_files=max_files)
        condition_data[cond] = (X_raw[:, feat_indices], y)

    # --- Normalizer fitté sur K001 uniquement ---
    # Le cache n'est réutilisable que s'il couvre TOUTES les features demandées : un cache
    # fitté sur un sous-ensemble différent (ex. top-5) ne contient pas toutes les clés natives
    # (rms, crest_factor…) et provoquerait un KeyError dans _apply_normalizer. Sinon → refit.
    normalizer = None
    if normalizer_path.exists():
        with open(normalizer_path) as f:
            cached = yaml.safe_load(f) or {}
        if all(name in cached for name in feature_names):
            normalizer = cached
    if normalizer is None:
        X_k001, _ = condition_data["K001"]
        normalizer = _fit_normalizer(X_k001, feature_names, output_path=normalizer_path)

    rng = np.random.default_rng(seed)
    tasks: list[dict] = []

    for task_id, domain in enumerate(DOMAIN_ORDER, start=1):
        X_sel, y_binary = condition_data[domain]
        X_norm = _apply_normalizer(X_sel, normalizer, feature_names).astype(np.float32)

        # Labels selon le mode
        if mode == "multiclass":
            y = np.full(len(X_norm), PADERBORN_MULTICLASS_MAP[domain], dtype=np.int64)
        else:
            y = y_binary  # float32, 0.0 ou 1.0

        # Split stratifié (ou aléatoire si classes uniques)
        unique_labels = np.unique(y)
        if len(unique_labels) > 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
            train_idx, val_idx = next(splitter.split(X_norm, y))
        else:
            idx = rng.permutation(len(X_norm))
            split_at = int(len(idx) * (1.0 - test_split))
            train_idx, val_idx = idx[:split_at], idx[split_at:]

        x_train = torch.from_numpy(X_norm[train_idx])
        x_val = torch.from_numpy(X_norm[val_idx])

        if mode == "multiclass":
            # MEM: labels int64 N×8 B
            y_train = torch.from_numpy(y[train_idx])  # shape [N], int64
            y_val = torch.from_numpy(y[val_idx])
        else:
            y_train = torch.from_numpy(y[train_idx]).reshape(-1, 1)  # shape [N, 1], float32
            y_val = torch.from_numpy(y[val_idx]).reshape(-1, 1)

        # MEM: batch 32×5×4 B @ FP32 = 640 B
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

        tasks.append(
            {
                "task_id": task_id,
                "domain": domain,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "n_train": len(x_train),
                "n_val": len(x_val),
            }
        )

    return tasks
