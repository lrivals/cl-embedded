"""
cmapss_loader.py — Loader PyTorch pour CMAPSS (NASA Turbofan Degradation).

Scénario CL : Domain-Incremental
    Task 1 = FD001 → Task 2 = FD002 → Task 3 = FD003 → Task 4 = FD004

RUL binarisé : faulty = 1 si RUL ≤ 30 (seuil défaillance imminente).
RUL capping : cap = 125 cycles (défini dans CMAPSS_RUL_CAP).
Normalisation MinMax fixée sur FD001 uniquement.

Usage :
    from src.data.cmapss_loader import get_cl_dataloaders
    tasks = get_cl_dataloaders(data_dir=Path("data/raw/CMAPSS Jet Engine Simulated Data/"),
                               config_path=Path("configs/cmapss_config.yaml"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes — toutes ici, jamais en dur dans les fonctions
# ---------------------------------------------------------------------------

CMAPSS_RUL_CAP: int = 125
CMAPSS_FAULTY_THRESHOLD: int = 30  # RUL ≤ 30 → faulty = 1
CMAPSS_N_FEATURES_RAW: int = 21  # capteurs s1–s21
CMAPSS_N_FEATURES_SELECTED: int = 5

DATA_DIR_DEFAULT: Path = Path("data/raw/CMAPSS Jet Engine Simulated Data/")

SENSOR_NAMES: list[str] = [
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "Phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
]

DOMAIN_ORDER: list[str] = ["FD001", "FD002", "FD003", "FD004"]

LABEL_COL: str = "faulty"
VAL_RATIO: float = 0.2
FEATURE_SUBSET_PATH: Path = Path("configs/cmapss_feature_subset.yaml")


# ---------------------------------------------------------------------------
# 1. Chargement et calcul RUL
# ---------------------------------------------------------------------------


def _load_raw(
    data_dir: Path,
    subset: str,
    faulty_threshold: int = CMAPSS_FAULTY_THRESHOLD,
) -> pd.DataFrame:
    """
    Lit train_FDxxx.csv, calcule RUL cappé et binarise en faulty.

    Parameters
    ----------
    data_dir : Path
        Dossier contenant les fichiers CMAPSS.
    subset : str
        Identifiant du sous-dataset, ex. "FD001".
    faulty_threshold : int
        Seuil de binarisation : faulty = 1 si RUL <= seuil.
        Default : CMAPSS_FAULTY_THRESHOLD (30).

    Returns
    -------
    pd.DataFrame
        Colonnes : SENSOR_NAMES + ["faulty", "RUL", "unit_nr", "time_cycles"]
    """
    csv_path = data_dir / f"train_{subset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CMAPSS introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    # RUL par unité moteur : max_cycles - time_cycles
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    rul_raw = max_cycles - df["time_cycles"]

    # Capping + binarisation
    df["RUL"] = rul_raw.clip(upper=CMAPSS_RUL_CAP)
    df[LABEL_COL] = (df["RUL"] <= faulty_threshold).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Sélection de features par mutual info
# ---------------------------------------------------------------------------


def compute_feature_selection(
    data_dir: Path,
    n_features: int = CMAPSS_N_FEATURES_SELECTED,
    faulty_threshold: int = CMAPSS_FAULTY_THRESHOLD,
) -> list[str]:
    """
    Calcule le top-N capteurs par mutual_info_classif sur FD001 (fit uniquement).

    Sauvegarde le résultat dans configs/cmapss_feature_subset.yaml.

    Parameters
    ----------
    data_dir : Path
        Dossier CMAPSS.
    n_features : int
        Nombre de capteurs à sélectionner.

    Returns
    -------
    list[str]
        Noms des capteurs sélectionnés (ordre décroissant de MI).
    """
    df = _load_raw(data_dir, "FD001", faulty_threshold=faulty_threshold)

    X = df[SENSOR_NAMES].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy()

    mi_scores = mutual_info_classif(X, y, random_state=42)
    ranked = sorted(zip(SENSOR_NAMES, mi_scores), key=lambda t: t[1], reverse=True)
    selected = [name for name, _ in ranked[:n_features]]

    # Sauvegarde
    FEATURE_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_SUBSET_PATH, "w") as f:
        yaml.dump(
            {"features": selected, "n_features": n_features, "method": "mutual_info_classif"},
            f,
            default_flow_style=False,
        )

    return selected


def _load_feature_selection() -> list[str] | None:
    """Charge la sélection depuis configs/cmapss_feature_subset.yaml si présente."""
    if not FEATURE_SUBSET_PATH.exists():
        return None
    with open(FEATURE_SUBSET_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("features")


# ---------------------------------------------------------------------------
# 3. Normalisation MinMax (fit sur FD001 uniquement)
# ---------------------------------------------------------------------------


def _fit_minmax(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Calcule min/max sur un DataFrame (FD001 uniquement)."""
    return {
        "min": df[feature_names].min().to_dict(),
        "max": df[feature_names].max().to_dict(),
    }


def _apply_minmax(df: pd.DataFrame, feature_names: list[str], scaler: dict) -> pd.DataFrame:
    """Applique normalisation MinMax [0, 1], clip pour éviter out-of-range."""
    df_out = df.copy()
    for feat in feature_names:
        feat_min = scaler["min"][feat]
        feat_max = scaler["max"][feat]
        denom = feat_max - feat_min if feat_max != feat_min else 1.0
        df_out[feat] = ((df_out[feat] - feat_min) / denom).clip(0.0, 1.0)
    return df_out


# ---------------------------------------------------------------------------
# 4. Helpers labels + conversion DataFrame → tenseurs PyTorch
# ---------------------------------------------------------------------------


def _make_labels(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Produit les labels selon le mode choisi."""
    if mode == "rul":
        return df["RUL"].values.astype(np.float32)  # cappé à CMAPSS_RUL_CAP dans _load_raw
    return df[LABEL_COL].values.astype(np.float32)  # mode "binary"


def df_to_tensors(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit un DataFrame en tenseurs (X, y) en mode binaire.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        X : shape [N, n_features], float32
        y : shape [N, 1], float32

    # MEM: batch 32 → 32×5×4 = 640 B @ FP32 / 160 B @ INT8
    """
    x_np = df[feature_names].to_numpy(dtype=np.float32).copy()
    y_np = df[LABEL_COL].to_numpy(dtype=np.float32).reshape(-1, 1).copy()

    # MEM: X complet — N×5×4 B @ FP32 / N×5×1 B @ INT8
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


# ---------------------------------------------------------------------------
# 5. Interface principale
# ---------------------------------------------------------------------------


def get_cl_dataloaders(
    data_dir: Path,
    config_path: Path,
    feature_names: list[str] | None = None,
    mode: Literal["binary", "rul"] = "binary",
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL sur CMAPSS.

    Pipeline :
        _load_raw (×4) → compute_feature_selection (FD001) → _fit_minmax (FD001)
        → [_apply_minmax + train_test_split + DataLoader] × 4

    Parameters
    ----------
    data_dir : Path
        Dossier contenant les fichiers CMAPSS.
    config_path : Path
        Chemin vers configs/cmapss_config.yaml.
    feature_names : list[str] | None
        Features à utiliser. Si None, chargé depuis cmapss_feature_subset.yaml
        ou calculé par mutual_info_classif sur FD001.
    mode : {"binary", "rul"}
        "binary" (défaut) : labels 0/1 (RUL ≤ 30).
        "rul" : RUL continu cappé à CMAPSS_RUL_CAP (float32).

    Returns
    -------
    list[dict]
        Liste ordonnée de 4 dicts (FD001 → FD004) :

        .. code-block:: python

            {
                "task_id": int,
                "domain": str,           # "FD001" … "FD004"
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
                "class_weights": torch.Tensor | None,  # None en mode "rul"
            }
    """
    cfg = load_config(str(config_path))
    seed: int = cfg.get("training", {}).get("seed", 42)
    batch_size: int = cfg.get("training", {}).get("batch_size", 32)
    val_ratio: float = cfg.get("data", {}).get("test_split", VAL_RATIO)
    faulty_threshold: int = cfg.get("data", {}).get(
        "faulty_threshold", CMAPSS_FAULTY_THRESHOLD
    )

    set_seed(seed)

    # Résolution des features
    if feature_names is None:
        feature_names = _load_feature_selection()
    if feature_names is None:
        feature_names = compute_feature_selection(
            data_dir, faulty_threshold=faulty_threshold
        )

    # Fit du scaler sur FD001 uniquement — pas de fuite inter-tâches
    df_fd001 = _load_raw(data_dir, "FD001", faulty_threshold=faulty_threshold)
    scaler = _fit_minmax(df_fd001, feature_names)

    tasks: list[dict] = []

    for task_id, domain in enumerate(DOMAIN_ORDER, start=1):
        df = _load_raw(data_dir, domain, faulty_threshold=faulty_threshold)
        df = _apply_minmax(df, feature_names, scaler)

        # Stratification uniquement en mode binaire (labels continus incompatibles)
        stratify_col = df[LABEL_COL] if mode == "binary" else None
        df_train, df_val = train_test_split(
            df,
            test_size=val_ratio,
            stratify=stratify_col,
            random_state=seed,
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)

        x_train_np = df_train[feature_names].to_numpy(dtype=np.float32)
        x_val_np = df_val[feature_names].to_numpy(dtype=np.float32)
        y_train_np = _make_labels(df_train, mode).reshape(-1, 1)
        y_val_np = _make_labels(df_val, mode).reshape(-1, 1)

        # MEM: X complet — N×5×4 B @ FP32 / N×5×1 B @ INT8
        x_train = torch.from_numpy(x_train_np)
        y_train = torch.from_numpy(y_train_np)
        x_val = torch.from_numpy(x_val_np)
        y_val = torch.from_numpy(y_val_np)

        # Poids de classe uniquement en mode binaire
        if mode == "binary":
            classes = np.array([0, 1])
            weights = compute_class_weight(
                "balanced",
                classes=classes,
                y=df_train[LABEL_COL].to_numpy(),
            )
            class_weights: torch.Tensor | None = torch.tensor(weights, dtype=torch.float32)
        else:
            class_weights = None

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
                "class_weights": class_weights,
            }
        )

    return tasks
