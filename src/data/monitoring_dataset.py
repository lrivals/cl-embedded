"""
monitoring_dataset.py — Loader PyTorch pour le Dataset 2 (Industrial Equipment Monitoring).

Scénario CL : Domain-Incremental
    Task 1 = Pump  →  Task 2 = Turbine  →  Task 3 = Compressor

Tailles mesurées (source : notebooks/01_data_exploration.ipynb) :
    Pump: 2534, Turbine: 2565, Compressor: 2573  — Total: 7672

Normalisation Z-score fixée sur Task 1 (Pump) uniquement,
chargée depuis configs/monitoring_normalizer.yaml.
Aucun re-fit en ligne pour éviter la fuite d'information inter-tâches.

Usage :
    from pathlib import Path
    from src.data.monitoring_dataset import get_cl_dataloaders

    tasks = get_cl_dataloaders(
        csv_path=Path("data/raw/equipment_monitoring/.../equipment_anomaly_data.csv"),
        normalizer_path=Path("configs/monitoring_normalizer.yaml"),
    )
    # tasks[0] = {"task_id": 1, "domain": "Pump", "train_loader": ..., "val_loader": ..., ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

# ---------------------------------------------------------------------------
# Constantes de configuration — toutes les valeurs sont ici, jamais en dur
# ---------------------------------------------------------------------------

# Ordre des domaines CL — conforme à CLAUDE.md et docs/models/ewc_mlp_spec.md
DOMAIN_ORDER: list[str] = ["Pump", "Turbine", "Compressor"]

# Features numériques à normaliser et à passer au modèle
NUMERIC_FEATURES: list[str] = ["temperature", "pressure", "vibration", "humidity"]

# Colonne définissant le domaine CL (exclue de X, utilisée pour le split)
DOMAIN_FEATURE: str = "equipment"

# Label binaire de défaut
LABEL_COL: str = "faulty"

# Ratio validation par tâche (stratifié sur faulty)
VAL_RATIO: float = 0.2

# Tailles par domaine (source : notebooks/01_data_exploration.ipynb)
DOMAIN_SIZES: dict[str, int] = {"Pump": 2534, "Turbine": 2565, "Compressor": 2573}

# Mapping ordinal fixe pour l'encodage catégoriel
EQUIPMENT_ENCODING: dict[str, int] = {"Pump": 0, "Turbine": 1, "Compressor": 2}


# ---------------------------------------------------------------------------
# 1. Chargement et validation du CSV brut
# ---------------------------------------------------------------------------


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Charge le CSV brut et valide les colonnes et labels attendus.

    Parameters
    ----------
    csv_path : Path
        Chemin vers equipment_anomaly_data.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame brut validé.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas.
    ValueError
        Si colonnes manquantes, labels inattendus, ou domaines manquants.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    # Validation des colonnes obligatoires
    required_cols = NUMERIC_FEATURES + [DOMAIN_FEATURE, LABEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    # Validation des labels binaires
    unexpected_labels = set(df[LABEL_COL].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            f"Labels inattendus dans '{LABEL_COL}' : {unexpected_labels}. "
            f"Attendu : {{0, 1}} uniquement."
        )

    # Validation de la présence des trois domaines
    present_domains = set(df[DOMAIN_FEATURE].unique())
    missing_domains = set(DOMAIN_ORDER) - present_domains
    if missing_domains:
        raise ValueError(
            f"Domaines manquants dans '{DOMAIN_FEATURE}' : {missing_domains}\n"
            f"Domaines présents : {present_domains}"
        )

    return df


# ---------------------------------------------------------------------------
# 2. Chargement du normaliseur
# ---------------------------------------------------------------------------


def load_normalizer(config_path: Path) -> dict:
    """
    Charge les statistiques mean/std depuis configs/monitoring_normalizer.yaml.

    Les stats sont calculées sur Task 1 (Pump) uniquement (S1-02).
    Ce module ne recalcule jamais ces valeurs.

    Parameters
    ----------
    config_path : Path
        Chemin vers monitoring_normalizer.yaml.

    Returns
    -------
    dict
        Dictionnaire avec clés ``mean`` et ``std``,
        chacun mappant les noms de features vers des floats.

    Raises
    ------
    ValueError
        Si les clés ``mean`` ou ``std`` sont absentes du fichier YAML.
    """
    cfg = load_config(str(config_path))

    if "mean" not in cfg or "std" not in cfg:
        raise ValueError(
            f"Le fichier normalizer ({config_path}) doit contenir 'mean' et 'std'."
        )

    return {"mean": cfg["mean"], "std": cfg["std"]}


# ---------------------------------------------------------------------------
# 3. Normalisation Z-score (sans re-fit)
# ---------------------------------------------------------------------------


def normalize_features(df: pd.DataFrame, normalizer: dict) -> pd.DataFrame:
    """
    Applique la normalisation Z-score sur NUMERIC_FEATURES.

    Les statistiques proviennent exclusivement de ``normalizer`` (Task 1 — Pump).
    Aucun calcul de mean/std n'est effectué ici pour éviter la fuite d'information.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut contenant au moins les colonnes NUMERIC_FEATURES.
    normalizer : dict
        Dictionnaire ``{mean: {feat: float}, std: {feat: float}}``
        tel que retourné par :func:`load_normalizer`.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame avec NUMERIC_FEATURES normalisées.
    """
    df_out = df.copy()
    mean = normalizer["mean"]
    std = normalizer["std"]

    for feat in NUMERIC_FEATURES:
        df_out[feat] = (df_out[feat] - mean[feat]) / std[feat]

    return df_out


# ---------------------------------------------------------------------------
# 4. Encodage des features catégorielles
# ---------------------------------------------------------------------------


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode la colonne ``equipment`` en label encoding ordinal.

    Mapping fixe (ordinal, pas one-hot) :
        Pump → 0, Turbine → 1, Compressor → 2

    La colonne originale ``equipment`` est conservée pour le split par domaine.
    La colonne encodée ``equipment_encoded`` est ajoutée mais exclue de X
    (``df_to_tensors`` utilise uniquement NUMERIC_FEATURES).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la colonne ``equipment``.

    Returns
    -------
    pd.DataFrame
        Copie avec la colonne ``equipment_encoded`` ajoutée.

    Notes
    -----
    TODO(arnaud) : confirmer si ``location`` doit être incluse comme feature.
    """
    df_out = df.copy()
    df_out["equipment_encoded"] = df_out[DOMAIN_FEATURE].map(EQUIPMENT_ENCODING)
    return df_out


# ---------------------------------------------------------------------------
# 5. Split par tâche CL
# ---------------------------------------------------------------------------


def get_task_split(
    df: pd.DataFrame,
    domain: str,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne les sous-DataFrames train/val pour un domaine donné.

    Le split est stratifié sur ``faulty`` pour conserver le taux ~10% de défaut.
    Seed fixe pour reproductibilité.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet (tous domaines, après normalisation et encodage).
    domain : str
        Domaine à extraire (ex. ``"Pump"``).
    val_ratio : float
        Fraction réservée à la validation. Default : VAL_RATIO (0.2).
    seed : int
        Seed pour le split. Default : 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_val) — aucune fuite entre tâches.

    Raises
    ------
    ValueError
        Si le domaine est absent du DataFrame.
    """
    df_domain = df[df[DOMAIN_FEATURE] == domain]

    if df_domain.empty:
        raise ValueError(f"Domaine '{domain}' absent du DataFrame.")

    df_train, df_val = train_test_split(
        df_domain,
        test_size=val_ratio,
        stratify=df_domain[LABEL_COL],
        random_state=seed,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Conversion DataFrame → tenseurs PyTorch
# ---------------------------------------------------------------------------


def df_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit un DataFrame en tenseurs (X, y).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame d'un domaine (train ou val), après normalisation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - X : shape [N, 4], dtype float32 — NUMERIC_FEATURES uniquement.
        - y : shape [N, 1], dtype float32 — label binaire faulty.

    Notes
    -----
    ``equipment`` et ``equipment_encoded`` sont exclus de X.
    Input dim final = 4 (confirmé → configs/ewc_config.yaml model.input_dim).

    # MEM: batch de 32 → 32×4×4 = 512 B @ FP32 / 128 B @ INT8
    """
    x_np: np.ndarray = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32).copy()
    y_np: np.ndarray = df[LABEL_COL].to_numpy(dtype=np.float32).reshape(-1, 1).copy()

    # MEM: X tenseur complet — N×4×4 B @ FP32 / N×4×1 B @ INT8
    x_tensor = torch.from_numpy(x_np)
    # MEM: y tenseur complet — N×1×4 B @ FP32 / N×1×1 B @ INT8
    y_tensor = torch.from_numpy(y_np)

    return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# 7. Interface principale
# ---------------------------------------------------------------------------


def get_cl_dataloaders(
    csv_path: Path,
    normalizer_path: Path,
    batch_size: int = 32,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> list[dict]:
    """
    Point d'entrée principal pour l'entraînement CL sur le Dataset 2.

    Pipeline :
        load_raw_dataset → load_normalizer → normalize_features
        → encode_categoricals → [get_task_split + df_to_tensors + DataLoader] × 3

    Parameters
    ----------
    csv_path : Path
        Chemin vers equipment_anomaly_data.csv.
    normalizer_path : Path
        Chemin vers configs/monitoring_normalizer.yaml.
    batch_size : int
        Taille des mini-batches. Default : 32.
    val_ratio : float
        Fraction de validation par tâche. Default : VAL_RATIO (0.2).
    seed : int
        Seed global. Default : 42.

    Returns
    -------
    list[dict]
        Liste ordonnée de 3 dicts (Pump → Turbine → Compressor) :

        .. code-block:: python

            {
                "task_id": int,          # 1, 2 ou 3
                "domain": str,           # "Pump", "Turbine" ou "Compressor"
                "train_loader": DataLoader,
                "val_loader": DataLoader,
                "n_train": int,
                "n_val": int,
            }
    """
    set_seed(seed)

    # Chargement et validation
    df = load_raw_dataset(csv_path)
    normalizer = load_normalizer(normalizer_path)

    # Prétraitement global (normalisation + encodage)
    df = normalize_features(df, normalizer)
    df = encode_categoricals(df)

    tasks: list[dict] = []

    for task_id, domain in enumerate(DOMAIN_ORDER, start=1):
        df_train, df_val = get_task_split(df, domain, val_ratio=val_ratio, seed=seed)

        x_train, y_train = df_to_tensors(df_train)
        x_val, y_val = df_to_tensors(df_val)

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
