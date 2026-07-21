"""
electricity_dataset.py — Loader du dataset Electricity / ELEC2 (Sprint 43, S4302).

Concept drift **classique supervisé** (Harries 1999 ; Gama 2004) : 45 312 instances du marché
de l'électricité NSW (mai 1996 → déc. 1998), label binaire ``class`` (UP/DOWN du prix vs moyenne
mobile 24 h). Le drift est établi dans la littérature mais **sans points ponctuels** →
``drift_points = None`` (ground-truth structurelle/absente). Seules les métriques de stabilité /
FAR (S44) sont calculables ; le délai de détection ne l'est pas — champ honnête.

- ``X`` : colonnes numériques (hors ``date``, ``class``), déjà normalisées [0,1] par Bifet.
- ``y`` : ``class`` (UP → 1, DOWN → 0).
- ``drift_points`` : ``None``.
- ``drift_type`` : ``"gradual"`` (documenté).

Source : data/raw/The Elec2 Dataset/electricity-normalized.csv
Licence : domaine public / usage recherche (aucune restriction citée).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.drift_dataset import DEFAULT_SEED, DriftDataset, freeze_zscore
from src.utils.config_loader import load_config

# Colonnes exclues des features (identifiant temporel + label).
DEFAULT_EXCLUDE: list[str] = ["date", "class"]

# Colonne de label.
LABEL_COLUMN: str = "class"


def load(config_path: str) -> DriftDataset:
    """
    Charge Electricity / ELEC2 en ``DriftDataset``.

    Parameters
    ----------
    config_path : str
        Chemin vers ``configs/electricity_drift_config.yaml``.

    Returns
    -------
    DriftDataset
        ``X`` [N, d], ``y`` binaire, ``drift_points = None``, ``drift_type = "gradual"``.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    csv_path = Path(data_cfg["raw_path"])
    exclude = data_cfg.get("exclude_columns", DEFAULT_EXCLUDE)
    label_column = data_cfg.get("label_column", LABEL_COLUMN)
    normalization = data_cfg.get("normalization", "none")
    drift_type = data_cfg.get("drift_type", "gradual")
    seed = int(cfg.get("seed", DEFAULT_SEED))

    np.random.seed(seed)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}\n"
            f"Lancer : python scripts/download_drift_datasets.py --dataset electricity"
        )

    df = pd.read_csv(csv_path)
    # Ordre chronologique préservé (le CSV est déjà trié temporellement).

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    y_raw = df[label_column]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(np.int64).to_numpy()
    else:
        # Labels textuels (UP/DOWN) → 1/0.
        y = (y_raw.astype(str).str.upper() == "UP").astype(np.int64).to_numpy()

    # Données déjà normalisées [0,1] par Bifet → pas de re-fit destructeur par défaut.
    if normalization == "zscore":
        X, _, _ = freeze_zscore(X, (0, len(X)))
        norm_meta = {"method": "zscore_global"}
    else:
        norm_meta = {"method": "none (pre-normalized by Bifet)"}

    return DriftDataset(
        X=X,
        y=y,
        drift_points=None,  # pas de ground-truth ponctuelle — honnête
        drift_type=drift_type,
        feature_names=feature_cols,
        segments=[(0, len(X))],
        metadata={
            "dataset": "electricity",
            "source": "ELEC2 (Harries 1999; Gama 2004; normalized by Bifet)",
            "license": "public / research use",
            "ground_truth": "none (structural drift established in literature, no exact points)",
            "label_column": label_column,
            "normalization": norm_meta,
            "config_snapshot": cfg,
        },
    )
