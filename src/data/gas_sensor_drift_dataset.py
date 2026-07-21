"""
gas_sensor_drift_dataset.py — Loader du Gas Sensor Array Drift Dataset (Sprint 43, S4302).

Dataset ⭐ dual-usage à drift capteur **réel** (UCI, Vergara 2012) : 13 910 mesures de
16 capteurs chimiques (128 features), 6 gaz, collectées sur 36 mois. La dérive de capteur
entre les **10 batches temporels** est la cible d'étude originale du dataset → les frontières
de batches constituent la **ground-truth structurelle** de drift.

- ``X`` : 128 features (16 capteurs × 8 descripteurs), Z-score figé sur le batch 1.
- ``y`` : identité du gaz ∈ {1..6} (dual-usage — tâche de « faute »/état supervisée).
- ``drift_points`` : offsets cumulés des frontières de batches (structurel).
- ``drift_type`` : ``"incremental"`` (dérive continue de capteur).

Format brut (libsvm) : ``label idx:val idx:val …`` par ligne, un fichier ``batchN.dat`` par batch.

Source : data/raw/Gas Sensor Array Drift Dataset/Dataset/batch{1..10}.dat
Licence : recherche uniquement (usage commercial exclu).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.drift_dataset import (
    DEFAULT_SEED,
    DriftDataset,
    freeze_zscore,
    segments_to_drift_points,
)
from src.utils.config_loader import load_config

# Nombre de batches temporels (vérité-terrain structurelle de drift).
N_BATCHES: int = 10

# Nombre de features par mesure (16 capteurs × 8 descripteurs).
N_FEATURES: int = 128


def _parse_libsvm_dat(path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse un fichier ``batchN.dat`` au format libsvm ``label idx:val …``.

    Parameters
    ----------
    path : Path
        Chemin du fichier batch.
    n_features : int
        Nombre de features attendu (128).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X [n, n_features] float32, y [n] int)``.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(int(float(parts[0])))
            vec = np.zeros(n_features, dtype=np.float32)
            for token in parts[1:]:
                idx_str, val_str = token.split(":")
                # index libsvm 1-based → 0-based
                vec[int(idx_str) - 1] = np.float32(val_str)
            rows.append(vec)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load(config_path: str) -> DriftDataset:
    """
    Charge le Gas Sensor Array Drift Dataset en ``DriftDataset``.

    Parameters
    ----------
    config_path : str
        Chemin vers ``configs/gas_sensor_drift_config.yaml``.

    Returns
    -------
    DriftDataset
        ``X`` [N, 128], ``y`` [N] (gaz 1..6), ``drift_points`` = frontières de batches,
        ``drift_type = "incremental"``.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_path"])
    n_features = int(data_cfg.get("n_features", N_FEATURES))
    n_batches = int(data_cfg.get("n_batches", N_BATCHES))
    file_template = data_cfg.get("file_template", "batch{i}.dat")
    normalization = data_cfg.get("normalization", "zscore")
    seed = int(cfg.get("seed", DEFAULT_SEED))

    np.random.seed(seed)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    cursor = 0
    for i in range(1, n_batches + 1):
        batch_path = raw_dir / file_template.format(i=i)
        if not batch_path.exists():
            raise FileNotFoundError(
                f"Batch introuvable : {batch_path}\n"
                f"Lancer : python scripts/download_drift_datasets.py --dataset gas_sensor_drift"
            )
        Xb, yb = _parse_libsvm_dat(batch_path, n_features)
        X_parts.append(Xb)
        y_parts.append(yb)
        segments.append((cursor, cursor + len(Xb)))
        cursor += len(Xb)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)

    # Normalisation figée sur le segment initial (batch 1) — drift laissé visible.
    if normalization == "zscore":
        X, mean, std = freeze_zscore(X, segments[0])
        norm_meta = {"method": "zscore_frozen_on_batch_1"}
    else:
        norm_meta = {"method": "none"}

    drift_points = segments_to_drift_points(segments)
    feature_names = [f"sensor_feat_{k}" for k in range(n_features)]

    return DriftDataset(
        X=X,
        y=y,
        drift_points=drift_points,
        drift_type="incremental",
        feature_names=feature_names,
        segments=segments,
        metadata={
            "dataset": "gas_sensor_drift",
            "source": "UCI Gas Sensor Array Drift Dataset (Vergara 2012, DOI 10.24432/C5RP6W)",
            "license": "research only (commercial use excluded)",
            "ground_truth": "structural (10 temporal batches)",
            "n_batches": n_batches,
            "normalization": norm_meta,
            "config_snapshot": cfg,
        },
    )
