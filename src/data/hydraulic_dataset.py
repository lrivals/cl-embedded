"""
hydraulic_dataset.py — Loader du Condition Monitoring of Hydraulic Systems (Sprint 43, S4302).

Dataset de **faute** (ZeMA/UCI, 2205 cycles de 60 s d'un banc hydraulique) réutilisé ici comme
support de **drift secondaire** : chaque cycle est agrégé en features par-capteur (moyenne par
cycle), puis les cycles sont **segmentés par la condition du refroidisseur** (profile col. 1,
cooler ∈ {3, 20, 100} %). L'ordre résultant fait apparaître le drift comme des blocs contigus →
``drift_points`` = frontières de segments (**ground-truth structurelle**, honnête : pas de points
ponctuels natifs).

- ``X`` : moyenne par cycle de chaque capteur (17 capteurs par défaut), Z-score figé sur le
  segment initial.
- ``y`` : label binaire configurable (défaut ``stable_flag`` — profile col. 5).
- ``drift_points`` : frontières des segments cooler.
- ``drift_type`` : ``"incremental"``.

Source : data/raw/Condition Monitoring of Hydraulic Systems/*.txt
Licence : usage recherche (ZeMA gGmbH).
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

# Capteurs agrégés par défaut (un fichier .txt par capteur, rows = cycles).
DEFAULT_SENSORS: list[str] = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
]

# Colonnes de profile.txt (tab-delimited, une ligne par cycle).
PROFILE_COLUMNS: list[str] = [
    "cooler_condition", "valve_condition", "pump_leakage",
    "accumulator_pressure", "stable_flag",
]


def _cycle_means(raw_dir: Path, sensors: list[str]) -> np.ndarray:
    """
    Charge chaque fichier capteur et calcule la moyenne par cycle (par ligne).

    Parameters
    ----------
    raw_dir : Path
        Répertoire brut du dataset.
    sensors : list[str]
        Noms de capteurs (fichiers ``<sensor>.txt``).

    Returns
    -------
    np.ndarray
        ``[n_cycles, len(sensors)]`` float32.
    """
    cols: list[np.ndarray] = []
    n_cycles: int | None = None
    for sensor in sensors:
        path = raw_dir / f"{sensor}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Fichier capteur introuvable : {path}")
        # Moyenne par ligne (cycle) — chaque ligne = points échantillonnés du cycle.
        arr = np.loadtxt(path, dtype=np.float32)
        mean_per_cycle = arr.mean(axis=1).astype(np.float32)
        if n_cycles is None:
            n_cycles = len(mean_per_cycle)
        elif len(mean_per_cycle) != n_cycles:
            raise ValueError(
                f"Incohérence de cycles pour {sensor}: {len(mean_per_cycle)} != {n_cycles}"
            )
        cols.append(mean_per_cycle)
    return np.stack(cols, axis=1)


def load(config_path: str) -> DriftDataset:
    """
    Charge le dataset hydraulique en ``DriftDataset`` (segmenté par condition cooler).

    Parameters
    ----------
    config_path : str
        Chemin vers ``configs/hydraulic_drift_config.yaml``.

    Returns
    -------
    DriftDataset
        ``X`` [N, n_sensors], ``y`` binaire, ``drift_points`` = frontières cooler,
        ``drift_type = "incremental"``.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_path"])
    sensors = data_cfg.get("sensors", DEFAULT_SENSORS)
    segment_by = data_cfg.get("segment_by", "cooler_condition")
    label_column = data_cfg.get("label_column", "stable_flag")
    normalization = data_cfg.get("normalization", "zscore")
    seed = int(cfg.get("seed", DEFAULT_SEED))

    np.random.seed(seed)

    profile_path = raw_dir / data_cfg.get("profile_file", "profile.txt")
    if not profile_path.exists():
        raise FileNotFoundError(
            f"profile.txt introuvable : {profile_path}\n"
            f"Lancer : python scripts/download_drift_datasets.py --dataset hydraulic"
        )
    profile = np.loadtxt(profile_path, dtype=np.float32)
    profile_map = {name: profile[:, i] for i, name in enumerate(PROFILE_COLUMNS)}

    X_raw = _cycle_means(raw_dir, sensors)  # [n_cycles, n_sensors], ordre cycles original

    if X_raw.shape[0] != profile.shape[0]:
        raise ValueError(
            f"Cycles capteurs ({X_raw.shape[0]}) != cycles profile ({profile.shape[0]})"
        )

    # Label binaire : par convention, valeur nominale/majoritaire → 0, sinon 1.
    label_raw = profile_map[label_column]
    if label_column == "stable_flag":
        y = label_raw.astype(np.int64)  # déjà {0,1}
    else:
        # Binarisation générique : classe la plus fréquente = 0 (nominal), reste = 1.
        vals, counts = np.unique(label_raw, return_counts=True)
        nominal = vals[np.argmax(counts)]
        y = (label_raw != nominal).astype(np.int64)

    # Segmentation par la condition choisie : trie les cycles pour rendre le drift contigu.
    seg_key = profile_map[segment_by]
    order = np.argsort(seg_key, kind="stable")
    X_ordered = X_raw[order]
    y = y[order]
    seg_key_sorted = seg_key[order]

    # Frontières des segments = changements de valeur de la clé de segmentation.
    boundaries = np.where(np.diff(seg_key_sorted) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(X_ordered)]])
    segments = [(int(s), int(e)) for s, e in zip(starts, ends)]

    if normalization == "zscore":
        X, _, _ = freeze_zscore(X_ordered, segments[0])
        norm_meta = {"method": f"zscore_frozen_on_segment_0_by_{segment_by}"}
    else:
        X = X_ordered.astype(np.float32)
        norm_meta = {"method": "none"}

    drift_points = segments_to_drift_points(segments)

    return DriftDataset(
        X=X,
        y=y,
        drift_points=drift_points,
        drift_type="incremental",
        feature_names=[f"{s}_mean" for s in sensors],
        segments=segments,
        metadata={
            "dataset": "hydraulic",
            "source": "Condition Monitoring of Hydraulic Systems (ZeMA gGmbH / UCI, 2018)",
            "license": "research use (ZeMA gGmbH)",
            "ground_truth": f"structural (segmented by {segment_by})",
            "label_column": label_column,
            "segment_by": segment_by,
            "segment_values": [float(v) for v in np.unique(seg_key_sorted)],
            "normalization": norm_meta,
            "config_snapshot": cfg,
        },
    )
