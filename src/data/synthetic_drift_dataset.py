"""
synthetic_drift_dataset.py — Générateur synthétique de flux à drift (Sprint 43, S4302).

Fournit une **vérité-terrain de drift EXACTE par construction** : les ``drift_points`` sont
imposés en config et retournés à l'identique. Sert à **calibrer** la chaîne de mesure des
métriques S44 (délai de détection, FAR) avant application aux datasets réels — usage **PC-only**,
non porté MCU.

Générateurs (numpy pur, **aucune dépendance `river`**) :
    - ``sea`` : concept SEA (Street & Kim 2001) — 3 features [0,10], règle ``f0 + f1 <= theta``,
      le seuil ``theta`` change à chaque drift (drift **sudden** de frontière de décision).
    - ``rotating_hyperplane`` : hyperplan ``w·x > w0`` dont les poids ``w`` tournent progressivement
      autour de chaque drift_point (drift **incremental/gradual**).
    - ``gradual_mixture`` : mélange progressif de deux gaussiennes sur une largeur de transition
      autour de chaque drift_point (drift **gradual**).

Aucune donnée sur disque : tout est généré à la volée, seed reproductible.
"""

from __future__ import annotations

import numpy as np

from src.data.drift_dataset import DEFAULT_SEED, DriftDataset
from src.utils.config_loader import load_config

# Descripteurs par générateur (nombre de features par défaut).
_SEA_N_FEATURES: int = 3


def _gen_sea(
    n_samples: int, drift_points: list[int], thresholds: list[float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Concept SEA : 3 features U[0,10], label = 1 si f0+f1 <= theta (theta change au drift)."""
    X = rng.uniform(0.0, 10.0, size=(n_samples, _SEA_N_FEATURES)).astype(np.float32)
    bounds = [0, *drift_points, n_samples]
    y = np.zeros(n_samples, dtype=np.int64)
    for seg_idx in range(len(bounds) - 1):
        start, end = bounds[seg_idx], bounds[seg_idx + 1]
        theta = thresholds[seg_idx % len(thresholds)]
        y[start:end] = (X[start:end, 0] + X[start:end, 1] <= theta).astype(np.int64)
    return X, y


def _gen_rotating_hyperplane(
    n_samples: int,
    drift_points: list[int],
    n_features: int,
    transition_width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hyperplan rotatif : w tourne progressivement autour de chaque drift (incremental)."""
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    bounds = [0, *drift_points, n_samples]
    n_concepts = len(bounds) - 1
    # Un vecteur de poids par concept (rotation par ajout d'angle progressif).
    base = rng.normal(size=(n_concepts, n_features))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    w0 = 0.5 * n_features  # seuil centré (features ~U[0,1])

    y = np.zeros(n_samples, dtype=np.int64)
    for seg_idx in range(n_concepts):
        start, end = bounds[seg_idx], bounds[seg_idx + 1]
        w = base[seg_idx]
        if seg_idx > 0 and transition_width > 0:
            # Transition progressive du concept précédent vers le courant.
            prev = base[seg_idx - 1]
            for i in range(start, min(start + transition_width, end)):
                alpha = (i - start) / transition_width
                wi = (1 - alpha) * prev + alpha * w
                y[i] = int(X[i] @ wi > w0)
            seg_start = min(start + transition_width, end)
        else:
            seg_start = start
        y[seg_start:end] = (X[seg_start:end] @ w > w0).astype(np.int64)
    return X, y


def _gen_gradual_mixture(
    n_samples: int,
    drift_points: list[int],
    n_features: int,
    transition_width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Mélange graduel de deux gaussiennes alternées autour de chaque drift (gradual)."""
    bounds = [0, *drift_points, n_samples]
    n_concepts = len(bounds) - 1
    means = [np.full(n_features, 2.0 * (c % 2)) for c in range(n_concepts)]
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for seg_idx in range(n_concepts):
        start, end = bounds[seg_idx], bounds[seg_idx + 1]
        cur_mean = means[seg_idx]
        prev_mean = means[seg_idx - 1] if seg_idx > 0 else cur_mean
        for i in range(start, end):
            if seg_idx > 0 and transition_width > 0 and i < start + transition_width:
                alpha = (i - start) / transition_width
                p_cur = alpha
            else:
                p_cur = 1.0
            mean = p_cur * cur_mean + (1 - p_cur) * prev_mean
            X[i] = rng.normal(loc=mean, scale=1.0).astype(np.float32)
            y[i] = seg_idx % 2
    return X, y


def load(config_path: str) -> DriftDataset:
    """
    Génère un flux synthétique à drift exact en ``DriftDataset``.

    Parameters
    ----------
    config_path : str
        Chemin vers ``configs/synthetic_drift_config.yaml``.

    Returns
    -------
    DriftDataset
        ``drift_points`` retournés == ceux imposés en config (vérité-terrain parfaite).
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    generator = data_cfg.get("generator", "sea")
    n_samples = int(data_cfg["n_samples"])
    drift_points = [int(p) for p in data_cfg["drift_points"]]
    n_features = int(data_cfg.get("n_features", _SEA_N_FEATURES))
    transition_width = int(data_cfg.get("transition_width", 0))
    thresholds = [float(t) for t in data_cfg.get("sea_thresholds", [8.0, 12.0])]
    seed = int(cfg.get("seed", DEFAULT_SEED))

    rng = np.random.default_rng(seed)

    if generator == "sea":
        n_features = _SEA_N_FEATURES
        X, y = _gen_sea(n_samples, drift_points, thresholds, rng)
        drift_type = "sudden"
    elif generator == "rotating_hyperplane":
        X, y = _gen_rotating_hyperplane(
            n_samples, drift_points, n_features, transition_width, rng
        )
        drift_type = "incremental"
    elif generator == "gradual_mixture":
        X, y = _gen_gradual_mixture(n_samples, drift_points, n_features, transition_width, rng)
        drift_type = "gradual"
    else:
        raise ValueError(
            f"Générateur inconnu : {generator!r}. "
            f"Attendu : sea | rotating_hyperplane | gradual_mixture."
        )

    bounds = [0, *drift_points, n_samples]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    return DriftDataset(
        X=X,
        y=y,
        drift_points=drift_points,  # exact == config (vérité-terrain parfaite)
        drift_type=drift_type,
        feature_names=[f"x{k}" for k in range(n_features)],
        segments=segments,
        metadata={
            "dataset": "synthetic",
            "source": f"numpy synthetic generator ({generator})",
            "license": "n/a (generated on-the-fly)",
            "ground_truth": "exact (drift_points imposed by construction)",
            "generator": generator,
            "seed": seed,
            "config_snapshot": cfg,
        },
    )
