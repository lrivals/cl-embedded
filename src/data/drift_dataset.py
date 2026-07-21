"""
drift_dataset.py — Structure commune et utilitaires pour les datasets de drift (Sprint 43).

Fournit :
    - ``DriftDataset`` : conteneur uniforme exposant la **vérité-terrain de drift**
      (``drift_points``, ``drift_type``) consommée par le harnais d'évaluation (S44) et
      le portage board (S45).
    - ``freeze_zscore`` : normalisation Z-score **figée sur le segment initial**, appliquée
      aux segments suivants **sans re-fit** — miroir de ``CLStreamSplitter`` de
      ``src/data/pump_dataset.py``. Essentiel pour que les détecteurs de drift *voient* la
      dérive non renormalisée (une normalisation glissante masquerait le drift).

Tous les loaders ``src/data/<dataset>_dataset.py`` retournent un ``DriftDataset`` via
``load(config_path)`` et sont enregistrés dans ``src/data/__init__.py`` (``DRIFT_LOADERS``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Taxonomie standard du concept drift (Gama et al. 2014).
DriftType = Literal["sudden", "gradual", "incremental", "recurring"]

# Seed projet (reproductibilité — cf. src/utils/reproducibility.py).
DEFAULT_SEED: int = 42

# Plancher d'écart-type : évite la division par zéro sur les features constantes
# (identique à CLStreamSplitter.fit_normalizer dans pump_dataset.py).
_STD_FLOOR: float = 1e-8


@dataclass
class DriftDataset:
    """
    Conteneur uniforme d'un dataset de drift.

    Attributes
    ----------
    X : np.ndarray
        Features, shape ``[N, d]``, dtype float32. Ordre **temporel/segmenté préservé**
        (aucun shuffle global — le drift est un phénomène ordonné).
    y : np.ndarray | None
        Labels de faute/classe si le dataset est dual-usage, sinon ``None``.
    drift_points : list[int] | None
        Indices (dans ``X``) de changement de distribution — **vérité-terrain**.
        ``None`` si le dataset n'a pas de ground-truth ponctuelle (ex. Electricity).
    drift_type : str | list[str]
        Type de drift global ou par segment (taxonomie ``DriftType``).
    feature_names : list[str]
        Noms des ``d`` colonnes de ``X``.
    segments : list[tuple[int, int]]
        Découpage CL/temporel en intervalles ``[start, end)`` sur ``X``.
    metadata : dict
        Source, licence, config_snapshot, et tout champ descriptif.

    Notes
    -----
    ``drift_points`` et ``segments`` sont cohérents : les frontières internes de
    ``segments`` coïncident avec ``drift_points`` lorsque la ground-truth est structurelle.
    """

    X: np.ndarray
    y: np.ndarray | None
    drift_points: list[int] | None
    drift_type: DriftType | list[DriftType]
    feature_names: list[str]
    segments: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Nombre d'échantillons ``N``."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Dimension de features ``d`` (impact direct sur l'état mémoire board S45)."""
        return int(self.X.shape[1])


def segments_to_drift_points(segments: list[tuple[int, int]]) -> list[int]:
    """
    Convertit une liste de segments ``[start, end)`` en points de drift (frontières internes).

    Parameters
    ----------
    segments : list[tuple[int, int]]
        Segments contigus couvrant ``[0, N)``.

    Returns
    -------
    list[int]
        Les indices de début de chaque segment sauf le premier (les changements de distribution).
    """
    return [start for start, _ in segments[1:]]


def freeze_zscore(
    X: np.ndarray,
    ref_slice: slice | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisation Z-score **figée** sur un segment de référence, appliquée à tout ``X``.

    Les statistiques (mean, std) sont calculées **uniquement** sur ``X[ref_slice]``
    (typiquement le segment initial) puis appliquées à l'ensemble **sans re-fit**. C'est
    la condition pour que la dérive reste visible : une re-normalisation par segment la
    masquerait.

    Parameters
    ----------
    X : np.ndarray
        Features, shape ``[N, d]``.
    ref_slice : slice | tuple[int, int]
        Segment de référence (``slice`` ou ``(start, end)``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(X_norm, mean, std)`` — ``X_norm`` dtype float32, ``mean``/``std`` shape ``[d]``.
    """
    if isinstance(ref_slice, tuple):
        ref_slice = slice(ref_slice[0], ref_slice[1])

    ref = X[ref_slice]
    mean = ref.mean(axis=0)
    std = ref.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)

    X_norm = ((X - mean) / std).astype(np.float32)
    return X_norm, mean.astype(np.float32), std.astype(np.float32)
