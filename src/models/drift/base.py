# ruff: noqa: N803, N806  — X est une convention mathématique ML (sklearn API)
"""
src/models/drift/base.py — Sprint 44 (S4401) : interface commune des détecteurs de drift.

Toutes les familles de détecteurs (supervisés flux d'erreur — S4402 ; non-supervisés features/score
— S4403) se conforment à ``BaseDriftDetector`` : un même contrat ``update(value) -> DriftVerdict``,
calibration sur segment d'enrôlement, empreinte mémoire majorée (``get_state_bytes``), et déclaration
explicite du besoin de labels (``requires_label``).

Verdict à **3 niveaux** (``NORMAL / WARNING / DRIFT``) : ``WARNING`` est requis par DDM/EDDM (zone
d'alerte avant le drift confirmé). Ce verdict est distinct :

- du baseline ``SlidingWindowDriftDetector`` (``src/evaluation/drift_detector.py``) qui émet des chaînes
  ``"NORMAL"/"FAULT"/"DRIFT"`` — laissé tel quel, catalogué comme baseline, jamais dupliqué ;
- de l'enum firmware ``DriftVerdict`` (``inc/drift_detector.h`` : ``DRIFT_NORMAL/FAULT/DRIFT``) — le
  mapping vers le binaire board est déféré au Sprint 45.

Références
----------
    Gama et al. 2014 (survey drift) · docs/context/drift_detectors.md (source de vérité S4401).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class DriftVerdict(Enum):
    """Verdict à 3 niveaux émis à chaque ``update``.

    NORMAL  : distribution stable.
    WARNING : zone d'alerte (dérive suspectée, pas encore confirmée) — DDM/EDDM.
    DRIFT   : changement de distribution confirmé (déclenche une action : MAJ, ré-enrôlement…).
    """

    NORMAL = 0
    WARNING = 1
    DRIFT = 2


class BaseDriftDetector(ABC):
    """Interface commune des détecteurs de drift en ligne (S4401).

    Chaque détecteur consomme **un échantillon par appel** ``update`` — une erreur ``0/1`` (familles
    supervisées) ou une feature scalaire / un score (familles non-supervisées) — et retourne un
    ``DriftVerdict``. L'état est **borné** (O(1) ou fenêtre à capacité fixe) : ``get_state_bytes``
    reflète l'empreinte majorée, argument de portabilité MCU (Sprint 45).

    Parameters
    ----------
    config : dict
        Sous-section du détecteur dans ``configs/sprint44_drift_detection.yaml``. Aucun
        hyperparamètre n'est codé en dur : tout provient de cette section.

    Attributes
    ----------
    config : dict
        Sous-section de configuration (conservée pour introspection).

    Notes
    -----
    ``requires_label`` classe explicitement le détecteur sur l'axe supervisé (flux d'erreur, label
    requis) vs non-supervisé (features, aucun label) — c'est le pivot scientifique du Sprint 44.
    """

    #: Redéfini par chaque sous-classe (True = supervisé, flux d'erreur ; False = non-supervisé).
    _REQUIRES_LABEL: bool = False

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = dict(config) if config else {}

    @property
    def requires_label(self) -> bool:
        """True si le détecteur surveille un flux d'erreur (labels requis)."""
        return self._REQUIRES_LABEL

    @abstractmethod
    def update(self, value: float) -> DriftVerdict:
        """Traite un échantillon (erreur ``0/1`` ou feature/score) et retourne le verdict courant."""

    def set_params_from_reference(self, reference_values: np.ndarray) -> None:
        """Calibre les paramètres (seuils / percentiles / fenêtre de référence) sur l'enrôlement.

        No-op par défaut (détecteurs auto-calibrants comme DDM/EDDM/Page-Hinkley qui apprennent
        leurs minima en ligne). Surchargé par les familles non-supervisées (KS/PSI/MMD…) qui figent
        une distribution de référence sur le segment d'enrôlement.

        Parameters
        ----------
        reference_values : np.ndarray
            Valeurs d'enrôlement — flux 1D (feature/score) ou matrice ``[N, d]`` (multivarié).
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """Réinitialise l'état interne (nouvelle machine / nouveau contexte)."""

    @abstractmethod
    def get_state_bytes(self) -> int:
        """Empreinte mémoire majorée de l'état interne, en octets (indépendante du n° d'échantillon).

        Doit être **constante** dans le temps : O(1) pour les détecteurs supervisés, majorée par la
        capacité de fenêtre/buckets/bins pour les non-supervisés. Utilisée pour le profilage S45.
        """

    def update_batch(self, values: np.ndarray) -> list[DriftVerdict]:
        """Traite une séquence de valeurs et retourne la liste des verdicts."""
        return [self.update(float(v)) for v in np.asarray(values).ravel()]


def error_stream(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Produit le flux d'erreur ``e_t = 1[ŷ_t ≠ y_t]`` d'un modèle de faute (source S4402).

    Réutilise l'inférence du modèle passé — n'en réimplémente pas la logique. Générique : accepte
    tout modèle exposant ``predict(X) -> array`` (EWC tête binaire, Mahalanobis seuillé…). Ne fige
    pas la décision ``TODO(arnaud)`` S4400 sur le modèle de référence.

    Parameters
    ----------
    model : object
        Modèle de faute exposant ``predict(X) -> np.ndarray`` de labels prédits.
    X : np.ndarray [N, d]
        Échantillons dans l'ordre chronologique du flux.
    y : np.ndarray [N]
        Labels de vérité-terrain (retour actif, scénario « active learning » P2 du Sprint 38).

    Returns
    -------
    np.ndarray [N] de ``int`` ∈ {0, 1}
        1 = erreur du modèle sur l'échantillon, 0 = prédiction correcte.
    """
    y_pred = np.asarray(model.predict(X)).ravel()
    y_true = np.asarray(y).ravel()
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"predict() a renvoyé {y_pred.shape[0]} labels pour {y_true.shape[0]} cibles."
        )
    return (y_pred != y_true).astype(np.int64)  # MEM: N×8 B (hôte uniquement, hors budget board)
