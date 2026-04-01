"""
base_cl_model.py — Classe abstraite commune aux trois modèles CL du projet.

Tous les modèles (EWC, HDC, TinyOL) doivent hériter de BaseCLModel
et implémenter les méthodes abstraites.

Référence architecture : docs/architecture.md
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BaseCLModel(ABC):
    """
    Interface commune pour tous les modèles de Continual Learning du projet.

    Cette classe garantit que chaque modèle peut être évalué, profilé en mémoire,
    et utilisé dans la boucle d'entraînement CL générique (CLTrainer).

    Notes
    -----
    Contraintes MCU rappelées :
        - RAM totale (modèle + activations + overhead CL) ≤ 64 Ko
        - Optimizer : SGD uniquement dans les phases CL
        - Activations : ReLU uniquement
        - Pas d'allocation dynamique dans forward()
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Configuration chargée depuis un fichier YAML (configs/*.yaml).
        """
        self.config = config
        self._ram_budget_bytes: int = config.get("memory", {}).get("target_ram_bytes", 65536)
        self._warn_threshold: int = config.get("memory", {}).get("warn_if_above_bytes", 52000)

    # ------------------------------------------------------------------
    # Interface obligatoire — à implémenter dans chaque sous-classe
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Inférence sur un batch d'observations.

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features]
            Observations normalisées.

        Returns
        -------
        np.ndarray [batch_size] ou [batch_size, n_classes]
            Prédictions (logits ou probabilités selon le modèle).
        """

    @abstractmethod
    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Mise à jour incrémentale en ligne (1 échantillon ou 1 mini-batch).

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features]
        y : np.ndarray [batch_size]
            Labels.

        Returns
        -------
        float
            Valeur de la perte (loss) pour cet update.
        """

    @abstractmethod
    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        """
        Callback appelé à la fin de chaque tâche CL.

        Utilisé pour :
        - EWC : calcul et accumulation de la Fisher diagonale
        - HDC : (optionnel) renormalisation des prototypes
        - TinyOL : (optionnel) calibration de la tête OtO

        Parameters
        ----------
        task_id : int
            Identifiant de la tâche qui vient de se terminer.
        dataloader : iterable
            Données de la tâche courante (pour calculer la Fisher, etc.).
        """

    @abstractmethod
    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres (entraînables ou non)."""

    @abstractmethod
    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """
        Estime l'empreinte RAM des poids seuls (hors activations et overhead CL).

        Parameters
        ----------
        dtype : str
            "fp32" (4 B/param) ou "int8" (1 B/param).

        Returns
        -------
        int
            Estimation en octets.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde l'état complet du modèle (poids + état CL)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Charge un état sauvegardé."""

    # ------------------------------------------------------------------
    # Méthodes communes (non abstraites)
    # ------------------------------------------------------------------

    def check_ram_budget(self) -> dict:
        """
        Vérifie si le modèle respecte le budget RAM cible.

        Returns
        -------
        dict
            {"within_budget": bool, "estimated_bytes": int, "budget_bytes": int}
        """
        estimated = self.estimate_ram_bytes("fp32")
        within = estimated <= self._ram_budget_bytes

        if estimated > self._warn_threshold:
            print(
                f"⚠️  RAM estimée ({estimated / 1024:.1f} Ko) > seuil d'alerte "
                f"({self._warn_threshold / 1024:.1f} Ko). "
                f"Budget total : {self._ram_budget_bytes / 1024:.0f} Ko."
            )

        return {
            "within_budget": within,
            "estimated_bytes": estimated,
            "budget_bytes": self._ram_budget_bytes,
            "utilization_pct": estimated / self._ram_budget_bytes * 100,
        }

    def summary(self) -> str:
        """Retourne un résumé compact du modèle (paramètres + RAM)."""
        n_params = self.count_parameters()
        ram_fp32 = self.estimate_ram_bytes("fp32")
        ram_int8 = self.estimate_ram_bytes("int8")
        budget = self._ram_budget_bytes

        lines = [
            f"Model: {self.__class__.__name__}",
            f"  Parameters : {n_params:,}",
            f"  RAM (FP32) : {ram_fp32:,} B ({ram_fp32 / 1024:.1f} Ko) "
            f"— {ram_fp32 / budget * 100:.1f}% of {budget // 1024} Ko budget",
            f"  RAM (INT8) : {ram_int8:,} B ({ram_int8 / 1024:.1f} Ko)",
        ]
        return "\n".join(lines)

    def save_summary(self, output_path: str) -> None:
        """Sauvegarde le résumé + bilan RAM dans un fichier JSON."""
        summary_data = {
            "model": self.__class__.__name__,
            "n_params": self.count_parameters(),
            "ram_fp32_bytes": self.estimate_ram_bytes("fp32"),
            "ram_int8_bytes": self.estimate_ram_bytes("int8"),
            "ram_budget_bytes": self._ram_budget_bytes,
            "within_budget": self.check_ram_budget()["within_budget"],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary_data, f, indent=2)
