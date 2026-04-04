"""
tests/conftest.py — Fixtures partagées pour les tests CL.

Fixtures principales :
    model           : EWCMlpClassifier() avec paramètres par défaut (input_dim=6, 769 params)
    synthetic_loader: factory → DataLoader synthétique input_dim=6
    cl_tasks        : liste de 3 tâches CL avec DataLoaders synthétiques
    ewc_config      : dict de config compatible ewc_config.yaml (données synthétiques)
    known_acc_matrix: matrice 3×3 aux valeurs connues pour tester AA/AF/BWT
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc import EWCMlpClassifier


@pytest.fixture
def model() -> EWCMlpClassifier:
    """EWCMlpClassifier avec paramètres par défaut (input_dim=6 → 769 params)."""
    return EWCMlpClassifier()


@pytest.fixture
def synthetic_loader():
    """Factory DataLoader synthétique — input_dim=6."""
    def make_loader(n: int = 64, seed: int = 42) -> DataLoader:
        torch.manual_seed(seed)
        X = torch.randn(n, 6)
        y = torch.randint(0, 2, (n, 1)).float()
        return DataLoader(TensorDataset(X, y), batch_size=32)
    return make_loader


@pytest.fixture
def cl_tasks(synthetic_loader) -> list[dict]:
    """3 tâches CL synthétiques (Pump / Turbine / Compressor simulées, input_dim=6)."""
    return [
        {
            "task_id": i + 1,
            "domain": d,
            "train_loader": synthetic_loader(seed=i),
            "val_loader": synthetic_loader(n=32, seed=i + 10),
        }
        for i, d in enumerate(["Pump", "Turbine", "Compressor"])
    ]


@pytest.fixture
def ewc_config() -> dict:
    """Config minimale compatible avec la structure de ewc_config.yaml."""
    return {
        "training": {
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "epochs_per_task": 2,  # réduit pour la rapidité des tests
            "batch_size": 32,
            "seed": 42,
        },
        "ewc": {
            "lambda": 1000,
            "gamma": 0.9,
            "n_fisher_samples": 32,
        },
        "model": {
            "input_dim": 6,
            "hidden_dims": [32, 16],
            "dropout": 0.0,  # désactivé pour reproductibilité des tests
        },
    }


@pytest.fixture
def known_acc_matrix() -> np.ndarray:
    """
    Matrice 3×3 avec valeurs connues pour tester les formules de métriques.

    acc_matrix[i, j] = accuracy sur tâche j après entraînement sur tâche i.
    Simule un oubli catastrophique modéré (EWC-like).

    Valeurs attendues :
        AA  = (0.86 + 0.83 + 0.89) / 3 ≈ 0.8600
        AF  > 0  (oubli présent)
        BWT < 0  (transfert backward négatif)
    """
    return np.array([
        [0.91, np.nan, np.nan],
        [0.88, 0.85,  np.nan],
        [0.86, 0.83,  0.89 ],
    ])
