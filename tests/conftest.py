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

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_cl_model import BaseCLModel
from src.models.ewc import EWCMlpClassifier


class MockCLModel(BaseCLModel):
    """Stub minimal de BaseCLModel pour les tests de scenarios.py."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 2, x.shape[0])

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        return 0.0

    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        pass

    def count_parameters(self) -> int:
        return 0

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        return 0

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


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


@pytest.fixture
def hdc_model_mock() -> MockCLModel:
    """Stub MockCLModel pour tester scenarios.py sans dépendance HDC réelle."""
    return MockCLModel({})


@pytest.fixture
def hdc_config() -> dict:
    """Config minimale compatible avec la structure de hdc_config.yaml.

    D=64 et n_levels=5 réduisent le temps de test à quelques ms.
    RAM estimée : (2×64×4 + 2×64×1 + 4×64×4 + 2×4) = 1 560 B — loin du budget 64 Ko.
    """
    return {
        "hdc": {
            "D": 64,
            "n_levels": 5,
            "seed": 42,
            "base_vectors_path": "/tmp/hdc_test_bv_D64_L5.npz",
        },
        "data": {
            "n_features": 4,
            "n_classes": 2,
        },
        "feature_bounds": {  # racine du dict — cf. HDCClassifier._load_feature_bounds
            "temperature": [-3.0, 3.0],
            "pressure":    [-3.0, 3.0],
            "vibration":   [-3.0, 3.0],
            "humidity":    [-3.0, 3.0],
        },
        "training": {"epochs_per_task": 1, "batch_size": 1, "seed": 42},
        "memory":   {"target_ram_bytes": 65536, "warn_if_above_bytes": 52000},
    }


@pytest.fixture
def hdc_synthetic_loader():
    """Factory DataLoader synthétique pour HDC — input_dim=4."""
    def make_loader(n: int = 32, seed: int = 42) -> DataLoader:
        torch.manual_seed(seed)
        X = torch.randn(n, 4)
        y = torch.randint(0, 2, (n, 1)).float()
        return DataLoader(TensorDataset(X, y), batch_size=8)
    return make_loader


@pytest.fixture
def hdc_cl_tasks(hdc_synthetic_loader) -> list[dict]:
    """3 tâches CL synthétiques pour HDC (input_dim=4)."""
    return [
        {
            "task_id": i + 1,
            "domain": d,
            "train_loader": hdc_synthetic_loader(seed=i),
            "val_loader": hdc_synthetic_loader(n=16, seed=i + 10),
        }
        for i, d in enumerate(["pump", "turbine", "compressor"])
    ]


# ---------------------------------------------------------------------------
# Fixtures — modèles non supervisés (S5-08)
# ---------------------------------------------------------------------------

_N_FEATURES = 4
_N_TRAIN = 100
_N_VAL = 40
_N_ANOMALY = 20


@pytest.fixture
def unsupervised_data() -> dict:
    """
    Données synthétiques 4 features : normal N(0,1) + anomalie N(5,1).

    Séparation nette pour des assertions déterministes (anomaly_score > normal_score).
    """
    rng = np.random.default_rng(42)
    X_normal = rng.normal(0, 1, (_N_TRAIN, _N_FEATURES))
    X_anomaly = rng.normal(5, 1, (_N_ANOMALY, _N_FEATURES))
    X_train = np.vstack([X_normal, X_anomaly[:10]])
    y_train = np.array([0] * _N_TRAIN + [1] * 10)
    X_val = np.vstack([rng.normal(0, 1, (_N_VAL, _N_FEATURES)), X_anomaly[10:]])
    y_val = np.array([0] * _N_VAL + [1] * 10)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


@pytest.fixture
def kmeans_config() -> dict:
    """Config minimale KMeansDetector — cl_strategy='refit' (défaut implémentation)."""
    return {
        "kmeans": {
            "k_min": 2,
            "k_max": 4,
            "k_method": "silhouette",
            "anomaly_percentile": 95,
            "cl_strategy": "refit",
            "seed": 42,
        }
    }


@pytest.fixture
def knn_config() -> dict:
    """Config minimale KNNDetector — cl_strategy='accumulate'."""
    return {
        "knn": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "anomaly_percentile": 95,
            "cl_strategy": "accumulate",
            "max_ref_samples": 200,
            "seed": 42,
        }
    }


@pytest.fixture
def pca_config() -> dict:
    """Config minimale PCABaseline — cl_strategy='refit' (défaut implémentation)."""
    return {
        "pca": {
            "n_components": 2,
            "anomaly_percentile": 95,
            "cl_strategy": "refit",
            "seed": 42,
        }
    }


@pytest.fixture
def mahalanobis_config() -> dict:
    """Config minimale MahalanobisDetector — cl_strategy='refit'."""
    return {
        "mahalanobis": {
            "anomaly_threshold": None,
            "anomaly_percentile": 95,
            "reg_covar": 1e-6,
            "cl_strategy": "refit",
        }
    }


@pytest.fixture
def dbscan_config() -> dict:
    """Config minimale DBSCANDetector — cl_strategy='refit'."""
    return {
        "dbscan": {
            "EPSILON": 2.0,       # large pour données synthétiques N(0,1) / N(5,1)
            "MIN_SAMPLES": 3,
            "metric": "euclidean",
            "algorithm": "auto",
            "anomaly_threshold": None,
            "anomaly_percentile": 95,
            "cl_strategy": "refit",
        }
    }
