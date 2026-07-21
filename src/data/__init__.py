"""
src.data — Loaders de datasets du projet.

Registre léger des **loaders de drift** (Sprint 43) exposant une interface commune
``load(config_path) -> DriftDataset``. Les loaders historiques (pump, cwru, cmapss, …)
restent importables directement via ``from src.data.<module> import …``.
"""

from __future__ import annotations

from src.data.drift_dataset import DriftDataset
from src.data.electricity_dataset import load as load_electricity
from src.data.gas_sensor_drift_dataset import load as load_gas_sensor_drift
from src.data.hydraulic_dataset import load as load_hydraulic
from src.data.synthetic_drift_dataset import load as load_synthetic

# Registre nom → fonction de chargement (consommé par scripts/characterize_drift.py).
DRIFT_LOADERS = {
    "gas_sensor_drift": load_gas_sensor_drift,
    "hydraulic": load_hydraulic,
    "electricity": load_electricity,
    "synthetic": load_synthetic,
}

# Config par défaut associée à chaque loader de drift.
DRIFT_CONFIGS = {
    "gas_sensor_drift": "configs/gas_sensor_drift_config.yaml",
    "hydraulic": "configs/hydraulic_drift_config.yaml",
    "electricity": "configs/electricity_drift_config.yaml",
    "synthetic": "configs/synthetic_drift_config.yaml",
}

__all__ = ["DriftDataset", "DRIFT_LOADERS", "DRIFT_CONFIGS"]
