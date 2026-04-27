"""
tinyol_anomaly_detector.py — TinyOL backbone utilisé comme détecteur d'anomalies (one-class).

Pas de tête OtO. Le score d'anomalie est l'erreur de reconstruction MSE :
une observation normale est bien reconstruite (MSE bas), une anomalie ne l'est pas (MSE élevé).

Stratégie CL : refit — l'autoencoder est réentraîné from scratch à chaque tâche.
Conforme au scénario anomaly detection du plan (S12-AD).

Références :
    Ren2021TinyOL — backbone TinyOL
    docs/models/tinyol_spec.md §5 — pré-entraînement sur données normales uniquement
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_cl_model import BaseCLModel
from src.models.tinyol.autoencoder import TinyOLAutoencoder

ANOMALY_PERCENTILE_DEFAULT: int = 95


class TinyOLAnomalyDetector(BaseCLModel):
    """
    Détecteur d'anomalies basé sur l'autoencoder TinyOL.

    Entraîne le backbone autoencoder sur données normales (faulty=0) uniquement.
    Le score d'anomalie est l'erreur de reconstruction MSE par échantillon.
    Aucune tête OtO — compatible avec le scénario anomaly detection.

    Budget mémoire (input_dim=4, encoder [4,4,2]) :
        Linear(4→4) : 20 params → MEM: 80 B @ FP32 / 20 B @ INT8
        Linear(4→4) : 20 params → MEM: 80 B @ FP32 / 20 B @ INT8
        Linear(4→2) : 10 params → MEM: 40 B @ FP32 / 10 B @ INT8
        décodeur : ~46 params (symétrique) → MEM: 184 B @ FP32
        TOTAL : ~200 B @ FP32 (très confortable vs 64 Ko)

    Parameters
    ----------
    config : dict
        Sous-sections attendues :
        - ``backbone`` : input_dim, encoder_dims, decoder_dims, checkpoint_path
        - ``pretrain``  : optimizer, learning_rate, epochs, batch_size
        - ``anomaly_percentile`` (int, défaut 95)
        - ``anomaly_threshold`` (float | null, calculé sur Task 0 si null)
        - ``memory``    : target_ram_bytes, warn_if_above_bytes

    Notes
    -----
    Contrainte bottleneck : encoder_dims[-1] < input_dim pour que l'erreur de
    reconstruction soit un signal d'anomalie utile. Recommandé : [4, 4, 2] avec input_dim=4.
    L'encodeur [8, 8, 8] du tinyol_monitoring_config.yaml est inadapté (bottleneck 8D > 4D input).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        backbone_cfg = config["backbone"]
        self.autoencoder = TinyOLAutoencoder(
            input_dim=backbone_cfg["input_dim"],
            encoder_dims=tuple(backbone_cfg["encoder_dims"]),
            decoder_dims=tuple(backbone_cfg["decoder_dims"]),
        )
        self._checkpoint_path: Path | None = (
            Path(backbone_cfg["checkpoint_path"]) if backbone_cfg.get("checkpoint_path") else None
        )

        pretrain_cfg = config["pretrain"]
        self._lr: float = float(pretrain_cfg["learning_rate"])
        self._n_epochs: int = int(pretrain_cfg["epochs"])
        self._batch_size: int = int(pretrain_cfg.get("batch_size", 64))
        self._pretrain_optimizer_name: str = pretrain_cfg.get("optimizer", "adam")

        self.anomaly_percentile: int = int(config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT))
        self.anomaly_threshold_: float | None = config.get("anomaly_threshold", None)

        self._buffer_X: list[np.ndarray] = []
        self._fitted: bool = False
        self._task_id: int = -1
        self._device = torch.device("cpu")  # MCU-ciblé : CPU uniquement

        self._optimizer: torch.optim.Optimizer = self._build_optimizer()

    # ------------------------------------------------------------------
    # Interface BaseCLModel
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prédiction binaire 0=normal / 1=anomalie via seuil sur le score MSE.

        Parameters
        ----------
        x : np.ndarray [N, input_dim]

        Returns
        -------
        np.ndarray [N], dtype=int64

        Raises
        ------
        RuntimeError
            Si le seuil n'a pas encore été calculé (appeler on_task_end sur Task 0).
        """
        if self.anomaly_threshold_ is None:
            raise RuntimeError(
                "Seuil non calculé. Appeler on_task_end() sur Task 0 d'abord."
            )
        scores = self.anomaly_score(x)
        return (scores >= self.anomaly_threshold_).astype(np.int64)

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Bufferise le batch (labels ignorés — entraînement non supervisé).

        L'entraînement réel de l'autoencoder est différé à on_task_end().
        Retourne 0.0 comme proxy de loss (les pertes sont calculées en on_task_end).

        Parameters
        ----------
        x : np.ndarray [batch_size, input_dim]
        y : np.ndarray [batch_size] — ignoré en mode anomaly detection

        Returns
        -------
        float — 0.0 (placeholder)
        """
        self._buffer_X.append(x.copy())
        return 0.0

    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        """
        Entraîne l'autoencoder sur les données bufferisées, puis vide le buffer.

        Sur Task 0 uniquement, calcule le seuil d'anomalie au percentile configuré.

        Parameters
        ----------
        task_id : int
            1-based. Task 1 = Pump (calibration du seuil).
        dataloader : Any
            Non utilisé — les données sont lues depuis self._buffer_X.
        """
        self._task_id = task_id
        if not self._buffer_X:
            return

        X_all = np.concatenate(self._buffer_X, axis=0).astype(np.float32)
        self._buffer_X = []

        self._train_autoencoder(X_all)
        self._fitted = True

        if task_id == 1 and self.anomaly_threshold_ is None:
            self.set_anomaly_threshold(X_all)
            print(
                f"  [TinyOL AE] Seuil calculé sur Task {task_id} : "
                f"{self.anomaly_threshold_:.6f} (percentile {self.anomaly_percentile})"
            )

        if self._checkpoint_path is not None:
            self.save(str(self._checkpoint_path).replace(".pt", f"_task{task_id}.pt"))

    def count_parameters(self) -> int:
        """Nombre total de paramètres de l'autoencoder (encodeur + décodeur)."""
        return sum(p.numel() for p in self.autoencoder.parameters())

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """
        Estime la RAM de l'autoencoder en octets.

        En inférence MCU seul l'encodeur est chargé.
        """
        n_params = sum(p.numel() for p in self.autoencoder.parameters())
        bytes_per_param = 4 if dtype == "fp32" else 1
        return n_params * bytes_per_param

    def save(self, path: str) -> None:
        """Sauvegarde les poids de l'autoencoder."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.autoencoder.state_dict(), p)

    def load(self, path: str) -> None:
        """Charge les poids de l'autoencoder."""
        state = torch.load(Path(path), map_location=self._device, weights_only=True)
        self.autoencoder.load_state_dict(state)
        self._fitted = True

    # ------------------------------------------------------------------
    # Méthodes anomaly detection
    # ------------------------------------------------------------------

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """
        Retourne le score d'anomalie : MSE de reconstruction par échantillon.

        Un score élevé indique une forte déviation par rapport aux données normales
        vues à l'entraînement.

        Parameters
        ----------
        x : np.ndarray [N, input_dim], dtype=float32

        Returns
        -------
        np.ndarray [N], dtype=float32
            Erreurs de reconstruction MSE par échantillon.

        Notes
        -----
        # MEM: batch N × input_dim × 4 B @ FP32 (activations forward, temporaire)
        """
        if not self._fitted:
            raise RuntimeError("TinyOLAnomalyDetector non entraîné. Appeler on_task_end() d'abord.")

        self.autoencoder.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32)).to(self._device)
            _, x_hat = self.autoencoder(x_t)
            # MSE par échantillon (moyenne sur les features)
            scores = ((x_hat - x_t) ** 2).mean(dim=1)  # MEM: N × 4 B @ FP32
        return scores.cpu().numpy().astype(np.float32)

    def set_anomaly_threshold(
        self,
        x_normal: np.ndarray,
        percentile: int | None = None,
    ) -> float:
        """
        Calcule et fixe le seuil d'anomalie depuis des données normales.

        Parameters
        ----------
        x_normal : np.ndarray [N, input_dim]
            Données normales (faulty=0) — typiquement le train de Task 0.
        percentile : int | None
            Percentile de la distribution des scores normaux. Défaut : self.anomaly_percentile.

        Returns
        -------
        float
            Seuil calculé.
        """
        p = percentile if percentile is not None else self.anomaly_percentile
        scores = self.anomaly_score(x_normal)
        self.anomaly_threshold_ = float(np.percentile(scores, p))
        return self.anomaly_threshold_

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self._pretrain_optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.autoencoder.parameters(), lr=self._lr)
        return torch.optim.SGD(self.autoencoder.parameters(), lr=self._lr)

    def _train_autoencoder(self, X: np.ndarray) -> None:
        """Entraîne l'autoencoder sur X pendant self._n_epochs epochs (refit complet)."""
        self._optimizer = self._build_optimizer()
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self.autoencoder.train()
        for _ in range(self._n_epochs):
            for (xb,) in loader:
                xb = xb.to(self._device)
                _, x_hat = self.autoencoder(xb)
                loss = F.mse_loss(x_hat, xb)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
        self.autoencoder.eval()
