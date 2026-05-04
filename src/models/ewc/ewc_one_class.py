"""
ewc_one_class.py — Détecteur d'anomalies one-class basé sur EWC Online.

Approche : MLP autoencoder (encoder-decoder) entraîné sur données normales
uniquement. Le score d'anomalie est l'erreur de reconstruction MSE. La
régularisation EWC empêche l'oubli des représentations des tâches précédentes.

Scénario CL : domain-incremental anomaly detection.
    - Entraînement : données normales (faulty=0) uniquement
    - Inférence : données normales + défectueuses → AUROC

RAM estimée (input_dim=4, hidden_dim=8) :
    Encoder : Linear(4→8) + Linear(8→4) = (32+8 + 32+4) × 4 = 304 B @ FP32
    Decoder : Linear(4→8) + Linear(8→4) = 304 B @ FP32
    Fisher  : ~304 B @ FP32 (diagonale)
    Snapshot θ* : ~304 B @ FP32
    TOTAL   : ~1.2 Ko @ FP32  ✅ << 64 Ko cible STM32N6

Références :
    Kirkpatrick2017EWC — EWC (régularisation)
    Ren2021TinyOL — inspiration architecture autoencoder embarqué
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ANOMALY_PERCENTILE_DEFAULT: int = 95
N_FISHER_SAMPLES_DEFAULT: int = 200


class _MLPAutoencoder(nn.Module):
    """
    Autoencoder MLP symétrique pour la reconstruction de vecteurs tabulaires.

    Architecture :
        Encoder : Linear(input_dim → hidden_dim) + ReLU → Linear(hidden_dim → bottleneck)
        Decoder : Linear(bottleneck → hidden_dim) + ReLU → Linear(hidden_dim → input_dim)

    Le bottleneck < input_dim crée une contrainte de compression utile pour la
    détection d'anomalies (points hors distribution = reconstruction dégradée).

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée. Default : 4.
    hidden_dim : int
        Dimension des couches cachées. Default : 8.
    bottleneck_dim : int
        Dimension du code latent (doit être < input_dim). Default : 2.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        bottleneck_dim: int = 2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        # MEM: Linear(4→8): (4×8+8)×4 = 160 B @ FP32 / 40 B @ INT8
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        # MEM: Linear(8→2): (8×2+2)×4 = 72 B @ FP32 / 18 B @ INT8
        self.enc2 = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder (symétrique)
        # MEM: Linear(2→8): (2×8+8)×4 = 96 B @ FP32 / 24 B @ INT8
        self.dec1 = nn.Linear(bottleneck_dim, hidden_dim)
        # MEM: Linear(8→4): (8×4+4)×4 = 144 B @ FP32 / 36 B @ INT8
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor [N, input_dim]

        Returns
        -------
        z : Tensor [N, bottleneck_dim] — code latent
        x_hat : Tensor [N, input_dim] — reconstruction
        """
        # MEM: activations encoder: N×8×4 B @ FP32 (temporaire)
        z = torch.relu(self.enc1(x))
        z = self.enc2(z)  # bottleneck — pas d'activation (linéaire)
        # MEM: activations decoder: N×8×4 B @ FP32 (temporaire)
        x_hat = torch.relu(self.dec1(z))
        x_hat = self.dec2(x_hat)  # sortie linéaire pour MSE
        return z, x_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EWCOneClassDetector:
    """
    Détecteur d'anomalies one-class avec régularisation EWC Online.

    Entraîne un MLP autoencoder sur les données normales de chaque tâche.
    La régularisation EWC préserve les poids importants pour les tâches
    précédentes, réduisant l'oubli des distributions de normalité passées.

    Interface compatible avec run_anomaly_detection_scenario() (fit_task API) :
        detector.fit_task(X_normal, task_id)
        scores = detector.anomaly_score(X_test)
        preds  = detector.predict(X_test)

    Parameters
    ----------
    config : dict
        Sous-sections attendues :
        - ``ewc_one_class.hidden_dim``        : int (default 8)
        - ``ewc_one_class.bottleneck_dim``    : int (default 2)
        - ``ewc_one_class.learning_rate``     : float (default 1e-3)
        - ``ewc_one_class.epochs_per_task``   : int (default 20)
        - ``ewc_one_class.batch_size``        : int (default 64)
        - ``ewc_one_class.ewc_lambda``        : float (default 500.0)
        - ``ewc_one_class.ewc_gamma``         : float (default 0.9)
        - ``ewc_one_class.n_fisher_samples``  : int (default 200)
        - ``ewc_one_class.cl_strategy``       : "refit" | "ewc" (default "ewc")
        - ``data.n_features``                 : int (default 4)
        - ``anomaly_percentile``              : int (default 95)
        - ``anomaly_threshold``               : float | null

    Notes
    -----
    cl_strategy="refit" : réentraîne from scratch à chaque tâche (pas d'EWC, baseline).
    cl_strategy="ewc"   : applique la régularisation EWC (comportement par défaut).
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("ewc_one_class", {})
        data_cfg = config.get("data", {})

        self._input_dim: int = int(data_cfg.get("n_features", 4))
        self._hidden_dim: int = int(cfg.get("hidden_dim", 8))
        self._bottleneck_dim: int = int(cfg.get("bottleneck_dim", 2))
        self._lr: float = float(cfg.get("learning_rate", 1e-3))
        self._epochs: int = int(cfg.get("epochs_per_task", 20))
        self._batch_size: int = int(cfg.get("batch_size", 64))
        self._ewc_lambda: float = float(cfg.get("ewc_lambda", 500.0))
        self._ewc_gamma: float = float(cfg.get("ewc_gamma", 0.9))
        self._n_fisher_samples: int = int(cfg.get("n_fisher_samples", N_FISHER_SAMPLES_DEFAULT))
        self._cl_strategy: str = cfg.get("cl_strategy", "ewc")

        self.anomaly_percentile: int = int(config.get("anomaly_percentile", ANOMALY_PERCENTILE_DEFAULT))
        self.threshold_: float | None = config.get("anomaly_threshold", None)

        self._device = torch.device("cpu")  # MCU-ciblé
        self._model = _MLPAutoencoder(
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            bottleneck_dim=self._bottleneck_dim,
        ).to(self._device)

        self._fisher: dict[str, torch.Tensor] | None = None
        self._theta_star: dict[str, torch.Tensor] | None = None
        self.task_id_: int = -1
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Interface anomaly detection (fit_task API)
    # ------------------------------------------------------------------

    def fit_task(self, X: np.ndarray, task_id: int) -> "EWCOneClassDetector":
        """
        Entraîne l'autoencoder sur les données normales d'une tâche.

        Sur Task 0 : initialise le seuil d'anomalie (percentile sur MSE de reconstruction).
        Sur Task 1+ : applique la régularisation EWC pour limiter l'oubli.

        Parameters
        ----------
        X : np.ndarray [N, input_dim]
            Données d'entraînement normales (faulty=0 uniquement).
        task_id : int
            Index 0-based de la tâche courante.

        Returns
        -------
        self
        """
        self.task_id_ = task_id

        if self._cl_strategy == "refit":
            self._reset_model()

        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        self._model.train()
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for (xb,) in loader:
                optimizer.zero_grad()
                _, x_hat = self._model(xb)

                loss = F.mse_loss(x_hat, xb)

                if self._cl_strategy == "ewc" and self._fisher is not None:
                    ewc_penalty = self._compute_ewc_penalty()
                    loss = loss + (self._ewc_lambda / 2.0) * ewc_penalty

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"  [EWCOneClass] Tâche {task_id}, epoch {epoch+1}/{self._epochs} "
                    f"— loss={epoch_loss/len(loader):.6f}"
                )

        self._model.eval()
        self._fitted = True

        if task_id == 0 and self.threshold_ is None:
            scores = self.anomaly_score(X)
            self.threshold_ = float(np.percentile(scores, self.anomaly_percentile))
            print(
                f"  [EWCOneClass] Seuil calculé sur Task 0 : {self.threshold_:.6f} "
                f"(percentile {self.anomaly_percentile})"
            )

        self._update_fisher(loader)
        self._theta_star = {
            name: param.detach().clone()
            for name, param in self._model.named_parameters()
        }
        print(
            f"  [EWCOneClass] Tâche {task_id} terminée — "
            f"RAM estimée={self._estimate_ram_bytes()} B"
        )
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Score d'anomalie = MSE de reconstruction par échantillon.

        Un score élevé indique une déviation par rapport aux données normales
        vues à l'entraînement.

        Parameters
        ----------
        X : np.ndarray [N, input_dim]

        Returns
        -------
        np.ndarray [N], dtype=float32
            Erreurs de reconstruction MSE.

        Notes
        -----
        # MEM: N × input_dim × 4 B @ FP32 (activations forward, temporaire)
        """
        if not self._fitted:
            raise RuntimeError(
                "EWCOneClassDetector non entraîné. Appeler fit_task() d'abord."
            )
        self._model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
            _, x_hat = self._model(x_t)
            # MSE par échantillon (moyenne sur les features)
            scores = ((x_hat - x_t) ** 2).mean(dim=1)  # MEM: N × 4 B @ FP32
        return scores.cpu().numpy().astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction binaire 0=normal / 1=anomalie via seuil sur le score MSE.

        Parameters
        ----------
        X : np.ndarray [N, input_dim]

        Returns
        -------
        np.ndarray [N], dtype=int64

        Raises
        ------
        RuntimeError
            Si le seuil n'a pas été calculé (fit_task sur Task 0 requis).
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Seuil non calculé. Appeler fit_task(X, task_id=0) sur Task 0 d'abord."
            )
        scores = self.anomaly_score(X)
        return (scores >= self.threshold_).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy binaire. Labels utilisés en évaluation uniquement."""
        preds = self.predict(X)
        return float((preds == y.astype(np.int64)).mean())

    # ------------------------------------------------------------------
    # Méthodes internes EWC
    # ------------------------------------------------------------------

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Pénalité EWC : Σ F_i (θ_i - θ*_i)² sur tous les paramètres."""
        penalty = torch.tensor(0.0, device=self._device)
        for name, param in self._model.named_parameters():
            if name in self._fisher and name in self._theta_star:
                f = self._fisher[name].to(self._device)
                ts = self._theta_star[name].to(self._device)
                penalty += (f * (param - ts) ** 2).sum()
        return penalty

    def _update_fisher(self, loader: DataLoader) -> None:
        """
        Calcule et accumule la Fisher diagonale sur la tâche courante (reconstruction).

        Utilise le gradient de la loss MSE (pas de BCE puisque one-class non supervisé).
        La Fisher Online est mise à jour avec décroissance γ.
        """
        fisher_new: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self._device)
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

        self._model.eval()
        n_seen = 0
        n_batches = 0

        for (xb,) in loader:
            if n_seen >= self._n_fisher_samples:
                break
            xb = xb.to(self._device)
            self._model.zero_grad()
            _, x_hat = self._model(xb)
            loss = F.mse_loss(x_hat, xb)
            loss.backward()

            for name, param in self._model.named_parameters():
                if param.grad is not None and name in fisher_new:
                    fisher_new[name] += param.grad.detach() ** 2

            n_seen += xb.size(0)
            n_batches += 1

        if n_batches > 0:
            fisher_new = {name: f / n_batches for name, f in fisher_new.items()}

        if self._fisher is None:
            self._fisher = {name: f.clone() for name, f in fisher_new.items()}
        else:
            self._fisher = {
                name: self._ewc_gamma * self._fisher[name] + fisher_new[name]
                for name in fisher_new
            }

    def _reset_model(self) -> None:
        """Réinitialise les poids du modèle (pour cl_strategy='refit')."""
        self._model = _MLPAutoencoder(
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            bottleneck_dim=self._bottleneck_dim,
        ).to(self._device)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Nombre total de paramètres de l'autoencoder."""
        return self._model.count_parameters()

    def _estimate_ram_bytes(self) -> int:
        """
        Estime la RAM totale (modèle + Fisher + θ*) en octets @ FP32.

        # MEM: n_params × 4 B (modèle) + n_params × 4 B (Fisher) + n_params × 4 B (θ*)
        """
        n = self.count_parameters()
        ram_model = n * 4
        ram_fisher = n * 4 if self._fisher is not None else 0
        ram_theta = n * 4 if self._theta_star is not None else 0
        return ram_model + ram_fisher + ram_theta

    def summary(self) -> str:
        n = self.count_parameters()
        ram = self._estimate_ram_bytes()
        threshold = f"{self.threshold_:.6f}" if self.threshold_ is not None else "—"
        return (
            f"EWCOneClassDetector | "
            f"arch=[{self._input_dim}→{self._hidden_dim}→{self._bottleneck_dim}]→[{self._hidden_dim}→{self._input_dim}] | "
            f"strategy={self._cl_strategy} | λ={self._ewc_lambda} | γ={self._ewc_gamma} | "
            f"params={n} | RAM≈{ram/1024:.2f} Ko FP32 | threshold={threshold}"
        )

    def save(self, path: str | Path) -> None:
        """Sauvegarde les poids et l'état EWC."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "fisher": self._fisher,
                "theta_star": self._theta_star,
                "threshold": self.threshold_,
                "task_id": self.task_id_,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        """Charge les poids et l'état EWC."""
        checkpoint = torch.load(Path(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._fisher = checkpoint.get("fisher")
        self._theta_star = checkpoint.get("theta_star")
        self.threshold_ = checkpoint.get("threshold")
        self.task_id_ = checkpoint.get("task_id", -1)
        self._fitted = True
