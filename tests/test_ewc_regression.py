"""
test_ewc_regression.py — Tests unitaires pour EWCMlpRegressor.

Invariants testés :
    1. Forward pass : output shape == (batch_size, 1)
    2. Output non borné : peut dépasser [0, 1] (régression linéaire, pas de Sigmoid)
    3. MSE loss calculable et propageable
    4. EWC penalty == 0.0 avant consolidation
    5. EWC penalty > 0.0 après consolidation
    6. Consolidation : Fisher et theta_star de même shape que les paramètres
    7. Backward transfer : RMSE sur tâche 1 ne dégrade pas catastrophiquement après tâche 2
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor


@pytest.fixture
def model() -> EWCMlpRegressor:
    return EWCMlpRegressor(input_dim=5, hidden_dims=[32, 16], ewc_lambda=400.0)


@pytest.fixture
def dummy_loader() -> DataLoader:
    """DataLoader factice : 64 exemples, input_dim=5, RUL ∈ [0, 125]."""
    x = torch.randn(64, 5)
    y = torch.rand(64) * 125
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_forward_shape(model: EWCMlpRegressor) -> None:
    """Output shape = (batch_size, 1)."""
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8, 1), f"Shape incorrecte : {out.shape} (attendu (8, 1))"


def test_output_unbounded(model: EWCMlpRegressor) -> None:
    """La sortie n'est pas bornée en [0, 1] (pas de Sigmoid)."""
    x = torch.randn(256, 5) * 10  # entrées larges
    with torch.no_grad():
        out = model(x).squeeze()
    # Au moins quelques sorties doivent dépasser [0, 1] avec des entrées larges
    # (sinon le modèle aurait une Sigmoid cachée)
    assert not (out.min() >= 0 and out.max() <= 1), (
        "Output borné en [0,1] — vérifier l'absence de Sigmoid finale"
    )


def test_mse_loss_backprop(model: EWCMlpRegressor) -> None:
    """MSE loss calculable + backpropagation sans erreur."""
    x = torch.randn(8, 5)
    y = torch.rand(8) * 125
    out = model(x).squeeze()
    loss = nn.MSELoss()(out, y)
    loss.backward()
    # Vérifier que les gradients existent
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient None pour {name}"


def test_ewc_penalty_before_consolidation(model: EWCMlpRegressor) -> None:
    """EWC penalty = 0.0 avant toute consolidation."""
    penalty = model.ewc_penalty()
    assert penalty.item() == 0.0, f"Penalty non nulle avant consolidation : {penalty.item()}"


def test_ewc_penalty_after_consolidation(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """EWC penalty > 0.0 après consolidation (Fisher non nulle)."""
    # Entraîner brièvement pour avoir des paramètres non nuls
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for x_batch, y_batch in dummy_loader:
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x_batch).squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    model.consolidate(dummy_loader, n_samples=32)

    # Modifier légèrement les paramètres
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    penalty = model.ewc_penalty()
    assert penalty.item() > 0.0, "Penalty nulle après consolidation — Fisher vide ?"


def test_consolidation_shapes(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """Fisher et theta_star de même shape que les paramètres."""
    model.consolidate(dummy_loader, n_samples=32)
    for name, param in model.named_parameters():
        assert name in model._fisher, f"Fisher manquante pour {name}"
        assert name in model._theta_star, f"theta_star manquant pour {name}"
        assert model._fisher[name].shape == param.shape, (
            f"Shape Fisher incorrecte pour {name}"
        )
        assert model._theta_star[name].shape == param.shape


def test_backward_transfer_not_catastrophic(
    model: EWCMlpRegressor,
    dummy_loader: DataLoader,
) -> None:
    """RMSE sur tâche 1 après entraînement sur tâche 2 n'est pas catastrophique (< 2× RMSE initial)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Tâche 1 : RUL normalisé ∈ [0.4, 1.0] (division par 125 pour éviter exploding gradients)
    x1 = torch.randn(64, 5)
    y1 = (50 + torch.rand(64) * 75) / 125.0
    loader1 = DataLoader(TensorDataset(x1, y1), batch_size=16)

    for _ in range(10):
        for xb, yb in loader1:
            optimizer.zero_grad()
            nn.MSELoss()(model(xb).squeeze(), yb).backward()
            optimizer.step()

    # RMSE initial sur tâche 1
    model.consolidate(loader1, n_samples=32)
    with torch.no_grad():
        rmse_task1_before = float(
            torch.sqrt(nn.MSELoss()(model(x1).squeeze(), y1)).item()
        )

    # Tâche 2 : RUL normalisé ∈ [0.0, 0.24] (distribution différente)
    x2 = torch.randn(64, 5)
    y2 = torch.rand(64) * 30 / 125.0
    loader2 = DataLoader(TensorDataset(x2, y2), batch_size=16)

    for _ in range(10):
        for xb, yb in loader2:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(xb).squeeze(), yb) + model.ewc_penalty()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        rmse_task1_after = float(
            torch.sqrt(nn.MSELoss()(model(x1).squeeze(), y1)).item()
        )

    # Avec EWC, la dégradation doit être < 2× (critère souple pour test unitaire)
    assert rmse_task1_after < rmse_task1_before * 2, (
        f"Oubli catastrophique : RMSE_before={rmse_task1_before:.2f}, "
        f"RMSE_after={rmse_task1_after:.2f}"
    )
