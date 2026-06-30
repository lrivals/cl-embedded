"""
test_ewc_multiclass.py — Tests unitaires pour EWCMlpMulticlass.

Invariants testés :
    1. Forward pass : output shape == (batch_size, n_classes)
    2. Softmax normalisé : probs.sum(dim=1) ≈ 1.0
    3. CrossEntropy loss calculable + backprop
    4. EWC penalty == 0.0 avant consolidation
    5. EWC penalty > 0.0 après consolidation
    6. Predict : argmax correct (valeur connue)
    7. F1-macro non nul sur données synthétiques séparables
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass


@pytest.fixture
def model_10() -> EWCMlpMulticlass:
    """Modèle CWRU : 9 features, 10 classes."""
    return EWCMlpMulticlass(input_dim=9, n_classes=10, ewc_lambda=400.0)


@pytest.fixture
def model_3() -> EWCMlpMulticlass:
    """Modèle Paderborn : 9 features, 3 classes."""
    return EWCMlpMulticlass(input_dim=9, n_classes=3, ewc_lambda=400.0)


def test_forward_shape_10_classes(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    logits = model_10(x)
    assert logits.shape == (8, 10), f"Shape : {logits.shape} (attendu (8, 10))"


def test_forward_shape_3_classes(model_3: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    logits = model_3(x)
    assert logits.shape == (8, 3), f"Shape : {logits.shape} (attendu (8, 3))"


def test_softmax_normalized(model_10: EWCMlpMulticlass) -> None:
    """softmax(logits).sum(dim=1) == 1.0 pour chaque exemple."""
    x = torch.randn(16, 9)
    with torch.no_grad():
        logits = model_10(x)
    probs = torch.softmax(logits, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(16), atol=1e-5), (
        f"Softmax non normalisé, max abs diff = {(sums - 1).abs().max().item():.2e}"
    )


def test_crossentropy_backprop(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(8, 9)
    y = torch.randint(0, 10, (8,))
    logits = model_10(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    for name, param in model_10.named_parameters():
        assert param.grad is not None, f"Gradient None pour {name}"


def test_ewc_penalty_zero_before_consolidation(model_10: EWCMlpMulticlass) -> None:
    assert model_10.ewc_penalty().item() == 0.0


def test_ewc_penalty_nonzero_after_consolidation(model_10: EWCMlpMulticlass) -> None:
    x = torch.randn(64, 9)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    # Entraîner brièvement
    opt = torch.optim.SGD(model_10.parameters(), lr=0.01)
    for xb, yb in loader:
        opt.zero_grad()
        nn.CrossEntropyLoss()(model_10(xb), yb).backward()
        opt.step()

    model_10.consolidate(loader, n_samples=32)

    # Perturber les paramètres
    with torch.no_grad():
        for p in model_10.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    assert model_10.ewc_penalty().item() > 0.0


def test_predict_argmax(model_10: EWCMlpMulticlass) -> None:
    """predict() retourne l'argmax des logits."""
    x = torch.randn(8, 9)
    model_10.eval()  # désactiver Dropout pour que les deux passes soient identiques
    with torch.no_grad():
        logits = model_10(x)
        preds = model_10.predict(x)
    expected = logits.argmax(dim=1)
    assert torch.all(preds == expected), "predict() != argmax(logits)"


def test_f1_macro_nonzero_on_separable_data(model_10: EWCMlpMulticlass) -> None:
    """F1-macro > 0 après entraînement sur données synthétiques séparables."""
    from src.evaluation.multiclass_metrics import compute_f1_macro

    # Données synthétiques : clusters bien séparés
    np.random.seed(42)
    x_list, y_list = [], []
    for cls in range(10):
        x_list.append(np.random.randn(20, 9).astype(np.float32) + cls * 5)
        y_list.extend([cls] * 20)
    x = torch.tensor(np.vstack(x_list))
    y = torch.tensor(y_list, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    opt = torch.optim.SGD(model_10.parameters(), lr=0.05)
    for _ in range(30):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model_10(xb), yb).backward()
            opt.step()

    with torch.no_grad():
        preds = model_10.predict(x).numpy()
    f1 = compute_f1_macro(y.numpy(), preds)
    assert f1 > 0.0, f"F1-macro = 0 après entraînement ({f1})"
