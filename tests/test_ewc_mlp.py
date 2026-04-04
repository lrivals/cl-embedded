"""
tests/test_ewc_mlp.py — Tests unitaires pour EWCMlpClassifier (S1-04).

Critères d'acceptation sprint S1-04 :
    pytest tests/test_ewc_mlp.py -v
"""

import torch

from src.models.ewc import EWCMlpClassifier


def test_forward_shape():
    """Le forward produit une sortie de forme [batch, 1] avec valeurs dans [0, 1]."""
    model = EWCMlpClassifier()
    x = torch.randn(32, 6)
    out = model(x)
    assert out.shape == (32, 1), f"Forme attendue (32, 1), obtenu {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid doit borner [0, 1]"


def test_bce_loss_task1():
    """Sur Task 1 (fisher=None), la perte doit être pure BCE scalaire positive."""
    model = EWCMlpClassifier()
    x = torch.randn(16, 6)
    y = torch.randint(0, 2, (16, 1)).float()
    loss = model.ewc_loss(x, y, fisher=None, theta_star=None, ewc_lambda=1000.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([]), "La perte doit être un scalaire"
    assert loss.item() > 0


def test_ewc_loss_increases_with_lambda():
    """La régularisation EWC doit augmenter avec λ."""
    model = EWCMlpClassifier()
    x = torch.randn(16, 6)
    y = torch.randint(0, 2, (16, 1)).float()
    # Fisher non-nulle : simule post-Task 1
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    theta_star = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    loss_low = model.ewc_loss(x, y, fisher=fisher, theta_star=theta_star, ewc_lambda=10.0)
    loss_high = model.ewc_loss(x, y, fisher=fisher, theta_star=theta_star, ewc_lambda=10000.0)
    assert loss_high.item() > loss_low.item(), "La perte EWC doit croître avec λ"


def test_n_params():
    """Vérifie le nombre total de paramètres (769 attendu pour input_dim=6)."""
    model = EWCMlpClassifier()
    n = sum(p.numel() for p in model.parameters())
    assert n == 769, f"Attendu 769 params, obtenu {n}"


def test_theta_star_detached():
    """Le snapshot θ* ne doit pas partager le graphe de calcul."""
    model = EWCMlpClassifier()
    theta_star = model.get_theta_star()
    assert len(theta_star) > 0, "get_theta_star() ne doit pas retourner un dict vide"
    for name, tensor in theta_star.items():
        assert not tensor.requires_grad, f"{name} ne devrait pas avoir requires_grad=True"
