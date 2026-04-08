"""
tests/test_tinyol_autoencoder.py — Tests unitaires pour TinyOLAutoencoder.

Exécution :
    pytest tests/test_tinyol_autoencoder.py -v
"""

from __future__ import annotations

import torch
import pytest

from src.models.tinyol import TinyOLAutoencoder


def test_encoder_output_shape():
    """L'encodeur doit produire un embedding de dimension 8."""
    model = TinyOLAutoencoder()
    x = torch.randn(16, 25)
    z = model.encode(x)
    assert z.shape == (16, 8), f"z.shape attendu (16, 8), obtenu {z.shape}"


def test_decoder_output_shape():
    """Le décodeur doit reconstruire un vecteur de dimension 25."""
    model = TinyOLAutoencoder()
    z = torch.randn(16, 8)
    x_hat = model.decode(z)
    assert x_hat.shape == (16, 25), f"x_hat.shape attendu (16, 25), obtenu {x_hat.shape}"


def test_forward_shapes():
    """forward() doit retourner (z, x_hat) avec les bonnes dimensions."""
    model = TinyOLAutoencoder()
    x = torch.randn(8, 25)
    z, x_hat = model(x)
    assert z.shape == (8, 8)
    assert x_hat.shape == (8, 25)


def test_n_encoder_params():
    """L'encodeur doit avoir exactement 1 496 paramètres (conforme tinyol_spec.md §2.1)."""
    model = TinyOLAutoencoder()
    n = model.n_encoder_params()
    assert n == 1496, f"Attendu 1 496 params encodeur, obtenu {n}"


def test_freeze_encoder_disables_grad():
    """Après freeze_encoder(), l'encodeur ne doit plus avoir de gradients."""
    model = TinyOLAutoencoder()
    model.freeze_encoder()
    for name, param in model.named_parameters():
        if name.startswith("enc"):
            assert not param.requires_grad, (
                f"Paramètre encodeur {name} a encore requires_grad=True après freeze"
            )


def test_freeze_encoder_decoder_still_trainable():
    """Après freeze_encoder(), le décodeur doit rester entraînable."""
    model = TinyOLAutoencoder()
    model.freeze_encoder()
    for name, param in model.named_parameters():
        if name.startswith("dec"):
            assert param.requires_grad, (
                f"Paramètre décodeur {name} n'est plus entraînable après freeze encodeur"
            )


def test_reconstruction_loss_positive_differentiable():
    """La perte MSE doit être positive et permettre la rétropropagation."""
    model = TinyOLAutoencoder()
    x = torch.randn(8, 25)
    z, x_hat = model(x)
    loss = model.reconstruction_loss(x, x_hat)
    assert loss.item() >= 0
    loss.backward()
    assert model.enc1.weight.grad is not None


def test_encode_batch_size_1():
    """Cas MCU : batch_size=1 (inférence échantillon par échantillon)."""
    model = TinyOLAutoencoder()
    model.freeze_encoder()
    x = torch.randn(1, 25)
    with torch.no_grad():
        z = model.encode(x)
    assert z.shape == (1, 8)
