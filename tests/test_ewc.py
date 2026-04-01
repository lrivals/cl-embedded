"""
tests/test_ewc.py — Tests unitaires pour EWCMlpClassifier.

Exécution :
    pytest tests/test_ewc.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.ewc.ewc_mlp import EWCMlpClassifier


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def model():
    return EWCMlpClassifier(input_dim=6, hidden_dims=[32, 16], ewc_lambda=1000.0)


@pytest.fixture
def dummy_batch():
    x = torch.randn(32, 6)
    y = torch.randint(0, 2, (32, 1)).float()
    return x, y


@pytest.fixture
def simple_dataloader(dummy_batch):
    x, y = dummy_batch
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


# ------------------------------------------------------------------
# Tests d'architecture
# ------------------------------------------------------------------

def test_forward_shape(model, dummy_batch):
    """Le forward produit une sortie de forme [batch, 1]."""
    x, _ = dummy_batch
    out = model(x)
    assert out.shape == (32, 1), f"Forme attendue (32, 1), obtenu {out.shape}"


def test_forward_range(model, dummy_batch):
    """La sortie sigmoid est dans [0, 1]."""
    x, _ = dummy_batch
    out = model(x).detach().numpy()
    assert np.all(out >= 0) and np.all(out <= 1), "Sortie hors de [0, 1]"


def test_count_parameters(model):
    """Le nombre de paramètres est dans la plage attendue (620–900)."""
    n = model.count_parameters()
    assert 620 <= n <= 900, f"Nombre de paramètres inattendu : {n}"


# ------------------------------------------------------------------
# Tests de contraintes MCU
# ------------------------------------------------------------------

def test_ram_within_budget(model):
    """La RAM estimée est < 64 Ko (cible STM32N6)."""
    ram = model.estimate_ram_bytes("fp32")
    assert ram < 65_536, f"RAM estimée ({ram} B) > 64 Ko"


def test_ram_with_ewc_overhead(model, simple_dataloader):
    """Après initialisation EWC, la RAM (poids + Fisher + snapshot) reste < 64 Ko."""
    model.update_ewc_state(simple_dataloader, torch.device("cpu"))
    ram = model.estimate_ram_bytes("fp32")
    assert ram < 65_536, f"RAM EWC ({ram} B) > 64 Ko"


# ------------------------------------------------------------------
# Tests EWC
# ------------------------------------------------------------------

def test_ewc_loss_before_init(model, dummy_batch):
    """Avant initialisation, la perte EWC est identique à la perte BCE."""
    x, y = dummy_batch
    total_loss, components = model.ewc_loss(x, y)
    assert components["ewc_reg"] == 0.0, "EWC reg doit être 0 avant init"
    assert abs(components["bce"] - components["total"]) < 1e-6


def test_ewc_state_update(model, simple_dataloader):
    """Après on_task_end, l'état EWC est initialisé."""
    assert not model._ewc_initialized
    model.update_ewc_state(simple_dataloader, torch.device("cpu"))
    assert model._ewc_initialized
    assert len(model._fisher) > 0
    assert len(model._params_star) > 0


def test_ewc_loss_after_init(model, simple_dataloader, dummy_batch):
    """Après initialisation, le terme EWC est non nul si les poids ont changé."""
    model.update_ewc_state(simple_dataloader, torch.device("cpu"))

    # Modifier les poids pour créer une divergence
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    x, y = dummy_batch
    _, components = model.ewc_loss(x, y)
    assert components["ewc_reg"] > 0.0, "EWC reg doit être > 0 après divergence des poids"


def test_no_gradient_leak_after_freezing(model, dummy_batch):
    """
    Simule le comportement MCU : vérifier que torch.no_grad() bloque les gradients.
    (Applicable quand EWC est utilisé avec un backbone gelé en contexte mixte.)
    """
    x, y = dummy_batch
    x.requires_grad_(True)

    with torch.no_grad():
        out = model(x)

    # Pas de gradient propagé si torch.no_grad()
    assert out.grad_fn is None, "grad_fn doit être None dans torch.no_grad()"


# ------------------------------------------------------------------
# Tests sauvegarde/chargement
# ------------------------------------------------------------------

def test_save_load_state(model, simple_dataloader, tmp_path):
    """Le modèle sauvegardé et rechargé produit les mêmes prédictions."""
    model.update_ewc_state(simple_dataloader, torch.device("cpu"))

    save_path = str(tmp_path / "ewc_model.pt")
    model.save_state(save_path)

    model2 = EWCMlpClassifier(input_dim=6, hidden_dims=[32, 16])
    model2.load_state(save_path)

    x = torch.randn(8, 6)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Prédictions divergent après chargement"
    assert model2._ewc_initialized, "État EWC non restauré après chargement"
