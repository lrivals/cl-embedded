"""
Tests unitaires pour le modèle TinyOL (M1).

Couvre :
- TinyOLAutoencoder : shape forward, reconstruction MSE, gel backbone
- OtOHead          : shape forward, valeur sigmoid, compte params
- TinyOLOnlineTrainer : update échantillon, prédiction, gel backbone vérifié
- Budget mémoire   : n_params encodeur = 1 496, OtO = 10

Ne requiert pas le Dataset 1 — utilise des tenseurs synthétiques.

Références : tinyol_spec.md §2, §6
"""

import pytest
import torch

from src.models.tinyol.autoencoder import TinyOLAutoencoder
from src.models.tinyol.oto_head import OtOHead, TinyOLOnlineTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> dict:
    """Configuration minimale pour les tests — clés identiques à tinyol_config.yaml."""
    return {
        "backbone": {
            "input_dim": 25,
            "encoder_dims": [32, 16, 8],
            "decoder_dims": [16, 32, 25],
        },
        "oto_head": {
            "learning_rate": 1e-2,
            "momentum": 0.0,
        },
    }


@pytest.fixture
def autoencoder(default_config) -> TinyOLAutoencoder:
    cfg = default_config["backbone"]
    return TinyOLAutoencoder(
        input_dim=cfg["input_dim"],
        encoder_dims=tuple(cfg["encoder_dims"]),
        decoder_dims=tuple(cfg["decoder_dims"]),
    )


@pytest.fixture
def oto_head() -> OtOHead:
    return OtOHead(input_dim=9)


@pytest.fixture
def trainer(autoencoder, oto_head, default_config) -> TinyOLOnlineTrainer:
    return TinyOLOnlineTrainer(autoencoder, oto_head, default_config)


@pytest.fixture
def dummy_window() -> torch.Tensor:
    """Fenêtre synthétique normalisée, shape [25]."""
    torch.manual_seed(42)
    return torch.randn(25)


@pytest.fixture
def dummy_label_normal() -> torch.Tensor:
    return torch.tensor(0.0)


@pytest.fixture
def dummy_label_fault() -> torch.Tensor:
    return torch.tensor(1.0)


# ---------------------------------------------------------------------------
# TestTinyOLAutoencoder
# ---------------------------------------------------------------------------


class TestTinyOLAutoencoder:
    def test_forward_output_shapes(self, autoencoder, dummy_window):
        """L'autoencoder retourne (embedding, reconstruction) avec les bonnes shapes."""
        x = dummy_window.unsqueeze(0)  # [1, 25]
        z, x_hat = autoencoder(x)
        assert z.shape == (1, 8), f"Embedding shape attendu (1,8), obtenu {z.shape}"
        assert x_hat.shape == (1, 25), f"Reconstruction shape attendu (1,25), obtenu {x_hat.shape}"

    def test_encoder_param_count(self, autoencoder):
        """L'encodeur doit avoir exactement 1 496 paramètres (tinyol_spec.md §2.1)."""
        assert autoencoder.n_encoder_params() == 1496, (
            f"Attendu 1496 params encodeur, obtenu {autoencoder.n_encoder_params()}"
        )

    def test_reconstruction_loss_positive(self, autoencoder, dummy_window):
        """La loss MSE de reconstruction est un scalaire positif."""
        x = dummy_window.unsqueeze(0)
        z, x_hat = autoencoder(x)
        loss = autoencoder.reconstruction_loss(x, x_hat)
        assert loss.item() >= 0.0

    def test_batch_forward(self, autoencoder):
        """L'autoencoder fonctionne sur un batch de taille > 1."""
        x_batch = torch.randn(8, 25)
        z, x_hat = autoencoder(x_batch)
        assert z.shape == (8, 8)
        assert x_hat.shape == (8, 25)


# ---------------------------------------------------------------------------
# TestOtOHead
# ---------------------------------------------------------------------------


class TestOtOHead:
    def test_forward_shape_1d(self, oto_head):
        """OtOHead accepte une entrée 1D [9] et retourne un scalaire."""
        x = torch.randn(9)
        y = oto_head(x)
        assert y.shape == (1,), f"Shape attendu (1,), obtenu {y.shape}"

    def test_forward_shape_batched(self, oto_head):
        """OtOHead accepte un batch [B, 9] et retourne [B, 1]."""
        x = torch.randn(4, 9)
        y = oto_head(x)
        assert y.shape == (4, 1)

    def test_output_in_range(self, oto_head):
        """La sortie Sigmoid est dans [0, 1] pour toute entrée."""
        x = torch.randn(100, 9) * 100  # entrées extrêmes
        y = oto_head(x)
        assert (y >= 0.0).all() and (y <= 1.0).all()

    def test_param_count(self, oto_head):
        """OtOHead doit avoir exactement 10 paramètres (tinyol_spec.md §2.2)."""
        assert oto_head.n_params() == 10, (
            f"Attendu 10 params OtO, obtenu {oto_head.n_params()}"
        )


# ---------------------------------------------------------------------------
# TestTinyOLOnlineTrainer
# ---------------------------------------------------------------------------


class TestTinyOLOnlineTrainer:
    def test_backbone_frozen(self, trainer):
        """Après initialisation, aucun paramètre du backbone ne doit avoir requires_grad=True."""
        for p in trainer.autoencoder.parameters():
            assert not p.requires_grad, "Backbone doit être complètement gelé"

    def test_oto_head_trainable(self, trainer):
        """Les paramètres de la tête OtO doivent être entraînables."""
        for p in trainer.oto_head.parameters():
            assert p.requires_grad, "OtO head doit être entraînable"

    def test_update_returns_float(self, trainer, dummy_window, dummy_label_normal):
        """trainer.update() retourne un float (loss BCE)."""
        loss = trainer.update(dummy_window, dummy_label_normal)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_modifies_oto_weights(self, trainer, dummy_window, dummy_label_fault):
        """Un appel à update() doit modifier les poids de la tête OtO."""
        weights_before = trainer.oto_head.fc.weight.clone().detach()
        trainer.update(dummy_window, dummy_label_fault)
        weights_after = trainer.oto_head.fc.weight.clone().detach()
        assert not torch.allclose(weights_before, weights_after), (
            "Les poids OtO ne doivent pas être identiques après une mise à jour"
        )

    def test_update_does_not_modify_backbone(self, trainer, dummy_window, dummy_label_fault):
        """Un appel à update() ne doit pas modifier le backbone."""
        backbone_weights_before = [
            p.clone().detach() for p in trainer.autoencoder.parameters()
        ]
        trainer.update(dummy_window, dummy_label_fault)
        for p_before, p_after in zip(
            backbone_weights_before, trainer.autoencoder.parameters()
        ):
            assert torch.allclose(p_before, p_after), (
                "Le backbone ne doit pas être modifié lors d'un update OtO"
            )

    def test_predict_returns_tuple(self, trainer, dummy_window):
        """trainer.predict() retourne (prob_panne, mse_recon), tous deux scalaires."""
        y_hat, mse = trainer.predict(dummy_window)
        assert isinstance(y_hat, float)
        assert isinstance(mse, float)
        assert 0.0 <= y_hat <= 1.0

    def test_online_loop_10_samples(self, trainer):
        """La boucle online tourne sur 10 échantillons successifs sans erreur."""
        torch.manual_seed(0)
        for _ in range(10):
            x = torch.randn(25)
            y = torch.tensor(float(torch.randint(0, 2, (1,)).item()))
            loss = trainer.update(x, y)
            assert loss >= 0.0


# ---------------------------------------------------------------------------
# TestMemoryBudget
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    def test_oto_ram_under_1ko(self, oto_head):
        """Les paramètres OtO tiennent dans moins de 1 Ko @ FP32."""
        param_bytes = oto_head.n_params() * 4  # 4 octets par float32
        assert param_bytes < 1024, (
            f"OtO params @ FP32 = {param_bytes} B — doit être < 1 Ko"
        )

    def test_encoder_under_6ko(self, autoencoder):
        """Les paramètres encodeur tiennent dans moins de 6 Ko @ FP32."""
        param_bytes = autoencoder.n_encoder_params() * 4
        assert param_bytes < 6144, (
            f"Encodeur params @ FP32 = {param_bytes} B — doit être < 6 Ko"
        )
