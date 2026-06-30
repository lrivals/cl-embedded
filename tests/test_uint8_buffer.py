"""Tests unitaires — buffer UINT8 de TinyOLOnlineTrainer (S4-02)."""

import pytest
import torch
import torch.nn as nn

from src.models.tinyol.oto_head import OtOHead, TinyOLOnlineTrainer


# --- Fixtures locales ---

INPUT_DIM = 9    # embed_dim (4) + MSE = 5, mais on simplifie à 9 pour OtO
EMBED_DIM = 4    # sortie du mock encoder


class _MockAutoencoder(nn.Module):
    """Autoencoder minimal pour les tests — encode retourne un vecteur fixe de dim EMBED_DIM."""

    def __init__(self, input_dim: int = 9, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, input_dim)
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return z, x_hat


@pytest.fixture()
def config_uint8_on() -> dict:
    return {
        "oto_head": {
            "learning_rate": 0.01,
            "momentum": 0.0,
            "use_uint8_buffer": True,
            "buffer_size": 10,
            "buffer_replay_ratio": 1.0,  # replay à chaque step
        }
    }


@pytest.fixture()
def config_uint8_off() -> dict:
    return {
        "oto_head": {
            "learning_rate": 0.01,
            "momentum": 0.0,
            "use_uint8_buffer": False,
            "buffer_size": 10,
            "buffer_replay_ratio": 0.2,
        }
    }


@pytest.fixture()
def trainer_on(config_uint8_on: dict) -> TinyOLOnlineTrainer:
    oto_input_dim = EMBED_DIM + 1  # embed + MSE
    autoencoder = _MockAutoencoder(input_dim=9, embed_dim=EMBED_DIM)
    oto_head = OtOHead(input_dim=oto_input_dim)
    # Patch config pour que l'input_dim de OtO soit cohérent
    config_uint8_on["oto_head"]["input_dim"] = oto_input_dim
    return TinyOLOnlineTrainer(autoencoder, oto_head, config_uint8_on)


@pytest.fixture()
def trainer_off(config_uint8_off: dict) -> TinyOLOnlineTrainer:
    oto_input_dim = EMBED_DIM + 1
    autoencoder = _MockAutoencoder(input_dim=9, embed_dim=EMBED_DIM)
    oto_head = OtOHead(input_dim=oto_input_dim)
    config_uint8_off["oto_head"]["input_dim"] = oto_input_dim
    return TinyOLOnlineTrainer(autoencoder, oto_head, config_uint8_off)


def _do_updates(trainer: TinyOLOnlineTrainer, n: int = 5) -> None:
    """Effectue n updates avec des données synthétiques."""
    for _ in range(n):
        x = torch.randn(9)
        y = torch.tensor(float(torch.randint(0, 2, (1,)).item()))
        trainer.update(x, y)


# --- Tests ---

def test_buffer_dtype_is_uint8(trainer_on: TinyOLOnlineTrainer) -> None:
    """Après ≥2 updates, _buffer_uint8 doit être de dtype torch.uint8."""
    _do_updates(trainer_on, n=5)
    assert trainer_on._buffer_uint8 is not None
    assert trainer_on._buffer_uint8.dtype == torch.uint8


def test_buffer_ram_reduces_4x(trainer_on: TinyOLOnlineTrainer) -> None:
    """get_buffer_ram_bytes() doit indiquer compression_ratio == 4.0."""
    _do_updates(trainer_on, n=5)
    ram = trainer_on.get_buffer_ram_bytes()
    assert ram["compression_ratio"] == 4.0
    assert ram["uint8_bytes"] * 4 == ram["fp32_equivalent_bytes"]


def test_update_returns_loss(trainer_on: TinyOLOnlineTrainer) -> None:
    """update() doit retourner un float scalaire (loss BCE > 0 en général)."""
    x = torch.randn(9)
    y = torch.tensor(1.0)
    loss = trainer_on.update(x, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_buffer_fifo_capacity(trainer_on: TinyOLOnlineTrainer) -> None:
    """Le buffer ne doit jamais dépasser buffer_size=10 éléments."""
    _do_updates(trainer_on, n=20)
    assert len(trainer_on._buffer_fp32) <= 10
    assert len(trainer_on._buffer_labels_raw) <= 10


def test_no_buffer_when_disabled(trainer_off: TinyOLOnlineTrainer) -> None:
    """use_uint8_buffer=False → _buffer_uint8 reste None même après updates."""
    _do_updates(trainer_off, n=5)
    assert trainer_off._buffer_uint8 is None
