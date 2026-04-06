# S3-03 — Implémenter `autoencoder.py` (backbone + décodeur)

| Champ | Valeur |
|-------|--------|
| **ID** | S3-03 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S3-02 (pour valider le forward pass avec données réelles) |
| **Fichiers cibles** | `src/models/tinyol/autoencoder.py`, `src/models/tinyol/__init__.py` |
| **Complété le** | — |

---

## Objectif

Implémenter l'autoencoder MLP qui constitue le **backbone TinyOL** :
1. **Encodeur** : extracteur de features (25 → 32 → 16 → 8), pré-entraîné offline, **gelé** pendant la phase CL
2. **Décodeur** : reconstructeur (8 → 16 → 32 → 25), utilisé uniquement au pré-entraînement (perte MSE)

Le modèle doit être :
- Autonome : aucune dépendance sauf `torch`
- Configurable via `configs/tinyol_config.yaml` uniquement (pas de valeur hardcodée)
- Annoté avec les empreintes mémoire `# MEM:` sur chaque couche
- Compatible avec l'interface attendue par `pretrain_tinyol.py` (S3-04) et `oto_head.py` (S3-05)

**Critère de succès** : `from src.models.tinyol import TinyOLAutoencoder` passe, n_params encodeur = 1 496, `freeze_encoder()` désactive correctement les gradients, et `pytest tests/test_tinyol_autoencoder.py -v` passe intégralement.

---

## Sous-tâches

### 1. Créer `src/models/tinyol/__init__.py`

```python
# src/models/tinyol/__init__.py
from .autoencoder import TinyOLAutoencoder

__all__ = ["TinyOLAutoencoder"]
```

### 2. Constantes de configuration

```python
# src/models/tinyol/autoencoder.py

# Conforme à configs/tinyol_config.yaml — ne pas modifier ici
INPUT_DIM: int = 25          # features statistiques par fenêtre (6 feat × 4 canaux + 1)
ENCODER_DIMS: tuple[int, ...] = (32, 16, 8)   # 25 → 32 → 16 → 8
DECODER_DIMS: tuple[int, ...] = (16, 32, 25)  # 8 → 16 → 32 → 25

# Budget mémoire encodeur (référence tinyol_spec.md §2.1)
# Linear(25→32) : 25×32 + 32 = 832 params  → MEM: 832×4 = 3 328 B @ FP32
# Linear(32→16) : 32×16 + 16 = 528 params  → MEM: 528×4 = 2 112 B @ FP32
# Linear(16→8)  : 16×8  +  8 = 136 params  → MEM: 136×4 =   544 B @ FP32
# TOTAL encodeur : 1 496 params → ~5 828 B ≈ 5,8 Ko @ FP32 / ~1,5 Ko @ INT8
```

### 3. Implémenter `TinyOLAutoencoder`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyOLAutoencoder(nn.Module):
    """
    Autoencoder MLP pour le pré-entraînement du backbone TinyOL.

    Séparation backbone/tête conforme à Ren2021TinyOL :
    - L'encodeur est pré-entraîné offline sur données normales, puis gelé.
    - Le décodeur est utilisé uniquement au pré-entraînement (perte MSE).
    - En phase CL, seul l'encodeur est utilisé (via encode()), le décodeur est ignoré.

    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée (défaut : 25).
    encoder_dims : tuple[int, ...]
        Dimensions des couches de l'encodeur (défaut : (32, 16, 8)).
    decoder_dims : tuple[int, ...]
        Dimensions des couches du décodeur (défaut : (16, 32, 25)).

    Notes
    -----
    Budget mémoire (conforme tinyol_spec.md §2.3) :
    - Encodeur (Flash après gel) : 1 496 params → ~5,8 Ko @ FP32 / ~1,5 Ko @ INT8
    - Décodeur (pré-entraînement seulement) : 1 656 params → ~6,5 Ko @ FP32
    - Activations forward : ~512 B @ FP32 / ~128 B @ INT8 (batch=1, MCU)
    - TOTAL RAM dynamique (inférence seule) : < 600 B (cible 64 Ko — très confortable)

    Références
    ----------
    Ren2021TinyOL, docs/models/tinyol_spec.md §2
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        encoder_dims: tuple[int, ...] = ENCODER_DIMS,
        decoder_dims: tuple[int, ...] = DECODER_DIMS,
    ) -> None:
        super().__init__()

        # --- Encodeur ---
        self.enc1 = nn.Linear(input_dim, encoder_dims[0])    # MEM: 800 B @ FP32 / 200 B @ INT8
        self.enc2 = nn.Linear(encoder_dims[0], encoder_dims[1])  # MEM: 512 B @ FP32 / 128 B @ INT8
        self.enc3 = nn.Linear(encoder_dims[1], encoder_dims[2])  # MEM: 128 B @ FP32 / 32 B @ INT8

        # --- Décodeur (utilisé au pré-entraînement uniquement) ---
        self.dec1 = nn.Linear(encoder_dims[2], decoder_dims[0])
        self.dec2 = nn.Linear(decoder_dims[0], decoder_dims[1])
        self.dec3 = nn.Linear(decoder_dims[1], decoder_dims[2])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait l'embedding latent z depuis les features d'entrée.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim], dtype float32

        Returns
        -------
        z : torch.Tensor, shape [batch, 8], dtype float32
            Embedding latent (8D).
            # MEM: batch × 8 × 4 B @ FP32 / batch × 8 B @ INT8
        """
        z = F.relu(self.enc1(x))   # MEM: batch × 32 × 4 B = 128 B @ FP32 (batch=1)
        z = F.relu(self.enc2(z))   # MEM: batch × 16 × 4 B = 64 B @ FP32 (batch=1)
        z = self.enc3(z)           # MEM: batch × 8 × 4 B = 32 B @ FP32 (batch=1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruit les features depuis l'embedding latent.

        Parameters
        ----------
        z : torch.Tensor, shape [batch, 8], dtype float32

        Returns
        -------
        x_hat : torch.Tensor, shape [batch, input_dim], dtype float32
            Reconstruction des features d'entrée.
        """
        x_hat = F.relu(self.dec1(z))
        x_hat = F.relu(self.dec2(x_hat))
        x_hat = self.dec3(x_hat)     # Pas d'activation finale (MSE sur features normalisées)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass complet (encodeur + décodeur).

        Utilisé pendant le pré-entraînement batch uniquement.
        En phase CL, utiliser encode() directement.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim], dtype float32

        Returns
        -------
        z : torch.Tensor, shape [batch, 8]
            Embedding latent.
        x_hat : torch.Tensor, shape [batch, input_dim]
            Reconstruction pour le calcul de la perte MSE.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la perte de reconstruction MSE.

        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim]
            Features d'entrée (normalisées).
        x_hat : torch.Tensor, shape [batch, input_dim]
            Reconstruction de l'encodeur-décodeur.

        Returns
        -------
        torch.Tensor (scalaire) — perte MSE différentiable.

        Notes
        -----
        Conforme à tinyol_spec.md §5 : PRE_TRAIN_LOSS = "mse".
        """
        return F.mse_loss(x_hat, x)

    def freeze_encoder(self) -> None:
        """
        Gèle les paramètres de l'encodeur (requires_grad = False).

        À appeler APRÈS le pré-entraînement, AVANT la phase CL.
        Après le gel, seule la tête OtO (oto_head.py) est entraînable.

        Analogie MCU : l'encodeur est stocké en Flash (lecture seule),
        la tête OtO est en RAM (lecture/écriture).

        Références
        ----------
        Ren2021TinyOL, tinyol_spec.md §1
        """
        for param in [
            *self.enc1.parameters(),
            *self.enc2.parameters(),
            *self.enc3.parameters(),
        ]:
            param.requires_grad = False

    def n_encoder_params(self) -> int:
        """Retourne le nombre de paramètres de l'encodeur (attendu : 1 496)."""
        return sum(p.numel() for p in [
            *self.enc1.parameters(),
            *self.enc2.parameters(),
            *self.enc3.parameters(),
        ])
```

### 4. Écrire les tests

Créer `tests/test_tinyol_autoencoder.py` :

```python
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
            assert not param.requires_grad, \
                f"Paramètre encodeur {name} a encore requires_grad=True après freeze"


def test_freeze_encoder_decoder_still_trainable():
    """Après freeze_encoder(), le décodeur doit rester entraînable."""
    model = TinyOLAutoencoder()
    model.freeze_encoder()
    for name, param in model.named_parameters():
        if name.startswith("dec"):
            assert param.requires_grad, \
                f"Paramètre décodeur {name} n'est plus entraînable après freeze encodeur"


def test_reconstruction_loss_positive_differentiable():
    """La perte MSE doit être positive et permettre la rétropropagation."""
    model = TinyOLAutoencoder()
    x = torch.randn(8, 25)
    z, x_hat = model(x)
    loss = model.reconstruction_loss(x, x_hat)
    assert loss.item() >= 0
    loss.backward()
    # Vérifier que les gradients sont bien calculés
    assert model.enc1.weight.grad is not None


def test_encode_batch_size_1():
    """Cas MCU : batch_size=1 (inférence échantillon par échantillon)."""
    model = TinyOLAutoencoder()
    model.freeze_encoder()
    x = torch.randn(1, 25)
    with torch.no_grad():
        z = model.encode(x)
    assert z.shape == (1, 8)
```

---

## Critères d'acceptation

- [ ] `from src.models.tinyol import TinyOLAutoencoder` — aucune erreur d'import
- [ ] `encode(x)` : shape `[batch, 8]` pour tout batch d'entrée
- [ ] `decode(z)` : shape `[batch, 25]` pour tout embedding
- [ ] `forward(x)` : retourne `(z, x_hat)` avec les bonnes dimensions
- [ ] `n_encoder_params() == 1496` (832 + 528 + 136, conforme `tinyol_spec.md §2.1`)
- [ ] `freeze_encoder()` : `requires_grad=False` sur tous les paramètres `enc1`, `enc2`, `enc3`
- [ ] `freeze_encoder()` : décodeur reste entraînable (`requires_grad=True`)
- [ ] `reconstruction_loss()` : perte scalaire positive, différentiable, rétropropageable
- [ ] `encode()` avec `batch_size=1` fonctionne (cas MCU)
- [ ] Annotations `# MEM:` présentes sur chaque couche Linear et activation intermédiaire
- [ ] `pytest tests/test_tinyol_autoencoder.py -v` — tous les tests passent
- [ ] `ruff check src/models/tinyol/autoencoder.py` + `black --check` passent

---

## Budget mémoire (récapitulatif)

| Composant | Params | FP32 | INT8 | Stockage MCU |
|-----------|--------|------|------|-------------|
| enc1 : Linear(25→32) | 832 | 3 328 B | 832 B | Flash |
| enc2 : Linear(32→16) | 528 | 2 112 B | 528 B | Flash |
| enc3 : Linear(16→8) | 136 | 544 B | 136 B | Flash |
| **Encodeur total** | **1 496** | **~5,8 Ko** | **~1,5 Ko** | **Flash** |
| Activations forward (batch=1) | — | ~512 B | ~128 B | RAM (temporaire) |
| **TOTAL RAM dynamique** | — | **< 600 B** | **< 160 B** | **RAM** |

> ✅ Très largement dans la cible de 64 Ko RAM. Marge disponible pour la tête OtO et le buffer UINT8.

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter | Statut |
|---------|-------------|--------|
| `n_encoder_params = 1496` confirmé | `docs/models/tinyol_spec.md §2.1` | ⬜ déjà documenté |
| Interface `encode(x) → z` | Utilisée dans S3-05 (`oto_head.py`) | ⬜ S3-05 |
| Interface `forward(x) → (z, x_hat)` | Utilisée dans S3-04 (`pretrain_tinyol.py`) | ⬜ S3-04 |
| `backbone.pt` (poids encodeur après pré-entraînement) | `experiments/exp_003_tinyol_dataset1/backbone.pt` | ⬜ S3-04 |

---

## Interface attendue par `pretrain_tinyol.py` (S3-04)

```python
from src.models.tinyol import TinyOLAutoencoder

# Initialisation depuis config
model = TinyOLAutoencoder(
    input_dim=config["backbone"]["input_dim"],         # 25
    encoder_dims=tuple(config["backbone"]["encoder_dims"]),   # (32, 16, 8)
    decoder_dims=tuple(config["backbone"]["decoder_dims"]),   # (16, 32, 25)
)

# Boucle de pré-entraînement batch
for x_batch, _ in pretrain_loader:
    optimizer.zero_grad()
    z, x_hat = model(x_batch)
    loss = model.reconstruction_loss(x_batch, x_hat)
    loss.backward()
    optimizer.step()

# Après pré-entraînement : geler l'encodeur
model.freeze_encoder()
torch.save(model.enc1.state_dict(), ...)  # Sauvegarder les poids → backbone.pt
```

---

## Interface attendue par `oto_head.py` (S3-05)

```python
from src.models.tinyol import TinyOLAutoencoder

# Phase CL online (MCU simulation)
model = TinyOLAutoencoder(...)
model.load_state_dict(...)       # Charger backbone.pt
model.freeze_encoder()

for x_sample, y_sample in cl_stream:
    with torch.no_grad():
        z = model.encode(x_sample)           # shape [1, 8]
        _, x_hat = model(x_sample)
        mse = model.reconstruction_loss(x_sample, x_hat)  # scalaire

    oto_input = torch.cat([z.squeeze(0), mse.unsqueeze(0)])  # shape [9]
    # → entrée de la tête OtO (S3-05)
```

---

## Questions ouvertes

- `TODO(dorra)` : backpropagation FP32 de l'autoencoder (1 496 + 1 656 = 3 152 params total) sur Cortex-M55 — overhead cycliste estimé pour 50 epochs × N_train samples ?
- `TODO(dorra)` : format recommandé pour la sauvegarde `backbone.pt` → quel format pour l'export vers TFLite Micro (ONNX intermédiaire ou direct PyTorch→C array) ?
- `FIXME(gap3)` : préparer un chemin de quantification INT8 de l'encodeur pour export TFLite Micro — activations ReLU sont INT8-friendly, à vérifier que `enc3` (pas de ReLU) ne pose pas de problème de quantification.
