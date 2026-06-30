# S2506–S2508 — Nouveaux modèles : EWC Régression, EWC Multi-classe, HDC Régressor

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🔴 Critique (S2506, S2507) / 🟡 Important (S2508) |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | S2506 : 3h / S2507 : 3h / S2508 : 3h = 9h total |
| **Dépendances** | `src/models/ewc/ewc_mlp.py` ✅ (pattern EWC à hériter), `src/models/hdc/hdc_classifier.py` ✅, `src/models/base_cl_model.py` ✅ |
| **Fichiers cibles** | `src/models/ewc/ewc_mlp_regression.py`, `src/models/ewc/ewc_mlp_multiclass.py`, `src/models/hdc/hdc_regressor.py` |
| **Référence** | `src/models/ewc/ewc_mlp.py` (pattern EWC Online), `src/models/hdc/hdc_classifier.py` (pattern HDC accumulateur), `src/models/ewc/ewc_mlp_int8.py` (variante quantifiée) |

---

## Contexte

Les modèles existants (`EWCMlpClassifier`, `HDCClassifier`) sont conçus pour la **classification binaire**. Sprint 25 exige des variantes pour :

- **Régression** : prédire un RUL continu (CMAPSS FD001–FD004, Pronostia) — 1 neurone de sortie, perte MSE, Fisher calculé via gradient MSE.
- **Multi-classe** : prédire le type de défaut (CWRU 10 classes, Paderborn 3 états) — N neurones softmax, perte cross-entropy, Fisher sur tous les poids.
- **HDC Régression** : pas de prototypes par classe, un seul vecteur accumulateur pondéré dont le produit scalaire avec l'embedding prédit le RUL.

**Règle de code** : les annotations `# MEM:` sont obligatoires sur toute couche ou activation temporaire. Les hyperparamètres de taille passent par les configs YAML, jamais codés en dur.

---

## S2506 — `src/models/ewc/ewc_mlp_regression.py`

### Architecture

MLP avec 1 neurone de sortie (régression linéaire) + régularisation EWC Online. Identique à `EWCMlpClassifier` sauf :
- `fc3 = nn.Linear(hidden_dims[-1], 1)` sans Sigmoid finale
- Perte `nn.MSELoss()` au lieu de `nn.BCELoss()`
- Fisher calculé via gradient MSE (même formule diagonale)

### Spec complète

```python
"""
ewc_mlp_regression.py — EWC Online + MLP pour la régression RUL.

Tâche : prédire le Remaining Useful Life continu (float) sur CMAPSS / Pronostia.
Méthode CL : Elastic Weight Consolidation Online (Schwarz et al., 2018).

RAM estimée (input_dim=5, hidden_dims=[32, 16]) :
    Poids : (5×32+32 + 32×16+16 + 16×1+1) × 4 = ~3 Ko @ FP32
    Fisher : ~3 Ko @ FP32
    θ*     : ~3 Ko @ FP32
    TOTAL  : ~9 Ko @ FP32  ✅ << 256 Ko NUCLEO-F439ZI

Références :
    Kirkpatrick2017EWC — EWC (régularisation)
    Schwarz et al., 2018 — EWC Online
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.models.base_cl_model import BaseCLModel


class EWCMlpRegressor(nn.Module):
    """
    MLP régression avec régularisation EWC Online.

    Architecture :
        Linear(input_dim → 32) + ReLU
        Dropout(p=dropout)
        Linear(32 → 16)        + ReLU
        Dropout(p=dropout)
        Linear(16 → 1)         [sortie linéaire — pas de Sigmoid]

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée.
    hidden_dims : list[int]
        Dimensions des couches cachées. Default : [32, 16].
    dropout : float
        Taux de dropout. Default : 0.2.
    ewc_lambda : float
        Coefficient de pénalité EWC. Default : 400.0.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        ewc_lambda: float = 400.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.ewc_lambda = ewc_lambda

        # MEM: Linear(5→32)  = (5×32 + 32) × 4 = 704 B @ FP32 / 176 B @ INT8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout)
        # MEM: Linear(32→16) = (32×16 + 16) × 4 = 2 112 B @ FP32 / 528 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout)
        # MEM: Linear(16→1)  = (16×1 + 1) × 4 = 68 B @ FP32 / 17 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        # Paramètres EWC Online
        self._fisher: dict[str, Tensor] = {}  # Fisher diagonale par paramètre
        self._theta_star: dict[str, Tensor] = {}  # θ* snapshot post-consolidation

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]

        Returns
        -------
        Tensor [batch_size, 1]
            Prédiction RUL ŷ ∈ ℝ (non bornée en inférence).
        """
        # MEM activations: 32 × 4 = 128 B @ FP32
        h = torch.relu(self.fc1(x))
        h = self.drop1(h)
        # MEM activations: 16 × 4 = 64 B @ FP32
        h = torch.relu(self.fc2(h))
        h = self.drop2(h)
        return self.fc3(h)  # shape [batch_size, 1]

    def ewc_penalty(self) -> Tensor:
        """Pénalité EWC = λ/2 · Σ F_i · (θ_i - θ*_i)²."""
        if not self._fisher:
            return torch.tensor(0.0, requires_grad=True)
        penalty = sum(
            (self._fisher[n] * (p - self._theta_star[n]) ** 2).sum()
            for n, p in self.named_parameters()
            if n in self._fisher
        )
        return self.ewc_lambda / 2.0 * penalty

    def consolidate(self, data_loader: DataLoader, n_samples: int = 200) -> None:
        """
        Calcule la Fisher diagonale sur `n_samples` exemples et snapshote θ*.

        Doit être appelé APRÈS l'entraînement d'une tâche, AVANT la suivante.
        Fisher via gradient MSE (pas de log-vraisemblance binaire).
        """
        self.eval()
        criterion = nn.MSELoss()
        fisher_accum: dict[str, Tensor] = {
            n: torch.zeros_like(p) for n, p in self.named_parameters()
        }
        count = 0
        for x_batch, y_batch in data_loader:
            if count >= n_samples:
                break
            self.zero_grad()
            y_pred = self(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch.float())
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher_accum[n] += p.grad.data.clone() ** 2
            count += len(x_batch)

        n_batches = max(1, count // data_loader.batch_size)
        self._fisher = {n: f / n_batches for n, f in fisher_accum.items()}
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters()}
        self.train()
```

### Vérification

```bash
python -c "
import torch
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor

model = EWCMlpRegressor(input_dim=5, hidden_dims=[32, 16])
x = torch.randn(4, 5)
y = torch.rand(4) * 125  # RUL en [0, 125]

# Forward
out = model(x)
assert out.shape == (4, 1), f'Shape incorrecte : {out.shape}'

# Loss MSE
loss = torch.nn.MSELoss()(out.squeeze(), y)
loss.backward()

# Penalty sans consolidation = 0
pen = model.ewc_penalty()
assert pen.item() == 0.0, 'Penalty non nulle avant consolidation'

print('EWCMlpRegressor forward + loss + penalty OK ✅')
print(f'  Output shape : {out.shape}')
print(f'  MSE loss : {loss.item():.4f}')
"
```

---

## S2507 — `src/models/ewc/ewc_mlp_multiclass.py`

### Architecture

MLP avec N neurones de sortie (softmax) + régularisation EWC Online. Identique à `EWCMlpClassifier` sauf :
- `fc3 = nn.Linear(hidden_dims[-1], n_classes)`
- Sortie brute (logits) — softmax appliqué par la perte `nn.CrossEntropyLoss()`
- Fisher calculé via gradient cross-entropy

### Spec complète

```python
"""
ewc_mlp_multiclass.py — EWC Online + MLP pour la classification multi-classe.

Tâche : classer le type et la sévérité de défaut (CWRU 10 classes, Paderborn 3 états).
Méthode CL : EWC Online.

RAM estimée (input_dim=9, n_classes=10, hidden_dims=[32, 16]) :
    Poids : (9×32+32 + 32×16+16 + 16×10+10) × 4 = ~4.7 Ko @ FP32
    Fisher : ~4.7 Ko @ FP32
    θ*     : ~4.7 Ko @ FP32
    TOTAL  : ~14 Ko @ FP32  ✅ << 256 Ko NUCLEO-F439ZI

Références :
    Kirkpatrick2017EWC — EWC (régularisation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class EWCMlpMulticlass(nn.Module):
    """
    MLP multi-classe avec régularisation EWC Online.

    Architecture :
        Linear(input_dim → 32) + ReLU
        Dropout(p=dropout)
        Linear(32 → 16)        + ReLU
        Dropout(p=dropout)
        Linear(16 → n_classes) [logits bruts — CrossEntropyLoss applique softmax]

    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée.
    n_classes : int
        Nombre de classes. CWRU : 10, Paderborn : 3.
    hidden_dims : list[int]
        Dimensions des couches cachées. Default : [32, 16].
    dropout : float
        Taux de dropout. Default : 0.2.
    ewc_lambda : float
        Coefficient de pénalité EWC. Default : 400.0.
    """

    def __init__(
        self,
        input_dim: int = 9,
        n_classes: int = 10,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        ewc_lambda: float = 400.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.ewc_lambda = ewc_lambda

        # MEM: Linear(9→32)  = (9×32 + 32) × 4 = 1 280 B @ FP32
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout)
        # MEM: Linear(32→16) = (32×16 + 16) × 4 = 2 112 B @ FP32
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout)
        # MEM: Linear(16→10) = (16×10 + 10) × 4 = 680 B @ FP32
        self.fc3 = nn.Linear(hidden_dims[1], n_classes)

        self._fisher: dict[str, Tensor] = {}
        self._theta_star: dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]

        Returns
        -------
        Tensor [batch_size, n_classes]
            Logits bruts (avant softmax).
        """
        # MEM activations: 32 × 4 = 128 B @ FP32
        h = torch.relu(self.fc1(x))
        h = self.drop1(h)
        # MEM activations: 16 × 4 = 64 B @ FP32
        h = torch.relu(self.fc2(h))
        h = self.drop2(h)
        return self.fc3(h)  # shape [batch_size, n_classes]

    def predict(self, x: Tensor) -> Tensor:
        """Retourne la classe prédite (argmax des logits)."""
        with torch.no_grad():
            return self(x).argmax(dim=1)

    def ewc_penalty(self) -> Tensor:
        """Pénalité EWC = λ/2 · Σ F_i · (θ_i - θ*_i)²."""
        if not self._fisher:
            return torch.tensor(0.0, requires_grad=True)
        penalty = sum(
            (self._fisher[n] * (p - self._theta_star[n]) ** 2).sum()
            for n, p in self.named_parameters()
            if n in self._fisher
        )
        return self.ewc_lambda / 2.0 * penalty

    def consolidate(self, data_loader: DataLoader, n_samples: int = 200) -> None:
        """
        Calcule la Fisher diagonale et snapshote θ*.

        Fisher via gradient cross-entropy (classification multi-classe).
        """
        self.eval()
        criterion = nn.CrossEntropyLoss()
        fisher_accum: dict[str, Tensor] = {
            n: torch.zeros_like(p) for n, p in self.named_parameters()
        }
        count = 0
        for x_batch, y_batch in data_loader:
            if count >= n_samples:
                break
            self.zero_grad()
            logits = self(x_batch)
            loss = criterion(logits, y_batch.long())
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher_accum[n] += p.grad.data.clone() ** 2
            count += len(x_batch)

        n_batches = max(1, count // data_loader.batch_size)
        self._fisher = {n: f / n_batches for n, f in fisher_accum.items()}
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters()}
        self.train()
```

### Vérification

```bash
python -c "
import torch
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

model = EWCMlpMulticlass(input_dim=9, n_classes=10)
x = torch.randn(8, 9)
y = torch.randint(0, 10, (8,))

# Forward — logits
logits = model(x)
assert logits.shape == (8, 10), f'Shape logits : {logits.shape}'

# Softmax normalisé
probs = torch.softmax(logits, dim=1)
assert abs(probs.sum(dim=1).mean().item() - 1.0) < 1e-5, 'Softmax non normalisé'

# CrossEntropy loss
loss = torch.nn.CrossEntropyLoss()(logits, y)
loss.backward()

# Predict
preds = model.predict(x)
assert preds.shape == (8,)

print('EWCMlpMulticlass OK ✅')
print(f'  Logits shape : {logits.shape}')
print(f'  Preds : {preds.tolist()}')
"
```

---

## S2508 — `src/models/hdc/hdc_regressor.py`

### Principe

HDC Régression = encodage hyperdimensionnel + prédiction linéaire. Pas de prototypes par classe : un seul vecteur de poids `w ∈ ℝ^D` appris par descente de gradient sur MSE. Prédiction : `ŷ = w · encode(x)`.

```
encode(x) : ℝ^n_features → {±1}^D    (encodage HDC existant)
w : ℝ^D                               (vecteur de poids — gradient MSE)
ŷ = dot(w, encode(x))                  (prédiction scalaire RUL)
```

### Spec complète

```python
"""
hdc_regressor.py — Régression linéaire sur embeddings HDC.

Tâche : prédire un RUL continu à partir d'embeddings hyperdimensionnels.
Apprentissage : descente de gradient (SGD) sur MSE — pas d'accumulation de prototypes.
Oubli catastrophique : atténué par la nature distribuite des embeddings HDC.

RAM estimée (D=1024) :
    Vecteurs de base : D × N_LEVELS × 1 = 10 240 B @ INT8
    Vecteur poids w  : D × 4 = 4 096 B @ FP32
    TOTAL            : ~14 Ko  ✅ << 256 Ko NUCLEO-F439ZI

Référence : Benatti2019HDC (encodage HDC), hdc_classifier.py (encode_sample)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.hdc.hdc_classifier import HDCClassifier  # réutilise encode_sample


class HDCRegressor:
    """
    Régression linéaire sur embeddings hyperdimensionnels.

    Hérite de l'encodage HDC existant (base vectors + niveau quantification)
    et apprend un vecteur de poids w par SGD sur MSE.

    Parameters
    ----------
    D : int
        Dimension des hypervecteurs. Default : 1024.
    n_levels : int
        Niveaux de quantification par feature. Default : 10.
    n_features : int
        Dimension de l'espace d'entrée. Default : 5 (CMAPSS top-5).
    lr : float
        Taux d'apprentissage SGD. Default : 0.01.
    """

    def __init__(
        self,
        D: int = 1024,
        n_levels: int = 10,
        n_features: int = 5,
        lr: float = 0.01,
    ) -> None:
        self.D = D
        self.n_levels = n_levels
        self.n_features = n_features

        # Vecteur de poids linéaire — MEM: D × 4 = 4 096 B @ FP32
        self.w = nn.Parameter(torch.zeros(D))
        self.optimizer = torch.optim.SGD([self.w], lr=lr)

        # Matrice de vecteurs de base (int8) — MEM: D × N_LEVELS = 10 240 B @ INT8
        self._base_vectors: np.ndarray | None = None  # chargés lors du premier fit
        self._feature_min: np.ndarray | None = None
        self._feature_max: np.ndarray | None = None

    def _encode(self, x: np.ndarray) -> Tensor:
        """
        Encode un batch x ∈ ℝ^(N × n_features) en hypervecteurs {±1}^(N × D).

        Réutilise la logique HDC de hdc_classifier.encode_sample() vectorisée.
        """
        # Implémentation : quantification features → lookup base vectors → XOR pondéré
        ...  # à implémenter en réutilisant encode_sample de HDCClassifier

    def fit_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Met à jour w sur un mini-batch (online learning).

        Returns
        -------
        float : MSE loss sur ce batch.
        """
        hvecs = self._encode(x)                       # [N, D]
        y_pred = (hvecs * self.w).sum(dim=1)          # [N]
        loss = nn.MSELoss()(y_pred, torch.tensor(y, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Prédiction RUL pour un batch de features."""
        with torch.no_grad():
            hvecs = self._encode(x)
            return (hvecs * self.w).sum(dim=1).numpy()
```

### Vérification

```bash
python -c "
import numpy as np
from src.models.hdc.hdc_regressor import HDCRegressor

model = HDCRegressor(D=256, n_features=5)  # D réduit pour le test
x = np.random.randn(8, 5).astype(np.float32)
y = np.random.rand(8).astype(np.float32) * 125

model.set_feature_bounds(x)  # requis avant fit/predict (bornes de quantification HDC)
loss = model.fit_batch(x, y)
preds = model.predict(x)
assert preds.shape == (8,), f'Shape prédictions : {preds.shape}'
print(f'HDCRegressor fit_batch loss={loss:.4f}, preds shape={preds.shape} ✅')
"
```

---

## Vérification end-to-end

```bash
# EWC Régression
python -c "
import torch
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor
m = EWCMlpRegressor(input_dim=5)
x, y = torch.randn(32, 5), torch.rand(32) * 125
out = m(x)
assert out.shape == (32, 1)
print('EWCMlpRegressor ✅')
"

# EWC Multi-classe
python -c "
import torch
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass
m = EWCMlpMulticlass(input_dim=9, n_classes=10)
x, y = torch.randn(32, 9), torch.randint(0, 10, (32,))
logits = m(x)
assert logits.shape == (32, 10)
print('EWCMlpMulticlass ✅')
"

# Empreinte mémoire (doit rester < 256 Ko)
python -c "
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass
for cls, kwargs in [(EWCMlpRegressor, {'input_dim': 5}),
                   (EWCMlpMulticlass, {'input_dim': 9, 'n_classes': 10})]:
    m = cls(**kwargs)
    n_params = sum(p.numel() for p in m.parameters())
    mem_fp32 = n_params * 3 * 4  # params + fisher + theta_star
    print(f'{cls.__name__}: {n_params} params, ~{mem_fp32 // 1024} Ko FP32 total')
    assert mem_fp32 < 256 * 1024
"
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2506 — `ewc_mlp_regression.py` | ✅ | 737 params, ~8 Ko FP32 (poids+Fisher+θ*) |
| S2507 — `ewc_mlp_multiclass.py` | ✅ | 1 018 params, ~11 Ko FP32 |
| S2508 — `hdc_regressor.py` | ✅ | D=1024 → ~14 Ko, `_encode` via `encode_observation` ; `set_feature_bounds(x)` ajouté hors spec — requis avant `fit_batch`/`predict` (bornes de quantification HDC) |

---

## Questions ouvertes

- `TODO(arnaud)` : Pour S2508 (HDC régression), l'accumulation additive de prototypes n'est plus applicable — confirmer que le SGD en ligne sur embeddings est acceptable comme variante "HDC incrémental" dans la taxonomie du manuscrit.
- `TODO(dorra)` : Le vecteur de poids `w` du HDC régresseur (FP32) est-il portable en C sur NUCLEO ? Ou faut-il prévoir une variante INT8 dès Sprint 25 ?
- `FIXME(gap3)` : EWC Régression utilise MSE — la Fisher diagonale via gradient MSE diffère de la Fisher classique (log-vraisemblance gaussienne). Vérifier que cela reste valide théoriquement pour le manuscrit.
