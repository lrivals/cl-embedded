# S1-04 — Implémenter `ewc_mlp.py` (MLP + perte EWC)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-04 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | S1-03 (loader disponible pour valider le forward pass) |
| **Fichiers cibles** | `src/models/ewc/ewc_mlp.py`, `src/models/ewc/__init__.py` |

---

## Objectif

Implémenter le modèle MLP (769 paramètres, ~9 Ko total EWC @ FP32) et sa perte EWC Online tels que spécifiés dans `docs/models/ewc_mlp_spec.md`.

Le modèle doit être :
- Autonome : aucune dépendance externe sauf `torch`
- Configurable exclusivement via `configs/ewc_config.yaml` (pas de valeur hardcodée)
- Annoté avec les empreintes mémoire `# MEM:` sur chaque couche
- Compatible avec l'interface attendue par `cl_trainer.py` (S1-06)

**Critère de succès** : `python -c "from src.models.ewc import EWCMlpClassifier"` passe, et `tests/test_ewc_mlp.py` passe intégralement.

---

## Sous-tâches

### 1. Créer `src/models/ewc/__init__.py`

```python
# src/models/ewc/__init__.py
from .ewc_mlp import EWCMlpClassifier

__all__ = ["EWCMlpClassifier"]
```

### 2. Implémenter le MLP

Architecture conforme à `docs/models/ewc_mlp_spec.md` §2 :

```
Input: [batch, 6]
   ├── Linear(6 → 32) + ReLU     # MEM: 224 B @ FP32 / 56 B @ INT8
   ├── Dropout(p=0.2)             # désactivé à l'inférence MCU
   ├── Linear(32 → 16) + ReLU    # MEM: 520 B @ FP32 / 130 B @ INT8
   └── Linear(16 → 1)  + Sigmoid  # MEM: 68 B @ FP32 / 17 B @ INT8
```

```python
import torch
import torch.nn as nn
from typing import Optional

# Constantes dérivées de ewc_config.yaml — ne pas modifier ici
INPUT_DIM: int = 6
HIDDEN_DIMS: tuple[int, int] = (32, 16)
OUTPUT_DIM: int = 1
DROPOUT_RATE: float = 0.2

class EWCMlpClassifier(nn.Module):
    """
    MLP binaire pour la classification de défauts industriels.

    Conçu pour l'entraînement EWC Online sur le Dataset 2 (Equipment Monitoring).
    Architecture fixe : Linear(6→32) → ReLU → Dropout → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid.

    Paramètres
    ----------
    input_dim : int
        Dimension des features d'entrée (défaut : 6, configurable via ewc_config.yaml)
    hidden_dims : list[int]
        Tailles des couches cachées (défaut : [32, 16])
    output_dim : int
        Dimension de sortie (défaut : 1, classification binaire)
    dropout : float
        Taux de dropout (désactivé en mode eval — inférence MCU)

    Notes
    -----
    Empreinte mémoire totale EWC :
    - Modèle seul   : 769 params × 4 B = ~3 Ko @ FP32
    - Fisher diag   : 769 scalaires × 4 B = ~3 Ko @ FP32
    - Snapshot θ*   : 769 scalaires × 4 B = ~3 Ko @ FP32
    - TOTAL         : ~9 Ko @ FP32 (cible : ≤ 64 Ko)

    Références
    ----------
    Kirkpatrick2017EWC, docs/models/ewc_mlp_spec.md §2
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = list(HIDDEN_DIMS),
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])   # MEM: 224 B @ FP32 / 56 B @ INT8
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # MEM: 520 B @ FP32 / 130 B @ INT8
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # MEM: 68 B @ FP32 / 17 B @ INT8
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [batch, input_dim], dtype float32

        Returns
        -------
        torch.Tensor, shape [batch, 1], dtype float32
            Probabilité de défaut ŷ ∈ [0, 1]
        """
        hidden = self.relu(self.fc1(x))    # MEM: 32 × 4 B = 128 B @ FP32 / 32 B @ INT8
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc2(hidden))  # MEM: 16 × 4 B = 64 B @ FP32 / 16 B @ INT8
        out = self.sigmoid(self.fc3(hidden))  # MEM: 1 × 4 B = 4 B @ FP32
        return out
```

### 3. Implémenter la perte EWC Online

```python
class EWCMlpClassifier(nn.Module):
    # ... (suite de l'implémentation)

    def ewc_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fisher: Optional[dict[str, torch.Tensor]],
        theta_star: Optional[dict[str, torch.Tensor]],
        ewc_lambda: float,
    ) -> torch.Tensor:
        """
        Calcule la perte EWC complète : L_BCE + régularisation Fisher.

        L_EWC(θ) = L_BCE(θ) + λ/2 · Σᵢ Fᵢ (θᵢ - θ*ᵢ)²

        Parameters
        ----------
        x : torch.Tensor
            Batch d'entrée, shape [batch, input_dim]
        y : torch.Tensor
            Labels, shape [batch, 1], dtype float32
        fisher : dict[str, Tensor] ou None
            Diagonale de Fisher par nom de paramètre (None pour Task 1)
        theta_star : dict[str, Tensor] ou None
            Snapshot des poids après la tâche précédente (None pour Task 1)
        ewc_lambda : float
            Coefficient de régularisation λ (depuis ewc_config.yaml → ewc.lambda)

        Returns
        -------
        torch.Tensor (scalaire)
            Perte totale L_EWC

        Notes
        -----
        Si fisher ou theta_star est None (Task 1), la perte est pure BCE.
        Référence : docs/models/ewc_mlp_spec.md §1, Kirkpatrick2017EWC eq. 3
        """
        pred = self.forward(x)
        bce_loss = nn.functional.binary_cross_entropy(pred, y)

        if fisher is None or theta_star is None:
            return bce_loss

        ewc_reg = torch.tensor(0.0, device=x.device)
        for name, param in self.named_parameters():
            if name in fisher and name in theta_star:
                ewc_reg += (fisher[name] * (param - theta_star[name]) ** 2).sum()

        return bce_loss + (ewc_lambda / 2.0) * ewc_reg

    def get_theta_star(self) -> dict[str, torch.Tensor]:
        """
        Retourne un snapshot des poids courants (θ* après la tâche courante).
        Doit être appelé APRÈS l'entraînement sur une tâche, AVANT la suivante.

        Returns
        -------
        dict[str, Tensor] : copie détachée des paramètres courants
        """
        return {name: param.detach().clone() for name, param in self.named_parameters()}
```

### 4. Écrire le test

Créer `tests/test_ewc_mlp.py` :

```python
import torch
from src.models.ewc import EWCMlpClassifier

def test_forward_shape():
    """Vérifie que le forward pass retourne la bonne forme."""
    model = EWCMlpClassifier()
    x = torch.randn(32, 6)
    out = model(x)
    assert out.shape == (32, 1)
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid doit borner [0,1]"

def test_bce_loss_task1():
    """Sur Task 1, la perte doit être pure BCE (fisher=None)."""
    model = EWCMlpClassifier()
    x = torch.randn(16, 6)
    y = torch.randint(0, 2, (16, 1)).float()
    loss = model.ewc_loss(x, y, fisher=None, theta_star=None, ewc_lambda=1000.0)
    assert loss.item() > 0

def test_ewc_loss_increases_with_lambda():
    """La régularisation EWC doit augmenter avec λ."""
    model = EWCMlpClassifier()
    x = torch.randn(16, 6)
    y = torch.randint(0, 2, (16, 1)).float()
    # Fisher non-nulle : simule post-Task 1
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    theta_star = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    loss_low = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=10.0)
    loss_high = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda=10000.0)
    assert loss_high.item() > loss_low.item()

def test_n_params():
    """Vérifie le nombre total de paramètres (769 attendu)."""
    model = EWCMlpClassifier()
    n = sum(p.numel() for p in model.parameters())
    assert n == 769, f"Attendu 769 params, obtenu {n}"

def test_theta_star_detached():
    """Le snapshot θ* ne doit pas partager le graphe de calcul."""
    model = EWCMlpClassifier()
    theta_star = model.get_theta_star()
    for name, tensor in theta_star.items():
        assert not tensor.requires_grad, f"{name} ne devrait pas avoir requires_grad"
```

---

## Critères d'acceptation

- [ ] `from src.models.ewc import EWCMlpClassifier` — aucune erreur d'import
- [ ] `forward()` : shape `[batch, 1]`, valeurs ∈ [0, 1] pour tout batch d'entrée
- [ ] `ewc_loss()` : retourne une perte scalaire positive, différentiable, rétropropageable
- [ ] `ewc_loss()` avec `fisher=None` == perte BCE pure (Task 1 sans régularisation)
- [ ] Régularisation EWC croît avec λ (test `test_ewc_loss_increases_with_lambda`)
- [ ] `n_params == 769` (conforme à `docs/models/ewc_mlp_spec.md` §2)
- [ ] `get_theta_star()` retourne des tenseurs détachés (`requires_grad=False`)
- [ ] Annotations `# MEM:` présentes sur chaque couche Linear et activation intermédiaire
- [ ] `ruff check src/models/ewc/ewc_mlp.py` et `black --check` passent
- [ ] `pytest tests/test_ewc_mlp.py -v` — tous les tests passent

---

## Interface attendue par `cl_trainer.py` (S1-06)

Le trainer appellera le modèle comme suit :

```python
# Initialisation
model = EWCMlpClassifier(
    input_dim=config["model"]["input_dim"],       # 6
    hidden_dims=config["model"]["hidden_dims"],   # [32, 16]
    dropout=config["model"]["dropout"],           # 0.2
)

# Boucle d'entraînement Task i
for x, y in train_loader:
    optimizer.zero_grad()
    loss = model.ewc_loss(x, y, fisher=fisher, theta_star=theta_star, ewc_lambda=lam)
    loss.backward()
    optimizer.step()

# Fin de Task i — snapshot et Fisher (calculée dans fisher.py, S1-05)
theta_star = model.get_theta_star()
fisher = compute_fisher_diagonal(model, train_loader, device)  # S1-05
```

---

## Questions ouvertes

- `TODO(arnaud)` : Dropout conservé à l'inférence MCU ou désactivé ? (actuellement désactivé en mode `eval()`)
- `TODO(dorra)` : faisabilité backprop FP32 sur Cortex-M55 pour ces 769 paramètres — overhead cycliste estimé ?
- `FIXME(gap3)` : préparer un chemin de quantification INT8 de ce MLP pour l'export ONNX → TFLite Micro
