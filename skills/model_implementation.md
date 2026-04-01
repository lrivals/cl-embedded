# Skill : Implémentation de Modèle Embarqué

> **Usage** : Demander à Claude d'implémenter ou de réviser le code d'un modèle CL.  
> **Déclencheur** : "implémente [modèle]" / "écris le code de [fichier].py" / "révise l'architecture de [modèle]"

---

## Checklist pré-implémentation (Claude doit vérifier)

Avant d'écrire du code, Claude doit confirmer :

1. ✅ La spec du modèle dans `docs/models/[modele]_spec.md` a été lue
2. ✅ Le budget mémoire cible (64 Ko RAM, STM32N6) est connu
3. ✅ L'optimizer autorisé est SGD (pas Adam sauf justification explicite)
4. ✅ Les activations utilisées sont ReLU (pas GELU, SiLU, etc.)
5. ✅ Pas de BatchNorm / LayerNorm dans le forward
6. ✅ Les annotations `# MEM:` seront présentes sur chaque couche

---

## Template de classe de modèle

Tout modèle de ce projet suit ce template :

```python
"""
[NomModèle] — [Description courte]

Référence : [Auteur(s), Année]
Gap(s) adressé(s) : Gap 1 / Gap 2 / Gap 3
RAM estimée FP32 : X Ko
RAM estimée INT8 : Y Ko
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class [NomModèle](nn.Module):
    """
    [Description du modèle].
    
    Parameters
    ----------
    input_dim : int
        Dimension du vecteur d'entrée.
    [autres params]
    
    Notes
    -----
    Budget mémoire (FP32) :
        - Poids : X Ko
        - Activations (forward) : Y Ko
        - Overhead CL : Z Ko
        - Total : W Ko (cible ≤ 64 Ko)
    
    Références
    ----------
    [Auteur, Année]. [Titre]. [Venue].
    """
    
    def __init__(self, input_dim: int, ...):
        super().__init__()
        # MEM: Linear(D_in → D_out) = D_in*D_out + D_out params = X B @ FP32
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : Tensor [batch_size, input_dim]
        
        Returns
        -------
        Tensor [batch_size, output_dim]
        """
        x = F.relu(self.fc1(x))  # MEM: hidden_dim × 4 B @ FP32
        ...
        return x
    
    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """Estime l'empreinte RAM des poids seuls (hors activations)."""
        bytes_per_param = 4 if dtype == "fp32" else 1
        return self.count_parameters() * bytes_per_param
```

---

## Règles de code spécifiques au projet

### Imports
```python
# Imports standard — ordre obligatoire
from __future__ import annotations  # toujours en premier
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict
import yaml
```

### Configuration via YAML (obligatoire)
```python
# ✅ Correct : hyperparamètres depuis config
import yaml
with open("configs/ewc_config.yaml") as f:
    cfg = yaml.safe_load(f)
lr = cfg["training"]["learning_rate"]

# ❌ Interdit : hyperparamètres hardcodés dans le code
lr = 0.01
```

### Gestion des gradients (backbone gelé)
```python
# ✅ Correct : simuler le comportement Flash MCU
model.backbone.eval()
for param in model.backbone.parameters():
    param.requires_grad = False

# Entraînement de la tête uniquement
optimizer = torch.optim.SGD(model.head.parameters(), lr=cfg["training"]["lr"])
```

### Sauvegarde des checkpoints
```python
# Format standardisé pour chaque checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": cfg,
    "metrics": metrics_dict,
    "epoch": epoch,
    "ram_bytes_measured": ram_bytes,
}
torch.save(checkpoint, f"experiments/{exp_id}/checkpoint.pt")
```

---

## Anti-patterns à éviter absolument

```python
# ❌ Taille variable dans le forward (incompatible MCU)
def forward(self, x):
    batch_size = x.shape[0]  # OK si utilisé comme info seulement
    hidden = torch.zeros(batch_size, self.hidden_dim)  # ❌ allocation dynamique

# ✅ Tailles fixes
def forward(self, x):
    return F.relu(self.fc1(x))
```

```python
# ❌ Adam (trop coûteux MCU)
optimizer = torch.optim.Adam(model.parameters())

# ✅ SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

```python
# ❌ Normalisation en ligne dans le modèle
class Model(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=0)  # ❌ calcul stats en ligne
        return (x - mean) / x.std(dim=0)

# ✅ Stats fixes depuis la config
class Model(nn.Module):
    def __init__(self, mean, std):
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
    def forward(self, x):
        return (x - self.mean) / self.std
```

---

## Tests minimaux requis pour chaque modèle

```python
# tests/test_[modele].py — template

def test_forward_shape():
    """Le forward produit la bonne forme de sortie."""

def test_count_parameters():
    """Le nombre de paramètres est dans la plage attendue."""

def test_estimate_ram():
    """La RAM estimée est < 64 Ko."""

def test_no_grad_backbone():
    """Le backbone ne reçoit pas de gradient après gel."""

def test_cl_update():
    """Une mise à jour incrémentale ne plante pas et modifie les poids."""

def test_onnx_export():
    """Le modèle est exportable en ONNX sans erreur."""
```
