# S3-05 — Implémenter `oto_head.py` (tête OtO + boucle SGD online)

| Champ | Valeur |
|-------|--------|
| **ID** | S3-05 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | S3-03 (`autoencoder.py` opérationnel) + S3-04 (`backbone.pt` disponible) |
| **Fichier cible** | `src/models/tinyol/oto_head.py` |
| **Complété le** | 9 avril 2026 |

---

## Objectif

Implémenter `src/models/tinyol/oto_head.py` qui contient :
1. La classe `OtOHead` — couche linéaire `Linear(9→1) + Sigmoid`, 10 paramètres, 40 octets @ FP32
2. La classe `TinyOLOnlineTrainer` — boucle d'apprentissage online (SGD, BCE loss, 1 échantillon à la fois)

**Le backbone est gelé** : aucun gradient ne doit remonter au-delà de la tête OtO. La seule partie mise à jour après déploiement est `OtOHead`.

**Critère de succès** : `trainer.update(x_window, y_label)` s'exécute sans erreur sur un seul échantillon, la RAM dynamique lors de l'update est < 100 octets (tracemalloc), et `pytest tests/test_tinyol.py -k oto` passe.

---

## Architecture cible

Conforme à `tinyol_spec.md §2.2` :

```
Input OtO : [9]   ← embedding z (8D) + MSE scalaire (1D)
   │
   └── Linear(9 → 1) + Sigmoid   # MEM: 36 B @ FP32 / 9 B @ INT8
       → probabilité de panne ŷ ∈ [0, 1]

Paramètres : 9×1 + 1 = 10 params → 40 octets @ FP32
```

---

## Sous-tâches

### 1. Classe `OtOHead`

```python
# src/models/tinyol/oto_head.py
"""
Tête OtO (One-to-One) pour TinyOL.

Architecture : Linear(9→1) + Sigmoid
Paramètres   : 10 (9 poids + 1 biais) → 40 octets @ FP32
MCU          : SGD uniquement, pas de momentum, pas d'Adam

Référence : Ren2021TinyOL, tinyol_spec.md §2.2 et §6
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OtOHead(nn.Module):
    """
    Tête One-to-One (OtO) entraînable du modèle TinyOL.

    Parameters
    ----------
    input_dim : int
        Dimension de l'entrée = embed_dim + 1 (MSE scalaire).
        Valeur attendue : 9 (8D embedding + 1D MSE).

    Notes
    -----
    Conformité MCU :
    - Pas d'Adam — état (m, v) trop coûteux en RAM.
    - ReLU absent (sortie = Sigmoid pour probabilité binaire).
    - Taille fixe : 10 paramètres = 40 octets @ FP32.
    """

    def __init__(self, input_dim: int = 9) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)  # MEM: 40 B @ FP32 / 10 B @ INT8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [input_dim] ou [batch, input_dim]

        Returns
        -------
        torch.Tensor, shape [1] ou [batch, 1]
            Probabilité de panne dans [0, 1].
        """
        return torch.sigmoid(self.fc(x))  # MEM: 4 B @ FP32 (scalaire)

    def n_params(self) -> int:
        """Retourne le nombre de paramètres (attendu : 10)."""
        return sum(p.numel() for p in self.parameters())
```

### 2. Classe `TinyOLOnlineTrainer`

```python
class TinyOLOnlineTrainer:
    """
    Boucle d'apprentissage online pour TinyOL.

    Encapsule le backbone gelé (TinyOLAutoencoder) et la tête OtO,
    et expose une méthode `update` pour la mise à jour échantillon par échantillon.

    Parameters
    ----------
    autoencoder : TinyOLAutoencoder
        Backbone pré-entraîné et gelé.
    oto_head : OtOHead
        Tête OtO entraînable.
    config : dict
        Configuration YAML (section `online`).

    Notes
    -----
    Conformité MCU — tinyol_spec.md §6 :
    - Optimiseur : SGD pur (pas de momentum → OTO_MOMENTUM=0.0)
    - Loss       : Binary Cross-Entropy
    - Fréquence  : 1 update par échantillon
    - Gradient   : limité à `oto_head` uniquement (backbone gelé)
    """

    def __init__(
        self,
        autoencoder: "TinyOLAutoencoder",
        oto_head: OtOHead,
        config: dict,
    ) -> None:
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)   # gel complet du backbone

        self.oto_head = oto_head
        self.optimizer = torch.optim.SGD(
            self.oto_head.parameters(),
            lr=config["online"]["learning_rate"],
            momentum=config["online"]["momentum"],   # doit être 0.0
        )

    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Effectue un pas d'apprentissage online sur un seul échantillon.

        Parameters
        ----------
        x : torch.Tensor, shape [25]
            Features normalisées d'une fenêtre.
        y : torch.Tensor, shape [1] ou scalaire
            Label binaire (0=normal, 1=panne).

        Returns
        -------
        float
            Valeur de la loss BCE pour cet échantillon.
        """
        # 1. Forward backbone gelé — pas de gradient
        with torch.no_grad():
            z, x_hat = self.autoencoder(x.unsqueeze(0))        # MEM: 32 B @ FP32 (z)
            mse = F.mse_loss(x_hat, x.unsqueeze(0)).unsqueeze(0)  # MEM: 4 B @ FP32

        # 2. Construction de l'entrée OtO : [embed_dim + 1] = [9]
        oto_input = torch.cat([z.squeeze(0), mse])              # MEM: 36 B @ FP32

        # 3. Forward + backward tête OtO
        y_hat = self.oto_head(oto_input)                        # MEM: 4 B @ FP32
        loss = F.binary_cross_entropy(
            y_hat.squeeze(), y.float().squeeze()
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Discard x, y — pas de stockage (online learning)
        return loss.item()

    def predict(self, x: torch.Tensor) -> tuple[float, float]:
        """
        Prédit sans mise à jour.

        Returns
        -------
        tuple[float, float]
            (probabilité_panne, mse_reconstruction)
        """
        with torch.no_grad():
            z, x_hat = self.autoencoder(x.unsqueeze(0))
            mse = F.mse_loss(x_hat, x.unsqueeze(0))
            oto_input = torch.cat([z.squeeze(0), mse.unsqueeze(0)])
            y_hat = self.oto_head(oto_input)
        return y_hat.item(), mse.item()
```

### 3. Mise à jour de `src/models/tinyol/__init__.py`

```python
# src/models/tinyol/__init__.py
from src.models.tinyol.autoencoder import TinyOLAutoencoder
from src.models.tinyol.oto_head import OtOHead, TinyOLOnlineTrainer

__all__ = ["TinyOLAutoencoder", "OtOHead", "TinyOLOnlineTrainer"]
```

### 4. Vérification budget mémoire

Ajouter en bas du fichier un bloc de vérification à exécuter manuellement :

```python
if __name__ == "__main__":
    import tracemalloc
    from src.models.tinyol.autoencoder import TinyOLAutoencoder

    config_online = {"online": {"learning_rate": 1e-2, "momentum": 0.0}}

    autoencoder = TinyOLAutoencoder()
    oto_head = OtOHead(input_dim=9)
    trainer = TinyOLOnlineTrainer(autoencoder, oto_head, config_online)

    x_dummy = torch.randn(25)
    y_dummy = torch.tensor(0.0)

    tracemalloc.start()
    _ = trainer.update(x_dummy, y_dummy)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"OtO params    : {oto_head.n_params()} (attendu : 10)")
    print(f"RAM peak update : {peak} B (cible : < 100 B hors PyTorch overhead)")
    # FIXME(gap2) : overhead PyTorch non représentatif de la RAM MCU réelle
    # → utiliser memory_profiler.py pour la mesure officielle dans S3-06
```

---

## Critères d'acceptation

- [x] `src/models/tinyol/oto_head.py` créé, importable sans erreur
- [x] `OtOHead(input_dim=9).n_params() == 10`
- [x] `OtOHead` : sortie dans `[0, 1]` pour toute entrée FP32
- [x] `TinyOLOnlineTrainer.update(x, y)` s'exécute sur un seul échantillon sans erreur
- [x] Backbone gelé : `sum(p.requires_grad for p in autoencoder.parameters()) == 0`
- [x] Optimiseur SGD sans momentum (`momentum=0.0` dans la config)
- [x] Aucune valeur hardcodée : `learning_rate` et `momentum` lus depuis `tinyol_config.yaml`
- [x] Annotations `# MEM:` présentes sur chaque tenseur intermédiaire
- [x] `ruff check src/models/tinyol/oto_head.py` + `black --check` passent
- [ ] `pytest tests/test_tinyol.py -k oto -v` passe (dépend de S3-07)

---

## Commande complète

```bash
# Prérequis : S3-03 et S3-04 terminés
pip install -e ".[dev]"

# Vérification rapide
python -c "from src.models.tinyol.oto_head import OtOHead, TinyOLOnlineTrainer; print('Import OK')"

# Vérification budget mémoire (bloc __main__ du fichier)
python src/models/tinyol/oto_head.py

# Linting
ruff check src/models/tinyol/oto_head.py
black --check src/models/tinyol/oto_head.py
```

---

## Questions ouvertes

- `TODO(dorra)` : l'optimiseur SGD sans momentum est-il suffisant pour la convergence sur les données industrielles du Dataset 1, ou faut-il envisager SGD avec momentum très faible (0.1) tout en restant dans le budget RAM ?
- `TODO(arnaud)` : la feature d'entrée OtO est `[z ∈ ℝ⁸, mse ∈ ℝ]` = 9D. Faut-il normaliser le scalaire MSE avant concaténation (min-max ou log) pour éviter des gradients déséquilibrés ?
- `FIXME(gap2)` : la mesure RAM avec `tracemalloc` inclut l'overhead Python et PyTorch, non représentatif de la RAM MCU. La mesure officielle sera faite dans S3-06 avec `memory_profiler.py` et documentée dans `experiments/exp_003_tinyol_dataset1/results/metrics.json`.
- `FIXME(gap3)` : explorer la possibilité d'une backprop INT8 sur la tête OtO (10 paramètres seulement) — à étudier en Phase 2 avec Dorra.
