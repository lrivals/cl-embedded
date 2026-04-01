# S1-05 — Implémenter `fisher.py` (calcul Fisher diagonale)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-05 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S1-04 (`EWCMlpClassifier` disponible) |
| **Fichier cible** | `src/models/ewc/fisher.py` |

---

## Objectif

Implémenter le calcul de la **diagonale de la matrice d'information de Fisher empirique** pour le modèle EWC, et la mise à jour Online (accumulation avec décroissance γ).

Le module doit être :
- Découplé du modèle (fonctionne sur tout `nn.Module`)
- Compatible avec la variante EWC Online (Schwarz et al., 2018) via le paramètre `gamma`
- Limité à `n_fisher_samples` exemples pour contrôler le coût de calcul (configurable via `ewc_config.yaml`)

**Critère de succès** : `tests/test_fisher.py` passe, et la Fisher produit bien des valeurs non-nulles et plus grandes sur les poids importants.

---

## Sous-tâches

### 1. Calcul de la Fisher diagonale empirique

```python
# src/models/ewc/fisher.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

# Nombre max d'échantillons pour estimer la Fisher — source : ewc_config.yaml → ewc.n_fisher_samples
N_FISHER_SAMPLES_DEFAULT: int = 200


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = N_FISHER_SAMPLES_DEFAULT,
) -> dict[str, torch.Tensor]:
    """
    Calcule la diagonale de la matrice de Fisher empirique.

    F_i ≈ E[(∂ log p(y|x,θ) / ∂θᵢ)²]

    Méthode : accumulation des carrés des gradients sur `n_samples` exemples,
    normalisée par le nombre de batches traités.

    Parameters
    ----------
    model : nn.Module
        Modèle dont on veut estimer l'importance des paramètres.
        Doit être en mode eval() pendant le calcul.
    dataloader : DataLoader
        Loader de la tâche courante (après son entraînement complet).
    device : torch.device
        Dispositif de calcul (cpu ou cuda).
    n_samples : int
        Nombre maximum d'exemples utilisés pour estimer la Fisher.
        Configurable via ewc_config.yaml → ewc.n_fisher_samples.

    Returns
    -------
    dict[str, torch.Tensor]
        Diagonale de Fisher par nom de paramètre, même shape que les paramètres.
        Valeurs ≥ 0 (carrés de gradients).

    Notes
    -----
    Cette fonction s'exécute une seule fois en fin de tâche, sur PC.
    Sur MCU, la Fisher est pré-calculée et chargée depuis Flash.
    Référence : docs/models/ewc_mlp_spec.md §5, Kirkpatrick2017EWC eq. 5

    Contrainte mémoire :
    - fisher dict : 769 scalaires × 4 B = ~3 Ko @ FP32 pour EWCMlpClassifier
    # MEM: 769 × 4 B = ~3 Ko @ FP32
    """
    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.eval()
    n_seen = 0
    n_batches = 0

    for x, y in dataloader:
        if n_seen >= n_samples:
            break

        x, y = x.to(device), y.to(device)
        model.zero_grad()

        output = model(x)
        loss = F.binary_cross_entropy(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.detach() ** 2

        n_seen += x.size(0)
        n_batches += 1

    if n_batches > 0:
        fisher = {name: f / n_batches for name, f in fisher.items()}

    return fisher
```

### 2. Mise à jour Online (EWC Online — Schwarz et al., 2018)

```python
def update_fisher_online(
    fisher_old: Optional[dict[str, torch.Tensor]],
    fisher_new: dict[str, torch.Tensor],
    gamma: float,
) -> dict[str, torch.Tensor]:
    """
    Accumule la Fisher diagonale avec décroissance (variante Online).

    F_online ← γ · F_old + F_new

    Avantage MCU : une seule copie de Fisher en RAM, overhead mémoire fixe
    quelle que soit le nombre de tâches vues.

    Parameters
    ----------
    fisher_old : dict[str, Tensor] ou None
        Fisher accumulée jusqu'à la tâche précédente.
        None pour la première tâche (initialise à zéro).
    fisher_new : dict[str, Tensor]
        Fisher calculée sur la tâche courante (output de compute_fisher_diagonal).
    gamma : float
        Facteur de décroissance ∈ [0, 1]. Depuis ewc_config.yaml → ewc.gamma.
        γ = 1.0 → accumulation pure (EWC multi-task classique)
        γ < 1.0 → oubli contrôlé des tâches anciennes (EWC Online)

    Returns
    -------
    dict[str, torch.Tensor]
        Fisher mise à jour, même structure que fisher_new.

    Références
    ----------
    Schwarz2018ProgressCompress eq. 4, docs/models/ewc_mlp_spec.md §1
    """
    if fisher_old is None:
        return {name: f.clone() for name, f in fisher_new.items()}

    return {
        name: gamma * fisher_old[name] + fisher_new[name]
        for name in fisher_new
    }
```

### 3. Utilitaire de diagnostic

```python
def fisher_stats(fisher: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """
    Retourne des statistiques descriptives de la Fisher par couche.

    Utile pour vérifier que la Fisher est non-triviale et diagnostiquer
    les poids les plus importants pour une tâche.

    Returns
    -------
    dict[str, dict] : {param_name: {"mean": float, "max": float, "sparsity": float}}
    """
    stats = {}
    for name, f in fisher.items():
        stats[name] = {
            "mean": f.mean().item(),
            "max": f.max().item(),
            "sparsity": (f == 0).float().mean().item(),
        }
    return stats
```

### 4. Écrire le test

Créer `tests/test_fisher.py` :

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.ewc import EWCMlpClassifier
from src.models.ewc.fisher import compute_fisher_diagonal, update_fisher_online


def _make_loader(n: int = 64, input_dim: int = 6) -> DataLoader:
    x = torch.randn(n, input_dim)
    y = torch.randint(0, 2, (n, 1)).float()
    return DataLoader(TensorDataset(x, y), batch_size=32)


def test_fisher_shape():
    """La Fisher doit avoir la même structure que les paramètres du modèle."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    for name, param in model.named_parameters():
        assert name in fisher
        assert fisher[name].shape == param.shape


def test_fisher_non_negative():
    """La Fisher est constituée de carrés de gradients → valeurs ≥ 0."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    for name, f in fisher.items():
        assert (f >= 0).all(), f"Fisher négative sur {name}"


def test_fisher_non_zero():
    """La Fisher ne doit pas être entièrement nulle sur un modèle entraînable."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    total = sum(f.sum().item() for f in fisher.values())
    assert total > 0, "Fisher entièrement nulle — gradients bloqués ?"


def test_update_fisher_online_none():
    """Avec fisher_old=None, update_fisher_online doit retourner une copie de fisher_new."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher_new = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_updated = update_fisher_online(None, fisher_new, gamma=0.9)
    for name in fisher_new:
        assert torch.allclose(fisher_updated[name], fisher_new[name])


def test_update_fisher_online_accumulates():
    """Après deux tâches, la Fisher Online doit être plus grande qu'après une seule."""
    model = EWCMlpClassifier()
    loader = _make_loader()
    fisher_t1 = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_t2 = compute_fisher_diagonal(model, loader, torch.device("cpu"))
    fisher_online = update_fisher_online(fisher_t1, fisher_t2, gamma=0.9)
    # γ·F_t1 + F_t2 > F_t2 si F_t1 > 0
    for name in fisher_t2:
        assert (fisher_online[name] >= fisher_t2[name]).all()


def test_n_fisher_samples_limit():
    """Avec n_samples petit, seule une partie du loader doit être utilisée."""
    model = EWCMlpClassifier()
    loader = _make_loader(n=256)
    # Ne pas planter avec n_samples < taille du loader
    fisher = compute_fisher_diagonal(model, loader, torch.device("cpu"), n_samples=32)
    assert fisher is not None
```

---

## Critères d'acceptation

- [ ] `from src.models.ewc.fisher import compute_fisher_diagonal, update_fisher_online` — aucune erreur
- [ ] Fisher retournée : même structure (`name → Tensor`) que `model.named_parameters()`
- [ ] Toutes les valeurs Fisher ≥ 0 (carrés de gradients)
- [ ] Fisher non entièrement nulle sur un modèle avec gradients actifs
- [ ] `update_fisher_online(None, fisher_new, gamma)` == copie de `fisher_new`
- [ ] `update_fisher_online(f_old, f_new, gamma=0.9)` == `0.9 * f_old + f_new`
- [ ] `n_samples` respecté : pas de crash si loader plus grand que `n_samples`
- [ ] Annotation `# MEM:` présente sur le dict Fisher (taille estimée)
- [ ] `pytest tests/test_fisher.py -v` — tous les tests passent
- [ ] `ruff check src/models/ewc/fisher.py` et `black --check` passent

---

## Rôle dans la boucle CL

```
Fin de Task i :
  fisher_new  = compute_fisher_diagonal(model, train_loader_i, device, n_samples=200)
  fisher      = update_fisher_online(fisher_old, fisher_new, gamma=0.9)   # accumulation
  theta_star  = model.get_theta_star()                                     # snapshot poids

Début de Task i+1 :
  loss = model.ewc_loss(x, y, fisher=fisher, theta_star=theta_star, ewc_lambda=1000.0)
```

Ce module est utilisé par `cl_trainer.py` (S1-06) — l'interface `compute_fisher_diagonal` et `update_fisher_online` ne doit pas changer sans mettre à jour le trainer.

---

## Questions ouvertes

- `TODO(arnaud)` : utiliser la Fisher empirique (gradients des labels vrais) ou la Fisher exacte (gradients des labels prédits) ? La version empirique est implémentée ici, conforme à Kirkpatrick2017EWC.
- `TODO(dorra)` : sur MCU, la Fisher est-elle calculée en ligne (batch par batch avec accumulation) ou offline puis chargée depuis Flash ? Impact sur `n_fisher_samples`.
- `FIXME(gap2)` : mesurer le coût RAM de `compute_fisher_diagonal` avec `tracemalloc` pour valider la contrainte ≤ 64 Ko (inclut les gradients intermédiaires de backprop).
