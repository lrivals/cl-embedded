"""
fisher.py — Diagonale de la Fisher empirique pour EWC Online.

Implémente le calcul de la diagonale de la matrice d'information de Fisher
et sa mise à jour avec décroissance (variante Online, Schwarz et al., 2018).

Ce module est découplé du modèle : fonctionne sur tout nn.Module.

Références :
    Kirkpatrick et al. (2017). Overcoming catastrophic forgetting. PNAS. eq. 5
    Schwarz et al. (2018). Progress & compress. ICML. eq. 4
    docs/models/ewc_mlp_spec.md §5
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        loss = nn.functional.binary_cross_entropy(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.detach() ** 2

        n_seen += x.size(0)
        n_batches += 1

    if n_batches > 0:
        fisher = {name: f / n_batches for name, f in fisher.items()}

    return fisher


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

    return {name: gamma * fisher_old[name] + fisher_new[name] for name in fisher_new}


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
