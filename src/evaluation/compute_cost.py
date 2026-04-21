"""
compute_cost.py — Estimation analytique du coût de calcul (MACs) par modèle.

Fournit des formules analytiques (multiply-accumulate) pour une inférence
de chacun des 6 modèles du benchmark. Indépendant de la machine hôte :
transposable au STM32N6 (contrainte latence ≤ 100 ms, cf. CLAUDE.md).

Méthodologie : compte uniquement les opérations multiply-accumulate dominantes
(produits matriciels, distances, projections). Les opérations mineures
(activations, normalisations) sont négligées — elles représentent typiquement
< 5 % du coût total pour les MLP petits et les modèles hyperdimensionnels.

Référence : les chiffres produits ici sont un PROXY portable. Les cycles CPU
réels sur Cortex-M55 dépendent du compilateur, du pipeline, et de l'usage
éventuel du NPU pour l'inférence (backprop toujours en SW).
"""

from __future__ import annotations


def macs_ewc_mlp(
    n_features: int, hidden_dims: list[int] | tuple[int, ...], n_classes: int
) -> int:
    """MACs par inférence d'un MLP (EWC Online).

    Architecture : chaîne Linear(n_features, h1) → ReLU → Linear(h1, h2) → ...
    → Linear(h_last, n_classes). Le coût MACs somme les produits successifs
    d'entrée × sortie de chaque Linear.

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    hidden_dims : list[int] | tuple[int, ...]
        Tailles successives des couches cachées (ex. [32, 16] pour EWC pump).
    n_classes : int
        Nombre de classes en sortie (1 pour sortie binaire sigmoïde).

    Returns
    -------
    int
        Nombre de multiply-accumulates par forward pass.
    """
    dims = [n_features, *hidden_dims, n_classes]
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def macs_tinyol(
    n_features: int,
    encoder_dims: list[int] | tuple[int, ...],
    n_classes: int,
) -> int:
    """MACs par inférence d'un TinyOL (encodeur + tête OtO).

    Architecture : encoder Linear(n_features, enc_1) → ... → Linear(enc_k-1,
    enc_k) → tête Linear(enc_k, n_classes). Le décodeur n'est pas utilisé
    en inférence (il sert à la reconstruction pendant l'entraînement).

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    encoder_dims : list[int] | tuple[int, ...]
        Tailles successives des couches de l'encodeur (la dernière est le
        latent / bottleneck).
    n_classes : int
        Nombre de classes en sortie de la tête.
    """
    dims = [n_features, *encoder_dims, n_classes]
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def macs_hdc(n_features: int, dim_hv: int, n_classes: int) -> int:
    """MACs par inférence d'un classifieur HDC (hyperdimensional).

    Architecture : encoding (bind feature→hypervecteur D) puis similarité
    cosinus entre hypervecteur requête et chaque prototype de classe.

    Parameters
    ----------
    n_features : int
        Nombre de features d'entrée (chaque feature binde un item vector).
    dim_hv : int
        Dimension des hypervecteurs (D, typiquement 2048–10000).
    n_classes : int
        Nombre de prototypes (un par classe).
    """
    return dim_hv * n_features + dim_hv * n_classes


def macs_kmeans(n_features: int, n_clusters: int) -> int:
    """MACs par inférence d'un classifieur basé KMeans.

    Distance euclidienne aux `n_clusters` centroïdes, chaque distance coûte
    `n_features` MACs (squared-diff + accumulation).

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    n_clusters : int
        Nombre de centroïdes (K).
    """
    return n_clusters * n_features


def macs_mahalanobis(n_features: int) -> int:
    """MACs par inférence d'un détecteur Mahalanobis.

    Distance de Mahalanobis : (x − μ)ᵀ Σ⁻¹ (x − μ). La forme quadratique
    avec Σ⁻¹ pré-calculée coûte n_features² MACs pour Σ⁻¹·(x−μ) puis
    n_features MACs pour le produit scalaire final.

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    """
    return n_features * n_features + n_features


def macs_dbscan(n_features: int, n_core_samples: int) -> int:
    """MACs par inférence DBSCAN (recherche du core sample le plus proche).

    DBSCAN ne produit pas de modèle paramétrique : l'inférence consiste
    à calculer la distance à chaque core sample retenu après fit.

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    n_core_samples : int
        Nombre de core samples issus du fit.
    """
    return n_core_samples * n_features


_DISPATCH = {
    "EWC": macs_ewc_mlp,
    "TinyOL": macs_tinyol,
    "HDC": macs_hdc,
    "KMeans": macs_kmeans,
    "Mahalanobis": macs_mahalanobis,
    "DBSCAN": macs_dbscan,
}


def compute_macs(model_name: str, **kwargs: int) -> int:
    """Dispatcher — MACs d'une inférence pour un modèle du benchmark.

    Parameters
    ----------
    model_name : str
        Un parmi : "EWC", "TinyOL", "HDC", "KMeans", "Mahalanobis", "DBSCAN".
    **kwargs
        Paramètres propres au modèle (cf. fonctions macs_*). Voir chaque
        fonction pour la liste attendue.

    Returns
    -------
    int
        Nombre de MACs par forward pass.

    Raises
    ------
    KeyError
        Si `model_name` n'est pas reconnu.
    """
    if model_name not in _DISPATCH:
        raise KeyError(
            f"Modèle inconnu: {model_name!r}. "
            f"Attendu parmi: {sorted(_DISPATCH)}."
        )
    return _DISPATCH[model_name](**kwargs)
