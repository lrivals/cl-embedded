"""
compute_cost.py — Estimation analytique du coût de calcul (MACs) par modèle.

Fournit des formules analytiques (multiply-accumulate) pour une inférence
et pour l'entraînement de chacun des modèles du benchmark. Indépendant de
la machine hôte : transposable au STM32N6 (contrainte latence ≤ 100 ms,
cf. CLAUDE.md).

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


def macs_tinyol_ae(
    n_features: int,
    encoder_dims: list[int] | tuple[int, ...],
    decoder_dims: list[int] | tuple[int, ...] | None = None,
) -> int:
    """MACs par inférence TinyOL Autoencoder (encoder + decoder + MSE).

    Contrairement à macs_tinyol, la tête OtO est absente ; le décodeur
    part du bottleneck vers n_features.

    Pour encoder_dims=[4,4,2], decoder_dims=[4,4,4] :
      Encoder : 16+16+8 = 40 MACs
      Decoder : 8+16+16 = 40 MACs
      MSE     : 4 MACs → Total : 84 MACs

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    encoder_dims : list[int] | tuple[int, ...]
        Tailles successives des couches de l'encodeur (la dernière est le
        bottleneck).
    decoder_dims : list[int] | tuple[int, ...] | None
        Tailles des couches du décodeur, sans le bottleneck (ex. [4,4,4]
        pour une sortie finale de dimension decoder_dims[-1]=n_features).
        Si None, symétrie automatique depuis encoder_dims.
    """
    enc_dims = [n_features, *encoder_dims]
    encoder_macs = sum(enc_dims[i] * enc_dims[i + 1] for i in range(len(enc_dims) - 1))
    bottleneck = encoder_dims[-1]
    if decoder_dims is None:
        dec_dims = [bottleneck, *list(reversed(encoder_dims[:-1])), n_features]
    else:
        dec_dims = [bottleneck, *decoder_dims]
    decoder_macs = sum(dec_dims[i] * dec_dims[i + 1] for i in range(len(dec_dims) - 1))
    return encoder_macs + decoder_macs + n_features


# ---------------------------------------------------------------------------
# Coûts d'entraînement
# ---------------------------------------------------------------------------


def training_macs_hdc(n_features: int, dim_hv: int, n_samples: int) -> int:
    """MACs d'entraînement HDC one-class (accumulative, one-pass, sans gradient).

    Chaque sample effectue le même encoding que l'inférence (n_classes=1).

    Parameters
    ----------
    n_features : int
        Nombre de features d'entrée.
    dim_hv : int
        Dimension des hypervecteurs.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    """
    return n_samples * macs_hdc(n_features, dim_hv, n_classes=1)


def training_macs_tinyol_ae(
    n_features: int,
    encoder_dims: list[int] | tuple[int, ...],
    n_samples: int,
    n_epochs: int,
    batch_size: int,
    decoder_dims: list[int] | tuple[int, ...] | None = None,
) -> int:
    """MACs d'entraînement TinyOL AE (refit, forward + backward ≈ 3× forward).

    L'approximation forward+backward = 3× forward est standard pour les MLP
    avec Adam (forward ×1 + gradients ×2).

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    encoder_dims : list[int] | tuple[int, ...]
        Couches de l'encodeur.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    n_epochs : int
        Nombre d'epochs de refit.
    batch_size : int
        Taille de batch (accepté pour la signature, coût calculé par sample).
    decoder_dims : list[int] | tuple[int, ...] | None
        Couches du décodeur (cf. macs_tinyol_ae).
    """
    per_sample = macs_tinyol_ae(n_features, encoder_dims, decoder_dims)
    return n_epochs * n_samples * 3 * per_sample


def training_macs_kmeans(
    n_features: int,
    n_clusters: int,
    n_samples: int,
    k_min: int,
    k_max: int,
    n_init: int,
    max_iter: int,
) -> int:
    """MACs d'entraînement KMeans avec sélection K par silhouette.

    Trois phases :
    - K-selection : sum_{k=k_min}^{k_max} n_init × max_iter × n_samples × k × n_features
    - Silhouette  : (k_max-k_min+1) × n_samples² × n_features  (distances pairwise)
    - Final fit   : n_init × max_iter × n_samples × n_clusters × n_features

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    n_clusters : int
        K retenu pour le fit final.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    k_min, k_max : int
        Borne de la grille de recherche K.
    n_init : int
        Nombre de restarts aléatoires par K.
    max_iter : int
        Nombre maximum d'itérations Lloyd.
    """
    k_select = sum(
        n_init * max_iter * n_samples * k * n_features
        for k in range(k_min, k_max + 1)
    )
    silhouette = (k_max - k_min + 1) * n_samples**2 * n_features
    final_fit = n_init * max_iter * n_samples * n_clusters * n_features
    return k_select + silhouette + final_fit


def training_macs_mahalanobis(n_features: int, n_samples: int) -> int:
    """MACs d'entraînement Mahalanobis (refit : mean + covariance + inversion).

    - Mean       : n_samples × n_features
    - Covariance : n_samples × n_features²  (produits extérieurs)
    - Inversion  : n_features³              (élimination gaussienne)

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    """
    return n_samples * n_features + n_samples * n_features**2 + n_features**3


def training_macs_ewc(
    n_features: int,
    hidden_dims: list[int] | tuple[int, ...],
    n_classes: int,
    n_samples: int,
    n_epochs: int,
    batch_size: int,
) -> int:
    """MACs d'entraînement EWC MLP (forward + backward ≈ 3× forward).

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    hidden_dims : list[int] | tuple[int, ...]
        Couches cachées du MLP.
    n_classes : int
        Nombre de classes en sortie.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    n_epochs : int
        Nombre d'epochs par tâche.
    batch_size : int
        Taille de batch (accepté pour la signature, coût calculé par sample).
    """
    per_sample = macs_ewc_mlp(n_features, hidden_dims, n_classes)
    return n_epochs * n_samples * 3 * per_sample


def training_macs_tinyol(
    n_features: int,
    encoder_dims: list[int] | tuple[int, ...],
    n_classes: int,
    n_samples: int,
    n_epochs: int = 1,
) -> int:
    """MACs d'entraînement TinyOL OtO head (backbone figé, update online SGD).

    Le backbone (encodeur) est pré-entraîné et figé ; seule la tête OtO est
    mise à jour sample par sample. L'approximation forward+backward = 3× forward
    s'applique sur le forward pass complet (encodeur + tête).

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    encoder_dims : list[int] | tuple[int, ...]
        Couches de l'encodeur (backbone figé inclus dans le forward).
    n_classes : int
        Nombre de classes en sortie de la tête OtO.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    n_epochs : int
        Nombre d'epochs (1 pour l'update online standard).
    """
    per_sample = macs_tinyol(n_features, encoder_dims, n_classes)
    return n_epochs * n_samples * 3 * per_sample


def training_macs_dbscan(n_features: int, n_samples: int) -> int:
    """MACs d'entraînement DBSCAN (calcul pairwise distances = O(n²·d)).

    DBSCAN construit un graphe de voisinage en calculant toutes les distances
    pairwise : n_samples² × n_features MACs.

    Parameters
    ----------
    n_features : int
        Dimension du vecteur d'entrée.
    n_samples : int
        Nombre d'échantillons d'entraînement.
    """
    return n_samples**2 * n_features


_DISPATCH = {
    "EWC": macs_ewc_mlp,
    "TinyOL": macs_tinyol,
    "TinyOL_AE": macs_tinyol_ae,
    "HDC": macs_hdc,
    "KMeans": macs_kmeans,
    "Mahalanobis": macs_mahalanobis,
    "DBSCAN": macs_dbscan,
}

_TRAINING_DISPATCH = {
    "EWC": training_macs_ewc,
    "TinyOL": training_macs_tinyol,
    "TinyOL_AE": training_macs_tinyol_ae,
    "HDC": training_macs_hdc,
    "KMeans": training_macs_kmeans,
    "Mahalanobis": training_macs_mahalanobis,
    "DBSCAN": training_macs_dbscan,
}


def compute_macs(model_name: str, **kwargs: int) -> int:
    """Dispatcher — MACs d'une inférence pour un modèle du benchmark.

    Parameters
    ----------
    model_name : str
        Un parmi : "EWC", "TinyOL", "TinyOL_AE", "HDC", "KMeans",
        "Mahalanobis", "DBSCAN".
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


def compute_training_macs(model_name: str, **kwargs) -> int:
    """Dispatcher — MACs d'entraînement total pour un modèle du benchmark.

    Parameters
    ----------
    model_name : str
        Un parmi : "EWC", "TinyOL", "TinyOL_AE", "HDC", "KMeans",
        "Mahalanobis", "DBSCAN".
    **kwargs
        Paramètres propres au modèle (cf. fonctions training_macs_*).

    Returns
    -------
    int
        Nombre de MACs pour l'entraînement complet (une tâche CL).

    Raises
    ------
    KeyError
        Si `model_name` n'est pas reconnu.
    """
    if model_name not in _TRAINING_DISPATCH:
        raise KeyError(
            f"Modèle inconnu: {model_name!r}. "
            f"Attendu parmi: {sorted(_TRAINING_DISPATCH)}."
        )
    return _TRAINING_DISPATCH[model_name](**kwargs)
