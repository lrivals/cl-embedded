#!/usr/bin/env python3
"""
measure_macs.py — Cross-check des MACs analytiques (compute_cost.py) vs torchinfo.

Confronte les MACs estimés analytiquement par src/evaluation/compute_cost.py
(somme de produits de dimensions) aux MACs mesurés par `torchinfo` pour les
modèles torch (EWC, TinyOL). Produit une table d'écart en pourcentage.

HDC et Mahalanobis ne sont PAS des torch.nn.Module (encodage binaire /
hypervecteurs, distance quadratique avec Σ⁻¹) : torchinfo ne s'y applique pas.
Pour eux, le script rapporte `tool_applicable=False` avec une justification
écrite plutôt que d'inventer un wrapper torch artificiel.

Usage
-----
    python scripts/measure_macs.py --model ewc    --config configs/board_ewc.yaml
    python scripts/measure_macs.py --model tinyol --config configs/board_tinyol.yaml
    python scripts/measure_macs.py --model hdc    --config configs/board_hdc.yaml   # tool_applicable=False
    python scripts/measure_macs.py --model mahalanobis --config configs/board_mahalanobis.yaml

`torchinfo` est optionnel (deps dev). S'il est absent, le script l'indique
clairement pour les modèles torch sans crasher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permet l'exécution directe (python scripts/measure_macs.py) sans installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.compute_cost import compute_macs  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def measure_macs_torchinfo(model, input_shape: tuple) -> int:
    """Mesure les MACs (total_mult_adds) d'un modèle torch via torchinfo.

    Parameters
    ----------
    model : torch.nn.Module
        Modèle instancié.
    input_shape : tuple
        Forme de l'entrée incluant le batch (ex. (1, 5)).

    Returns
    -------
    int
        Nombre total de multiply-adds rapporté par torchinfo.

    Raises
    ------
    ImportError
        Si `torchinfo` n'est pas installé (deps dev — `pip install torchinfo`).
    """
    try:
        from torchinfo import summary
    except ImportError as exc:  # pragma: no cover - dépend de l'env
        raise ImportError(
            "torchinfo non installé — requis pour le cross-check des modèles torch. "
            "Installer via : pip install torchinfo (ou pip install -e \".[dev]\")."
        ) from exc

    stats = summary(model, input_size=input_shape, verbose=0)
    return int(stats.total_mult_adds)


def _build_ewc(cfg: dict):
    """Instancie EWCMlpClassifier depuis la config board (n_in, n_h1, n_h2)."""
    from src.models.ewc.ewc_mlp import EWCMlpClassifier

    n_in = int(cfg.get("n_in", cfg.get("EWC_IN", 5)))
    n_h1 = int(cfg.get("n_h1", cfg.get("EWC_H1", 32)))
    n_h2 = int(cfg.get("n_h2", cfg.get("EWC_H2", 16)))
    model = EWCMlpClassifier(input_dim=n_in, hidden_dims=[n_h1, n_h2])
    model.eval()
    # Sortie réelle du classifieur = 1 (sigmoïde binaire, fc3 → 1).
    macs_analytical = compute_macs("EWC", n_features=n_in, hidden_dims=[n_h1, n_h2], n_classes=1)
    return model, (1, n_in), macs_analytical


def _build_tinyol(cfg: dict):
    """Instancie TinyOLAutoencoder ; les dims sont introspectées du module.

    On lit les dimensions réelles des couches Linear pour que l'analytique
    (macs_tinyol_ae) et torchinfo voient strictement la même architecture.
    """
    from src.models.tinyol.autoencoder import TinyOLAutoencoder

    n_in = int(cfg.get("n_in", cfg.get("TINYOL_IN", 25)))
    model = TinyOLAutoencoder(input_dim=n_in)
    model.eval()

    encoder_dims = [model.enc1.out_features, model.enc2.out_features, model.enc3.out_features]
    decoder_dims = [model.dec1.out_features, model.dec2.out_features, model.dec3.out_features]
    macs_analytical = compute_macs(
        "TinyOL_AE",
        n_features=n_in,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
    )
    return model, (1, n_in), macs_analytical


def compare_analytical_vs_tool(
    model_name: str,
    model,
    input_shape: tuple | None,
    macs_analytical: int,
) -> dict:
    """Compare les MACs analytiques et l'outil torchinfo pour un modèle.

    Parameters
    ----------
    model_name : str
        Nom du modèle ("ewc", "tinyol", "hdc", "mahalanobis").
    model : torch.nn.Module | None
        Modèle torch instancié, ou None pour les modèles non-torch.
    input_shape : tuple | None
        Forme de l'entrée (avec batch), ou None si non applicable.
    macs_analytical : int
        MACs estimés par compute_cost.py.

    Returns
    -------
    dict
        {model, macs_analytical, macs_torchinfo, delta_pct, tool_applicable,
         justification}.
    """
    if model is None or input_shape is None:
        justification = {
            "hdc": (
                "HDC : encodage hyperdimensionnel binaire/bipolaire (bind + bundle), "
                "pas de couches linéaires torch.nn — torchinfo non applicable, "
                "estimation analytique seule."
            ),
            "mahalanobis": (
                "Mahalanobis : forme quadratique (x−μ)ᵀ Σ⁻¹ (x−μ), pas un "
                "torch.nn.Module — torchinfo non applicable, analytique seul."
            ),
        }.get(model_name, f"{model_name} : modèle non-torch, torchinfo non applicable.")
        return {
            "model": model_name,
            "macs_analytical": macs_analytical,
            "macs_torchinfo": None,
            "delta_pct": None,
            "tool_applicable": False,
            "justification": justification,
        }

    macs_tool = measure_macs_torchinfo(model, input_shape)
    delta_pct = (
        100.0 * (macs_analytical - macs_tool) / macs_tool if macs_tool else None
    )
    return {
        "model": model_name,
        "macs_analytical": macs_analytical,
        "macs_torchinfo": macs_tool,
        "delta_pct": delta_pct,
        "tool_applicable": True,
        "justification": (
            "Modèle torch (couches Linear) — comparaison analytique ↔ torchinfo. "
            "Écart résiduel attribuable aux termes que les deux comptes traitent "
            "différemment : torchinfo inclut les additions de biais (Σ out_features) "
            "alors que compute_cost ne compte que les produits in×out ; à l'inverse "
            "macs_tinyol_ae ajoute le terme MSE (n_features) que torchinfo ignore."
        ),
    }


_TORCH_BUILDERS = {"ewc": _build_ewc, "tinyol": _build_tinyol}


def _non_torch_analytical(model_name: str, cfg: dict) -> int:
    """MACs analytiques pour les modèles non-torch (HDC, Mahalanobis)."""
    if model_name == "hdc":
        return compute_macs(
            "HDC",
            n_features=int(cfg.get("HDC_N_FEATURES", cfg.get("n_features", 5))),
            dim_hv=int(cfg.get("HDC_DIM", cfg.get("dim_hv", 1000))),
            n_classes=int(cfg.get("HDC_N_CLASSES", cfg.get("n_classes", 2))),
        )
    if model_name == "mahalanobis":
        return compute_macs(
            "Mahalanobis",
            n_features=int(cfg.get("MAHA_DIM", cfg.get("n_features", 5))),
        )
    raise KeyError(f"Modèle non-torch inconnu : {model_name!r}")


def run(model_name: str, config_path: str) -> dict:
    """Charge la config, construit le modèle et retourne le dict de comparaison."""
    cfg = load_config(config_path)

    if model_name in _TORCH_BUILDERS:
        try:
            model, input_shape, macs_analytical = _TORCH_BUILDERS[model_name](cfg)
        except ImportError as exc:
            # torch absent : on rapporte au moins l'analytique sans outil.
            return {
                "model": model_name,
                "macs_analytical": None,
                "macs_torchinfo": None,
                "delta_pct": None,
                "tool_applicable": False,
                "justification": f"Dépendance manquante pour le modèle torch : {exc}",
            }
        try:
            return compare_analytical_vs_tool(model_name, model, input_shape, macs_analytical)
        except ImportError as exc:
            # torchinfo absent : analytique disponible, outil non.
            return {
                "model": model_name,
                "macs_analytical": macs_analytical,
                "macs_torchinfo": None,
                "delta_pct": None,
                "tool_applicable": False,
                "justification": str(exc),
            }

    # HDC / Mahalanobis : non-torch.
    macs_analytical = _non_torch_analytical(model_name, cfg)
    return compare_analytical_vs_tool(model_name, None, None, macs_analytical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["ewc", "tinyol", "hdc", "mahalanobis"],
        help="Modèle à cross-checker.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config YAML board fournissant les dimensions (ex. configs/board_ewc.yaml).",
    )
    args = parser.parse_args()

    result = run(args.model, args.config)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
