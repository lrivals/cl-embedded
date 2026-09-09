#!/usr/bin/env python3
"""compute_article_cost.py — Coût de calcul de la tête EWC de l'article (Sprint 40).

Produit les MACs / FLOPs / BOPs / paramètres des deux cellules de l'article
« EWC INT8 sur MCU » : Monitoring (k=4) et Pronostia (k=5), tête multi-classe
``k -> 32 -> 16 -> 2`` — celle réellement portée sur carte (``ewc_head.c``,
``EWCMlpMulticlass``), et non la variante binaire de ``measure_macs.py``.

Le BOPs rend la comparaison FP32 / INT8 honnête : à FLOPs identiques,
``BOPs_INT8 / BOPs_FP32 = (8/32)² = 1/16``. Ce gain est **théorique** — la
latence réellement mesurée sur Cortex-M4 FPU *augmente* (paradoxe FPU,
Sprint 50) et l'énergie ne bouge pas (Sprint 53). Le JSON porte cet
avertissement pour que l'article ne le perde pas.

Aucune valeur n'est saisie à la main : tout provient de
``src/evaluation/compute_cost.py``. Le cross-check ``torchinfo`` est optionnel
et rapporté ``null`` s'il est indisponible (jamais 0).

Usage
-----
    python scripts/compute_article_cost.py
    python scripts/compute_article_cost.py --out experiments/exp_S40_article_metrics/compute_cost_ewc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.compute_cost import (  # noqa: E402
    compute_bops_for_model,
    compute_flops_for_model,
    compute_macs,
    compute_params_for_model,
)

DEFAULT_OUT = ROOT / "experiments" / "exp_S40_article_metrics" / "compute_cost_ewc.json"

#: Cellules de l'article — dimensions natives mesurées (Sprints 36/40/46/49).
CELLS: dict[str, dict[str, int]] = {
    "monitoring": {"n_features": 4, "n_classes": 2},
    "pronostia": {"n_features": 5, "n_classes": 2},
}
HIDDEN_DIMS = [32, 16]
FP32_BITS, INT8_BITS = 32, 8


def _torchinfo_macs(n_features: int, n_classes: int) -> tuple[int | None, str]:
    """MACs mesurés par torchinfo sur la tête multi-classe réelle, ou ``None``.

    Retourne ``(macs, justification)``. ``None`` si torch/torchinfo absents —
    l'analytique reste la source de vérité, aucune valeur n'est inventée.
    """
    try:
        import torch  # noqa: F401
        from torchinfo import summary

        from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass
    except ImportError as exc:
        return None, f"cross-check indisponible ({exc})"

    model = EWCMlpMulticlass(
        input_dim=n_features, n_classes=n_classes, hidden_dims=list(HIDDEN_DIMS)
    )
    model.eval()
    stats = summary(model, input_size=(1, n_features), verbose=0)
    return int(stats.total_mult_adds), (
        "torchinfo inclut les additions de biais (Σ out_features) que "
        "compute_cost ne compte pas — l'écart attendu vaut donc Σ des tailles "
        "de sortie des couches."
    )


def build() -> dict:
    """Construit le dict de coût de calcul pour les deux cellules de l'article."""
    cells: dict[str, dict] = {}
    for dataset, dims in CELLS.items():
        kwargs = {"hidden_dims": list(HIDDEN_DIMS), **dims}
        macs = compute_macs("EWC", **kwargs)
        flops = compute_flops_for_model("EWC", **kwargs)
        bops_fp32 = compute_bops_for_model("EWC", n_bits=FP32_BITS, **kwargs)
        bops_int8 = compute_bops_for_model("EWC", n_bits=INT8_BITS, **kwargs)
        macs_tool, justification = _torchinfo_macs(dims["n_features"], dims["n_classes"])

        cells[dataset] = {
            "n_features": dims["n_features"],
            "n_classes": dims["n_classes"],
            "hidden_dims": list(HIDDEN_DIMS),
            "n_params": compute_params_for_model("EWC", **kwargs),
            "macs": macs,
            "flops": flops,
            "bops_fp32": bops_fp32,
            "bops_int8": bops_int8,
            "bops_ratio_fp32_over_int8": bops_fp32 / bops_int8 if bops_int8 else None,
            "macs_torchinfo": macs_tool,
            "macs_delta_vs_torchinfo": (macs - macs_tool) if macs_tool is not None else None,
            "cross_check_note": justification,
            "platform": "théorique",
        }

    return {
        "generated_by": "scripts/compute_article_cost.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "ewc",
        "architecture": "k -> 32 -> 16 -> 2 (EWCMlpMulticlass, tête portée ewc_head.c)",
        "source": "src/evaluation/compute_cost.py (analytique) + torchinfo (cross-check)",
        "cells": cells,
        "honesty_note": (
            "BOPs = FLOPs × n_bits² : le rapport FP32/INT8 = (32/8)² = 16 est un "
            "gain THÉORIQUE de coût arithmétique. Il ne se traduit ni en latence "
            "(la latence INT8 mesurée sur Cortex-M4 FPU est SUPÉRIEURE au FP32, "
            "Sprint 50) ni en énergie (identique à l'incertitude près, Sprint 53). "
            "Le bénéfice INT8 réellement mesuré porte sur la RAM des poids (÷4)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data = build()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[compute_article_cost] écrit → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
