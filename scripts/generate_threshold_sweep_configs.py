"""
generate_threshold_sweep_configs.py — Sprint 32 / S3202.

Génère les 15 configs de balayage du seuil RUL→faulty (5 seuils × 3 datasets)
dans ``configs/sweep/``, à partir des configs de base. Seul le champ seuil
(et ``label_mode`` pour Pronostia) est injecté — conformément à la règle
CLAUDE.md : aucun hyperparamètre n'est modifié dans le code source, tout passe
par YAML.

Usage
-----
    python scripts/generate_threshold_sweep_configs.py

Vérification
------------
    ls configs/sweep/*.yaml | wc -l   # attendu : 15
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.utils.config_loader import load_config

# Mapping (dataset → config de base, champ seuil injecté, liste de seuils).
# Seuils exprimés dans l'unité native du RUL de chaque dataset :
#   - CMAPSS    : cycles restants (rul_cap=125)
#   - Pronostia : secondes restantes (rul_cap≈300) — mode rul_threshold opt-in
#   - Battery   : cycles restants (durée de vie ≈1134)
SWEEPS: dict[str, tuple[str, str, list[int]]] = {
    "cmapss": (
        "configs/cmapss_config.yaml",
        "faulty_threshold",
        [10, 20, 30, 40, 50],
    ),
    "pronostia": (
        "configs/pronostia_config.yaml",
        "faulty_threshold",
        [24, 48, 72, 96, 120],
    ),
    "battery": (
        "configs/battery_config.yaml",
        "rul_failure_threshold",
        [67, 133, 200, 267, 333],
    ),
}

OUTPUT_DIR = Path("configs/sweep")


def generate() -> list[Path]:
    """Génère toutes les configs de balayage et retourne les chemins écrits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for dataset, (base_path, field, thresholds) in SWEEPS.items():
        base_cfg = load_config(base_path)

        for thr in thresholds:
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("data", {})
            cfg["data"][field] = thr

            # Pronostia : la base est le config binaire (failure_ratio par défaut).
            # On active explicitement la binarisation par seuil RUL (opt-in).
            if dataset == "pronostia":
                cfg["data"]["label_mode"] = "rul_threshold"

            out_path = OUTPUT_DIR / f"{dataset}_thr{thr}.yaml"
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    cfg,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            written.append(out_path)
            print(f"  écrit : {out_path}  ({field}={thr})")

    return written


def main() -> None:
    written = generate()
    print(f"\n{len(written)} configs de balayage générées dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
