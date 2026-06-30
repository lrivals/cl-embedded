#!/usr/bin/env python3
"""battery_feature_selection.py — Sélection top-5 features Battery pour le board.

Le board NUCLEO-F439ZI traite 5 features (HDC_N_FEATURES=5, EWC_IN=5, MAHA_DIM=5).
Le dataset Battery expose 7 features électrochimiques. Ce script classe les 7
features par information mutuelle avec ``faulty`` (seuil RUL par défaut) et émet
``configs/battery_feature_subset.yaml`` (5 indices retenus) — analogue à
``pronostia_feature_subset.yaml``.

Usage :
    python scripts/battery_feature_selection.py [--n-features 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from src.data.battery_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    RUL_FAILURE_THRESHOLD,
    load_raw_dataset,
)
from src.utils.reproducibility import set_seed  # noqa: E402

CSV_PATH = Path("data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv")
OUT_PATH = Path("configs/battery_feature_subset.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sélection top-5 features Battery (board)")
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    set_seed(42)
    df = load_raw_dataset(args.csv, rul_failure_threshold=RUL_FAILURE_THRESHOLD)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["faulty"].to_numpy(dtype=np.int64)

    mi = mutual_info_classif(X, y, random_state=42)
    order = np.argsort(mi)[::-1]
    top = sorted(order[: args.n_features].tolist())  # indices triés croissants

    subset = {
        "method": "mutual_info",
        "n_features_total": len(FEATURE_COLUMNS),
        "n_features_selected": args.n_features,
        "feature_indices": [int(i) for i in top],
        "feature_names": [FEATURE_COLUMNS[i] for i in top],
        "ranking": {FEATURE_COLUMNS[i]: round(float(mi[i]), 4) for i in order},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# battery_feature_subset.yaml — Top-5 features Battery pour board (N_FEATURES=5)\n"
        "# Généré par scripts/battery_feature_selection.py\n"
    )
    args.out.write_text(header + yaml.safe_dump(subset, sort_keys=False, allow_unicode=True))
    print(f"[battery] subset écrit → {args.out}")
    print(f"  indices={top}  noms={subset['feature_names']}")


if __name__ == "__main__":
    main()
