"""
cmapss_feature_selection.py — Sélection top-N features CMAPSS par mutual info.

Fit uniquement sur FD001 (train set) pour éviter la fuite de données inter-domaines.
Sauvegarde dans configs/cmapss_feature_subset.yaml.

Usage :
    python scripts/cmapss_feature_selection.py
    python scripts/cmapss_feature_selection.py --n-features 5 --subset-id FD001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif

CMAPSS_FAULTY_THRESHOLD: int = 30
CMAPSS_RUL_CAP: int = 125

SENSOR_NAMES: list[str] = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30",
    "Nf", "Nc", "epr", "Ps30", "Phi", "NRf", "NRc",
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]
COL_NAMES: list[str] = (
    ["unit_nr", "time_cycles", "op1", "op2", "op3"] + SENSOR_NAMES
)

DATA_DIR: Path = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
OUTPUT_PATH: Path = Path("configs/cmapss_feature_subset.yaml")


def _load_fd001(data_dir: Path, faulty_threshold: int, rul_cap: int) -> pd.DataFrame:
    csv = data_dir / "train_FD001.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv}")

    df = pd.read_csv(csv, sep=r"\s+", header=None, names=COL_NAMES)

    # RUL = max(time_cycles) par unité − time_cycles courant
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["RUL"] = (max_cycles - df["time_cycles"]).clip(upper=rul_cap)
    df["faulty"] = (df["RUL"] <= faulty_threshold).astype(int)

    return df


def select_features(
    data_dir: Path,
    n_features: int,
    subset_id: str,
    faulty_threshold: int,
    rul_cap: int,
) -> list[str]:
    df = _load_fd001(data_dir, faulty_threshold, rul_cap)

    X = df[SENSOR_NAMES].to_numpy(dtype=np.float32)
    y = df["faulty"].to_numpy()

    print(f"FD001 — {len(df)} lignes, taux faulty : {y.mean():.3f}")

    mi_scores = mutual_info_classif(X, y, random_state=42)
    ranked = sorted(zip(SENSOR_NAMES, mi_scores), key=lambda t: t[1], reverse=True)

    print("\nScores mutual info (top 10) :")
    for name, score in ranked[:10]:
        print(f"  {name:>12s} : {score:.4f}")

    selected = [name for name, _ in ranked[:n_features]]
    return selected


def save_yaml(
    output_path: Path,
    selected: list[str],
    n_features: int,
    subset_id: str,
    faulty_threshold: int,
    method: str = "mutual_info_classif",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        # selected_features : clé lue par sensor_stream.py (S2306)
        "selected_features": selected,
        # features : alias lu par cmapss_loader.py (_load_feature_selection)
        "features": selected,
        "n_features": n_features,
        "fit_subset": subset_id,
        "method": method,
        "faulty_threshold": faulty_threshold,
    }
    with open(output_path, "w") as f:
        f.write(f"# cmapss_feature_subset.yaml — Top-{n_features} features CMAPSS\n")
        f.write(f"# Généré par scripts/cmapss_feature_selection.py — NE PAS éditer manuellement.\n")
        f.write(f"# Fit sur {subset_id} uniquement.\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\nSauvegardé : {output_path}")
    print(f"Features sélectionnées : {selected}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sélection top-N features CMAPSS par mutual_info_classif (fit FD001)."
    )
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--subset-id", type=str, default="FD001")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--faulty-threshold", type=int, default=CMAPSS_FAULTY_THRESHOLD)
    parser.add_argument("--rul-cap", type=int, default=CMAPSS_RUL_CAP)
    args = parser.parse_args()

    selected = select_features(
        data_dir=args.data_dir,
        n_features=args.n_features,
        subset_id=args.subset_id,
        faulty_threshold=args.faulty_threshold,
        rul_cap=args.rul_cap,
    )
    save_yaml(
        output_path=args.output,
        selected=selected,
        n_features=args.n_features,
        subset_id=args.subset_id,
        faulty_threshold=args.faulty_threshold,
    )


if __name__ == "__main__":
    main()
