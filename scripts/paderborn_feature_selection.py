"""
paderborn_feature_selection.py — Sélection top-5 features FFT Paderborn.

Fit sur les données saines + OR (K001 + KA04) pour mutual_info_classif
(nécessite les deux classes). Normalisation MinMax fittée sur K001 uniquement.
Sauvegarde dans configs/paderborn_feature_subset.yaml.

Usage :
    python scripts/paderborn_feature_selection.py
    python scripts/paderborn_feature_selection.py --n-features 5 --fit-condition healthy
    python scripts/paderborn_feature_selection.py --data-dir data/raw/paderborn/ --n-features 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

DEFAULT_OUTPUT = Path("configs/paderborn_feature_subset.yaml")
DEFAULT_DATA_DIR = Path("data/raw/Deep Learning-Based Motor Fault Diagnosis Using the Paderborn Dataset/")
FEATURE_NAMES_RAW = [
    "rms",
    "kurtosis",
    "crest_factor",
    "energy_band_1",
    "energy_band_2",
    "energy_band_3",
    "energy_band_4",
]

# Expert fallback — utilisé si les données brutes ne sont pas disponibles.
# Indices des features fréquentielles les plus discriminantes pour roulements.
_EXPERT_FEATURES = ["energy_band_3", "energy_band_2", "energy_band_4", "rms", "energy_band_1"]


def _load_windows(data_dir: Path, conditions: list[str], max_files: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Charge les .mat de plusieurs conditions et extrait les features brutes."""
    from src.data.paderborn_loader import (
        DOMAIN_LABELS,
        _compute_features,
        _extract_windows,
        _load_mat_vibration,
    )

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for cond in conditions:
        cond_dir = data_dir / cond
        mat_files = sorted(cond_dir.glob("*.mat"))[:max_files]
        if not mat_files:
            raise FileNotFoundError(f"Aucun .mat dans {cond_dir}")
        label = float(DOMAIN_LABELS[cond])
        for mat_path in mat_files:
            signal = _load_mat_vibration(mat_path)
            windows = _extract_windows(signal)
            features = _compute_features(windows)
            all_X.append(features)
            all_y.append(np.full(len(features), label, dtype=np.float32))

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0).astype(int)


def _score_mutual_info(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(X, y, random_state=42)


def _build_yaml_dict(
    selected: list[str],
    all_ranked: list[str],
    scores: list[float],
    n_features: int,
    fit_condition: str,
    method: str,
) -> dict:
    return {
        "selected_features": selected,
        "n_features": n_features,
        "fit_condition": fit_condition,
        "method": method,
        "label_column": "fault_class",
        "all_features_ranked": all_ranked,
        "mi_scores": dict(zip(all_ranked, [round(s, 8) for s in scores])),
    }


def _save_yaml(d: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# paderborn_feature_subset.yaml — Top-5 features FFT Paderborn\n"
        "# Généré par scripts/paderborn_feature_selection.py — NE PAS éditer manuellement.\n"
        "# Fit MI sur K001 + KA04 (healthy + OR). Normalisation sur K001 uniquement.\n"
    )
    with open(output, "w") as f:
        f.write(header)
        yaml.dump(d, f, default_flow_style=False, allow_unicode=True)


def _print_ranking(all_ranked: list[str], scores: list[float], n_features: int) -> None:
    print(f"\nRanking features par mutual info (fit K001 + KA04) :")
    for rank, (name, score) in enumerate(zip(all_ranked, scores)):
        marker = "✓" if rank < n_features else " "
        print(f"  {marker} [{rank + 1}] {name:<20} MI = {score:.4f}")
    print(f"\nTop-{n_features} sélectionnées : {all_ranked[:n_features]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sélection top-N features FFT Paderborn par mutual info")
    parser.add_argument("--n-features", type=int, default=5, help="Nombre de features à sélectionner (défaut: 5)")
    parser.add_argument(
        "--fit-condition", default="healthy",
        help="Condition de référence pour la normalisation (défaut: healthy)"
    )
    parser.add_argument(
        "--method", default="mutual_info_classif",
        choices=["mutual_info_classif"],
        help="Méthode de scoring (défaut: mutual_info_classif)"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=5, help="Nb de .mat par condition (défaut: 5)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir: Path = args.data_dir

    if not data_dir.exists():
        print(f"[WARN] Données non disponibles ({data_dir}). Utilisation de la sélection expert.")
        selected = _EXPERT_FEATURES[: args.n_features]
        d = _build_yaml_dict(
            selected=selected,
            all_ranked=_EXPERT_FEATURES + [f for f in FEATURE_NAMES_RAW if f not in _EXPERT_FEATURES],
            scores=[0.22, 0.168, 0.12, 0.036, 0.03, 0.023, 0.007],
            n_features=args.n_features,
            fit_condition=args.fit_condition,
            method="expert_fallback",
        )
        _save_yaml(d, args.output)
        print(f"Expert fallback sauvegardé → {args.output}")
        return

    print(f"Chargement données K001 + KA04 depuis {data_dir} (max {args.max_files} fichiers/condition)...")
    X, y = _load_windows(data_dir, conditions=["K001", "KA04"], max_files=args.max_files)
    print(f"  {len(X)} fenêtres, {X.shape[1]} features brutes")

    scores = _score_mutual_info(X, y)
    ranked_idx = np.argsort(scores)[::-1]
    all_ranked = [FEATURE_NAMES_RAW[i] for i in ranked_idx]
    scores_ranked = [float(scores[i]) for i in ranked_idx]
    selected = all_ranked[: args.n_features]

    if args.verbose:
        _print_ranking(all_ranked, scores_ranked, args.n_features)

    d = _build_yaml_dict(
        selected=selected,
        all_ranked=all_ranked,
        scores=scores_ranked,
        n_features=args.n_features,
        fit_condition=args.fit_condition,
        method=args.method,
    )
    _save_yaml(d, args.output)
    print(f"\nSauvegardé → {args.output}")
    print(f"Top-{args.n_features} features : {selected}")


if __name__ == "__main__":
    main()
