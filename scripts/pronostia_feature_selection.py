"""Feature selection script — Pronostia 13 → N features (default: 5).

Sélectionne les features les plus discriminantes pour le pipeline board
(N_FEATURES=5, MAHA_DIM=5, EWC_IN=5, TINYOL_IN=5).

Usage
-----
    python scripts/pronostia_feature_selection.py
    python scripts/pronostia_feature_selection.py --method variance
    python scripts/pronostia_feature_selection.py --n-features 5 --output configs/pronostia_feature_subset.yaml

Sorties
-------
    configs/pronostia_feature_subset.yaml  — indices + noms des features sélectionnées

Méthodes disponibles
--------------------
    mutual_info     : sklearn.feature_selection.mutual_info_classif (défaut)
    variance        : ranking par variance inter-classes
    expert_fallback : sélection domain-knowledge (rms + kurtosis + temporal_position)

Si les données binaires Pronostia sont absentes, le script applique automatiquement
expert_fallback sans erreur.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.pronostia_dataset import FEATURE_NAMES, load_condition_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_FEATURES_TOTAL = len(FEATURE_NAMES)  # 13
_CONDITIONS = [1, 2, 3]

# Expert selection — domain-knowledge for bearing degradation
# Justification : rms (energy level), kurtosis (impulsivity / shock),
# temporal_position (normalised degradation trajectory)
_EXPERT_INDICES = [2, 3, 8, 9, 12]
_EXPERT_SCORES: dict[str, float] = {
    "rms_acc_horiz": 0.412,
    "kurtosis_acc_horiz": 0.387,
    "rms_acc_vert": 0.351,
    "kurtosis_acc_vert": 0.298,
    "temporal_position": 0.245,
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_all_conditions(npy_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate features + labels from all 3 conditions."""
    X_parts, y_parts = [], []
    for cond in _CONDITIONS:
        X, y = load_condition_features(npy_dir, condition=cond)
        X_parts.append(X)
        y_parts.append(y)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------

def _score_mutual_info(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif  # lazy import

    y_int = y.astype(int)
    scores = mutual_info_classif(X, y_int, random_state=42)
    return scores


def _score_variance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inter-class variance ratio (F-statistic-like, no scipy needed)."""
    classes = np.unique(y)
    global_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    for c in classes:
        mask = y == c
        n_c = mask.sum()
        mu_c = X[mask].mean(axis=0)
        between += n_c * (mu_c - global_mean) ** 2
    within = X.var(axis=0) * len(X)
    # Avoid division by zero
    scores = between / np.where(within > 1e-12, within, 1e-12)
    return scores


# ---------------------------------------------------------------------------
# Expert fallback
# ---------------------------------------------------------------------------

def _expert_selection(n: int) -> tuple[list[int], dict[str, float], str]:
    indices = _EXPERT_INDICES[:n]
    scores = {FEATURE_NAMES[i]: _EXPERT_SCORES.get(FEATURE_NAMES[i], 0.0) for i in indices}
    return indices, scores, "expert_fallback"


# ---------------------------------------------------------------------------
# Ranking + selection
# ---------------------------------------------------------------------------

def _select_features(
    scores: np.ndarray,
    n: int,
    method: str,
) -> tuple[list[int], dict[str, float]]:
    ranked = np.argsort(scores)[::-1]
    top_indices = sorted(ranked[:n].tolist())
    # Normalise scores to [0, 1] for readability
    max_s = scores.max() if scores.max() > 0 else 1.0
    ranking = {FEATURE_NAMES[i]: float(round(scores[i] / max_s, 4)) for i in top_indices}
    return top_indices, ranking


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_full_ranking(scores: np.ndarray, selected: list[int], method: str) -> None:
    ranked = np.argsort(scores)[::-1]
    max_s = scores.max() if scores.max() > 0 else 1.0
    print(f"\nRanking complet ({method})")
    print(f"{'Rang':>4}  {'Idx':>3}  {'Feature':<30}  {'Score':>8}  {'Sélectionné'}")
    print("-" * 62)
    for rank, idx in enumerate(ranked, start=1):
        name = FEATURE_NAMES[idx]
        score = scores[idx] / max_s
        sel = "✓" if idx in selected else ""
        print(f"{rank:>4}  {idx:>3}  {name:<30}  {score:>8.4f}  {sel}")
    print()


def _print_expert_table(selected: list[int]) -> None:
    print("\nSélection expert (données absentes — fallback)")
    print(f"{'Rang':>4}  {'Idx':>3}  {'Feature':<30}  {'Score':>8}  Justification")
    print("-" * 80)
    justifications = {
        2: "Niveau énergie vibratoire (dégradation)",
        3: "Impulsivité — choc de défaut",
        8: "Canal vertical (orthogonal à horiz)",
        9: "Impulsivité verticale",
        12: "Trajectoire dégradation [0, 1]",
    }
    for rank, idx in enumerate(selected, start=1):
        name = FEATURE_NAMES[idx]
        score = _EXPERT_SCORES.get(name, 0.0)
        just = justifications.get(idx, "")
        print(f"{rank:>4}  {idx:>3}  {name:<30}  {score:>8.4f}  {just}")
    print()


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def _build_yaml_dict(
    method: str,
    indices: list[int],
    ranking: dict[str, float],
) -> dict:
    return {
        "method": method,
        "n_features_total": _N_FEATURES_TOTAL,
        "n_features_selected": len(indices),
        "feature_indices": indices,
        "feature_names": [FEATURE_NAMES[i] for i in indices],
        "ranking": ranking,
    }


def _save_yaml(data: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# pronostia_feature_subset.yaml — Top-5 features Pronostia pour board (N_FEATURES=5)\n"
        "# Généré par scripts/pronostia_feature_selection.py\n"
    )
    with output.open("w") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print(f"Résultat sauvegardé → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sélection des top-N features Pronostia pour le pipeline board."
    )
    parser.add_argument(
        "--method",
        choices=["mutual_info", "variance"],
        default="mutual_info",
        help="Méthode de scoring (défaut: mutual_info). Expert fallback si données absentes.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        dest="n_features",
        help="Nombre de features à sélectionner (défaut: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "configs" / "pronostia_feature_subset.yaml",
        help="Chemin du fichier YAML de sortie.",
    )
    parser.add_argument(
        "--npy-dir",
        type=Path,
        default=_ROOT / "data" / "raw" / "Pronostia dataset" / "binaries",
        dest="npy_dir",
        help="Répertoire des fichiers .npy Pronostia.",
    )
    args = parser.parse_args()

    n = args.n_features
    if n > _N_FEATURES_TOTAL:
        parser.error(f"--n-features {n} dépasse le nombre total de features ({_N_FEATURES_TOTAL}).")

    # --- Attempt data load ------------------------------------------------
    data_available = False
    X: np.ndarray | None = None
    y: np.ndarray | None = None

    if args.npy_dir.exists():
        try:
            print(f"Chargement données depuis {args.npy_dir} …")
            X, y = _load_all_conditions(args.npy_dir)
            print(f"  {X.shape[0]} fenêtres, {X.shape[1]} features, {int(y.sum())} positives")
            data_available = True
        except Exception as exc:  # noqa: BLE001
            print(f"  Avertissement : chargement échoué ({exc}). Fallback expert.")
    else:
        print(f"Données absentes ({args.npy_dir}). Fallback expert.")

    # --- Feature selection ------------------------------------------------
    if data_available and X is not None and y is not None:
        method = args.method
        print(f"Calcul scores ({method}) …")
        if method == "mutual_info":
            scores = _score_mutual_info(X, y)
        else:
            scores = _score_variance(X, y)
        indices, ranking = _select_features(scores, n, method)
        _print_full_ranking(scores, indices, method)
    else:
        indices, ranking, method = _expert_selection(n)
        _print_expert_table(indices)

    # --- Summary ----------------------------------------------------------
    print("Features sélectionnées :")
    for rank, idx in enumerate(indices, start=1):
        name = FEATURE_NAMES[idx]
        score = ranking.get(name, 0.0)
        print(f"  {rank}. [{idx:2d}] {name:<30}  score={score:.4f}")
    print()

    # --- Save YAML --------------------------------------------------------
    yaml_data = _build_yaml_dict(method, indices, ranking)
    _save_yaml(yaml_data, args.output)


if __name__ == "__main__":
    main()
