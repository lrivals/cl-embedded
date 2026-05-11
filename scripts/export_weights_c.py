"""
export_weights_c.py — Export des poids Python vers tableaux C statiques.

Génère firmware/stm32f4_blink/inc/model_weights.h depuis :
  - Un checkpoint MahalanobisDetector (.pkl)
  - Un checkpoint EWCMlpClassifier (.pt) (optionnel)

Usage :
    python scripts/export_weights_c.py --mahal <path.pkl> [--ewc <path.pt>] [--out <dir>]

Exemple :
    python scripts/export_weights_c.py \
        --mahal experiments/exp_007/checkpoints/mahalanobis_task0.pkl \
        --out firmware/stm32f4_blink/inc/
"""

from __future__ import annotations

import argparse
import pickle
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector


# ── Helpers ───────────────────────────────────────────────────────────────

def _fmt_float(v: float) -> str:
    return f"{v:.8f}f"


def _array1d_to_c(name: str, arr: np.ndarray) -> str:
    vals = ", ".join(_fmt_float(float(v)) for v in arr.flatten())
    return f"static const float {name}[{arr.size}] = {{{vals}}};"


def _array2d_to_c(name: str, arr: np.ndarray) -> str:
    rows, cols = arr.shape
    inner = ",\n    ".join(
        "{" + ", ".join(_fmt_float(float(v)) for v in row) + "}"
        for row in arr
    )
    return f"static const float {name}[{rows}][{cols}] = {{\n    {inner}\n}};"


# ── Export Mahalanobis ─────────────────────────────────────────────────────

def export_mahalanobis_to_c(
    detector: "MahalanobisDetector",
    out_path: Path,
    zscore_mean: np.ndarray | None = None,
    zscore_std: np.ndarray | None = None,
) -> None:
    """
    Génère inc/model_weights.h depuis un MahalanobisDetector entraîné.

    Parameters
    ----------
    detector : MahalanobisDetector
        Modèle chargé depuis pickle (mu_ et sigma_inv_ requis).
    out_path : Path
        Répertoire de sortie (inc/).
    zscore_mean : np.ndarray [d] | None
        Moyenne Z-score par feature. Si None : vecteur nul.
    zscore_std : np.ndarray [d] | None
        Écart-type Z-score par feature. Si None : vecteur unité.
    """
    assert detector.mu_ is not None, "fit_task() requis avant export"
    assert detector.sigma_inv_ is not None, "fit_task() requis avant export"

    d = detector.n_features_
    mu = detector.mu_.astype(np.float32)
    sigma_inv = detector.sigma_inv_.astype(np.float32)
    threshold = float(detector.threshold_) if detector.threshold_ is not None else 1.0

    if zscore_mean is None:
        zscore_mean = np.zeros(d, dtype=np.float32)
    if zscore_std is None:
        zscore_std = np.ones(d, dtype=np.float32)

    lines = [
        "/**",
        " * model_weights.h — Poids Mahalanobis générés automatiquement",
        " * Généré par scripts/export_weights_c.py — ne pas modifier à la main.",
        " */",
        "",
        "#pragma once",
        '#include "mahalanobis.h"',
        "",
        f"/* d = {d}, seuil = {threshold:.6f} */",
        "",
        _array1d_to_c("ZSCORE_MEAN", zscore_mean),
        _array1d_to_c("ZSCORE_STD", zscore_std),
        "",
        _array1d_to_c("MAHA_MEAN_INIT", mu),
        "",
        _array2d_to_c("MAHA_PRECISION_INIT", sigma_inv),
        "",
        f"static const float MAHA_THRESHOLD_INIT = {_fmt_float(threshold)};",
        f"static const float MAHA_EMA_ALPHA      = {_fmt_float(0.1)};",
    ]

    header_path = out_path / "model_weights.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] model_weights.h écrit → {header_path}")
    print(f"  μ shape={mu.shape}, Σ⁻¹ shape={sigma_inv.shape}, seuil={threshold:.6f}")


# ── Export EWC MLP ─────────────────────────────────────────────────────────

def export_ewc_head_to_c(
    model_path: Path,
    out_path: Path,
    ewc_lambda: float = 400.0,
) -> None:
    """
    Génère inc/ewc_weights.h depuis un checkpoint EWCMlpClassifier (.pt).

    Parameters
    ----------
    model_path : Path
        Chemin vers le .pt (state_dict format {'model_state_dict': ...}).
    out_path : Path
        Répertoire de sortie (inc/).
    ewc_lambda : float
        Coefficient λ EWC (depuis configs/ewc_config.yaml).
    """
    import torch  # noqa: PLC0415 — import tardif, torch optionnel

    checkpoint = torch.load(model_path, map_location="cpu")
    sd = checkpoint.get("model_state_dict", checkpoint)

    def _get(key: str) -> np.ndarray:
        return sd[key].detach().cpu().numpy().astype(np.float32)

    w1 = _get("fc1.weight")   # [H1, IN]
    b1 = _get("fc1.bias")     # [H1]
    w2 = _get("fc2.weight")   # [H2, H1]
    b2 = _get("fc2.bias")     # [H2]
    w3 = _get("fc3.weight")   # [OUT, H2]
    b3 = _get("fc3.bias")     # [OUT]

    lines = [
        "/**",
        " * ewc_weights.h — Poids EWC MLP générés automatiquement",
        " * Généré par scripts/export_weights_c.py — ne pas modifier à la main.",
        " */",
        "",
        "#pragma once",
        '#include "ewc_head.h"',
        "",
        _array2d_to_c("EWC_W1_INIT", w1),
        _array1d_to_c("EWC_B1_INIT", b1),
        "",
        _array2d_to_c("EWC_W2_INIT", w2),
        _array1d_to_c("EWC_B2_INIT", b2),
        "",
        _array2d_to_c("EWC_W3_INIT", w3),
        _array1d_to_c("EWC_B3_INIT", b3),
        "",
        f"static const float EWC_LAMBDA_INIT = {_fmt_float(ewc_lambda)};",
    ]

    header_path = out_path / "ewc_weights.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] ewc_weights.h écrit → {header_path}")
    print(f"  w1={w1.shape}, w2={w2.shape}, w3={w3.shape}, λ={ewc_lambda}")


# ── Export test vectors (validation bitwise Python vs C) ───────────────────

def export_test_vectors_h(
    mu: np.ndarray,
    sigma_inv: np.ndarray,
    out_path: Path,
    test_input: np.ndarray | None = None,
) -> None:
    """
    Génère tests/test_vectors.h : vecteur d'entrée + distance Mahalanobis
    attendue (calculée en numpy FP32), pour valider la correspondance bitwise
    entre l'implémentation C et Python.

    Parameters
    ----------
    mu : np.ndarray [d]
        Moyenne du détecteur (FP32).
    sigma_inv : np.ndarray [d, d]
        Matrice de précision Σ⁻¹ (FP32).
    out_path : Path
        Répertoire de sortie (tests/).
    test_input : np.ndarray [d] | None
        Vecteur d'entrée à utiliser. Si None : vecteur [1, 2, …, d].
    """
    d = mu.shape[0]
    mu = mu.astype(np.float32)
    sigma_inv = sigma_inv.astype(np.float32)

    if test_input is None:
        test_input = np.arange(1, d + 1, dtype=np.float32)
    else:
        test_input = test_input.astype(np.float32)

    # Distance Mahalanobis en FP32 strict (même arithmétique que le C)
    diff = test_input - mu
    left = sigma_inv @ diff
    dist_sq = float(np.dot(left, diff))
    dist = float(np.sqrt(max(dist_sq, 0.0)))

    lines = [
        "/**",
        " * test_vectors.h — Vecteurs de test générés par export_weights_c.py",
        " * Valider : dist C == dist Python (tolérance 1e-5 FP32).",
        " * NE PAS MODIFIER À LA MAIN.",
        " */",
        "",
        "#pragma once",
        "",
        f"/* d = {d} features */",
        _array1d_to_c("TV_MAHA_MEAN", mu),
        "",
        _array2d_to_c("TV_MAHA_PRECISION", sigma_inv),
        "",
        _array1d_to_c("TV_MAHA_INPUT", test_input),
        "",
        f"/* Distance numpy FP32 attendue (tolérance 1e-5 en C) */",
        f"static const float TV_MAHA_EXPECTED_DIST = {_fmt_float(dist)};",
    ]

    header_path = out_path / "test_vectors.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] test_vectors.h écrit → {header_path}")
    print(f"  input={test_input.tolist()}, dist_numpy={dist:.8f}")


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mahal", type=Path, default=None,
        help="Chemin vers le checkpoint MahalanobisDetector (.pkl)",
    )
    p.add_argument(
        "--ewc", type=Path, default=None,
        help="Chemin vers le checkpoint EWCMlpClassifier (.pt) [optionnel]",
    )
    p.add_argument(
        "--zscore", type=Path, default=None,
        help="YAML de normalisation (configs/*_normalizer*.yaml) [optionnel]",
    )
    p.add_argument(
        "--out", type=Path, default=Path("firmware/stm32f4_blink/inc"),
        help="Répertoire de sortie (défaut : firmware/stm32f4_blink/inc/)",
    )
    p.add_argument(
        "--ewc-lambda", type=float, default=400.0,
        help="Coefficient λ EWC pour l'export (défaut : 400.0)",
    )
    p.add_argument(
        "--dump-test-vectors", action="store_true",
        help="Génère firmware/stm32f4_blink/tests/test_vectors.h (validation bitwise C vs Python)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = args.out
    out_path.mkdir(parents=True, exist_ok=True)

    zscore_mean: np.ndarray | None = None
    zscore_std: np.ndarray | None = None

    if args.zscore is not None:
        import yaml  # noqa: PLC0415
        with open(args.zscore) as f:
            norm = yaml.safe_load(f)
        zscore_mean = np.array(norm["mean"], dtype=np.float32)
        zscore_std = np.array(norm["std"], dtype=np.float32)
        print(f"[export] Z-score chargé depuis {args.zscore}")

    detector = None
    if args.mahal is not None:
        with open(args.mahal, "rb") as f:
            detector = pickle.load(f)
        export_mahalanobis_to_c(detector, out_path, zscore_mean, zscore_std)
    else:
        print("[export] --mahal non fourni : model_weights.h inchangé")

    if args.ewc is not None:
        export_ewc_head_to_c(args.ewc, out_path, ewc_lambda=args.ewc_lambda)
    else:
        print("[export] --ewc non fourni : ewc_weights.h non généré")

    if args.dump_test_vectors:
        test_vectors_dir = Path("firmware/stm32f4_blink/tests")
        test_vectors_dir.mkdir(parents=True, exist_ok=True)
        if detector is not None:
            mu = detector.mu_.astype(np.float32)
            sigma_inv = detector.sigma_inv_.astype(np.float32)
        else:
            # Valeurs par défaut analytiques (mean=0, precision=I)
            d = 5
            mu = np.zeros(d, dtype=np.float32)
            sigma_inv = np.eye(d, dtype=np.float32)
        export_test_vectors_h(mu, sigma_inv, test_vectors_dir)


if __name__ == "__main__":
    main()
