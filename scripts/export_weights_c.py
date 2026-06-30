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
import json
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
        f"#define MAHA_NATIVE_DIM {d}   /* dim des poids ci-dessous (cf. MAHA_DIM au build, S3507) */",
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

    # Preserve non-Mahalanobis sections (e.g. TinyOL weights) from existing file
    header_path = out_path / "model_weights.h"
    _TINYOL_MARKER = "/* ── TinyOL"
    if header_path.exists():
        existing = header_path.read_text()
        idx = existing.find(_TINYOL_MARKER)
        if idx != -1:
            lines.append("")
            lines.append(existing[idx:].rstrip())

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


# ── Export tête EWC single-mode board (g_ewc_head, Sprint 32 / S3205) ──────

def export_ewc_head_board_to_c(model_path: Path, out_path: Path) -> None:
    """Génère inc/model_weights_ewc.h depuis un checkpoint EWCMlpMulticlass 5→32→16→2.

    Cible la tête EWC single-mode du firmware (``g_ewc_head``, EWC_IN=5, EWC_OUT=2).
    L'architecture EWCMlpMulticlass(input_dim=5, n_classes=2, hidden=[32,16]) est
    bit-pour-bit équivalente à ``ewc_forward`` (relu→relu→logits→argmax), d'où la
    parité board↔PC. nn.Linear.weight est [out, in] = layout board w[out][in].

    Le header émet ``#define EWC_HEAD_WEIGHTS_PROVIDED 1`` qui active le chargement
    Flash dans ``pipeline_init`` (sinon : init Xavier historique).
    """
    import torch  # noqa: PLC0415

    checkpoint = torch.load(model_path, map_location="cpu")
    sd = checkpoint.get("model_state_dict", checkpoint)

    def _get(key: str) -> np.ndarray:
        return sd[key].detach().cpu().numpy().astype(np.float32)

    w1, b1 = _get("fc1.weight"), _get("fc1.bias")   # [32, IN], [32]
    w2, b2 = _get("fc2.weight"), _get("fc2.bias")   # [16,32], [16]
    w3, b3 = _get("fc3.weight"), _get("fc3.bias")   # [2,16], [2]

    # IN est configurable au build (S3506/S3507 : `make EWC_IN=k`). Seules les couches
    # cachées sont figées (EWC_H1=32, EWC_H2=16) ; la dim d'entrée k varie par condition.
    ewc_in = int(w1.shape[1])
    assert w1.shape[0] == 32, f"w1 {w1.shape} — archi board EWC_H1=32 attendu"
    assert w3.shape == (2, 16), f"w3 {w3.shape} ≠ (2,16) — archi board EWC_OUT=2, EWC_H2=16"
    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("w3", w3), ("b3", b3)]:
        if not np.isfinite(arr).all():  # NaN/inf → "nanf" invalide en C ; échec explicite
            raise ValueError(f"EWC {name} contient NaN/inf — entraînement board divergé, "
                             "ré-entraîner (cf. clip_grad dans train_board_reference.py)")

    lines = [
        "/**",
        " * model_weights_ewc.h — Poids tête EWC single-mode (g_ewc_head) générés.",
        " * Généré par scripts/export_weights_c.py --ewc-head — ne pas modifier à la main.",
        f" * Parité : EWCMlpMulticlass({ewc_in}, 2, [32,16]) == ewc_forward (FP32).",
        f" * Dim d'entrée k={ewc_in} → builder avec `make EWC_IN={ewc_in}` (S3507).",
        " */",
        "",
        "#pragma once",
        '#include "ewc_head.h"',
        "",
        "#define EWC_HEAD_WEIGHTS_PROVIDED 1",
        f"#define EWC_HEAD_NATIVE_DIM {ewc_in}   /* dim k des poids ci-dessous (cf. EWC_IN au build) */",
        "",
        _array2d_to_c("EWC_W1_INIT", w1),
        _array1d_to_c("EWC_B1_INIT", b1),
        "",
        _array2d_to_c("EWC_W2_INIT", w2),
        _array1d_to_c("EWC_B2_INIT", b2),
        "",
        _array2d_to_c("EWC_W3_INIT", w3),
        _array1d_to_c("EWC_B3_INIT", b3),
    ]

    header_path = out_path / "model_weights_ewc.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] model_weights_ewc.h écrit → {header_path}")
    print(f"  w1={w1.shape}, w2={w2.shape}, w3={w3.shape} (g_ewc_head, EWC_IN={ewc_in}, parité board↔PC)")


# ── Export seuils du gate de dérive (Sprint 38 / S3803) ────────────────────

def export_drift_thresholds_to_c(thresholds_json: Path, out_path: Path) -> dict:
    """Émet inc/drift_thresholds.h depuis drift_thresholds.json (run_sprint38_pc).

    Le header généré pose ``DRIFT_THRESHOLDS_PROVIDED`` + les seuils calibrés sur
    l'enrôlement healthy. Consommé par pipeline.c sous ``-DEWC_AUTO_UPDATE`` (S3803).
    Jamais édité à la main (règle CLAUDE.md).

    Parameters
    ----------
    thresholds_json : Path
        JSON {fault_threshold, drift_threshold, window_size, drift_ratio} (S3802).
    out_path : Path
        Répertoire de sortie (firmware/stm32f4_blink/inc).

    Returns
    -------
    dict
        Les seuils chargés (pour log / réutilisation).
    """
    with open(thresholds_json) as f:
        th = json.load(f)
    fault = float(th["fault_threshold"])
    drift = float(th["drift_threshold"])
    window = int(th["window_size"])
    ratio = float(th["drift_ratio"])

    lines = [
        "/* drift_thresholds.h — Seuils du gate de dérive (Sprint 38 S3803) */",
        "/* GÉNÉRÉ par scripts/export_weights_c.py --drift-thresholds. NE PAS ÉDITER. */",
        f"/* Source : {thresholds_json} */",
        "",
        "#ifndef DRIFT_THRESHOLDS_H",
        "#define DRIFT_THRESHOLDS_H",
        "",
        "#define DRIFT_THRESHOLDS_PROVIDED 1",
        "",
        f"#define DRIFT_FAULT_THRESHOLD {_fmt_float(fault)}",
        f"#define DRIFT_DRIFT_THRESHOLD {_fmt_float(drift)}",
        f"#define DRIFT_WINDOW_SIZE     {window}",
        f"#define DRIFT_RATIO           {_fmt_float(ratio)}",
        "",
        "#endif /* DRIFT_THRESHOLDS_H */",
    ]
    header_path = out_path / "drift_thresholds.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] drift_thresholds.h écrit → {header_path}")
    print(f"  fault={fault:.4f} drift={drift:.4f} window={window} ratio={ratio}")
    return th


# ── Export méta-modèle (stacking, Sprint 31 / S3105) ───────────────────────

def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _meta_forward_np(weights: dict, feats: np.ndarray) -> float:
    """Reproduit meta_forward (C) en numpy FP32 → proba sigmoïde ∈ [0, 1].

    Sert de référence pour la parité C ↔ Python (test_meta_head.c).
    """
    feats = feats.astype(np.float32)
    if weights["kind"] == "logreg":
        w = np.asarray(weights["w"], dtype=np.float32)
        z = float(np.dot(w, feats)) + float(weights["b"])
        return float(_sigmoid(z))
    # mlp 1 couche cachée : relu(W1·x + b1) → sigmoid(W2·h + b2)
    w1 = np.asarray(weights["w1"], dtype=np.float32)   # [H, F]
    b1 = np.asarray(weights["b1"], dtype=np.float32)   # [H]
    w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(-1)  # [H]
    b2 = float(weights["b2"])
    hidden = np.maximum(w1 @ feats + b1, 0.0)
    z = float(np.dot(w2, hidden)) + b2
    return float(_sigmoid(z))


def export_meta_to_c(meta_json: Path, out_path: Path) -> dict:
    """Génère inc/meta_weights.h depuis un meta_weights.json (MetaLearner.export_weights).

    Parameters
    ----------
    meta_json : Path
        Chemin vers le JSON produit par `MetaLearner.export_weights()` (logreg ou mlp).
    out_path : Path
        Répertoire de sortie (inc/).

    Returns
    -------
    dict
        Les poids chargés (réutilisés pour les vecteurs de test).
    """
    import json  # noqa: PLC0415

    weights = json.loads(Path(meta_json).read_text())
    kind = weights["kind"]
    feat_names = weights.get("feature_names", [])

    if kind == "logreg":
        w = np.asarray(weights["w"], dtype=np.float32)
        b = float(weights["b"])
        n_features = int(w.size)
        hidden = 0
        body = [
            _array1d_to_c("META_W", w),
            f"static const float META_B = {_fmt_float(b)};",
        ]
    elif kind == "mlp":
        w1 = np.asarray(weights["w1"], dtype=np.float32)         # [H, F]
        b1 = np.asarray(weights["b1"], dtype=np.float32)         # [H]
        w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(-1)  # [H]
        b2 = float(weights["b2"])
        hidden, n_features = w1.shape
        body = [
            _array2d_to_c("META_W1", w1),
            _array1d_to_c("META_B1", b1),
            "",
            _array1d_to_c("META_W2", w2),
            f"static const float META_B2 = {_fmt_float(b2)};",
        ]
    else:
        raise ValueError(f"kind méta inconnu : {kind!r} (attendu logreg/mlp).")

    lines = [
        "/**",
        " * meta_weights.h — Poids méta-modèle (stacking) générés automatiquement",
        " * Généré par scripts/export_weights_c.py — ne pas modifier à la main.",
        f" * kind = {kind}, features = {feat_names}",
        " */",
        "",
        "#pragma once",
        "",
        f"#define META_N_FEATURES {n_features}",
        f"#define META_HIDDEN     {hidden}   /* 0 = logreg ; >0 = MLP 1 couche cachée */",
        "",
        *body,
    ]

    header_path = out_path / "meta_weights.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] meta_weights.h écrit → {header_path}")
    print(f"  kind={kind}, n_features={n_features}, hidden={hidden}")
    return weights


def export_meta_test_vectors_h(weights: dict, out_path: Path) -> None:
    """Génère tests/test_vectors_meta.h : feats de référence + sortie sigmoïde attendue.

    Valide la parité C ↔ Python de `meta_forward` (tolérance 1e-5 FP32).
    """
    if weights["kind"] == "logreg":
        n_features = len(weights["w"])
    else:
        n_features = np.asarray(weights["w1"]).shape[1]

    # Quelques vecteurs de features bornés [0, 1] (cohérent build_meta_features).
    test_inputs = np.array(
        [
            [0.1, 0.2, 0.0, 0.6][:n_features] + [0.5] * max(0, n_features - 4),
            [0.9, 0.8, 1.0, 0.3][:n_features] + [0.5] * max(0, n_features - 4),
            [0.5, 0.5, 0.0, 0.0][:n_features] + [0.5] * max(0, n_features - 4),
        ],
        dtype=np.float32,
    )
    expected = np.array([_meta_forward_np(weights, x) for x in test_inputs], dtype=np.float32)

    inner = ",\n    ".join(
        "{" + ", ".join(_fmt_float(float(v)) for v in row) + "}" for row in test_inputs
    )
    exp_vals = ", ".join(_fmt_float(float(v)) for v in expected)

    lines = [
        "/**",
        " * test_vectors_meta.h — Vecteurs de test méta (parité C ↔ Python).",
        " * Valider : meta_forward(C) == proba sigmoïde Python (tolérance 1e-5 FP32).",
        " * NE PAS MODIFIER À LA MAIN — généré par scripts/export_weights_c.py.",
        " */",
        "",
        "#pragma once",
        "",
        f"#define TV_META_N_CASES    {len(test_inputs)}",
        f"#define TV_META_N_FEATURES {n_features}",
        "",
        f"static const float TV_META_INPUT[TV_META_N_CASES][TV_META_N_FEATURES] = {{\n    {inner}\n}};",
        "",
        f"static const float TV_META_EXPECTED[TV_META_N_CASES] = {{{exp_vals}}};",
    ]

    header_path = out_path / "test_vectors_meta.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] test_vectors_meta.h écrit → {header_path}")
    print(f"  cases={len(test_inputs)}, expected={expected.tolist()}")


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


# ── Export Mahalanobis Q15 (Sprint 34 / S3407) ────────────────────────────

def _array1d_int_to_c(name: str, arr: np.ndarray, ctype: str) -> str:
    vals = ", ".join(str(int(v)) for v in arr.flatten())
    return f"static const {ctype} {name}[{arr.size}] = {{{vals}}};"


def _array2d_int_to_c(name: str, arr: np.ndarray, ctype: str) -> str:
    rows, cols = arr.shape
    inner = ",\n    ".join(
        "{" + ", ".join(str(int(v)) for v in row) + "}" for row in arr
    )
    return f"static const {ctype} {name}[{rows}][{cols}] = {{\n    {inner}\n}};"


def export_maha_q15_to_c(detector: "MahalanobisDetector", out_path: Path) -> dict:
    """Génère inc/mahalanobis_q15_weights.h depuis un détecteur Mahalanobis (S3407).

    mu_ → INT8 affine (mu_q8, scale, zero_point) ; sigma_inv_ → int16 Q15 (scale par-tenseur).
    Réutilise EXACTEMENT les quantifieurs de mahalanobis_int8.py (parité board↔PC garantie par
    construction). Le z-score reste celui de model_weights.h (export --mahal), partagé.

    Returns
    -------
    dict des poids quantifiés (réutilisable pour la génération de test-vectors).
    """
    from src.models.unsupervised.mahalanobis_int8 import (  # noqa: PLC0415
        _quantize_affine_int8,
        _quantize_sigma_inv_q15,
    )

    assert detector.mu_ is not None, "fit_task() requis avant export"
    assert detector.sigma_inv_ is not None, "fit_task() requis avant export"

    d = int(detector.n_features_)
    mu = detector.mu_.astype(np.float32)
    sigma_inv = detector.sigma_inv_.astype(np.float32)
    threshold = float(detector.threshold_) if detector.threshold_ is not None else 1.0

    mu_q8, mu_scale, mu_zp = _quantize_affine_int8(mu)
    # uint8 affine [0,255] → int8 stocké tel quel côté C avec zero_point (mu = (q - zp) * scale)
    sigma_q15, sigma_scale = _quantize_sigma_inv_q15(sigma_inv)

    lines = [
        "#pragma once",
        "/* mahalanobis_q15_weights.h — GÉNÉRÉ par export_weights_c.py --maha-q15 (S3407).",
        " * NE PAS MODIFIER À LA MAIN (règle CLAUDE.md). mu INT8 affine + sigma_inv int16 Q15. */",
        "#include <stdint.h>",
        "",
        "#define MAHA_Q15_WEIGHTS_PROVIDED",
        f"#define MAHA_Q15_NATIVE_DIM {d}",
        "",
        _array1d_int_to_c("MAHA_Q15_MU_Q8", mu_q8.astype(np.int32), "uint8_t"),
        f"static const float   MAHA_Q15_MU_SCALE   = {_fmt_float(mu_scale)};",
        f"static const int32_t MAHA_Q15_MU_ZP      = {int(mu_zp)};",
        _array2d_int_to_c("MAHA_Q15_SIGMA_INV", sigma_q15, "int16_t"),
        f"static const float   MAHA_Q15_SIGMA_SCALE = {_fmt_float(sigma_scale)};",
        f"static const float   MAHA_Q15_THRESHOLD   = {_fmt_float(threshold)};",
    ]
    header_path = out_path / "mahalanobis_q15_weights.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] mahalanobis_q15_weights.h écrit → {header_path}  (d={d})")
    return {
        "d": d,
        "mu_q8": mu_q8,
        "mu_scale": mu_scale,
        "mu_zp": mu_zp,
        "sigma_q15": sigma_q15,
        "sigma_scale": sigma_scale,
        "threshold": threshold,
    }


def export_maha_q15_test_vectors_h(detector: "MahalanobisDetector", out_path: Path) -> None:
    """Génère tests/test_vectors_q15.h : entrée + distance Q15 attendue (parité C↔Python, S3409).

    La distance attendue est calculée par MahalanobisDetectorInt8.score_q15 (chemin Python exact),
    de sorte que le test C (maha_q15_score) doit la reproduire à la tolérance flottante près.
    """
    from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8  # noqa: PLC0415

    d = int(detector.n_features_)

    # Reconstruire un détecteur Q15 calibré identique pour produire la distance de référence.
    q15 = MahalanobisDetectorInt8({"quantization": "q15"})
    q15.mu_ = detector.mu_.astype(np.float32)
    q15.sigma_inv_ = detector.sigma_inv_.astype(np.float32)
    q15.threshold_ = detector.threshold_
    q15.n_features_ = d
    q15.calibrate_q15()

    rng = np.random.default_rng(34)
    test_input = (detector.mu_.astype(np.float32) + rng.standard_normal(d).astype(np.float32))
    dist = q15.score_q15(test_input)

    lines = [
        "#pragma once",
        "/* test_vectors_q15.h — GÉNÉRÉ par export_weights_c.py --maha-q15-test-vectors (S3409).",
        " * Parité forward Q15 C (maha_q15_score) ↔ Python (score_q15). NE PAS MODIFIER À LA MAIN. */",
        "#include <stdint.h>",
        "",
        f"#define TV_Q15_DIM {d}",
        _array1d_int_to_c("TV_Q15_MU_Q8", q15.mu_q_.astype(np.int32), "uint8_t"),
        f"static const float   TV_Q15_MU_SCALE   = {_fmt_float(q15._mu_scale)};",
        f"static const int32_t TV_Q15_MU_ZP      = {int(q15._mu_zp)};",
        _array2d_int_to_c("TV_Q15_SIGMA_INV", q15.sigma_inv_q15_, "int16_t"),
        f"static const float   TV_Q15_SIGMA_SCALE = {_fmt_float(q15._sigma_q15_scale)};",
        _array1d_to_c("TV_Q15_INPUT", test_input),
        f"static const float   TV_Q15_EXPECTED_DIST = {_fmt_float(dist)};",
    ]
    header_path = out_path / "test_vectors_q15.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] test_vectors_q15.h écrit → {header_path}  dist_q15={dist:.8f}")


def export_maha_int8_to_c(detector: "MahalanobisDetector", out_path: Path) -> dict:
    """Génère inc/mahalanobis_int8_weights.h depuis un détecteur Mahalanobis (S2912).

    mu_ → INT8 affine (mu_q8, scale, zero_point) ; sigma_inv_ → INT8 affine (scale, zero_point
    par-matrice). Réutilise EXACTEMENT les quantifieurs de mahalanobis_int8.py (parité board↔PC
    garantie par construction). Reproduit fidèlement la dégradation INT8 sur grande dynamique
    (≠ Q15 S3407) — le but est de MESURER ce comportement sur board.

    Returns
    -------
    dict des poids quantifiés (réutilisable pour la génération de test-vectors).
    """
    from src.models.unsupervised.mahalanobis_int8 import (  # noqa: PLC0415
        _quantize_affine_int8,
    )

    assert detector.mu_ is not None, "fit_task() requis avant export"
    assert detector.sigma_inv_ is not None, "fit_task() requis avant export"

    d = int(detector.n_features_)
    mu = detector.mu_.astype(np.float32)
    sigma_inv = detector.sigma_inv_.astype(np.float32)
    threshold = float(detector.threshold_) if detector.threshold_ is not None else 1.0

    mu_q8, mu_scale, mu_zp = _quantize_affine_int8(mu)
    sigma_q8, sigma_scale, sigma_zp = _quantize_affine_int8(sigma_inv)

    lines = [
        "#pragma once",
        "/* mahalanobis_int8_weights.h — GÉNÉRÉ par export_weights_c.py --maha-int8 (S2912).",
        " * NE PAS MODIFIER À LA MAIN (règle CLAUDE.md). mu + sigma_inv INT8 affine. */",
        "#include <stdint.h>",
        "",
        "#define MAHA_INT8_WEIGHTS_PROVIDED",
        f"#define MAHA_INT8_NATIVE_DIM {d}",
        "",
        _array1d_int_to_c("MAHA_INT8_MU_Q8", mu_q8.astype(np.int32), "uint8_t"),
        f"static const float   MAHA_INT8_MU_SCALE   = {_fmt_float(mu_scale)};",
        f"static const int32_t MAHA_INT8_MU_ZP      = {int(mu_zp)};",
        _array2d_int_to_c("MAHA_INT8_SIGMA_INV", sigma_q8.astype(np.int32), "uint8_t"),
        f"static const float   MAHA_INT8_SIGMA_SCALE = {_fmt_float(sigma_scale)};",
        f"static const int32_t MAHA_INT8_SIGMA_ZP    = {int(sigma_zp)};",
        f"static const float   MAHA_INT8_THRESHOLD   = {_fmt_float(threshold)};",
    ]
    header_path = out_path / "mahalanobis_int8_weights.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] mahalanobis_int8_weights.h écrit → {header_path}  (d={d})")
    return {
        "d": d,
        "mu_q8": mu_q8,
        "mu_scale": mu_scale,
        "mu_zp": mu_zp,
        "sigma_q8": sigma_q8,
        "sigma_scale": sigma_scale,
        "sigma_zp": sigma_zp,
        "threshold": threshold,
    }


def export_maha_int8_test_vectors_h(detector: "MahalanobisDetector", out_path: Path) -> None:
    """Génère tests/test_vectors_maha_int8.h : entrée + distance INT8 attendue (S2912).

    La distance attendue est calculée par MahalanobisDetectorInt8.score_int8 (chemin Python
    exact), de sorte que le test C (maha_int8_score) doit la reproduire à la tolérance flottante.
    """
    from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8  # noqa: PLC0415

    d = int(detector.n_features_)

    int8 = MahalanobisDetectorInt8({"quantization": "int8"})
    int8.mu_ = detector.mu_.astype(np.float32)
    int8.sigma_inv_ = detector.sigma_inv_.astype(np.float32)
    int8.threshold_ = detector.threshold_
    int8.n_features_ = d
    int8.calibrate_int8()

    rng = np.random.default_rng(29)
    test_input = (detector.mu_.astype(np.float32) + rng.standard_normal(d).astype(np.float32))
    dist = int8.score_int8(test_input)

    lines = [
        "#pragma once",
        "/* test_vectors_maha_int8.h — GÉNÉRÉ par export_weights_c.py --maha-int8-test-vectors (S2912).",
        " * Parité forward INT8 C (maha_int8_score) ↔ Python (score_int8). NE PAS MODIFIER À LA MAIN. */",
        "#include <stdint.h>",
        "",
        f"#define TV_MAHA_INT8_DIM {d}",
        _array1d_int_to_c("TV_MAHA_INT8_MU_Q8", int8.mu_q_.astype(np.int32), "uint8_t"),
        f"static const float   TV_MAHA_INT8_MU_SCALE   = {_fmt_float(int8._mu_scale)};",
        f"static const int32_t TV_MAHA_INT8_MU_ZP      = {int(int8._mu_zp)};",
        _array2d_int_to_c("TV_MAHA_INT8_SIGMA_INV", int8.sigma_inv_q_.astype(np.int32), "uint8_t"),
        f"static const float   TV_MAHA_INT8_SIGMA_SCALE = {_fmt_float(int8._sigma_scale)};",
        f"static const int32_t TV_MAHA_INT8_SIGMA_ZP    = {int(int8._sigma_zp)};",
        _array1d_to_c("TV_MAHA_INT8_INPUT", test_input),
        f"static const float   TV_MAHA_INT8_EXPECTED_DIST = {_fmt_float(dist)};",
    ]
    header_path = out_path / "test_vectors_maha_int8.h"
    header_path.write_text("\n".join(lines) + "\n")
    print(f"[export] test_vectors_maha_int8.h écrit → {header_path}  dist_int8={dist:.8f}")


# ── Résolution de checkpoint par condition (Sprint 35 / S3507) ─────────────

_AUTO = "__auto__"  # sentinelle : --mahal/--ewc-head sans valeur → résoudre via condition


def _resolve_ckpt(condition: str, model: str, dataset: str, kind: str) -> Path:
    """Localise le checkpoint board d'une cellule (condition, model, dataset).

    Convention (produite par run_feature_condition_board.py) :
    ``experiments/exp_S35_board_{condition}_{model}_{dataset}/checkpoints/{name}``.

    Parameters
    ----------
    kind : str
        ``"ewc"`` → ``ewc_head.pt`` ; ``"mahalanobis"`` → ``mahalanobis_task0.pkl``.
    """
    name = "ewc_head.pt" if kind == "ewc" else "mahalanobis_task0.pkl"
    ckpt = (
        Path("experiments")
        / f"exp_S35_board_{condition}_{model}_{dataset}"
        / "checkpoints"
        / name
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {ckpt}. Lancer d'abord l'entraînement board "
            f"(run_feature_condition_board.py) pour la cellule {condition}/{model}/{dataset}."
        )
    return ckpt


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mahal", nargs="?", default=None, const=_AUTO,
        help="Checkpoint MahalanobisDetector (.pkl). Sans valeur + "
             "--condition/--model/--dataset : résolution auto (S3507).",
    )
    p.add_argument(
        "--ewc", type=Path, default=None,
        help="Chemin vers le checkpoint EWCMlpClassifier (.pt) [optionnel]",
    )
    p.add_argument(
        "--ewc-head", nargs="?", default=None, const=_AUTO,
        help="Checkpoint EWCMlpMulticlass(k,2,[32,16]) → model_weights_ewc.h "
             "(tête board g_ewc_head, parité S3205). Sans valeur + "
             "--condition/--model/--dataset : résolution auto (S3507) [optionnel].",
    )
    p.add_argument(
        "--condition", choices=["5feat", "all", "best"], default=None,
        help="Condition Sprint 35 pour la résolution auto des checkpoints (S3507).",
    )
    p.add_argument(
        "--model", default=None,
        help="Modèle (ewc/mahalanobis/...) pour la résolution auto des checkpoints (S3507).",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Dataset pour la résolution auto des checkpoints (S3507).",
    )
    p.add_argument(
        "--meta", type=Path, default=None,
        help="Chemin vers meta_weights.json (MetaLearner.export_weights) → meta_weights.h [optionnel]",
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
        "--maha-q15", nargs="?", default=None, const=_AUTO,
        help="Checkpoint MahalanobisDetector (.pkl) → mahalanobis_q15_weights.h "
             "(g_maha_q15, sigma_inv int16 Q15, S3407). Sans valeur + "
             "--condition/--model/--dataset : résolution auto [optionnel].",
    )
    p.add_argument(
        "--dump-test-vectors", action="store_true",
        help="Génère firmware/stm32f4_blink/tests/test_vectors.h (validation bitwise C vs Python)",
    )
    p.add_argument(
        "--maha-q15-test-vectors", action="store_true",
        help="Génère tests/test_vectors_q15.h (parité forward Q15 C ↔ Python, S3409). "
             "Nécessite --maha-q15 (ou --mahal) pour fournir le détecteur.",
    )
    p.add_argument(
        "--maha-int8", nargs="?", default=None, const=_AUTO,
        help="Checkpoint MahalanobisDetector (.pkl) → mahalanobis_int8_weights.h "
             "(g_maha_int8, mu+sigma_inv INT8 affine, S2912). Sans valeur + "
             "--condition/--model/--dataset : résolution auto [optionnel].",
    )
    p.add_argument(
        "--maha-int8-test-vectors", action="store_true",
        help="Génère tests/test_vectors_maha_int8.h (parité forward INT8 C ↔ Python, S2912). "
             "Nécessite --maha-int8 (ou --mahal) pour fournir le détecteur.",
    )
    p.add_argument(
        "--drift-thresholds", type=Path, default=None,
        help="JSON {fault_threshold,drift_threshold,window_size,drift_ratio} "
             "(run_sprint38_pc) → inc/drift_thresholds.h (gate -DEWC_AUTO_UPDATE, S3803).",
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

    def _resolve(arg_val: str, kind: str) -> Path:
        """Traduit la valeur CLI (--AUTO ⇒ résolution par condition) en chemin."""
        if arg_val == _AUTO:
            if not (args.condition and args.model and args.dataset):
                raise SystemExit(
                    "Résolution auto demandée mais --condition/--model/--dataset manquant."
                )
            return _resolve_ckpt(args.condition, args.model, args.dataset, kind)
        return Path(arg_val)

    detector = None
    if args.mahal is not None:
        mahal_path = _resolve(args.mahal, "mahalanobis")
        with open(mahal_path, "rb") as f:
            detector = pickle.load(f)
        export_mahalanobis_to_c(detector, out_path, zscore_mean, zscore_std)
    else:
        print("[export] --mahal non fourni : model_weights.h inchangé")

    # Sprint 34 — Q15 : charge le détecteur Q15 (réutilise --mahal s'il est déjà chargé).
    maha_q15_detector = detector
    if args.maha_q15 is not None:
        q15_path = _resolve(args.maha_q15, "mahalanobis")
        with open(q15_path, "rb") as f:
            maha_q15_detector = pickle.load(f)
        export_maha_q15_to_c(maha_q15_detector, out_path)
    else:
        print("[export] --maha-q15 non fourni : mahalanobis_q15_weights.h inchangé")

    # Sprint 29 — INT8 : charge le détecteur Maha INT8 (réutilise --mahal s'il est déjà chargé).
    maha_int8_detector = detector
    if args.maha_int8 is not None:
        int8_path = _resolve(args.maha_int8, "mahalanobis")
        with open(int8_path, "rb") as f:
            maha_int8_detector = pickle.load(f)
        export_maha_int8_to_c(maha_int8_detector, out_path)
    else:
        print("[export] --maha-int8 non fourni : mahalanobis_int8_weights.h inchangé")

    if args.ewc is not None:
        export_ewc_head_to_c(args.ewc, out_path, ewc_lambda=args.ewc_lambda)
    else:
        print("[export] --ewc non fourni : ewc_weights.h non généré")

    if args.ewc_head is not None:
        export_ewc_head_board_to_c(_resolve(args.ewc_head, "ewc"), out_path)
    else:
        print("[export] --ewc-head non fourni : model_weights_ewc.h inchangé")

    meta_weights: dict | None = None
    if args.meta is not None:
        meta_weights = export_meta_to_c(args.meta, out_path)
    else:
        print("[export] --meta non fourni : meta_weights.h inchangé")

    if args.drift_thresholds is not None:
        export_drift_thresholds_to_c(args.drift_thresholds, out_path)
    else:
        print("[export] --drift-thresholds non fourni : drift_thresholds.h inchangé")

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
        if meta_weights is not None:
            export_meta_test_vectors_h(meta_weights, test_vectors_dir)

    if args.maha_q15_test_vectors:
        test_vectors_dir = Path("firmware/stm32f4_blink/tests")
        test_vectors_dir.mkdir(parents=True, exist_ok=True)
        if maha_q15_detector is None:
            # Cas par défaut synthétique reproductible (pas de checkpoint requis).
            from src.models.unsupervised.mahalanobis_detector import (  # noqa: PLC0415
                MahalanobisDetector,
            )

            rng = np.random.default_rng(34)
            d = 5
            X = (rng.standard_normal((400, d)) @ np.diag([1.0, 50.0, 1.0, 2000.0, 5.0])).astype(
                np.float32
            )
            maha_q15_detector = MahalanobisDetector({"threshold_percentile": 95})
            maha_q15_detector.fit_task(X, task_id=0)
        export_maha_q15_test_vectors_h(maha_q15_detector, test_vectors_dir)

    if args.maha_int8_test_vectors:
        test_vectors_dir = Path("firmware/stm32f4_blink/tests")
        test_vectors_dir.mkdir(parents=True, exist_ok=True)
        if maha_int8_detector is None:
            # Cas par défaut synthétique reproductible (pas de checkpoint requis).
            from src.models.unsupervised.mahalanobis_detector import (  # noqa: PLC0415
                MahalanobisDetector,
            )

            rng = np.random.default_rng(29)
            d = 5
            X = (rng.standard_normal((400, d)) @ np.diag([1.0, 50.0, 1.0, 2000.0, 5.0])).astype(
                np.float32
            )
            maha_int8_detector = MahalanobisDetector({"threshold_percentile": 95})
            maha_int8_detector.fit_task(X, task_id=0)
        export_maha_int8_test_vectors_h(maha_int8_detector, test_vectors_dir)


if __name__ == "__main__":
    main()
