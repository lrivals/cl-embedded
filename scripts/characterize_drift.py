#!/usr/bin/env python3
"""
characterize_drift.py — Caractérisation & quantification offline du drift (Sprint 43, S4303).

Charge un ``DriftDataset`` (S4302) et calcule, **sans détecteur en ligne** (analyse offline
exhaustive), une description quantitative de la dérive : type confirmé, statistiques glissantes
(KS / PSI / Jensen-Shannon / MMD / Mahalanobis / résidu PCA), et **validation vs ground-truth**
(alignement pics ↔ ``drift_points``). Produit un JSON de référence, **aucun chiffre en dur**.

Réutilise ``MahalanobisDetector`` (pas de nouvelle distance) et n'implémente PAS de détecteur en
ligne (``SlidingWindowDriftDetector`` = baseline S44).

Usage :
    python scripts/characterize_drift.py --dataset synthetic
    python scripts/characterize_drift.py --dataset gas_sensor_drift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from src.data import DRIFT_CONFIGS, DRIFT_LOADERS
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

OUT_ROOT = Path("experiments/exp_S43_drift_char")

# Défauts de fenêtrage (surchargés par la section characterization: de la config).
WINDOW_SIZE_DEFAULT = 500
STRIDE_DEFAULT = 250
HIST_BINS_DEFAULT = 20
TOP_K_DEFAULT = 10
_EPS = 1e-8


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int) -> float:
    """Population Stability Index entre deux échantillons 1D (histogrammes alignés sur ref)."""
    lo, hi = np.min(ref), np.max(ref)
    if hi - lo < _EPS:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r = r / max(r.sum(), 1) + _EPS
    c = c / max(c.sum(), 1) + _EPS
    return float(np.sum((c - r) * np.log(c / r)))


def _js_divergence(ref: np.ndarray, cur: np.ndarray, bins: int) -> float:
    """Divergence de Jensen-Shannon (base 2) entre deux échantillons 1D."""
    lo = min(np.min(ref), np.min(cur))
    hi = max(np.max(ref), np.max(cur))
    if hi - lo < _EPS:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p = p / max(p.sum(), 1) + _EPS
    q = q / max(q.sum(), 1) + _EPS
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))  # noqa: E731
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _mmd_rbf(ref: np.ndarray, cur: np.ndarray, gamma: float | None, rng: np.random.Generator,
             max_n: int = 300) -> float:
    """MMD² non biaisé (noyau RBF) entre deux ensembles multivariés (sous-échantillonnés)."""
    def _sub(a: np.ndarray) -> np.ndarray:
        if len(a) > max_n:
            idx = rng.choice(len(a), max_n, replace=False)
            return a[idx]
        return a

    x, y = _sub(ref), _sub(cur)
    if gamma is None:
        # Heuristique de la médiane des distances.
        z = np.vstack([x, y])
        d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, _EPS)

    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def characterize(dataset: str, config_path: str) -> dict:
    """Calcule la caractérisation complète d'un dataset de drift → dict sérialisable."""
    set_seed(42)
    rng = np.random.default_rng(42)

    cfg = load_config(config_path)
    char_cfg = cfg.get("characterization", {})
    window = int(char_cfg.get("window_size", WINDOW_SIZE_DEFAULT))
    stride = int(char_cfg.get("stride", STRIDE_DEFAULT))
    bins = int(char_cfg.get("hist_bins", HIST_BINS_DEFAULT))
    gamma = char_cfg.get("mmd_gamma", None)
    top_k = int(char_cfg.get("top_k_features", TOP_K_DEFAULT))

    d = DRIFT_LOADERS[dataset](config_path)
    X, drift_points = d.X, d.drift_points
    n, dim = X.shape

    # Fenêtre de référence = segment initial (début du flux).
    ref = X[: min(window, n)]

    # Détecteur Mahalanobis calibré sur le segment 0 (réutilisé, pas réimplémenté).
    maha = MahalanobisDetector({"cl_strategy": "refit", "reg_covar": 1e-3})
    maha.fit_task(ref, task_id=0)

    # PCA sur le segment 0 → résidu de reconstruction.
    n_comp = max(1, min(dim - 1, dim // 2)) if dim > 1 else 1
    pca = PCA(n_components=n_comp, random_state=42).fit(ref)

    centers: list[int] = []
    ks_series: list[float] = []
    psi_series: list[float] = []
    js_series: list[float] = []
    mmd_series: list[float] = []
    mean_shift_series: list[float] = []
    var_shift_series: list[float] = []
    maha_series: list[float] = []
    pca_series: list[float] = []
    per_feature_ks_accum = np.zeros(dim)
    n_windows = 0

    ref_mean, ref_var = ref.mean(axis=0), ref.var(axis=0)

    for start in range(0, n - window + 1, stride):
        cur = X[start : start + window]
        center = start + window // 2
        centers.append(center)

        ks_vals = np.array([ks_2samp(ref[:, j], cur[:, j]).statistic for j in range(dim)])
        per_feature_ks_accum += ks_vals
        ks_series.append(float(ks_vals.mean()))
        psi_series.append(float(np.mean([_psi(ref[:, j], cur[:, j], bins) for j in range(dim)])))
        js_series.append(
            float(np.mean([_js_divergence(ref[:, j], cur[:, j], bins) for j in range(dim)]))
        )
        mmd_series.append(_mmd_rbf(ref, cur, gamma, rng))
        mean_shift_series.append(float(np.linalg.norm(cur.mean(axis=0) - ref_mean)))
        var_shift_series.append(float(np.linalg.norm(cur.var(axis=0) - ref_var)))
        maha_series.append(float(maha.anomaly_score(cur).mean()))
        recon = pca.inverse_transform(pca.transform(cur))
        pca_series.append(float(np.mean(np.sum((cur - recon) ** 2, axis=1))))
        n_windows += 1

    # Validation vs ground-truth par détection de CHANGE-POINTS (robuste aux concepts
    # récurrents, où une statistique vs référence revient à sa valeur initiale) : on prend le
    # gradient d'un composite z-normalisé (MMD + Mahalanobis + décalage de moyenne), dont les
    # pics localisent les transitions. On retient les ``len(drift_points)`` plus forts.
    alignment_score = None
    peak_centers: list[int] = []
    if drift_points and len(centers) >= 3:
        centers_arr = np.array(centers)

        def _z(a: list[float]) -> np.ndarray:
            arr = np.array(a)
            s = arr.std()
            return (arr - arr.mean()) / s if s > _EPS else np.zeros_like(arr)

        composite = _z(mmd_series) + _z(maha_series) + _z(mean_shift_series)
        grad = np.abs(np.diff(composite))  # magnitude de changement entre fenêtres adjacentes
        # Centres de transition = milieu de chaque paire de fenêtres adjacentes.
        edge_centers = (centers_arr[:-1] + centers_arr[1:]) // 2

        # Maxima locaux du gradient, triés par intensité, top-N = nb de drift_points.
        local_max = [
            i for i in range(1, len(grad) - 1) if grad[i] >= grad[i - 1] and grad[i] >= grad[i + 1]
        ]
        local_max.sort(key=lambda i: grad[i], reverse=True)
        chosen = sorted(local_max[: len(drift_points)])
        peak_centers = [int(edge_centers[i]) for i in chosen]

        if peak_centers:
            peaks = np.array(peak_centers)
            dists = [int(np.min(np.abs(peaks - dp))) for dp in drift_points]
            alignment_score = float(np.median(dists))

    # Top-k features les plus dérivées (KS moyen cumulé).
    mean_ks_per_feat = per_feature_ks_accum / max(n_windows, 1)
    top_idx = np.argsort(mean_ks_per_feat)[::-1][:top_k]
    features_most_drifted = [
        {"feature": d.feature_names[j], "mean_ks": float(mean_ks_per_feat[j])} for j in top_idx
    ]

    return {
        "dataset": dataset,
        "drift_type_confirmed": d.drift_type,
        "n_features": dim,
        "n_samples": n,
        "drift_points": drift_points,
        "window_size": window,
        "stride": stride,
        "window_centers": centers,
        "series": {
            "ks": ks_series,
            "psi": psi_series,
            "js": js_series,
            "mmd": mmd_series,
            "mean_shift": mean_shift_series,
            "var_shift": var_shift_series,
            "mahalanobis": maha_series,
            "pca_residual": pca_series,
        },
        "peak_centers": peak_centers,
        "alignment_score": alignment_score,
        "features_most_drifted": features_most_drifted,
        "metadata": d.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Caractérisation offline du drift (Sprint 43).")
    parser.add_argument("--dataset", choices=list(DRIFT_LOADERS.keys()), required=True)
    parser.add_argument("--config", default=None, help="Config override (défaut = DRIFT_CONFIGS).")
    args = parser.parse_args()

    config_path = args.config or DRIFT_CONFIGS[args.dataset]
    result = characterize(args.dataset, config_path)

    out_dir = OUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "characterization.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    align = result["alignment_score"]
    print(f"✅ {args.dataset}: n={result['n_samples']} d={result['n_features']} "
          f"drift_points={result['drift_points']} alignment_score={align}")
    print(f"   → {out_path}")


if __name__ == "__main__":
    main()
