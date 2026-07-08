#!/usr/bin/env python3
"""run_s39_quant_sweep.py — Campagne PC trade-off latence/RAM/accuracy (S3906).

Cœur de la réponse à « existe-t-il des quantifications intermédiaires qui équilibrent
latence, RAM et accuracy ? ». On balaie **4 modèles × 5 datasets × schémas** et on
reporte, par cellule : la **métrique** (F1 / AUROC), la **RAM analytique** (octets de
poids purs), un **proxy de BOPs** et un **proxy de latence** analytique. Tout tourne au
PC (l'émulateur reproduit le chemin C bit-à-bit) ; la latence FPU réelle — où l'INT8 est
*plus lent* sur Cortex-M4 — exige la board et est renvoyée à S3915 (marqueur ``lat_proxy``).

Schémas balayés par modèle (cf. S3900) :
    - EWC          : fp32, int8_legacy, int8_perchannel, q15, mixed  (émulateur S3902)
    - Mahalanobis  : fp32, int8, q15                                 (mahalanobis_int8.py)
    - HDC          : fp32, int8 (exact, INT8==FP32 → témoin)
    - TinyOL       : fp32, int8 (QAT tinyol_int8.py)

Table de mapping **unique** ``SCHEME_TO_QUANTCONFIG`` (aussi la source de vérité des
configs ``configs/quant_intermediate/*.yaml``, S3905) : aucune duplication.

Aucune valeur écrite à la main : chaque cellule sort d'une exécution ; les cellules non
mesurables sortent ``metric: null`` + ``na_reason`` (honnêteté).

Usage :
    python scripts/run_s39_quant_sweep.py                       # 4 modèles × 5 datasets
    python scripts/run_s39_quant_sweep.py --model ewc
    python scripts/run_s39_quant_sweep.py --dataset pronostia --max-samples 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import compute_cost  # noqa: E402
from src.evaluation.feature_conditions import (  # noqa: E402
    load_condition_arrays,
    load_native_task_arrays,
    resolve_feature_indices,
    train_and_evaluate,
)
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    QuantConfig,
    forward_fp32,
    forward_quant,
    predict,
)
from src.utils.reproducibility import set_seed  # noqa: E402

# Réutilise l'entraîneur de tête EWC board du script d'ablation (source unique).
from scripts.run_s39_int8_ablation import train_ewc_head  # noqa: E402

DATASETS: list[str] = ["cmapss", "cwru", "monitoring", "pronostia", "paderborn"]
MODELS: list[str] = ["ewc", "mahalanobis", "hdc", "tinyol"]
CONDITION = "5feat"
OUT_DIR = Path("experiments/exp_S39_quant_sweep")

# Table de mapping unique schéma → QuantConfig (émulateur). ``None`` = FP32 de référence.
SCHEME_TO_QUANTCONFIG: dict[str, QuantConfig | None] = {
    "fp32": None,
    "int8_legacy": QuantConfig.legacy_c(),
    "int8_perchannel": QuantConfig.per_channel_int8(),
    "q15": QuantConfig.q15(),
    "mixed": QuantConfig.mixed_int8w_q15act(),
}

# Largeur d'opérande (bits) par schéma → proxy BOPs/latence analytique (compute_cost).
SCHEME_BITS: dict[str, int] = {
    "fp32": 32,
    "int8_legacy": 8,
    "int8_perchannel": 8,
    "q15": 16,
    "mixed": 8,       # poids int8 (dominant RAM) ; activations Q15
    "int8": 8,
}

# Octets par poids stocké selon le schéma (RAM analytique des poids purs, scales exclus).
SCHEME_WEIGHT_BYTES: dict[str, int] = {
    "fp32": 4,
    "int8_legacy": 1,
    "int8_perchannel": 1,
    "q15": 2,
    "mixed": 1,       # poids int8
    "int8": 1,
}

# Schémas balayés par famille de modèle.
EWC_SCHEMES = ["fp32", "int8_legacy", "int8_perchannel", "q15", "mixed"]
MAHA_SCHEMES = ["fp32", "int8", "q15"]
HDC_SCHEMES = ["fp32", "int8"]
TINYOL_SCHEMES = ["fp32", "int8"]


# ── Proxies de coût (compute_cost) ──────────────────────────────────────────

def _proxies(macs: int, scheme: str) -> dict:
    """Retourne bops_proxy (rel. FP32) + lat_proxy_rel (proxy analytique, PAS FPU réelle).

    ``bops_proxy`` = BOPs(scheme) / BOPs(fp32) = (bits/32)². ``lat_proxy_rel`` reprend ce
    proxy analytique : la latence FPU réelle (INT8 plus lent, déquant→FP32) ne se mesure
    que sur board (S3915) → champ ``lat_proxy: true`` explicite.
    """
    bits = SCHEME_BITS[scheme]
    bops = compute_cost.compute_bops(macs, bits)
    bops_fp32 = compute_cost.compute_bops(macs, 32)
    rel = round(bops / bops_fp32, 4) if bops_fp32 else None
    return {"bops_proxy": rel, "lat_proxy_rel": rel, "lat_proxy": True}


def _subsample(X: np.ndarray, y: np.ndarray, n: int | None, seed: int = 42):
    if n is None or len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC binaire (score élevé = anormal). None si une seule classe présente."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


# ── EWC : 5 schémas via l'émulateur bit-exact ───────────────────────────────

def sweep_ewc(dataset: str, seed: int, max_samples: int | None) -> dict:
    X, y, idx, names = load_condition_arrays(dataset, CONDITION, "ewc", seed=seed)
    X, y = _subsample(X, y, max_samples, seed)
    model = train_ewc_head(X, y, seed=seed)

    import torch

    with torch.no_grad():
        state = {k: v.cpu() for k, v in model.state_dict().items()}
    w = EWCHeadWeights.from_state_dict(state)

    k = int(X.shape[1])
    params = compute_cost.params_ewc_mlp(k, [32, 16], 2)
    macs = compute_cost.macs_ewc_mlp(k, [32, 16], 2)

    schemes: dict[str, dict] = {}
    for scheme in EWC_SCHEMES:
        cfg = SCHEME_TO_QUANTCONFIG[scheme]
        logits = forward_fp32(w, X) if cfg is None else forward_quant(w, X, cfg)
        f1 = float(compute_fault_f1(y, predict(logits))["f1_faulty"])
        cell = {
            "metric": round(f1, 4),
            "ram_weights_bytes": params * SCHEME_WEIGHT_BYTES[scheme],
            **_proxies(macs, scheme),
        }
        schemes[scheme] = cell
    return {"metric_name": "f1_faulty", "n_features": k, "n_params": int(params),
            "feature_names": list(names), "schemes": schemes}


# ── Mahalanobis : fp32 / int8 / q15 ─────────────────────────────────────────

def sweep_mahalanobis(dataset: str, seed: int, max_samples: int | None) -> dict:
    from src.models.unsupervised import MahalanobisDetector
    from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8

    X, y, idx, names = load_condition_arrays(dataset, CONDITION, "mahalanobis", seed=seed)
    X, y = _subsample(X, y, max_samples, seed)
    d = int(X.shape[1])

    base = MahalanobisDetector({"threshold_percentile": 95})
    base.fit_task(X.astype(np.float32), task_id=0)

    def _int8_like(quant: str) -> MahalanobisDetectorInt8:
        det = MahalanobisDetectorInt8({"quantization": quant})
        det.mu_ = base.mu_.astype(np.float32)
        det.sigma_inv_ = base.sigma_inv_.astype(np.float32)
        det.threshold_ = base.threshold_
        det.n_features_ = d
        det.calibrate()
        return det

    macs = compute_cost.macs_mahalanobis(d)
    params = compute_cost.params_mahalanobis(d)
    schemes: dict[str, dict] = {}

    # fp32
    auroc = _auroc(y, base.anomaly_score(X))
    schemes["fp32"] = {
        "metric": None if auroc is None else round(auroc, 4),
        "ram_weights_bytes": params * 4,
        **_proxies(macs, "fp32"),
        **({"na_reason": "tâche mono-classe (AUROC indéfinie)"} if auroc is None else {}),
    }
    # int8 / q15
    for scheme, ram_key in (("int8", "get_memory_footprint_int8"),
                            ("q15", "get_memory_footprint_q15")):
        det = _int8_like(scheme)
        if scheme == "int8":
            scores = det.anomaly_score_int8(X)
        else:
            scores = np.array([det.score_q15(x) for x in X])
        auroc = _auroc(y, scores)
        ram = getattr(det, ram_key)()["total_bytes"]
        schemes[scheme] = {
            "metric": None if auroc is None else round(auroc, 4),
            "ram_weights_bytes": int(ram),
            **_proxies(macs, scheme),
            **({"na_reason": "tâche mono-classe (AUROC indéfinie)"} if auroc is None else {}),
        }
    return {"metric_name": "auroc", "n_features": d, "n_params": int(params),
            "feature_names": list(names), "schemes": schemes}


# ── HDC : fp32 / int8 (témoin, INT8 == FP32) ────────────────────────────────

def sweep_hdc(dataset: str, seed: int, max_samples: int | None) -> dict:
    from src.utils.config_loader import load_config

    idx, _ = resolve_feature_indices(CONDITION, "hdc", dataset)
    tasks = load_native_task_arrays(dataset, seed=seed)
    if max_samples:  # tractabilité : plafonne train/val par tâche (HDC batch_size=1)
        cap = max(1, max_samples // max(1, len(tasks)))
        tasks = [{**t, "X_train": t["X_train"][:cap], "y_train": t["y_train"][:cap],
                  "X_val": t["X_val"][:cap], "y_val": t["y_val"][:cap]} for t in tasks]
    res = train_and_evaluate("hdc", tasks, idx, seed=seed)
    f1 = res["f1_faulty"]

    d = len(idx)
    D = int(load_config("configs/hdc_config.yaml")["hdc"]["D"])
    macs = compute_cost.macs_hdc(d, D, 2)
    params = compute_cost.params_hdc(d, D, 2)  # hypervecteurs bipolaires (int8 natif)

    note = "HDC nativement bipolaire : INT8 == FP32 (témoin) ; seule la RAM diffère."
    schemes = {
        "fp32": {"metric": round(f1, 4), "ram_weights_bytes": params * 2,  # AM int16
                 **_proxies(macs, "fp32"), "note": note},
        "int8": {"metric": round(f1, 4), "ram_weights_bytes": params * 1,  # AM int8/bipolaire
                 **_proxies(macs, "int8"), "note": note},
    }
    return {"metric_name": "f1_faulty", "n_features": d, "n_params": int(params),
            "schemes": schemes}


# ── TinyOL : fp32 / int8 (QAT autoencoder) ──────────────────────────────────

def sweep_tinyol(dataset: str, seed: int, max_samples: int | None) -> dict:
    from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector
    from src.models.tinyol.tinyol_int8 import TinyOLAutoencoderInt8
    from src.utils.config_loader import load_config

    set_seed(seed)
    X, y, idx, names = load_condition_arrays(dataset, CONDITION, "tinyol", seed=seed)
    X, y = _subsample(X, y, max_samples, seed)
    k = int(X.shape[1])

    # Archi autoencodeur dérivée de k (3 couches enc/dec, cohérent feature_conditions).
    bottleneck = max(2, k // 2)
    h2 = max(bottleneck + 1, k)
    h1 = max(2 * k, h2 + 1)
    base = load_config("configs/tinyol_config.yaml")
    cfg = {
        "backbone": {"input_dim": k, "encoder_dims": [h1, h2, bottleneck],
                     "decoder_dims": [h2, h1, k], "checkpoint_path": None},
        "pretrain": {"optimizer": base["pretrain"].get("optimizer", "adam"),
                     "learning_rate": base["pretrain"].get("learning_rate", 0.001),
                     "epochs": base["pretrain"].get("epochs", 50),
                     "batch_size": base["pretrain"].get("batch_size", 64)},
        "anomaly_percentile": base.get("anomaly_percentile", 95),
        "anomaly_threshold": None,
    }
    det = TinyOLAnomalyDetector(cfg)
    x_norm = X[y == 0]
    if len(x_norm) == 0:
        x_norm = X
    det.update(x_norm, np.zeros(len(x_norm), dtype=np.int64))
    det.on_task_end(task_id=0, dataloader=None)

    # FP32 : erreur de reconstruction (score anomalie).
    scores_fp32 = det.anomaly_score(X)
    auroc_fp32 = _auroc(y, scores_fp32)

    # INT8 : wrap QAT + erreur de reconstruction INT8.
    int8 = TinyOLAutoencoderInt8(det.autoencoder)
    int8.calibrate_int8(x_norm)
    scores_int8 = np.array([int8.reconstruction_error_int8(x) for x in X])
    auroc_int8 = _auroc(y, scores_int8)

    params = compute_cost.params_tinyol_ae(k, [h1, h2, bottleneck], [h2, h1, k])
    macs = compute_cost.macs_tinyol_ae(k, [h1, h2, bottleneck], [h2, h1, k])
    ram_int8 = int8.get_memory_footprint_int8().get("total_bytes", params)

    def _cell(auroc, ram, scheme):
        return {"metric": None if auroc is None else round(auroc, 4),
                "ram_weights_bytes": int(ram), **_proxies(macs, scheme),
                **({"na_reason": "tâche mono-classe (AUROC indéfinie)"} if auroc is None else {})}

    schemes = {
        "fp32": _cell(auroc_fp32, params * 4, "fp32"),
        "int8": _cell(auroc_int8, ram_int8, "int8"),
    }
    return {"metric_name": "auroc", "n_features": k, "n_params": int(params),
            "feature_names": list(names), "schemes": schemes}


SWEEP_FN = {
    "ewc": sweep_ewc,
    "mahalanobis": sweep_mahalanobis,
    "hdc": sweep_hdc,
    "tinyol": sweep_tinyol,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Campagne trade-off quantif PC (S3906)")
    parser.add_argument("--model", choices=MODELS, default=None, help="Un seul modèle.")
    parser.add_argument("--dataset", choices=DATASETS, default=None, help="Un seul dataset.")
    parser.add_argument("--max-samples", type=int, default=6000,
                        help="Sous-échantillonnage train+eval (tractabilité). None = tout.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [args.model] if args.model else MODELS
    datasets = [args.dataset] if args.dataset else DATASETS
    max_samples = None if args.max_samples <= 0 else args.max_samples

    summary: dict[str, dict] = {}
    for model in models:
        summary.setdefault(model, {})
        for ds in datasets:
            print(f"[S3906] {model} × {ds} ({CONDITION}) …")
            try:
                result = SWEEP_FN[model](ds, args.seed, max_samples)
            except Exception as exc:  # skip honnête : chargement/entraînement KO
                print(f"  ⚠️  {model}×{ds} ignoré : {type(exc).__name__}: {exc}")
                result = {"metric_name": None, "schemes": {},
                          "na_reason": f"{type(exc).__name__}: {exc}"}
            result.update({"model": model, "dataset": ds, "condition": CONDITION,
                           "seed": args.seed})
            out = OUT_DIR / f"{model}_{ds}.json"
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            summary[model][ds] = result.get("schemes", {})
            cells = " | ".join(
                f"{s}={c.get('metric')}" for s, c in result.get("schemes", {}).items()
            )
            print(f"  {result.get('metric_name')}: {cells} → {out}")

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"[S3906] summary.json → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
