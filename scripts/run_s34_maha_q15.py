"""
scripts/run_s34_maha_q15.py — Expérience PC Q15 vs INT8 vs FP32 Mahalanobis (Sprint 34, S3406).

Pour chaque dataset, entraîne le détecteur de Mahalanobis et évalue trois variantes :
    - fp32 : MahalanobisDetector (référence)
    - int8 : MahalanobisDetectorInt8(quant="int8") — sigma_inv_ INT8 affine (bug grande dynamique)
    - q15  : MahalanobisDetectorInt8(quant="q15")  — sigma_inv_ int16 Q15 (fallback S3405)

Écrit experiments/exp_S34_maha_q15/{dataset}_{fp32,int8,q15}.json + summary.json.

Critère (docs/triple_gap.md, S3406) :
    - CWRU/Pronostia : ΔAUROC(q15 vs fp32) < 0.02  (vs −0.236/−0.238 en INT8 — Sprint 28)
    - Monitoring/CMAPSS/Paderborn : non-régression (q15 ne dégrade pas fp32/int8)

100 % PC Python — pas de board. Réutilise MahalanobisAdapter de benchmark_int8_fp32.py
(load_tasks, _maha_cfg) + les helpers AUROC/loader, pour rester strictement aligné sur S28.

Usage :
    python scripts/run_s34_maha_q15.py [--datasets cwru pronostia ...] [--n_samples 500]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector  # noqa: E402
from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8  # noqa: E402
from src.utils.config_loader import load_config_extends  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

DATASETS = ["cwru", "pronostia", "monitoring", "cmapss", "paderborn"]
TARGET_DATASETS = {"cwru", "pronostia"}  # cibles du bug INT8 Sprint 28
OUT_DIR = _ROOT / "experiments" / "exp_S34_maha_q15"
DELTA_THRESHOLD = 0.02


def _load_benchmark_module():
    """Charge scripts/benchmark_int8_fp32.py pour réutiliser adapter + helpers (S28)."""
    spec = importlib.util.spec_from_file_location(
        "_bench_int8", _ROOT / "scripts" / "benchmark_int8_fp32.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BENCH = _load_benchmark_module()


def _config_for(dataset: str) -> str:
    return str(_ROOT / "configs" / f"mahalanobis_q15_{dataset}.yaml")


def _scores_over_tasks(model, tasks: list[dict], variant: str) -> tuple[float, np.ndarray]:
    """(AUROC moyenne par tâche, scores concaténés) pour une variante (fp32/int8/q15).

    Les scores concaténés servent à mesurer la *fidélité numérique* d'une variante quantifiée
    vis-à-vis du FP32 (mean abs error, corrélation) — métrique robuste là où l'AUROC binarisée
    par-tâche est dégénérée (ex. CWRU, FP32 déjà < 0.5).
    """
    per_task = []
    all_scores: list[float] = []
    for task in tasks:
        loader = task.get("test_loader") or task["val_loader"]
        scores, labels = [], []
        for x, y in loader:
            xa = x.numpy().astype(np.float32)
            if variant == "fp32":
                s = model.anomaly_score(xa)
            elif variant == "int8":
                s = model.anomaly_score_int8(xa)
            else:  # q15
                s = model.anomaly_score_q15(xa)
            sl = np.asarray(s).ravel().tolist()
            scores.extend(sl)
            all_scores.extend(sl)
            labels.extend(y.numpy().ravel().tolist())
        per_task.append((labels, scores))
    return _BENCH._mean_auroc_over_tasks(per_task), np.asarray(all_scores, dtype=np.float64)


def _fidelity(scores_q: np.ndarray, scores_fp32: np.ndarray) -> dict:
    """Fidélité numérique d'une variante quantifiée vs FP32 (S3406)."""
    err = np.abs(scores_q - scores_fp32)
    if scores_q.size > 1 and np.std(scores_q) > 0 and np.std(scores_fp32) > 0:
        corr = float(np.corrcoef(scores_q, scores_fp32)[0, 1])
    else:
        corr = float("nan")
    return {
        "mean_abs_error": round(float(np.mean(err)), 6),
        "max_abs_error": round(float(np.max(err)), 6),
        "corr_with_fp32": _r(corr),
    }


def _latency_ms(fn, d: int) -> float:
    sample = np.zeros((1, d), dtype=np.float32)
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        fn(sample)
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lat))


def _fit_tasks(model, tasks: list[dict]) -> None:
    for i, task in enumerate(tasks):
        X = _BENCH._loader_to_numpy(task["train_loader"])
        model.fit_task(X, task_id=i)


def run_dataset(dataset: str, n_samples: int | None) -> dict:
    """Évalue fp32/int8/q15 sur un dataset, écrit les 3 JSON, renvoie le résumé."""
    adapter = _BENCH.MahalanobisAdapter()
    config_path = _config_for(dataset)
    cfg = load_config_extends(config_path)
    maha_cfg = adapter._maha_cfg(cfg)
    seed = cfg.get("training", {}).get("seed", 42)

    print(f"\n{'=' * 60}\n  Q15 Mahalanobis — dataset={dataset}\n{'=' * 60}")
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _BENCH._truncate_tasks(tasks, n_samples)

    results: dict[str, dict] = {}

    # --- FP32 (référence) ---
    set_seed(seed)
    m_fp32 = MahalanobisDetector(maha_cfg)
    _fit_tasks(m_fp32, tasks)
    d = int(m_fp32.n_features_)
    auroc_fp32, scores_fp32 = _scores_over_tasks(m_fp32, tasks, "fp32")
    ram_fp32 = (d + d * d) * 4
    lat_fp32 = _latency_ms(m_fp32.anomaly_score, d)
    results["fp32"] = _result(dataset, config_path, "fp32", auroc_fp32, ram_fp32, lat_fp32, None)

    # --- INT8 (bug grande dynamique) ---
    set_seed(seed)
    m_int8 = MahalanobisDetectorInt8({**maha_cfg, "quantization": "int8"})
    _fit_tasks(m_int8, tasks)
    m_int8.calibrate_int8()
    auroc_int8, scores_int8 = _scores_over_tasks(m_int8, tasks, "int8")
    ram_int8 = m_int8.get_memory_footprint_int8()["total_bytes"]
    lat_int8 = _latency_ms(m_int8.anomaly_score_int8, d)
    fid_int8 = _fidelity(scores_int8, scores_fp32)
    results["int8"] = _result(dataset, config_path, "int8", auroc_int8, ram_int8, lat_int8, fid_int8)

    # --- Q15 (fallback S3405) ---
    set_seed(seed)
    m_q15 = MahalanobisDetectorInt8({**maha_cfg, "quantization": "q15"})
    _fit_tasks(m_q15, tasks)
    m_q15.calibrate_q15()
    auroc_q15, scores_q15 = _scores_over_tasks(m_q15, tasks, "q15")
    ram_q15 = m_q15.get_memory_footprint_q15()["total_bytes"]
    lat_q15 = _latency_ms(m_q15.anomaly_score_q15, d)
    fid_q15 = _fidelity(scores_q15, scores_fp32)
    results["q15"] = _result(dataset, config_path, "q15", auroc_q15, ram_q15, lat_q15, fid_q15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant, res in results.items():
        path = OUT_DIR / f"{dataset}_{variant}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  → {path.name}  AUROC={res['auroc']}  RAM={res['ram_bytes']} B")

    delta_q15 = _safe_delta(auroc_q15, auroc_fp32)
    delta_int8 = _safe_delta(auroc_int8, auroc_fp32)
    is_target = dataset in TARGET_DATASETS
    # Fidélité de RANG (corrélation au FP32) : métrique de recouvrement primaire, car c'est le
    # rang des scores qui détermine le seuil et l'AUROC. Q15 doit corréler ≥ INT8 au FP32.
    # NB : sur les datasets à très grande dynamique (paderborn, sigma_inv ~6e5), l'erreur
    # ABSOLUE de score (MAE) peut être pire en Q15 qu'en INT8 — non parce que Q15 est moins
    # fidèle (il reconstruit sigma_inv 200× mieux) mais parce que mu reste INT8 : sa petite
    # erreur est amplifiée par les grandes valeurs de sigma_inv (que Q15 préserve, et qu'INT8
    # écrase vers 0, collapsant les distances). La corrélation reste donc le bon critère.
    c_q15 = fid_q15["corr_with_fp32"]
    c_int8 = fid_int8["corr_with_fp32"]
    fidelity_improved = (
        bool(c_q15 + 1e-9 >= c_int8) if (c_q15 is not None and c_int8 is not None) else None
    )
    summary = {
        "dataset": dataset,
        "auroc_fp32": _r(auroc_fp32),
        "auroc_int8": _r(auroc_int8),
        "auroc_q15": _r(auroc_q15),
        "delta_int8_vs_fp32": _r(delta_int8),
        "delta_q15_vs_fp32": _r(delta_q15),
        "ram_bytes": {"fp32": ram_fp32, "int8": ram_int8, "q15": ram_q15},
        "score_fidelity": {"int8": fid_int8, "q15": fid_q15},
        "is_target_dataset": is_target,
        # Recouvrement AUROC (vrai sur cibles à AUROC non-dégénérée, ex. Pronostia).
        "q15_recovers_auroc": (
            bool(abs(delta_q15) < DELTA_THRESHOLD) if delta_q15 == delta_q15 else None
        ),
        # Métrique de recouvrement robuste (corrélation de rang) — vaut aussi sur AUROC
        # dégénérée (CWRU) ou indéfinie (Paderborn mono-classe).
        "q15_rank_fidelity_better_than_int8": fidelity_improved,
        "q15_no_regression_vs_int8_auroc": (
            bool(auroc_q15 + 1e-9 >= auroc_int8) if auroc_q15 == auroc_q15 else None
        ),
    }
    print(
        f"  AUROC fp32={summary['auroc_fp32']} int8={summary['auroc_int8']} q15={summary['auroc_q15']}"
        f"  |  Δq15={summary['delta_q15_vs_fp32']} Δint8={summary['delta_int8_vs_fp32']}"
        f"  |  corr-fp32 int8={c_int8} q15={c_q15}"
        f"  |  MAE int8={fid_int8['mean_abs_error']} q15={fid_q15['mean_abs_error']}"
        f"  {'[CIBLE]' if is_target else ''}"
    )
    return summary


def _result(dataset, config_path, variant, auroc, ram, lat, fidelity) -> dict:
    res = {
        "model": "mahalanobis",
        "dataset": dataset,
        "variant": variant,
        "config_path": str(config_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metric_name": "auroc",
        "auroc": _r(auroc),
        "ram_bytes": int(ram),
        "latency_ms": round(float(lat), 6),
    }
    if fidelity is not None:
        res["score_fidelity_vs_fp32"] = fidelity
    return res


def _safe_delta(a: float, b: float) -> float:
    if a != a or b != b:  # NaN
        return float("nan")
    return a - b


def _r(v: float):
    return None if v != v else round(float(v), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expérience PC Q15 Mahalanobis (S3406)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    summaries = [run_dataset(ds, args.n_samples) for ds in args.datasets]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sprint": "S34",
                "task": "S3406",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "delta_threshold": DELTA_THRESHOLD,
                "datasets": summaries,
            },
            f,
            indent=2,
        )
    print(f"\n✅ summary → {summary_path}")


if __name__ == "__main__":
    main()
