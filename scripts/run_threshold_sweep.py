"""
run_threshold_sweep.py — Sprint 32 / S3203 + S3204.

Orchestrateur du balayage du seuil RUL→faulty : boucle
``{modèle} × {dataset} × {seuil}`` en réutilisant les scripts d'entraînement
existants (jamais réimplémenter la boucle CL). Pour chaque combinaison :

1. Fusionne la config de base (modèle × dataset) avec le seuil balayé
   (``configs/sweep/_runs/exp_S32_*.yaml``), seul le champ seuil diffère.
2. Lance ``train_{model}.py --config {merged} --profile_memory`` en subprocess.
3. Consigne ``positive_ratio`` (part de faulty=1, model-indépendant) — métrique
   clé de S3204 expliquant la dérive des métriques de perf.
4. Consolide perf + HW de chaque run dans
   ``experiments/exp_S32_sweep_summary.json``.

Les échecs sont isolés : un run qui échoue est loggé et n'interrompt pas le
balayage (cellule N/A honnête dans le résumé).

Usage
-----
    # balayage complet (4 modèles × 3 datasets × 5 seuils = 60 runs)
    python scripts/run_threshold_sweep.py

    # smoke-test d'une combinaison
    python scripts/run_threshold_sweep.py --models ewc --datasets cmapss --thresholds 30
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_threshold_sweep_configs import SWEEPS  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

# --- Matrice de balayage --------------------------------------------------
MODELS = ["mahalanobis", "hdc", "ewc", "tinyol"]
DATASETS = ["cmapss", "pronostia", "battery"]

# Config de base par (modèle, dataset). Pour CMAPSS, cmapss_config.yaml est un
# config multi-modèle complet (sections model/ewc/hdc/mahalanobis) ; TinyOL a
# son propre config. Pour Pronostia, un config par modèle existe. Pour Battery,
# EWC natif + 3 configs créés par analogie (S3203).
MODEL_DATASET_BASE: dict[tuple[str, str], str] = {
    ("ewc", "cmapss"): "configs/cmapss_config.yaml",
    ("hdc", "cmapss"): "configs/cmapss_config.yaml",
    ("mahalanobis", "cmapss"): "configs/cmapss_config.yaml",
    ("tinyol", "cmapss"): "configs/cmapss_tinyol_config.yaml",
    ("ewc", "pronostia"): "configs/ewc_pronostia_by_condition_config.yaml",
    ("hdc", "pronostia"): "configs/hdc_pronostia_by_condition_config.yaml",
    ("mahalanobis", "pronostia"): "configs/mahalanobis_pronostia_by_condition_config.yaml",
    ("tinyol", "pronostia"): "configs/tinyol_pronostia_by_condition_config.yaml",
    ("ewc", "battery"): "configs/battery_config.yaml",
    ("hdc", "battery"): "configs/hdc_battery_config.yaml",
    ("mahalanobis", "battery"): "configs/mahalanobis_battery_config.yaml",
    ("tinyol", "battery"): "configs/tinyol_battery_config.yaml",
}

RUNS_CONFIG_DIR = Path("configs/sweep/_runs")
EXPERIMENTS_DIR = Path("experiments")
SUMMARY_PATH = EXPERIMENTS_DIR / "exp_S32_sweep_summary.json"


def _inject_threshold(base_cfg: dict, dataset: str, threshold: int) -> dict:
    """Retourne une copie de base_cfg avec le seuil injecté dans data."""
    _, field, _ = SWEEPS[dataset]
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("data", {})
    cfg["data"][field] = threshold
    if dataset == "pronostia":
        cfg["data"]["label_mode"] = "rul_threshold"
    return cfg


def _positive_ratio(dataset: str, threshold: int) -> float | None:
    """Part de faulty=1 pour (dataset, seuil) — identique pour tous les modèles."""
    try:
        if dataset == "battery":
            from src.data.battery_dataset import load_raw_dataset

            df = load_raw_dataset(
                Path("data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv"),
                rul_failure_threshold=threshold,
            )
            return float(df["faulty"].mean())

        if dataset == "cmapss":
            from src.data.cmapss_loader import _load_raw

            data_dir = Path("data/raw/CMAPSS Jet Engine Simulated Data/")
            labels = []
            for subset in ["FD001", "FD002", "FD003", "FD004"]:
                df = _load_raw(data_dir, subset, faulty_threshold=threshold)
                labels.append(df["faulty"].to_numpy())
            return float(np.concatenate(labels).mean())

        if dataset == "pronostia":
            from src.data.pronostia_dataset import load_condition_features

            npy_dir = Path("data/raw/Pronostia dataset/binaries")
            labels = []
            for condition in (1, 2, 3):
                _, lbls = load_condition_features(
                    npy_dir,
                    condition,
                    label_mode="rul_threshold",
                    faulty_threshold=threshold,
                )
                labels.append(lbls)
            return float(np.concatenate(labels).mean())
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] positive_ratio({dataset}, {threshold}) indisponible : {exc}")
        return None
    return None


def _load_run_metrics(exp_dir: Path) -> dict:
    """Récupère les métriques perf+HW écrites par le train script."""
    for name in ("results/metrics_cl.json", "results.json", "results/metrics_single_task.json"):
        path = exp_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


def run_one(model: str, dataset: str, threshold: int) -> dict:
    """Lance un run et retourne sa ligne de résumé (status + métriques)."""
    exp_id = f"exp_S32_{model}_{dataset}_thr{threshold}"
    exp_dir = EXPERIMENTS_DIR / exp_id
    base_path = MODEL_DATASET_BASE.get((model, dataset))

    row: dict = {
        "exp_id": exp_id,
        "model": model,
        "dataset": dataset,
        "threshold": threshold,
        "positive_ratio": _positive_ratio(dataset, threshold),
        "status": "pending",
    }

    if base_path is None or not Path(base_path).exists():
        row["status"] = "skipped_no_base_config"
        print(f"  [skip] {exp_id} : base config absente ({base_path})")
        return row

    # Config fusionnée (base modèle×dataset ⊕ seuil) — un fichier complet par run
    # pour contourner le re-read YAML des loaders CMAPSS/Mahalanobis.
    merged = _inject_threshold(load_config(base_path), dataset, threshold)
    merged["exp_id"] = exp_id
    RUNS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = RUNS_CONFIG_DIR / f"{exp_id}.yaml"
    with open(merged_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    cmd = [
        sys.executable,
        f"scripts/train_{model}.py",
        "--config",
        str(merged_path),
        "--profile_memory",
        "--exp_id",
        exp_id,
        "--exp_dir",
        str(exp_dir),
    ]
    print(f"\n=== {exp_id} ===\n  $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            row["status"] = "failed"
            row["error"] = (proc.stderr or proc.stdout)[-1500:]
            print(f"  [FAIL] returncode={proc.returncode}\n{row['error']}")
            return row
    except Exception as exc:  # noqa: BLE001 — échec isolé, on continue le balayage
        row["status"] = "error"
        row["error"] = f"{exc}\n{traceback.format_exc()[-800:]}"
        print(f"  [ERROR] {exc}")
        return row

    metrics = _load_run_metrics(exp_dir)
    for key in (
        "acc_final",
        "avg_forgetting",
        "backward_transfer",
        "ram_peak_bytes",
        "inference_latency_ms",
        "n_params",
    ):
        if key in metrics:
            row[key] = metrics[key]
    row["status"] = "ok"

    # Persiste positive_ratio dans le run (S3204 : métrique consignée par seuil).
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "sweep_meta.json", "w") as f:
        json.dump(
            {"positive_ratio": row["positive_ratio"], "threshold": threshold, "dataset": dataset},
            f,
            indent=2,
        )
    print(f"  [OK] positive_ratio={row['positive_ratio']} | acc_final={row.get('acc_final')}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Balayage du seuil RUL→faulty (Sprint 32)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--thresholds", nargs="+", type=int, default=None,
                        help="Sous-ensemble de seuils (défaut : tous ceux du dataset)")
    args = parser.parse_args()

    new_rows: list[dict] = []
    for dataset in args.datasets:
        _, _, all_thresholds = SWEEPS[dataset]
        thresholds = args.thresholds or all_thresholds
        for model in args.models:
            for thr in thresholds:
                new_rows.append(run_one(model, dataset, thr))

    # Fusion par exp_id dans le résumé existant (préserve les runs hors sous-ensemble).
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if SUMMARY_PATH.exists():
        for r in json.load(open(SUMMARY_PATH)):
            merged[r["exp_id"]] = r
    for r in new_rows:
        merged[r["exp_id"]] = r
    summary = sorted(merged.values(), key=lambda r: r["exp_id"])
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    ok = sum(1 for r in summary if r["status"] == "ok")
    print(f"\n{'=' * 60}\nBalayage terminé : {ok}/{len(summary)} runs OK")
    print(f"Résumé consolidé : {SUMMARY_PATH}")
    for r in summary:
        if r["status"] != "ok":
            print(f"  - {r['exp_id']} : {r['status']}")


if __name__ == "__main__":
    main()
