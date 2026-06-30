"""
board_experiment_recorder.py — Capture résultats board → experiments/ unifié Phase 1.

Orchestre :
  1. Streaming via sensor_stream.py (multi-tâches)
  2. Collecte métriques CL (accuracy par tâche, forgetting, BWT)
  3. Profiling (latence, RAM, throughput)
  4. Sauvegarde experiments/exp_S19_XX/ avec le même format que Phase 1 Python

Format results.json identique à evaluate_all.py (Phase 1) :
  acc_final, avg_forgetting, backward_transfer,
  ram_peak_bytes, inference_latency_ms, n_params

Usage :
    # Dry-run (sans board) — ancienne API
    python scripts/board_experiment_recorder.py --model mahalanobis \\
        --dataset cwru --dry-run --output experiments/exp_S19_01

    # Via config YAML avec exp-id (S2005)
    python scripts/board_experiment_recorder.py \\
        --config configs/board_ewc.yaml --exp-id ewc \\
        --dry-run --update --output experiments/exp_S19_02

    # Baseline lambda=0
    python scripts/board_experiment_recorder.py \\
        --config configs/board_ewc.yaml --override lambda_ewc=0.0 \\
        --exp-id baseline --dry-run --update --output experiments/exp_S19_02
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


# Nombre de paramètres par modèle (calculé offline depuis model.parameters())
_N_PARAMS = {
    "mahalanobis": 30,     # mean(5) + precision(25) = 30 floats
    "ewc":         1538,   # (5×32+32) + (32×16+16) + (16×2+2) = 802 poids, ×2 avec Fisher/star ≈ 1538
    "tinyol":      881,    # encoder: (5×32+32) + (32×16+16) = 720, decoder: (16×32+32) + (32×5+5) = 677 → 881 total encoder-only
    "hdc":         7000,   # AM: HDC_N_CLASSES×HDC_DIM=2×1000 + proj: HDC_N_FEATURES×HDC_DIM=5×1000
    "ewc-int8":    1538,   # même architecture EWC, stockage INT8 (n_params identique pour l'eval)
    "dual":        1538 + 4680,  # EWCHeadReg(~1538) + EWCHeadMC(~4680) — Sprint 27
    # Sprint 30 — paires Mahalanobis (30) + supervisé
    "pair-maha-ewc":    30 + 1538,
    "pair-maha-hdc":    30 + 7000,
    "pair-maha-tinyol": 30 + 881,
}

# RAM théorique par modèle
_EWC_RAM_BYTES     = 9728   # 3 Ko poids + 3 Ko Fisher + 3 Ko star_w + 200 B activations
_HDC_RAM_BYTES     = 28364  # 27.7 Ko : AM(8 Ko) + proj(20 Ko) + buffer retrain(300 B)
_EWC_INT8_RAM_BYTES = 4800  # 3.6 Ko poids INT8 + 1.2 Ko biais FP32

# Paramètres de simulation dry-run par (modèle, dataset) — valeurs calibrées sur specs sprint 23
# base_acc  : accuracy diagonale simulée
# step_drop : chute par tâche supplémentaire (forgetting)
# lat_lo/hi : bornes de la distribution uniforme latence (µs)
# ram_bytes : RAM statique du modèle
_GENERIC_DRY_RUN_PARAMS: dict[str, dict] = {
    "mahalanobis/cmapss":    {"base_acc": 0.67, "step_drop": 0.12, "lat_lo":   40, "lat_hi":   80, "ram_bytes":  1200},
    "tinyol/cmapss":         {"base_acc": 0.74, "step_drop": 0.08, "lat_lo": 2000, "lat_hi": 4500, "ram_bytes":  7040},
    "hdc/cmapss":            {"base_acc": 0.70, "step_drop": 0.07, "lat_lo":  800, "lat_hi": 1600, "ram_bytes": 28364},
    "ewc-int8/cmapss":       {"base_acc": 0.80, "step_drop": 0.03, "lat_lo":  250, "lat_hi":  700, "ram_bytes":  4800},
    "mahalanobis/paderborn": {"base_acc": 0.71, "step_drop": 0.10, "lat_lo":   35, "lat_hi":   75, "ram_bytes":  1200},
    "ewc/paderborn":         {"base_acc": 0.85, "step_drop": 0.025,"lat_lo":  300, "lat_hi":  750, "ram_bytes":  9728},
    # Fallback pour les datasets existants (cwru/monitoring/pronostia)
    "mahalanobis/cwru":      {"base_acc": 0.75, "step_drop": 0.10, "lat_lo":   40, "lat_hi":   80, "ram_bytes":  1200},
    "mahalanobis/monitoring":{"base_acc": 0.78, "step_drop": 0.08, "lat_lo":   40, "lat_hi":   80, "ram_bytes":  1200},
    "mahalanobis/pronostia": {"base_acc": 0.72, "step_drop": 0.11, "lat_lo":   40, "lat_hi":   80, "ram_bytes":  1200},
    "tinyol/cwru":           {"base_acc": 0.77, "step_drop": 0.07, "lat_lo": 2000, "lat_hi": 4500, "ram_bytes":  7040},
    "tinyol/monitoring":     {"base_acc": 0.80, "step_drop": 0.06, "lat_lo": 2000, "lat_hi": 4500, "ram_bytes":  7040},
    "tinyol/pronostia":      {"base_acc": 0.74, "step_drop": 0.09, "lat_lo": 2000, "lat_hi": 4500, "ram_bytes":  7040},
    "hdc/cwru":              {"base_acc": 0.73, "step_drop": 0.07, "lat_lo":  800, "lat_hi": 1600, "ram_bytes": 28364},
    "hdc/monitoring":        {"base_acc": 0.76, "step_drop": 0.06, "lat_lo":  800, "lat_hi": 1600, "ram_bytes": 28364},
    "hdc/pronostia":         {"base_acc": 0.70, "step_drop": 0.08, "lat_lo":  800, "lat_hi": 1600, "ram_bytes": 28364},
    "ewc-int8/cwru":         {"base_acc": 0.79, "step_drop": 0.04, "lat_lo":  250, "lat_hi":  700, "ram_bytes":  4800},
    "ewc-int8/paderborn":    {"base_acc": 0.81, "step_drop": 0.03, "lat_lo":  250, "lat_hi":  700, "ram_bytes":  4800},
    # Sprint 27 — DUAL_MODE : CMAPSS (RUL) + CWRU (faute) simultané
    "dual/cmapss+cwru":      {"base_acc": 0.61, "step_drop": 0.05, "lat_lo":  550, "lat_hi":  750, "ram_bytes": 66748},
}


def _load_stream_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sensor_stream", Path(__file__).parent / "sensor_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_config(config_path: Path, overrides: list[str]) -> dict:
    """Charge un fichier YAML et applique les overrides key=value."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override attend key=value, reçu : {item!r}")
        key, val = item.split("=", 1)
        try:
            cfg[key] = float(val)
        except ValueError:
            cfg[key] = val
    return cfg


def _run_ewc_dry_run_cl(
    n_tasks: int,
    ewc_lambda: float,
    n_samples: int,
    verbose: bool,
) -> tuple[np.ndarray, list[dict]]:
    """
    Simule un experiment EWC CL avec acc_matrix réaliste.

    - ewc_lambda >= 100 : faible oubli (EWC actif), AF ≈ 0.05
    - ewc_lambda == 0   : oubli catastrophique, AF ≈ 0.48
    """
    rng = np.random.default_rng(seed=42)
    T = n_tasks

    # Précision en diagonale (tâche courante) et chute par tâche supplémentaire entraînée
    # Modèle : acc[i,j] = diag - step_drop*(i-j), borné à [0, 1]
    # step_drop=0.03 → AF≈0.045 (EWC) ; step_drop=0.20 → AF≈0.30 (catastrophic)
    diag_acc = 0.81
    step_drop = 0.03 if ewc_lambda >= 100 else 0.20

    acc_matrix = np.full((T, T), np.nan)
    for i in range(T):
        for j in range(i + 1):
            noise = float(rng.uniform(-0.01, 0.01))
            if i == j:
                acc_matrix[i, j] = min(1.0, diag_acc + noise)
            else:
                acc_matrix[i, j] = max(0.0, diag_acc - step_drop * (i - j) + noise)

    # Résultats bruts pour profiling latence/RAM (un pass de streaming simulé)
    raw_results: list[dict] = []
    per_task = n_samples // T
    for task_id in range(T):
        final_acc = acc_matrix[T - 1, task_id]
        for _ in range(per_task):
            label = int(rng.integers(0, 2))
            pred = label if rng.random() < final_acc else 1 - label
            raw_results.append({
                "task_id":        task_id,
                "ts_ms":          len(raw_results) * 10,
                "true":           label,
                "pred":           int(pred),
                "confidence":     float(final_acc),
                "latency_us":     int(rng.uniform(300, 800)),  # Sprint 20 mesuré: 3.7 µs inférence; ~300-800 µs avec update
                "ram_bytes":      _EWC_RAM_BYTES,
                "throughput_ips": 1400,
                "status":         0,
            })

    if verbose:
        print(f"[EWC dry-run] λ={ewc_lambda} acc_matrix:\n{acc_matrix}")

    return acc_matrix, raw_results


def _run_generic_dry_run_cl(
    model: str,
    dataset: str,
    n_tasks: int,
    n_samples: int,
    base_acc: float,
    step_drop: float,
    lat_lo: int,
    lat_hi: int,
    ram_bytes: int,
    verbose: bool,
) -> tuple[np.ndarray, list[dict]]:
    """Simulation dry-run générique pour mahalanobis, tinyol, hdc, ewc-int8."""
    rng = np.random.default_rng(seed=42)
    T = n_tasks

    acc_matrix = np.full((T, T), np.nan)
    for i in range(T):
        for j in range(i + 1):
            noise = float(rng.uniform(-0.015, 0.015))
            if i == j:
                acc_matrix[i, j] = min(1.0, base_acc + noise)
            else:
                acc_matrix[i, j] = max(0.0, base_acc - step_drop * (i - j) + noise)

    raw_results: list[dict] = []
    per_task = n_samples // T
    for task_id in range(T):
        final_acc = acc_matrix[T - 1, task_id]
        for _ in range(per_task):
            label = int(rng.integers(0, 2))
            pred = label if rng.random() < final_acc else 1 - label
            raw_results.append({
                "task_id":        task_id,
                "ts_ms":          len(raw_results) * 10,
                "true":           label,
                "pred":           int(pred),
                "confidence":     float(final_acc),
                "latency_us":     int(rng.uniform(lat_lo, lat_hi)),
                "ram_bytes":      ram_bytes,
                "throughput_ips": max(1, int(1e6 / ((lat_lo + lat_hi) / 2))),
                "status":         0,
            })

    if verbose:
        print(f"[{model}/{dataset} dry-run] base_acc={base_acc} step_drop={step_drop}\n{acc_matrix}")

    return acc_matrix, raw_results


def _run_experiment(
    model: str, dataset: str, dry_run: bool, port: str, baud: int,
    n_samples: int, n_tasks: int, request_update: bool, verbose: bool,
    ewc_lambda: float | None = None,
) -> tuple[list[dict], float, np.ndarray | None]:
    """Lance le streaming et retourne (résultats bruts, durée collection, acc_matrix|None)."""
    # Dry-run : simulation entièrement synthétique (pas de dataset requis)
    if dry_run:
        t0 = time.time()
        if model == "ewc" and ewc_lambda is not None:
            acc_matrix, raw_results = _run_ewc_dry_run_cl(n_tasks, ewc_lambda, n_samples, verbose)
        else:
            key = f"{model}/{dataset}"
            params = _GENERIC_DRY_RUN_PARAMS.get(key)
            if params is None:
                raise ValueError(
                    f"Pas de paramètres dry-run pour '{key}'. "
                    f"Clés disponibles : {list(_GENERIC_DRY_RUN_PARAMS)}"
                )
            acc_matrix, raw_results = _run_generic_dry_run_cl(
                model, dataset, n_tasks, n_samples, verbose=verbose, **params
            )
        return raw_results, time.time() - t0, acc_matrix

    mod = _load_stream_module()
    X, y = mod._load_dataset(dataset)

    # Flags firmware selon le modèle (cohérence avec pipeline.h)
    _flag_map = {
        "ewc":     getattr(mod, "FRAME_FLAGS_EWC_MODE",    0x10),
        "ewc-int8":getattr(mod, "FRAME_FLAGS_INT8_MODE",   0x40),
        "hdc":     getattr(mod, "FRAME_FLAGS_HDC_MODE",    0x20),
        "tinyol":  getattr(mod, "FRAME_FLAGS_TINYOL_MODE", 0x80),
        # Sprint 30 — paires Mahalanobis + supervisé (PAIR_MODE pipeline.c)
        "pair-maha-ewc":    getattr(mod, "FRAME_FLAGS_PAIR_MAHA_EWC",    0x90),
        "pair-maha-hdc":    getattr(mod, "FRAME_FLAGS_PAIR_MAHA_HDC",    0xA0),
        "pair-maha-tinyol": getattr(mod, "FRAME_FLAGS_PAIR_MAHA_TINYOL", 0xB0),
    }
    model_flags = _flag_map.get(model, 0)

    t0 = time.time()
    # Pour EWC/INT8 board : reset avant l'expérience (réinitialise poids + lambda)
    reset_lambda = ewc_lambda if (model in ("ewc", "ewc-int8") and ewc_lambda is not None) else None
    results = mod._stream_uart(port, baud, X, y, n_samples, n_tasks,
                                0.0, request_update, verbose,
                                protocol_version=3,
                                model_flags=model_flags,
                                reset_lambda=reset_lambda)
    return results, time.time() - t0, None


def _compute_per_task_acc(results: list[dict]) -> dict[int, float]:
    """Accuracy par tâche CL."""
    per_task: dict[int, list] = {}
    for r in results:
        tid = r["task_id"]
        if tid not in per_task:
            per_task[tid] = []
        per_task[tid].append(int(r["pred"] == r["true"]))
    return {tid: float(np.mean(hits)) for tid, hits in per_task.items()}


def _compute_forgetting(per_task_acc: dict[int, float]) -> float:
    """AF simplifié : mean(1 - current_acc[t]) pour les tâches antérieures."""
    if len(per_task_acc) < 2:
        return 0.0
    task_ids = sorted(per_task_acc.keys())
    drops = []
    best_acc = per_task_acc[task_ids[0]]
    for tid in task_ids[1:]:
        # On suppose que la dernière tâche est la courante
        drops.append(max(0.0, best_acc - per_task_acc[tid]))
    return float(np.mean(drops)) if drops else 0.0


def _build_results_json(
    model: str, dataset: str, platform: str, output_dir: Path,
    results: list[dict], collection_time_s: float, n_tasks: int,
    *,
    acc_matrix: np.ndarray | None = None,
    ewc_lambda: float | None = None,
    exp_id: str | None = None,
    ram_model_bytes: int | None = None,
) -> dict:
    if not results:
        return {}

    latencies_us = [r["latency_us"] for r in results]
    rams_raw     = [r["ram_bytes"]  for r in results]
    # Protocole v3 ne rapporte pas la RAM (toujours 0) → utiliser valeur statique du config
    if ram_model_bytes and all(v == 0 for v in rams_raw):
        rams = [ram_model_bytes] * len(rams_raw)
    else:
        rams = rams_raw

    if acc_matrix is not None:
        # Métriques CL depuis acc_matrix complète (compute_cl_metrics de Phase 1)
        _repo_root = Path(__file__).parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from src.evaluation.metrics import compute_cl_metrics
        cl = compute_cl_metrics(acc_matrix)
        acc_final = round(cl["aa"], 4)
        af        = round(cl["af"], 4)
        bwt       = round(cl["bwt"], 4)
        cl_acc_matrix = cl["acc_matrix"]
    else:
        per_task_acc = _compute_per_task_acc(results)
        acc_final    = round(float(np.mean(list(per_task_acc.values()))), 4)
        af           = round(_compute_forgetting(per_task_acc), 4)
        bwt          = round(-af, 4)
        cl_acc_matrix = None

    out: dict = {
        "exp_id": exp_id or output_dir.name,
        "model": model,
        "dataset": dataset,
        "platform": platform,
        "date": datetime.now().strftime("%Y-%m-%d"),
        # Métriques obligatoires (compatibles evaluate_all.py)
        "acc_final":          acc_final,
        "avg_forgetting":     af,
        "backward_transfer":  bwt,
        "ram_peak_bytes":     int(max(rams)),
        "inference_latency_ms": round(float(np.mean(latencies_us)) / 1000.0, 4),
        "n_params":           _N_PARAMS.get(model),
        # Métriques supplémentaires board
        "n_tasks":            n_tasks,
        "n_samples_total":    len(results),
        "latency_p99_ms":     round(float(np.percentile(latencies_us, 99)) / 1000.0, 4),
        "throughput_mean_ips": int(np.mean([r["throughput_ips"] for r in results])),
        "per_task_acc":       _compute_per_task_acc_str(results),
        "collection_time_s":  round(collection_time_s, 2),
        "config_snapshot":    str(output_dir / "config_snapshot.yaml"),
        # Gap 2 — validation
        "gap2_ram_compliant":     int(max(rams)) < 64000,
        "gap2_latency_compliant": float(np.mean(latencies_us)) / 1000.0 < 100.0,
    }

    if ewc_lambda is not None:
        out["lambda_ewc"] = ewc_lambda
    if cl_acc_matrix is not None:
        out["acc_matrix"] = cl_acc_matrix

    return out


def _compute_per_task_acc_str(results: list[dict]) -> dict[str, float]:
    """Accuracy par tâche, clés en string pour JSON."""
    per_task = _compute_per_task_acc(results)
    return {str(k): round(v, 4) for k, v in per_task.items()}


def _build_run_conditions(args: argparse.Namespace, rep: int) -> dict:
    return {
        "run_type": args.run_type,
        "repetition": rep,
        "reset_method": args.reset_method,
        "flash_fresh": args.flash_fresh,
        "board_temp_c": None,
    }


def _aggregate_repetitions(all_results: list[dict]) -> dict:
    metrics = ["inference_latency_ms", "acc_final", "avg_forgetting",
               "backward_transfer", "ram_peak_bytes"]
    stats: dict = {"n_repetitions": len(all_results)}
    for m in metrics:
        vals = [r[m] for r in all_results if m in r]
        if vals:
            stats[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            stats[f"{m}_std"]  = round(float(np.std(vals)), 4)
    return stats


def _print_aggregated_summary(stats: dict) -> None:
    n = stats["n_repetitions"]
    print(f"\n=== Résultats ({n} répétitions) ===")
    for m in ["inference_latency_ms", "acc_final", "avg_forgetting", "ram_peak_bytes"]:
        mean = stats.get(f"{m}_mean")
        std  = stats.get(f"{m}_std")
        if mean is None:
            continue
        cv = (std / mean * 100) if mean else 0
        flag = "✅" if (m == "inference_latency_ms" and cv <= 5) else ""
        print(f"  {m}: {mean} ± {std}  (σ/μ={cv:.1f}% {flag})")


def _save_config_snapshot(
    args: argparse.Namespace,
    output_dir: Path,
    cfg: dict,
    model: str,
    dataset: str,
) -> None:
    snap = {
        "model": model,
        "dataset": dataset,
        "n_samples": cfg.get("n_samples", args.n_samples),
        "n_tasks": cfg.get("n_tasks", args.n_tasks),
        "update_requested": args.update,
        "dry_run": args.dry_run,
        "port": None if args.dry_run else args.port,
        "baud": args.baud,
        "platform": args.platform,
        "date": datetime.now().isoformat(),
        "protocol_version": 3,
        "board_config": str(args.config) if args.config else f"configs/board_{model}.yaml",
    }
    if cfg.get("lambda_ewc") is not None:
        snap["lambda_ewc"] = cfg["lambda_ewc"]
    if cfg.get("fisher_decay") is not None:
        snap["fisher_decay"] = cfg["fisher_decay"]
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snap, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enregistre une expérience board dans experiments/ (format unifié Phase 1)")
    # Arguments existants (backward compat)
    parser.add_argument("--model", choices=["mahalanobis", "ewc", "tinyol", "hdc", "ewc-int8"])
    parser.add_argument("--dataset", choices=["cwru", "monitoring", "pronostia", "cmapss", "paderborn", "battery"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--update", action="store_true", help="Active la mise à jour incrémentale")
    parser.add_argument("--platform", default="nucleo_f439zi",
                        choices=["nucleo_f439zi", "stm32n6_eval", "edge_spectrum"])
    parser.add_argument("--output", required=True, help="Répertoire experiments/exp_S19_XX")
    parser.add_argument("--verbose", action="store_true")
    # Nouveaux arguments S2005
    parser.add_argument("--config", type=Path, metavar="YAML",
                        help="Config board YAML (ex. configs/board_ewc.yaml)")
    parser.add_argument("--exp-id", metavar="ID",
                        help="Identifiant condition → fichier results_{exp_id}.json")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                        help="Override clé=valeur du config YAML (ex. lambda_ewc=0.0)")
    # Protocole expérimental S2113
    parser.add_argument("--repetitions", type=int, default=1, metavar="N",
                        help="Nombre de répétitions indépendantes (défaut : 1)")
    parser.add_argument("--run-type", choices=["cold", "warm"], default="warm",
                        help="Type de run : cold (après reset) ou warm (régime établi)")
    parser.add_argument("--reset-method", choices=["nrst", "openocd"], default="nrst",
                        help="Méthode de reset entre répétitions (journalisation)")
    parser.add_argument("--flash-fresh", action="store_true",
                        help="Indique que le binaire a été re-flashé avant ce run")
    args = parser.parse_args()

    # Chargement config YAML + overrides (même sans --config)
    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config, args.override or [])
    elif args.override:
        for item in args.override:
            if "=" not in item:
                raise ValueError(f"--override attend key=value, reçu : {item!r}")
            key, val = item.split("=", 1)
            try:
                cfg[key] = float(val)
            except ValueError:
                cfg[key] = val

    # Résolution des valeurs finales (CLI explicite > YAML > défaut)
    model   = args.model   or cfg.get("model")
    dataset = args.dataset or cfg.get("dataset")
    if not model or not dataset:
        parser.error("--model et --dataset sont requis (ou --config avec model/dataset)")

    n_tasks = int(cfg.get("n_tasks") or cfg.get("n_tasks_board") or args.n_tasks)
    _per_task = cfg.get("n_samples_per_task")
    n_samples = int(_per_task * n_tasks if _per_task else cfg.get("n_samples", args.n_samples))
    ewc_lambda   = cfg.get("lambda_ewc") or cfg.get("EWC_LAMBDA")
    fisher_decay = cfg.get("fisher_decay") or cfg.get("FISHER_EMA_DECAY") or 0.99
    ram_model_bytes = cfg.get("ram_model_bytes") or cfg.get("RAM_MAHA_BYTES") or cfg.get("RAM_EWC_BYTES")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie
    out_filename = f"results_{args.exp_id}.json" if args.exp_id else "results.json"

    print(f"Expérience board — modèle={model} dataset={dataset} "
          f"({'dry-run' if args.dry_run else args.port})"
          + (f" λ={ewc_lambda}" if ewc_lambda is not None else "")
          + (f" [{args.repetitions} répétitions]" if args.repetitions > 1 else ""))

    all_rep_results: list[dict] = []
    for rep in range(1, args.repetitions + 1):
        if args.repetitions > 1:
            print(f"\n=== Répétition {rep}/{args.repetitions} ===")
            if rep > 1 and not args.dry_run:
                if args.reset_method == "openocd" or not sys.stdin.isatty():
                    import subprocess
                    print("  → Reset via OpenOCD...")
                    subprocess.run(
                        ["openocd", "-f", "interface/stlink.cfg", "-f", "target/stm32f4x.cfg",
                         "-c", "init; reset; exit"],
                        capture_output=True, timeout=15,
                    )
                    time.sleep(2)
                else:
                    input("  → Effectuez le reset NRST, attendez 2 s, puis appuyez sur Entrée...")

        raw_results, elapsed, acc_matrix = _run_experiment(
            model, dataset, args.dry_run, args.port, args.baud,
            int(n_samples), int(n_tasks), args.update, args.verbose,
            ewc_lambda=ewc_lambda,
        )

        results_json = _build_results_json(
            model, dataset, args.platform,
            output_dir, raw_results, elapsed, int(n_tasks),
            acc_matrix=acc_matrix,
            ewc_lambda=ewc_lambda,
            exp_id=args.exp_id,
            ram_model_bytes=int(ram_model_bytes) if ram_model_bytes else None,
        )

        results_json["run_conditions"] = _build_run_conditions(args, rep)

        if args.repetitions > 1:
            rep_filename = (f"results_{args.exp_id}_rep{rep}.json" if args.exp_id
                            else f"results_rep{rep}.json")
            with open(output_dir / rep_filename, "w") as f:
                json.dump(results_json, f, indent=2)

        all_rep_results.append(results_json)

    # Résultat final = dernière répétition + run_statistics agrégées si N > 1
    final_result = all_rep_results[-1].copy()
    if args.repetitions > 1:
        final_result["run_statistics"] = _aggregate_repetitions(all_rep_results)

    with open(output_dir / out_filename, "w") as f:
        json.dump(final_result, f, indent=2)

    _save_config_snapshot(args, output_dir, cfg, model, dataset)

    print(f"\n--- Résultats {args.exp_id or output_dir.name} ---")
    for key in ["acc_final", "avg_forgetting", "backward_transfer",
                "ram_peak_bytes", "inference_latency_ms", "n_params"]:
        print(f"  {key}: {final_result.get(key)}")
    if ewc_lambda is not None:
        print(f"  lambda_ewc: {ewc_lambda}")
    print(f"  Gap 2 RAM  : {'✅' if final_result.get('gap2_ram_compliant') else '❌'}")
    print(f"  Gap 2 lat  : {'✅' if final_result.get('gap2_latency_compliant') else '❌'}")

    if args.repetitions > 1 and "run_statistics" in final_result:
        _print_aggregated_summary(final_result["run_statistics"])

    print(f"\nSauvé : {output_dir}/{out_filename}")


if __name__ == "__main__":
    main()
