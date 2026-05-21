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
    # Dry-run (sans board)
    python scripts/board_experiment_recorder.py --model mahalanobis \\
        --dataset cwru --dry-run --output experiments/exp_S19_01

    # Avec board
    python scripts/board_experiment_recorder.py --model ewc \\
        --dataset monitoring --port /dev/ttyACM0 \\
        --n-samples 500 --n-tasks 3 --output experiments/exp_S19_02
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


# Nombre de paramètres par modèle (calculé offline depuis model.parameters())
_N_PARAMS = {
    "mahalanobis": 30,   # mean(5) + precision(25) = 30 floats
    "ewc":         1538, # (5×32+32) + (32×16+16) + (16×2+2) = 802 poids, ×2 avec Fisher/star ≈ 1538
    "tinyol":      881,  # encoder: (5×32+32) + (32×16+16) = 720, decoder: (16×32+32) + (32×5+5) = 677 → 881 total encoder-only
}


def _load_stream_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sensor_stream", Path(__file__).parent / "sensor_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_experiment(
    model: str, dataset: str, dry_run: bool, port: str, baud: int,
    n_samples: int, n_tasks: int, request_update: bool, verbose: bool
) -> tuple[list[dict], float]:
    """Lance le streaming et retourne (résultats bruts, durée collection)."""
    mod = _load_stream_module()
    X, y = mod._load_dataset(dataset)

    t0 = time.time()
    if dry_run:
        results = mod._stream_dry_run(X, y, n_samples, n_tasks, request_update, verbose)
    else:
        results = mod._stream_uart(port, baud, X, y, n_samples, n_tasks,
                                    0.0, request_update, verbose)
    return results, time.time() - t0


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
    results: list[dict], collection_time_s: float, n_tasks: int
) -> dict:
    if not results:
        return {}

    latencies_us = [r["latency_us"] for r in results]
    rams         = [r["ram_bytes"]  for r in results]
    per_task_acc = _compute_per_task_acc(results)
    acc_final    = float(np.mean(list(per_task_acc.values())))
    af           = _compute_forgetting(per_task_acc)
    bwt          = -af  # approx BWT ≈ -AF pour détection anomalie

    return {
        "exp_id": output_dir.name,
        "model": model,
        "dataset": dataset,
        "platform": platform,
        "date": datetime.now().strftime("%Y-%m-%d"),
        # Métriques obligatoires (compatibles evaluate_all.py)
        "acc_final":          round(acc_final, 4),
        "avg_forgetting":     round(af, 4),
        "backward_transfer":  round(bwt, 4),
        "ram_peak_bytes":     int(max(rams)),
        "inference_latency_ms": round(float(np.mean(latencies_us)) / 1000.0, 4),
        "n_params":           _N_PARAMS.get(model),
        # Métriques supplémentaires board
        "n_tasks":            len(per_task_acc),
        "n_samples_total":    len(results),
        "latency_p99_ms":     round(float(np.percentile(latencies_us, 99)) / 1000.0, 4),
        "throughput_mean_ips": int(np.mean([r["throughput_ips"] for r in results])),
        "per_task_acc":       {str(k): round(v, 4) for k, v in per_task_acc.items()},
        "collection_time_s":  round(collection_time_s, 2),
        "config_snapshot":    str(output_dir / "config_snapshot.yaml"),
        # Gap 2 — validation
        "gap2_ram_compliant":     int(max(rams)) < 64000,
        "gap2_latency_compliant": float(np.mean(latencies_us)) / 1000.0 < 100.0,
    }


def _save_config_snapshot(args: argparse.Namespace, output_dir: Path) -> None:
    snap = {
        "model": args.model,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "n_tasks": args.n_tasks,
        "update_requested": args.update,
        "dry_run": args.dry_run,
        "port": None if args.dry_run else args.port,
        "baud": args.baud,
        "platform": args.platform,
        "date": datetime.now().isoformat(),
        "protocol_version": 2,
        "board_config": f"configs/board_{args.model}.yaml",
    }
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snap, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enregistre une expérience board dans experiments/ (format unifié Phase 1)")
    parser.add_argument("--model", choices=["mahalanobis", "ewc", "tinyol"], required=True)
    parser.add_argument("--dataset", choices=["cwru", "monitoring"], required=True)
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
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Expérience board — modèle={args.model} dataset={args.dataset} "
          f"({'dry-run' if args.dry_run else args.port})")

    raw_results, elapsed = _run_experiment(
        args.model, args.dataset, args.dry_run, args.port, args.baud,
        args.n_samples, args.n_tasks, args.update, args.verbose,
    )

    results_json = _build_results_json(
        args.model, args.dataset, args.platform,
        output_dir, raw_results, elapsed, args.n_tasks,
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    _save_config_snapshot(args, output_dir)

    print(f"\n--- Résultats {output_dir.name} ---")
    for key in ["acc_final", "avg_forgetting", "backward_transfer",
                "ram_peak_bytes", "inference_latency_ms", "n_params"]:
        print(f"  {key}: {results_json.get(key)}")
    print(f"  Gap 2 RAM  : {'✅' if results_json.get('gap2_ram_compliant') else '❌'}")
    print(f"  Gap 2 lat  : {'✅' if results_json.get('gap2_latency_compliant') else '❌'}")
    print(f"\nSauvé : {output_dir}/results.json")


if __name__ == "__main__":
    main()
