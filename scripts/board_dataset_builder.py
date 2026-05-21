"""
board_dataset_builder.py — Collecte réponses firmware → CSV/HDF5 + JSON metadata.

Orchestre sensor_stream.py et profiling_reader.py pour produire un répertoire
experiments/exp_S18_XX/ contenant :
  - dataset.csv      : samples (features + pred + true + latency + task_id)
  - profiling.json   : métriques board (latence, RAM, throughput)
  - results.json     : résumé statistiques (acc, latence, RAM)
  - config_snapshot.yaml : config utilisée

Usage :
    # Dry-run (sans board)
    python scripts/board_dataset_builder.py --dataset cwru --dry-run \\
        --output experiments/exp_S18_01

    # Avec board
    python scripts/board_dataset_builder.py --dataset monitoring \\
        --port /dev/ttyACM0 --n-samples 500 --output experiments/exp_S18_01
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def _run_stream(
    dataset: str, dry_run: bool, port: str, baud: int,
    n_samples: int, n_tasks: int, rate_hz: float, request_update: bool,
    verbose: bool,
) -> list[dict]:
    """Lance sensor_stream et retourne les résultats bruts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sensor_stream", Path(__file__).parent / "sensor_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X, y = mod._load_dataset(dataset)
    if dry_run:
        return mod._stream_dry_run(X, y, n_samples, n_tasks, request_update, verbose)
    else:
        return mod._stream_uart(port, baud, X, y, n_samples, n_tasks,
                                rate_hz, request_update, verbose)


def _save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _compute_results_json(results: list[dict], dataset: str,
                           platform: str, output_dir: Path) -> dict:
    if not results:
        return {}

    preds = [r["pred"] for r in results]
    trues = [r["true"] for r in results]
    latencies = [r["latency_us"] for r in results]
    acc = sum(p == t for p, t in zip(preds, trues)) / len(results)

    return {
        "exp_id": output_dir.name,
        "model": "streaming_pipeline",
        "dataset": dataset,
        "platform": platform,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "acc_final": round(acc, 4),
        "avg_forgetting": None,     # calculé par board_experiment_recorder
        "backward_transfer": None,
        "ram_peak_bytes": int(max(r["ram_bytes"] for r in results)),
        "inference_latency_ms": round(float(np.mean(latencies)) / 1000.0, 4),
        "n_params": None,           # renseigné par board_experiment_recorder
        "n_samples": len(results),
        "n_tasks": len(set(r["task_id"] for r in results)),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)) / 1000.0, 4),
        "throughput_mean_ips": int(np.mean([r["throughput_ips"] for r in results])),
        "config_snapshot": str(output_dir / "config_snapshot.yaml"),
    }


def _save_config_snapshot(args: argparse.Namespace, output_dir: Path) -> None:
    snap = {
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "n_tasks": args.n_tasks,
        "rate_hz": args.rate_hz,
        "update_requested": args.update,
        "dry_run": args.dry_run,
        "port": None if args.dry_run else args.port,
        "baud": args.baud,
        "platform": args.platform,
        "date": datetime.now().isoformat(),
        "protocol_version": 2,
    }
    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snap, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collecte réponses firmware → CSV + JSON (pipeline données S18)")
    parser.add_argument("--dataset", choices=["cwru", "monitoring"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=0.0)
    parser.add_argument("--update", action="store_true", help="Demande mise à jour incrémentale")
    parser.add_argument("--platform", default="nucleo_f439zi",
                        choices=["nucleo_f439zi", "stm32n6_eval", "edge_spectrum"])
    parser.add_argument("--output", required=True, help="Répertoire experiments/exp_S18_XX")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming '{args.dataset}' → {output_dir} ({'dry-run' if args.dry_run else args.port})...")
    t0 = time.time()
    results = _run_stream(
        args.dataset, args.dry_run, args.port, args.baud,
        args.n_samples, args.n_tasks, args.rate_hz, args.update, args.verbose,
    )
    elapsed = time.time() - t0

    # 1. CSV
    csv_path = output_dir / "dataset.csv"
    _save_csv(results, csv_path)
    print(f"  CSV sauvé : {csv_path} ({len(results)} lignes)")

    # 2. results.json
    results_json = _compute_results_json(results, args.dataset, args.platform, output_dir)
    results_json["collection_time_s"] = round(elapsed, 2)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  results.json sauvé : acc={results_json.get('acc_final')}")

    # 3. config_snapshot.yaml
    _save_config_snapshot(args, output_dir)
    print(f"  config_snapshot.yaml sauvé")

    print(f"\nDone — {len(results)} samples collectés en {elapsed:.1f}s")
    print(f"  RAM peak : {results_json.get('ram_peak_bytes')} B")
    print(f"  Latence moyenne : {results_json.get('inference_latency_ms')} ms")


if __name__ == "__main__":
    main()
