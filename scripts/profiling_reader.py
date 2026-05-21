"""
profiling_reader.py — Parse les métriques de profiling firmware (UART) et sauve JSON.

Lit le protocole v2 étendu (réponse 14 B) en continu pendant N inférences,
puis calcule les statistiques et sauve un profiling.json dans experiments/.

Peut aussi fonctionner en mode "parse-only" sur un fichier CSV existant
produit par board_dataset_builder.py.

Usage :
    # Mode live (board connectée)
    python scripts/profiling_reader.py --port /dev/ttyACM0 --n-samples 100 \\
        --save experiments/exp_S18_01/profiling.json

    # Mode parse-only (depuis CSV board_dataset_builder)
    python scripts/profiling_reader.py --from-csv experiments/exp_S18_01/dataset.csv \\
        --save experiments/exp_S18_01/profiling.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def _load_profiling_config() -> dict:
    cfg_path = Path("configs/profiling_config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {"LATENCY_ALERT_MS": 10.0, "RAM_ALERT_BYTES": 52000, "THROUGHPUT_MIN_IPS": 10}


def _compute_profiling_stats(records: list[dict], platform: str, cfg: dict) -> dict:
    latencies_us = [r["latency_us"] for r in records]
    rams = [r["ram_bytes"] for r in records]
    thrs = [r["throughput_ips"] for r in records]

    lat_mean_ms = float(np.mean(latencies_us)) / 1000.0
    ram_peak    = int(max(rams))
    thr_mean    = int(np.mean(thrs))

    alerts = []
    if lat_mean_ms > cfg.get("LATENCY_ALERT_MS", 10.0):
        alerts.append(f"LATENCY: {lat_mean_ms:.2f} ms > seuil {cfg['LATENCY_ALERT_MS']} ms")
    if ram_peak > cfg.get("RAM_ALERT_BYTES", 52000):
        alerts.append(f"RAM: {ram_peak} B > seuil {cfg['RAM_ALERT_BYTES']} B")
    if thr_mean < cfg.get("THROUGHPUT_MIN_IPS", 10):
        alerts.append(f"THROUGHPUT: {thr_mean} ips < seuil {cfg['THROUGHPUT_MIN_IPS']} ips")

    return {
        "platform": platform,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(records),
        "latency_mean_us": round(float(np.mean(latencies_us)), 2),
        "latency_p50_us":  round(float(np.percentile(latencies_us, 50)), 2),
        "latency_p99_us":  round(float(np.percentile(latencies_us, 99)), 2),
        "latency_mean_ms": round(lat_mean_ms, 4),
        "latency_p99_ms":  round(float(np.percentile(latencies_us, 99)) / 1000.0, 4),
        "ram_mean_bytes":  int(np.mean(rams)),
        "ram_peak_bytes":  ram_peak,
        "throughput_mean_ips": thr_mean,
        "throughput_min_ips":  int(min(thrs)),
        "alerts": alerts,
        "gap2_compliant": ram_peak < 64000 and lat_mean_ms < 100.0,
    }


def parse_from_csv(csv_path: Path) -> list[dict]:
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "latency_us":    int(row["latency_us"]),
                "ram_bytes":     int(row["ram_bytes"]),
                "throughput_ips": int(row["throughput_ips"]),
            })
    return records


def collect_from_uart(port: str, baud: int, n_samples: int, verbose: bool) -> list[dict]:
    """Utilise sensor_stream en mode passif pour collecter les métriques."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sensor_stream", Path(__file__).parent / "sensor_stream.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X, y = mod._load_dataset("monitoring")
    results = mod._stream_uart(port, baud, X, y, n_samples, 1, 0.0, False, verbose)
    return [{"latency_us": r["latency_us"], "ram_bytes": r["ram_bytes"],
             "throughput_ips": r["throughput_ips"]} for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse métriques profiling firmware → JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", help="Port UART (ex: /dev/ttyACM0)")
    group.add_argument("--from-csv", help="CSV existant de board_dataset_builder")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--platform", default="nucleo_f439zi")
    parser.add_argument("--save", required=True, help="Chemin profiling.json de sortie")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = _load_profiling_config()

    if args.from_csv:
        print(f"Parsing CSV : {args.from_csv}")
        records = parse_from_csv(Path(args.from_csv))
    else:
        print(f"Collecte UART sur {args.port}...")
        records = collect_from_uart(args.port, args.baud, args.n_samples, args.verbose)

    stats = _compute_profiling_stats(records, args.platform, cfg)

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))

    print(f"\n--- Profiling ({stats['n_samples']} samples) ---")
    print(f"  Latence moyenne : {stats['latency_mean_ms']} ms (P99: {stats['latency_p99_ms']} ms)")
    print(f"  RAM peak        : {stats['ram_peak_bytes']} B")
    print(f"  Throughput      : {stats['throughput_mean_ips']} ips")
    print(f"  Gap 2 compliant : {stats['gap2_compliant']}")
    if stats["alerts"]:
        print("  ⚠️  ALERTES :")
        for a in stats["alerts"]:
            print(f"    - {a}")
    print(f"\nSauvegardé : {out_path}")


if __name__ == "__main__":
    main()
