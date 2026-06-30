"""
edge_spectrum_demo.py — Pipeline capteur Edge Spectrum → NUCLEO → décision temps réel.

Deux scénarios :
    Scénario A : données Edge Spectrum réelles (CSV fourni par Fred).
        Format CSV attendu (à confirmer avec Fred — TODO(fred)) :
            timestamp_ms, sensor_1, sensor_2, ..., sensor_N, label (0=normal, 1=fault)
        Si N > 5, sélection automatique du top-5 par mutual information.

    Scénario B (repli) : CWRU comme proxy industriel.
        Activé si --dataset cwru_proxy. 3 tâches CL : ball → inner_race → outer_race.

Usage :
    # Dry-run Scénario B (CWRU proxy — ne requiert pas la board)
    python scripts/edge_spectrum_demo.py \\
        --dataset cwru_proxy --model ewc \\
        --dry-run --n-samples 50 --tasks 3

    # Live Scénario B avec board connectée
    python scripts/edge_spectrum_demo.py \\
        --dataset cwru_proxy --model ewc \\
        --port /dev/ttyACM0 --baud 115200 \\
        --n-samples 300 --tasks 3 --update --consolidate \\
        --output experiments/exp_S23_benchmark/stream_cwru_proxy.json

    # Dry-run Scénario A (CSV Edge Spectrum)
    python scripts/edge_spectrum_demo.py \\
        --input data/raw/edge_spectrum/demo_feed.csv \\
        --model ewc --dry-run \\
        --output experiments/exp_S23_benchmark/stream_live.json

    # Live Scénario A avec board connectée
    python scripts/edge_spectrum_demo.py \\
        --input data/raw/edge_spectrum/demo_feed.csv \\
        --model ewc --port /dev/ttyACM0 --baud 115200 \\
        --rate-hz 10 --update --consolidate \\
        --output experiments/exp_S23_benchmark/stream_live.json
"""

# TODO(fred) : Confirmer le format de fichier avant 2026-06-22.
#   - Format CSV ou JSON ou binary ?
#   - Fréquence d'échantillonnage ? Nombre de features ?
#   - Labels inclus ou à inférer depuis notes de maintenance ?

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Réutilisation des primitives UART depuis sensor_stream.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sensor_stream import (
    FRAME_FLAGS_CONSOLIDATE,
    FRAME_FLAGS_EWC_MODE,
    FRAME_FLAGS_HDC_MODE,
    FRAME_FLAGS_INT8_MODE,
    FRAME_FLAGS_PROFILING,
    FRAME_FLAGS_UPDATE,
    RESPONSE_V2_FMT,
    RESPONSE_V2_SIZE,
    RESPONSE_V3_FMT,
    RESPONSE_V3_SIZE,
    STATUS_CRC_ERR,
    STATUS_OK,
    UART_TIMEOUT_S,
    build_frame_v2,
    crc8,
    parse_response,
)

MAX_FEATURES = 5
CWRU_CSV_PATH = Path("data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------


def _load_edge_spectrum_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Charge un CSV Edge Spectrum et sélectionne le top-5 features si nécessaire.

    Returns
    -------
    X : (N, ≤5) float32
    y : (N,) int
    feature_names : list[str]
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("La colonne 'label' est absente du CSV Edge Spectrum.")

    y = df["label"].to_numpy(dtype=int)

    # Exclure timestamp et label des features
    exclude = {"label"}
    if "timestamp_ms" in df.columns:
        exclude.add("timestamp_ms")
    feature_cols = [c for c in df.columns if c not in exclude]

    if not feature_cols:
        raise ValueError("Aucune colonne de feature trouvée dans le CSV Edge Spectrum.")

    X_all = df[feature_cols].to_numpy(dtype=np.float32)

    if len(feature_cols) > MAX_FEATURES:
        print(f"  {len(feature_cols)} features détectées — sélection top-{MAX_FEATURES} par mutual info...")
        from sklearn.feature_selection import mutual_info_classif

        # Fit mutual info sur les 500 premières lignes de la classe normale
        normal_mask = y == 0
        X_normal = X_all[normal_mask][:500]
        y_normal = y[normal_mask][:500]
        if len(X_normal) < 10:
            X_normal, y_normal = X_all[:500], y[:500]

        scores = mutual_info_classif(X_normal, y_normal, random_state=42)
        top_idx = np.argsort(scores)[::-1][:MAX_FEATURES]
        top_idx = sorted(top_idx.tolist())
        feature_cols = [feature_cols[i] for i in top_idx]
        X_all = X_all[:, top_idx]
        print(f"  Features retenues : {feature_cols}")

    return X_all, y, feature_cols


def _load_cwru_proxy(n_samples_per_task: int = 100) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Charge CWRU (3 tâches CL) comme proxy industriel — Scénario B.

    Returns
    -------
    X : (N, 5) float32 — top-5 features CWRU (sélectionnées par variance)
    y : (N,) int
    segments : [(task_name, n_samples), ...]
    """
    print("  AVERTISSEMENT: Utilisation de CWRU comme proxy Edge Spectrum (Scénario B).")
    print("  TODO(fred): remplacer par données Edge Spectrum réelles.")

    from src.data.cwru_dataset import CWRUDataset, CWRUFaultTypeStream

    if not CWRU_CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV CWRU introuvable : {CWRU_CSV_PATH}\n"
            "Télécharger le dataset CWRU Bearing et le placer dans data/raw/CWRU Bearing Dataset/"
        )

    ds = CWRUDataset(csv_path=CWRU_CSV_PATH)
    stream = CWRUFaultTypeStream(ds)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    segments: list[tuple[str, int]] = []

    for task_id, task_name, X_task, y_task in stream.iter_tasks():
        n = min(n_samples_per_task, len(X_task))
        idx = np.random.default_rng(42 + task_id).choice(len(X_task), size=n, replace=False)
        X_sel = X_task[idx]
        y_sel = y_task[idx]
        all_X.append(X_sel)
        all_y.append(y_sel)
        segments.append((task_name, n))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(int)

    # CWRU a 9 features — on prend les 5 premières (variance maximale dans l'ordre standard)
    if X.shape[1] > MAX_FEATURES:
        variances = X.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:MAX_FEATURES]
        top_idx = sorted(top_idx.tolist())
        X = X[:, top_idx]

    return X, y, segments


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Dispatcher de chargement de données selon le scénario.

    Returns
    -------
    X : (N, ≤5) float32, normalisé (StandardScaler fit sur tâche 0)
    y : (N,) int
    segments : [(task_name, n_samples), ...]
    """
    if args.dataset == "cwru_proxy":
        n_per_task = args.n_samples // max(args.tasks, 1)
        X, y, segments = _load_cwru_proxy(n_samples_per_task=n_per_task)

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier CSV Edge Spectrum introuvable : {input_path}")
        print(f"  Chargement CSV Edge Spectrum : {input_path}")
        X_raw, y, feature_names = _load_edge_spectrum_csv(input_path)
        # Découpage en tâches temporelles égales
        n_tasks = args.tasks
        n_total = len(X_raw)
        n_per = n_total // n_tasks
        segments = [(f"task_{i}", min(n_per, args.n_samples // n_tasks)) for i in range(n_tasks)]
        X = X_raw

    else:
        raise ValueError("Spécifier --dataset cwru_proxy OU --input <csv_path>")

    # Normalisation : StandardScaler fit sur les données de la tâche 0
    n_task0 = segments[0][1] if segments else len(X)
    scaler = StandardScaler()
    scaler.fit(X[:n_task0])
    X = scaler.transform(X).astype(np.float32)

    return X, y, segments


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _stream_demo_dry_run(
    X: np.ndarray,
    y: np.ndarray,
    segments: list[tuple[str, int]],
    request_update: bool,
    consolidate: bool,
    verbose: bool,
) -> list[dict]:
    """Simule la board en dry-run : renvoie des réponses synthétiques cohérentes."""
    results: list[dict] = []
    t0_ms = int(time.time() * 1000)
    offset = 0

    for task_id, (task_name, n_samples) in enumerate(segments):
        is_last_task = task_id == len(segments) - 1
        end = min(offset + n_samples, len(X))
        indices = np.arange(offset, end)
        if len(indices) == 0:
            indices = np.random.choice(len(X), size=n_samples, replace=False)
        offset = end

        for local_i, idx in enumerate(indices):
            is_last = local_i == len(indices) - 1
            features, label = X[idx], int(y[idx])
            ts_ms = int(time.time() * 1000) - t0_ms

            flags = FRAME_FLAGS_PROFILING
            if request_update:
                flags |= FRAME_FLAGS_UPDATE
            if consolidate and is_last and not is_last_task:
                flags |= FRAME_FLAGS_CONSOLIDATE

            frame = build_frame_v2(features, label, task_id, ts_ms, flags)
            payload, recv_crc = frame[:-1], frame[-1]
            crc_ok = crc8(payload) == recv_crc

            entry: dict = {
                "task_id": task_id,
                "task_name": task_name,
                "ts_ms": ts_ms,
                "true": label,
                "pred": label,
                "confidence": 1.0,
                "latency_us": 3,
                "latency_ms": 0.003,
                "ram_bytes": 200,
                "gap2_latency_compliant": True,
                "throughput_ips": 333333,
                "status": STATUS_OK if crc_ok else STATUS_CRC_ERR,
                "consolidate": bool(flags & FRAME_FLAGS_CONSOLIDATE),
            }
            results.append(entry)

            if verbose:
                marker = " [CONSOLIDATE]" if entry["consolidate"] else ""
                print(f"[{task_name} ts={ts_ms}ms] true={label} pred={label} "
                      f"conf=1.000 lat=3µs ram=200B{marker}")

        print(f"  → Tâche {task_id} ({task_name}) terminée — {len(indices)} samples (dry-run)")

    return results


def _stream_demo_uart(
    port: str,
    baud: int,
    X: np.ndarray,
    y: np.ndarray,
    segments: list[tuple[str, int]],
    rate_hz: float,
    request_update: bool,
    consolidate: bool,
    verbose: bool,
    model_flags: int = 0,
    protocol_version: int = 2,
) -> list[dict]:
    """Stream les données via UART vers la NUCLEO et collecte les réponses firmware."""
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial requis : pip install pyserial")

    results: list[dict] = []
    t0_ms = int(time.time() * 1000)
    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    offset = 0
    resp_size = RESPONSE_V3_SIZE if protocol_version >= 3 else RESPONSE_V2_SIZE

    with serial.Serial(port, baud, timeout=UART_TIMEOUT_S, dsrdtr=False, rtscts=False) as ser:
        ser.dtr = True
        time.sleep(0.05)
        ser.dtr = False
        time.sleep(0.5)
        ser.reset_input_buffer()

        for task_id, (task_name, n_samples) in enumerate(segments):
            is_last_task = task_id == len(segments) - 1
            end = min(offset + n_samples, len(X))
            indices = np.arange(offset, end)
            if len(indices) == 0:
                indices = np.random.choice(len(X), size=n_samples, replace=False)
            offset = end

            for local_i, idx in enumerate(indices):
                is_last = local_i == len(indices) - 1
                features, label = X[idx], int(y[idx])
                ts_ms = int(time.time() * 1000) - t0_ms
                t_send = time.monotonic()

                flags = FRAME_FLAGS_PROFILING | model_flags
                if request_update:
                    flags |= FRAME_FLAGS_UPDATE
                if consolidate and is_last and not is_last_task:
                    flags |= FRAME_FLAGS_CONSOLIDATE

                frame = build_frame_v2(features, label, task_id, ts_ms, flags)
                ser.write(frame)

                raw = ser.read(resp_size)
                if len(raw) != resp_size:
                    if verbose:
                        print(f"[WARN] Timeout {task_name} ({len(raw)}/{resp_size} B)")
                    elapsed = time.monotonic() - t_send
                    if interval_s > elapsed:
                        time.sleep(interval_s - elapsed)
                    continue

                entry: dict = {
                    "task_id": task_id,
                    "task_name": task_name,
                    "ts_ms": ts_ms,
                    "true": label,
                    "consolidate": bool(flags & FRAME_FLAGS_CONSOLIDATE),
                }
                entry.update(parse_response(raw))
                lat_ms = entry["latency_us"] / 1000.0
                entry["latency_ms"] = round(lat_ms, 4)
                entry["gap2_latency_compliant"] = lat_ms < 100.0
                results.append(entry)

                if verbose:
                    marker = " [CONSOLIDATE]" if entry["consolidate"] else ""
                    print(f"[{task_name} ts={ts_ms}ms] true={label} "
                          f"pred={entry['pred']} conf={entry['confidence']:.3f} "
                          f"lat={entry['latency_us']}µs{marker}")

                elapsed = time.monotonic() - t_send
                if interval_s > elapsed:
                    time.sleep(interval_s - elapsed)

            print(f"  → Tâche {task_id} ({task_name}) terminée — {len(indices)} samples")

    return results


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------


def _compute_demo_stats(results: list[dict]) -> dict:
    if not results:
        return {"n_samples": 0}

    latencies_us = [r["latency_us"] for r in results]
    preds = [r["pred"] for r in results]
    trues = [r["true"] for r in results]
    acc = sum(p == t for p, t in zip(preds, trues)) / len(results)
    gap2_ok = all(r.get("gap2_latency_compliant", True) for r in results)

    return {
        "n_samples": len(results),
        "accuracy": round(acc, 4),
        "latency_mean_us": round(float(np.mean(latencies_us)), 2),
        "latency_p50_us": round(float(np.percentile(latencies_us, 50)), 2),
        "latency_p99_us": round(float(np.percentile(latencies_us, 99)), 2),
        "latency_mean_ms": round(float(np.mean(latencies_us)) / 1000.0, 4),
        "ram_mean_bytes": int(np.mean([r.get("ram_bytes", 0) for r in results])),
        "gap2_latency_compliant": gap2_ok,
        "crc_errors": sum(1 for r in results if r.get("status", 0) & STATUS_CRC_ERR),
    }


def _print_summary(results: list[dict], stats: dict, scenario: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Benchmark Edge Spectrum — {scenario}")
    print(f"{'='*60}")
    print(f"  {'Métrique':<30} {'Valeur'}")
    print(f"  {'-'*50}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
    print(f"{'='*60}")

    if results:
        print("\n  Échantillons (10 derniers) :")
        print(f"  {'task':<12} {'true':<6} {'pred':<6} {'conf':<8} {'lat_µs':<10} {'gap2'}")
        print(f"  {'-'*55}")
        for r in results[-10:]:
            gap2 = "✅" if r.get("gap2_latency_compliant", True) else "❌"
            print(f"  {r.get('task_name', str(r['task_id'])):<12} "
                  f"{r['true']:<6} {r['pred']:<6} "
                  f"{r['confidence']:<8.3f} {r['latency_us']:<10} {gap2}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo Edge Spectrum — pipeline capteur industriel → NUCLEO-F439ZI"
    )

    # Source de données
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--input", type=str, metavar="CSV",
        help="CSV Edge Spectrum (Scénario A — format: timestamp_ms, feat_1..N, label)"
    )
    src.add_argument(
        "--dataset", choices=["cwru_proxy"],
        help="Scénario B : proxy CWRU (3 tâches CL par type de défaut)"
    )

    # Board
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true",
                        help="Simuler la board (pas de port série requis)")

    # Modèle
    parser.add_argument("--model", choices=["ewc", "ewc-int8", "mahalanobis", "hdc"],
                        default="ewc")

    # Streaming
    parser.add_argument("--rate-hz", type=float, default=10.0,
                        help="Fréquence d'envoi (0 = vitesse max)")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Nombre total d'échantillons à streamer")
    parser.add_argument("--tasks", type=int, default=3,
                        help="Nombre de tâches CL (Scénario A uniquement)")
    parser.add_argument("--update", action="store_true",
                        help="Activer la mise à jour incrémentale sur la board")
    parser.add_argument("--consolidate", action="store_true",
                        help="Envoyer signal EWC consolidate entre tâches")

    # Protocole
    parser.add_argument("--protocol-version", type=int, default=2, choices=[2, 3])

    # Sortie
    parser.add_argument("--output", type=str,
                        help="Fichier JSON de sortie (flux + statistiques)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.input is None and args.dataset is None:
        parser.error("Spécifier --input <csv> (Scénario A) ou --dataset cwru_proxy (Scénario B)")

    # Flags modèle
    model_flags = 0
    if args.model == "ewc":
        model_flags = FRAME_FLAGS_EWC_MODE
    elif args.model == "ewc-int8":
        model_flags = FRAME_FLAGS_INT8_MODE
    elif args.model == "hdc":
        model_flags = FRAME_FLAGS_HDC_MODE

    scenario = "A_edge_spectrum" if args.input else "B_cwru_proxy"
    print(f"\nEdge Spectrum Demo — Scénario {scenario}")
    print(f"  Modèle : {args.model}  |  Mode : {'dry-run' if args.dry_run else 'UART'}")

    # Chargement
    print("\nChargement données...")
    X, y, segments = load_data(args)
    total = sum(n for _, n in segments)
    print(f"  {total} samples | {X.shape[1]} features | {len(segments)} tâches")
    for tid, (name, n) in enumerate(segments):
        print(f"    Tâche {tid} ({name}) : {n} samples")

    # Streaming
    print("\nDémarrage streaming...")
    if args.dry_run or args.port is None:
        results = _stream_demo_dry_run(
            X, y, segments,
            request_update=args.update,
            consolidate=args.consolidate,
            verbose=args.verbose,
        )
    else:
        results = _stream_demo_uart(
            port=args.port,
            baud=args.baud,
            X=X, y=y,
            segments=segments,
            rate_hz=args.rate_hz,
            request_update=args.update,
            consolidate=args.consolidate,
            verbose=args.verbose,
            model_flags=model_flags,
            protocol_version=args.protocol_version,
        )

    # Statistiques
    stats = _compute_demo_stats(results)
    stats["scenario"] = scenario
    stats["model"] = args.model
    stats["mode"] = "dry-run" if args.dry_run else "uart"
    stats["n_tasks"] = len(segments)
    stats["per_task"] = [
        {
            "task_id": i,
            "task_name": name,
            "n_samples": n,
            "accuracy": round(
                sum(r["pred"] == r["true"] for r in results if r["task_id"] == i)
                / max(sum(1 for r in results if r["task_id"] == i), 1),
                4,
            ),
        }
        for i, (name, n) in enumerate(segments)
    ]

    _print_summary(results, stats, scenario)

    # Sauvegarde
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stats": stats, "stream": results}
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSauvegardé : {out_path}")


if __name__ == "__main__":
    main()
