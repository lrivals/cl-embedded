"""
sensor_sim.py — Simulateur de capteur : injecte des données dataset via UART vers le firmware STM32.

Protocole UART (little-endian, binaire) :
    Trame envoyée :
        [MAGIC 0xABCD : 2B] [N_FEATURES : 1B] [features : float32 × N] [label : 1B] [CRC8 : 1B]
    Réponse firmware :
        [pred_label : 1B] [confidence : float32 : 4B] [latency_us : uint32 : 4B]

Usage :
    # Dry-run (pas de board)
    python scripts/sensor_sim.py --dataset cwru --dry-run --n-samples 100

    # Board connectée
    python scripts/sensor_sim.py --dataset monitoring --port /dev/ttyUSB0 --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = 0xABCD
UART_TIMEOUT_S = 2.0
RESPONSE_FMT = "<BfI"  # pred_label (u8), confidence (f32), latency_us (u32)
RESPONSE_SIZE = struct.calcsize(RESPONSE_FMT)


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc


def build_frame(features: np.ndarray, label: int) -> bytes:
    n = len(features)
    header = struct.pack("<HB", MAGIC, n)
    feat_bytes = features.astype(np.float32).tobytes()
    lbl_byte = struct.pack("<B", label)
    payload = header + feat_bytes + lbl_byte
    return payload + struct.pack("<B", crc8(payload))


_MONITORING_CSV = Path(
    "data/raw/equipment_monitoring/Industrial_Equipment_Monitoring_Dataset"
    "/equipment_anomaly_data.csv"
)
_CWRU_PROCESSED = Path("data/processed/cwru_features.npz")
_CWRU_SUBSET_CFG = Path("configs/cwru_feature_subset.yaml")
_PRONOSTIA_BINARIES = Path("data/raw/Pronostia dataset/binaries")
_PRONOSTIA_SUBSET_CFG = Path("configs/pronostia_feature_subset.yaml")
_BATTERY_CSV = Path("data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv")
_BATTERY_SUBSET_CFG = Path("configs/battery_feature_subset.yaml")
_BATTERY_NORMALIZER_CFG = Path("configs/battery_normalizer.yaml")


def _load_monitoring() -> tuple[np.ndarray, np.ndarray]:
    """Charge le dataset monitoring : 4 features numériques + label faulty.

    Le firmware NUCLEO est câblé à ``PROTO_MAX_N=5`` features. Monitoring n'en a
    que 4 (temperature, pressure, vibration, humidity) → on ajoute une **5ᵉ feature
    synthétique nulle** (zéro-pad) pour rendre le flux compatible board sans
    toucher au firmware. Cette colonne ne porte aucune information (contribution
    nulle) : elle n'est pas un chiffre « inventé » mais une feature construite,
    à mentionner en note de figure pour HDC×monitoring.
    """
    from src.data.monitoring_dataset import NUMERIC_FEATURES, LABEL_COL, load_raw_dataset

    df = load_raw_dataset(_MONITORING_CSV)
    X = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)  # MEM: 4 col @ FP32
    # 5ᵉ feature synthétique (zéro-pad → firmware 5-feat)
    X = np.column_stack([X, np.zeros(len(X), dtype=np.float32)])  # MEM: 5 col @ FP32
    y = df[LABEL_COL].to_numpy(dtype=np.int64)
    return X, y


def _load_cwru() -> tuple[np.ndarray, np.ndarray]:
    """Charge les features CWRU prétraitées (npz) si disponibles, sinon CSV + sélection 9→5."""
    if _CWRU_PROCESSED.exists():
        data = np.load(_CWRU_PROCESSED)
        return data["X"].astype(np.float32), data["y"].astype(np.int64)

    import yaml
    from src.data.cwru_dataset import CWRUDataset
    from pathlib import Path as _Path

    subset = yaml.safe_load(_CWRU_SUBSET_CFG.read_text())
    indices = subset["feature_indices"]  # [3, 4, 6, 7, 8] = sd, rms, kurtosis, crest, form

    cwru_csv = _Path("data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv")
    ds = CWRUDataset(cwru_csv)
    return ds.X[:, indices].astype(np.float32), ds.y.astype(np.int64)


def _load_pronostia() -> tuple[np.ndarray, np.ndarray]:
    """Charge Pronostia (3 conditions concaténées) et applique la sélection 13→5 features."""
    import yaml
    from src.data.pronostia_dataset import load_condition_features, N_CONDITIONS

    subset = yaml.safe_load(_PRONOSTIA_SUBSET_CFG.read_text())
    indices = subset["feature_indices"]  # [1, 2, 4, 8, 12]

    all_X, all_y = [], []
    for cond in range(1, N_CONDITIONS + 1):
        X_cond, y_cond = load_condition_features(_PRONOSTIA_BINARIES, condition=cond)
        all_X.append(X_cond[:, indices].astype(np.float32))
        all_y.append(y_cond.astype(np.int64))
    return np.concatenate(all_X), np.concatenate(all_y)


def _load_battery() -> tuple[np.ndarray, np.ndarray]:
    """Charge Battery RUL z-scoré et applique la sélection 7→5 features (board).

    Features normalisées via ``configs/battery_normalizer.yaml`` (grande dynamique
    brute → stabilité Mahalanobis), puis sous-ensemble 5 features
    (``configs/battery_feature_subset.yaml``). Le label ``faulty`` utilise le seuil
    RUL par défaut ; le balayage S3205 ré-étiquette côté entraîneur board.
    """
    import yaml

    from src.data.battery_dataset import (
        FEATURE_COLUMNS,
        load_battery_normalizer,
        load_raw_dataset,
        normalize_features,
    )

    subset = yaml.safe_load(_BATTERY_SUBSET_CFG.read_text())
    indices = subset["feature_indices"]  # [0, 1, 2, 4, 5]

    df = load_raw_dataset(_BATTERY_CSV)
    normalizer = load_battery_normalizer(_BATTERY_NORMALIZER_CFG)
    df = normalize_features(df, normalizer)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, indices]
    y = df["faulty"].to_numpy(dtype=np.int64)
    return X, y


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge un dataset Phase 1 pour la simulation.

    Parameters
    ----------
    name : str
        "cwru", "monitoring", "pronostia" ou "battery"

    Returns
    -------
    X : np.ndarray [N, n_features]
    y : np.ndarray [N]
    """
    loaders = {
        "monitoring": _load_monitoring,
        "cwru": _load_cwru,
        "pronostia": _load_pronostia,
        "battery": _load_battery,
    }
    if name not in loaders:
        raise ValueError(f"Dataset inconnu : {name}. Choisir parmi {list(loaders)}")
    return loaders[name]()


def stream_samples(
    X: np.ndarray, y: np.ndarray, n_samples: int
) -> Iterator[tuple[np.ndarray, int]]:
    indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    for idx in indices:
        yield X[idx], int(y[idx])


def dry_run(
    X: np.ndarray, y: np.ndarray, n_samples: int, verbose: bool
) -> dict:
    """Valide le protocole en loopback sans board."""
    results = []
    for i, (features, label) in enumerate(stream_samples(X, y, n_samples)):
        frame = build_frame(features, label)
        # Loopback : on relit la trame pour vérifier le CRC
        payload, recv_crc = frame[:-1], frame[-1]
        assert crc8(payload) == recv_crc, f"CRC mismatch à l'échantillon {i}"
        results.append({"true": label, "pred": label})  # loopback → pred = true
        if verbose:
            print(f"[{i}/{n_samples}] features={features[:3].tolist()} label={label} → OK")
    acc = sum(r["pred"] == r["true"] for r in results) / len(results)
    return {"mode": "dry-run", "n_samples": len(results), "accuracy": acc, "crc_errors": 0}


def run_uart(
    port: str,
    baud: int,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    verbose: bool,
) -> dict:
    """Envoie les trames via UART et collecte les réponses firmware."""
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial requis : pip install pyserial")

    results = []
    crc_errors = 0

    import time as _time

    with serial.Serial(port, baud, timeout=UART_TIMEOUT_S,
                       dsrdtr=False, rtscts=False) as ser:
        # Reset DTR bref → NRST pour remettre le firmware dans un état propre
        ser.dtr = True
        _time.sleep(0.05)
        ser.dtr = False
        # Attendre que le firmware finisse hw_info_print + pipeline_init (~300ms)
        _time.sleep(0.5)
        ser.reset_input_buffer()   # vide les bytes hw_info accumulés au boot
        for i, (features, label) in enumerate(stream_samples(X, y, n_samples)):
            frame = build_frame(features, label)

            # Vérification CRC locale avant envoi
            if crc8(frame[:-1]) != frame[-1]:
                crc_errors += 1
                continue

            ser.write(frame)
            raw = ser.read(RESPONSE_SIZE)
            if len(raw) != RESPONSE_SIZE:
                print(f"[WARN] Timeout à l'échantillon {i} ({len(raw)}/{RESPONSE_SIZE} octets)")
                continue

            pred_label, confidence, latency_us = struct.unpack(RESPONSE_FMT, raw)
            results.append({
                "true": label,
                "pred": pred_label,
                "confidence": confidence,
                "latency_us": latency_us,
            })
            if verbose:
                print(
                    f"[{i}/{n_samples}] true={label} pred={pred_label} "
                    f"conf={confidence:.3f} lat={latency_us}µs"
                )

    if not results:
        return {"mode": "uart", "n_samples": 0, "accuracy": 0.0, "crc_errors": crc_errors}

    acc = sum(r["pred"] == r["true"] for r in results) / len(results)
    latencies = [r["latency_us"] for r in results]
    return {
        "mode": "uart",
        "port": port,
        "baud": baud,
        "n_samples": len(results),
        "accuracy": acc,
        "crc_errors": crc_errors,
        "latency_mean_us": float(np.mean(latencies)),
        "latency_p99_us": float(np.percentile(latencies, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulateur de capteur UART pour firmware STM32")
    parser.add_argument("--dataset", choices=["cwru", "monitoring", "pronostia", "battery"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Loopback, pas de board nécessaire")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=str, help="Fichier JSON pour les résultats")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Chargement du dataset '{args.dataset}'...")
    X, y = load_dataset(args.dataset)
    print(f"  {len(X)} échantillons, {X.shape[1]} features")

    if args.dry_run:
        stats = dry_run(X, y, args.n_samples, args.verbose)
    else:
        stats = run_uart(args.port, args.baud, X, y, args.n_samples, args.verbose)

    print(f"\n--- Résultats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2))
        print(f"\nRésultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
