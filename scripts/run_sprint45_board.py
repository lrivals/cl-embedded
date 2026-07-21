#!/usr/bin/env python3
"""
scripts/run_sprint45_board.py — Sprint 45 (S4503) : driver board des détecteurs de drift.

Par cellule ``(détecteur, dataset)`` :
  1. charge le ``DriftDataset`` (même loader / enrôlement / ``freeze_zscore`` que la grille S44) ;
  2. calibre les paramètres sur le segment d'enrôlement (miroir exact PC → parité par construction)
     et entraîne le modèle de référence board (tête EWC pour le flux d'erreur supervisé ;
     Mahalanobis d'enrôlement pour le score des détecteurs non-supervisés) ;
  3. ``export_weights_c.py [--ewc-head] [--mahal] --drift-methods`` → headers C générés ;
  4. ``make EXTRA_CFLAGS="-DDRIFT_DETECT -DDRIFT_METHOD=<id>"`` (dims par modèle) → ``.bss`` → flash ;
  5. **streame le split test DANS L'ORDRE CHRONOLOGIQUE** (le drift est ordonné — on n'utilise pas
     le shuffle de ``sensor_stream._stream_uart``), sans ``--update``, protocole v3, et récupère le
     **verdict** (``snap.auroc`` réinterprété, S4502) + latence DWT par échantillon.

Écrit ``experiments/exp_S45_board_{detector}_{dataset}/`` : ``results.json`` (métriques de détection
vs vérité-terrain, latences, ``.bss``, honnêteté N/A) + ``board_samples.json`` (features/pred/true/
verdict par échantillon, consommé par ``board_pc_parity45.py``).

Règles projet : seed 42 ; ``metric_value=null`` + ``na_reason`` si pas de vérité-terrain ponctuelle ;
``sensor_stream.py`` **inchangé** au niveau wire (verdict via champ snapshot réinterprété).

Usage
-----
    python scripts/run_sprint45_board.py --detector page_hinkley --dataset gas_sensor_drift \
        --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.sensor_stream as ss  # noqa: E402
from src.data import DRIFT_CONFIGS, DRIFT_LOADERS  # noqa: E402
from src.data.drift_dataset import DriftDataset, freeze_zscore  # noqa: E402
from src.evaluation.drift_metrics import (  # noqa: E402
    alarms_from_verdicts,
    compute_drift_metrics,
)
from src.utils.reproducibility import set_seed  # noqa: E402

FW_DIR = ROOT / "firmware" / "stm32f4_blink"
EXPERIMENTS = ROOT / "experiments"
DEFAULT_CONFIG = ROOT / "configs" / "sprint44_drift_detection.yaml"

# détecteur → (macro -DDRIFT_METHOD, famille, requires_label)
METHOD_SPEC = {
    "page_hinkley": ("DRIFT_PAGE_HINKLEY", "supervised", True),
    "ddm":          ("DRIFT_DDM", "supervised", True),
    "psi":          ("DRIFT_PSI", "unsupervised", False),
}
# code snapshot → nom de verdict (miroir enum DriftMethodVerdict, S4502)
CODE_VERDICT = {0: "NORMAL", 1: "WARNING", 2: "DRIFT"}
TOLERANCE_DEFAULT = 200


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _binarize_labels(y: np.ndarray, y_enroll: np.ndarray) -> np.ndarray:
    """« normal (classe dominante à l'enrôlement) vs reste » — miroir run_sprint44_pc."""
    y = np.asarray(y)
    uniq = np.unique(y)
    if uniq.size <= 2:
        return (y != uniq[0]).astype(np.int64) if uniq.size == 2 and set(uniq) != {0, 1} else y
    vals, counts = np.unique(y_enroll, return_counts=True)
    normal_cls = vals[int(np.argmax(counts))]
    return (y != normal_cls).astype(np.int64)


# ── Modèles de référence board (miroir exact PC) ────────────────────────────────────────

def train_ewc_head(X_enr: np.ndarray, y_enr: np.ndarray, exp_dir: Path, seed: int) -> Path:
    """Tête EWC de référence (EWCMlpMulticlass k→32→16→2), entraînée sur l'enrôlement.

    Architecture bit-pour-bit équivalente à ``ewc_forward`` (parité board↔PC, S3205/S36).
    """
    import torch  # noqa: PLC0415
    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: PLC0415

    set_seed(seed)
    k = X_enr.shape[1]
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16])
    Xt = torch.tensor(X_enr, dtype=torch.float32)
    yt = torch.tensor(y_enr, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(200):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "ewc_head.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    return ckpt


def fit_enrollment_maha(X_enr: np.ndarray, exp_dir: Path) -> Path:
    """Détecteur Mahalanobis figé sur l'enrôlement (score des détecteurs non-supervisés)."""
    import pickle  # noqa: PLC0415
    from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector  # noqa: PLC0415

    det = MahalanobisDetector({"cl_strategy": "welford", "anomaly_percentile": 95})
    det.fit_task(X_enr, task_id=0)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "mahalanobis_task0.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(det, f)
    return ckpt


def calibrate_psi_params(scores_enr: np.ndarray, cfg_psi: dict) -> dict:
    """Bornes de bacs + distribution de référence PSI figées sur les scores d'enrôlement.

    Miroir exact de PSI.set_params_from_reference (psi.py) : edges = linspace(min,max,bins+1),
    ref_probs = counts/sum + eps. Exportées telles quelles → parité par construction.
    """
    eps = 1e-8
    bins = int(cfg_psi.get("bins", 10))
    ref = np.asarray(scores_enr, dtype=np.float64).ravel()
    lo, hi = float(ref.min()), float(ref.max())
    if hi - lo < eps:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(ref, bins=edges)
    ref_probs = counts / max(counts.sum(), 1) + eps
    return {
        "bins": bins,
        "block_size": int(cfg_psi.get("block_size", 200)),
        "threshold": float(cfg_psi.get("psi_threshold", 0.2)),
        "edges": edges.tolist(),
        "ref_probs": ref_probs.tolist(),
    }


# ── Streaming chronologique (l'ordre du drift est significatif) ─────────────────────────

def stream_ordered(port: str, baud: int, X: np.ndarray, y: np.ndarray,
                   flags: int, rate_hz: float, verbose: bool = False) -> list[dict]:
    """Streame X/y DANS L'ORDRE, une trame v3 par échantillon (pas de shuffle par tâche).

    Réutilise les primitives wire de sensor_stream (build_frame_v2 / parse_response) → aucune
    modification de sensor_stream.py. Retourne un dict par échantillon (pred, verdict via auroc…).
    """
    import serial  # noqa: PLC0415

    results: list[dict] = []
    interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    with serial.Serial(port, baud, timeout=ss.UART_TIMEOUT_S, dsrdtr=False, rtscts=False) as ser:
        ser.dtr = True; time.sleep(0.05); ser.dtr = False; time.sleep(0.5)
        ser.reset_input_buffer()
        for i in range(len(X)):
            feats = np.asarray(X[i], dtype=np.float32)
            label = int(y[i]) if y is not None else 0
            frame = ss.build_frame_v2(feats, label, task_id=0, ts_ms=i, flags=flags)
            ser.write(frame)
            data = ser.read(ss.RESPONSE_V3_SIZE)
            if len(data) != ss.RESPONSE_V3_SIZE:
                raise RuntimeError(f"trame {i} : réponse tronquée ({len(data)} B) — CRC/timeout")
            resp = ss.parse_response(data)
            resp["idx"] = i
            resp["true"] = label
            resp["features"] = feats.tolist()
            results.append(resp)
            if verbose and i % 1000 == 0:
                print(f"  [{i}/{len(X)}] pred={resp['pred']} verdict={int(round(resp['auroc']))}")
            if interval_s:
                time.sleep(interval_s)
    return results


# ── Build / flash ───────────────────────────────────────────────────────────────────────

def build_and_flash(k: int, method_macro: str, family: str, flash: bool) -> int:
    """make clean → make (dims=k, -DDRIFT_DETECT -DDRIFT_METHOD) → .bss → [flash]. Retourne .bss.

    Dims minimales : le chemin EWC est toujours utilisé (pred / flux d'erreur) → EWC_IN=k.
    MAHA_DIM=k seulement pour les non-supervisés (PSI branché sur maha_score). HDC/TinyOL
    gardent leur dim par défaut (chemins non exécutés en mode EWC) → pas de gonflement RAM
    (la projection HDC k·HDC_DIM déborderait la SRAM à k élevé).
    """
    dims = [f"EWC_IN={k}"]
    if family == "unsupervised":
        dims.append(f"MAHA_DIM={k}")
    if k > 16:
        dims.append(f"PROTO_MAX_N={k}")
    extra = f"-DDRIFT_DETECT -DDRIFT_METHOD={method_macro}"
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    r = _run(["make", "-C", str(FW_DIR), f"EXTRA_CFLAGS={extra}", *dims, "all"])
    if r.returncode != 0:
        raise RuntimeError(f"make échec :\n{r.stderr[-2000:]}")
    bss = _read_bss()
    if flash:
        rf = _run(["make", "-C", str(FW_DIR), "flash"])
        if rf.returncode != 0:
            raise RuntimeError(f"flash échec :\n{rf.stderr[-2000:]}")
    return bss


def _read_bss() -> int:
    elf = FW_DIR / "build" / "stm32f4_blink.elf"
    out = _run(["arm-none-eabi-size", str(elf)]).stdout.strip().splitlines()
    return int(out[-1].split()[2])  # colonne bss


# ── Cellule ─────────────────────────────────────────────────────────────────────────────

def run_cell(detector: str, dataset: str, cfg: dict, args) -> dict:
    method_macro, family, requires_label = METHOD_SPEC[detector]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    print(f"\n{'='*72}\n=== BOARD S45  {detector}  ×  {dataset}  (famille={family})  ===\n{'='*72}")

    d: DriftDataset = DRIFT_LOADERS[dataset](DRIFT_CONFIGS[dataset])
    X, y, drift_points = d.X, d.y, d.drift_points
    n_samples = X.shape[0]
    n_enroll = max(2, int(cfg.get("calibration", {}).get("enrollment_fraction", 0.25) * n_samples))
    k = int(X.shape[1])

    # Normalisation z-score figée sur l'enrôlement (le drift reste visible — S43).
    X_norm, _, _ = freeze_zscore(X, (0, n_enroll))
    X_enr = X_norm[:n_enroll]

    exp_id = f"exp_S45_board_{detector}_{dataset}"
    exp_dir = EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Référence + calibration + export headers ────────────────────────────────────────
    export_cmd = [sys.executable, "scripts/export_weights_c.py"]
    methods_params: dict = {}
    y_bin = None
    if requires_label:
        if y is None:
            return _na_result(exp_id, detector, dataset, family, k, n_samples, n_enroll,
                              drift_points, "supervisé : labels absents pour ce dataset")
        y_bin = _binarize_labels(y, y[:n_enroll])
        ewc_ckpt = train_ewc_head(X_enr, y_bin[:n_enroll], exp_dir, seed)
        export_cmd += ["--ewc-head", str(ewc_ckpt)]
        sec = dict(cfg.get(detector, {}))
        methods_params[detector] = {  # seuils = config S44 (miroir exact PC)
            "delta": sec.get("delta"), "lambda": sec.get("lambda_"),
            "warning_level": sec.get("warning_level"), "drift_level": sec.get("drift_level"),
            "min_instances": sec.get("min_instances", 30),
        }
    else:  # non-supervisé : score Maha + calibration PSI
        maha_ckpt = fit_enrollment_maha(X_enr, exp_dir)
        export_cmd += ["--mahal", str(maha_ckpt)]
        import pickle  # noqa: PLC0415
        with open(maha_ckpt, "rb") as f:
            maha = pickle.load(f)
        scores_enr = maha.anomaly_score(X_enr)
        methods_params["psi"] = calibrate_psi_params(scores_enr, dict(cfg.get("psi", {})))

    methods_json = exp_dir / "drift_methods_params.json"
    methods_json.write_text(json.dumps(methods_params, indent=2))
    export_cmd += ["--drift-methods", str(methods_json)]
    if _run(export_cmd).returncode != 0:
        raise RuntimeError("export_weights_c échec")

    # ── Build / flash ───────────────────────────────────────────────────────────────────
    bss = build_and_flash(k, method_macro, family, flash=not args.no_flash)

    if args.no_stream:
        print("[S45] --no-stream : build/flash OK, streaming différé (métriques « à mesurer »).")
        return _pending_result(exp_id, detector, dataset, family, k, n_samples, n_enroll,
                               drift_points, bss)

    # ── Stream chronologique (sans --update : on mesure la détection) ────────────────────
    y_stream = y_bin if y_bin is not None else (y if y is not None else np.zeros(n_samples, int))
    results = stream_ordered(args.port, args.baud, X_norm, y_stream,
                             flags=ss.FRAME_FLAGS_EWC_MODE, rate_hz=float(args.rate_hz),
                             verbose=args.verbose)

    verdict_codes = [int(round(float(r["auroc"]))) for r in results]
    verdict_board = [CODE_VERDICT.get(c, "NORMAL") for c in verdict_codes]
    lats = np.array([float(r["latency_us"]) for r in results])

    # ── Métriques de détection vs vérité-terrain (N/A honnête si drift_points absents) ──
    if drift_points is None:
        drift_metrics = None
        metric_value = None
        na_reason = "vérité-terrain de drift ponctuelle absente (ex. electricity)"
    else:
        alarms = alarms_from_verdicts(verdict_board)
        drift_metrics = compute_drift_metrics(alarms, drift_points, n_samples, args.tolerance)
        metric_value = drift_metrics.get("f1")
        na_reason = None

    # ── Persistance par échantillon pour la parité (features telles que streamées) ──────
    board_samples = [{
        "idx": r["idx"], "true": int(r["true"]),
        "pred": int(r["pred"]), "verdict_code": verdict_codes[i],
        "verdict": verdict_board[i], "latency_us": float(r["latency_us"]),
        "features": r["features"],
    } for i, r in enumerate(results)]
    (exp_dir / "board_samples.json").write_text(json.dumps(board_samples))

    lat_arr = lats[np.isfinite(lats)]
    result = {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "detector": detector, "dataset": dataset,
        "family": family, "requires_label": requires_label,
        "date": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "n_samples": n_samples, "n_enroll": n_enroll, "n_features": k,
        "drift_points": list(drift_points) if drift_points else None,
        "tolerance": args.tolerance,
        "stream_mode": "chronologique (sans --update, proto v3, EWC path)",
        "verdict_counts_board": {v: int(sum(1 for x in verdict_board if x == v))
                                 for v in ("NORMAL", "WARNING", "DRIFT")},
        "drift_metrics": drift_metrics, "metric_value": metric_value, "na_reason": na_reason,
        "mean_latency_us": float(lat_arr.mean()) if len(lat_arr) else None,
        "latency_us_p50": float(np.percentile(lat_arr, 50)) if len(lat_arr) else None,
        "latency_us_p99": float(np.percentile(lat_arr, 99)) if len(lat_arr) else None,
        "gap2_ok": bool(len(lat_arr) and np.percentile(lat_arr, 99) < 100_000),
        "bss_bytes": bss,
    }
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"[S45] {exp_id} : verdicts={result['verdict_counts_board']} "
          f"metric={metric_value} lat_p50={result['latency_us_p50']}µs bss={bss} B")
    return result


def _base(exp_id, detector, dataset, family, k, n, n_enr, dp):
    return {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "detector": detector, "dataset": dataset,
        "family": family, "n_features": k, "n_samples": n, "n_enroll": n_enr,
        "drift_points": list(dp) if dp else None,
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def _na_result(exp_id, detector, dataset, family, k, n, n_enr, dp, reason) -> dict:
    r = _base(exp_id, detector, dataset, family, k, n, n_enr, dp)
    r.update({"metric_value": None, "na_reason": reason})
    (EXPERIMENTS / exp_id).mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS / exp_id / "results.json").write_text(json.dumps(r, indent=2))
    print(f"[S45] {exp_id} : N/A — {reason}")
    return r


def _pending_result(exp_id, detector, dataset, family, k, n, n_enr, dp, bss) -> dict:
    r = _base(exp_id, detector, dataset, family, k, n, n_enr, dp)
    r.update({"metric_value": "à mesurer", "na_reason": None, "bss_bytes": bss,
              "stream_mode": "différé (--no-stream)"})
    (EXPERIMENTS / exp_id / "results.json").write_text(json.dumps(r, indent=2))
    return r


def main() -> None:
    p = argparse.ArgumentParser(description="Driver board des détecteurs de drift (S4503)")
    p.add_argument("--detector", required=True, choices=list(METHOD_SPEC))
    p.add_argument("--dataset", required=True, choices=list(DRIFT_LOADERS))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--tolerance", type=int, default=TOLERANCE_DEFAULT)
    p.add_argument("--no-flash", action="store_true", help="build seulement (pas de flash)")
    p.add_argument("--no-stream", action="store_true", help="build/flash sans streamer")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_cell(args.detector, args.dataset, cfg, args)


if __name__ == "__main__":
    main()
