#!/usr/bin/env python3
"""run_sprint38_board.py — Board P0/P1 : références frozen / always (S3804).

Bornes de l'étude d'économie de mise à jour autonome (Sprint 38) mesurées sur la
**NUCLEO-F439ZI réelle**. P0 (`frozen`) et P1 (`always`) utilisent le firmware **par
défaut** (sans ``-DEWC_AUTO_UPDATE``) → déclencheur UART historique. Le gate embarqué
(S3803) n'est exercé que par les cellules gated (S3805, hors périmètre ici).

Réutilise les helpers éprouvés de ``run_sprint36_board.py`` / ``run_feature_condition_board.py``
(``_bss_bytes``, ``_pc_pred_ewc``, ``train_maha_board``) et le streaming in-process de
``sensor_stream.py`` (aucune modif de ces scripts).

Pour chaque ``(policy ∈ {frozen, always} × dataset × init_mode)`` :
  1. Charge le **checkpoint PC** ``exp_S38_PC_{policy}_{ds}_{init}/checkpoints/ewc_head.pt``
     (produit par run_sprint38_pc, S3802) ⇒ modèle flashé == modèle PC.
  2. Entraîne un Maha de référence (mêmes arrays) pour des ``model_weights.h`` cohérents en dim.
  3. ``export_weights_c.py --mahal --ewc-head`` → headers C.
  4. ``make clean`` puis ``make EWC_IN=5 MAHA_DIM=5 all`` ; ``.bss`` lu ; ``make flash``.

  --policy frozen (P0) : stream **sans --update** → latence **inférence seule** (DWT) ;
    parité **exacte** pred_board vs ``_pc_pred_ewc`` (même checkpoint).
  --policy always (P1) : stream **avec --update** (flag UART) → latence **inférence + SGD** ;
    ``n_updates == n_samples`` ; miroir PC (rejoue 1 pas SGD/échantillon depuis le même
    checkpoint, dans l'ordre streamé) → parité **approchée** (float32 board ≠ float64 PC).

Usage :
    python scripts/run_sprint38_board.py --policy frozen --dataset monitoring --port /dev/ttyACM0
    python scripts/run_sprint38_board.py --policy always --dataset pronostia --init-mode pretrained
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import (  # noqa: E402
    _bss_bytes,
    _pc_pred_ewc,
    train_maha_board,
)
from src.evaluation.drift_detector import SlidingWindowDriftDetector  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: E402
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
GAP2_LATENCY_US = 100_000   # 100 ms (Gap 2)
DEFAULT_CONFIG = "configs/sprint38_autonomous_update.yaml"
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")
GATED_POLICIES = ("gated_truelabel", "gated_pseudolabel")
INIT_MODES = ("pretrained", "scratch")
# Codes du verdict — DOIVENT correspondre à l'enum firmware DriftVerdict
# (drift_detector.h : DRIFT_NORMAL=0, DRIFT_FAULT=1, DRIFT_DRIFT=2).
VERDICT_CODE = {"NORMAL": 0, "FAULT": 1, "DRIFT": 2}
CODE_VERDICT = {v: k for k, v in VERDICT_CODE.items()}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def build_and_flash(k: int, X: np.ndarray, pc_ckpt: Path, exp_dir: Path) -> int:
    """Export → build → flash à la dim k (firmware par défaut). Retourne .bss (B)."""
    maha_ckpt = train_maha_board(X, exp_dir)   # Maha de référence (cohérence dims ; non streamé)
    if _run([sys.executable, "scripts/export_weights_c.py",
             "--mahal", str(maha_ckpt), "--ewc-head", str(pc_ckpt)]).returncode != 0:
        raise RuntimeError("export_weights_c échec")
    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    if _run(["make", "-C", str(FW_DIR), *make_dims, "all"]).returncode != 0:
        raise RuntimeError("make échec")
    bss = _bss_bytes()
    if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


def _fit_enrollment_maha(X: np.ndarray, y: np.ndarray, n_enr: int, exp_dir: Path) -> Path:
    """Maha d'enrôlement = miroir EXACT de ``run_sprint38_pc`` (welford, P95 sur les
    ``n_enr`` premiers échantillons SAINS). C'est ce détecteur qui est flashé dans le gate
    (``g_detector``) → parité du gate board↔PC par construction. ≠ ``train_maha_board``
    (qui fit sur tout X, threshold global) qui ne convient PAS au gate."""
    X_healthy = X[y == 0][:n_enr]
    if len(X_healthy) < 2:
        raise RuntimeError(f"Pas assez d'échantillons sains pour l'enrôlement ({len(X_healthy)}).")
    maha = MahalanobisDetector({"cl_strategy": "welford", "anomaly_percentile": 95})
    maha.fit_task(X_healthy, task_id=0)
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "mahalanobis_task0.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(maha, f)
    print(f"  [maha-enrol] μ={maha.mu_.shape} seuil={maha.threshold_:.4f} "
          f"(n_healthy={len(X_healthy)}) → {ckpt}")
    return ckpt


def build_and_flash_gated(k: int, X: np.ndarray, y: np.ndarray, n_enr: int,
                          pc_ckpt: Path, drift_json: Path, exp_dir: Path,
                          pseudo: bool) -> int:
    """Export (maha enrôlement + tête EWC + seuils du gate) → build ``-DEWC_AUTO_UPDATE``
    [+``-DGATE_PSEUDO_LABEL``] → flash. Retourne .bss (B)."""
    maha_ckpt = _fit_enrollment_maha(X, y, n_enr, exp_dir)
    if _run([sys.executable, "scripts/export_weights_c.py",
             "--mahal", str(maha_ckpt), "--ewc-head", str(pc_ckpt),
             "--drift-thresholds", str(drift_json)]).returncode != 0:
        raise RuntimeError("export_weights_c (gated) échec")
    extra = "-DEWC_AUTO_UPDATE" + (" -DGATE_PSEUDO_LABEL" if pseudo else "")
    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    if _run(["make", "-C", str(FW_DIR), f"EXTRA_CFLAGS={extra}", *make_dims, "all"]).returncode != 0:
        raise RuntimeError("make (gated) échec")
    bss = _bss_bytes()
    if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


def _pc_gate_replay(maha_ckpt: Path, drift_json: Path, feats: np.ndarray,
                    pseudo: bool) -> list[str]:
    """Rejoue le gate (maha enrôlement + SlidingWindowDriftDetector) sur les features
    streamées DANS L'ORDRE BOARD → verdict_pc par échantillon (référence de parité S3806).
    Réplique les mises à jour maha de la politique : P3 (pseudo) adapte le normal sur DRIFT
    (``maha.partial_fit``) ; P2 ne touche pas maha. Mêmes seuils exportés que la board."""
    with open(maha_ckpt, "rb") as f:
        maha = pickle.load(f)
    th = json.loads(Path(drift_json).read_text())
    drift = SlidingWindowDriftDetector(window_size=int(th["window_size"]), drift_ratio=float(th["drift_ratio"]))
    drift.fault_threshold = float(th["fault_threshold"])
    drift.drift_threshold = float(th["drift_threshold"])
    verdicts: list[str] = []
    for x in feats:
        score = float(maha.anomaly_score(x[None, :])[0])
        v = drift.update(score)
        verdicts.append(v)
        if pseudo and v == "DRIFT":
            maha.partial_fit(x)   # P3 : adapte le normal (lockstep avec le firmware)
    return verdicts


def _pc_always_mirror(pc_ckpt: Path, feats: np.ndarray, trues: np.ndarray,
                      k: int, ewc_lr: float, ewc_lambda: float) -> np.ndarray:
    """Rejoue la trajectoire `always` sur PC : pour chaque échantillon (dans l'ordre
    EXACT streamé par la board) : prédire → 1 pas SGD (vrai label). Démarre du même
    checkpoint que la board → parité approchée (float32 board ≠ float64 PC)."""
    sd = torch.load(pc_ckpt, map_location="cpu")["model_state_dict"]
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16], ewc_lambda=ewc_lambda)
    model.load_state_dict(sd)
    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    preds = []
    for x, lab in zip(feats, trues):
        xb = torch.tensor(x[None, :], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            preds.append(int(model(xb).argmax(dim=1).item()))
        model.train()
        optimizer.zero_grad()
        yb = torch.tensor([int(lab)], dtype=torch.long)
        (criterion(model(xb), yb) + model.ewc_penalty()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return np.array(preds)


def run_cell(policy: str, dataset: str, init_mode: str, cfg: dict, args,
             ewc_lr: float, ewc_lambda: float) -> dict:
    if policy in GATED_POLICIES:
        return run_cell_gated(policy, dataset, init_mode, cfg, args)
    print(f"\n{'='*72}\n=== BOARD {policy.upper()}  dataset={dataset}  init={init_mode}  ===\n{'='*72}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])
    condition = cfg["condition"]

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_ckpt = (EXPERIMENTS / f"exp_S38_PC_{policy}_{dataset}_{init_mode}"
               / "checkpoints" / "ewc_head.pt")
    if not pc_ckpt.exists():
        raise FileNotFoundError(f"{pc_ckpt} absent — lancer scripts/run_sprint38_pc.py d'abord")

    exp_id = f"exp_S38_board_{policy}_{dataset}_{init_mode}"
    exp_dir = EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    bss = build_and_flash(k, X, pc_ckpt, exp_dir)

    request_update = (policy == "always")
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=n_tasks,
        rate_hz=float(cfg["uart"]["rate_hz"]), request_update=request_update,
        verbose=args.verbose, protocol_version=int(cfg["uart"]["proto"]),
        model_flags=ss.FRAME_FLAGS_EWC_MODE,
    )
    stats = ss._compute_stats(results)
    feats = np.array([r["features"] for r in results], dtype=np.float32)
    board_pred = np.array([int(r["pred"]) for r in results])
    board_true = np.array([int(r["true"]) for r in results])
    lat = stats.get("latency_p50_us")
    f1 = compute_fault_f1(board_true, board_pred)

    result = {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "policy": policy, "init_mode": init_mode,
        "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "always (--update)" if request_update else "frozen (sans --update)",
        "online_accuracy": stats.get("accuracy"),
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"], "metric_value": f1["f1_faulty"],
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "bss_bytes": bss, "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "n_updates": (len(results) if request_update else 0),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
    }

    if policy == "frozen":
        # Parité EXACTE : board (gelé) == _pc_pred_ewc sur les mêmes features.
        ref_pred = _pc_pred_ewc(pc_ckpt, feats)
        n_mismatch = int((board_pred != ref_pred).sum())
        result.update({
            "parity_class": "exact",
            "parity_ok": bool(n_mismatch == 0),
            "parity_rate": float((board_pred == ref_pred).mean()),
            "parity_mismatch_count": n_mismatch, "n_compared": len(results),
        })
        print(f"  k={k} .bss={bss} lat_p50={lat}µs parity={result['parity_ok']} "
              f"({result['parity_rate']:.4f}) F1={f1['f1_faulty']:.3f} → {exp_dir}/results.json")
    else:
        # Parité APPROCHÉE : miroir PC rejoue le SGD par échantillon (ordre streamé).
        mirror_pred = _pc_always_mirror(pc_ckpt, feats, board_true, k, ewc_lr, ewc_lambda)
        result.update({
            "parity_class": "approx",
            "parity_rate": float((board_pred == mirror_pred).mean()),
            "parity_mismatch_count": int((board_pred != mirror_pred).sum()),
            "n_compared": len(results),
        })
        print(f"  k={k} .bss={bss} lat_inf+SGD_p50={lat}µs n_updates={result['n_updates']} "
              f"parity~{result['parity_rate']:.4f} F1={f1['f1_faulty']:.3f} → {exp_dir}/results.json")

    # Dump par échantillon (parité fine S3806).
    samples = [{"idx": i, "true": int(board_true[i]), "pred_board": int(board_pred[i]),
                "confidence": results[i].get("confidence")} for i in range(len(results))]
    (exp_dir / "board_samples.json").write_text(json.dumps(samples))
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def run_cell_gated(policy: str, dataset: str, init_mode: str, cfg: dict, args) -> dict:
    """P2/P3 autonomes : le gate embarqué (-DEWC_AUTO_UPDATE) décide des MAJ. On streame
    SANS --update (le bit UART n'a plus d'effet) ; le vrai label reste transmis (SGD P2 +
    scoring). Le firmware renvoie le verdict (snap.auroc) et le compteur cumulé de MAJ
    (snap.forgetting) — réinterprétation S3805, wire format V3 inchangé."""
    pseudo = (policy == "gated_pseudolabel")
    print(f"\n{'='*72}\n=== BOARD {policy.upper()}  dataset={dataset}  init={init_mode}  "
          f"(gate {'pseudo-label' if pseudo else 'true-label'})  ===\n{'='*72}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])
    condition = cfg["condition"]
    n_enr = int(cfg["enrollment"]["n_samples"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_dir = EXPERIMENTS / f"exp_S38_PC_{policy}_{dataset}_{init_mode}"
    pc_ckpt = pc_dir / "checkpoints" / "ewc_head.pt"
    drift_json = pc_dir / "drift_thresholds.json"
    if not pc_ckpt.exists() or not drift_json.exists():
        raise FileNotFoundError(f"{pc_dir} incomplet (ckpt/seuils) — lancer run_sprint38_pc.py d'abord")

    exp_id = f"exp_S38_board_{policy}_{dataset}_{init_mode}"
    exp_dir = EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    bss = build_and_flash_gated(k, X, y, n_enr, pc_ckpt, drift_json, exp_dir, pseudo)

    # Stream SANS --update : le gate firmware tranche seul.
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=n_tasks,
        rate_hz=float(cfg["uart"]["rate_hz"]), request_update=False,
        verbose=args.verbose, protocol_version=int(cfg["uart"]["proto"]),
        model_flags=ss.FRAME_FLAGS_EWC_MODE,
    )
    stats = ss._compute_stats(results)
    feats = np.array([r["features"] for r in results], dtype=np.float32)
    board_pred = np.array([int(r["pred"]) for r in results])
    board_true = np.array([int(r["true"]) for r in results])
    lats = np.array([float(r["latency_us"]) for r in results])
    f1 = compute_fault_f1(board_true, board_pred)

    # ── Décodage du gate depuis le snapshot réinterprété (S3805) ──
    verdict_codes = [int(round(float(r["auroc"]))) for r in results]
    verdict_board = [CODE_VERDICT.get(c, "NORMAL") for c in verdict_codes]
    n_updates = int(round(float(results[-1]["forgetting"]))) if results else 0
    update_rate = (n_updates / len(results)) if results else None

    # gate_overhead_us : surcoût/échantillon sur les NORMAL (gate seul, pas de SGD)
    # vs la latence inférence pure (frozen, S3804).
    frozen_dir = EXPERIMENTS / f"exp_S38_board_frozen_{dataset}_{init_mode}"
    frozen_res = json.loads((frozen_dir / "results.json").read_text()) if (frozen_dir / "results.json").exists() else None
    frozen_lat_p50 = frozen_res.get("latency_us_p50") if frozen_res else None
    normal_lats = lats[np.array(verdict_codes) == VERDICT_CODE["NORMAL"]]
    gate_overhead = (float(normal_lats.mean()) - float(frozen_lat_p50)
                     if (frozen_lat_p50 is not None and len(normal_lats)) else None)
    bss_delta = (bss - int(frozen_res["bss_bytes"])
                 if (frozen_res and frozen_res.get("bss_bytes") is not None) else None)

    # ── Parité verdicts board↔PC : rejoue le gate PC sur l'ordre board ──
    maha_ckpt = exp_dir / "checkpoints" / "mahalanobis_task0.pkl"
    verdict_pc = _pc_gate_replay(maha_ckpt, drift_json, feats, pseudo)
    verdict_match = [vb == vp for vb, vp in zip(verdict_board, verdict_pc)]
    verdict_parity_rate = float(np.mean(verdict_match)) if verdict_match else None
    # Parité prédiction (approchée : EWC mis à jour board ≠ PC).
    pred_pc = _pc_pred_ewc(pc_ckpt, feats)
    pred_parity_rate = float((board_pred == pred_pc).mean()) if len(results) else None

    lat_p50 = stats.get("latency_p50_us")
    result = {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "policy": policy, "init_mode": init_mode,
        "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": f"gated (sans --update, gate {'pseudo' if pseudo else 'true'}-label)",
        "online_accuracy": stats.get("accuracy"),
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"], "metric_value": f1["f1_faulty"],
        "af": None,   # forgetting board non reconstruit (état modèle non interrogeable) — cf. PC
        "n_updates": n_updates, "update_rate": update_rate,
        "mean_latency_us": float(lats.mean()) if len(lats) else None,
        "inference_latency_us": frozen_lat_p50,   # latence inférence pure (frozen, même cellule)
        "gate_overhead_us": gate_overhead,
        "latency_us_p50": lat_p50, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "bss_bytes": bss, "bss_delta_vs_default": bss_delta,
        "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "verdict_counts_board": {v: int(sum(1 for x in verdict_board if x == v))
                                 for v in VERDICT_CODE},
        "prediction_parity_rate": pred_parity_rate,
        "verdict_parity_rate": verdict_parity_rate,
        "verdict_mismatch_count": int(sum(1 for m in verdict_match if not m)),
        "n_compared": len(results),
        "gap2_latency_compliant": (lat_p50 is not None and lat_p50 < GAP2_LATENCY_US),
    }

    # Dump par échantillon enrichi (parité fine S3806).
    samples = [{"idx": i, "true": int(board_true[i]), "pred_board": int(board_pred[i]),
                "confidence": results[i].get("confidence"),
                "verdict_board": verdict_board[i], "verdict_pc": verdict_pc[i],
                "pred_pc": int(pred_pc[i])} for i in range(len(results))]
    (exp_dir / "board_samples.json").write_text(json.dumps(samples))
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"  k={k} .bss={bss} (Δ={bss_delta}) lat_mean={result['mean_latency_us']:.1f}µs "
          f"n_updates={n_updates}/{len(results)} (rate={update_rate:.3f}) "
          f"gate_ovh={gate_overhead}µs verdict_parity={verdict_parity_rate} "
          f"F1={f1['f1_faulty']:.3f} → {exp_dir}/results.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Board P0/P1 références frozen/always (S3804)")
    p.add_argument("--policy", choices=POLICIES, default=None)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", default=None)
    p.add_argument("--init-mode", choices=INIT_MODES, default=None)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    ewc_lr, ewc_lambda = float(base["EWC_LR"]), float(base["EWC_LAMBDA"])

    policies = [args.policy] if args.policy else list(POLICIES)
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    init_modes = [args.init_mode] if args.init_mode else cfg.get("init_modes", list(INIT_MODES))

    rows = []
    for im in init_modes:
        for d in datasets:
            for pol in policies:
                try:
                    rows.append(run_cell(pol, d, im, cfg, args, ewc_lr, ewc_lambda))
                except Exception as exc:  # noqa: BLE001 — cellule robuste
                    print(f"  [FAIL {pol}/{d}/{im}] {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}\nBoard S38 : {len(rows)} cellules.")
    for r in rows:
        print(f"  {r['exp_id']:46s} lat_p50={r.get('latency_us_p50')}µs "
              f"parity={r.get('parity_ok', r.get('parity_rate'))} F1={r.get('f1_faulty')}")


if __name__ == "__main__":
    main()
