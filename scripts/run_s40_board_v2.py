#!/usr/bin/env python3
"""run_s40_board_v2.py — S4002 : validation board du kernel EWC INT8 v2 (récupération F1).

Preuve **matérielle** (NUCLEO-F439ZI réelle) que le kernel INT8 v2 calibré (S4001) récupère
la F1 que la PTQ « legacy » (v1) effondrait (Sprint 36 : F1 board 0.07–0.15 vs FP32 ≈ 0.92).
La récupération n'était jusqu'ici qu'**émulée PC** (Sprint 39) ; ce driver flashe le v2 et
mesure, **dans les conditions strictement identiques à exp_S36/exp_S39** (mêmes checkpoints,
seed=42, condition 5feat, même ordre de streaming), la F1 board, l'accord INT8↔FP32 et la
parité board↔émulateur PC.

Réutilise ``run_sprint36_board.py`` (squelette apparié, mêmes séquences) sans le modifier :
``build/flash`` et ``_pc_online_mirror`` importés ; l'émulateur ``int8_c_emulation`` fournit la
référence PC **bit-exacte** du kernel v2 (parité frozen attendue = 1.000).

Grille (12 cellules) :
  schemes {per_channel, q15, int8_legacy} × datasets {pronostia, monitoring}
  × protocoles {frozen, online}
  → experiments/exp_S40_board_v2/results_{scheme}_{dataset}_{proto}.json

Sélection du kernel **à la compilation** (le nibble protocole 0x40 est saturé) :
  - per_channel : make ... EXTRA_CFLAGS="-DEWC_INT8_V2"          (acc int32, scales par-canal)
  - q15         : make ... EXTRA_CFLAGS="-DEWC_INT8_V2 -DEWC_INT8_Q15"  (poids/act int16)
  - int8_legacy : make ... (aucun flag → v1, chemin 0x40 historique = A/B)
Dans les 3 cas le stream utilise le flag UART ``FRAME_FLAGS_INT8_MODE`` (0x40).

Règle « aucun chiffre inventé » : ``experiments/exp_S40_board_v2/`` n'est créé qu'au premier
stream réussi ; rien n'est écrit sans carte.

Usage :
    python scripts/run_s40_board_v2.py --scheme per_channel --dataset pronostia --proto frozen
    python scripts/run_s40_board_v2.py                       # grille complète (12 cellules)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import _bss_bytes, train_maha_board  # noqa: E402
from scripts.run_sprint36_board import (  # noqa: E402
    FW_DIR,
    GAP2_LATENCY_US,
    _ewc_weight_bytes,
    _pc_online_mirror,
)
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    QuantConfig,
    calibrate_activations,
    forward_quant,
    predict,
)
from src.utils.reproducibility import set_seed  # noqa: E402

EXPERIMENTS = Path("experiments")
OUT_DIR = EXPERIMENTS / "exp_S40_board_v2"
DEFAULT_CONFIG = "configs/sprint36_ewc_comparison.yaml"

# scheme → (flags de build, config émulateur PC, kernel). int8_legacy = v1 (aucun flag).
SCHEMES: dict[str, dict] = {
    "per_channel": {"cflags": ["-DEWC_INT8_V2"], "kernel": "v2",
                    "qcfg": QuantConfig.per_channel_int8, "ram_div": 4},
    "q15":         {"cflags": ["-DEWC_INT8_V2", "-DEWC_INT8_Q15"], "kernel": "v2",
                    "qcfg": QuantConfig.q15, "ram_div": 2},
    "int8_legacy": {"cflags": [], "kernel": "v1_legacy",
                    "qcfg": QuantConfig.legacy_c, "ram_div": 4},
}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _pc_ckpt(dataset: str, condition: str) -> Path:
    p = EXPERIMENTS / f"exp_S36_PC_{condition}_ewc_{dataset}" / "checkpoints" / "ewc_head.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} absent — lancer scripts/run_sprint36_pc.py d'abord")
    return p


def _load_head(pc_ckpt: Path) -> EWCHeadWeights:
    sd = torch.load(pc_ckpt, map_location="cpu")["model_state_dict"]
    return EWCHeadWeights.from_state_dict(sd)


def _emu_pred(w: EWCHeadWeights, feats: np.ndarray, scheme: str,
              act_max: dict[str, float]) -> np.ndarray:
    """Référence PC bit-exacte du kernel embarqué (parité par construction).

    ``act_max`` figé = mêmes bornes que le header exporté (``calibrate_activations`` sur le
    même lot condition, seed 42) ⇒ frozen v2 attendu parité 1.000 vs board.
    """
    cfg = SCHEMES[scheme]["qcfg"]()
    logits = forward_quant(w, feats, cfg, act_max=act_max)
    return predict(logits)


# ── Build/flash (sélection du schéma via EXTRA_CFLAGS) ───────────────────────

def build_and_flash_s40(scheme: str, dataset: str, condition: str, k: int,
                        X: np.ndarray, pc_ckpt: Path, exp_dir: Path,
                        flash: bool = True) -> int:
    """Export → build (schéma) → flash à la dim k. Retourne .bss (B)."""
    info = SCHEMES[scheme]
    maha_ckpt = train_maha_board(X, exp_dir)   # cohérence dims du build (non streamé)
    export_cmd = [sys.executable, "scripts/export_weights_c.py",
                  "--mahal", str(maha_ckpt), "--ewc-head", str(pc_ckpt)]
    if info["kernel"] == "v2":
        # header v2 : scales par-canal + act_max calibrés sur les mêmes colonnes/seed.
        export_cmd += ["--int8-v2", str(pc_ckpt),
                       "--condition", condition, "--dataset", dataset, "--model", "ewc"]
    if _run(export_cmd).returncode != 0:
        raise RuntimeError("export_weights_c échec")

    make_dims = [f"EWC_IN={k}", f"MAHA_DIM={k}", f"TINYOL_IN={k}", f"HDC_N_FEATURES={k}"]
    if k > 16:
        make_dims.append(f"PROTO_MAX_N={k}")
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    make_cmd = ["make", "-C", str(FW_DIR), *make_dims]
    if info["cflags"]:
        make_cmd.append(f"EXTRA_CFLAGS={' '.join(info['cflags'])}")
    make_cmd.append("all")
    if _run(make_cmd).returncode != 0:
        raise RuntimeError("make échec")
    bss = _bss_bytes()
    if flash and _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


def _ram_block(scheme: str, k: int) -> dict:
    """Empreinte RAM des poids selon le schéma (int8 ÷4, q15 ÷2 vs FP32 — Gap 3)."""
    fp32_b, int8_b = _ewc_weight_bytes(k)
    div = SCHEMES[scheme]["ram_div"]
    quant_b = int8_b if div == 4 else fp32_b // 2
    return {
        "ram_weights_fp32_bytes": fp32_b,
        "ram_weights_quant_bytes": quant_b,
        "ram_ratio_fp32_over_quant": round(fp32_b / quant_b, 3) if quant_b else None,
    }


# ── Passe FROZEN ─────────────────────────────────────────────────────────────

def run_frozen(scheme: str, dataset: str, condition: str, cfg: dict, args) -> dict:
    info = SCHEMES[scheme]
    print(f"\n{'='*70}\n=== S40 FROZEN  scheme={scheme}  dataset={dataset}  "
          f"kernel={info['kernel']}  ===\n{'='*70}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_ckpt = _pc_ckpt(dataset, condition)
    w = _load_head(pc_ckpt)
    act_max = calibrate_activations(w, X)   # même lot/seed que l'export → act_max identique

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = OUT_DIR / f"cell_{scheme}_{dataset}_frozen"
    exp_dir.mkdir(parents=True, exist_ok=True)
    bss = build_and_flash_s40(scheme, dataset, condition, k, X, pc_ckpt, exp_dir)

    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=n_tasks,
        rate_hz=float(cfg["uart"]["rate_hz"]), request_update=False, verbose=args.verbose,
        protocol_version=int(cfg["uart"]["proto"]), model_flags=ss.FRAME_FLAGS_INT8_MODE,
    )
    stats = ss._compute_stats(results)
    feats = np.array([r["features"] for r in results], dtype=np.float32)
    board_pred = np.array([int(r["pred"]) for r in results])

    # Référence FP32 (checkpoint PC = board FP32) et référence émulateur (kernel quantifié).
    fp32_pred = _fp32_pred(w, feats)
    emu_pred = _emu_pred(w, feats, scheme, act_max)
    lat = stats.get("latency_p50_us")

    # Persistance par échantillon (consommée par board_pc_parity.py --exp exp_S40_board_v2).
    board_samples = [
        {"idx": i, "true": int(results[i]["true"]),
         "pred_board": int(results[i]["pred"]),
         "pred_pc": int(emu_pred[i]),           # émulateur = référence bit-exacte v2
         "pred_fp32_ref": int(fp32_pred[i]),
         "conf_board": results[i].get("confidence")}
        for i in range(len(results))
    ]
    (exp_dir / "board_samples.json").write_text(json.dumps(board_samples))

    parity_class = "exact_vs_emulator" if info["kernel"] == "v2" else "approx_int8"
    result = {
        "exp_id": f"exp_S40_board_v2/results_{scheme}_{dataset}_frozen",
        "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "scheme": scheme, "kernel": info["kernel"],
        "protocol": "frozen", "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "frozen (sans --update)",
        "online_accuracy": stats.get("accuracy"),
        "f1_faulty": stats.get("f1_faulty"), "f1_macro": stats.get("f1_macro"),
        "metric_value": stats.get("f1_faulty"),
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "bss_bytes": bss, "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
        "parity_class": parity_class,
        "parity_rate": float((board_pred == emu_pred).mean()),
        "parity_mismatch_count": int((board_pred != emu_pred).sum()),
        "agreement_int8_vs_fp32": float((board_pred == fp32_pred).mean()),
        "n_compared": len(results),
        "emulator_f1_ref": _emulator_f1_ref(dataset, scheme),
    }
    result.update(_ram_block(scheme, k))
    (exp_dir.parent / f"results_{scheme}_{dataset}_frozen.json").write_text(
        json.dumps(result, indent=2))
    print(f"  k={k} .bss={bss} lat_p50={lat}µs F1={result['f1_faulty']} "
          f"parity_emu={result['parity_rate']:.4f} agree_fp32={result['agreement_int8_vs_fp32']:.4f} "
          f"ramx={result['ram_ratio_fp32_over_quant']}")
    return result


def _fp32_pred(w: EWCHeadWeights, feats: np.ndarray) -> np.ndarray:
    """Prédiction FP32 de référence (forward exact, = board FP32 même checkpoint)."""
    from src.utils.int8_c_emulation import forward_fp32
    return predict(forward_fp32(w, feats))


def _emulator_f1_ref(dataset: str, scheme: str):
    """Récupère la F1 émulateur PC (exp_S39_quant_sweep) pour comparaison — jamais inventée."""
    p = EXPERIMENTS / "exp_S39_quant_sweep" / f"ewc_{dataset}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    key = {"per_channel": "int8_perchannel", "q15": "q15",
           "int8_legacy": "int8_legacy"}[scheme]
    return data.get("schemes", {}).get(key, {}).get("metric")


# ── Passe ONLINE ─────────────────────────────────────────────────────────────

def run_online(scheme: str, dataset: str, condition: str, cfg: dict, args,
               ewc_lr: float, ewc_lambda: float) -> dict:
    info = SCHEMES[scheme]
    print(f"\n{'='*70}\n=== S40 ONLINE  scheme={scheme}  dataset={dataset}  "
          f"kernel={info['kernel']}  ===\n{'='*70}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_ckpt = _pc_ckpt(dataset, condition)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = OUT_DIR / f"cell_{scheme}_{dataset}_online"
    exp_dir.mkdir(parents=True, exist_ok=True)
    bss = build_and_flash_s40(scheme, dataset, condition, k, X, pc_ckpt, exp_dir)

    size = len(X) // n_tasks
    segments = [(f"t{i}", size if i < n_tasks - 1 else len(X) - size * (n_tasks - 1))
                for i in range(n_tasks)]

    all_results, per_task = ss._stream_cl_sequence(
        X, y, segments=segments, request_update=True, consolidate=True,
        verbose=args.verbose, dry_run=False, port=args.port, baud=args.baud,
        rate_hz=float(cfg["uart"]["rate_hz"]), protocol_version=int(cfg["uart"]["proto"]),
        output_dir=str(exp_dir), model_flags=ss.FRAME_FLAGS_INT8_MODE,
    )
    stats = ss._compute_stats(all_results)
    board_pred = np.array([int(r["pred"]) for r in all_results])
    board_true = np.array([int(r["true"]) for r in all_results])
    last = all_results[-1] if all_results else {}

    # Miroir PC online : trajectoire FP32 (SGD+consolidate, ref approchée — float32 board ≠
    # float64 PC). Réutilise le miroir S36 (identique aux séquences exp_S36).
    mirror = _pc_online_mirror(X, y, k, pc_ckpt, segments, ewc_lr, ewc_lambda)
    pc_pred = np.array(mirror["preds"][:len(board_pred)])
    parity_rate = float((board_pred == pc_pred).mean()) if len(pc_pred) == len(board_pred) else None

    n_align = min(len(all_results), len(mirror["preds"]))
    board_samples = [
        {"idx": i, "task_id": int(all_results[i].get("task_id", 0)),
         "true": int(all_results[i]["true"]),
         "pred_board": int(all_results[i]["pred"]),
         "conf_board": all_results[i].get("confidence"),
         "pred_pc": int(mirror["preds"][i])}
        for i in range(n_align)
    ]
    (exp_dir / "board_samples.json").write_text(json.dumps(board_samples))

    frozen_p = OUT_DIR / f"results_{scheme}_{dataset}_frozen.json"
    lat_inf_p50 = json.loads(frozen_p.read_text()).get("latency_us_p50") if frozen_p.exists() else None
    lat = stats.get("latency_p50_us")
    f1 = compute_fault_f1(board_true, board_pred)

    result = {
        "exp_id": f"exp_S40_board_v2/results_{scheme}_{dataset}_online",
        "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "scheme": scheme, "kernel": info["kernel"],
        "protocol": "online", "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "online (--update + consolidate)",
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "latency_inference_only_us_p50": lat_inf_p50,
        "latency_update_overhead_us_p50": (lat - lat_inf_p50)
        if (lat is not None and lat_inf_p50 is not None) else None,
        "online_accuracy": stats.get("accuracy"),
        "online_accuracy_firmware": last.get("acc"),
        "online_auroc_firmware": last.get("auroc"),
        "online_forgetting_firmware": last.get("forgetting"),
        "pc_online_accuracy": mirror["online_accuracy"],
        "online_forgetting": mirror["forgetting"],
        "per_task_board_acc": [t.get("accuracy") for t in per_task],
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"],
        "metric_value": f1["f1_faulty"],
        "bss_bytes": bss, "n_streamed": len(all_results), "crc_errors": stats.get("crc_errors"),
        "parity_class": "approx",
        "parity_rate": parity_rate, "n_compared": len(board_pred),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
        "emulator_f1_ref": _emulator_f1_ref(dataset, scheme),
    }
    result.update(_ram_block(scheme, k))
    frozen_board = OUT_DIR / f"cell_{scheme}_{dataset}_frozen" / "board_samples.json"
    if frozen_board.exists():   # accord INT8↔FP32 : board vs référence FP32 (frozen, même stream)
        fp32_ref = np.array([int(s["pred_fp32_ref"]) for s in json.loads(frozen_board.read_text())])
        n = min(len(fp32_ref), len(board_pred))
        result["agreement_int8_vs_fp32"] = float((board_pred[:n] == fp32_ref[:n]).mean()) if n else None
    (OUT_DIR / f"results_{scheme}_{dataset}_online.json").write_text(json.dumps(result, indent=2))
    print(f"  k={k} .bss={bss} lat_inf+MAJ_p50={lat}µs (inf={lat_inf_p50}µs) "
          f"parity~{parity_rate} F1={f1['f1_faulty']:.3f} ramx={result['ram_ratio_fp32_over_quant']}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Validation board kernel EWC INT8 v2 (S4002)")
    p.add_argument("--scheme", choices=list(SCHEMES), action="append", default=None,
                   help="répétable ; défaut = tous (per_channel, q15, int8_legacy)")
    p.add_argument("--dataset", choices=["pronostia", "monitoring"], default=None)
    p.add_argument("--proto", choices=["frozen", "online"], default=None)
    p.add_argument("--condition", default="5feat")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    ewc_lr, ewc_lambda = float(base["EWC_LR"]), float(base["EWC_LAMBDA"])

    schemes = args.scheme if args.scheme else list(SCHEMES)
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    protos = [args.proto] if args.proto else ["frozen", "online"]

    rows = []
    for scheme in schemes:
        for ds in datasets:
            for proto in protos:
                try:
                    if proto == "frozen":
                        rows.append(run_frozen(scheme, ds, args.condition, cfg, args))
                    else:
                        rows.append(run_online(scheme, ds, args.condition, cfg, args,
                                                ewc_lr, ewc_lambda))
                except Exception as exc:  # noqa: BLE001 — cellule robuste
                    print(f"  [FAIL {scheme}/{ds}/{proto}] {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}\nBoard S40 v2 : {len(rows)} cellules → {OUT_DIR}/")
    for r in rows:
        print(f"  {r['scheme']:12s} {r['dataset']:10s} {r['protocol']:7s} "
              f"F1={r.get('f1_faulty')} lat_p50={r.get('latency_us_p50')}µs "
              f"parity={r.get('parity_rate')}")


if __name__ == "__main__":
    main()
