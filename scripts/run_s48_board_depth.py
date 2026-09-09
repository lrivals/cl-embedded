#!/usr/bin/env python3
"""run_s48_board_depth.py — Driver board sub-INT8 pour la tête EWC (Sprint 48, S4804).

Matérialise sur NUCLEO-F439ZI réelle les schémas gagnants du Sprint 47 (émulateur) :
mesure la **RAM `.bss` réelle** (packé vs non-packé), la **latence DWT** (coût du dépacking)
et **valide la parité board↔PC** (émulateur subint8 = réplique bit-exacte).

Pour chaque cellule ``(dataset, mode, packed)`` du cœur scientifique (S4801) :

  1. Charge (X, y) 5feat (``load_condition_arrays``) → dim native k (monitoring 4, pronostia 5).
  2. Entraîne une tête EWC FP32 de référence (``train_ewc_head``) — 1× par dataset, réutilisée.
  3. ``export_weights_c.py --ewc-subint8`` (mode/bits/granularité/symétrie [--packed]) — réutilise
     les primitives de l'émulateur S47 ⇒ **parité par construction**.
  4. ``make clean`` puis ``make EXTRA_CFLAGS="-D<FLAG> [-DEWC_INTx_PACKED]
     -DEWC_SUBINT8_WEIGHTS_PROVIDED" EWC_IN=k all`` ; ``.bss`` lu ; ``make flash``.
  5. Stream **frozen** (sans ``--update``, ``_stream_uart`` flag 0x40) → latence DWT P50/P99,
     CRC, prédictions ; ``board_samples.json`` persisté (parité S4805).
  6. Parité board↔émulateur (``emulator_predict`` rejoue le schéma subint8) + AUROC board.
  7. Écrit ``experiments/exp_S48_board/exp_S48_<ds>_<mode>_<gran>[_packed].json``.

Aucun chiffre inventé : tous les champs mesurés sont ``null``/``"à mesurer"`` avant flash ;
N/A honnête (``na_reason``) si un schéma déborde la SRAM (précédent PSI×gas_sensor S45).

Usage :
    python scripts/run_s48_board_depth.py --all --port /dev/ttyACM0
    python scripts/run_s48_board_depth.py --dataset monitoring --mode ternary --packed
    python scripts/run_s48_board_depth.py --cell pronostia_binary_per_channel_packed --no-flash
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import _bss_bytes  # noqa: E402
from scripts.run_sprint45_board import train_ewc_head  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
OUT_DIR = EXPERIMENTS / "exp_S48_board"
GAP2_LATENCY_US = 100_000        # 100 ms (Gap 2)
RAM_BUDGET_BYTES = 256 * 1024    # NUCLEO-F439ZI SRAM (Gap 3)
CONDITION = "5feat"
GRANULARITY = "per_channel"
SYMMETRY = "symmetric"
DEFAULT_SEED = 42
A_MESURER = "à mesurer"

# mode → (weight_mode émulateur, bits effectifs, flag de build). Cf. _SUBINT8_SCHEMES (export).
MODES: dict[str, tuple[str, int, str]] = {
    "int4":    ("linear", 4, "EWC_INT4"),
    "ternary": ("ternary", 2, "EWC_INT2"),
    "binary":  ("binary", 1, "EWC_INT1"),
}
DATASETS = ["monitoring", "pronostia"]


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Émulateur subint8 = réplique PC bit-exacte (parité par construction) ──────

def emulator_predict(ckpt: Path, mode: str, bits: int, feats: np.ndarray):
    """Rejoue le schéma subint8 (S47) du checkpoint FP32 sur ``feats``.

    Utilise les MÊMES primitives que ``export_weights_c.py --ewc-subint8`` (mêmes poids
    quantifiés, mêmes scales, même calibration seed-48) ⇒ parité board↔PC par construction.
    Retourne ``(pred, prob1, logits)``.
    """
    import torch  # noqa: PLC0415

    from src.utils.int8_c_emulation import (  # noqa: PLC0415
        EWCHeadWeights,
        QuantConfig,
        calibrate_activations,
        forward_quant,
        predict,
        softmax_prob1,
    )

    checkpoint = torch.load(ckpt, map_location="cpu")
    sd = checkpoint.get("model_state_dict", checkpoint)
    w = EWCHeadWeights.from_state_dict(sd)
    k = int(w.w1.shape[1])
    # Calibration identique au défaut de l'export (aucun --condition) → act_max identique.
    rng = np.random.default_rng(48)
    calib = rng.standard_normal((256, k)).astype(np.float32)
    act_max = calibrate_activations(w, calib)
    cfg = QuantConfig.subint8(bits, granularity=GRANULARITY, symmetry=SYMMETRY, mode=mode)
    logits = forward_quant(w, np.asarray(feats, dtype=np.float32), cfg, act_max=act_max)
    return predict(logits), softmax_prob1(logits), logits


# ── Métriques ────────────────────────────────────────────────────────────────

def _auroc(y_true: np.ndarray, scores: np.ndarray):
    """AUROC binaire ; None si une seule classe présente (N/A honnête, mono-classe)."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    return float(roc_auc_score(y, np.asarray(scores)))


# ── Build/flash ──────────────────────────────────────────────────────────────

def export_and_build(ckpt: Path, mode: str, bits: int, flag: str, packed: bool,
                     k: int, flash: bool) -> int:
    """Export sub-INT8 → build (packé/non-packé) → .bss → flash. Retourne .bss (B)."""
    weight_mode = MODES[mode][0]
    # L'export ne valide --weight-bits que pour le mode linéaire ({4,2}) ; en
    # ternaire/binaire les bits effectifs (2/1) sont dérivés de --weight-mode
    # (_SUBINT8_SCHEMES) ⇒ on passe une valeur valide ignorée.
    export_bits = bits if weight_mode == "linear" else 2
    export_cmd = [sys.executable, "scripts/export_weights_c.py",
                  "--ewc-subint8", str(ckpt),
                  "--weight-mode", weight_mode, "--weight-bits", str(export_bits),
                  "--granularity", GRANULARITY, "--symmetry", SYMMETRY]
    if packed:
        export_cmd.append("--packed")
    if _run(export_cmd).returncode != 0:
        raise RuntimeError("export_weights_c échec")

    extra = f"-D{flag} -DEWC_SUBINT8_WEIGHTS_PROVIDED"
    if packed:
        extra += " -DEWC_INTx_PACKED"
    make_args = [f"EXTRA_CFLAGS={extra}", f"EWC_IN={k}"]
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    r = _run(["make", "-C", str(FW_DIR), *make_args, "all"])
    if r.returncode != 0:
        raise RuntimeError(f"make échec:\n{r.stderr[-1500:]}")
    bss = _bss_bytes()
    if flash and _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
        raise RuntimeError("flash échec")
    return bss


# ── Assemblage d'une cellule ─────────────────────────────────────────────────

def _cell_id(dataset: str, mode: str, packed: bool) -> str:
    return f"exp_S48_{dataset}_{mode}_{GRANULARITY}{'_packed' if packed else ''}"


def _base(dataset: str, mode: str, bits: int, packed: bool, k: int, names: list[str]) -> dict:
    return {
        "exp_id": _cell_id(dataset, mode, packed),
        "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "weight_bits": bits, "weight_mode": MODES[mode][0], "mode": mode,
        "granularity": GRANULARITY, "symmetry": SYMMETRY, "packed": packed,
        "n_features": k, "feature_names": names,
        "seed": DEFAULT_SEED, "date": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": {
            "condition": CONDITION, "mode": mode, "weight_bits": bits,
            "granularity": GRANULARITY, "symmetry": SYMMETRY, "packed": packed,
            "build_flag": MODES[mode][2], "input_dim": k, "hidden_dims": [32, 16], "seed": DEFAULT_SEED,
        },
    }


def _pending(base: dict, bss: int | None, reason: str) -> dict:
    """Cellule non streamée (--no-stream) : mesures HW présentes, métriques différées."""
    base.update({
        "bss_bytes": bss, "bss_bytes_int8_ref": None, "ram_ratio_measured_vs_int8": None,
        "auroc_board": A_MESURER, "auroc_pc_emulator": None, "parity_pred": None,
        "latency_dwt_p50_us": None, "latency_dwt_p99_us": None,
        "crc_errors": None, "gap2_ok": None, "gap3_ram_ok": (bss is not None and bss < RAM_BUDGET_BYTES),
        "na_reason": None, "stream_mode": reason,
    })
    return base


def _na(base: dict, bss: int | None, reason: str) -> dict:
    """Cellule non mesurable (débordement SRAM, mono-classe…) : N/A honnête."""
    base.update({
        "bss_bytes": bss, "bss_bytes_int8_ref": None, "ram_ratio_measured_vs_int8": None,
        "auroc_board": None, "auroc_pc_emulator": None, "parity_pred": None,
        "latency_dwt_p50_us": None, "latency_dwt_p99_us": None,
        "crc_errors": None, "gap2_ok": None, "gap3_ram_ok": None, "na_reason": reason,
    })
    return base


def run_cell(dataset: str, mode: str, packed: bool, ckpt: Path,
             X: np.ndarray, y: np.ndarray, names: list[str], args) -> dict:
    weight_mode, bits, flag = MODES[mode]
    k = X.shape[1]
    print(f"\n{'='*70}\n=== BOARD sub-INT8  dataset={dataset}  mode={mode}  "
          f"packed={packed}  k={k}  ===\n{'='*70}")
    base = _base(dataset, mode, bits, packed, k, names)
    exp_dir = OUT_DIR / base["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    bss = export_and_build(ckpt, mode, bits, flag, packed, k, flash=not args.no_flash)

    if args.no_stream:
        result = _pending(base, bss, "différé (--no-stream)")
        (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
        print(f"  .bss={bss} (stream différé)")
        return result

    # Stream frozen (sans --update). model_flags=0x40 → route sub-INT8 (pipeline.c S4804).
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=args.n_tasks, rate_hz=args.rate_hz,
        request_update=False, verbose=args.verbose,
        protocol_version=3, model_flags=ss.FRAME_FLAGS_INT8_MODE,
    )
    if not results:
        result = _na(base, bss, "aucune réponse UART (timeout stream)")
        (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result
    stats = ss._compute_stats(results)

    trues = np.array([int(r["true"]) for r in results])
    board_pred = np.array([int(r["pred"]) for r in results])
    board_conf = np.array([float(r["confidence"]) for r in results])
    feats = np.array([r["features"] for r in results], dtype=np.float32)

    # Parité board↔émulateur (réplique PC bit-exacte).
    pc_pred, pc_prob1, _ = emulator_predict(ckpt, weight_mode, bits, feats)
    parity_pred = float((board_pred == pc_pred).mean())

    # board_samples.json pour la parité détaillée (S4805).
    samples = [{"idx": i, "true": int(trues[i]), "pred": int(board_pred[i]),
                "confidence": float(board_conf[i]), "latency_us": float(results[i]["latency_us"]),
                "features": [float(v) for v in feats[i]]} for i in range(len(results))]
    (exp_dir / "board_samples.json").write_text(json.dumps({
        "dataset": dataset, "mode": mode, "weight_bits": bits, "packed": packed,
        "checkpoint": str(ckpt), "samples": samples}, indent=2))

    p50 = stats.get("latency_p50_us")
    p99 = stats.get("latency_p99_us")
    f1 = compute_fault_f1(trues, board_pred)
    result = base | {
        "bss_bytes": bss, "bss_bytes_int8_ref": None, "ram_ratio_measured_vs_int8": None,
        "auroc_board": _auroc(trues, board_conf),
        "auroc_pc_emulator": _auroc(trues, pc_prob1),
        "parity_pred": parity_pred,
        "accuracy": stats.get("accuracy"), "f1_faulty": round(f1["f1_faulty"], 4),
        "f1_macro": round(f1["f1_macro"], 4), "metric_value": round(f1["f1_faulty"], 4),
        "latency_dwt_p50_us": p50, "latency_dwt_p99_us": p99,
        "latency_dwt_mean_us": stats.get("latency_mean_us"),
        "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "gap2_ok": bool(p99 is not None and p99 < GAP2_LATENCY_US),
        "gap3_ram_ok": bool(bss < RAM_BUDGET_BYTES),
        "na_reason": None, "stream_mode": "frozen (sans --update)",
    }
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"  .bss={bss} lat_p50={p50}µs p99={p99}µs parity={parity_pred:.4f} "
          f"auroc_board={result['auroc_board']} crc={result['crc_errors']} "
          f"F1={result['f1_faulty']} → {exp_dir}/results.json")
    return result


# ── Sélection des cellules ───────────────────────────────────────────────────

def _cells(args) -> list[tuple[str, str, bool]]:
    if args.cell:
        # format <ds>_<mode>_<gran>[_packed]
        packed = args.cell.endswith("_packed")
        core = args.cell[:-len("_packed")] if packed else args.cell
        # core = <ds>_<mode>_<gran> ; mode/ds connus
        for ds in DATASETS:
            for mode in MODES:
                if core == f"{ds}_{mode}_{GRANULARITY}":
                    return [(ds, mode, packed)]
        raise SystemExit(f"cellule inconnue : {args.cell}")
    datasets = [args.dataset] if args.dataset else DATASETS
    modes = [args.mode] if args.mode else list(MODES)
    if args.packed and args.no_packed:
        packings = [True, False]
    elif args.packed:
        packings = [True]
    elif args.no_packed:
        packings = [False]
    else:
        packings = [False, True]   # cœur scientifique : les deux
    return [(ds, m, p) for ds in datasets for m in modes for p in packings]


def main() -> None:
    ap = argparse.ArgumentParser(description="Driver board sub-INT8 EWC (Sprint 48, S4804)")
    ap.add_argument("--all", action="store_true", help="12 cellules cœur (défaut si rien d'autre)")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--mode", choices=list(MODES))
    ap.add_argument("--cell", help="cellule unique <ds>_<mode>_per_channel[_packed]")
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--no-packed", action="store_true")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seed(DEFAULT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = _cells(args)
    print(f"[S4804] {len(cells)} cellule(s) : {cells}")

    # Tête EWC FP32 entraînée 1× par dataset (réutilisée pour ses cellules).
    ckpts: dict[str, tuple[Path, np.ndarray, np.ndarray, list[str]]] = {}
    for ds in {c[0] for c in cells}:
        X, y, idx, names = load_condition_arrays(ds, CONDITION, "ewc", seed=DEFAULT_SEED)
        ck_dir = OUT_DIR / f"_ref_{ds}"
        ckpt = train_ewc_head(X, y, ck_dir, DEFAULT_SEED)
        ckpts[ds] = (ckpt, X, y, names)
        print(f"  réf FP32 {ds} : k={X.shape[1]} → {ckpt}")

    summary = []
    for ds, mode, packed in cells:
        ckpt, X, y, names = ckpts[ds]
        try:
            summary.append(run_cell(ds, mode, packed, ckpt, X, y, names, args))
        except Exception as exc:  # noqa: BLE001 — robustesse par cellule (patron S36)
            print(f"  [ERREUR] {ds}/{mode}/packed={packed} : {exc}")
            base = _base(ds, mode, MODES[mode][1], packed, X.shape[1], names)
            exp_dir = OUT_DIR / base["exp_id"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            res = _na(base, None, f"échec cellule : {exc}")
            (exp_dir / "results.json").write_text(json.dumps(res, indent=2))
            summary.append(res)

    print(f"\n[S4804] {len(summary)} cellule(s) écrite(s) dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
