#!/usr/bin/env python3
"""run_sprint36_board.py — Passes board appariées EWC (S3603 frozen / S3604 online).

Réutilise les helpers éprouvés de ``run_feature_condition_board.py`` (Sprint 35) et les
fonctions de streaming de ``sensor_stream.py`` (in-process, aucune modif de ces scripts).

Pour chaque cellule ``(condition, dataset)`` de ``configs/sprint36_ewc_comparison.yaml`` :

  Étapes communes (build/flash) :
    1. Charge le **checkpoint PC** ``exp_S36_PC_{cond}_ewc_{ds}/checkpoints/ewc_head.pt``
       (produit par S3602) → ⇒ modèle PC == modèle flashé ⇒ parité exacte par construction.
    2. Entraîne un Maha de référence (mêmes arrays) pour que le build firmware ait des
       ``model_weights.h`` cohérents en dim (on ne streame que l'EWC).
    3. ``export_weights_c.py --mahal --ewc-head`` → headers C.
    4. ``make clean`` puis ``make EWC_IN=k MAHA_DIM=k TINYOL_IN=k HDC_N_FEATURES=k all`` ;
       ``.bss`` lu ; ``make flash``.

  --pass frozen (S3603) :
    5. Stream **sans --update** (``_stream_uart``), split complet → latence **inférence seule**.
    6. Parité **exacte** : pred_board vs ``_pc_pred_ewc`` (même checkpoint).
    → experiments/exp_S36_board_frozen_{cond}_ewc_{ds}/results.json

  --pass online (S3604) :
    5. Stream **avec --update + consolidate** (``_stream_cl_sequence``, séquence 3 tâches),
       split complet → latence **inférence + MAJ CL** (DWT).
    6. Miroir PC online (rejoue la même séquence : prédire→1 pas SGD→consolidate aux
       frontières) → parité **approchée** (taux de concordance, jamais exacte).
    7. Delta latence vs frozen (S3603) repris de son results.json.
    → experiments/exp_S36_board_online_{cond}_ewc_{ds}/results.json

Sprint 36 rework (S3611) : option ``--precision {fp32,int8}``. En ``int8`` le build/flash est
identique (poids FP32 exportés, convertis en INT8 par le firmware au boot — S3610), seul le
flag UART change (``FRAME_FLAGS_INT8_MODE`` 0x40 → ``g_ewc_int8``). Sorties dans
``exp_S36_board_{frozen,online}_int8_*`` avec ratios RAM/latence et accord INT8↔FP32 board.

Usage :
    python scripts/run_sprint36_board.py --pass frozen --port /dev/ttyACM0
    python scripts/run_sprint36_board.py --pass online --port /dev/ttyACM0
    python scripts/run_sprint36_board.py --pass frozen --precision int8 --port /dev/ttyACM0
    python scripts/run_sprint36_board.py --pass online --precision int8 --port /dev/ttyACM0
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_feature_condition_board import (  # noqa: E402
    _bss_bytes,
    _pc_pred_ewc,
    train_maha_board,
)
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.evaluation.online_metrics import OnlineAccuracy, OnlineForgetting  # noqa: E402
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

FW_DIR = Path("firmware/stm32f4_blink")
ELF = FW_DIR / "build/stm32f4_blink.elf"
EXPERIMENTS = Path("experiments")
GAP2_LATENCY_US = 100_000   # 100 ms (Gap 2)
DEFAULT_CONFIG = "configs/sprint36_ewc_comparison.yaml"

# Tête EWC binaire (cf. EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16])).
EWC_H1, EWC_H2, EWC_OUT = 32, 16, 2


def _ewc_weight_bytes(k: int) -> tuple[int, int]:
    """Empreinte des tenseurs de poids EWC (fp32, int8) à la dim k.

    Compte analytique des poids des 3 couches (k→32→16→2). FP32 = 4 B/poids,
    INT8 = 1 B/poids ⇒ ratio structurel 4.0 (cf. Sprint 28/29). Distinct de ``.bss``
    qui héberge les DEUX têtes (FP32 + INT8) simultanément côté firmware.
    """
    n_w = k * EWC_H1 + EWC_H1 * EWC_H2 + EWC_H2 * EWC_OUT
    return n_w * 4, n_w * 1


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Build/flash (commun aux deux passes) ─────────────────────────────────────

def build_and_flash(condition: str, dataset: str, k: int, X: np.ndarray,
                    pc_ckpt: Path, exp_dir: Path) -> int:
    """Export → build → flash à la dim k. Retourne .bss (B)."""
    # Maha de référence (cohérence dims du build ; non streamé).
    maha_ckpt = train_maha_board(X, exp_dir)
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


# ── Passe FROZEN (S3603) ─────────────────────────────────────────────────────

def run_frozen(condition: str, dataset: str, cfg: dict, args, precision: str = "fp32") -> dict:
    is_int8 = (precision == "int8")
    tag = "_int8" if is_int8 else ""
    print(f"\n{'='*68}\n=== BOARD FROZEN  condition={condition}  dataset={dataset}  "
          f"precision={precision}  ===\n{'='*68}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_ckpt = EXPERIMENTS / f"exp_S36_PC_{condition}_ewc_{dataset}" / "checkpoints" / "ewc_head.pt"
    if not pc_ckpt.exists():
        raise FileNotFoundError(f"{pc_ckpt} absent — lancer scripts/run_sprint36_pc.py d'abord")

    exp_id = f"exp_S36_board_frozen{tag}_{condition}_ewc_{dataset}"
    exp_dir = EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    bss = build_and_flash(condition, dataset, k, X, pc_ckpt, exp_dir)

    # Stream gelé (split complet → n_samples = len(X)) ; features attachées par _stream_uart.
    # INT8 : flag 0x40 (PROTO_FLAG_INT8_MODE) → g_ewc_int8 (chargé depuis FP32 par S3610).
    model_flags = ss.FRAME_FLAGS_INT8_MODE if is_int8 else ss.FRAME_FLAGS_EWC_MODE
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=n_tasks,
        rate_hz=float(cfg["uart"]["rate_hz"]), request_update=False, verbose=args.verbose,
        protocol_version=int(cfg["uart"]["proto"]), model_flags=model_flags,
    )
    stats = ss._compute_stats(results)

    feats = np.array([r["features"] for r in results], dtype=np.float32)
    board_pred = np.array([int(r["pred"]) for r in results])
    # FP32 board == PC (même checkpoint) ⇒ _pc_pred_ewc sert de référence FP32.
    fp32_ref_pred = _pc_pred_ewc(pc_ckpt, feats)
    lat = stats.get("latency_p50_us")

    result = {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "precision": precision.upper(), "n_features": k,
        "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "frozen (sans --update)",
        "online_accuracy": stats.get("accuracy"),
        "f1_faulty": stats.get("f1_faulty"), "f1_macro": stats.get("f1_macro"),
        "metric_value": stats.get("f1_faulty"),
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "bss_bytes": bss, "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
    }
    if is_int8:
        fp32_b, int8_b = _ewc_weight_bytes(k)
        result.update({
            "parity_class": "approx_int8",   # quantification ⇒ pas de parité exacte
            "agreement_int8_vs_fp32": float((board_pred == fp32_ref_pred).mean()),
            "n_compared": len(results),
            "ram_weights_fp32_bytes": fp32_b, "ram_weights_int8_bytes": int8_b,
            "ram_ratio_fp32_over_int8": round(fp32_b / int8_b, 3) if int8_b else None,
        })
        print(f"  k={k} INT8 .bss={bss} lat_p50={lat}µs "
              f"agree_vs_fp32={result['agreement_int8_vs_fp32']:.4f} "
              f"F1={result['f1_faulty']} ramx={result['ram_ratio_fp32_over_int8']} "
              f"→ {exp_dir}/results.json")
    else:
        n_mismatch = int((board_pred != fp32_ref_pred).sum())
        result.update({
            "parity_class": "exact",
            "parity_ok": bool(n_mismatch == 0),
            "parity_rate": float((board_pred == fp32_ref_pred).mean()),
            "parity_mismatch_count": n_mismatch, "n_compared": len(results),
        })
        print(f"  k={k} .bss={bss} lat_p50={lat}µs parity={result['parity_ok']} "
              f"({result['parity_rate']:.4f}) F1={result['f1_faulty']} → {exp_dir}/results.json")
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


# ── Passe ONLINE (S3604) ─────────────────────────────────────────────────────

def _pc_online_mirror(X: np.ndarray, y: np.ndarray, k: int, pc_ckpt: Path,
                      segments: list[tuple[str, int]], ewc_lr: float,
                      ewc_lambda: float) -> dict:
    """Rejoue la séquence online sur PC : prédire→1 pas SGD→consolidate aux frontières."""
    sd = torch.load(pc_ckpt, map_location="cpu")["model_state_dict"]
    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16], ewc_lambda=ewc_lambda)
    model.load_state_dict(sd)
    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    online_acc = OnlineAccuracy()
    forget = OnlineForgetting()
    preds: list[int] = []
    n_tasks = len(segments)
    offset = 0
    task_X: list[np.ndarray] = []
    task_y: list[np.ndarray] = []
    for task_id, (_name, n) in enumerate(segments):
        end = min(offset + n, len(X))
        Xt, yt = X[offset:end], y[offset:end]
        task_X.append(Xt)
        task_y.append(yt)
        for i in range(len(Xt)):
            xb = torch.tensor(Xt[i:i + 1], dtype=torch.float32)
            yb = torch.tensor(yt[i:i + 1], dtype=torch.long)
            model.eval()
            with torch.no_grad():
                p = int(model(xb).argmax(dim=1).item())
            preds.append(p)
            online_acc.update(int(yt[i]), p)
            # 1 pas SGD (inférence + MAJ).
            model.train()
            optimizer.zero_grad()
            (criterion(model(xb), yb) + model.ewc_penalty()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        # Acc « pic » de la tâche (juste après l'avoir vue).
        forget.record_task_end(task_id, _eval_acc(model, Xt, yt))
        # Consolidation aux frontières (sauf dernière tâche).
        if task_id < n_tasks - 1:
            ld = DataLoader(TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                                          torch.tensor(yt, dtype=torch.long)),
                            batch_size=32, shuffle=True)
            model.consolidate(ld, n_samples=200)
        offset = end

    # Acc finale par tâche → forgetting.
    for task_id in range(n_tasks):
        forget.record_final(task_id, _eval_acc(model, task_X[task_id], task_y[task_id]))
    af = forget.compute()

    return {"preds": preds, "online_accuracy": online_acc.compute(),
            "forgetting": af.get("af"), "per_task_af": af.get("per_task")}


def _agreement_vs_fp32_online(fp32_samples_path: Path, board_pred: np.ndarray):
    """Taux d'accord INT8↔FP32 board online (mêmes échantillons, même ordre).

    Compare les prédictions board INT8 aux préds board FP32 persistées par la passe
    online FP32 (``board_samples.json``). Retourne ``None`` si la référence FP32 manque.
    """
    if not fp32_samples_path.exists():
        return None
    fp32 = json.loads(fp32_samples_path.read_text())
    fp32_pred = np.array([int(s["pred_board"]) for s in fp32])
    n = min(len(fp32_pred), len(board_pred))
    if n == 0:
        return None
    return float((board_pred[:n] == fp32_pred[:n]).mean())


def _eval_acc(model: EWCMlpMulticlass, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((pred == y).mean())


def run_online(condition: str, dataset: str, cfg: dict, args,
               ewc_lr: float, ewc_lambda: float, precision: str = "fp32") -> dict:
    is_int8 = (precision == "int8")
    tag = "_int8" if is_int8 else ""
    print(f"\n{'='*68}\n=== BOARD ONLINE  condition={condition}  dataset={dataset}  "
          f"precision={precision}  ===\n{'='*68}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    pc_ckpt = EXPERIMENTS / f"exp_S36_PC_{condition}_ewc_{dataset}" / "checkpoints" / "ewc_head.pt"
    if not pc_ckpt.exists():
        raise FileNotFoundError(f"{pc_ckpt} absent — lancer scripts/run_sprint36_pc.py d'abord")

    exp_id = f"exp_S36_board_online{tag}_{condition}_ewc_{dataset}"
    exp_dir = EXPERIMENTS / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    bss = build_and_flash(condition, dataset, k, X, pc_ckpt, exp_dir)

    # Séquence CL contiguë couvrant le split complet (3 tâches), tranches égales.
    size = len(X) // n_tasks
    segments = [(f"t{i}", size if i < n_tasks - 1 else len(X) - size * (n_tasks - 1))
                for i in range(n_tasks)]

    # Board online : forward + backprop + consolidate aux frontières.
    # INT8 : flag 0x40 → g_ewc_int8 (ewc_int8_update + ewc_int8_consolidate firmware).
    model_flags = ss.FRAME_FLAGS_INT8_MODE if is_int8 else ss.FRAME_FLAGS_EWC_MODE
    all_results, per_task = ss._stream_cl_sequence(
        X, y, segments=segments, request_update=True, consolidate=True,
        verbose=args.verbose, dry_run=False, port=args.port, baud=args.baud,
        rate_hz=float(cfg["uart"]["rate_hz"]), protocol_version=int(cfg["uart"]["proto"]),
        output_dir=str(exp_dir), model_flags=model_flags,
    )
    stats = ss._compute_stats(all_results)
    board_pred = np.array([int(r["pred"]) for r in all_results])
    board_true = np.array([int(r["true"]) for r in all_results])
    last = all_results[-1] if all_results else {}

    # Miroir PC online (même séquence contiguë).
    mirror = _pc_online_mirror(X, y, k, pc_ckpt, segments, ewc_lr, ewc_lambda)
    pc_pred = np.array(mirror["preds"][:len(board_pred)])
    parity_rate = float((board_pred == pc_pred).mean()) if len(pc_pred) == len(board_pred) else None

    # Persistance des prédictions par échantillon (S3605, parité online réelle).
    # idx = indice positionnel global de la séquence contiguë (identique board↔miroir PC).
    n_align = min(len(all_results), len(mirror["preds"]))
    board_samples = [
        {
            "idx": i,
            "task_id": int(all_results[i].get("task_id", 0)),
            "true": int(all_results[i]["true"]),
            "pred_board": int(all_results[i]["pred"]),
            "conf_board": all_results[i].get("confidence"),
            "pred_pc": int(mirror["preds"][i]),
        }
        for i in range(n_align)
    ]
    (exp_dir / "board_samples.json").write_text(json.dumps(board_samples))

    # Latence inférence seule reprise de la passe frozen de MÊME précision (S3603/S3611).
    frozen_p = (EXPERIMENTS / f"exp_S36_board_frozen{tag}_{condition}_ewc_{dataset}"
                / "results.json")
    lat_inf_p50 = None
    if frozen_p.exists():
        lat_inf_p50 = json.loads(frozen_p.read_text()).get("latency_us_p50")
    lat = stats.get("latency_p50_us")

    f1 = compute_fault_f1(board_true, board_pred)
    result = {
        "exp_id": exp_id, "platform": "nucleo_f439zi", "model": "ewc", "dataset": dataset,
        "condition": condition, "precision": precision.upper(), "n_features": k,
        "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "online (--update + consolidate)",
        "latency_us_p50": lat, "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "latency_inference_only_us_p50": lat_inf_p50,
        "latency_update_overhead_us_p50": (lat - lat_inf_p50)
        if (lat is not None and lat_inf_p50 is not None) else None,
        # Métriques online firmware (proto v3, dernier échantillon = cumulatif).
        "online_accuracy": stats.get("accuracy"),
        "online_accuracy_firmware": last.get("acc"),
        "online_auroc_firmware": last.get("auroc"),
        "online_forgetting_firmware": last.get("forgetting"),
        # Miroir PC online (FP32, référence).
        "pc_online_accuracy": mirror["online_accuracy"],
        "online_forgetting": mirror["forgetting"],
        "pc_per_task_af": mirror["per_task_af"],
        "per_task_board_acc": [t.get("accuracy") for t in per_task],
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"],
        "metric_value": f1["f1_faulty"],
        "bss_bytes": bss, "n_streamed": len(all_results), "crc_errors": stats.get("crc_errors"),
        "parity_class": "approx_int8" if is_int8 else "approx",
        "parity_rate": parity_rate, "n_compared": len(board_pred),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
    }
    if is_int8:
        fp32_b, int8_b = _ewc_weight_bytes(k)
        result.update({
            "ram_weights_fp32_bytes": fp32_b, "ram_weights_int8_bytes": int8_b,
            "ram_ratio_fp32_over_int8": round(fp32_b / int8_b, 3) if int8_b else None,
            "agreement_int8_vs_fp32": _agreement_vs_fp32_online(
                EXPERIMENTS / f"exp_S36_board_online_{condition}_ewc_{dataset}"
                / "board_samples.json", board_pred),
        })
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"  k={k} {precision.upper()} .bss={bss} lat_inf+MAJ_p50={lat}µs "
          f"(inf seul={lat_inf_p50}µs) parity~{parity_rate} F1={f1['f1_faulty']:.3f} "
          f"→ {exp_dir}/results.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Passes board appariées EWC (S3603/S3604)")
    p.add_argument("--pass", dest="pass_", choices=["frozen", "online"], required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--condition", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                   help="fp32 (défaut, EWC head FP32) ou int8 (g_ewc_int8, flag 0x40)")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    ewc_lr, ewc_lambda = float(base["EWC_LR"]), float(base["EWC_LAMBDA"])

    conditions = [args.condition] if args.condition else cfg["conditions"]
    datasets = [args.dataset] if args.dataset else cfg["datasets"]

    rows = []
    for d in datasets:
        for c in conditions:
            try:
                if args.pass_ == "frozen":
                    rows.append(run_frozen(c, d, cfg, args, precision=args.precision))
                else:
                    rows.append(run_online(c, d, cfg, args, ewc_lr, ewc_lambda,
                                            precision=args.precision))
            except Exception as exc:  # noqa: BLE001 — cellule robuste
                print(f"  [FAIL {c}/{d}] {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}\nBoard S36 ({args.pass_}) : {len(rows)} cellules.")
    for r in rows:
        print(f"  {r['exp_id']:42s} lat_p50={r.get('latency_us_p50')}µs "
              f"parity={r.get('parity_ok', r.get('parity_rate'))}")


if __name__ == "__main__":
    main()
