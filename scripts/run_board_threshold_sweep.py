#!/usr/bin/env python3
"""run_board_threshold_sweep.py — Balayage board réel du seuil RUL→faulty (S3205).

Orchestre, sur la NUCLEO-F439ZI réelle, pour chaque ``(dataset, seuil)`` :

1. **Entraîne** les modèles de référence board 5-feat (``train_board_reference.py``) :
   Mahalanobis + EWC (EWCMlpMulticlass 5→32→16→2), label au seuil balayé.
2. **Exporte** les poids → headers C (``export_weights_c.py --mahal --ewc-head``).
3. **Recompile + flashe** une fois (les 2 modèles à parité partagent un flash ;
   HDC/TinyOL sont embarqués dans le même binaire).
4. **Streame** chaque modèle (``sensor_stream.py --dump-samples``) :
   - EWC, Mahalanobis → **parité board↔PC** (pred board == pred PC sur les mêmes
     features ; streaming **sans** ``--update`` pour figer les poids exportés) ;
   - HDC, TinyOL → **HW-only** (latence DWT µs + RAM, parité N/A par construction :
     projection HDC embarquée / init en ligne — cf. S3205, choix utilisateur).
5. **Consigne** ``experiments/exp_S32_board_{model}_{dataset}_thr{XX}/results.json``
   (latence, .bss, parité, online metrics) + ``exp_S32_board_sweep_summary.json``.

Idempotent : ``--skip-existing`` saute les cellules déjà produites. ``--dry-run``
valide la boucle sans toucher la board.

Usage :
    python scripts/run_board_threshold_sweep.py --datasets cmapss --thresholds 30   # 1 cellule
    python scripts/run_board_threshold_sweep.py                                      # matrice complète
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

from scripts.generate_threshold_sweep_configs import SWEEPS  # noqa: E402

FW_DIR = Path("firmware/stm32f4_blink")
EXPERIMENTS = Path("experiments")
SUMMARY = EXPERIMENTS / "exp_S32_board_sweep_summary.json"

PARITY_MODELS = ["mahalanobis", "ewc"]      # parité board↔PC exacte
HWONLY_MODELS = ["hdc", "tinyol"]           # latence/.bss seulement (parité N/A)
DATASETS = ["cmapss", "pronostia", "battery"]

GAP2_LATENCY_US = 100_000   # 100 ms (Gap 2)


# ── Sous-process helpers ────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _bss_bytes() -> int:
    """Lit .bss du binaire courant via arm-none-eabi-size."""
    out = subprocess.run(["arm-none-eabi-size", str(FW_DIR / "build/stm32f4_blink.elf")],
                         capture_output=True, text=True)
    line = out.stdout.strip().splitlines()[-1].split()
    return int(line[2])  # text data bss ...


# ── Référence PC pour la parité ─────────────────────────────────────────────

def _pc_pred_maha(ckpt: Path, feats: np.ndarray) -> np.ndarray:
    with open(ckpt, "rb") as f:
        model = pickle.load(f)
    scores = model.anomaly_score(feats)
    return (scores > model.threshold_).astype(int)


def _pc_pred_ewc(ckpt: Path, feats: np.ndarray) -> np.ndarray:
    import torch

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    model = EWCMlpMulticlass(input_dim=5, n_classes=2, hidden_dims=[32, 16])
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(feats, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


def _parity(model: str, ckpt: Path, samples: list[dict]) -> dict:
    """Compare pred board vs pred PC sur les features réellement envoyées."""
    valid = [s for s in samples if s.get("features")]
    if not valid:
        return {"parity_ok": None, "n_compared": 0, "parity_mismatch_count": None}
    feats = np.array([s["features"] for s in valid], dtype=np.float32)
    board = np.array([int(s["pred"]) for s in valid])
    pc = _pc_pred_maha(ckpt, feats) if model == "mahalanobis" else _pc_pred_ewc(ckpt, feats)
    n_mismatch = int((board != pc).sum())
    return {
        "parity_ok": bool(n_mismatch == 0),
        "n_compared": len(valid),
        "parity_mismatch_count": n_mismatch,
        "parity_rate": float((board == pc).mean()),
    }


# ── Streaming + collecte ────────────────────────────────────────────────────

def _stream(dataset: str, model: str, out_json: Path, args) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/sensor_stream.py",
        "--dataset", dataset, "--model", model,
        "--n-samples", str(args.n_samples),
        "--rate-hz", str(args.rate_hz),
        "--protocol-version", "3",
        "--dump-samples",
        "--output", str(out_json),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    proc = _run(cmd, timeout=600)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout)[-800:]}
    return json.loads(out_json.read_text()) if out_json.exists() else {}


def _collect_hw(stats: dict) -> dict:
    samples = stats.get("samples", [])
    lat = stats.get("latency_p50_us", stats.get("latency_mean_us"))
    ram = stats.get("ram_mean_bytes")
    return {
        "latency_us_p50": lat,
        "latency_us_p99": stats.get("latency_p99_us"),
        "ram_response_bytes": ram,
        "online_accuracy": stats.get("accuracy"),
        "crc_errors": stats.get("crc_errors"),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
    }


# ── Cellule (dataset, seuil) ────────────────────────────────────────────────

def run_cell(dataset: str, thr: int, args, rows: list[dict]) -> None:
    field, *_ = (SWEEPS[dataset][1],)
    print(f"\n{'='*70}\n=== BOARD CELL  dataset={dataset}  seuil={thr}  ===\n{'='*70}")

    # 1) Entraîner + exporter les modèles de référence board (Maha + EWC).
    ckpts: dict[str, Path] = {}
    for model in PARITY_MODELS:
        exp_dir = EXPERIMENTS / f"exp_S32_board_{model}_{dataset}_thr{thr}"
        ck = exp_dir / "checkpoints" / ("mahalanobis_task0.pkl" if model == "mahalanobis" else "ewc_head.pt")
        if not (args.skip_existing and ck.exists()):
            proc = _run([sys.executable, "scripts/train_board_reference.py",
                         "--model", model, "--dataset", dataset, "--threshold", str(thr),
                         "--exp_dir", str(exp_dir)], timeout=1200)
            if proc.returncode != 0:
                print(f"  [FAIL train {model}] {(proc.stderr or proc.stdout)[-500:]}")
                return
        ckpts[model] = ck

    if not args.dry_run:
        export_cmd = [sys.executable, "scripts/export_weights_c.py",
                      "--mahal", str(ckpts["mahalanobis"]),
                      "--ewc-head", str(ckpts["ewc"])]
        if _run(export_cmd).returncode != 0:
            print("  [FAIL export]"); return
        # 2) Recompile + flash (1 fois pour la cellule).
        subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
        if _run(["make", "-C", str(FW_DIR), "all"]).returncode != 0:
            print("  [FAIL build]"); return
        bss = _bss_bytes()
        if _run(["make", "-C", str(FW_DIR), "flash"]).returncode != 0:
            print("  [FAIL flash]"); return
    else:
        bss = _bss_bytes() if (FW_DIR / "build/stm32f4_blink.elf").exists() else 0

    date = datetime.now().isoformat(timespec="seconds")

    # 3) Streamer chaque modèle + consigner.
    for model in PARITY_MODELS + HWONLY_MODELS:
        exp_dir = EXPERIMENTS / f"exp_S32_board_{model}_{dataset}_thr{thr}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        stats = _stream(dataset, model, exp_dir / "stream.json", args)
        hw = _collect_hw(stats)

        result = {
            "exp_id": f"exp_S32_board_{model}_{dataset}_thr{thr}",
            "model": model, "dataset": dataset, "threshold": thr,
            "platform": "nucleo_f439zi", "date": date,
            "bss_bytes": bss, "n_features": 5,
            "parity_class": "parity" if model in PARITY_MODELS else "hw_only",
            **hw,
        }
        if model in PARITY_MODELS and not args.dry_run:
            result.update(_parity(model, ckpts[model], stats.get("samples", [])))
        elif model in HWONLY_MODELS:
            result["parity_ok"] = None
            result["parity_note"] = "N/A par construction (HDC projection embarquée / init en ligne)"

        (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
        rows.append(result)
        pflag = result.get("parity_ok")
        print(f"  [{model:11s}] lat_p50={hw['latency_us_p50']}µs parity={pflag} "
              f".bss={bss} gap2={hw['gap2_latency_compliant']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Balayage board réel du seuil (S3205)")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--thresholds", nargs="+", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Pas de flash/stream board")
    args = p.parse_args()

    rows: list[dict] = []
    for dataset in args.datasets:
        thresholds = args.thresholds or SWEEPS[dataset][2]
        for thr in thresholds:
            run_cell(dataset, thr, args, rows)

    # Consolidation (fusion par exp_id).
    merged: dict[str, dict] = {}
    if SUMMARY.exists():
        for r in json.load(open(SUMMARY)):
            merged[r["exp_id"]] = r
    for r in rows:
        merged[r["exp_id"]] = r
    summary = sorted(merged.values(), key=lambda r: r["exp_id"])
    SUMMARY.write_text(json.dumps(summary, indent=2))

    par = [r for r in summary if r.get("parity_class") == "parity" and r.get("parity_ok") is not None]
    ok = sum(1 for r in par if r["parity_ok"])
    print(f"\n{'='*60}\nBoard sweep terminé : {len(summary)} cellules ; "
          f"parité {ok}/{len(par)} OK\nRésumé : {SUMMARY}")


if __name__ == "__main__":
    main()
