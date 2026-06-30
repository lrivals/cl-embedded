"""
scripts/run_s28_pc_benchmarks.py — Orchestrateur des 20 benchmarks PC FP32 vs INT8 (S2807/S2808).

Lance benchmark_int8_fp32.run_benchmark pour chaque couple (modèle × dataset) du tableau
4×5 du Sprint 28. Chaque échec produit un JSON ``status: "blocked"`` documentant la cause
(dataset incompatible, dépendance manquante) au lieu d'interrompre la campagne.

Usage :
    python scripts/run_s28_pc_benchmarks.py [--n_samples N]
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from benchmark_int8_fp32 import run_benchmark  # exécuté depuis scripts/ ou via sys.path

_ROOT = Path(__file__).resolve().parent.parent
EWC_HDC_DIR = _ROOT / "experiments" / "exp_S28_PC_ewc_hdc"
TINYOL_MAHA_DIR = _ROOT / "experiments" / "exp_S28_PC_tinyol_maha"

DATASETS = ["cmapss", "monitoring", "cwru", "pronostia", "paderborn"]

# (model, output_dir, config_name_for_dataset) — config monitoring EWC garde son nom historique.
MATRIX = [
    ("ewc", EWC_HDC_DIR),
    ("hdc", EWC_HDC_DIR),
    ("tinyol", TINYOL_MAHA_DIR),
    ("mahalanobis", TINYOL_MAHA_DIR),
]


def _config_for(model: str, dataset: str) -> str:
    return f"configs/{model}_int8_{dataset}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Campagne PC S2807/S2808")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    EWC_HDC_DIR.mkdir(parents=True, exist_ok=True)
    TINYOL_MAHA_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for model, out_dir in MATRIX:
        for dataset in DATASETS:
            config = _config_for(model, dataset)
            out = out_dir / f"results_{model}_{dataset}.json"
            print(f"\n########## {model} × {dataset} ##########")
            try:
                res = run_benchmark(
                    model_name=model,
                    config_path=config,
                    output_path=str(out),
                    n_samples=args.n_samples,
                )
                summary.append((model, dataset, "ok", res.get("delta_metric"), res.get("ram_ratio")))
            except Exception as exc:  # noqa: BLE001 — on veut documenter, pas planter
                tb = traceback.format_exc()
                print(f"⚠️  ÉCHEC {model}×{dataset} : {exc}")
                blocked = {
                    "model": model,
                    "dataset": dataset,
                    "config_path": config,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "blocked",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                }
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(blocked, f, indent=2)
                summary.append((model, dataset, "blocked", None, None))

    print("\n" + "=" * 70)
    print("  RÉSUMÉ CAMPAGNE S2807/S2808")
    print("=" * 70)
    for model, dataset, status, dmetric, ratio in summary:
        extra = f"Δ={dmetric:+.4f} ratio={ratio:.2f}×" if status == "ok" else ""
        print(f"  {model:12s} {dataset:11s} {status:8s} {extra}")


if __name__ == "__main__":
    main()
