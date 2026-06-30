"""
run_feature_condition_sweep.py — Re-run PC des 3 conditions de features (Sprint 35 / S3503).

Balaye ``condition × modèle × dataset`` (3 × 4 × 5 = 60 cellules) et produit, pour la
détection de panne, **F1 (classe faulty) ET acc_final** (+ avg_forgetting, RAM, n_params).

Conditions :
  - ``5feat`` → ``configs/{dataset}_feature_subset.yaml`` (subset top-5 existant)
  - ``all``   → ``configs/all_features/{dataset}.yaml`` (S3502, dims natives)
  - ``best``  → ``configs/best_features/{model}_{dataset}.yaml`` (S3501, par modèle)

Cas monitoring : pas de subset top-5 (4 features natives) → ``5feat`` ≡ ``all`` (documenté
dans le results.json via ``note``).

Sortie : ``experiments/exp_S35_PC_{condition}_{model}_{dataset}/results.json`` (+ snapshot).

Usage
-----
    python scripts/run_feature_condition_sweep.py --dry-run                 # liste 60 cellules
    python scripts/run_feature_condition_sweep.py --condition best --model ewc --dataset cwru
    python scripts/run_feature_condition_sweep.py --all                     # sweep complet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.evaluation.feature_conditions import (
    CONDITIONS,
    DATASETS,
    MODELS,
    NATIVE_FEATURE_NAMES,
    load_native_task_arrays,
    resolve_feature_indices,  # noqa: F401 — re-exporté (profile_memory.py S3505)
    train_and_evaluate,
)

EXP_ROOT = Path("experiments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep PC 3 conditions de features (S3503)")
    p.add_argument("--condition", choices=CONDITIONS, default=None)
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None)
    p.add_argument("--all", action="store_true", help="Sweep complet 3 × 4 × 5")
    p.add_argument("--dry-run", action="store_true", help="Liste les cellules sans entraîner")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_cell(condition: str, model: str, dataset: str, seed: int) -> dict:
    exp_id = f"exp_S35_PC_{condition}_{model}_{dataset}"
    idx, note = resolve_feature_indices(condition, model, dataset)
    print(f"\n=== {exp_id} | {len(idx)} features | {note} ===")

    tasks = load_native_task_arrays(dataset, seed=seed)
    res = train_and_evaluate(model, tasks, idx, seed=seed)

    results = {
        "exp_id": exp_id,
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "platform": "pc",
        "sprint": 35,
        "n_features": int(res["n_features"]),
        "selected_indices": [int(i) for i in idx],
        "acc_final": res["acc_final"],
        "f1_faulty": res["f1_faulty"],
        "f1_macro": res["f1_macro"],
        "precision_faulty": res["precision_faulty"],
        "recall_faulty": res["recall_faulty"],
        "avg_forgetting": res["avg_forgetting"],
        "backward_transfer": res["backward_transfer"],
        "ram_peak_bytes": res["ram_peak_bytes"],
        "n_params": res["n_params"],
        "note": note,
    }

    exp_dir = EXP_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    snapshot = {
        "exp_id": exp_id,
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "feature_indices": [int(i) for i in idx],
        "feature_names": [NATIVE_FEATURE_NAMES[dataset][i] for i in idx],
    }
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)

    print(
        f"  ✅ acc_final={results['acc_final']:.4f} | "
        f"F1_faulty={results['f1_faulty']:.4f} | F1_macro={results['f1_macro']:.4f} "
        f"→ {exp_dir/'results.json'}"
    )
    return results


def main() -> None:
    args = parse_args()

    if args.all or args.dry_run and not (args.condition or args.model or args.dataset):
        conditions = CONDITIONS
        models = MODELS
        datasets = DATASETS
    else:
        conditions = [args.condition] if args.condition else CONDITIONS
        models = [args.model] if args.model else MODELS
        datasets = [args.dataset] if args.dataset else DATASETS

    cells = [(c, m, d) for c in conditions for m in models for d in datasets]

    if args.dry_run:
        print(f"{len(cells)} cellules :")
        for c, m, d in cells:
            print(f"  exp_S35_PC_{c}_{m}_{d}")
        return

    failures: list[tuple] = []
    for c, m, d in cells:
        try:
            run_cell(c, m, d, args.seed)
        except Exception as exc:  # noqa: BLE001 — isolation par cellule (sweep robuste)
            print(f"  ❌ {c}_{m}_{d} : {type(exc).__name__}: {exc}")
            failures.append((c, m, d, str(exc)))

    if failures:
        print(f"\n{len(failures)} cellule(s) en échec :")
        for c, m, d, e in failures:
            print(f"  - {c}_{m}_{d} : {e}")


if __name__ == "__main__":
    main()
