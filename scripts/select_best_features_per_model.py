"""
select_best_features_per_model.py — Sélection des meilleures features PAR MODÈLE (Sprint 35 / S3501).

Contrairement aux subsets `configs/*_feature_subset.yaml` (top-5 par *dataset*), ce script
calcule la condition `best` = meilleures features **spécifiques à chaque modèle** (un détecteur
de distance et un MLP ne valorisent pas les mêmes variables).

Procédure (fittée sur train/val uniquement — pas de fuite test) :
  1. Charger le dataset à sa dimension native, entraîner le modèle sur toutes les features.
  2. Classer les features par ``permutation_importance`` (réutilisé tel quel — métrique accuracy).
  3. Balayer k=1..n : ré-entraîner sur le top-k, mesurer F1 (classe faulty) sur le val.
  4. Retenir k* = plus petit k à moins de ``--parcimonie`` du F1 max.
  5. Écrire ``configs/best_features/{model}_{dataset}.yaml`` (généré — ne pas éditer à la main).

Usage
-----
    python scripts/select_best_features_per_model.py --model ewc --dataset cwru
    python scripts/select_best_features_per_model.py --all          # 4 modèles × 5 datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.evaluation.feature_conditions import (
    DATASETS,
    MODELS,
    NATIVE_FEATURE_NAMES,
    load_native_task_arrays,
    train_and_evaluate,
)
from src.evaluation.feature_importance import permutation_importance

OUTPUT_DIR = Path("configs/best_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sélection de features par modèle (S3501)")
    p.add_argument("--model", choices=MODELS, default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None)
    p.add_argument("--all", action="store_true", help="Boucle complète 4 modèles × 5 datasets")
    p.add_argument("--n-repeats", type=int, default=5, help="Permutations par feature (ranking)")
    p.add_argument(
        "--parcimonie",
        type=float,
        default=0.01,
        help="Tolérance relative au F1 max pour retenir le plus petit k (défaut 1%%)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sous-échantillonne (stratifié) train ET val à ≤ N par tâche pendant la SÉLECTION "
        "du k* uniquement (0 = données complètes). Accélère les datasets lourds (CMAPSS) sans "
        "toucher au sweep final S3503 qui garde les données complètes.",
    )
    return p.parse_args()


def _stratified_sample(y: "np.ndarray", max_n: int, rng: "np.random.Generator") -> "np.ndarray":
    """Indices d'un sous-échantillon stratifié ≤ max_n (chaque classe plafonnée à max_n/n_classes)."""
    classes = np.unique(y)
    per = max(1, max_n // len(classes))
    keep = []
    for c in classes:
        ci = np.where(y == c)[0]
        if len(ci) > per:
            ci = rng.choice(ci, size=per, replace=False)
        keep.append(ci)
    return np.sort(np.concatenate(keep))


def _subsample_tasks(tasks: list[dict], max_samples: int, seed: int) -> list[dict]:
    """Plafonne train+val de chaque tâche (stratifié, classes préservées) pour la sélection."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for t in tasks:
        nt = dict(t)
        for split in ("train", "val"):
            X, y = t[f"X_{split}"], t[f"y_{split}"]
            if len(X) > max_samples:
                idx = _stratified_sample(y, max_samples, rng)
                nt[f"X_{split}"], nt[f"y_{split}"] = X[idx], y[idx]
        out.append(nt)
    return out


def _select_one(
    model: str, dataset: str, n_repeats: int, parcimonie: float, seed: int, max_samples: int = 0
) -> dict:
    feature_names = NATIVE_FEATURE_NAMES[dataset]
    n = len(feature_names)
    print(f"\n=== {model} × {dataset} (n_features={n}) ===")

    tasks = load_native_task_arrays(dataset, seed=seed)
    if max_samples:
        sizes_before = [len(t["X_train"]) for t in tasks]
        tasks = _subsample_tasks(tasks, max_samples, seed)
        print(f"  sous-échantillonnage sélection : train/tâche {sizes_before} → "
              f"{[len(t['X_train']) for t in tasks]} (≤{max_samples})")

    # 1+2. Entraînement sur toutes les features → ranking par permutation importance.
    import numpy as np

    full = train_and_evaluate(model, tasks, list(range(n)), seed=seed)
    X_val = np.concatenate([t["X_val"] for t in tasks])
    y_val = np.concatenate([t["y_val"] for t in tasks])
    ranking = permutation_importance(
        full["predict_fn"], X_val, y_val, feature_names, n_repeats=n_repeats, random_state=seed
    )
    ranked_names = list(ranking.keys())
    ranked_idx = [feature_names.index(name) for name in ranked_names]
    print(f"  ranking: {ranked_names}")

    # 3. Balayage k : F1 val sur le top-k.
    val_f1_by_k: dict[int, float] = {}
    for k in range(1, n + 1):
        idx_k = ranked_idx[:k]
        res = train_and_evaluate(model, tasks, idx_k, seed=seed)
        val_f1_by_k[k] = round(res["f1_faulty"], 6)
        print(f"  k={k:2d} → F1_faulty={val_f1_by_k[k]:.4f}  (features={ranked_names[:k]})")

    # 4. k* = plus petit k à <parcimonie du max.
    f1_max = max(val_f1_by_k.values())
    k_star = next(
        k for k in range(1, n + 1) if val_f1_by_k[k] >= f1_max - parcimonie
    )
    selected_idx = ranked_idx[:k_star]
    selected_names = ranked_names[:k_star]
    print(f"  → k*={k_star} (F1_max={f1_max:.4f}, parcimonie={parcimonie}) : {selected_names}")

    return {
        "model": model,
        "dataset": dataset,
        "method": "permutation_importance",
        "metric": "f1_faulty",
        "n_features_total": n,
        "n_features_selected": k_star,
        "selected_indices": [int(i) for i in selected_idx],
        "selected_features": selected_names,
        "val_f1_by_k": {int(k): float(v) for k, v in val_f1_by_k.items()},
        "fit_split": f"train (seed {seed})",
        "permutation_n_repeats": n_repeats,
        "parcimonie": parcimonie,
        "selection_max_samples": max_samples or None,
    }


def _write_yaml(payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{payload['model']}_{payload['dataset']}.yaml"
    header = (
        f"# configs/best_features/{payload['model']}_{payload['dataset']}.yaml\n"
        "# Généré par scripts/select_best_features_per_model.py (Sprint 35 / S3501).\n"
        "# NE PAS éditer manuellement.\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    print(f"  ✅ écrit → {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    if args.all:
        pairs = [(m, d) for m in MODELS for d in DATASETS]
    else:
        if not args.model or not args.dataset:
            raise SystemExit("Préciser --model ET --dataset, ou utiliser --all.")
        pairs = [(args.model, args.dataset)]

    for model, dataset in pairs:
        payload = _select_one(
            model, dataset, args.n_repeats, args.parcimonie, args.seed, args.max_samples
        )
        _write_yaml(payload)


if __name__ == "__main__":
    main()
