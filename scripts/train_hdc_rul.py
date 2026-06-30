"""
train_hdc_rul.py — Boucle CL HDC Régression RUL.

Scénario supporté :
    CMAPSS : domain-incremental FD001 → FD002 → FD003 → FD004

Usage :
    python scripts/train_hdc_rul.py \\
        --config configs/cmapss_rul_config.yaml \\
        --exp_id exp_S25_04 \\
        --output_dir experiments/exp_S25_04/

    # Dry-run
    python scripts/train_hdc_rul.py \\
        --config configs/cmapss_rul_config.yaml \\
        --exp_id dry_run_hdc_rul \\
        --output_dir /tmp/dry_run/ \\
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.rul_metrics import compute_avg_forgetting_rmse, compute_rul_metrics_task
from src.models.hdc.hdc_regressor import HDCRegressor
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed


def _load_tasks(cfg: dict, config_path: str) -> list[dict]:
    dataset = cfg["dataset"]
    mode = cfg.get("task_mode", "rul")
    if dataset == "cmapss":
        from src.data.cmapss_loader import get_cl_dataloaders as load_cmapss
        return load_cmapss(
            data_dir=Path(cfg["data_dir"]),
            config_path=Path(config_path),
            mode=mode,
        )
    else:
        raise ValueError(f"Dataset non supporté pour HDC RUL : {dataset}")


def _collect_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def evaluate_task(model: HDCRegressor, loader, rul_scale: float = 125.0) -> dict[str, float]:
    x_all, y_all = _collect_numpy(loader)
    y_pred = model.predict(x_all) * rul_scale
    y_all = y_all.squeeze()
    return compute_rul_metrics_task(y_all, y_pred)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    set_seed(cfg.get("seed", 42))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    D: int = cfg.get("D", 1024)
    # Normaliser le LR par 1/D : chaque gradient de w_i accumule D contributions ±1,
    # un LR calibré pour ~700 params (EWC) serait 1000x trop grand pour HDC(D=1024).
    base_lr = float(cfg.get("HDC_LR", cfg.get("EWC_LR", 0.01)))
    hdc_lr = base_lr / D  # ≈ 9.8e-6 pour EWC_LR=0.01, D=1024
    model = HDCRegressor(
        D=D,
        n_levels=cfg.get("N_LEVELS", 10),
        n_features=cfg["INPUT_DIM"],
        lr=hdc_lr,
        seed=cfg.get("seed", 42),
    )

    tasks = _load_tasks(cfg, args.config)
    n_tasks = len(tasks)
    n_epochs = cfg.get("N_EPOCHS_PER_TASK", 20)
    # Normalisation des cibles : train sur [0,1] pour stabilité
    rul_scale: float = float(cfg.get("rul_cap", 125))

    # Initialiser les bornes de features sur la première tâche
    x_first, _ = _collect_numpy(tasks[0]["train_loader"])
    model.set_feature_bounds(x_first)

    task_rmse_per_epoch: list[list[float]] = []
    task_results: list[dict] = []

    t_start = time.time()
    for task_idx, task in enumerate(tasks):
        task_name = task.get("domain", task.get("condition", task_idx + 1))
        print(f"\n--- Tâche {task_idx + 1}/{n_tasks} : {task_name} ---")
        train_loader = task["train_loader"]
        val_loader = task["val_loader"]

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                y_np = y_batch.numpy().squeeze() / rul_scale
                loss = model.fit_batch(x_batch.numpy(), y_np)
                epoch_loss += loss
                n_batches += 1

            row: list[float] = []
            for prev_task in tasks[: task_idx + 1]:
                metrics = evaluate_task(model, prev_task["val_loader"], rul_scale)
                row.append(metrics["rmse"])
            task_rmse_per_epoch.append(row + [0.0] * (n_tasks - len(row)))

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / max(1, n_batches)
                print(
                    f"  Epoch {epoch+1}/{n_epochs} — loss={avg_loss:.4f},"
                    f" RMSE_t{task_idx+1}={row[-1]:.2f}"
                )

        final_metrics = evaluate_task(model, val_loader, rul_scale)
        task_results.append({"task_id": task_idx + 1, **final_metrics})
        print(f"  Final — RMSE={final_metrics['rmse']:.2f}, MAE={final_metrics['mae']:.2f}")

    elapsed = time.time() - t_start
    af_rmse = compute_avg_forgetting_rmse(task_rmse_per_epoch)

    # Comparaison vs EWC RUL (exp_S25_01) si disponible
    ewc_rmse_t1 = None
    ref_path = Path("experiments/exp_S25_01/results.json")
    if ref_path.exists():
        import json as _json
        ref = _json.load(open(ref_path))
        ewc_rmse_t1 = ref.get("per_task_metrics", [{}])[0].get("rmse")

    results = {
        "exp_id": args.exp_id,
        "model": "hdc_regressor",
        "dataset": cfg["dataset"],
        "task_mode": cfg.get("task_mode", "rul"),
        "n_tasks": n_tasks,
        "per_task_metrics": task_results,
        "avg_forgetting_rmse": af_rmse,
        "training_time_s": elapsed,
        "n_params": model.count_parameters(),
        "config": str(args.config),
        "comparison_vs_ewc_regression": {
            "exp_S25_01_rmse_task1": ewc_rmse_t1,
            "exp_S25_04_rmse_task1": task_results[0]["rmse"] if task_results else None,
        },
    }

    if not args.dry_run:
        import yaml
        with open(output_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump({**cfg, "exp_id": args.exp_id, "model": "hdc_regressor"}, f, allow_unicode=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRésultats sauvegardés dans {output_dir}/")

    print(f"\n=== Résumé exp_id={args.exp_id} ===")
    print(f"  AF RMSE : {af_rmse:.4f}")
    for t in task_results:
        print(f"  Task {t['task_id']} — RMSE={t['rmse']:.2f}, MAE={t['mae']:.2f}")
    if ewc_rmse_t1 is not None:
        print(f"  Comparaison Task 1 : HDC={task_results[0]['rmse']:.2f}  EWC={ewc_rmse_t1:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDC RUL Régression — boucle CL")
    parser.add_argument("--config", required=True, help="Chemin vers la config YAML")
    parser.add_argument("--exp_id", required=True, help="Identifiant de l'expérience")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie des résultats")
    parser.add_argument("--dry_run", action="store_true", help="Ne pas écrire sur disque")
    args = parser.parse_args()
    main(args)
