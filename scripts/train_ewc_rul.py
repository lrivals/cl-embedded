"""
train_ewc_rul.py — Boucle CL EWC Régression RUL.

Scénarios supportés :
    CMAPSS : domain-incremental FD001 → FD002 → FD003 → FD004
    Pronostia : domain-incremental Condition 1 → 2 → 3

Usage :
    python scripts/train_ewc_rul.py \\
        --config configs/cmapss_rul_config.yaml \\
        --exp_id exp_S25_01 \\
        --output_dir experiments/exp_S25_01/

    python scripts/train_ewc_rul.py \\
        --config configs/pronostia_rul_config.yaml \\
        --exp_id exp_S25_02 \\
        --output_dir experiments/exp_S25_02/

    # Dry-run (vérifie pipeline sans écriture disque)
    python scripts/train_ewc_rul.py \\
        --config configs/cmapss_rul_config.yaml \\
        --exp_id dry_run_rul \\
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
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.rul_metrics import compute_avg_forgetting_rmse, compute_rul_metrics_task
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed


def _load_tasks(cfg: dict, config_path: str) -> list[dict]:
    """Charge les DataLoaders selon le dataset spécifié dans la config."""
    dataset = cfg["dataset"]
    mode = cfg.get("task_mode", "rul")
    if dataset == "cmapss":
        from src.data.cmapss_loader import get_cl_dataloaders as load_cmapss
        return load_cmapss(
            data_dir=Path(cfg["data_dir"]),
            config_path=Path(config_path),
            mode=mode,
        )
    elif dataset == "pronostia":
        from src.data.pronostia_dataset import get_pronostia_dataloaders as load_pronostia
        return load_pronostia(
            npy_dir=Path(cfg["npy_dir"]),
            normalizer_path=Path(cfg["normalizer_path"]),
            batch_size=cfg.get("BATCH_SIZE", 32),
            val_ratio=cfg.get("val_ratio", 0.2),
            seed=cfg.get("seed", 42),
            mode=mode,
        )
    else:
        raise ValueError(f"Dataset non supporté en mode RUL : {dataset}")


def train_one_epoch(
    model: EWCMlpRegressor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    rul_scale: float = 125.0,
) -> float:
    """Entraîne une époque sur cibles normalisées [0,1]. Retourne MSE loss moyen."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(x_batch).squeeze()
        y_target = y_batch.float().squeeze() / rul_scale
        loss = criterion(y_pred, y_target) + model.ewc_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_task(
    model: EWCMlpRegressor,
    loader: torch.utils.data.DataLoader,
    rul_scale: float = 125.0,
) -> dict[str, float]:
    """Évalue le modèle sur une tâche. Retourne RMSE / MAE / Horizon Score en cycles."""
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            # Dé-normalisation : modèle entraîné sur [0,1], metrics en cycles
            y_pred = model(x_batch).squeeze().numpy() * rul_scale
            y_true_all.append(y_batch.numpy().squeeze())
            y_pred_all.append(np.atleast_1d(y_pred))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_rul_metrics_task(y_true, y_pred)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    set_seed(cfg.get("seed", 42))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EWCMlpRegressor(
        input_dim=cfg["INPUT_DIM"],
        hidden_dims=cfg.get("HIDDEN_DIMS", [32, 16]),
        dropout=cfg.get("DROPOUT", 0.2),
        ewc_lambda=cfg.get("EWC_LAMBDA", 400.0),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["EWC_LR"],
        momentum=cfg.get("MOMENTUM", 0.9),
    )
    criterion = nn.MSELoss()

    tasks = _load_tasks(cfg, args.config)
    n_tasks = len(tasks)
    n_epochs = cfg.get("N_EPOCHS_PER_TASK", 20)
    # Normalisation des cibles : entraîner sur [0,1] pour stabilité gradient
    rul_scale: float = float(cfg.get("rul_cap", 125))

    task_rmse_per_epoch: list[list[float]] = []
    task_results: list[dict] = []

    t_start = time.time()
    for task_idx, task in enumerate(tasks):
        task_name = task.get("domain", task.get("condition", task_idx + 1))
        print(f"\n--- Tâche {task_idx + 1}/{n_tasks} : {task_name} ---")
        train_loader = task["train_loader"]
        val_loader = task["val_loader"]

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, rul_scale)
            row: list[float] = []
            for prev_task in tasks[: task_idx + 1]:
                metrics = evaluate_task(model, prev_task["val_loader"], rul_scale)
                row.append(metrics["rmse"])
            task_rmse_per_epoch.append(row + [0.0] * (n_tasks - len(row)))

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch+1}/{n_epochs} — loss={train_loss:.4f},"
                    f" RMSE_t{task_idx+1}={row[-1]:.2f}"
                )

        model.consolidate(train_loader, n_samples=cfg.get("FISHER_N_SAMPLES", 200), rul_scale=rul_scale)

        final_metrics = evaluate_task(model, val_loader, rul_scale)
        task_results.append({"task_id": task_idx + 1, **final_metrics})
        print(f"  Final — RMSE={final_metrics['rmse']:.2f}, MAE={final_metrics['mae']:.2f}")

    elapsed = time.time() - t_start
    af_rmse = compute_avg_forgetting_rmse(task_rmse_per_epoch)

    results = {
        "exp_id": args.exp_id,
        "model": "ewc_regression",
        "dataset": cfg["dataset"],
        "task_mode": cfg.get("task_mode", "rul"),
        "n_tasks": n_tasks,
        "per_task_metrics": task_results,
        "avg_forgetting_rmse": af_rmse,
        "training_time_s": elapsed,
        "n_params": sum(p.numel() for p in model.parameters()),
        "config": str(args.config),
    }

    if not args.dry_run:
        import yaml
        with open(output_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump({**cfg, "exp_id": args.exp_id}, f, allow_unicode=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        torch.save(model.state_dict(), output_dir / "model_ewc_reg.pt")
        print(f"\nRésultats sauvegardés dans {output_dir}/")

    print(f"\n=== Résumé exp_id={args.exp_id} ===")
    print(f"  AF RMSE : {af_rmse:.4f}")
    for t in task_results:
        print(f"  Task {t['task_id']} — RMSE={t['rmse']:.2f}, MAE={t['mae']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EWC RUL Regression — boucle CL")
    parser.add_argument("--config", required=True, help="Chemin vers la config YAML")
    parser.add_argument("--exp_id", required=True, help="Identifiant de l'expérience")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie des résultats")
    parser.add_argument("--dry_run", action="store_true", help="Ne pas écrire sur disque")
    args = parser.parse_args()
    main(args)
