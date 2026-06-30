"""
train_ewc_multiclass.py — Boucle CL EWC Classification Multi-classe.

Scénarios supportés :
    CWRU : by_fault_type (3 tâches, 10 classes)
    Paderborn : domain-incremental (3 états bearing)

Usage :
    python scripts/train_ewc_multiclass.py \\
        --config configs/cwru_multiclass_config.yaml \\
        --exp_id exp_S25_03 \\
        --output_dir experiments/exp_S25_03/

    python scripts/train_ewc_multiclass.py \\
        --config configs/paderborn_multiclass_config.yaml \\
        --exp_id exp_S25_03b \\
        --output_dir experiments/exp_S25_03b/

    # Dry-run (vérifie pipeline sans écriture disque)
    python scripts/train_ewc_multiclass.py \\
        --config configs/cwru_multiclass_config.yaml \\
        --exp_id dry_run_mc \\
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

from src.evaluation.multiclass_metrics import (
    compute_avg_forgetting_f1,
    compute_multiclass_metrics_task,
)
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed


def _load_tasks(cfg: dict, config_path: str) -> list[dict]:
    """Charge les DataLoaders selon le dataset spécifié dans la config."""
    dataset = cfg["dataset"]
    mode = cfg.get("task_mode", "multiclass")

    if dataset == "cwru":
        from torch.utils.data import DataLoader, TensorDataset

        from src.data.cwru_dataset import get_cl_splits

        splits = get_cl_splits(
            csv_path=Path(cfg["data_path"]),
            scenario=cfg.get("scenario", "by_fault_type"),
            mode=mode,
            test_size=cfg.get("val_ratio", 0.2),
            random_state=cfg.get("seed", 42),
        )
        batch_size = cfg.get("BATCH_SIZE", 32)
        tasks = []
        for split in splits:
            X_train = torch.from_numpy(split["X_train"])
            y_train = torch.from_numpy(split["y_train"])
            X_val = torch.from_numpy(split["X_val"])
            y_val = torch.from_numpy(split["y_val"])
            tasks.append(
                {
                    "task_id": split["task_id"] + 1,
                    "domain": split["task_name"],
                    "train_loader": DataLoader(
                        TensorDataset(X_train, y_train),
                        batch_size=batch_size,
                        shuffle=True,
                    ),
                    "val_loader": DataLoader(
                        TensorDataset(X_val, y_val),
                        batch_size=batch_size,
                        shuffle=False,
                    ),
                }
            )
        return tasks

    elif dataset == "paderborn":
        from src.data.paderborn_loader import get_cl_dataloaders as load_paderborn

        return load_paderborn(
            data_dir=Path(cfg["data_dir"]),
            config_path=Path(config_path),
            mode=mode,
        )

    else:
        raise ValueError(f"Dataset non supporté en mode multiclass : {dataset}")


def train_one_epoch(
    model: EWCMlpMulticlass,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Entraîne une époque, retourne Cross-Entropy loss moyen."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch.long()) + model.ewc_penalty()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_task(
    model: EWCMlpMulticlass,
    loader: torch.utils.data.DataLoader,
    n_classes: int,
) -> dict[str, float | list]:
    """Évalue le modèle sur une tâche, retourne F1-macro / confusion matrix."""
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            preds = logits.argmax(dim=1).numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(preds)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_multiclass_metrics_task(y_true, y_pred, n_classes)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    set_seed(cfg.get("seed", 42))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = cfg["N_CLASSES"]
    model = EWCMlpMulticlass(
        input_dim=cfg["INPUT_DIM"],
        n_classes=n_classes,
        hidden_dims=cfg.get("HIDDEN_DIMS", [32, 16]),
        dropout=cfg.get("DROPOUT", 0.2),
        ewc_lambda=cfg.get("EWC_LAMBDA", 400.0),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["EWC_LR"],
        momentum=cfg.get("MOMENTUM", 0.9),
    )
    criterion = nn.CrossEntropyLoss()

    tasks = _load_tasks(cfg, args.config)
    n_tasks = len(tasks)
    n_epochs = cfg.get("N_EPOCHS_PER_TASK", 30)

    task_f1_per_epoch: list[list[float]] = []
    task_results: list[dict] = []

    t_start = time.time()
    for task_idx, task in enumerate(tasks):
        task_name = task.get("domain", task_idx + 1)
        print(f"\n--- Tâche {task_idx + 1}/{n_tasks} : {task_name} ---")
        train_loader = task["train_loader"]
        val_loader = task["val_loader"]

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            row: list[float] = []
            for prev_task in tasks[: task_idx + 1]:
                metrics = evaluate_task(model, prev_task["val_loader"], n_classes)
                row.append(float(metrics["f1_macro"]))
            task_f1_per_epoch.append(row + [0.0] * (n_tasks - len(row)))

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch+1}/{n_epochs} — loss={train_loss:.4f},"
                    f" F1_t{task_idx+1}={row[-1]:.3f}"
                )

        model.consolidate(train_loader, n_samples=cfg.get("FISHER_N_SAMPLES", 200))

        final_metrics = evaluate_task(model, val_loader, n_classes)
        task_results.append({"task_id": task_idx + 1, **final_metrics})
        print(f"  Final — F1-macro={final_metrics['f1_macro']:.3f}")

    elapsed = time.time() - t_start
    af_f1 = compute_avg_forgetting_f1(task_f1_per_epoch)

    results = {
        "exp_id": args.exp_id,
        "model": "ewc_multiclass",
        "dataset": cfg["dataset"],
        "task_mode": cfg.get("task_mode", "multiclass"),
        "n_tasks": n_tasks,
        "n_classes": n_classes,
        "per_task_metrics": task_results,
        "avg_forgetting_f1": af_f1,
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
        torch.save(model.state_dict(), output_dir / "model_ewc_mc.pt")
        print(f"\nRésultats sauvegardés dans {output_dir}/")

    print(f"\n=== Résumé exp_id={args.exp_id} ===")
    print(f"  AF F1-macro : {af_f1:.4f}")
    for t in task_results:
        print(f"  Task {t['task_id']} — F1-macro={t['f1_macro']:.3f}")

    # Critère de validation tâche 1
    if task_results:
        f1_task1 = float(task_results[0]["f1_macro"])
        threshold = cfg.get("f1_macro_min_task1", 0.70)
        status = "✅" if f1_task1 >= threshold else "⚠"
        print(f"\n  {status} Tâche 1 F1-macro={f1_task1:.3f} (seuil={threshold:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EWC Multi-classe — boucle CL")
    parser.add_argument("--config", required=True, help="Chemin vers la config YAML")
    parser.add_argument("--exp_id", required=True, help="Identifiant de l'expérience")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie des résultats")
    parser.add_argument("--dry_run", action="store_true", help="Ne pas écrire sur disque")
    args = parser.parse_args()
    main(args)
