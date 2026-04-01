"""
scripts/train_ewc.py — Script d'entraînement EWC Online sur Dataset 2.

Usage :
    python scripts/train_ewc.py --config configs/ewc_config.yaml
    python scripts/train_ewc.py --config configs/ewc_config.yaml --lambda 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from src.evaluation.memory_profiler import full_memory_report
from src.evaluation.metrics import compute_cl_metrics, format_metrics_report, save_metrics
from src.models.ewc.ewc_mlp import EWCMlpClassifier
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="EWC Online Training — Dataset 2")
    parser.add_argument("--config", default="configs/ewc_config.yaml")
    parser.add_argument("--lambda", dest="ewc_lambda", type=float, default=None,
                        help="Override EWC lambda coefficient")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Vérifie la config et l'architecture sans entraîner")
    return parser.parse_args()


def make_dummy_cl_stream(cfg: dict) -> list[dict]:
    """
    Crée un stream CL factice pour valider le pipeline.
    À remplacer par le vrai loader quand Dataset 2 est téléchargé.

    Returns
    -------
    list[dict] : [{"train": DataLoader, "test": DataLoader, "name": str}, ...]
    """
    print("⚠️  Données réelles non trouvées — utilisation de données synthétiques.")
    print(f"   → Télécharger Dataset 2 dans : {cfg['data']['path']}")

    tasks = []
    n_features = cfg["model"]["input_dim"]
    n_samples = 500

    domain_names = cfg["data"]["domain_order"]
    for i, domain in enumerate(domain_names):
        # Simuler un shift de distribution par domaine (mean différente)
        X = torch.randn(n_samples, n_features) + i * 0.5
        y = (torch.randn(n_samples) > 0.3 + i * 0.1).float().unsqueeze(1)

        split = int(n_samples * 0.8)
        train_ds = data.TensorDataset(X[:split], y[:split])
        test_ds = data.TensorDataset(X[split:], y[split:])

        tasks.append({
            "name": domain,
            "train": data.DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True),
            "test": data.DataLoader(test_ds, batch_size=64, shuffle=False),
        })

    return tasks


def evaluate_on_all_tasks(model, tasks, seen_up_to: int, device) -> list[float]:
    """Évalue le modèle sur toutes les tâches vues."""
    model.eval()
    accs = []
    for i, task in enumerate(tasks[:seen_up_to + 1]):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in task["test"]:
                x, y = x.to(device), y.to(device)
                pred = model(x) >= 0.5
                correct += (pred == y.bool()).sum().item()
                total += y.size(0)
        accs.append(correct / total if total > 0 else 0.0)
    model.train()
    return accs


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides CLI
    if args.ewc_lambda is not None:
        cfg["ewc"]["lambda"] = args.ewc_lambda
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = get_exp_dir(cfg)

    print(f"\n{'=' * 60}")
    print(f"  EWC Online — {cfg['exp_id']}")
    print(f"  λ={cfg['ewc']['lambda']} | γ={cfg['ewc']['gamma']} | device={device}")
    print(f"{'=' * 60}\n")

    # --- Modèle ---
    model = EWCMlpClassifier(
        input_dim=cfg["model"]["input_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
        ewc_lambda=cfg["ewc"]["lambda"],
        ewc_gamma=cfg["ewc"]["gamma"],
    ).to(device)

    print(model.summary())
    budget_check = model.check_ram_budget()
    print(f"\nBudget RAM : {budget_check['utilization_pct']:.1f}% utilisé")

    if args.dry_run:
        print("\n✅ Dry run terminé — config et architecture OK.")
        return

    # --- Data ---
    tasks = make_dummy_cl_stream(cfg)
    T = len(tasks)

    # --- Sauvegarde config ---
    save_config_snapshot(cfg, str(exp_dir))

    # --- Boucle CL ---
    acc_matrix = np.full((T, T), np.nan)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
    )

    for task_id, task in enumerate(tasks):
        print(f"\n--- Tâche {task_id + 1}/{T} : {task['name']} ---")

        # Entraînement
        model.train()
        for epoch in range(cfg["training"]["epochs_per_task"]):
            epoch_losses = []
            for x_batch, y_batch in task["train"]:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                total_loss, components = model.ewc_loss(x_batch, y_batch)
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(components["total"])

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1:2d}/{cfg['training']['epochs_per_task']} "
                      f"| Loss: {np.mean(epoch_losses):.4f} "
                      f"(BCE: {components['bce']:.4f}, EWC: {components['ewc_reg']:.4f})")

        # Mise à jour EWC après la tâche
        model.update_ewc_state(task["train"], device)
        print(f"  ✅ État EWC mis à jour (Fisher + snapshot)")

        # Évaluation sur toutes les tâches vues
        accs = evaluate_on_all_tasks(model, tasks, task_id, device)
        for j, acc in enumerate(accs):
            acc_matrix[task_id, j] = acc
        print(f"  Accuracy : {' | '.join(f'{tasks[j][\"name\"]}: {accs[j]:.3f}' for j in range(len(accs)))}")

    # --- Métriques finales ---
    metrics = compute_cl_metrics(acc_matrix)
    print(f"\n{format_metrics_report(metrics, 'EWC Online')}")

    # --- Profiling mémoire ---
    mem_report = full_memory_report(
        model,
        input_shape=(1, cfg["model"]["input_dim"]),
        model_name="EWC Online MLP",
    )

    # --- Sauvegarde résultats ---
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(
        metrics,
        str(results_dir / "metrics.json"),
        extra_info={
            "exp_id": cfg["exp_id"],
            "model": "EWC Online MLP",
            "dataset": "equipment_monitoring",
            "memory": mem_report["forward"],
            "ewc_lambda": cfg["ewc"]["lambda"],
        },
    )

    # Sauvegarde checkpoint
    model.save_state(str(exp_dir / "checkpoint.pt"))
    print(f"\n✅ Expérience terminée → {exp_dir}")


if __name__ == "__main__":
    main()
