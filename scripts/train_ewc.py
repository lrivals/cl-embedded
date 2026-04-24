"""
scripts/train_ewc.py — Script principal pour l'entraînement EWC Online sur Dataset 2.

Usage :
    python scripts/train_ewc.py --config configs/ewc_config.yaml
    python scripts/train_ewc.py --config configs/ewc_config.yaml --device cpu
    python scripts/train_ewc.py --config configs/ewc_config.yaml --skip-baselines
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.memory_profiler import full_memory_report
from src.evaluation.metrics import compute_cl_metrics, format_metrics_report, save_metrics
from src.models.ewc import EWCMlpClassifier
from src.models.ewc.fisher import compute_fisher_diagonal, update_fisher_online
from src.training.baselines import evaluate_task, train_joint, train_naive_sequential
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def _get_tasks(cfg: dict) -> list[dict]:
    """Charge les tâches CL selon le dataset spécifié dans config."""
    dataset = cfg["data"].get("dataset", "equipment_monitoring")

    if dataset == "pronostia":
        from src.data.pronostia_dataset import (
            get_pronostia_dataloaders,
            get_pronostia_dataloaders_single_task,
        )
        npy_dir = Path(cfg["data"]["npy_dir"])
        batch_size = cfg["training"]["batch_size"]
        seed = cfg["training"]["seed"]
        task_split = cfg["data"].get("task_split", "by_condition")
        if task_split == "no_split":
            st = get_pronostia_dataloaders_single_task(
                npy_dir=npy_dir,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
                window_size=cfg["data"].get("window_size", 2560),
                step_size=cfg["data"].get("step_size", 2560),
                failure_ratio=cfg["data"].get("failure_ratio", 0.10),
            )
            st["_single_task_mode"] = True
            return [st]
        return get_pronostia_dataloaders(
            npy_dir=npy_dir,
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=batch_size,
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=seed,
            window_size=cfg["data"].get("window_size", 2560),
            step_size=cfg["data"].get("step_size", 2560),
            failure_ratio=cfg["data"].get("failure_ratio", 0.10),
        )

    csv_path = Path(cfg["data"]["csv_path"])
    normalizer_path = Path(cfg["data"]["normalizer_path"])
    batch_size = cfg["training"]["batch_size"]
    val_ratio = cfg["data"]["test_split"]
    seed = cfg["training"]["seed"]

    if dataset == "pump_maintenance":
        task_split = cfg["data"].get("task_split", "chronological")
        if task_split == "by_pump_id":
            from src.data.pump_dataset import get_pump_dataloaders_by_id
            return get_pump_dataloaders_by_id(
                csv_path=str(csv_path),
                normalizer_path=str(normalizer_path),
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
        elif task_split == "by_temporal_window":
            from src.data.pump_dataset import get_pump_dataloaders_by_temporal_window
            return get_pump_dataloaders_by_temporal_window(
                csv_path=str(csv_path),
                normalizer_path=str(normalizer_path),
                n_tasks=cfg["data"].get("n_tasks", 4),
                entries_per_task=cfg["data"].get("entries_per_task", 5000),
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
        elif task_split == "no_split":
            from src.data.pump_dataset import get_pump_dataloaders_single_task
            st = get_pump_dataloaders_single_task(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
                window_size=cfg["data"]["window_size"],
                step_size=cfg["data"]["step_size"],
            )
            st["_single_task_mode"] = True
            return [st]
        from src.data.pump_dataset import get_pump_dataloaders
        return get_pump_dataloaders(
            csv_path=csv_path,
            normalizer_path=normalizer_path,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
            window_size=cfg["data"]["window_size"],
            step_size=cfg["data"]["step_size"],
        )
    else:
        task_split = cfg["data"].get("task_split", "by_equipment")
        if task_split == "no_split":
            from src.data.monitoring_dataset import get_monitoring_dataloaders_single_task
            st = get_monitoring_dataloaders_single_task(
                csv_path=csv_path,
                batch_size=batch_size,
                test_ratio=cfg["data"].get("test_ratio", 0.2),
                val_ratio=cfg["data"].get("val_ratio", 0.1),
                seed=seed,
            )
            st["_single_task_mode"] = True
            return [st]
        if task_split == "by_location":
            from src.data.monitoring_dataset import get_cl_dataloaders_by_location
            return get_cl_dataloaders_by_location(
                csv_path=csv_path,
                normalizer_path=normalizer_path,
                batch_size=batch_size,
                val_ratio=val_ratio,
                seed=seed,
                location_order=cfg["data"].get("location_order"),
            )
        from src.data.monitoring_dataset import get_cl_dataloaders
        return get_cl_dataloaders(
            csv_path=csv_path,
            normalizer_path=normalizer_path,
            batch_size=batch_size,
            val_ratio=val_ratio,
            seed=seed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EWC Online Training — Dataset 2")
    parser.add_argument("--config", default="configs/ewc_config.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Sauter les baselines (naive + joint) pour un test rapide",
    )
    parser.add_argument(
        "--data_config",
        default=None,
        help="Config data override (ex. configs/pump_by_id_config.yaml)",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Répertoire expérience override (ex. experiments/exp_013_ewc_pump_by_id)",
    )
    return parser.parse_args()


def train_ewc(
    model: EWCMlpClassifier,
    tasks: list[dict],
    config: dict,
    device: str,
) -> np.ndarray:
    """
    Boucle d'entraînement EWC Online sur T tâches séquentielles.

    Parameters
    ----------
    model : EWCMlpClassifier
        Modèle à entraîner (modifié in-place).
    tasks : list[dict]
        Liste de dicts avec clés task_id, domain, train_loader, val_loader.
    config : dict
        Config complète chargée depuis ewc_config.yaml.
    device : str
        Device torch ("cpu" ou "cuda").

    Returns
    -------
    np.ndarray [T, T]
        acc_matrix[i, j] = accuracy sur la tâche j après entraînement sur la tâche i.
        NaN pour j > i (tâches futures non encore vues).
    """
    T = len(tasks)
    acc_matrix = np.full((T, T), np.nan)

    epochs = config["training"]["epochs_per_task"]
    lr = config["training"]["learning_rate"]
    momentum = config["training"]["momentum"]
    ewc_lambda = config["ewc"]["lambda"]
    gamma = config["ewc"]["gamma"]
    n_fisher_samples = config["ewc"]["n_fisher_samples"]

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    fisher = None      # Pas de régularisation pour la tâche 1
    theta_star = None  # Pas de snapshot pour la tâche 1

    model.to(device)

    for i, task in enumerate(tasks):
        domain = task.get("domain", f"Tâche {task['task_id']}")
        print(f"\n--- Tâche {i + 1}/{T} : {domain} ---")

        model.train()
        for epoch in range(epochs):
            epoch_losses = []
            for x, y in task["train_loader"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model.ewc_loss(x, y, fisher, theta_star, ewc_lambda)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"  Epoch {epoch + 1:2d}/{epochs}"
                    f" | Loss moy: {np.mean(epoch_losses):.4f}"
                )

        # Fin de tâche : snapshot θ* + mise à jour Fisher Online
        theta_star = model.get_theta_star()
        new_fisher = compute_fisher_diagonal(
            model, task["train_loader"], device, n_samples=n_fisher_samples
        )
        fisher = update_fisher_online(fisher, new_fisher, gamma=gamma)
        print(f"  Fisher mis à jour (γ={gamma})")

        # Évaluation sur toutes les tâches vues jusqu'ici
        model.eval()
        accs = []
        for j in range(i + 1):
            acc = evaluate_task(model, tasks[j]["val_loader"], device)
            acc_matrix[i, j] = acc
            accs.append(f"{tasks[j].get('domain', f'T{j+1}')}: {acc:.3f}")
        print(f"  Accuracy : {' | '.join(accs)}")

    return acc_matrix


def run_memory_profiling(
    model: EWCMlpClassifier,
    config: dict,
    device: str,
) -> dict:
    """
    Profile le forward pass et la mise à jour CL pour le rapport Gap 2.

    Parameters
    ----------
    model : EWCMlpClassifier
        Modèle après entraînement complet.
    config : dict
        Config pour récupérer input_dim.
    device : str
        Device torch.

    Returns
    -------
    dict
        Rapport complet (forward + update) depuis full_memory_report().
    """
    input_dim = config["model"]["input_dim"]

    def update_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = model.ewc_loss(x, y, None, None, ewc_lambda=0.0)
        loss.backward()
        optimizer.step()
        return loss.item()

    return full_memory_report(
        model,
        input_shape=(1, input_dim),
        update_fn=update_fn,
        model_name="EWCMlpClassifier",
    )


def _fresh_model(config: dict) -> EWCMlpClassifier:
    """Crée un nouveau EWCMlpClassifier depuis la config (poids réinitialisés)."""
    return EWCMlpClassifier(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        dropout=config["model"]["dropout"],
    )


def _run_single_task_ewc(
    model: EWCMlpClassifier,
    data: dict,
    cfg: dict,
    device: str,
    results_dir: Path,
    exp_dir: Path,
) -> None:
    """
    Entraîne le MLP EWC sur une seule tâche (tout le dataset monitoring fusionné).

    Pas de pénalité EWC (pas de tâches précédentes) — entraînement BCE standard.
    Sortie : ``results/metrics_single_task.json`` avec les 6 métriques obligatoires.

    Parameters
    ----------
    model : EWCMlpClassifier
    data : dict
        Sortie de get_monitoring_dataloaders_single_task().
    cfg : dict
        Config complète.
    device : str
    results_dir : Path
    exp_dir : Path
    """
    epochs = cfg["training"]["epochs_per_task"] * 3  # plus d'epochs — pas de CL
    lr = cfg["training"]["learning_rate"]
    momentum = cfg["training"]["momentum"]

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # pos_weight corrige le déséquilibre de classes (ex. 9.0 si ~10% faulty)
    pos_weight_val = cfg["training"].get("pos_weight", None)
    if pos_weight_val is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        print(f"  pos_weight={pos_weight_val:.1f} (correction déséquilibre de classes)")
    else:
        criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    print(f"\n{'=' * 40}")
    print(f"  EWC Single-Task — {cfg['exp_id']}")
    print(f"  {data['n_train']} train | {data['n_val']} val | {data['n_test']} test")
    print(f"  {epochs} epochs | lr={lr} | momentum={momentum}")
    print(f"{'=' * 40}")

    for epoch in range(epochs):
        model.train()
        for x, y in data["train_loader"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            val_probs_ep, val_true_ep = [], []
            with torch.no_grad():
                for x, y in data["val_loader"]:
                    logits = model(x.to(device))
                    val_probs_ep.extend(torch.sigmoid(logits).cpu().numpy().ravel())
                    val_true_ep.extend(y.numpy().ravel())
            # F1 sur val (seuil 0.5) — indicateur de progression
            val_pred_ep = (np.array(val_probs_ep) > 0.5).astype(int)
            val_f1 = f1_score(np.array(val_true_ep), val_pred_ep, zero_division=0)
            print(f"  Epoch {epoch + 1:3d}/{epochs} | val_f1={val_f1:.4f}")

    # Seuil optimal sur val : maximise F1 (important avec pos_weight ou déséquilibre)
    model.eval()
    val_probs_all, val_true_all = [], []
    with torch.no_grad():
        for x, y in data["val_loader"]:
            logits = model(x.to(device))
            val_probs_all.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_true_all.extend(y.numpy().ravel().astype(int))

    val_probs_arr = np.array(val_probs_all)
    val_true_arr = np.array(val_true_all)
    # Cherche le seuil qui maximise F1 sur 50 candidats
    thresholds = np.linspace(0.01, 0.99, 50)
    best_thresh = 0.5
    best_f1_val = 0.0
    for t in thresholds:
        f1_t = f1_score(val_true_arr, (val_probs_arr > t).astype(int), zero_division=0)
        if f1_t > best_f1_val:
            best_f1_val, best_thresh = f1_t, t
    print(f"\n  Seuil optimal (val F1={best_f1_val:.4f}) : {best_thresh:.3f}")

    # Évaluation finale sur test avec seuil optimal
    test_probs, test_true = [], []
    with torch.no_grad():
        for x, y in data["test_loader"]:
            logits = model(x.to(device))
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_true.extend(y.numpy().ravel().astype(int))

    y_true = np.array(test_true)
    y_scores = np.array(test_probs)
    y_pred = (y_scores > best_thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = float("nan")

    print(f"\n  Test → accuracy={acc:.4f} | f1={f1:.4f} | auc_roc={auc:.4f}")

    # Profiling mémoire
    print("\n  Profiling mémoire...")
    memory_report = run_memory_profiling(model, cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics: dict = {
        "exp_id": cfg["exp_id"],
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "ram_peak_bytes": memory_report.get("forward", {}).get("ram_peak_bytes", 0),
        "inference_latency_ms": memory_report.get("forward", {}).get("inference_latency_ms", 0.0),
        "n_params": n_params,
        "threshold": float(best_thresh),  # seuil optimal calculé sur val
    }

    metrics_path = results_dir / "metrics_single_task.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")
    print(f"✅ EWC single-task terminé → {exp_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Override section data depuis --data_config (ex. pump_by_id_config.yaml)
    if args.data_config:
        data_cfg = load_config(args.data_config)
        cfg["data"].update(data_cfg.get("data", {}))

    # Override répertoire expérience depuis --exp_dir
    if args.exp_dir:
        cfg["evaluation"]["output_dir"] = str(Path(args.exp_dir) / "results")
        cfg["exp_id"] = Path(args.exp_dir).name

    device = args.device

    set_seed(cfg["training"]["seed"])

    exp_dir = get_exp_dir(cfg)
    results_dir = exp_dir / "results"
    checkpoints_dir = exp_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  EWC Online — {cfg['exp_id']}")
    print(f"  λ={cfg['ewc']['lambda']} | γ={cfg['ewc']['gamma']} | device={device}")
    print(f"  Sortie : {exp_dir}")
    print(f"{'=' * 60}\n")

    # Sauvegarde snapshot config
    save_config_snapshot(cfg, str(exp_dir))
    print("Config snapshot sauvegardé.")

    # Chargement données
    dataset = cfg["data"].get("dataset", "equipment_monitoring")
    if dataset != "pronostia":
        csv_path = Path(cfg["data"]["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")
        print(f"Dataset : {csv_path}")
    else:
        print(f"Dataset : {cfg['data']['npy_dir']} (FEMTO PRONOSTIA .npy)")

    tasks = _get_tasks(cfg)
    T = len(tasks)

    # Mode single-task (task_split: no_split) — baseline hors-CL
    if tasks[0].get("_single_task_mode"):
        _run_single_task_ewc(
            model=_fresh_model(cfg),
            data=tasks[0],
            cfg=cfg,
            device=device,
            results_dir=results_dir,
            exp_dir=exp_dir,
        )
        return

    labels = [t.get("domain", f"T{t['task_id']}") for t in tasks]
    print(f"Tâches chargées : {labels} ({T} tâches)\n")

    # ── EWC ──────────────────────────────────────────────────────────────────
    print("=" * 40)
    print("  Entraînement EWC Online")
    print("=" * 40)
    model = _fresh_model(cfg)
    acc_matrix_ewc = train_ewc(model, tasks, cfg, device)
    np.save(results_dir / "acc_matrix_ewc.npy", acc_matrix_ewc)
    metrics_ewc = compute_cl_metrics(acc_matrix_ewc)

    # ── Baselines ─────────────────────────────────────────────────────────────
    metrics_naive: dict | None = None
    metrics_joint: dict | None = None

    if not args.skip_baselines:
        print("\n" + "=" * 40)
        print("  Baseline : Fine-tuning naïf")
        print("=" * 40)
        set_seed(cfg["training"]["seed"])
        naive_model = _fresh_model(cfg)
        acc_matrix_naive = train_naive_sequential(naive_model, tasks, cfg, device)
        np.save(results_dir / "acc_matrix_naive.npy", acc_matrix_naive)
        metrics_naive = compute_cl_metrics(acc_matrix_naive)

        print("\n" + "=" * 40)
        print("  Baseline : Joint training")
        print("=" * 40)
        set_seed(cfg["training"]["seed"])
        joint_model = _fresh_model(cfg)
        acc_matrix_joint = train_joint(joint_model, tasks, cfg, device)
        np.save(results_dir / "acc_matrix_joint.npy", acc_matrix_joint)
        metrics_joint = compute_cl_metrics(acc_matrix_joint)

    # ── Profiling mémoire ─────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  Profiling mémoire")
    print("=" * 40)
    memory_report = run_memory_profiling(model, cfg, device)
    with open(results_dir / "memory_report.json", "w") as f:
        json.dump(memory_report, f, indent=2)

    # ── Sauvegarde résultats ──────────────────────────────────────────────────
    all_metrics: dict = {
        "ewc": metrics_ewc,
        "memory": memory_report,
    }
    if metrics_naive is not None:
        all_metrics["naive"] = metrics_naive
    if metrics_joint is not None:
        all_metrics["joint"] = metrics_joint

    save_metrics(
        all_metrics,
        str(results_dir / "metrics.json"),
        extra_info={"exp_id": cfg["exp_id"]},
    )

    # Checkpoint modèle EWC final
    model.save_state(str(checkpoints_dir / "ewc_task3_final.pt"))

    # ── Rapport stdout ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    report = format_metrics_report(
        metrics_ewc,
        model_name="EWCMlpClassifier",
        baseline_finetune=metrics_naive,
        baseline_joint=metrics_joint,
    )
    print(report)

    # Vérification critère principal
    if metrics_naive is not None:
        aa_ewc = metrics_ewc.get("aa", 0.0)
        aa_naive = metrics_naive.get("aa", 0.0)
        if aa_ewc > aa_naive:
            print(f"✓  EWC bat le fine-tuning naïf (aa_ewc={aa_ewc:.3f} > aa_naive={aa_naive:.3f})")
        else:
            print(
                f"⚠  EWC ne bat pas le fine-tuning naïf "
                f"(aa_ewc={aa_ewc:.3f} ≤ aa_naive={aa_naive:.3f}) "
                f"— augmenter ewc.lambda ?"
            )

    ram_ok = memory_report.get("forward", {}).get("within_budget_64ko", False)
    print(f"{'✓' if ram_ok else '⚠'}  Budget RAM 64 Ko : {'OK' if ram_ok else 'DÉPASSÉ'}")

    print(f"\n✅ Expérience terminée → {exp_dir}")


if __name__ == "__main__":
    main()
