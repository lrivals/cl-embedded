"""
Entraînement TinyOL en scénario Domain-Incremental sur Dataset 1 (Pump).

Protocole :
  - Tâche 1 : fenêtres t=0..T/3    (pompe saine)
  - Tâche 2 : fenêtres t=T/3..2T/3 (usure naissante)
  - Tâche 3 : fenêtres t=2T/3..T   (pré-panne)
  Après chaque tâche → évaluation sur toutes les tâches vues.

Sorties :
    experiments/exp_003_tinyol_dataset1/results/metrics.json
    experiments/exp_003_tinyol_dataset1/results/cl_accuracy_matrix.png
    experiments/exp_003_tinyol_dataset1/config_snapshot.yaml

Références :
    Ren2021TinyOL, tinyol_spec.md §4 et §6
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.memory_profiler import profile_cl_update, profile_forward_pass
from src.evaluation.metrics import compute_cl_metrics
from src.models.tinyol import OtOHead, TinyOLAutoencoder, TinyOLOnlineTrainer
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement TinyOL — exp_003")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tinyol_config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--data_config",
        default=None,
        help="Config data override (ex. configs/pump_by_id_config.yaml)",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Répertoire expérience override (ex. experiments/exp_012_tinyol_pump_by_id)",
    )
    return parser.parse_args()


def _save_accuracy_matrix(acc_matrix: list[list[float]], path: Path, exp_id: str = "exp") -> None:
    """Sauvegarde la heatmap de la matrice d'accuracy CL."""
    n_tasks = len(acc_matrix)
    mat = np.full((n_tasks, n_tasks), np.nan)
    for i, row in enumerate(acc_matrix):
        for j, val in enumerate(row):
            mat[i, j] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn")
    ax.set_xlabel("Tâche évaluée")
    ax.set_ylabel("Après tâche d'entraînement")
    ax.set_title(f"TinyOL — Matrice accuracy CL ({exp_id})")
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f"T{i + 1}" for i in range(n_tasks)])
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f"T{i + 1}" for i in range(n_tasks)])
    for i in range(n_tasks):
        for j in range(i + 1):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{exp_id}] Accuracy matrix sauvegardée → {path}")


def _save_config_snapshot(config: dict, path: Path, exp_id: str = "exp") -> None:
    """Sauvegarde le snapshot de configuration pour reproductibilité."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[{exp_id}] Config snapshot → {path}")


def _load_tasks(config: dict, seed: int) -> list[dict]:
    """Charge les tâches CL selon config["data"]["dataset"]."""
    dataset = config["data"].get("dataset", "pump_maintenance")
    if dataset == "equipment_monitoring":
        task_split = config["data"].get("task_split", "by_equipment")
        if task_split == "no_split":
            from src.data.monitoring_dataset import get_monitoring_dataloaders_single_task
            st = get_monitoring_dataloaders_single_task(
                csv_path=Path(config["data"]["csv_path"]),
                batch_size=1,  # TinyOL : online, un échantillon à la fois
                test_ratio=config["data"].get("test_ratio", 0.2),
                val_ratio=config["data"].get("val_ratio", 0.1),
                seed=seed,
            )
            st["_single_task_mode"] = True
            return [st]
        if task_split == "by_location":
            from src.data.monitoring_dataset import get_cl_dataloaders_by_location
            return get_cl_dataloaders_by_location(
                csv_path=Path(config["data"]["csv_path"]),
                normalizer_path=Path(config["data"]["normalizer_path"]),
                batch_size=1,
                seed=seed,
                location_order=config["data"].get("location_order"),
            )
        from src.data.monitoring_dataset import get_cl_dataloaders
        return get_cl_dataloaders(
            csv_path=Path(config["data"]["csv_path"]),
            normalizer_path=Path(config["data"]["normalizer_path"]),
            batch_size=1,
            seed=seed,
        )
    task_split = config["data"].get("task_split", "chronological")
    if task_split == "by_pump_id":
        from src.data.pump_dataset import get_pump_dataloaders_by_id
        return get_pump_dataloaders_by_id(
            csv_path=str(config["data"]["csv_path"]),
            normalizer_path=str(config["data"]["normalizer_path"]),
            batch_size=1,
            val_ratio=0.0,
            seed=seed,
            window_size=config["data"]["window_size"],
            step_size=config["data"]["step_size"],
        )
    elif task_split == "by_temporal_window":
        from src.data.pump_dataset import get_pump_dataloaders_by_temporal_window
        return get_pump_dataloaders_by_temporal_window(
            csv_path=str(config["data"]["csv_path"]),
            normalizer_path=str(config["data"]["normalizer_path"]),
            n_tasks=config["data"].get("n_tasks", 4),
            entries_per_task=config["data"].get("entries_per_task", 5000),
            batch_size=1,
            val_ratio=0.0,
            seed=seed,
            window_size=config["data"]["window_size"],
            step_size=config["data"]["step_size"],
        )
    elif task_split == "no_split":
        from src.data.pump_dataset import get_pump_dataloaders_single_task
        st = get_pump_dataloaders_single_task(
            csv_path=Path(config["data"]["csv_path"]),
            normalizer_path=Path(config["data"].get("normalizer_path", "configs/pump_normalizer.yaml")),
            batch_size=1,  # TinyOL : online, un échantillon à la fois
            test_ratio=config["data"].get("test_ratio", 0.2),
            val_ratio=config["data"].get("val_ratio", 0.1),
            seed=seed,
            window_size=config["data"]["window_size"],
            step_size=config["data"]["step_size"],
        )
        st["_single_task_mode"] = True
        return [st]
    from src.data.pump_dataset import get_pump_dataloaders
    return get_pump_dataloaders(
        csv_path=Path(config["data"]["csv_path"]),
        normalizer_path=Path(config["data"]["normalizer_path"]),
        batch_size=1,
        val_ratio=0.0,
        seed=seed,
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"],
    )


def _run_single_task_tinyol(data: dict, config: dict, output_dir: Path) -> None:
    """
    Entraîne TinyOL en mode single-task sur le dataset monitoring complet.

    Construit et pré-entraîne le backbone depuis les données (pas de checkpoint.pt requis),
    puis entraîne la tête OtO en ligne sur le train split.

    Parameters
    ----------
    data : dict
        Sortie de get_monitoring_dataloaders_single_task(), batch_size=1.
    config : dict
        Config complète.
    output_dir : Path
        Répertoire de sortie des résultats.
    """
    exp_id = config.get("exp_id", "exp")

    # Détecter input_dim depuis les données
    first_x = None
    for x_batch, _ in data["train_loader"]:
        first_x = x_batch
        break
    n_features: int = first_x.shape[-1]

    # Dims autoencoder adaptées à n_features
    enc_dims: tuple[int, ...] = (max(8, n_features * 2), max(4, n_features), max(2, n_features // 2))
    embedding_dim: int = enc_dims[-1]
    dec_dims: tuple[int, ...] = (enc_dims[-2], enc_dims[-3], n_features)

    print(f"\n{'=' * 40}")
    print(f"  TinyOL Single-Task — {exp_id}")
    print(f"  n_features={n_features} | enc={enc_dims} | embedding_dim={embedding_dim}")
    print(f"  {data['n_train']} train | {data['n_val']} val | {data['n_test']} test")
    print(f"{'=' * 40}")

    # --- Pré-entraînement autoencoder (MSE, offline) ---
    autoencoder = TinyOLAutoencoder(
        input_dim=n_features,
        encoder_dims=enc_dims,
        decoder_dims=dec_dims,
    )
    pretrain_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    n_pretrain_epochs = config.get("pretrain", {}).get("epochs", 10)
    print(f"\n  Pré-entraînement backbone ({n_pretrain_epochs} epochs)...")
    autoencoder.train()
    for epoch in range(n_pretrain_epochs):
        losses = []
        for x_batch, _ in data["train_loader"]:
            x = x_batch.squeeze(0)
            pretrain_optimizer.zero_grad()
            _, x_hat = autoencoder(x.unsqueeze(0))
            loss = nn.functional.mse_loss(x_hat, x.unsqueeze(0))
            loss.backward()
            pretrain_optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % max(1, n_pretrain_epochs // 5) == 0:
            print(f"    Epoch {epoch + 1:3d}/{n_pretrain_epochs} | MSE={np.mean(losses):.6f}")
    autoencoder.eval()

    # --- Tête OtO ---
    oto_input_dim: int = embedding_dim + 1  # embed + MSE scalaire
    oto_head = OtOHead(input_dim=oto_input_dim)

    cfg_patched = copy.deepcopy(config)
    cfg_patched["oto_head"]["input_dim"] = oto_input_dim

    trainer = TinyOLOnlineTrainer(autoencoder, oto_head, cfg_patched)

    # --- Entraînement online sur train split ---
    print(f"\n  Entraînement OtO online ({data['n_train']} exemples)...")
    for x_batch, y_batch in data["train_loader"]:
        x = x_batch.squeeze(0)
        y = y_batch.squeeze()
        trainer.update(x, y)

    # --- Évaluation sur test split ---
    test_probs, test_preds, test_true = [], [], []
    for x_batch, y_batch in data["test_loader"]:
        x = x_batch.squeeze(0)
        prob, _ = trainer.predict(x)
        pred = int(prob > 0.5)
        test_probs.append(float(prob))
        test_preds.append(pred)
        test_true.append(int(y_batch.squeeze().item()))

    y_true = np.array(test_true)
    y_pred = np.array(test_preds)
    y_scores = np.array(test_probs)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = float("nan")

    print(f"\n  Test → accuracy={acc:.4f} | f1={f1:.4f} | auc_roc={auc:.4f}")

    # --- Profiling mémoire (tête OtO uniquement — ce qui tourne sur MCU) ---
    fwd_profile = profile_forward_pass(oto_head, input_shape=(1, oto_input_dim), n_runs=100)
    ram_peak = fwd_profile.get("ram_peak_bytes", 0)
    latency_ms = fwd_profile.get("inference_latency_ms", 0.0)

    n_params = oto_head.n_params() + autoencoder.n_encoder_params()

    metrics: dict = {
        "exp_id": exp_id,
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "ram_peak_bytes": ram_peak,
        "inference_latency_ms": latency_ms,
        "n_params": n_params,
    }

    metrics_path = output_dir / "metrics_single_task.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Résultats → {metrics_path}")
    print(f"✅ TinyOL single-task terminé → {output_dir.parent}")


def run_experiment(config: dict) -> None:
    """Exécute l'expérience TinyOL complète (pré-entraînement exclu)."""
    exp_id = config.get("exp_id", "exp")
    seed = config["evaluation"]["seed"]
    set_seed(seed)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Données ---
    print(f"[{exp_id}] Chargement des données...")
    task_loaders = _load_tasks(config, seed)
    n_tasks = len(task_loaders)

    # Mode single-task (task_split: no_split) — baseline hors-CL
    if task_loaders[0].get("_single_task_mode"):
        _run_single_task_tinyol(task_loaders[0], config, output_dir)
        _save_config_snapshot(config, output_dir.parent / "config_snapshot.yaml", exp_id=exp_id)
        return

    # --- Prérequis : backbone.pt ---
    checkpoint_path = Path(config["backbone"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Backbone introuvable : {checkpoint_path}\n"
            f"Lancer d'abord : python scripts/pretrain_tinyol.py --config <config>"
        )

    n_tasks = len(task_loaders)  # dynamique : 3 tâches (chronologique) ou 5 (by_pump_id)
    print(f"[{exp_id}] Stream CL : {n_tasks} tâches")
    for t, tl in enumerate(task_loaders):
        print(f"  Tâche {t + 1} : {tl['n_train']} exemples train")

    # --- Modèle : backbone gelé + tête OtO fraîche ---
    autoencoder = TinyOLAutoencoder(
        input_dim=config["backbone"]["input_dim"],
        encoder_dims=tuple(config["backbone"]["encoder_dims"]),
        decoder_dims=tuple(config["backbone"]["decoder_dims"]),
    )
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    autoencoder.eval()

    oto_head = OtOHead(input_dim=config["oto_head"]["input_dim"])
    trainer = TinyOLOnlineTrainer(autoencoder, oto_head, config)

    print(f"[{exp_id}] Backbone chargé depuis {checkpoint_path}")
    print(f"[{exp_id}] OtO params : {oto_head.n_params()}")
    print(f"[{exp_id}] Encodeur params : {autoencoder.n_encoder_params()}")

    # --- Boucle CL ---
    # acc_matrix[i][j] = accuracy sur tâche j après entraînement sur tâches 0..i
    acc_matrix: list[list[float]] = []

    for task_id in range(n_tasks):
        print(f"\n--- Tâche {task_id + 1}/{n_tasks} ---")

        # Entraînement online (1 pass, échantillon par échantillon)
        n_correct_train, total_train = 0, 0
        for batch in task_loaders[task_id]["train_loader"]:
            x = batch[0].squeeze(0)  # [25]
            y = batch[1].squeeze()  # scalaire
            _ = trainer.update(x, y)
            y_hat, _ = trainer.predict(x)
            n_correct_train += int((y_hat > 0.5) == bool(y.item()))
            total_train += 1

        print(f"  Train acc (online) : {n_correct_train / total_train:.4f}")

        # Évaluation sur toutes les tâches vues (0..task_id)
        task_accs = []
        for eval_id in range(task_id + 1):
            correct, total = 0, 0
            for batch in task_loaders[eval_id]["train_loader"]:
                x = batch[0].squeeze(0)
                y_np = batch[1].item()
                y_hat, _ = trainer.predict(x)
                correct += int((y_hat > 0.5) == bool(y_np))
                total += 1
            acc = correct / total
            task_accs.append(acc)
            print(f"  Eval tâche {eval_id + 1} : {acc:.4f}")

        acc_matrix.append(task_accs)

    # --- Métriques CL ---
    mat_np = np.full((n_tasks, n_tasks), np.nan)
    for i, row in enumerate(acc_matrix):
        for j, val in enumerate(row):
            mat_np[i, j] = val
    cl_metrics = compute_cl_metrics(mat_np)

    print(f"\n[{exp_id}] AA  = {cl_metrics['aa']:.4f}")
    print(f"[{exp_id}] AF  = {cl_metrics['af']:.4f}")
    print(f"[{exp_id}] BWT = {cl_metrics['bwt']:.4f}")

    # --- Profiling RAM et latence ---
    # Inférence : tête OtO seule (ce qui tourne sur le MCU)
    fwd_profile = profile_forward_pass(
        oto_head,
        input_shape=(1, config["oto_head"]["input_dim"]),
        n_runs=100,
    )

    # Mise à jour : pipeline complet (backbone encode + OtO update)
    def _update_wrapper(x_dummy: torch.Tensor, y_dummy: torch.Tensor) -> float:
        return trainer.update(x_dummy.squeeze(0), y_dummy.squeeze())

    upd_profile = profile_cl_update(
        _update_wrapper,
        input_shape=(config["backbone"]["input_dim"],),
        label_shape=(1,),
    )

    ram_update_bytes = upd_profile["ram_peak_bytes_update"]
    latency_ms = fwd_profile["inference_latency_ms"]

    print(f"[{exp_id}] RAM peak update   : {ram_update_bytes} B")
    print(f"[{exp_id}] Latence inférence : {latency_ms:.3f} ms")
    # FIXME(gap2) : overhead Python non représentatif MCU — référence uniquement

    # --- Sauvegarde des résultats ---
    results = {
        "exp_id": exp_id,
        "model": "tinyol",
        "dataset": config["data"].get("dataset", "unknown"),
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        # Métriques CL (noms conformes à S306 + tinyol_spec.md §7)
        "acc_final": cl_metrics["aa"],
        "avg_forgetting": cl_metrics["af"],
        "backward_transfer": cl_metrics["bwt"],
        "acc_matrix": acc_matrix,
        # Métriques embarquées
        "ram_peak_bytes": ram_update_bytes,
        "inference_latency_ms": latency_ms,
        "n_params_oto": oto_head.n_params(),
        "n_params_encoder": autoencoder.n_encoder_params(),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{exp_id}] Résultats sauvegardés → {metrics_path}")

    _save_accuracy_matrix(acc_matrix, output_dir / "cl_accuracy_matrix.png", exp_id=exp_id)
    _save_config_snapshot(config, output_dir.parent / "config_snapshot.yaml", exp_id=exp_id)

    print(f"[{exp_id}] Terminé ✓")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override section data depuis --data_config (ex. pump_by_id_config.yaml)
    if args.data_config:
        data_cfg = load_config(args.data_config)
        config["data"].update(data_cfg.get("data", {}))

    # Override répertoire expérience depuis --exp_dir
    if args.exp_dir:
        config["evaluation"]["output_dir"] = str(Path(args.exp_dir) / "results")
        config["exp_id"] = Path(args.exp_dir).name

    run_experiment(config)


if __name__ == "__main__":
    main()
