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
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

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


def run_experiment(config: dict) -> None:
    """Exécute l'expérience TinyOL complète (pré-entraînement exclu)."""
    exp_id = config.get("exp_id", "exp")
    seed = config["evaluation"]["seed"]
    set_seed(seed)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prérequis : backbone.pt ---
    checkpoint_path = Path(config["backbone"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Backbone introuvable : {checkpoint_path}\n"
            f"Lancer d'abord : python scripts/pretrain_tinyol.py --config <config>"
        )

    # --- Données ---
    print(f"[{exp_id}] Chargement des données...")
    task_loaders = _load_tasks(config, seed)
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
