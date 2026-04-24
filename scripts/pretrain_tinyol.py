"""
Pré-entraînement offline du backbone TinyOL (autoencoder MLP).

Usage :
    python scripts/pretrain_tinyol.py --config configs/tinyol_config.yaml

Sorties :
    experiments/exp_003_tinyol_dataset1/backbone.pt
    configs/pump_normalizer.yaml
    experiments/exp_003_tinyol_dataset1/pretrain_loss_curve.png
    experiments/exp_003_tinyol_dataset1/config_snapshot.yaml

Références :
    Ren2021TinyOL, tinyol_spec.md §5
"""

from __future__ import annotations

import argparse
import sys
from math import inf
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif — compatible serveur
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pump_dataset import CLStreamSplitter, PumpMaintenanceDataset
from src.models.tinyol import TinyOLAutoencoder
from src.utils.config_loader import get_exp_dir, load_config, save_config_snapshot
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pré-entraînement backbone TinyOL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tinyol_config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    return parser.parse_args()


def _get_pretrain_data_pump(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge les données saines du Pump dataset pour le pré-entraînement."""
    dataset = PumpMaintenanceDataset(Path(config["data"]["csv_path"]))
    dataset.load()
    features, labels = dataset.extract_features(
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"],
    )

    cl_stream = CLStreamSplitter(
        features,
        labels,
        n_tasks=config["data"]["n_tasks"],
        strategy=config["data"]["task_strategy"],
    )

    # Normalisation Z-score fixée sur Task 1 (données saines)
    normalizer = cl_stream.fit_normalizer(task_id=0)
    # Sauvegarder les stats avant normalisation (format compatible MCU)
    cl_stream.save_normalizer(Path(config["data"]["normalizer_path"]), normalizer)
    cl_stream.apply_normalizer(normalizer)

    # Filtrage : premières pretrain_fraction de Task 1 = état sain (avant drift)
    x_task0, _ = cl_stream._slices[0]  # [N_task0, 25] normalisé  # noqa: N806
    n_task0 = len(x_task0)
    n_pretrain = int(n_task0 * config["pretrain"]["pretrain_fraction"])
    n_val = int(n_task0 * config["pretrain"]["val_fraction"])

    # MEM: x_train shape [n_pretrain, 25] @ FP32
    x_train = torch.from_numpy(x_task0[:n_pretrain].copy())
    # MEM: x_val shape [n_val, 25] @ FP32
    x_val = torch.from_numpy(x_task0[-n_val:].copy())

    print(f"[Pretrain] Données saines : {n_pretrain} fenêtres (Task 1 : {n_task0} total)")
    print(f"[Pretrain] Validation     : {n_val} fenêtres")
    print(f"[Pretrain] Normalizer sauvegardé → {config['data']['normalizer_path']}")

    return x_train, x_val


def _get_pretrain_data_monitoring(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge les données saines du Monitoring dataset (Task 1 — Pump) pour le pré-entraînement."""
    import numpy as np

    from src.data.monitoring_dataset import (
        DOMAIN_FEATURE,
        DOMAIN_ORDER,
        LABEL_COL,
        NUMERIC_FEATURES,
        load_normalizer,
        load_raw_dataset,
        normalize_features,
    )

    df = load_raw_dataset(Path(config["data"]["csv_path"]))
    normalizer = load_normalizer(Path(config["data"]["normalizer_path"]))
    df = normalize_features(df, normalizer)

    # Task 1 (Pump) — données saines uniquement (label == 0)
    df_healthy = df[(df[DOMAIN_FEATURE] == DOMAIN_ORDER[0]) & (df[LABEL_COL] == 0)]
    x_np = df_healthy[NUMERIC_FEATURES].values.astype(np.float32)
    n = len(x_np)
    n_pretrain = int(n * config["pretrain"]["pretrain_fraction"])
    n_val = int(n * config["pretrain"]["val_fraction"])

    print(f"[Pretrain] Données saines ({DOMAIN_ORDER[0]}) : {n_pretrain} exemples (Task 1 : {n} total)")
    print(f"[Pretrain] Validation : {n_val} exemples")

    x_train = torch.from_numpy(x_np[:n_pretrain].copy())  # MEM: [n_pretrain, 4] @ FP32
    x_val = torch.from_numpy(x_np[-n_val:].copy())         # MEM: [n_val, 4] @ FP32
    return x_train, x_val


def _get_pretrain_data_pronostia(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge les données saines de la Condition 1 PRONOSTIA pour le pré-entraînement."""
    import numpy as np

    from src.data.pronostia_dataset import (
        fit_normalizer,
        load_condition_features,
        load_pronostia_normalizer,
        save_normalizer,
    )

    npy_dir = Path(config["data"]["npy_dir"])
    normalizer_path = Path(config["data"]["normalizer_path"])

    # Condition 1 = données de référence (1 800 rpm, 4 000 N)
    features, labels = load_condition_features(npy_dir, condition=1)

    # Générer le normaliseur si absent
    if not normalizer_path.exists():
        normalizer = fit_normalizer(features)
        save_normalizer(normalizer_path, normalizer)
        print(f"[Pretrain] Normalizer généré → {normalizer_path}")

    normalizer = load_pronostia_normalizer(normalizer_path)
    features = (features - normalizer["mean"]) / normalizer["std"]

    # Données saines uniquement (label=0 = premières 90% des fenêtres)
    healthy_mask = labels == 0
    x_healthy = features[healthy_mask].astype(np.float32)

    n_total = len(x_healthy)
    n_pretrain = int(n_total * config["pretrain"]["pretrain_fraction"])
    n_val = int(n_total * config["pretrain"]["val_fraction"])

    x_train = torch.from_numpy(x_healthy[:n_pretrain].copy())
    x_val = torch.from_numpy(x_healthy[-n_val:].copy())

    print(f"[Pretrain] Données saines Cond.1 : {n_pretrain} fenêtres (total sain : {n_total})")
    print(f"[Pretrain] Validation             : {n_val} fenêtres")

    return x_train, x_val


def get_pretrain_data(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Charge et filtre les données normales pour le pré-entraînement.

    Dispatche vers le loader approprié selon ``config["data"]["dataset"]``.

    Parameters
    ----------
    config : dict
        Configuration lue depuis le fichier YAML.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        X_train et X_val, dtype float32, normalisés.
    """
    dataset = config["data"].get("dataset", "pump_maintenance")
    if dataset == "equipment_monitoring":
        return _get_pretrain_data_monitoring(config)
    if dataset == "pronostia":
        return _get_pretrain_data_pronostia(config)
    return _get_pretrain_data_pump(config)


def _save_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    path: Path,
) -> None:
    """Sauvegarde la courbe de convergence MSE train/val."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train MSE", color="steelblue")
    ax.plot(val_losses, label="Val MSE", color="coral", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("TinyOL Backbone — Courbe de pré-entraînement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Pretrain] Courbe sauvegardée → {path}")


def pretrain(config: dict) -> None:
    """
    Pré-entraîne le backbone TinyOL sur les données normales.

    Paramètres lus depuis configs/tinyol_config.yaml :
      - pretrain.epochs, pretrain.learning_rate, pretrain.batch_size
      - pretrain.pretrain_fraction, pretrain.val_fraction, pretrain.seed
      - backbone.input_dim, backbone.encoder_dims, backbone.decoder_dims
      - backbone.checkpoint_path

    Conforme à tinyol_spec.md §5.

    Parameters
    ----------
    config : dict
        Configuration lue depuis tinyol_config.yaml.
    """
    set_seed(config["pretrain"]["seed"])

    # 1. Données
    x_train, x_val = get_pretrain_data(config)

    train_loader = DataLoader(
        TensorDataset(x_train, x_train),  # cible = entrée (autoencoder)
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["pretrain"]["seed"]),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, x_val),
        batch_size=config["pretrain"]["batch_size"],
        shuffle=False,
    )

    # 2. Modèle
    model = TinyOLAutoencoder(
        input_dim=config["backbone"]["input_dim"],
        encoder_dims=tuple(config["backbone"]["encoder_dims"]),
        decoder_dims=tuple(config["backbone"]["decoder_dims"]),
    )
    print(f"[Pretrain] Modèle : {model.n_encoder_params()} params encodeur")

    # 3. Optimiseur — Adam autorisé ici (phase offline sur PC uniquement)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["pretrain"]["learning_rate"],
    )

    # 4. Préparer les répertoires de sortie
    exp_dir = get_exp_dir(config)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["backbone"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Boucle d'entraînement
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = inf
    epochs = config["pretrain"]["epochs"]

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for x_batch, _ in train_loader:
            optimizer.zero_grad()
            _, x_hat = model(x_batch)
            loss = model.reconstruction_loss(x_batch, x_hat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                _, x_hat = model(x_val)
                val_loss += model.reconstruction_loss(x_val, x_hat).item()
        val_losses.append(val_loss / max(len(val_loader), 1))

        # Sauvegarde meilleur checkpoint
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 10 == 0:
            best_marker = " ← best" if val_losses[-1] == best_val_loss else ""
            print(
                f"Epoch {epoch + 1:3d}/{epochs}"
                f" | Train MSE: {train_losses[-1]:.6f}"
                f" | Val MSE: {val_losses[-1]:.6f}"
                f"{best_marker}"
            )

    print(f"[Pretrain] Meilleur val MSE : {best_val_loss:.6f}")
    print(f"[Pretrain] backbone.pt sauvegardé → {checkpoint_path}")

    # 6. Courbe de convergence
    _save_loss_curve(train_losses, val_losses, exp_dir / "pretrain_loss_curve.png")

    # 7. Config snapshot (reproductibilité)
    save_config_snapshot(config, str(exp_dir))

    print("[Pretrain] Terminé ✓")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pretrain(config)


if __name__ == "__main__":
    main()
