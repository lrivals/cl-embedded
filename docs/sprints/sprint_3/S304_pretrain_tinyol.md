# S3-04 — Pré-entraînement backbone (données normales uniquement)

| Champ | Valeur |
|-------|--------|
| **ID** | S3-04 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S3-02 (`pump_dataset.py` opérationnel) + S3-03 (`autoencoder.py` opérationnel) |
| **Fichier cible** | `scripts/pretrain_tinyol.py` |
| **Complété le** | — |

---

## Objectif

Implémenter le script CLI `scripts/pretrain_tinyol.py` qui :
1. Charge le Dataset 1 via `PumpMaintenanceDataset` et `CLStreamSplitter` (S3-02)
2. Filtre les données **normales uniquement** (fraction `pretrain_fraction=0.3` = 30% les plus anciens, avant drift)
3. Entraîne le backbone `TinyOLAutoencoder` en batch (Adam, MSE, 50 epochs) sur ces données saines
4. Sauvegarde les trois artefacts de sortie dans `experiments/exp_003_tinyol_dataset1/`

**Tous les hyperparamètres sont lus depuis `configs/tinyol_config.yaml`. Aucune valeur hardcodée dans le script.**

**Critère de succès** : `python scripts/pretrain_tinyol.py --config configs/tinyol_config.yaml` s'exécute sans erreur, produit `backbone.pt`, `pump_normalizer.yaml` (committé), et une courbe de convergence qui décroît.

---

## Sorties du script

| Fichier | Chemin | Committable | Utilisé par |
|---------|--------|-------------|-------------|
| Poids backbone gelé | `experiments/exp_003_tinyol_dataset1/backbone.pt` | ❌ (gitignore experiments/) | S3-05, S3-06 |
| Stats normalisation | `configs/pump_normalizer.yaml` | ✅ | `pump_dataset.py`, MCU |
| Courbe convergence | `experiments/exp_003_tinyol_dataset1/pretrain_loss_curve.png` | ❌ | Documentation |
| Config snapshot | `experiments/exp_003_tinyol_dataset1/config_snapshot.yaml` | ❌ | Reproductibilité |

---

## Sous-tâches

### 1. Structure du script

```python
# scripts/pretrain_tinyol.py
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
import argparse
import json
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pré-entraînement backbone TinyOL")
    parser.add_argument("--config", type=str, default="configs/tinyol_config.yaml",
                        help="Chemin vers le fichier de configuration YAML")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    pretrain(config)
```

### 2. Filtrage des données normales (pré-entraînement)

```python
from src.data.pump_dataset import PumpMaintenanceDataset, CLStreamSplitter
from src.utils.reproducibility import set_seed


def get_pretrain_data(config: dict) -> tuple:
    """
    Charge et filtre les données normales pour le pré-entraînement.

    Stratégie : les premières `pretrain_fraction` de la Task 1 correspondent
    à l'état sain de la pompe (avant drift). Pas de labels utilisés — apprentissage
    non supervisé (autoencoder MSE).

    Conforme à tinyol_spec.md §5 : "Données utilisées : première portion du dataset
    (avant drift). Label utilisé : aucun."
    """
    # Chargement et feature engineering
    dataset = PumpMaintenanceDataset(config["data"]["path"])
    features, labels = dataset.extract_features(
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"],
    )

    # Split CL (chronologique)
    cl_stream = CLStreamSplitter(
        features, labels,
        n_tasks=config["data"]["n_tasks"],
        strategy=config["data"]["task_strategy"],
    )

    # Normalisation sur Task 1
    normalizer = cl_stream.fit_normalizer(task_id=0)
    cl_stream.apply_normalizer(normalizer)
    cl_stream.save_normalizer(config["data"]["normalizer_path"])

    # Filtrage : prendre uniquement les premières pretrain_fraction de Task 1
    # Ces données correspondent à l'état sain (avant tout drift de dégradation)
    task0_start, task0_end = cl_stream._task_slices[0]
    n_task0 = task0_end - task0_start
    n_pretrain = int(n_task0 * config["pretrain"]["pretrain_fraction"])

    X_pretrain = torch.tensor(
        cl_stream.features[task0_start : task0_start + n_pretrain],
        dtype=torch.float32,
    )
    # MEM: X_pretrain shape [n_pretrain, 25] @ FP32

    # Validation / fraction finale de Task 1
    n_val = int(n_task0 * config["pretrain"]["val_fraction"])
    X_val = torch.tensor(
        cl_stream.features[task0_end - n_val : task0_end],
        dtype=torch.float32,
    )

    print(f"[Pretrain] Données saines : {n_pretrain} fenêtres")
    print(f"[Pretrain] Validation : {n_val} fenêtres")
    print(f"[Pretrain] Normalizer sauvegardé → {config['data']['normalizer_path']}")

    return X_pretrain, X_val, cl_stream
```

### 3. Boucle de pré-entraînement

```python
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")   # pas d'affichage interactif — compatible serveur
import matplotlib.pyplot as plt

from src.models.tinyol import TinyOLAutoencoder


def pretrain(config: dict) -> None:
    """
    Pré-entraîne le backbone TinyOL sur les données normales.

    Paramètres lus depuis configs/tinyol_config.yaml :
      - pretrain.epochs, pretrain.learning_rate, pretrain.batch_size
      - pretrain.pretrain_fraction, pretrain.val_fraction
      - backbone.input_dim, backbone.encoder_dims, backbone.decoder_dims
      - backbone.checkpoint_path

    Conforme à tinyol_spec.md §5.
    """
    set_seed(config["pretrain"]["seed"])

    # 1. Données
    X_train, X_val, _ = get_pretrain_data(config)

    train_loader = DataLoader(
        TensorDataset(X_train, X_train),   # x → x (autoencoder : cible = entrée)
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["pretrain"]["seed"]),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, X_val),
        batch_size=config["pretrain"]["batch_size"],
        shuffle=False,
    )

    # 2. Modèle
    model = TinyOLAutoencoder(
        input_dim=config["backbone"]["input_dim"],
        encoder_dims=tuple(config["backbone"]["encoder_dims"]),
        decoder_dims=tuple(config["backbone"]["decoder_dims"]),
    )
    print(f"[Pretrain] Modèle : {model.n_encoder_params()} params encodeur (attendu : 1 496)")

    # 3. Optimiseur — Adam autorisé ici (phase offline sur PC uniquement)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["pretrain"]["learning_rate"],
    )

    # 4. Boucle d'entraînement
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["backbone"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["pretrain"]["epochs"]):
        # Train
        model.train()
        epoch_loss = 0.0
        for x_batch, _ in train_loader:
            optimizer.zero_grad()
            z, x_hat = model(x_batch)
            loss = model.reconstruction_loss(x_batch, x_hat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                _, x_hat = model(x_val)
                val_loss += model.reconstruction_loss(x_val, x_hat).item()
        val_losses.append(val_loss / max(len(val_loader), 1))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            # Sauvegarder checkpoint complet (sera gelé ensuite)
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{config['pretrain']['epochs']} "
                f"| Train MSE: {train_losses[-1]:.6f} "
                f"| Val MSE: {val_losses[-1]:.6f}"
                f"{' ← best' if val_losses[-1] == best_val_loss else ''}"
            )

    print(f"[Pretrain] Meilleur val MSE : {best_val_loss:.6f}")
    print(f"[Pretrain] backbone.pt sauvegardé → {checkpoint_path}")

    # 5. Courbe de convergence
    _save_loss_curve(train_losses, val_losses, output_dir / "pretrain_loss_curve.png")

    # 6. Config snapshot (reproductibilité)
    _save_config_snapshot(config, output_dir / "config_snapshot.yaml")

    print("[Pretrain] Terminé ✓")
```

### 4. Helpers

```python
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


def _save_config_snapshot(config: dict, path: Path) -> None:
    """Sauvegarde un snapshot de la config pour la reproductibilité."""
    import datetime
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
    }
    with open(path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
```

---

## Critères d'acceptation

- [ ] `python scripts/pretrain_tinyol.py --config configs/tinyol_config.yaml` s'exécute sans erreur
- [ ] `experiments/exp_003_tinyol_dataset1/backbone.pt` créé et chargeable avec `torch.load()`
- [ ] `configs/pump_normalizer.yaml` généré (YAML valide, committable)
- [ ] `experiments/exp_003_tinyol_dataset1/pretrain_loss_curve.png` créé — courbe décroissante
- [ ] `experiments/exp_003_tinyol_dataset1/config_snapshot.yaml` créé (reproductibilité)
- [ ] Val MSE finale < Train MSE initiale (convergence effective)
- [ ] Aucun hyperparamètre hardcodé dans le script — tout vient de `tinyol_config.yaml`
- [ ] `set_seed(config["pretrain"]["seed"])` appelé en début de script
- [ ] `ruff check scripts/pretrain_tinyol.py` + `black --check` passent

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter | Statut |
|---------|-------------|--------|
| Val MSE finale | `experiments/exp_003_tinyol_dataset1/results/metrics.json` | ⬜ S3-06 |
| N fenêtres pré-entraînement | Commentaire dans `pump_dataset.py` | ⬜ |
| `configs/pump_normalizer.yaml` | Versionné dans git (committable) | ⬜ |
| `backbone.pt` path | `configs/tinyol_config.yaml` → `backbone.checkpoint_path` | ✅ déjà configuré |

---

## Commande complète

```bash
# Setup
pip install -e ".[dev]"

# Étape 1 — Télécharger le dataset (S3-01)
# Étape 2 — Vérifier que pump_dataset.py et autoencoder.py sont en place (S3-02, S3-03)

# Étape 3 — Lancer le pré-entraînement
python scripts/pretrain_tinyol.py --config configs/tinyol_config.yaml

# Vérification des sorties
ls experiments/exp_003_tinyol_dataset1/
# Attendu : backbone.pt, pretrain_loss_curve.png, config_snapshot.yaml

ls configs/pump_normalizer.yaml
# Attendu : fichier YAML avec mean/std par feature
```

---

## Questions ouvertes

- `TODO(arnaud)` : `pretrain_fraction=0.3` (30% les plus anciens) — confirmer que cette fraction correspond bien à l'état "sain" de la pompe dans ce dataset simulé. Si la dégradation commence dès le début, la fraction devra être ajustée.
- `TODO(arnaud)` : faut-il filtrer les fenêtres avec `maintenance_required=1` dans les données de pré-entraînement ? La spec dit "données normales uniquement" mais ne précise pas si on filtre sur le label ou sur la position temporelle.
- `TODO(dorra)` : format de sauvegarde `backbone.pt` (state_dict PyTorch) → quel format intermédiaire pour l'export vers TFLite Micro ? ONNX ou directement extraction des poids en tableaux C ?
- `FIXME(gap2)` : mesurer la RAM utilisée lors du pré-entraînement avec `memory_profiler.py` — non bloquant pour S3-04 mais requis pour S3-06 (exp_003).
