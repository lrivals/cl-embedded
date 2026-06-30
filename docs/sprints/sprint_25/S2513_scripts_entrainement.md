# S2513–S2514 — Scripts d'entraînement CL natifs

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | S2513 : 2h / S2514 : 2h = 4h total |
| **Dépendances** | S2501 ✅ (loaders mode `rul`), S2504 ✅ (loader mode `multiclass`), S2506 ✅ (`EWCMlpRegressor`), S2507 ✅ (`EWCMlpMulticlass`), S2509 ✅ (`rul_metrics.py`), S2510 ✅ (`multiclass_metrics.py`), S2511/S2512 ✅ (configs YAML) |
| **Fichiers cibles** | `scripts/train_ewc_rul.py`, `scripts/train_ewc_multiclass.py` |
| **Référence** | `scripts/train_ewc.py` (boucle CL binaire — pattern à reproduire), `scripts/train_hdc.py` (structure CLI + logging), `src/utils/config_loader.py` |

---

## Contexte

Les scripts `train_ewc.py` et `train_hdc.py` implémentent la boucle CL binaire standard. Sprint 25 crée deux nouveaux scripts suivant exactement le même pattern (CLI, logging, `config_snapshot.yaml`, `results.json`), adaptés aux nouvelles tâches :

- `train_ewc_rul.py` : régression RUL sur CMAPSS ou Pronostia, logge RMSE par tâche
- `train_ewc_multiclass.py` : classification multi-classe sur CWRU ou Paderborn, logge F1-macro par tâche

Les scripts sont autonomes (pas de Jupyter) et déposent leurs résultats dans `experiments/exp_S25_XX/`.

---

## S2513 — `scripts/train_ewc_rul.py`

### Spécification complète

```python
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
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data.cmapss_loader import get_cl_dataloaders as load_cmapss
from src.data.pronostia_dataset import get_pronostia_dataloaders as load_pronostia
from src.evaluation.rul_metrics import (
    compute_rul_metrics_task,
    compute_avg_forgetting_rmse,
)
from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed


def _load_tasks(cfg: dict) -> list[dict]:
    """Charge les DataLoaders selon le dataset spécifié dans la config."""
    dataset = cfg["dataset"]
    mode = cfg.get("task_mode", "rul")
    if dataset == "cmapss":
        return load_cmapss(
            data_dir=Path(cfg["data_dir"]),
            config_path=Path(args.config),
            mode=mode,
        )
    elif dataset == "pronostia":
        return load_pronostia(
            npy_dir=Path(cfg["npy_dir"]),
            normalizer_path=Path(cfg["normalizer_path"]),
            mode=mode,
        )
    else:
        raise ValueError(f"Dataset non supporté en mode RUL : {dataset}")


def train_one_epoch(
    model: EWCMlpRegressor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Entraîne une époque, retourne MSE loss moyen."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(x_batch).squeeze()
        loss = criterion(y_pred, y_batch.float()) + model.ewc_penalty()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_task(
    model: EWCMlpRegressor,
    loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    """Évalue le modèle sur une tâche, retourne RMSE / MAE / Horizon Score."""
    import numpy as np
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch).squeeze().numpy()
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(y_pred)
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

    tasks = _load_tasks(cfg)
    n_tasks = len(tasks)
    n_epochs = cfg.get("N_EPOCHS_PER_TASK", 20)

    # Matrice RMSE : task_rmse_matrix[époque_globale][task_idx]
    task_rmse_per_epoch: list[list[float]] = []
    task_results: list[dict] = []

    t_start = time.time()
    for task_idx, task in enumerate(tasks):
        print(f"\n--- Tâche {task_idx + 1}/{n_tasks} : {task.get('task_id', task_idx)} ---")
        train_loader = task["train_loader"]
        val_loader = task["val_loader"]

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            # Évaluer toutes les tâches précédentes (backward transfer)
            row: list[float] = []
            for prev_task in tasks[: task_idx + 1]:
                metrics = evaluate_task(model, prev_task["val_loader"])
                row.append(metrics["rmse"])
            task_rmse_per_epoch.append(row + [0.0] * (n_tasks - len(row)))

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} — loss={train_loss:.4f}, RMSE_t{task_idx+1}={row[-1]:.2f}")

        # Consolidation EWC
        model.consolidate(train_loader, n_samples=cfg.get("FISHER_N_SAMPLES", 200))

        # Métriques finales sur cette tâche
        final_metrics = evaluate_task(model, val_loader)
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
        # Config snapshot
        import yaml
        with open(output_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump({**cfg, "exp_id": args.exp_id}, f, allow_unicode=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
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
```

### Vérification

```bash
# Dry-run CMAPSS (vérifie le pipeline sans écrire)
python scripts/train_ewc_rul.py \
    --config configs/cmapss_rul_config.yaml \
    --exp_id dry_rul \
    --output_dir /tmp/dry_rul/ \
    --dry_run

# Attendu : affichage des tâches + métriques RMSE, pas d'erreur
```

---

## S2514 — `scripts/train_ewc_multiclass.py`

### Spécification complète

```python
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
"""

# ... (même structure que train_ewc_rul.py)

# Différences clés vs train_ewc_rul.py :
#   - Modèle : EWCMlpMulticlass (n_classes depuis cfg["N_CLASSES"])
#   - Criterion : nn.CrossEntropyLoss()
#   - Labels : y_batch.long() (pas float)
#   - Métriques : compute_multiclass_metrics_task (F1-macro, confusion_matrix)
#   - Forgetting : compute_avg_forgetting_f1
#   - Critère de validation : f1_macro >= cfg.get("f1_macro_min_task1", 0.70)
```

### Vérification

```bash
# Dry-run CWRU multiclass
python scripts/train_ewc_multiclass.py \
    --config configs/cwru_multiclass_config.yaml \
    --exp_id dry_multiclass \
    --output_dir /tmp/dry_mc/ \
    --dry_run

# Attendu : affichage tâches + F1-macro par tâche, pas d'erreur
```

---

## Vérification end-to-end

```bash
# Les deux scripts en dry-run
for script in train_ewc_rul train_ewc_multiclass; do
    cfg_rul="configs/cmapss_rul_config.yaml"
    cfg_mc="configs/cwru_multiclass_config.yaml"
    cfg=$([ "$script" = "train_ewc_rul" ] && echo $cfg_rul || echo $cfg_mc)
    python scripts/${script}.py \
        --config $cfg \
        --exp_id dry_${script} \
        --output_dir /tmp/dry_${script}/ \
        --dry_run && echo "$script dry-run OK ✅"
done

# Vérifier que les scripts n'impactent pas les tests binaires existants
pytest tests/ -v -k "not ewc_regression and not ewc_multiclass"
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2513 — `scripts/train_ewc_rul.py` | ✅ | Bugfix `args.config` hors scope → paramètre `config_path` ; `np.atleast_1d` pour batch_size=1 |
| S2514 — `scripts/train_ewc_multiclass.py` | ✅ | CWRU via `get_cl_splits` → DataLoaders manuels ; Paderborn via `get_cl_dataloaders(mode="multiclass")` |

---

## Questions ouvertes

- `TODO(arnaud)` : La boucle CL actuelle consolide EWC **après** toutes les époques d'une tâche. Est-il préférable de consolider après chaque batch (EWC Online pur) ou garder le schéma task-by-task (plus stable empiriquement) ?
- `FIXME(gap2)` : Les scripts mesurent `training_time_s` (CPU) mais pas `inference_latency_ms`. Ce profilage est dans exp_S25_05 — ne pas l'ajouter dans ces scripts pour garder la séparation des responsabilités.
