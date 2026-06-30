#!/usr/bin/env python3
"""train_board_reference.py — Modèles de référence board 5-features (Sprint 32 / S3205).

Le firmware NUCLEO-F439ZI traite **5 features**. Les runs PC du balayage S32
utilisent les dimensions natives (CMAPSS=5, Battery=7, Pronostia=13), ce qui rend
la parité board↔PC impossible pour Battery/Pronostia. Ce script entraîne, pour un
``(modèle ∈ {mahalanobis, ewc}, dataset ∈ {cmapss, battery, pronostia}, seuil)``,
un modèle **board-compatible 5-features** sur EXACTEMENT les features que
``sensor_stream.py`` envoie (même extraction que ``sensor_sim.load_dataset`` /
loader CMAPSS board), avec le label ``faulty`` ré-étiqueté au seuil balayé.

- **Mahalanobis** → ``checkpoints/mahalanobis_task0.pkl`` (export via
  ``export_weights_c.py --mahal``, z-score identité côté board car features déjà
  normalisées en streaming).
- **EWC** → ``checkpoints/ewc_head.pt`` (EWCMlpMulticlass 5→32→16→2, parité exacte
  avec ``ewc_forward`` ; export via ``export_weights_c.py --ewc-head``).

La parité est garantie par construction : board et PC consomment les mêmes nombres.

Usage :
    python scripts/train_board_reference.py --model ewc --dataset cmapss --threshold 30 \
        --exp_dir experiments/exp_S32_board_ewc_cmapss_thr30
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_seed  # noqa: E402

N_TASKS_DEFAULT = 3
EWC_EPOCHS_PER_TASK = 15
EWC_LR = 0.01
EWC_LAMBDA = 400.0


# ── Extraction des features board (alignée sur sensor_stream) ───────────────

def board_features(dataset: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (X[N,5], y[N]) board, label ``faulty`` au seuil donné.

    X reproduit EXACTEMENT l'extraction de ``sensor_sim.load_dataset(dataset)`` /
    du loader CMAPSS board (mêmes 5 features, même normalisation), garantissant la
    parité board↔PC. y est ré-étiqueté au seuil balayé.
    """
    if dataset == "cmapss":
        return _features_cmapss(int(threshold))
    if dataset == "pronostia":
        return _features_pronostia(float(threshold))
    if dataset == "battery":
        return _features_battery(float(threshold))
    raise ValueError(f"dataset board inconnu : {dataset}")


def _features_cmapss(threshold: int) -> tuple[np.ndarray, np.ndarray]:
    from src.data.cmapss_loader import get_cl_dataloaders

    subset = yaml.safe_load(Path("configs/cmapss_feature_subset.yaml").read_text())
    feature_names = subset.get("selected_features") or subset.get("features")

    # board_cmapss.yaml + seuil injecté → y au seuil ; features identiques au stream.
    cfg = yaml.safe_load(Path("configs/board_cmapss.yaml").read_text())
    cfg.setdefault("data", {})["faulty_threshold"] = threshold
    tmp_cfg = Path("configs/sweep/_runs/_board_cmapss_thr.yaml")
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    tasks = get_cl_dataloaders(
        data_dir=Path("data/raw/CMAPSS Jet Engine Simulated Data/"),
        config_path=tmp_cfg,
        feature_names=feature_names,
    )
    Xs, ys = [], []
    for t in tasks[:2]:  # FD001 + FD002 (n_tasks_board=2, cohérent _load_cmapss)
        for xb, yb in t["train_loader"]:
            Xs.append(xb.numpy())
            ys.append(yb.numpy().flatten())
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def _features_pronostia(threshold: float) -> tuple[np.ndarray, np.ndarray]:
    from src.data.pronostia_dataset import N_CONDITIONS, load_condition_features

    subset = yaml.safe_load(Path("configs/pronostia_feature_subset.yaml").read_text())
    indices = subset["feature_indices"]
    binaries = Path("data/raw/Pronostia dataset/binaries")

    Xs, ys = [], []
    for cond in range(1, N_CONDITIONS + 1):
        X_cond, y_cond = load_condition_features(
            binaries, condition=cond,
            label_mode="rul_threshold", faulty_threshold=threshold,
        )
        Xs.append(X_cond[:, indices].astype(np.float32))
        ys.append(y_cond.astype(np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def _features_battery(threshold: float) -> tuple[np.ndarray, np.ndarray]:
    from src.data.battery_dataset import (
        FEATURE_COLUMNS,
        load_battery_normalizer,
        load_raw_dataset,
        normalize_features,
    )

    subset = yaml.safe_load(Path("configs/battery_feature_subset.yaml").read_text())
    indices = subset["feature_indices"]
    csv = Path("data/raw/Battery Remaining Useful Life (RUL)/Battery_RUL.csv")

    df = load_raw_dataset(csv, rul_failure_threshold=threshold)  # y au seuil
    normalizer = load_battery_normalizer(Path("configs/battery_normalizer.yaml"))
    df = normalize_features(df, normalizer)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[:, indices]
    y = df["faulty"].to_numpy(dtype=np.int64)
    return X, y


def _temporal_tasks(X: np.ndarray, y: np.ndarray, n_tasks: int) -> list[tuple]:
    size = len(X) // n_tasks
    return [(X[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_tasks)]


# ── Entraînement Mahalanobis board ──────────────────────────────────────────

def train_maha_board(X: np.ndarray, y: np.ndarray, exp_dir: Path) -> Path:
    from src.models.unsupervised import MahalanobisDetector

    cfg = yaml.safe_load(Path("configs/board_mahalanobis.yaml").read_text())
    maha_cfg = cfg.get("mahalanobis", {"anomaly_percentile": 95, "cl_strategy": "refit"})
    model = MahalanobisDetector(maha_cfg)
    model.fit_task(X, task_id=0)  # référence board initiale (EMA online ensuite)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "mahalanobis_task0.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(model, f)
    print(f"  [maha] μ={model.mu_.shape} Σ⁻¹={model.sigma_inv_.shape} seuil={model.threshold_:.4f} → {ckpt}")
    return ckpt


# ── Entraînement EWC board (EWCMlpMulticlass 5→32→16→2) ─────────────────────

def train_ewc_board(X: np.ndarray, y: np.ndarray, exp_dir: Path, n_tasks: int) -> Path:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    model = EWCMlpMulticlass(input_dim=5, n_classes=2, hidden_dims=[32, 16],
                             dropout=0.2, ewc_lambda=EWC_LAMBDA)
    optimizer = torch.optim.SGD(model.parameters(), lr=EWC_LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for task_id, (Xt, yt) in enumerate(_temporal_tasks(X, y, n_tasks)):
        ds = TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                           torch.tensor(yt, dtype=torch.long))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model.train()
        for _ in range(EWC_EPOCHS_PER_TASK):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb) + model.ewc_penalty()
                loss.backward()
                # Clip de gradient : stabilise sur features brutes (Pronostia non normalisé)
                # → évite la divergence NaN. N'affecte pas la parité (poids finaux finis).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.consolidate(loader, n_samples=200)  # Fisher + θ* pour la tâche suivante
        with torch.no_grad():
            acc = (model.predict(torch.tensor(Xt, dtype=torch.float32)).numpy() == yt).mean()
        print(f"  [ewc] tâche {task_id}: acc_train={acc:.3f}")

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "ewc_head.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    print(f"  [ewc] state_dict → {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement modèle de référence board 5-feat (S3205)")
    parser.add_argument("--model", required=True, choices=["mahalanobis", "ewc"])
    parser.add_argument("--dataset", required=True, choices=["cmapss", "battery", "pronostia"])
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--exp_dir", required=True, type=Path)
    parser.add_argument("--n-tasks", type=int, default=N_TASKS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    X, y = board_features(args.dataset, args.threshold)
    print(f"[board-ref] {args.model} × {args.dataset} thr={args.threshold} : "
          f"X={X.shape} pos_ratio={float(y.mean()):.3f}")

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "mahalanobis":
        train_maha_board(X, y, args.exp_dir)
    else:
        train_ewc_board(X, y, args.exp_dir, args.n_tasks)
    print(f"✅ board-ref {args.model}/{args.dataset}/thr{args.threshold} → {args.exp_dir}")


if __name__ == "__main__":
    main()
