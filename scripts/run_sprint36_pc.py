#!/usr/bin/env python3
"""run_sprint36_pc.py — Référence PC appariée EWC (S3602).

Pour chaque cellule ``(condition ∈ {5feat, all}, dataset ∈ {pronostia, monitoring})`` :

1. Charge **exactement** les colonnes/le split que le board verra :
   ``load_condition_arrays(dataset, condition, "ewc", seed=42)`` (source unique S35).
2. Entraîne ``EWCMlpMulticlass(k, 2, [32, 16])`` en CL séquentiel sur un **split temporel
   3 tâches** (miroir de ``run_feature_condition_board.train_ewc_board`` → le checkpoint PC
   est **réutilisé tel quel** par le board en S3603 ⇒ parité exacte par construction).
3. Construit ``acc_matrix[T×T]`` (eval de toutes les tâches vues après chaque tâche) puis
   ``compute_cl_metrics`` (AA/AF/BWT — **AF rapporté** : lien Sprint 26 oubli catastrophique).
4. Calcule F1 (``compute_fault_f1``), ROC-AUC (``compute_anomaly_metrics`` sur la proba de
   classe 1), ``n_params``, ``ram_peak_bytes`` (tracemalloc), ``inference_latency_ms``.
5. Sauve checkpoint ``experiments/exp_S36_PC_{cond}_ewc_{ds}/checkpoints/ewc_head.pt`` +
   ``results.json`` (avec dump ``samples`` par échantillon pour la parité S3603/S3604).

Hyperparamètres : LR/LAMBDA ← ``configs/board_ewc.yaml`` ; n_tasks/epochs ←
``configs/sprint36_ewc_comparison.yaml`` (aucun hyperparamètre en dur — règle CLAUDE.md).

Usage :
    python scripts/run_sprint36_pc.py --config configs/sprint36_ewc_comparison.yaml
    python scripts/run_sprint36_pc.py --condition all --dataset pronostia
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.evaluation.anomaly_metrics import compute_anomaly_metrics  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_cl_metrics, compute_fault_f1  # noqa: E402
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

EXPERIMENTS = Path("experiments")
DEFAULT_CONFIG = "configs/sprint36_ewc_comparison.yaml"


# ── Split temporel + train/test par tâche ───────────────────────────────────

def _temporal_tasks(X: np.ndarray, y: np.ndarray, n_tasks: int) -> list[tuple]:
    """Découpe chronologique en n_tasks blocs (== run_feature_condition_board)."""
    size = max(1, len(X) // n_tasks)
    return [(X[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_tasks)]


def _split_task(Xt: np.ndarray, yt: np.ndarray, test_ratio: float) -> tuple:
    """Split déterministe train/test (ordre temporel : train = début, test = fin)."""
    n_test = max(1, int(len(Xt) * test_ratio))
    n_train = len(Xt) - n_test
    return Xt[:n_train], yt[:n_train], Xt[n_train:], yt[n_train:]


def _accuracy(model: EWCMlpMulticlass, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((pred == y).mean())


# ── Entraînement CL + métriques ──────────────────────────────────────────────

def train_and_eval(X: np.ndarray, y: np.ndarray, k: int, tr_cfg: dict,
                   ewc_lr: float, ewc_lambda: float) -> tuple:
    """Entraîne EWC séquentiel ; retourne (model, acc_matrix[T×T])."""
    n_tasks = int(tr_cfg["n_tasks"])
    epochs = int(tr_cfg["epochs_per_task"])
    batch_size = int(tr_cfg["batch_size"])
    test_ratio = float(tr_cfg["test_ratio"])

    tasks = _temporal_tasks(X, y, n_tasks)
    splits = [_split_task(Xt, yt, test_ratio) for Xt, yt in tasks]  # (Xtr,ytr,Xte,yte)

    model = EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16],
                             dropout=0.2, ewc_lambda=ewc_lambda)
    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    for i, (Xtr, ytr, _Xte, _yte) in enumerate(splits):
        ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                           torch.tensor(ytr, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb) + model.ewc_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.consolidate(loader, n_samples=200)
        # Évaluer toutes les tâches vues (j ≤ i) sur leur split test.
        for j in range(i + 1):
            _, _, Xte, yte = splits[j]
            acc_matrix[i, j] = _accuracy(model, Xte, yte)
    return model, acc_matrix


def run_cell(condition: str, dataset: str, cfg: dict, ewc_lr: float,
             ewc_lambda: float, out_root: Path) -> dict:
    print(f"\n{'='*64}\n=== PC CELL  condition={condition}  dataset={dataset}  ===\n{'='*64}")
    set_seed(int(cfg["seed"]))

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    print(f"  X={X.shape}  k={k}  features={names}")

    tracemalloc.start()
    model, acc_matrix = train_and_eval(X, y, k, cfg["training"], ewc_lr, ewc_lambda)
    _cur, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cl = compute_cl_metrics(acc_matrix)

    # Prédictions finales sur le split COMPLET (n_inference: full).
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    conf = prob[np.arange(len(pred)), pred]
    score_pos = prob[:, 1]  # proba classe faulty → score d'anomalie pour ROC-AUC

    f1 = compute_fault_f1(y, pred)
    try:
        roc = compute_anomaly_metrics(y, score_pos)["auroc"]
    except Exception:  # noqa: BLE001
        roc = None

    # Latence inférence single-sample (moyenne 200 runs).
    x1 = torch.tensor(X[:1], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x1)
        lat_ms = (time.perf_counter() - t0) / 200 * 1e3

    samples = [
        {"idx": int(i), "true": int(y[i]), "pred": int(pred[i]),
         "confidence": float(conf[i]), "features": [float(v) for v in X[i]]}
        for i in range(len(X))
    ]

    exp_id = f"exp_S36_PC_{condition}_ewc_{dataset}"
    exp_dir = out_root / exp_id
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "ewc_head.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)

    result = {
        "exp_id": exp_id, "platform": "pc", "model": "ewc", "dataset": dataset,
        "condition": condition, "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "acc_matrix": cl["acc_matrix"],
        "aa": cl["aa"], "af": cl["af"], "bwt": cl["bwt"], "fwt": cl["fwt"],
        "acc_final": cl["aa"],   # AA = accuracy moyenne finale toutes tâches
        "forgetting_per_task": cl["forgetting_per_task"],
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"],
        "precision_faulty": f1["precision_faulty"], "recall_faulty": f1["recall_faulty"],
        "roc_auc": roc,
        "n_params": model.count_parameters(),
        "ram_peak_bytes": int(ram_peak),
        "inference_latency_ms": float(lat_ms),
        "per_task_acc": {str(j): (None if np.isnan(acc_matrix[-1, j]) else float(acc_matrix[-1, j]))
                         for j in range(acc_matrix.shape[1])},
        "checkpoint": str(ckpt),
        "samples": samples,
    }
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"  AA={cl['aa']:.3f} AF={cl['af']:.3f} BWT={cl['bwt']:.3f} "
          f"F1_faulty={f1['f1_faulty']:.3f} ROC-AUC={roc} → {exp_dir}/results.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Référence PC appariée EWC (S3602)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--condition", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out-root", default=str(EXPERIMENTS))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    ewc_lr = float(base["EWC_LR"])
    ewc_lambda = float(base["EWC_LAMBDA"])
    out_root = Path(args.out_root)

    conditions = [args.condition] if args.condition else cfg["conditions"]
    datasets = [args.dataset] if args.dataset else cfg["datasets"]

    rows = []
    for d in datasets:
        for c in conditions:
            rows.append(run_cell(c, d, cfg, ewc_lr, ewc_lambda, out_root))

    print(f"\n{'='*60}\nPC S36 : {len(rows)} cellules produites.")
    for r in rows:
        print(f"  {r['exp_id']:38s} AA={r['aa']:.3f} AF={r['af']:.3f} "
              f"F1={r['f1_faulty']:.3f}")


if __name__ == "__main__":
    main()
