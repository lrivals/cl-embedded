#!/usr/bin/env python3
"""run_sprint38_pc.py — Référence PC : mise à jour autonome EWC pilotée par gate (S3802).

Pour chaque cellule ``(policy × dataset × init_mode)`` :

1. Charge **exactement** les colonnes/le split que le board verra :
   ``load_condition_arrays(dataset, "5feat", "ewc", seed=42)`` (source unique S35).
2. **init_mode** (décision utilisateur — étude des deux) :
   - ``pretrained`` : base CL offline partagée (split temporel 3 tâches, 15 epochs,
     ``consolidate`` — miroir ``run_sprint36_pc.train_and_eval``) → checkpoint réutilisé tel
     quel par le board (S3804) ⇒ parité exacte en frozen.
   - ``scratch`` : ``EWCMlpMulticlass`` à l'init Xavier, **aucun** entraînement offline ;
     le streaming EST l'apprentissage.
3. **Enrôlement one-class** : les ``n_samples`` premiers échantillons SAINS calibrent un
   ``MahalanobisDetector`` ET les seuils du gate (``set_thresholds_from_normal`` = P95 × mult).
4. **Streaming séquentiel** du split test complet, échantillon par échantillon. À chaque pas :
   ``score = maha.anomaly_score(x)`` ; ``verdict = drift.update(score)`` ; application de la
   politique (frozen / always / gated_truelabel / gated_pseudolabel). Le **vrai label** ne sert
   qu'au scoring, jamais au SGD en P3.
5. **acc_matrix(T×T)** reconstruit aux frontières de tâche pendant le stream → ``compute_cl_metrics``.
6. Métriques → ``results.json`` (AA/AF/BWT, F1, ROC-AUC, n_updates, update_rate, confusion
   verdict↔vérité, RAM, latence) + checkpoint + ``drift_thresholds.json`` + dump ``samples``.

Hyperparamètres : LR/LAMBDA ← ``configs/board_ewc.yaml`` ; n_tasks/epochs ←
``configs/sprint38_autonomous_update.yaml`` (aucun hyperparamètre en dur — règle CLAUDE.md).

Usage :
    python scripts/run_sprint38_pc.py --config configs/sprint38_autonomous_update.yaml
    python scripts/run_sprint38_pc.py --policy gated_pseudolabel --dataset monitoring --init-mode scratch
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
from src.evaluation.drift_detector import SlidingWindowDriftDetector  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_cl_metrics, compute_fault_f1  # noqa: E402
from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass  # noqa: E402
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

EXPERIMENTS = Path("experiments")
DEFAULT_CONFIG = "configs/sprint38_autonomous_update.yaml"
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")
INIT_MODES = ("pretrained", "scratch")
VERDICTS = ("NORMAL", "DRIFT", "FAULT")


# ── Split temporel + train/test par tâche (== run_sprint36_pc / board) ───────

def _temporal_tasks(X: np.ndarray, y: np.ndarray, n_tasks: int) -> list[tuple]:
    """Découpe chronologique en n_tasks blocs."""
    size = max(1, len(X) // n_tasks)
    return [(X[i * size:(i + 1) * size], y[i * size:(i + 1) * size]) for i in range(n_tasks)]


def _split_task(Xt: np.ndarray, yt: np.ndarray, test_ratio: float) -> tuple:
    """Split déterministe train/test (train = début, test = fin)."""
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


def _build_model(k: int, ewc_lambda: float) -> EWCMlpMulticlass:
    return EWCMlpMulticlass(input_dim=k, n_classes=2, hidden_dims=[32, 16],
                            dropout=0.2, ewc_lambda=ewc_lambda)


def _pretrain_cl(model: EWCMlpMulticlass, splits: list[tuple], tr_cfg: dict,
                 ewc_lr: float) -> np.ndarray:
    """Entraînement CL offline (== run_sprint36_pc.train_and_eval). Retourne acc_matrix de base."""
    epochs = int(tr_cfg["epochs_per_task"])
    batch_size = int(tr_cfg["batch_size"])
    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    n_tasks = len(splits)
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
        for j in range(i + 1):
            _, _, Xte, yte = splits[j]
            acc_matrix[i, j] = _accuracy(model, Xte, yte)
    return acc_matrix


def _sgd_step(model: EWCMlpMulticlass, optimizer: torch.optim.Optimizer,
              x: np.ndarray, label: int) -> None:
    """1 pas de SGD online sur un échantillon (équivalent PC de ``ewc_sgd_step`` firmware)."""
    model.train()
    xb = torch.tensor(x[None, :], dtype=torch.float32)
    yb = torch.tensor([int(label)], dtype=torch.long)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(xb), yb) + model.ewc_penalty()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()


def _predict_one(model: EWCMlpMulticlass, x: np.ndarray) -> tuple[int, float, float]:
    """Retourne (pred, confidence, proba_classe1) pour un échantillon."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[None, :], dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).numpy()[0]
    pred = int(np.argmax(prob))
    return pred, float(prob[pred]), float(prob[1])


# ── Cellule ──────────────────────────────────────────────────────────────────

def run_cell(policy: str, dataset: str, init_mode: str, cfg: dict, ewc_lr: float,
             ewc_lambda: float, out_root: Path) -> dict:
    print(f"\n{'='*72}\n=== PC CELL  policy={policy}  dataset={dataset}  init={init_mode}  ===\n{'='*72}")
    set_seed(int(cfg["seed"]))

    condition = cfg["condition"]
    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)
    print(f"  X={X.shape}  k={k}  features={names}")

    tr_cfg = cfg["training"]
    n_tasks = int(tr_cfg["n_tasks"])
    test_ratio = float(tr_cfg["test_ratio"])
    tasks = _temporal_tasks(X, y, n_tasks)
    splits = [_split_task(Xt, yt, test_ratio) for Xt, yt in tasks]  # (Xtr,ytr,Xte,yte)

    tracemalloc.start()

    # ── Modèle : pré-entraîné (base CL) ou from scratch ──
    model = _build_model(k, ewc_lambda)
    base_acc_matrix = None
    if init_mode == "pretrained":
        base_acc_matrix = _pretrain_cl(model, splits, tr_cfg, ewc_lr)

    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)

    # ── Enrôlement one-class : maha + seuils du gate sur les N premiers sains ──
    enr = cfg["enrollment"]
    n_enr = int(enr["n_samples"])
    healthy_mask = (y == 0)
    X_healthy = X[healthy_mask][:n_enr]
    if len(X_healthy) < 2:
        raise RuntimeError(f"Pas assez d'échantillons sains pour l'enrôlement ({len(X_healthy)}).")
    maha = MahalanobisDetector({"cl_strategy": "welford", "anomaly_percentile": 95})
    maha.fit_task(X_healthy, task_id=0)

    dd_cfg = cfg["drift_detector"]
    drift = SlidingWindowDriftDetector(
        window_size=int(dd_cfg["window_size"]),
        fault_multiplier=float(dd_cfg["fault_multiplier"]),
        drift_multiplier=float(dd_cfg["drift_multiplier"]),
        drift_ratio=float(dd_cfg["drift_ratio"]),
    )
    drift.set_thresholds_from_normal(maha.anomaly_score(X_healthy))

    # ── Streaming séquentiel sur le split test complet ──
    # On rejoue les splits test tâche par tâche pour reconstruire acc_matrix(T×T)
    # aux frontières (eval du modèle COURANT sur chaque tâche vue).
    test_splits = [(Xte, yte) for (_Xtr, _ytr, Xte, yte) in splits]
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)

    preds: list[int] = []
    trues: list[int] = []
    scores_pos: list[float] = []
    samples: list[dict] = []
    n_updates = 0
    # confusion verdict↔vérité : lignes {NORMAL,DRIFT,FAULT} × colonnes {sain(0),faulty(1)}
    confusion = {v: [0, 0] for v in VERDICTS}

    global_idx = 0
    for ti, (Xte, yte) in enumerate(test_splits):
        for xi, yi in zip(Xte, yte):
            score = float(maha.anomaly_score(xi[None, :])[0])
            verdict = drift.update(score)
            confusion[verdict][int(yi)] += 1

            pred, conf, p1 = _predict_one(model, xi)
            updated = False
            if policy == "frozen":
                pass
            elif policy == "always":
                _sgd_step(model, optimizer, xi, int(yi)); n_updates += 1; updated = True
            elif policy == "gated_truelabel":
                if verdict != "NORMAL":
                    _sgd_step(model, optimizer, xi, int(yi)); n_updates += 1; updated = True
            elif policy == "gated_pseudolabel":
                if verdict == "FAULT":
                    _sgd_step(model, optimizer, xi, 1); n_updates += 1; updated = True
                elif verdict == "DRIFT":
                    maha.partial_fit(xi); updated = True  # adapte le normal (pas de SGD faute)
            else:
                raise ValueError(f"Politique inconnue : {policy}")

            preds.append(pred); trues.append(int(yi)); scores_pos.append(p1)
            samples.append({"idx": int(global_idx), "true": int(yi), "pred": int(pred),
                            "confidence": float(conf), "verdict": verdict, "updated": bool(updated)})
            global_idx += 1

        # Frontière de tâche : éval du modèle courant sur toutes les tâches vues.
        for j in range(ti + 1):
            Xj, yj = test_splits[j]
            acc_matrix[ti, j] = _accuracy(model, Xj, yj)

    _cur, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Métriques ──
    cl = compute_cl_metrics(acc_matrix)
    y_arr = np.array(trues)
    pred_arr = np.array(preds)
    f1 = compute_fault_f1(y_arr, pred_arr)
    try:
        roc = compute_anomaly_metrics(y_arr, np.array(scores_pos))["auroc"]
    except Exception:  # noqa: BLE001
        roc = None

    # Latence inférence single-sample (moyenne 200 runs).
    x1 = torch.tensor(X[:1], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x1)
        lat_ms = (time.perf_counter() - t0) / 200 * 1e3

    # ── Artefacts ──
    exp_id = f"exp_S38_PC_{policy}_{dataset}_{init_mode}"
    exp_dir = out_root / exp_id
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ck_dir / "ewc_head.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)

    drift_thresholds = {
        "fault_threshold": float(drift.fault_threshold),
        "drift_threshold": float(drift.drift_threshold),
        "window_size": int(drift.window_size),
        "drift_ratio": float(drift.drift_ratio),
    }
    (exp_dir / "drift_thresholds.json").write_text(json.dumps(drift_thresholds, indent=2))

    n_samples = len(preds)
    result = {
        "exp_id": exp_id, "platform": "pc", "model": "ewc", "dataset": dataset,
        "condition": condition, "policy": policy, "init_mode": init_mode,
        "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "acc_matrix": cl["acc_matrix"],
        "base_acc_matrix": (None if base_acc_matrix is None
                            else [[None if np.isnan(v) else float(v) for v in row]
                                  for row in base_acc_matrix]),
        "aa": cl["aa"], "af": cl["af"], "bwt": cl["bwt"], "fwt": cl["fwt"],
        "acc_final": cl["aa"],
        "forgetting_per_task": cl["forgetting_per_task"],
        "f1_faulty": f1["f1_faulty"], "f1_macro": f1["f1_macro"],
        "precision_faulty": f1["precision_faulty"], "recall_faulty": f1["recall_faulty"],
        "roc_auc": roc,
        "n_updates": int(n_updates), "n_samples": int(n_samples),
        "update_rate": (float(n_updates) / n_samples if n_samples else None),
        "confusion_verdict_truth": {v: confusion[v] for v in VERDICTS},
        "drift_thresholds": drift_thresholds,
        "n_params": model.count_parameters(),
        "ram_peak_bytes": int(ram_peak),
        "inference_latency_ms": float(lat_ms),
        "checkpoint": str(ckpt),
        "samples": samples,
    }
    (exp_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"  AA={cl['aa']:.3f} AF={cl['af']:.3f} F1_faulty={f1['f1_faulty']:.3f} "
          f"ROC-AUC={roc} n_updates={n_updates}/{n_samples} "
          f"(rate={result['update_rate']:.3f}) → {exp_dir}/results.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Référence PC mise à jour autonome EWC (S3802)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--policy", default=None, choices=POLICIES)
    p.add_argument("--dataset", default=None)
    p.add_argument("--init-mode", default=None, choices=INIT_MODES)
    p.add_argument("--out-root", default=str(EXPERIMENTS))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    ewc_lr = float(base["EWC_LR"])
    ewc_lambda = float(base["EWC_LAMBDA"])
    out_root = Path(args.out_root)

    policies = [args.policy] if args.policy else cfg["policies"]
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    init_modes = [args.init_mode] if args.init_mode else cfg.get("init_modes", list(INIT_MODES))

    rows = []
    for im in init_modes:
        for d in datasets:
            for pol in policies:
                rows.append(run_cell(pol, d, im, cfg, ewc_lr, ewc_lambda, out_root))

    print(f"\n{'='*72}\nPC S38 : {len(rows)} cellules produites.")
    for r in rows:
        print(f"  {r['exp_id']:48s} AA={r['aa']:.3f} F1={r['f1_faulty']:.3f} "
              f"update_rate={r['update_rate']:.3f}")


if __name__ == "__main__":
    main()
