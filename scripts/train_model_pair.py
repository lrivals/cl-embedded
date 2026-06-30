# ruff: noqa: N802, N803  — X / _loader_Xy suivent la convention mathématique ML (sklearn)
"""
scripts/train_model_pair.py — Sprint 30 (S3005 + S3007) — entraînement & évaluation
d'une paire « Mahalanobis (non-supervisé) + modèle supervisé ».

Pour une paire × dataset, ce script :
  1. entraîne le détecteur Mahalanobis et le modèle supervisé (EWC / HDC / TinyOL) sur
     la *même* séquence de tâches CL ;
  2. évalue chaque modèle individuellement (entraînement CL : AA/AF/BWT ; inférence :
     AUROC/F1/précision/rappel) ET l'ensemble (les 4 règles de fusion) ;
  3. analyse le désaccord entre les 2 modèles (taux, kappa, qui a raison, origine).

Deux cadres :
  - `--mode binary` (Partie A) : tout binarisé normal-vs-fault → sorties comparables.
  - `--mode native` (Partie B) : modèle supervisé en tâche native (RUL CMAPSS régression,
    multi-classe CWRU) ; le désaccord compare la décision binaire dérivée par
    `native_to_fault` (cf. bloc `native:` de la config) à la sortie du détecteur.

Réutilise au maximum l'existant : `_get_tasks` / `train_ewc` (scripts/train_ewc.py),
`run_cl_scenario_full` (src/training/scenarios.py), helpers RUL/multiclasse
(scripts/train_ewc_rul.py, scripts/train_ewc_multiclass.py), `ModelPair`,
`compute_cl_metrics`, `compute_anomaly_metrics`, `disagreement_metrics`.

Usage :
    python scripts/train_model_pair.py --config configs/board_pair_maha_ewc.yaml --dataset monitoring
    python scripts/train_model_pair.py --config configs/board_pair_maha_ewc.yaml --dataset cmapss --mode native
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.ensemble.model_pair import ModelPair, _binarize_labels, native_to_fault  # noqa: E402
from src.evaluation.anomaly_metrics import compute_anomaly_metrics  # noqa: E402
from src.evaluation.disagreement_metrics import (  # noqa: E402
    analyze_disagreement_origin,
    cohen_kappa,
    disagreement_confusion,
    disagreement_rate,
    per_sample_disagreement_mask,
)
from src.evaluation.metrics import compute_cl_metrics  # noqa: E402
from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector  # noqa: E402
from src.training.scenarios import run_cl_scenario_full  # noqa: E402
from src.utils.config_loader import load_config, load_config_extends  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

DATASETS = ("monitoring", "cwru", "pronostia", "cmapss", "paderborn")


# ----------------------------------------------------------------------------
# Import dynamique des scripts/ (non packagés) pour réutiliser leurs helpers
# ----------------------------------------------------------------------------
def _load_script(name: str):
    """Charge un module scripts/<name>.py pour réutiliser ses fonctions."""
    spec = importlib.util.spec_from_file_location(f"_pair_{name}", _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Extraction numpy des DataLoaders
# ----------------------------------------------------------------------------
def _loader_X(loader) -> np.ndarray:
    xs = [x.numpy().astype(np.float32) for x, _ in loader]
    return np.concatenate(xs, axis=0) if xs else np.empty((0, 0), dtype=np.float32)


def _loader_Xy(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy().astype(np.float32))
        ys.append(y.numpy().ravel())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _concat_eval(tasks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Jeu d'évaluation commun = concaténation des val_loaders (mêmes échantillons)."""
    Xs, ys = [], []
    for t in tasks:
        loader = t.get("test_loader") or t["val_loader"]
        X, y = _loader_Xy(loader)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ----------------------------------------------------------------------------
# Wrappers supervisés — homogénéisent predict / predict_proba sur numpy (duck-typing
# attendu par ModelPair : predict_proba sinon predict).
# ----------------------------------------------------------------------------
class _EWCWrap:
    """EWCMlpClassifier (nn.Module binaire) → predict_proba/predict sur numpy."""

    def __init__(self, model) -> None:
        self.m = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:  # noqa: N803
        self.m.eval()
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return self.m(t).cpu().numpy().ravel()

    def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(x) >= 0.5).astype(int)


class _HDCWrap:
    """HDCClassifier → predict (labels). Pas de proba native → labels servent de score."""

    def __init__(self, model) -> None:
        self.m = model

    def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(self.m.predict(np.asarray(x, dtype=np.float32))).ravel()


class _DetectorWrap:
    """Détecteur d'anomalie (TinyOLAnomalyDetector) → predict + proba sigmoïde(score-seuil)."""

    def __init__(self, det) -> None:
        self.d = det

    def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(self.d.predict(np.asarray(x, dtype=np.float32))).ravel()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:  # noqa: N803
        s = np.asarray(self.d.anomaly_score(np.asarray(x, dtype=np.float32)), dtype=float).ravel()
        thr = self.d.anomaly_threshold_
        z = np.clip(s - thr, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))


def _sup_scores(sup, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Score continu de panne du modèle supervisé (proba si dispo, sinon labels)."""
    if hasattr(sup, "predict_proba"):
        return np.asarray(sup.predict_proba(X), dtype=float).ravel()
    return np.asarray(sup.predict(X), dtype=float).ravel()


# ----------------------------------------------------------------------------
# Entraînement des modèles (chacun renvoie le modèle entraîné + acc_matrix [T, T])
# ----------------------------------------------------------------------------
def _train_mahalanobis(maha_cfg: dict, tasks: list[dict]) -> tuple[MahalanobisDetector, np.ndarray]:
    det = MahalanobisDetector(maha_cfg)
    T = len(tasks)
    acc = np.full((T, T), np.nan)
    for i, task in enumerate(tasks):
        det.fit_task(_loader_X(task["train_loader"]), task_id=i)
        for j in range(i + 1):
            Xv, yv = _loader_Xy(tasks[j]["val_loader"])
            acc[i, j] = float((det.predict(Xv) == _binarize_labels(yv)).mean())
    return det, acc


def _train_detector_basecl(model, tasks: list[dict]) -> np.ndarray:
    """Boucle CL pour détecteur BaseCLModel (TinyOLAnomalyDetector) : seuil sur task 1."""
    T = len(tasks)
    acc = np.full((T, T), np.nan)
    for i, task in enumerate(tasks):
        for xb, yb in task["train_loader"]:
            model.update(xb.numpy().astype(np.float32), yb.numpy().ravel())
        model.on_task_end(i + 1, task["train_loader"])  # task_id 1-based → seuil sur task 1
        for j in range(i + 1):
            Xv, yv = _loader_Xy(tasks[j]["val_loader"])
            pred = np.asarray(model.predict(Xv)).ravel()
            acc[i, j] = float((pred == _binarize_labels(yv)).mean())
    return acc


def _train_supervised(name: str, sup_cfg: dict, tasks: list[dict], device: str):
    """Construit + entraîne le modèle supervisé (binaire). Renvoie (wrapper, acc_matrix)."""
    if name == "ewc":
        from src.models.ewc import EWCMlpClassifier

        te = _load_script("train_ewc")
        model = EWCMlpClassifier(
            input_dim=sup_cfg["model"]["input_dim"],
            hidden_dims=sup_cfg["model"]["hidden_dims"],
            dropout=sup_cfg["model"]["dropout"],
        )
        acc = te.train_ewc(model, tasks, sup_cfg, device)
        return _EWCWrap(model), acc

    if name == "hdc":
        from src.models.hdc.hdc_classifier import HDCClassifier

        model = HDCClassifier(sup_cfg)
        acc, _ = run_cl_scenario_full(model, tasks, sup_cfg)
        return _HDCWrap(model), acc

    if name == "tinyol":
        from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector

        model = TinyOLAnomalyDetector(sup_cfg)
        acc = _train_detector_basecl(model, tasks)
        return _DetectorWrap(model), acc

    raise ValueError(f"Modèle supervisé inconnu : {name!r} (ewc | hdc | tinyol).")


# ----------------------------------------------------------------------------
# Évaluation commune (binary + native)
# ----------------------------------------------------------------------------
def _cl_block(name: str, acc_matrix: np.ndarray) -> dict:
    m = compute_cl_metrics(acc_matrix)
    return {"name": name, "aa": m["aa"], "af": m["af"], "bwt": m["bwt"]}


def _ensemble_metrics(pair: ModelPair, X: np.ndarray, y_bin: np.ndarray, rules: list[str]) -> dict:  # noqa: N803
    """F1/précision/rappel (sur décision binaire) + AUROC (sur proba fusionnée) par règle."""
    by_rule: dict[str, dict] = {}
    for rule in rules:
        preds = pair.predict_ensemble(X, rule)
        proba = pair._fused_proba(X, rule)  # proba fusionnée pour l'AUROC de cette règle
        auroc = compute_anomaly_metrics(y_bin, proba)["auroc"] if np.unique(y_bin).size > 1 else None
        by_rule[rule] = {
            "f1": float(f1_score(y_bin, preds, zero_division=0)),
            "precision": float(precision_score(y_bin, preds, zero_division=0)),
            "recall": float(recall_score(y_bin, preds, zero_division=0)),
            "auroc": auroc,
        }
    best_rule = max(by_rule, key=lambda r: by_rule[r]["f1"])
    return {"by_rule": by_rule, "best_rule": best_rule}


def _disagreement_block(
    pred_maha: np.ndarray,
    pred_sup: np.ndarray,
    X: np.ndarray,  # noqa: N803
    y_bin: np.ndarray,
    maha_scores: np.ndarray,
) -> dict:
    mask = per_sample_disagreement_mask(pred_maha, pred_sup)
    conf = disagreement_confusion(pred_maha, pred_sup, y_bin)
    origin = analyze_disagreement_origin(X, mask, y_bin, maha_scores=maha_scores)
    return {
        "rate": float(disagreement_rate(pred_maha, pred_sup)),
        "kappa": float(cohen_kappa(pred_maha, pred_sup)),
        "a_correct": int(conf["a_correct"]),
        "b_correct": int(conf["b_correct"]),
        "both_wrong": int(conf["both_wrong"]),
        "n_disagree": int(conf.get("n_disagree", int(mask.sum()))),
        "origin": _jsonify(origin),
    }


def _jsonify(obj):
    """Convertit récursivement ndarray/np.scalaire en types JSON-sérialisables."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ----------------------------------------------------------------------------
# Partie A — cadre binarisé
# ----------------------------------------------------------------------------
def run_binary(pair_cfg: dict, dataset: str, device: str) -> dict:
    sup_name = pair_cfg["pair"]["supervised"]
    sup_path = pair_cfg["supervised_configs"][dataset]
    seed = int(pair_cfg.get("seed", 42))
    rules = list(pair_cfg["fusion"]["rules"])

    # --- Tâches (depuis la config supervisée validée) ---
    set_seed(seed)
    sup_cfg = load_config_extends(sup_path)
    sup_cfg["_config_path"] = sup_path
    te = _load_script("train_ewc")
    tasks = te._get_tasks(sup_cfg)
    sup_cfg.pop("_config_path", None)

    # --- Entraînement des 2 modèles sur les mêmes tâches ---
    set_seed(seed)
    detector, maha_acc = _train_mahalanobis(pair_cfg["mahalanobis"], tasks)
    set_seed(seed)
    sup, sup_acc = _train_supervised(sup_name, sup_cfg, tasks, device)

    # --- Jeu d'évaluation commun ---
    X_eval, y_raw = _concat_eval(tasks)
    y_bin = _binarize_labels(y_raw)

    # --- Paire ---
    fusion = pair_cfg["fusion"]
    pair = ModelPair(
        detector=detector,
        classifier=sup,
        mode="binary",
        config={
            "fusion_rule": rules[0],
            "weights": fusion["weights"],
            "ensemble_threshold": fusion["ensemble_threshold"],
        },
    )

    # --- Inférence individuelle ---
    maha_scores = detector.anomaly_score(X_eval)
    m_a = _cl_block("mahalanobis", maha_acc)
    am = compute_anomaly_metrics(y_bin, maha_scores) if np.unique(y_bin).size > 1 else {}
    m_a.update({"auroc": am.get("auroc"), "f1": am.get("f1"), "precision": am.get("precision"),
                "recall": am.get("recall")})

    m_b = _cl_block(sup_name, sup_acc)
    bm = compute_anomaly_metrics(y_bin, _sup_scores(sup, X_eval)) if np.unique(y_bin).size > 1 else {}
    m_b.update({"auroc": bm.get("auroc"), "f1": bm.get("f1"), "precision": bm.get("precision"),
                "recall": bm.get("recall")})

    # --- Ensemble (4 règles) + désaccord ---
    ensemble = _ensemble_metrics(pair, X_eval, y_bin, rules)
    pred_maha, pred_sup = pair.predict_individual(X_eval)
    disagreement = _disagreement_block(pred_maha, pred_sup, X_eval, y_bin, maha_scores)

    return {
        "exp_id": f"exp_S30_PC_{pair_cfg['pair']['name']}_{dataset}",
        "pair": pair_cfg["pair"]["name"],
        "dataset": dataset,
        "frame": "binary",
        "n_eval": int(X_eval.shape[0]),
        "model_a": m_a,
        "model_b": m_b,
        "ensemble": ensemble,
        "disagreement": disagreement,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------
# Partie B — tâches natives (S3007)
# ----------------------------------------------------------------------------
def _train_native_supervised(sup_name: str, dataset: str, native_path: str):
    """Entraîne le modèle supervisé en tâche native. Renvoie (predict_fn, tasks, native_summary).

    predict_fn(X) → sortie native (RUL cycles pour régression, label classe pour multiclasse).
    """
    if sup_name != "ewc":
        # Les configs natives existantes (cmapss_rul / cwru_multiclass) sont au schéma
        # EWC (régression / multi-classe). HDC/TinyOL n'ont pas de config native calibrée
        # (feature_bounds, backbone) → on ne fabrique aucun chiffre (règle CLAUDE.md).
        raise ValueError(
            f"mode native non supporté pour supervised={sup_name!r} (EWC uniquement ; "
            f"configs natives HDC/TinyOL non disponibles)."
        )

    cfg = load_config(native_path)
    set_seed(int(cfg.get("seed", 42)))

    if dataset == "cmapss":  # RUL régression
        from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor

        tr = _load_script("train_ewc_rul")
        tasks = tr._load_tasks(cfg, native_path)
        model = EWCMlpRegressor(
            input_dim=cfg["INPUT_DIM"],
            hidden_dims=cfg.get("HIDDEN_DIMS", [32, 16]),
            dropout=cfg.get("DROPOUT", 0.2),
            ewc_lambda=cfg.get("EWC_LAMBDA", 400.0),
        )
        opt = torch.optim.SGD(model.parameters(), lr=cfg["EWC_LR"], momentum=cfg.get("MOMENTUM", 0.9))
        crit = torch.nn.MSELoss()
        rul_scale = float(cfg.get("rul_cap", 125))
        rmses = []
        for task in tasks:
            for _ in range(cfg.get("N_EPOCHS_PER_TASK", 20)):
                tr.train_one_epoch(model, task["train_loader"], opt, crit, rul_scale)
            model.consolidate(task["train_loader"], n_samples=cfg.get("FISHER_N_SAMPLES", 200),
                              rul_scale=rul_scale)
            rmses.append(tr.evaluate_task(model, task["val_loader"], rul_scale)["rmse"])

        def predict_fn(X):  # noqa: N803
            model.eval()
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
                return (model(t).squeeze().cpu().numpy().ravel()) * rul_scale

        return predict_fn, tasks, {"native_metric": "rmse_mean", "value": float(np.mean(rmses))}

    if dataset == "cwru":  # multi-classe
        from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

        tm = _load_script("train_ewc_multiclass")
        tasks = tm._load_tasks(cfg, native_path)
        n_classes = cfg["N_CLASSES"]
        model = EWCMlpMulticlass(
            input_dim=cfg["INPUT_DIM"],
            n_classes=n_classes,
            hidden_dims=cfg.get("HIDDEN_DIMS", [32, 16]),
            dropout=cfg.get("DROPOUT", 0.2),
            ewc_lambda=cfg.get("EWC_LAMBDA", 400.0),
        )
        opt = torch.optim.SGD(model.parameters(), lr=cfg["EWC_LR"], momentum=cfg.get("MOMENTUM", 0.9))
        crit = torch.nn.CrossEntropyLoss()
        f1s = []
        for task in tasks:
            for _ in range(cfg.get("N_EPOCHS_PER_TASK", 30)):
                tm.train_one_epoch(model, task["train_loader"], opt, crit)
            model.consolidate(task["train_loader"], n_samples=cfg.get("FISHER_N_SAMPLES", 200))
            f1s.append(float(tm.evaluate_task(model, task["val_loader"], n_classes)["f1_macro"]))

        def predict_fn(X):  # noqa: N803
            model.eval()
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
                return model(t).argmax(dim=1).cpu().numpy().ravel()

        return predict_fn, tasks, {"native_metric": "f1_macro_mean", "value": float(np.mean(f1s))}

    raise ValueError(f"Mode natif non supporté pour le dataset {dataset!r}.")


def run_native(pair_cfg: dict, dataset: str, device: str) -> dict:
    sup_name = pair_cfg["pair"]["supervised"]
    native_cfgs = pair_cfg.get("native_configs", {})
    if dataset not in native_cfgs:
        raise SystemExit(
            f"[native] pas de config native pour dataset={dataset} dans {pair_cfg['pair']['name']}. "
            f"Datasets natifs disponibles : {sorted(native_cfgs)}"
        )
    rule_cfg = pair_cfg["native"][dataset]
    seed = int(pair_cfg.get("seed", 42))

    # --- Modèle supervisé natif ---
    predict_fn, tasks, native_summary = _train_native_supervised(
        sup_name, dataset, native_cfgs[dataset]
    )

    # --- Mahalanobis sur les mêmes tâches natives ---
    set_seed(seed)
    detector, maha_acc = _train_mahalanobis(pair_cfg["mahalanobis"], tasks)

    # --- Jeu d'évaluation commun ---
    X_eval, y_native = _concat_eval(tasks)
    y_fault = native_to_fault(y_native, rule_cfg["rule"], rule_cfg.get("threshold"))

    # Décision binaire dérivée de la sortie native du supervisé (désaccord).
    sup_native_pred = predict_fn(X_eval)
    sup_fault = native_to_fault(sup_native_pred, rule_cfg["rule"], rule_cfg.get("threshold"))

    maha_scores = detector.anomaly_score(X_eval)
    maha_fault = detector.predict(X_eval)

    has_two = np.unique(y_fault).size > 1
    m_a = _cl_block("mahalanobis", maha_acc)
    am = compute_anomaly_metrics(y_fault, maha_scores) if has_two else {}
    m_a.update({"auroc": am.get("auroc"), "f1": am.get("f1")})

    m_b = {
        "name": f"{sup_name}_native",
        **native_summary,
        "f1_fault": float(f1_score(y_fault, sup_fault, zero_division=0)),
    }

    # Désaccord entre décisions binaires faute (maha vs native_to_fault(supervisé)).
    disagreement = _disagreement_block(
        np.asarray(maha_fault).ravel(), np.asarray(sup_fault).ravel(), X_eval, y_fault, maha_scores
    )

    return {
        "exp_id": f"exp_S30_PC_native_{pair_cfg['pair']['name']}_{dataset}",
        "pair": pair_cfg["pair"]["name"],
        "dataset": dataset,
        "frame": "native",
        "native_rule": rule_cfg,
        "n_eval": int(X_eval.shape[0]),
        "model_a": m_a,
        "model_b": m_b,
        "disagreement": disagreement,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _write_outputs(result: dict, pair_cfg: dict, dataset: str, mode: str) -> Path:
    exp_dir = _ROOT / "experiments" / result["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(result), f, indent=2, ensure_ascii=False)
    snapshot = {**pair_cfg, "_dataset": dataset, "_mode": mode}
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, allow_unicode=True, sort_keys=False)
    return exp_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 30 — entraînement d'une paire de modèles")
    p.add_argument("--config", required=True, help="configs/board_pair_*.yaml")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", default="binary", choices=("binary", "native"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pair_cfg = load_config(args.config)

    print(f"\n{'=' * 64}")
    print(f"  Paire {pair_cfg['pair']['name']} × {args.dataset} — mode={args.mode}")
    print(f"{'=' * 64}")

    try:
        if args.mode == "native":
            result = run_native(pair_cfg, args.dataset, args.device)
        else:
            result = run_binary(pair_cfg, args.dataset, args.device)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        # Limitation de construction (ex. HDC×Paderborn : feature_bounds non calibrés) :
        # on n'invente aucun chiffre (règle CLAUDE.md) — artefact « à mesurer » explicite.
        prefix = "exp_S30_PC_native" if args.mode == "native" else "exp_S30_PC"
        result = {
            "exp_id": f"{prefix}_{pair_cfg['pair']['name']}_{args.dataset}",
            "pair": pair_cfg["pair"]["name"],
            "dataset": args.dataset,
            "frame": args.mode,
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
            "model_a": {"name": "mahalanobis"},
            "model_b": {"name": pair_cfg["pair"]["supervised"]},
            "disagreement": {"rate": None},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        exp_dir = _write_outputs(result, pair_cfg, args.dataset, args.mode)
        print(f"\n⚠️  Paire ignorée ({result['reason']}) — artefact « à mesurer » : "
              f"{exp_dir}/results.json")
        return

    exp_dir = _write_outputs(result, pair_cfg, args.dataset, args.mode)

    ma, mb = result["model_a"], result["model_b"]
    print(f"\n  model_a (maha)     : {ma}")
    print(f"  model_b ({mb['name']}) : {mb}")
    if "ensemble" in result:
        print(f"  ensemble best_rule : {result['ensemble']['best_rule']}")
    print(f"  disagreement.rate  : {result['disagreement']['rate']:.4f}")
    print(f"  → {exp_dir}/results.json")


if __name__ == "__main__":
    main()
