# ruff: noqa: N802, N803  — X / _loader_Xy suivent la convention mathématique ML (sklearn)
"""
scripts/train_meta_learner.py — Sprint 31 (S3103/S3104) — entraînement & évaluation du
méta-modèle d'arbitrage (stacking) au-dessus d'une paire « Mahalanobis + supervisé ».

Démontre que l'arbitrage *appris* (:class:`~src.ensemble.meta_learner.MetaLearner`) **bat ou
égale** les meilleures alternatives du Sprint 30 :
  (a) le meilleur modèle individuel (Mahalanobis ou supervisé) ;
  (b) la meilleure règle d'ensemble fixe (or / and / soft_vote / weighted).

Anti-fuite (S3101) : les bases sont entraînées sur les `train_loader` ; le jeu d'évaluation
(concaténation des `val_loader`/`test_loader`) est donc déjà out-of-fold pour elles. Ce jeu est
ensuite scindé en `meta-fit` / `meta-eval` disjoints : le méta est *entraîné* sur meta-fit et
*évalué* sur meta-eval, où sont aussi mesurées toutes les baselines (comparaison équitable).

Chaque run entraîne et compare les deux types de méta (`logreg`, `mlp`) et retient le meilleur
F1 (`best_meta`). Méthodologie de mesure (F1/AUROC) identique à `scripts/train_model_pair.py`
(Sprint 30) pour des deltas comparables.

Réutilise au maximum `scripts/train_model_pair.py` (helpers d'entraînement des bases) via import
dynamique.

Usage :
    python scripts/train_meta_learner.py --config configs/meta_stacking.yaml --pair maha_ewc --dataset monitoring
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.ensemble.meta_learner import MetaLearner, build_meta_features  # noqa: E402
from src.ensemble.model_pair import ModelPair, _binarize_labels  # noqa: E402
from src.evaluation.anomaly_metrics import compute_anomaly_metrics  # noqa: E402
from src.utils.config_loader import load_config, load_config_extends  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

DATASETS = ("monitoring", "cwru", "pronostia", "cmapss", "paderborn")


def _load_script(name: str):
    """Charge un module scripts/<name>.py pour réutiliser ses fonctions."""
    spec = importlib.util.spec_from_file_location(f"_meta_{name}", _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC si 2 classes présentes, sinon None (aucun chiffre inventé)."""
    if np.unique(y).size <= 1:
        return None
    return float(compute_anomaly_metrics(y, scores)["auroc"])


def _ext_f1(y: np.ndarray, scores: np.ndarray) -> float | None:
    """F1 au seuil optimal sur un score continu (méthodo Sprint 30 : compute_anomaly_metrics)."""
    if np.unique(y).size <= 1:
        return None
    return float(compute_anomaly_metrics(y, scores)["f1"])


# ----------------------------------------------------------------------------
# Run principal (cadre binaire, Partie A)
# ----------------------------------------------------------------------------
def run_meta(meta_cfg: dict, pair_cfg: dict, dataset: str, device: str) -> tuple[dict, dict]:
    """Entraîne les bases + le méta, évalue contre les baselines. Renvoie (result, best_weights)."""
    tmp = _load_script("train_model_pair")  # helpers d'entraînement des bases (Sprint 30)
    mcfg = meta_cfg["meta"]
    seed = int(mcfg["seed"])
    feats = list(mcfg["input_features"])
    kinds = list(mcfg["kinds"])
    hidden = int(mcfg.get("hidden_size", 8))
    class_weight = mcfg.get("class_weight", "balanced")
    oof = float(mcfg["oof_test_size"])

    sup_name = pair_cfg["pair"]["supervised"]
    sup_path = pair_cfg["supervised_configs"][dataset]
    rules = list(pair_cfg["fusion"]["rules"])

    # --- Tâches (config supervisée validée) ---
    set_seed(seed)
    sup_cfg = load_config_extends(sup_path)
    sup_cfg["_config_path"] = sup_path
    te = _load_script("train_ewc")
    tasks = te._get_tasks(sup_cfg)
    sup_cfg.pop("_config_path", None)

    # --- Entraînement des 2 bases sur les mêmes tâches (train_loader) ---
    set_seed(seed)
    detector, _ = tmp._train_mahalanobis(pair_cfg["mahalanobis"], tasks)
    set_seed(seed)
    sup, _ = tmp._train_supervised(sup_name, sup_cfg, tasks, device)

    # --- Jeu d'évaluation commun (déjà out-of-fold pour les bases) ---
    X_eval, y_raw = tmp._concat_eval(tasks)
    y_bin = _binarize_labels(y_raw)

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

    # --- Split out-of-fold meta-fit / meta-eval (disjoints) ---
    idx = np.arange(y_bin.shape[0])
    strat = y_bin if np.unique(y_bin).size > 1 else None
    fit_idx, ev_idx = train_test_split(idx, test_size=oof, random_state=seed, stratify=strat)

    meta_X_all, feat_names = build_meta_features(pair, X_eval, feats)
    y_fit, y_ev = y_bin[fit_idx], y_bin[ev_idx]

    # --- Méta (logreg + mlp) entraînés sur meta-fit, évalués sur meta-eval ---
    meta_block: dict[str, dict] = {}
    trained: dict[str, MetaLearner] = {}
    for kind in kinds:
        ml = MetaLearner(
            kind=kind,
            input_features=feat_names,
            hidden_size=hidden,
            class_weight=class_weight,
            seed=seed,
        )
        ml.fit(meta_X_all[fit_idx], y_fit)
        preds = ml.predict(meta_X_all[ev_idx])
        proba = ml.predict_proba(meta_X_all[ev_idx])
        meta_block[kind] = {
            "f1": float(f1_score(y_ev, preds, zero_division=0)),
            "auroc": _auroc(y_ev, proba),
        }
        trained[kind] = ml
    best_meta = max(meta_block, key=lambda k: meta_block[k]["f1"])

    # --- Baselines sur le MÊME meta-eval ---
    X_ev = X_eval[ev_idx]
    maha_scores = detector.anomaly_score(X_ev)
    model_a = {"name": "mahalanobis", "f1": _ext_f1(y_ev, maha_scores), "auroc": _auroc(y_ev, maha_scores)}
    sup_scores = tmp._sup_scores(sup, X_ev)
    model_b = {"name": sup_name, "f1": _ext_f1(y_ev, sup_scores), "auroc": _auroc(y_ev, sup_scores)}

    ensemble: dict[str, dict] = {}
    for rule in rules:
        preds = pair.predict_ensemble(X_ev, rule)
        proba = pair._fused_proba(X_ev, rule)
        ensemble[rule] = {
            "f1": float(f1_score(y_ev, preds, zero_division=0)),
            "auroc": _auroc(y_ev, proba),
        }
    best_rule = max(ensemble, key=lambda r: ensemble[r]["f1"])

    # --- Deltas (F1 du best_meta vs meilleures alternatives Sprint 30) ---
    meta_f1 = meta_block[best_meta]["f1"]
    best_indiv_f1 = max(v for v in (model_a["f1"], model_b["f1"]) if v is not None)
    best_ens_f1 = ensemble[best_rule]["f1"]

    result = {
        "exp_id": f"exp_S31_PC_{pair_cfg['pair']['name']}_{dataset}",
        "pair": pair_cfg["pair"]["name"],
        "dataset": dataset,
        "frame": "binary",
        "n_eval": int(X_eval.shape[0]),
        "oof": {"meta_fit": int(fit_idx.size), "meta_eval": int(ev_idx.size), "test_size": oof},
        "input_features": feat_names,
        "meta": meta_block,
        "best_meta": best_meta,
        "baselines": {
            "model_a": model_a,
            "model_b": model_b,
            "ensemble": ensemble,
            "best_ensemble_rule": best_rule,
            "best_individual_f1": float(best_indiv_f1),
            "best_ensemble_f1": float(best_ens_f1),
        },
        "delta_vs_best_individual": float(meta_f1 - best_indiv_f1),
        "delta_vs_ensemble": float(meta_f1 - best_ens_f1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result, trained[best_meta].export_weights()


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _write_outputs(
    result: dict, meta_cfg: dict, pair_cfg: dict, dataset: str, weights: dict | None
) -> Path:
    tmp = _load_script("train_model_pair")
    exp_dir = _ROOT / "experiments" / result["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(tmp._jsonify(result), f, indent=2, ensure_ascii=False)
    snapshot = {**meta_cfg, "_pair_config": pair_cfg, "_dataset": dataset, "_mode": "binary"}
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, allow_unicode=True, sort_keys=False)
    if weights is not None:
        with open(exp_dir / "meta_weights.json", "w", encoding="utf-8") as f:
            json.dump(tmp._jsonify(weights), f, indent=2, ensure_ascii=False)
    return exp_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 31 — méta-modèle d'arbitrage (stacking)")
    p.add_argument("--config", required=True, help="configs/meta_stacking.yaml")
    p.add_argument("--pair", required=True, help="clé de paire (ex. maha_ewc)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_cfg = load_config(args.config)
    if args.pair not in meta_cfg["pairs"]:
        raise SystemExit(f"paire inconnue : {args.pair!r} (disponibles : {sorted(meta_cfg['pairs'])}).")
    pair_cfg = load_config(meta_cfg["pairs"][args.pair])

    print(f"\n{'=' * 64}")
    print(f"  Méta {args.pair} × {args.dataset} — kinds={meta_cfg['meta']['kinds']}")
    print(f"{'=' * 64}")

    try:
        result, weights = run_meta(meta_cfg, pair_cfg, args.dataset, args.device)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        # Limitation de construction (ex. maha_hdc×paderborn : feature_bounds non calibrés) :
        # artefact « à mesurer » explicite, aucun chiffre inventé (règle CLAUDE.md).
        result = {
            "exp_id": f"exp_S31_PC_{pair_cfg['pair']['name']}_{args.dataset}",
            "pair": pair_cfg["pair"]["name"],
            "dataset": args.dataset,
            "frame": "binary",
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        exp_dir = _write_outputs(result, meta_cfg, pair_cfg, args.dataset, None)
        print(f"\n⚠️  Méta ignoré ({result['reason']}) — artefact « à mesurer » : "
              f"{exp_dir}/results.json")
        return

    exp_dir = _write_outputs(result, meta_cfg, pair_cfg, args.dataset, weights)
    bm = result["best_meta"]
    print(f"\n  meta[{bm}]            : {result['meta'][bm]}")
    print(f"  best_individual_f1   : {result['baselines']['best_individual_f1']:.4f}")
    print(f"  best_ensemble ({result['baselines']['best_ensemble_rule']}) : "
          f"{result['baselines']['best_ensemble_f1']:.4f}")
    print(f"  Δ vs best individual : {result['delta_vs_best_individual']:+.4f}")
    print(f"  Δ vs best ensemble   : {result['delta_vs_ensemble']:+.4f}")
    print(f"  → {exp_dir}/results.json")


if __name__ == "__main__":
    main()
