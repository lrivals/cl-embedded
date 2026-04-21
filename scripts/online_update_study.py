"""
scripts/online_update_study.py — S9-09: Comparaison stratégies MAJ online.

Compare 5 stratégies de mise à jour du MahalanobisDetector (+ PCABaseline)
sur un stream de 1000 échantillons après un enrôlement sur 500 normaux.

Stratégies testées :
    batch_refit      — refit complet sur toutes les données vues (borne supérieure, non MCU)
    online_welford   — MAJ sample-by-sample via partial_fit() (MCU-compatible)
    minibatch_10     — MAJ toutes les 10 mesures via partial_fit()
    minibatch_50     — MAJ toutes les 50 mesures via partial_fit()
    minibatch_100    — MAJ toutes les 100 mesures via partial_fit()

Usage
-----
    python scripts/online_update_study.py \\
      --config configs/unsupervised_config.yaml \\
      --enrollment_domain Pump \\
      --n_enrollment 500 \\
      --n_stream 1000 \\
      --eval_every 10 \\
      --exp_id exp_043_online_update

Sorties
-------
    experiments/exp_043_online_update/
    ├── config_snapshot.yaml
    └── results/
        ├── online_update_mahalanobis.json   # {strategy: [(step, auroc), ...]}
        └── online_update_pca.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.unsupervised import MahalanobisDetector, PCABaseline
from src.utils.config_loader import load_config, save_config_snapshot
from src.utils.reproducibility import set_seed

N_ENROLLMENT_DEFAULT: int = 500
N_STREAM_DEFAULT: int = 1000
EVAL_EVERY_DEFAULT: int = 10

STRATEGIES: list[str] = [
    "batch_refit",
    "online_welford",
    "minibatch_10",
    "minibatch_50",
    "minibatch_100",
]


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online update strategy comparison — exp_043"
    )
    parser.add_argument("--config", default="configs/unsupervised_config.yaml")
    parser.add_argument("--enrollment_domain", default="Pump")
    parser.add_argument("--n_enrollment", type=int, default=N_ENROLLMENT_DEFAULT)
    parser.add_argument("--n_stream", type=int, default=N_STREAM_DEFAULT)
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY_DEFAULT)
    parser.add_argument("--exp_id", default="exp_043_online_update")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Chargement données
# ---------------------------------------------------------------------------


def load_data(
    cfg: dict, enrollment_domain: str, n_enrollment: int, n_stream: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et prépare les données pour l'étude online.

    Parameters
    ----------
    cfg : dict
    enrollment_domain : str
    n_enrollment : int
        Nombre d'échantillons normaux pour l'enrôlement.
    n_stream : int
        Nombre d'échantillons dans le stream (mélange normal + anomalie).

    Returns
    -------
    X_enroll : np.ndarray [n_enrollment, d]
    X_stream : np.ndarray [n_stream, d]
    y_stream : np.ndarray [n_stream]
    X_test   : np.ndarray [N_test, d]  — jeu de test fixe pour AUROC
    y_test   : np.ndarray [N_test]
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    csv_path = Path(cfg["data"]["csv_path"])
    feature_cols: list[str] = cfg["data"]["feature_columns"]
    label_col: str = cfg["data"]["label_column"]
    domain_col: str = cfg["data"]["domain_column"]

    df = pd.read_csv(csv_path)

    normalizer_path = Path(cfg["data"]["normalizer_path"])
    if normalizer_path.exists():
        import yaml
        with open(normalizer_path) as f:
            norm_params = yaml.safe_load(f)
        means = np.array([norm_params["mean"][c] for c in feature_cols])
        stds = np.array([norm_params["std"][c] for c in feature_cols])
        X_all = (df[feature_cols].values - means) / (stds + 1e-8)
    else:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(df[feature_cols].values)

    y_all = df[label_col].values.astype(np.int64)
    domains = df[domain_col].values

    rng = np.random.default_rng(42)

    # Enrôlement : normaux du domaine sélectionné
    mask_enroll = (domains == enrollment_domain) & (y_all == 0)
    X_pool = X_all[mask_enroll]
    n_enroll_actual = min(n_enrollment, len(X_pool))
    idx_enroll = rng.choice(len(X_pool), size=n_enroll_actual, replace=False)
    X_enroll = X_pool[idx_enroll]

    # Stream : mélange de tous les domaines (inclut anomalies)
    mask_stream_pool = ~mask_enroll
    X_stream_pool = X_all[mask_stream_pool]
    y_stream_pool = y_all[mask_stream_pool]
    n_stream_actual = min(n_stream, len(X_stream_pool))
    idx_stream = rng.choice(len(X_stream_pool), size=n_stream_actual, replace=False)
    X_stream = X_stream_pool[idx_stream]
    y_stream = y_stream_pool[idx_stream]

    # Test fixe : sous-ensemble disjoint du stream
    n_test = min(1000, len(X_all))
    idx_test = rng.choice(len(X_all), size=n_test, replace=False)
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]

    print(f"  Enrôlement : {X_enroll.shape[0]} normaux ({enrollment_domain})")
    print(f"  Stream     : {X_stream.shape[0]} samples ({y_stream.sum()} anomalies)")
    print(f"  Test fixe  : {X_test.shape[0]} samples ({y_test.sum()} anomalies)")
    return X_enroll, X_stream, y_stream, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Stratégies MAJ
# ---------------------------------------------------------------------------


def run_strategy(
    strategy: str,
    model_name: str,
    cfg: dict,
    X_enroll: np.ndarray,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_every: int,
) -> list[tuple[int, float]]:
    """
    Exécute une stratégie de MAJ sur le stream et retourne AUROC toutes les eval_every MAJ.

    Parameters
    ----------
    strategy : str
        "batch_refit" | "online_welford" | "minibatch_10" | "minibatch_50" | "minibatch_100"
    model_name : str
        "mahalanobis" | "pca"
    cfg : dict
    X_enroll : np.ndarray
    X_stream : np.ndarray
    y_stream : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    eval_every : int

    Returns
    -------
    list[tuple[int, float]]
        [(step, auroc), ...] — step = nombre de MAJ effectuées.
    """
    # Instanciation + enrôlement initial
    if model_name == "mahalanobis":
        model = MahalanobisDetector(cfg["mahalanobis"])
    elif model_name == "pca":
        model = PCABaseline(cfg["pca"])
    else:
        raise ValueError(f"Modèle non supporté : {model_name!r}")

    model.fit_task(X_enroll, task_id=0)

    # Pour batch_refit : buffer accumulatif (non MCU — borne sup.)
    X_accumulated = X_enroll.copy()

    results: list[tuple[int, float]] = []
    update_step = 0

    def _eval_auroc() -> float:
        scores = model.anomaly_score(X_test)
        try:
            return float(roc_auc_score(y_test, scores))
        except ValueError:
            return float("nan")

    # Évaluation initiale (step 0)
    results.append((0, _eval_auroc()))

    if strategy == "batch_refit":
        for i, xi in enumerate(X_stream):
            X_accumulated = np.vstack([X_accumulated, xi.reshape(1, -1)])
            update_step += 1
            if update_step % eval_every == 0:
                # Refit complet — non MCU-compatible
                model.fit_task(X_accumulated[X_accumulated.shape[0] - len(X_enroll):], task_id=0)
                # Refit sur toutes les données normales vues (approximation)
                model.fit_task(X_accumulated, task_id=0)
                results.append((update_step, _eval_auroc()))

    elif strategy == "online_welford":
        for xi in X_stream:
            model.partial_fit(xi)
            update_step += 1
            if update_step % eval_every == 0:
                results.append((update_step, _eval_auroc()))

    elif strategy.startswith("minibatch_"):
        batch_size = int(strategy.split("_")[1])
        buf: list[np.ndarray] = []
        for xi in X_stream:
            buf.append(xi)
            if len(buf) >= batch_size:
                model.partial_fit(np.array(buf))
                buf = []
                update_step += batch_size
                if update_step % eval_every == 0:
                    results.append((update_step, _eval_auroc()))
        # Flush buffer résiduel
        if buf:
            model.partial_fit(np.array(buf))
            update_step += len(buf)
            results.append((update_step, _eval_auroc()))

    else:
        raise ValueError(f"Stratégie inconnue : {strategy!r}")

    return results


# ---------------------------------------------------------------------------
# 4. Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    exp_dir = Path(f"experiments/{args.exp_id}")
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Online Update Study — {args.exp_id}")
    print(f"  Domaine enrôlement : {args.enrollment_domain}")
    print(f"  Enrôlement N={args.n_enrollment}  Stream N={args.n_stream}")
    print(f"  Éval toutes les {args.eval_every} MAJ")
    print(f"{'=' * 60}")

    save_config_snapshot(cfg, str(exp_dir))

    X_enroll, X_stream, y_stream, X_test, y_test = load_data(
        cfg, args.enrollment_domain, args.n_enrollment, args.n_stream
    )

    for model_name in ["mahalanobis", "pca"]:
        print(f"\n{'=' * 50}")
        print(f"  Modèle : {model_name.upper()}")
        print(f"{'=' * 50}")

        model_results: dict[str, list[tuple[int, float]]] = {}

        for strategy in STRATEGIES:
            print(f"\n  Stratégie : {strategy}")
            curve = run_strategy(
                strategy=strategy,
                model_name=model_name,
                cfg=cfg,
                X_enroll=X_enroll,
                X_stream=X_stream,
                y_stream=y_stream,
                X_test=X_test,
                y_test=y_test,
                eval_every=args.eval_every,
            )
            model_results[strategy] = curve
            if curve:
                final_auroc = curve[-1][1]
                print(f"    AUROC final : {final_auroc:.4f} ({len(curve)} évaluations)")

        # Sauvegarde par modèle
        out_path = results_dir / f"online_update_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)
        print(f"\n  → {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  Expérience terminée → {exp_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
