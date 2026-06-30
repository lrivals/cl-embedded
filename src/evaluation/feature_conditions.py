"""
feature_conditions.py — Moteur partagé pour la sélection de features par modèle (Sprint 35).

Fournit aux scripts S3501 (`select_best_features_per_model.py`) et S3503
(`run_feature_condition_sweep.py`) :

  1. ``NATIVE_FEATURE_NAMES`` / ``load_native_task_arrays`` — chargement des tâches CL
     à la dimension **native** de chaque dataset (les colonnes sont ensuite tranchées).
  2. ``train_and_evaluate`` — entraînement CL + évaluation (F1 classe faulty, acc_final,
     avg_forgetting, RAM, n_params) d'un modèle sur un sous-ensemble de colonnes donné,
     plus un ``predict_fn`` (scores) réutilisable par ``permutation_importance``.

Aucune logique CL n'est réimplémentée : les boucles reprennent celles de
``scripts/train_ewc.py`` (EWC) et ``scripts/train_mahalanobis.py`` /
``src/training/scenarios.py`` (détecteurs). Les hyperparamètres viennent des configs
canoniques (`configs/{model}_config.yaml`), jamais codés en dur.
"""

from __future__ import annotations

import tempfile
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.data.cmapss_loader import SENSOR_NAMES as _CMAPSS_SENSORS
from src.data.cwru_dataset import FEATURE_COLS as _CWRU_FEATS
from src.data.paderborn_loader import FEATURE_NAMES_RAW as _PADERBORN_FEATS
from src.data.pronostia_dataset import FEATURE_NAMES as _PRONOSTIA_FEATS
from src.evaluation.metrics import compute_cl_metrics
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed

MODELS: list[str] = ["mahalanobis", "ewc", "tinyol", "hdc"]
DATASETS: list[str] = ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"]

# Noms de features natives par dataset (importés des constantes des loaders).
NATIVE_FEATURE_NAMES: dict[str, list[str]] = {
    "cwru": list(_CWRU_FEATS),
    "monitoring": ["temperature", "pressure", "vibration", "humidity"],
    "pronostia": list(_PRONOSTIA_FEATS),
    "cmapss": list(_CMAPSS_SENSORS),
    "paderborn": list(_PADERBORN_FEATS),
}

# Config canonique de chaque modèle (hyperparamètres — jamais en dur ici).
MODEL_CONFIG_PATHS: dict[str, str] = {
    "ewc": "configs/ewc_config.yaml",
    "hdc": "configs/hdc_config.yaml",
    "tinyol": "configs/tinyol_config.yaml",
    # board_mahalanobis est flat ; on lit le bloc `mahalanobis` d'une config dataset.
    "mahalanobis": "configs/cwru_by_fault_config.yaml",
}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des tâches à la dimension native
# ──────────────────────────────────────────────────────────────────────────────


def _extract_arrays(loader) -> tuple[np.ndarray, np.ndarray]:
    """Concatène un DataLoader en (X, y) numpy."""
    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy().ravel())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _tasks_to_arrays(tasks: list[dict]) -> list[dict]:
    """Transforme une liste de tâches DataLoader en tâches numpy (train/val)."""
    out: list[dict] = []
    for t in tasks:
        x_tr, y_tr = _extract_arrays(t["train_loader"])
        x_va, y_va = _extract_arrays(t["val_loader"])
        out.append(
            {
                "task_id": int(t.get("task_id", len(out))),
                "domain": str(t.get("domain", f"task_{len(out)}")),
                "X_train": x_tr.astype(np.float32),
                "y_train": y_tr.astype(np.int64),
                "X_val": x_va.astype(np.float32),
                "y_val": y_va.astype(np.int64),
            }
        )
    return out


def load_native_task_arrays(dataset: str, seed: int = 42) -> list[dict]:
    """
    Charge les tâches CL d'un dataset à sa **dimension native** (numpy).

    Pour cmapss/paderborn, le loader applique par défaut un subset top-5 ; on force
    la liste native complète via ``feature_names`` pour obtenir toutes les colonnes.

    Returns
    -------
    list[dict]
        Chaque tâche : ``task_id``, ``domain``, ``X_train``, ``y_train``,
        ``X_val``, ``y_val`` (colonnes dans l'ordre de ``NATIVE_FEATURE_NAMES``).
    """
    set_seed(seed)

    if dataset == "cwru":
        from src.data.cwru_dataset import get_cwru_cl_dataloaders_by_fault_type

        cfg = load_config("configs/cwru_by_fault_config.yaml")
        tasks = get_cwru_cl_dataloaders_by_fault_type(
            csv_path=Path(cfg["data"]["csv_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            test_ratio=cfg["data"].get("test_ratio", 0.2),
            val_ratio=cfg["data"].get("val_ratio", 0.1),
            seed=seed,
        )
    elif dataset == "monitoring":
        from src.data.monitoring_dataset import get_cl_dataloaders

        cfg = load_config("configs/ewc_config.yaml")
        tasks = get_cl_dataloaders(
            csv_path=Path(cfg["data"]["csv_path"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["training"].get("batch_size", 32),
            val_ratio=cfg["data"].get("test_split", 0.2),
            seed=seed,
        )
    elif dataset == "pronostia":
        from src.data.pronostia_dataset import get_pronostia_dataloaders

        cfg = load_config("configs/pronostia_config.yaml")
        tasks = get_pronostia_dataloaders(
            npy_dir=Path(cfg["data"]["npy_dir"]),
            normalizer_path=Path(cfg["data"]["normalizer_path"]),
            batch_size=cfg["data"].get("batch_size", 32),
            val_ratio=cfg["data"].get("val_ratio", 0.2),
            seed=seed,
            window_size=cfg["data"].get("window_size", 2560),
            step_size=cfg["data"].get("step_size", 2560),
            failure_ratio=cfg["data"].get("failure_ratio", 0.10),
            label_mode=cfg["data"].get("label_mode", "failure_ratio"),
            faulty_threshold=cfg["data"].get("faulty_threshold"),
        )
    elif dataset == "cmapss":
        from src.data.cmapss_loader import get_cl_dataloaders

        cfg_path = "configs/cmapss_config.yaml"
        cfg = load_config(cfg_path)
        tasks = get_cl_dataloaders(
            data_dir=Path(cfg["data"]["data_dir"]),
            config_path=Path(cfg_path),
            feature_names=NATIVE_FEATURE_NAMES["cmapss"],  # force 21 capteurs natifs
        )
    elif dataset == "paderborn":
        from src.data.paderborn_loader import get_cl_dataloaders

        cfg_path = "configs/paderborn_config.yaml"
        cfg = load_config(cfg_path)
        tasks = get_cl_dataloaders(
            data_dir=Path(cfg["data"]["data_dir"]),
            config_path=Path(cfg_path),
            feature_names=NATIVE_FEATURE_NAMES["paderborn"],  # force 7 features natives
        )
    else:
        raise ValueError(f"Dataset inconnu : {dataset!r}. Attendu : {DATASETS}")

    return _tasks_to_arrays(tasks)


CONDITIONS: list[str] = ["5feat", "all", "best"]


def resolve_feature_indices(condition: str, model: str, dataset: str) -> tuple[list[int], str]:
    """
    Retourne (indices natifs sélectionnés, note explicative) pour une cellule.

    Source de vérité unique partagée par le sweep PC (S3503), le driver board (S3508)
    et ``sensor_stream.py`` (S3508), garantissant que board et PC consomment EXACTEMENT
    les mêmes colonnes (parité par construction).

    Conditions :
      - ``all``   → toutes les dims natives ;
      - ``5feat`` → ``configs/{dataset}_feature_subset.yaml`` (subset top-5, ≡ all si absent) ;
      - ``best``  → ``configs/best_features/{model}_{dataset}.yaml`` (S3501, par modèle).

    Lève FileNotFoundError si la config de features requise est absente.
    """
    native = NATIVE_FEATURE_NAMES[dataset]
    n = len(native)

    if condition == "all":
        return list(range(n)), "dims natives"

    if condition == "5feat":
        path = Path(f"configs/{dataset}_feature_subset.yaml")
        if not path.exists():
            # monitoring (4 features natives) : pas de subset top-5 → all.
            return list(range(n)), f"pas de subset top-5 ({dataset}) → 5feat≡all ({n} feats)"
        sub = yaml.safe_load(path.read_text())
        if "feature_indices" in sub and sub["feature_indices"]:
            idx = [int(i) for i in sub["feature_indices"]]
        else:
            names = sub.get("selected_features") or sub.get("features") or sub.get("feature_names")
            idx = [native.index(name) for name in names]
        return idx, f"subset {path.name}"

    if condition == "best":
        path = Path(f"configs/best_features/{model}_{dataset}.yaml")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} manquant — lancer d'abord scripts/select_best_features_per_model.py "
                f"--model {model} --dataset {dataset} (S3501)."
            )
        best = yaml.safe_load(path.read_text())
        return [int(i) for i in best["selected_indices"]], f"best {path.name}"

    raise ValueError(f"Condition inconnue : {condition!r}. Attendu : {CONDITIONS}")


def load_condition_arrays(
    dataset: str, condition: str, model: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """
    Charge (X[N,k], y[N]) d'une cellule (dataset, condition, model) — features tranchées.

    Concatène train+val de toutes les tâches CL (ordre temporel conservé) à la
    dimension ``k`` de la condition. Utilisé par le driver board (entraînement de
    référence + streaming) pour que board et PC voient les mêmes nombres.

    Returns
    -------
    (X, y, feature_indices, feature_names)
    """
    idx, _note = resolve_feature_indices(condition, model, dataset)
    tasks = load_native_task_arrays(dataset, seed=seed)
    Xs, ys = [], []
    for t in tasks:
        Xs.append(t["X_train"][:, idx])
        ys.append(t["y_train"])
        Xs.append(t["X_val"][:, idx])
        ys.append(t["y_val"])
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    names = [NATIVE_FEATURE_NAMES[dataset][i] for i in idx]
    return X, y, list(idx), names


def _slice(task_arrays: list[dict], idx: list[int]) -> list[dict]:
    """Restreint les colonnes de X_train/X_val aux indices ``idx`` (ordre conservé)."""
    cols = list(idx)
    out = []
    for t in task_arrays:
        out.append(
            {
                **t,
                "X_train": t["X_train"][:, cols],
                "X_val": t["X_val"][:, cols],
            }
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Construction + entraînement des modèles (dispatch par type)
# ──────────────────────────────────────────────────────────────────────────────


def _make_loaders(task_arrays: list[dict], batch_size: int):
    """Construit des DataLoaders torch (train/val) par tâche depuis les arrays."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = []
    for t in task_arrays:
        tr = TensorDataset(
            torch.from_numpy(t["X_train"]).float(),
            torch.from_numpy(t["y_train"].astype(np.float32)).unsqueeze(1),
        )
        va = TensorDataset(
            torch.from_numpy(t["X_val"]).float(),
            torch.from_numpy(t["y_val"].astype(np.float32)).unsqueeze(1),
        )
        loaders.append(
            {
                "task_id": t["task_id"],
                "domain": t["domain"],
                "train_loader": DataLoader(tr, batch_size=batch_size, shuffle=True),
                "val_loader": DataLoader(va, batch_size=max(batch_size, 64)),
            }
        )
    return loaders


def _feature_bounds_dict(X: np.ndarray) -> dict[str, list[float]]:
    """Bornes [min, max] par colonne — format racine attendu par HDCClassifier."""
    return {
        f"feat_{i}": [float(X[:, i].min()), float(X[:, i].max())]
        for i in range(X.shape[1])
    }


def _eval_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """F1 faulty + macro + précision/rappel — définition partagée PC↔board (S3504)."""
    from src.evaluation.metrics import compute_fault_f1

    return compute_fault_f1(y_true, y_pred)


def train_and_evaluate(
    model_name: str,
    task_arrays: list[dict],
    feature_indices: list[int],
    seed: int = 42,
) -> dict:
    """
    Entraîne ``model_name`` sur les colonnes ``feature_indices`` et évalue.

    Returns
    -------
    dict
        ``acc_final``, ``avg_forgetting``, ``backward_transfer``, ``f1_faulty``,
        ``f1_macro``, ``precision_faulty``, ``recall_faulty``, ``n_features``,
        ``n_params``, ``ram_peak_bytes``,
        ``predict_fn`` (Callable[np.ndarray]->scores, sur dim len(feature_indices)).
    """
    set_seed(seed)
    sliced = _slice(task_arrays, feature_indices)
    k = len(feature_indices)

    tracemalloc.start()
    if model_name == "ewc":
        result = _train_ewc(sliced, k, seed)
    elif model_name == "mahalanobis":
        result = _train_mahalanobis(sliced, seed)
    elif model_name == "hdc":
        result = _train_hdc(sliced, k, seed)
    elif model_name == "tinyol":
        result = _train_tinyol(sliced, k, seed)
    else:
        tracemalloc.stop()
        raise ValueError(f"Modèle inconnu : {model_name!r}. Attendu : {MODELS}")
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    acc_matrix = result["acc_matrix"]
    cl = compute_cl_metrics(acc_matrix)

    # F1 sur la concaténation des val de toutes les tâches (état final du modèle).
    y_all = np.concatenate([t["y_val"] for t in sliced])
    preds_all = np.concatenate([result["predict_labels"](t["X_val"]) for t in sliced])
    f1 = _eval_f1(y_all, preds_all)

    return {
        "acc_final": float(cl["aa"]),
        "avg_forgetting": float(cl["af"]),
        "backward_transfer": float(cl["bwt"]),
        "f1_faulty": f1["f1_faulty"],
        "f1_macro": f1["f1_macro"],
        "precision_faulty": f1["precision_faulty"],
        "recall_faulty": f1["recall_faulty"],
        "n_features": k,
        "n_params": int(result["n_params"]),
        "ram_peak_bytes": int(ram_peak),
        "predict_fn": result["predict_scores"],
    }


# --- EWC (regularization-based, supervisé) -----------------------------------


def _train_ewc(sliced: list[dict], k: int, seed: int) -> dict:
    import torch
    import torch.optim as optim

    from scripts.train_ewc import evaluate_task, train_ewc
    from src.models.ewc import EWCMlpClassifier

    cfg = load_config(MODEL_CONFIG_PATHS["ewc"])
    cfg["model"]["input_dim"] = k
    cfg["training"]["seed"] = seed

    batch_size = cfg["training"].get("batch_size", 32)
    loaders = _make_loaders(sliced, batch_size)

    model = EWCMlpClassifier(
        input_dim=k,
        hidden_dims=cfg["model"].get("hidden_dims", [32, 16]),
        dropout=cfg["model"].get("dropout", 0.2),
    )
    acc_matrix = train_ewc(model, loaders, cfg, device="cpu")

    def predict_scores(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X).float()).cpu().numpy().flatten()

    def predict_labels(X: np.ndarray) -> np.ndarray:
        return (predict_scores(X) >= 0.5).astype(np.int64)

    return {
        "acc_matrix": acc_matrix,
        "predict_scores": predict_scores,
        "predict_labels": predict_labels,
        "n_params": model.count_parameters(),
    }


# --- Mahalanobis (distance, non supervisé) -----------------------------------


def _train_mahalanobis(sliced: list[dict], seed: int) -> dict:
    from sklearn.metrics import accuracy_score

    from src.models.unsupervised import MahalanobisDetector

    cfg = load_config(MODEL_CONFIG_PATHS["mahalanobis"])
    model = MahalanobisDetector(cfg["mahalanobis"])

    T = len(sliced)
    acc_matrix = np.full((T, T), np.nan)
    for i, t in enumerate(sliced):
        model.fit_task(t["X_train"], task_id=i)
        thr = model.threshold_
        for j in range(i + 1):
            preds = (model.anomaly_score(sliced[j]["X_val"]) > thr).astype(int)
            acc_matrix[i, j] = float(accuracy_score(sliced[j]["y_val"], preds))

    thr_final = model.threshold_

    def predict_scores(X: np.ndarray) -> np.ndarray:
        return model.anomaly_score(X)

    def predict_labels(X: np.ndarray) -> np.ndarray:
        return (model.anomaly_score(X) > thr_final).astype(np.int64)

    return {
        "acc_matrix": acc_matrix,
        "predict_scores": predict_scores,
        "predict_labels": predict_labels,
        "n_params": int(model.count_parameters()),
    }


# --- HDC (architecture-based, supervisé binaire) -----------------------------


def _train_hdc(sliced: list[dict], k: int, seed: int) -> dict:
    from src.models.hdc import HDCClassifier
    from src.training.scenarios import evaluate_task_generic, run_cl_scenario

    base = load_config(MODEL_CONFIG_PATHS["hdc"])
    # Bornes calculées sur le train de la 1re tâche (cohérent avec la spec HDC).
    bounds = _feature_bounds_dict(sliced[0]["X_train"])
    bv_path = Path(tempfile.gettempdir()) / f"hdc_bv_k{k}_{seed}.npz"
    cfg = {
        "hdc": {
            "D": base["hdc"]["D"],
            "n_levels": base["hdc"]["n_levels"],
            "seed": seed,
            "base_vectors_path": str(bv_path),
        },
        "data": {"n_features": k, "n_classes": 2},
        "feature_bounds": bounds,
        "one_class_mode": False,
        "training": {"epochs_per_task": 1, "batch_size": 1},
    }
    model = HDCClassifier(cfg)
    loaders = _make_loaders(sliced, batch_size=1)
    acc_matrix = run_cl_scenario(model, loaders, cfg)

    def predict_labels(X: np.ndarray) -> np.ndarray:
        return model.predict(X).astype(np.int64)

    def predict_scores(X: np.ndarray) -> np.ndarray:
        return predict_labels(X).astype(np.float32)

    n_params = int(getattr(model, "count_parameters", lambda: 0)())
    return {
        "acc_matrix": acc_matrix,
        "predict_scores": predict_scores,
        "predict_labels": predict_labels,
        "n_params": n_params,
    }


# --- TinyOL (architecture-based, anomaly one-class) --------------------------


def _train_tinyol(sliced: list[dict], k: int, seed: int) -> dict:
    from sklearn.metrics import accuracy_score

    from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector

    base = load_config(MODEL_CONFIG_PATHS["tinyol"])
    # L'autoencodeur a 3 couches encodeur/décodeur (enc1→enc2→enc3) → dims de longueur 3,
    # dérivées de k (constantes nommées, pas de dim board en dur). Bottleneck < input_dim.
    bottleneck = max(2, k // 2)
    h2 = max(bottleneck + 1, k)
    h1 = max(2 * k, h2 + 1)
    cfg = {
        "backbone": {
            "input_dim": k,
            "encoder_dims": [h1, h2, bottleneck],
            "decoder_dims": [h2, h1, k],
            "checkpoint_path": None,
        },
        "pretrain": {
            "optimizer": base["pretrain"].get("optimizer", "adam"),
            "learning_rate": base["pretrain"].get("learning_rate", 0.001),
            "epochs": base["pretrain"].get("epochs", 50),
            "batch_size": base["pretrain"].get("batch_size", 64),
        },
        "anomaly_percentile": base.get("anomaly_percentile", 95),
        "anomaly_threshold": None,
    }
    model = TinyOLAnomalyDetector(cfg)

    T = len(sliced)
    acc_matrix = np.full((T, T), np.nan)
    for i, t in enumerate(sliced):
        # Entraînement one-class : données normales (faulty==0) uniquement.
        x_norm = t["X_train"][t["y_train"] == 0]
        if len(x_norm) == 0:
            x_norm = t["X_train"]
        model.update(x_norm, np.zeros(len(x_norm), dtype=np.int64))
        model.on_task_end(task_id=i, dataloader=None)
        if model.anomaly_threshold_ is None:
            continue
        for j in range(i + 1):
            preds = model.predict(sliced[j]["X_val"])
            acc_matrix[i, j] = float(accuracy_score(sliced[j]["y_val"], preds))

    def predict_scores(X: np.ndarray) -> np.ndarray:
        return model.anomaly_score(X)

    def predict_labels(X: np.ndarray) -> np.ndarray:
        if model.anomaly_threshold_ is None:
            return np.zeros(len(X), dtype=np.int64)
        return model.predict(X).astype(np.int64)

    n_params = sum(p.numel() for p in model.autoencoder.parameters())
    return {
        "acc_matrix": acc_matrix,
        "predict_scores": predict_scores,
        "predict_labels": predict_labels,
        "n_params": int(n_params),
    }


def predict_fn_factory(result: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Expose le predict_fn (scores) d'un résultat ``train_and_evaluate``."""
    return result["predict_fn"]
