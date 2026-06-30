"""
scripts/benchmark_int8_fp32.py — Benchmark unifié FP32 vs INT8 (Sprint 28, S2801).

Pour un couple (modèle, config), entraîne et évalue les variantes FP32 et INT8 du
modèle, mesure la RAM (poids) et la latence d'inférence, puis écrit un JSON normalisé
ingéré par generate_int8_heatmaps.py (S2810). 100 % PC Python — pas de board.

Usage :
    python scripts/benchmark_int8_fp32.py \
        --model {ewc,hdc,tinyol,mahalanobis} \
        --config configs/ewc_int8_monitoring.yaml \
        --output experiments/exp_S28_PC_ewc_hdc/results_ewc_monitoring.json \
        [--n_samples 500]

Périmètre (S2801 étendu par S2807/S2808) :
    - ewc : EWCMlpClassifier FP32 + EWCMlpInt8Classifier INT8. Modèle binaire → AUROC de
      détection de panne sur labels binarisés *normal-vs-fault* (multiclasse/régression).
    - hdc : best-effort — HDC est nativement INT8, donc métrique INT8 == FP32, seule la
      RAM diffère. Un avertissement explicite est émis.
    - tinyol : TinyOLAnomalyDetector FP32 + TinyOLAutoencoderInt8 INT8 (S2804). AUROC sur
      l'erreur de reconstruction MSE.
    - mahalanobis : MahalanobisDetector FP32 + MahalanobisDetectorInt8 INT8 (S2805). AUROC
      sur la distance de Mahalanobis.

Critères Gap 3 (docs/triple_gap.md) :
    gap3_metric_ok = abs(delta_metric) < 0.02
    gap3_ram_ok    = ram_ratio > 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from src.evaluation.memory_profiler import profile_forward_pass  # noqa: E402
from src.utils.config_loader import load_config_extends  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402


# ----------------------------------------------------------------------------
# Import des helpers EWC existants (scripts/ n'est pas un package)
# ----------------------------------------------------------------------------
def _load_train_ewc_module():
    """Charge scripts/train_ewc.py comme module pour réutiliser ses helpers."""
    spec = importlib.util.spec_from_file_location(
        "_train_ewc_helpers", _ROOT / "scripts" / "train_ewc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Utilitaires communs
# ----------------------------------------------------------------------------
def _truncate_tasks(tasks: list[dict], n_samples: int) -> list[dict]:
    """
    Tronque chaque DataLoader des tâches à ``n_samples`` exemples (tests rapides).

    Reconstruit train/val/test loaders à partir d'un Subset des n premiers indices,
    en préservant le batch_size d'origine. Déterministe (pas de shuffle).
    """
    from torch.utils.data import DataLoader, Subset

    truncated = []
    for task in tasks:
        new_task = dict(task)
        for key in ("train_loader", "val_loader", "test_loader"):
            loader = task.get(key)
            if loader is None:
                continue
            dataset = loader.dataset
            n = min(n_samples, len(dataset))
            subset = Subset(dataset, list(range(n)))
            batch_size = getattr(loader, "batch_size", 32) or 32
            new_task[key] = DataLoader(subset, batch_size=batch_size, shuffle=False)
        truncated.append(new_task)
    return truncated


def _measure_callable_latency(fn, sample: np.ndarray, n_runs: int = 100) -> float:
    """Latence moyenne (ms) d'un appel ``fn(sample)`` sur n_runs (modèles non-nn)."""
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(sample)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(latencies))


# ----------------------------------------------------------------------------
# Détection de panne binaire — AUROC commun (EWC + détecteurs d'anomalie)
# ----------------------------------------------------------------------------
def _binarize_labels(y: np.ndarray) -> np.ndarray:
    """Réduit des labels {0,1}, multiclasse ou continus à du binaire *normal-vs-fault*.

    Convention : la classe « normale/saine » est la plus petite valeur (typiquement 0).
    Tout le reste est considéré comme une panne (1). Sur des labels déjà binaires {0,1}
    l'opération est l'identité.
    """
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if uniq.size <= 1:
        return y.astype(int)
    return (y != uniq.min()).astype(int)


def _auroc_fault(labels, scores) -> float:
    """AUROC de détection de panne : score d'anomalie/proba vs label binarisé.

    Retourne NaN si une seule classe est présente (AUROC indéfini).
    """
    yb = _binarize_labels(np.asarray(labels))
    s = np.asarray(scores, dtype=float).ravel()
    if np.unique(yb).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yb, s))
    except ValueError:
        return float("nan")


def _mean_auroc_over_tasks(per_task: list[tuple]) -> float:
    """Moyenne des AUROC par tâche (NaN ignorés)."""
    vals = [_auroc_fault(lab, sc) for lab, sc in per_task]
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _loader_to_numpy(loader) -> np.ndarray:
    """Concatène tous les X (features) d'un DataLoader en un array float32 [N, d]."""
    xs = [x.numpy().astype(np.float32) for x, _ in loader]
    return np.concatenate(xs, axis=0) if xs else np.empty((0, 0), dtype=np.float32)


def _first_task_train_X(tasks: list[dict]) -> np.ndarray:
    """X d'entraînement de la 1re tâche — jeu de calibration INT8 représentatif."""
    return _loader_to_numpy(tasks[0]["train_loader"])


# ----------------------------------------------------------------------------
# Adaptateur EWC (complet)
# ----------------------------------------------------------------------------
class EWCAdapter:
    """FP32 = EWCMlpClassifier, INT8 = EWCMlpInt8Classifier — câblage réel."""

    metric_name = "auroc"

    def __init__(self) -> None:
        self._te = _load_train_ewc_module()
        from src.models.ewc import EWCMlpClassifier, EWCMlpInt8Classifier

        self._fp32_cls = EWCMlpClassifier
        self._int8_cls = EWCMlpInt8Classifier

    def load_tasks(self, cfg: dict, config_path: str) -> list[dict]:
        cfg = dict(cfg)
        cfg["_config_path"] = config_path
        tasks = self._te._get_tasks(cfg)
        cfg.pop("_config_path", None)
        return tasks

    def build_fp32(self, cfg: dict):
        return self._fp32_cls(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
        )

    def build_int8(self, cfg: dict):
        return self._int8_cls(
            input_dim=cfg["model"]["input_dim"],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
        )

    def train(self, model, tasks: list[dict], cfg: dict, device: str) -> None:
        model.train()
        self._te.train_ewc(model, tasks, cfg, device)

    def evaluate(self, model, tasks: list[dict], device: str) -> float:
        """AUROC de détection de panne, moyennée par tâche (labels binarisés).

        EWCMlpClassifier est binaire (sortie sigmoïde unique) : pour les datasets
        multiclasses (cwru/pronostia/paderborn) ou de régression (cmapss), les labels
        sont réduits à *normal-vs-fault* via :func:`_binarize_labels`. La proba de sortie
        sert de score d'anomalie.
        """
        model.eval()
        per_task = []
        with torch.no_grad():
            for task in tasks:
                loader = task.get("test_loader") or task["val_loader"]
                probs, labels = [], []
                for x, y in loader:
                    out = model(x.to(device))
                    probs.extend(np.asarray(out.cpu()).ravel().tolist())
                    labels.extend(np.asarray(y).ravel().tolist())
                per_task.append((labels, probs))
        return _mean_auroc_over_tasks(per_task)

    def ram_bytes(self, model, dtype: str) -> int:
        return int(model.estimate_ram_bytes(dtype))

    def latency_ms(self, model, cfg: dict, tasks: list[dict]) -> float:
        res = profile_forward_pass(model, (1, cfg["model"]["input_dim"]))
        return float(res["inference_latency_ms"])


# ----------------------------------------------------------------------------
# Adaptateur HDC (best-effort — métrique INT8 == FP32 par construction)
# ----------------------------------------------------------------------------
class HDCAdapter:
    """HDC nativement INT8 : seule la RAM diffère entre FP32 et INT8."""

    metric_name = "f1_macro"
    native_int8 = True

    def __init__(self) -> None:
        self._te = _load_train_ewc_module()
        from src.models.hdc.hdc_classifier import HDCClassifier

        self._cls = HDCClassifier

    def load_tasks(self, cfg: dict, config_path: str) -> list[dict]:
        cfg = dict(cfg)
        cfg["_config_path"] = config_path
        tasks = self._te._get_tasks(cfg)
        cfg.pop("_config_path", None)
        return tasks

    def build_fp32(self, cfg: dict):
        return self._cls(cfg)

    def build_int8(self, cfg: dict):
        return self._cls(cfg)

    def train(self, model, tasks: list[dict], cfg: dict, device: str) -> None:
        for task in tasks:
            for x, y in task["train_loader"]:
                model.update(
                    x.numpy().astype(np.float32),
                    y.numpy().ravel().astype(int),
                )
            model.on_task_end(task["task_id"], task["train_loader"])

    def evaluate(self, model, tasks: list[dict], device: str) -> float:
        y_true, y_pred = [], []
        for task in tasks:
            loader = task.get("test_loader") or task["val_loader"]
            for x, y in loader:
                preds = model.predict(x.numpy().astype(np.float32))
                y_pred.extend(np.asarray(preds).ravel().tolist())
                y_true.extend(y.numpy().ravel().astype(int).tolist())
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def ram_bytes(self, model, dtype: str) -> int:
        return int(model.estimate_ram_bytes(dtype))

    def latency_ms(self, model, cfg: dict, tasks: list[dict]) -> float:
        n_features = cfg["data"]["n_features"]
        sample = np.zeros((1, n_features), dtype=np.float32)
        return _measure_callable_latency(model.predict, sample)


# ----------------------------------------------------------------------------
# Adaptateur TinyOL (détecteur d'anomalie autoencoder — S2804)
# ----------------------------------------------------------------------------
class TinyOLAdapter:
    """FP32 = TinyOLAnomalyDetector (MSE), INT8 = TinyOLAutoencoderInt8 (poids INT8).

    Le score d'anomalie est l'erreur de reconstruction MSE ; l'AUROC est calculée sur
    le label binarisé *normal-vs-fault*. Le variant INT8 enveloppe l'autoencoder FP32
    entraîné et applique une fake-quantization (poids INT8, activations UINT8).
    """

    metric_name = "auroc"

    def __init__(self) -> None:
        self._te = _load_train_ewc_module()
        from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector
        from src.models.tinyol.tinyol_int8 import TinyOLAutoencoderInt8

        self._fp32_cls = TinyOLAnomalyDetector
        self._int8_wrap = TinyOLAutoencoderInt8

    def load_tasks(self, cfg: dict, config_path: str) -> list[dict]:
        cfg = dict(cfg)
        cfg["_config_path"] = config_path
        tasks = self._te._get_tasks(cfg)
        cfg.pop("_config_path", None)
        return tasks

    def build_fp32(self, cfg: dict):
        return self._fp32_cls(cfg)

    def build_int8(self, cfg: dict):
        model = self._fp32_cls(cfg)
        model._use_int8 = True  # marqueur consommé par train/evaluate/ram/latency
        return model

    def train(self, model, tasks: list[dict], cfg: dict, device: str) -> None:
        # Refit : l'autoencoder est réentraîné par tâche (on_task_end vide le buffer).
        for i, task in enumerate(tasks):
            for x, y in task["train_loader"]:
                model.update(x.numpy().astype(np.float32), y.numpy().ravel())
            model.on_task_end(i + 1, task["train_loader"])  # task_id 1-based
        if getattr(model, "_use_int8", False):
            model._int8 = self._int8_wrap(model.autoencoder)
            model._int8.calibrate_int8(_first_task_train_X(tasks))

    def evaluate(self, model, tasks: list[dict], device: str) -> float:
        use_int8 = getattr(model, "_use_int8", False)
        per_task = []
        for task in tasks:
            loader = task.get("test_loader") or task["val_loader"]
            scores, labels = [], []
            for x, y in loader:
                xa = x.numpy().astype(np.float32)
                if use_int8:
                    s = np.array([model._int8.reconstruction_error_int8(xi) for xi in xa])
                else:
                    s = model.anomaly_score(xa)
                scores.extend(np.asarray(s).ravel().tolist())
                labels.extend(y.numpy().ravel().tolist())
            per_task.append((labels, scores))
        return _mean_auroc_over_tasks(per_task)

    def ram_bytes(self, model, dtype: str) -> int:
        if dtype == "int8" and getattr(model, "_int8", None) is not None:
            return int(model._int8.get_memory_footprint_int8()["total_bytes"])
        return int(model.estimate_ram_bytes(dtype))

    def latency_ms(self, model, cfg: dict, tasks: list[dict]) -> float:
        n = cfg["backbone"]["input_dim"]
        sample = np.zeros((1, n), dtype=np.float32)
        if getattr(model, "_use_int8", False):
            return _measure_callable_latency(model._int8.reconstruction_error_int8, sample)
        return _measure_callable_latency(model.anomaly_score, sample)


# ----------------------------------------------------------------------------
# Adaptateur Mahalanobis (détecteur d'anomalie par distance — S2805)
# ----------------------------------------------------------------------------
class MahalanobisAdapter:
    """FP32 = MahalanobisDetector, INT8 = MahalanobisDetectorInt8 (μ/Σ⁻¹ INT8).

    Score = distance de Mahalanobis ; AUROC sur label binarisé *normal-vs-fault*.
    """

    metric_name = "auroc"

    def __init__(self) -> None:
        self._te = _load_train_ewc_module()
        from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector
        from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8

        self._fp32_cls = MahalanobisDetector
        self._int8_cls = MahalanobisDetectorInt8

    @staticmethod
    def _maha_cfg(cfg: dict) -> dict:
        return dict(cfg.get("mahalanobis", {}))

    def load_tasks(self, cfg: dict, config_path: str) -> list[dict]:
        cfg = dict(cfg)
        cfg["_config_path"] = config_path
        tasks = self._te._get_tasks(cfg)
        cfg.pop("_config_path", None)
        return tasks

    def build_fp32(self, cfg: dict):
        return self._fp32_cls(self._maha_cfg(cfg))

    def build_int8(self, cfg: dict):
        model = self._int8_cls(self._maha_cfg(cfg))
        model._use_int8 = True
        return model

    def train(self, model, tasks: list[dict], cfg: dict, device: str) -> None:
        # Refit/Welford selon cl_strategy ; labels ignorés (fit non supervisé).
        for i, task in enumerate(tasks):
            X = _loader_to_numpy(task["train_loader"])
            model.fit_task(X, task_id=i)  # task_id 0-based ; seuil calculé sur task 0
        if getattr(model, "_use_int8", False):
            model.calibrate_int8()

    def evaluate(self, model, tasks: list[dict], device: str) -> float:
        use_int8 = getattr(model, "_use_int8", False)
        per_task = []
        for task in tasks:
            loader = task.get("test_loader") or task["val_loader"]
            scores, labels = [], []
            for x, y in loader:
                xa = x.numpy().astype(np.float32)
                s = model.anomaly_score_int8(xa) if use_int8 else model.anomaly_score(xa)
                scores.extend(np.asarray(s).ravel().tolist())
                labels.extend(y.numpy().ravel().tolist())
            per_task.append((labels, scores))
        return _mean_auroc_over_tasks(per_task)

    def ram_bytes(self, model, dtype: str) -> int:
        if dtype == "int8" and getattr(model, "_use_int8", False):
            return int(model.get_memory_footprint_int8()["total_bytes"])
        d = int(model.n_features_)
        return (d + d * d) * 4  # μ + Σ⁻¹ en float32

    def latency_ms(self, model, cfg: dict, tasks: list[dict]) -> float:
        d = int(model.n_features_) or int(self._maha_cfg(cfg).get("n_features", 5))
        sample = np.zeros((1, d), dtype=np.float32)
        fn = model.anomaly_score_int8 if getattr(model, "_use_int8", False) else model.anomaly_score
        return _measure_callable_latency(fn, sample)


MODEL_ADAPTERS = {
    "ewc": EWCAdapter,
    "hdc": HDCAdapter,
    "tinyol": TinyOLAdapter,
    "mahalanobis": MahalanobisAdapter,
}


# ----------------------------------------------------------------------------
# Construction du dict résultat (testable isolément)
# ----------------------------------------------------------------------------
def build_result_dict(
    model_name: str,
    dataset: str,
    config_path: str,
    metric_name: str,
    fp32_metric: float,
    fp32_ram: int,
    fp32_latency: float,
    int8_metric: float,
    int8_ram: int,
    int8_latency: float,
) -> dict:
    """Construit le dict de sortie normalisé (schéma S2801)."""
    delta_metric = int8_metric - fp32_metric
    ram_ratio = (fp32_ram / int8_ram) if int8_ram else float("inf")
    return {
        "model": model_name,
        "dataset": dataset,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fp32": {
            "metric_name": metric_name,
            "metric_value": round(float(fp32_metric), 6),
            "ram_bytes": int(fp32_ram),
            "latency_ms": round(float(fp32_latency), 6),
        },
        "int8": {
            "metric_name": metric_name,
            "metric_value": round(float(int8_metric), 6),
            "ram_bytes": int(int8_ram),
            "latency_ms": round(float(int8_latency), 6),
        },
        "delta_metric": round(float(delta_metric), 6),
        "ram_ratio": round(float(ram_ratio), 6),
        "gap3_metric_ok": bool(abs(delta_metric) < 0.02),
        "gap3_ram_ok": bool(ram_ratio > 1.0),
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_benchmark(
    model_name: str,
    config_path: str,
    output_path: str,
    n_samples: int | None = None,
    device: str = "cpu",
) -> dict:
    """Exécute le benchmark FP32 vs INT8 et écrit le JSON normalisé."""
    if model_name not in MODEL_ADAPTERS:
        raise ValueError(
            f"Modèle inconnu : {model_name}. Choix : {sorted(MODEL_ADAPTERS)}"
        )

    adapter = MODEL_ADAPTERS[model_name]()

    cfg = load_config_extends(config_path)
    dataset = cfg["data"].get("dataset", "unknown")
    seed = cfg.get("training", {}).get("seed", 42)

    if getattr(adapter, "native_int8", False):
        print(
            f"⚠️  {model_name.upper()} est nativement INT8 : la métrique INT8 est "
            f"identique à FP32, seule la RAM diffère."
        )

    print(f"\n{'=' * 60}")
    print(f"  Benchmark INT8 vs FP32 — modèle={model_name} dataset={dataset}")
    print(f"{'=' * 60}")

    # --- Chargement des tâches (une fois) ---
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)

    # --- Phase FP32 ---
    print("\n[1/2] FP32...")
    set_seed(seed)
    fp32_model = adapter.build_fp32(cfg)
    adapter.train(fp32_model, tasks, cfg, device)
    fp32_metric = adapter.evaluate(fp32_model, tasks, device)
    fp32_ram = adapter.ram_bytes(fp32_model, "fp32")
    fp32_latency = adapter.latency_ms(fp32_model, cfg, tasks)
    print(
        f"  {adapter.metric_name}={fp32_metric:.4f} | "
        f"ram={fp32_ram} B | lat={fp32_latency:.4f} ms"
    )

    # --- Phase INT8 ---
    print("\n[2/2] INT8...")
    set_seed(seed)
    int8_model = adapter.build_int8(cfg)
    adapter.train(int8_model, tasks, cfg, device)
    int8_metric = adapter.evaluate(int8_model, tasks, device)
    int8_ram = adapter.ram_bytes(int8_model, "int8")
    int8_latency = adapter.latency_ms(int8_model, cfg, tasks)
    print(
        f"  {adapter.metric_name}={int8_metric:.4f} | "
        f"ram={int8_ram} B | lat={int8_latency:.4f} ms"
    )

    # --- Résultat ---
    result = build_result_dict(
        model_name=model_name,
        dataset=dataset,
        config_path=config_path,
        metric_name=adapter.metric_name,
        fp32_metric=fp32_metric,
        fp32_ram=fp32_ram,
        fp32_latency=fp32_latency,
        int8_metric=int8_metric,
        int8_ram=int8_ram,
        int8_latency=int8_latency,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Δ {adapter.metric_name} : {result['delta_metric']:+.4f}  "
          f"(Gap 3 métrique : {'✅' if result['gap3_metric_ok'] else '❌'})")
    print(f"  RAM ratio FP32/INT8 : {result['ram_ratio']:.2f}×  "
          f"(Gap 3 RAM : {'✅' if result['gap3_ram_ok'] else '❌'})")
    print(f"  → {out}")
    print(f"{'=' * 60}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark INT8 vs FP32 (S2801)")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ADAPTERS))
    parser.add_argument("--config", required=True, help="Config YAML (support extends:)")
    parser.add_argument("--output", required=True, help="Chemin du JSON de sortie")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limite d'exemples par tâche (tests rapides)",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
