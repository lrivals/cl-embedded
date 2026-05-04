"""
profile_memory.py — Profiling RAM des modèles CL embarqués via tracemalloc.

Usage :
    python scripts/profile_memory.py --model ewc_oneclass --dataset monitoring \
        --config configs/ewc_oneclass_config.yaml

    python scripts/profile_memory.py --model all --dataset pronostia \
        --config configs/unsupervised_config.yaml \
        --ewc_config configs/ewc_oneclass_config.yaml

Objectif : valider le Gap 2 (sub-100 Ko RAM avec chiffres précis mesurés).

Note méthodologique — deux mesures distinctes à reporter :
    1. RAM statique (get_ram_bytes / estimate_ram_bytes)  : poids modèle + éventuels
       buffers internes, en octets FP32. Représente la RAM statique nécessaire sur MCU.
    2. RAM peak inférence (tracemalloc PC, forward pass) : proxy de la RAM
       d'activation pendant une inférence. Forward pass seul, sans backprop.
    3. RAM lifecycle PC (tracemalloc, training inclus) : NON représentatif MCU
       (inclut le overhead PyTorch autograd et les buffers d'optimisation).

Les mesures MCU réelles seront effectuées en Phase 2 (portage STM32N6).
"""

from __future__ import annotations

import argparse
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import yaml

# Assure que src/ est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.memory_profiler import full_memory_report
from src.models.ewc.ewc_oneclass import EWCOneClassDetector

BUDGET_BYTES: int = 65_536  # 64 Ko


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_inference_ram(model, x_single: np.ndarray) -> int:
    """tracemalloc peak pendant un appel anomaly_score (1 sample)."""
    tracemalloc.start()
    model.anomaly_score(x_single)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def _print_model_block(label: str, input_dim: int, n_params: int,
                       ram_static: int, ram_inf_peak: int,
                       dbscan_variable: bool = False) -> None:
    within_static = ram_static < BUDGET_BYTES
    within_inf = ram_inf_peak < BUDGET_BYTES
    s1 = "✅ DANS LE BUDGET" if within_static else "❌ DÉPASSE LE BUDGET"
    s2 = "⚠️  RAM variable (N_train×d×4B — non MCU)" if dbscan_variable else (
        "✅ DANS LE BUDGET" if within_inf else "❌ DÉPASSE LE BUDGET"
    )
    print(f"\n   ┌─ {label} (input_dim={input_dim}) ──────────────────────┐")
    print(f"   │  Paramètres          : {n_params:>8,}")
    print(f"   │  RAM statique        : {ram_static:>8,} B  ({ram_static/1024:.2f} Ko)")
    print(f"   │  STM32N6 64Ko        : {s1}")
    print(f"   ├─ RAM inférence (tracemalloc PC, forward seul) ────┤")
    print(f"   │  RAM peak fwd        : {ram_inf_peak:>8,} B  ({ram_inf_peak/1024:.2f} Ko)")
    print(f"   │  STM32N6 64Ko        : {s2}")
    print(f"   └────────────────────────────────────────────────────┘")


def _print_comparison_table(results: list[dict]) -> None:
    """Tableau comparatif RAM Monitoring (4D) vs Pronostia (13D)."""
    BUDGET = BUDGET_BYTES
    print("\n" + "=" * 72)
    print("  TABLEAU COMPARATIF RAM — tous les modèles")
    print("=" * 72)
    header = f"{'Modèle':<18} {'dim':>4} {'RAM statique':>14} {'RAM inf peak':>14} {'≤ 64 Ko':>10}"
    print(header)
    print("-" * 72)
    for r in results:
        label = r["model_label"]
        dim = r["input_dim"]
        static = r["ram_static_bytes"]
        inf_pk = r["ram_inf_peak_bytes"]
        variable = r.get("ram_variable", False)
        ok_s = "✅" if static < BUDGET else "❌"
        ok_i = "⚠️ var" if variable else ("✅" if inf_pk < BUDGET else "❌")
        print(
            f"{label:<18} {dim:>4} "
            f"{static/1024:>11.2f} Ko "
            f"{inf_pk/1024:>11.2f} Ko "
            f"  {ok_s} / {ok_i}"
        )
    print("=" * 72)
    print("  ⚠️  DBSCAN : RAM statique croît avec N_train (non embarquable tel quel).")
    print("  Mesures PC uniquement — MCU validé en Phase 2 (portage STM32N6).\n")


# ---------------------------------------------------------------------------
# Profiling functions — un par modèle
# ---------------------------------------------------------------------------

def _profile_ewc_oneclass(config_path: Path, dataset: str) -> dict:
    """
    Profil RAM complet de EWCOneClassDetector.

    Mesure séparément :
    - RAM statique estimée (get_ram_bytes) — modèle + Fisher + θ*
    - RAM peak inférence (tracemalloc, forward pass seul) — proxy MCU
    - Cohérence get_ram_bytes vs n_params × 4 × N_matrices (±10%)

    Parameters
    ----------
    config_path : Path
        Chemin vers ewc_oneclass_config.yaml.
    dataset : str
        Clé dans DATASETS (ex. "monitoring", "pronostia", "cwru").

    Returns
    -------
    dict : rapport complet
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("MODEL", {})
    train_cfg = config.get("TRAINING", {})
    dataset_cfg = config.get("DATASETS", {}).get(dataset, {})

    input_dim: int = int(dataset_cfg.get("INPUT_DIM", 4))
    hidden_dim: int = int(dataset_cfg.get("HIDDEN_DIM", model_cfg.get("HIDDEN_DIM", 32)))
    latent_dim: int = int(model_cfg.get("LATENT_DIM", 8))
    lambda_ewc: float = float(model_cfg.get("LAMBDA_EWC", 400.0))
    n_epochs: int = int(train_cfg.get("N_EPOCHS", 20))
    lr: float = float(train_cfg.get("LR", 1e-3))
    batch_size: int = int(train_cfg.get("BATCH_SIZE", 32))

    print(f"\n🔍 Profiling EWCOneClassDetector — dataset={dataset}")
    print(f"   Arch : [{input_dim}→{hidden_dim}→{latent_dim}]→[{hidden_dim}→{input_dim}]")
    print(f"   λ_ewc={lambda_ewc}, epochs={n_epochs}, batch={batch_size}")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((100, input_dim)).astype(np.float32)

    # --- Entraînement (hors tracemalloc — overhead PC non représentatif MCU) ---
    detector = EWCOneClassDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lambda_ewc=lambda_ewc,
        n_epochs=n_epochs,
        lr=lr,
    )
    detector._batch_size = batch_size
    detector.fit_task(X_normal, task_id=0)
    X_task1 = rng.standard_normal((80, input_dim)).astype(np.float32)
    detector.fit_task(X_task1, task_id=1)

    # --- RAM statique : get_ram_bytes (poids + Fisher + θ*) ---
    get_ram = detector.get_ram_bytes()
    n_params = detector.count_parameters()
    theoretical_ram = n_params * 4 * 3  # modèle + Fisher + θ*
    ratio_internal = get_ram / theoretical_ram if theoretical_ram > 0 else float("inf")
    coherent_internal = abs(ratio_internal - 1.0) <= 0.10

    # --- RAM peak inférence (tracemalloc, forward pass seul) ---
    peak_inf = _measure_inference_ram(detector, X_normal[:1])

    within_budget_static = get_ram < BUDGET_BYTES
    within_budget_fwd = peak_inf < BUDGET_BYTES

    print(f"\n   ┌─ RAM statique (embarquée) ────────────────────────┐")
    print(f"   │  Paramètres          : {n_params:>8,} params")
    print(f"   │  get_ram_bytes()     : {get_ram:>8,} B  ({get_ram/1024:.2f} Ko)")
    print(f"   │  Théorique (×3×4B)   : {theoretical_ram:>8,} B  ({theoretical_ram/1024:.2f} Ko)")
    print(f"   │  Cohérence ±10%      : {'✅' if coherent_internal else '⚠️ '} (ratio={ratio_internal:.3f})")
    s1 = "✅ DANS LE BUDGET" if within_budget_static else "❌ DÉPASSE LE BUDGET"
    print(f"   │  STM32N6 64Ko        : {s1}")
    print(f"   ├─ RAM inférence (tracemalloc PC, forward seul) ────┤")
    print(f"   │  RAM peak fwd        : {peak_inf:>8,} B  ({peak_inf/1024:.2f} Ko)")
    s2 = "✅ DANS LE BUDGET" if within_budget_fwd else "❌ DÉPASSE LE BUDGET"
    print(f"   │  STM32N6 64Ko        : {s2}  ({peak_inf/BUDGET_BYTES*100:.1f}%)")
    print(f"   └────────────────────────────────────────────────────┘")

    # --- Forward pass complet via memory_profiler ---
    print()
    full_memory_report(
        model=detector._model,
        input_shape=(1, input_dim),
        model_name=f"_MLPAutoencoder({input_dim}→{hidden_dim}→{latent_dim})",
    )

    return {
        "model_label": "EWC OneClass",
        "model": f"EWCOneClassDetector_{dataset}",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "n_params": n_params,
        "get_ram_bytes": get_ram,
        "theoretical_ram_bytes": theoretical_ram,
        "coherent_internal": coherent_internal,
        "ram_static_bytes": get_ram,
        "ram_inf_peak_bytes": peak_inf,
        "within_budget_64ko_static": within_budget_static,
        "within_budget_64ko_inference": within_budget_fwd,
    }


def _profile_hdc(hdc_config_path: Path, dataset: str) -> dict:
    """Profil RAM HDCClassifier (one_class_mode=True)."""
    from src.models.hdc.hdc_classifier import HDCClassifier

    with open(hdc_config_path) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("DATASETS", {}).get(dataset, {})
    input_dim: int = int(ds_cfg.get("n_features", cfg.get("data", {}).get("n_features", 13)))
    D: int = int(cfg.get("hdc", {}).get("D", 1024))

    print(f"\n🔍 Profiling HDCClassifier — dataset={dataset}, D={D}, input_dim={input_dim}")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((200, input_dim)).astype(np.float32)

    cfg["one_class_mode"] = True
    cfg["cl_strategy"] = "refit"
    model = HDCClassifier(cfg)
    # HDC uses update()+set_anomaly_threshold() — no fit_task
    y_normal = np.zeros(X_normal.shape[0], dtype=np.int64)
    for i in range(0, len(X_normal), 64):
        model.update(X_normal[i : i + 64], y_normal[i : i + 64])
    model.set_anomaly_threshold(X_normal)

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")
    peak_inf = _measure_inference_ram(model, X_normal[:1])

    _print_model_block("HDCClassifier", input_dim, n_params, ram_static, peak_inf)

    return {
        "model_label": "HDC",
        "model": f"HDCClassifier_{dataset}",
        "input_dim": input_dim,
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inf_peak_bytes": peak_inf,
        "within_budget_64ko_static": ram_static < BUDGET_BYTES,
        "within_budget_64ko_inference": peak_inf < BUDGET_BYTES,
    }


def _profile_tinyol_ae(tinyol_config_path: Path, dataset: str) -> dict:
    """Profil RAM TinyOLAnomalyDetector (autoencoder seul)."""
    from src.models.tinyol.tinyol_anomaly_detector import TinyOLAnomalyDetector

    with open(tinyol_config_path) as f:
        cfg = yaml.safe_load(f)

    backbone_cfg = cfg.get("backbone", {})
    input_dim: int = int(backbone_cfg.get("input_dim", 13))

    print(f"\n🔍 Profiling TinyOLAnomalyDetector — dataset={dataset}, input_dim={input_dim}")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((200, input_dim)).astype(np.float32)

    # checkpoint_path=None → pas de chargement de poids pré-entraînés
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("backbone", {})
    cfg_copy["backbone"] = dict(backbone_cfg)
    cfg_copy["backbone"]["checkpoint_path"] = None

    model = TinyOLAnomalyDetector(cfg_copy)
    # TinyOL uses update()+on_task_end() — no fit_task
    y_zeros = np.zeros(X_normal.shape[0], dtype=np.int64)
    model.update(X_normal, y_zeros)  # buffers data
    model.on_task_end(1, None)  # trains AE + sets threshold

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")
    peak_inf = _measure_inference_ram(model, X_normal[:1])

    _print_model_block("TinyOL AE", input_dim, n_params, ram_static, peak_inf)

    return {
        "model_label": "TinyOL AE",
        "model": f"TinyOLAnomalyDetector_{dataset}",
        "input_dim": input_dim,
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inf_peak_bytes": peak_inf,
        "within_budget_64ko_static": ram_static < BUDGET_BYTES,
        "within_budget_64ko_inference": peak_inf < BUDGET_BYTES,
    }


def _profile_kmeans(config_path: Path, dataset: str) -> dict:
    """Profil RAM KMeansDetector."""
    from src.models.unsupervised import KMeansDetector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dim: int = int(cfg.get("DATASETS", {}).get(dataset, {}).get("INPUT_DIM", 13))
    kmeans_cfg = dict(cfg.get("kmeans", {}))

    print(f"\n🔍 Profiling KMeansDetector — dataset={dataset}, input_dim={input_dim}")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((200, input_dim)).astype(np.float32)

    model = KMeansDetector(kmeans_cfg)
    model.fit_task(X_normal, task_id=0)

    n_params = model.count_parameters()
    ram_static = n_params * 4  # centroides FP32
    peak_inf = _measure_inference_ram(model, X_normal[:1])

    _print_model_block("KMeansDetector", input_dim, n_params, ram_static, peak_inf)

    return {
        "model_label": "KMeans",
        "model": f"KMeansDetector_{dataset}",
        "input_dim": input_dim,
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inf_peak_bytes": peak_inf,
        "within_budget_64ko_static": ram_static < BUDGET_BYTES,
        "within_budget_64ko_inference": peak_inf < BUDGET_BYTES,
    }


def _profile_mahalanobis(config_path: Path, dataset: str) -> dict:
    """Profil RAM MahalanobisDetector."""
    from src.models.unsupervised import MahalanobisDetector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dim: int = int(cfg.get("DATASETS", {}).get(dataset, {}).get("INPUT_DIM", 13))
    maha_cfg = dict(cfg.get("mahalanobis", {}))

    print(f"\n🔍 Profiling MahalanobisDetector — dataset={dataset}, input_dim={input_dim}")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((200, input_dim)).astype(np.float32)

    model = MahalanobisDetector(maha_cfg)
    model.fit_task(X_normal, task_id=0)

    n_params = model.count_parameters()
    ram_static = (input_dim + input_dim * input_dim) * 4  # mu_ + sigma_inv_ @ FP32
    peak_inf = _measure_inference_ram(model, X_normal[:1])

    _print_model_block("MahalanobisDetector", input_dim, n_params, ram_static, peak_inf)
    print(f"   ℹ️  RAM théorique Mahalanobis : d + d² = {input_dim} + {input_dim**2} = "
          f"{input_dim + input_dim**2} floats → {ram_static} B @ FP32")

    return {
        "model_label": "Mahalanobis",
        "model": f"MahalanobisDetector_{dataset}",
        "input_dim": input_dim,
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inf_peak_bytes": peak_inf,
        "within_budget_64ko_static": ram_static < BUDGET_BYTES,
        "within_budget_64ko_inference": peak_inf < BUDGET_BYTES,
    }


def _profile_dbscan(config_path: Path, dataset: str) -> dict:
    """Profil RAM DBSCANDetector.

    RAM est variable (N_train × d × 4B) — DBSCAN conserve tous les points
    d'entraînement pour la prédiction (non MCU-compatible sans adaptation).
    """
    from src.models.unsupervised import DBSCANDetector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dim: int = int(cfg.get("DATASETS", {}).get(dataset, {}).get("INPUT_DIM", 13))
    dbscan_cfg = dict(cfg.get("dbscan", {}))

    N_TRAIN = 200
    print(f"\n🔍 Profiling DBSCANDetector — dataset={dataset}, input_dim={input_dim}")
    print(f"   ⚠️  RAM variable : dépend de N_train (ici N={N_TRAIN})")

    rng = np.random.default_rng(42)
    X_normal = rng.standard_normal((N_TRAIN, input_dim)).astype(np.float32)

    model = DBSCANDetector(dbscan_cfg)
    model.fit_task(X_normal, task_id=0)

    n_params = model.count_parameters()
    ram_static = n_params * 4  # points stockés FP32
    peak_inf = _measure_inference_ram(model, X_normal[:1])

    _print_model_block("DBSCANDetector", input_dim, n_params, ram_static, peak_inf,
                       dbscan_variable=True)
    print(f"   ℹ️  RAM DBSCAN : N_train({N_TRAIN}) × d({input_dim}) × 4B = {ram_static} B")
    print(f"   ℹ️  Pour N_train=1000 → {1000*input_dim*4} B ({1000*input_dim*4/1024:.1f} Ko)")

    return {
        "model_label": "DBSCAN",
        "model": f"DBSCANDetector_{dataset}",
        "input_dim": input_dim,
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inf_peak_bytes": peak_inf,
        "ram_variable": True,
        "within_budget_64ko_static": False,  # variable → ne peut pas garantir ≤ 64Ko
        "within_budget_64ko_inference": peak_inf < BUDGET_BYTES,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profiling RAM des modèles CL embarqués (STM32N6 — Gap 2)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["ewc_oneclass", "hdc", "tinyol_ae", "kmeans", "mahalanobis", "dbscan", "all"],
        help="Modèle à profiler ('all' pour tableau comparatif complet).",
    )
    parser.add_argument(
        "--dataset",
        default="monitoring",
        help="Clé dataset dans le fichier config (ex. monitoring, pronostia, cwru).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unsupervised_config.yaml"),
        help="Config principal (kmeans/mahalanobis/dbscan).",
    )
    parser.add_argument(
        "--ewc_config",
        type=Path,
        default=Path("configs/ewc_oneclass_config.yaml"),
        help="Config EWCOneClassDetector.",
    )
    parser.add_argument(
        "--hdc_config",
        type=Path,
        default=Path("configs/hdc_pronostia_by_condition_config.yaml"),
        help="Config HDCClassifier.",
    )
    parser.add_argument(
        "--tinyol_config",
        type=Path,
        default=Path("configs/tinyol_pronostia_by_condition_config.yaml"),
        help="Config TinyOLAnomalyDetector.",
    )
    args = parser.parse_args()

    for p, name in [
        (args.config, "--config"),
        (args.ewc_config, "--ewc_config"),
    ]:
        if not p.exists():
            print(f"❌ Config introuvable : {p} ({name})", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  RAM Profiling — model={args.model}  dataset={args.dataset}")
    print(f"{'=' * 60}")

    if args.model == "all":
        results = []
        results.append(_profile_hdc(args.hdc_config, args.dataset))
        results.append(_profile_tinyol_ae(args.tinyol_config, args.dataset))
        results.append(_profile_kmeans(args.config, args.dataset))
        results.append(_profile_mahalanobis(args.config, args.dataset))
        results.append(_profile_dbscan(args.config, args.dataset))
        results.append(_profile_ewc_oneclass(args.ewc_config, args.dataset))
        _print_comparison_table(results)
        return

    dispatch = {
        "ewc_oneclass": lambda: _profile_ewc_oneclass(args.ewc_config, args.dataset),
        "hdc": lambda: _profile_hdc(args.hdc_config, args.dataset),
        "tinyol_ae": lambda: _profile_tinyol_ae(args.tinyol_config, args.dataset),
        "kmeans": lambda: _profile_kmeans(args.config, args.dataset),
        "mahalanobis": lambda: _profile_mahalanobis(args.config, args.dataset),
        "dbscan": lambda: _profile_dbscan(args.config, args.dataset),
    }

    report = dispatch[args.model]()

    print(
        f"\n📋 Résumé — {report['model']}\n"
        f"   RAM statique              : {report['ram_static_bytes']/1024:.2f} Ko\n"
        f"   RAM peak inférence (PC)   : {report['ram_inf_peak_bytes']/1024:.2f} Ko\n"
        f"   Budget 64Ko (statique)    : {'✅' if report['within_budget_64ko_static'] else '❌'}\n"
        f"   Budget 64Ko (inférence)   : {'✅' if report['within_budget_64ko_inference'] else '❌'}"
    )


if __name__ == "__main__":
    main()
