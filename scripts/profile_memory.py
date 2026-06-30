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
import json
import sys
import time
import tracemalloc
from datetime import date
from pathlib import Path

import numpy as np
import yaml

# Assure que src/ est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.memory_profiler import full_memory_report
from src.models.ewc.ewc_oneclass import EWCOneClassDetector

BUDGET_BYTES: int = 65_536       # 64 Ko
BUDGET_256KO_BYTES: int = 262_144  # 256 Ko — NUCLEO-F439ZI (SRAM)

# ---------------------------------------------------------------------------
# Grille multi-dataset (S2404)
# ---------------------------------------------------------------------------

_MODELS   = ["ewc", "hdc", "tinyol", "mahalanobis"]
_DATASETS = ["monitoring", "pump", "cwru", "pronostia", "cmapss", "paderborn"]

# (model, dataset) → (ewc_config_path, hdc_config_path)
# ewc_config_path : utilisé par EWC, TinyOL, Mahalanobis (lit model.input_dim / backbone.input_dim)
# hdc_config_path : utilisé par HDC (lit data.n_features et hdc.D)
_PROFILE_CONFIG_MAP: dict[tuple[str, str], tuple[str | None, str | None]] = {
    ("ewc",         "monitoring"):  ("configs/ewc_config.yaml",                        None),
    ("ewc",         "pump"):        ("configs/ewc_pump_config.yaml",                   None),
    ("ewc",         "cwru"):        ("configs/cwru_by_fault_config.yaml",              None),
    ("ewc",         "pronostia"):   ("configs/ewc_pronostia_by_condition_config.yaml", None),
    ("ewc",         "cmapss"):      ("configs/cmapss_config.yaml",                     None),
    ("ewc",         "paderborn"):   ("configs/paderborn_config.yaml",                  None),
    ("hdc",         "monitoring"):  (None, "configs/hdc_config.yaml"),
    ("hdc",         "cwru"):        (None, "configs/cwru_by_fault_config.yaml"),
    ("hdc",         "pump"):        (None, "configs/hdc_pump_config.yaml"),
    ("hdc",         "cmapss"):      (None, "configs/cmapss_config.yaml"),
    ("hdc",         "paderborn"):   (None, "configs/paderborn_config.yaml"),
    ("tinyol",      "monitoring"):  ("configs/tinyol_monitoring_config.yaml",          None),
    ("tinyol",      "pump"):        ("configs/tinyol_config.yaml",                     None),
    ("tinyol",      "cwru"):        ("configs/cwru_by_fault_config.yaml",              None),
    ("mahalanobis", "monitoring"):  ("configs/ewc_config.yaml",                        None),
    ("mahalanobis", "pump"):        ("configs/ewc_pump_config.yaml",                   None),
    ("mahalanobis", "cwru"):        ("configs/cwru_by_fault_config.yaml",              None),
    ("mahalanobis", "pronostia"):   ("configs/pronostia_config.yaml",                  None),
    ("mahalanobis", "cmapss"):      ("configs/cmapss_config.yaml",                     None),
    ("mahalanobis", "paderborn"):   ("configs/paderborn_config.yaml",                  None),
}

_SKIP_COMBOS: dict[tuple[str, str], str] = {
    ("hdc",    "pronostia"):  "pas d'expérience HDC correspondante sur Pronostia",
    ("tinyol", "cmapss"):     "pas de loader temporel approprié pour CMAPSS",
    ("tinyol", "paderborn"):  "pas de loader temporel approprié pour Paderborn",
    ("tinyol", "pronostia"):  "pas d'expérience TinyOL correspondante sur Pronostia",
}


def _to_standard_entry(model: str, dataset: str, raw: dict) -> dict:
    """Normalise la sortie hétérogène des profilers vers le format S2404."""
    inf_bytes = int(raw.get("ram_inference_bytes") or raw.get("ram_inf_peak_bytes") or 0)
    upd_bytes = raw.get("ram_update_bytes")
    lat_ms    = raw.get("inference_latency_ms") or raw.get("inference_latency_ms_mean") or 0.0
    return {
        "model":                    model,
        "dataset":                  dataset,
        "inference_ram_peak_bytes": inf_bytes,
        "inference_ram_peak_kb":    round(inf_bytes / 1024, 2),
        "update_ram_peak_bytes":    int(upd_bytes) if upd_bytes is not None else None,
        "update_ram_peak_kb":       round(upd_bytes / 1024, 2) if upd_bytes is not None else None,
        "gap2_compliant":           inf_bytes < BUDGET_256KO_BYTES,
        "gap2_budget_bytes":        BUDGET_256KO_BYTES,
        "n_params":                 raw.get("n_params", 0),
        "inference_latency_ms_mean":round(float(lat_ms), 4),
        "n_latency_runs":           100,
    }


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

    data_cfg = cfg.get("data", {})
    input_dim: int = int(
        cfg.get("DATASETS", {}).get(dataset, {}).get("INPUT_DIM")
        or cfg.get("model", {}).get("input_dim")
        or data_cfg.get("n_features")
        or data_cfg.get("n_features_selected")
        or 13
    )
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
# Sprint 4 — Profiling modèles CL supervisés (EWCMlpClassifier, HDCClassifier, TinyOL+OtO)
# ---------------------------------------------------------------------------


def _profile_ewc_mlp(
    ewc_config_path: Path,
    n_runs_inf: int = 100,
    n_runs_upd: int = 50,
) -> dict:
    """
    Profil RAM EWCMlpClassifier (MLP supervisé binaire avec régularisation EWC).

    Mesures :
        - RAM statique (poids FP32) via estimate_ram_bytes()
        - RAM EWC state = poids × 3 (modèle + Fisher diagonal + θ*)
        - RAM peak inférence (tracemalloc, forward seul, batch=1)
        - RAM peak update (tracemalloc, forward + backward + SGD step, batch=32)
        - Latences inférence et update
    """
    import torch
    import torch.optim as optim

    from src.models.ewc.ewc_mlp import EWCMlpClassifier

    with open(ewc_config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    input_dim: int = int(model_cfg.get("input_dim", 4))
    hidden_dims: list[int] = list(model_cfg.get("hidden_dims", [32, 16]))
    ewc_lambda: float = float(cfg.get("ewc", {}).get("lambda", 1000.0))
    lr: float = float(cfg.get("training", {}).get("learning_rate", 0.01))

    print(f"\n🔍 Profiling EWCMlpClassifier — input_dim={input_dim}, hidden={hidden_dims}")

    model = EWCMlpClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
    model.eval()

    rng = np.random.default_rng(42)
    x_single = torch.from_numpy(rng.standard_normal((1, input_dim)).astype(np.float32))
    x_batch = torch.from_numpy(rng.standard_normal((32, input_dim)).astype(np.float32))
    y_batch = torch.from_numpy(rng.integers(0, 2, (32, 1)).astype(np.float32))

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")
    # Fisher diagonal + θ* = 2 copies supplémentaires des poids
    ram_ewc_state = n_params * 4 * 2

    # --- RAM inférence ---
    tracemalloc.start()
    with torch.no_grad():
        for _ in range(n_runs_inf):
            _ = model(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- RAM update ---
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    tracemalloc.start()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        loss = model.ewc_loss(x_batch, y_batch, None, None, ewc_lambda)
        loss.backward()
        optimizer.step()
    _, peak_upd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Latences ---
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs_inf):
            model(x_single)
        inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    model.train()
    t0 = time.perf_counter()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        loss = model.ewc_loss(x_batch, y_batch, None, None, ewc_lambda)
        loss.backward()
        optimizer.step()
    upd_ms = (time.perf_counter() - t0) / n_runs_upd * 1000

    _print_model_block("EWC MLP", input_dim, n_params, ram_static, peak_inf)
    print(f"   │  RAM update (tracemalloc) : {peak_upd:>8,} B  ({peak_upd/1024:.2f} Ko)")
    s_upd = "✅ DANS LE BUDGET 256Ko" if peak_upd < BUDGET_256KO_BYTES else "❌ DÉPASSE 256Ko"
    print(f"   │  NUCLEO-F439ZI 256Ko     : {s_upd}")

    return {
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_ewc_state_bytes": ram_ewc_state,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": peak_upd,
        "inference_latency_ms": round(inf_ms, 3),
        "update_latency_ms": round(upd_ms, 3),
        "within_budget_64ko": peak_upd < BUDGET_BYTES,
        "within_budget_256ko": peak_upd < BUDGET_256KO_BYTES,
    }


def _profile_hdc_supervised(
    hdc_config_path: Path,
    n_runs_inf: int = 100,
) -> dict:
    """
    Profil RAM HDCClassifier en mode supervisé binaire (faulty=0/1).

    HDC n'a pas de backpropagation — update = accumulation de prototypes (O(1) RAM).
    Seule la RAM d'inférence (encode + dot product) est mesurée.
    """
    from src.models.hdc.hdc_classifier import HDCClassifier

    with open(hdc_config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    n_features: int = int(
        data_cfg.get("n_features")
        or data_cfg.get("n_features_selected")
        or cfg.get("model", {}).get("input_dim")
        or 4
    )

    # Injecte les défauts HDC si la section est absente (configs non-HDC réutilisées)
    if "hdc" not in cfg:
        # base_vectors_path inexistant → génération on-the-fly dans HDCClassifier
        cfg["hdc"] = {
            "D": 1024,
            "n_levels": 10,
            "seed": 42,
            "base_vectors_path": f"_profiling_bv_{n_features}d_nonexistent.npz",
        }
    else:
        # Force génération on-the-fly si le fichier .npz ne correspond pas au bon n_features
        bv_path = cfg["hdc"].get("base_vectors_path", "")
        if bv_path:
            import numpy as _np
            try:
                _bv = _np.load(bv_path)
                # H_pos shape = [D, n_features] — vérifier la compatibilité
                if _bv.get("H_pos", _np.empty((0, 0))).shape[1] != n_features:
                    cfg["hdc"]["base_vectors_path"] = f"_profiling_bv_{n_features}d_nonexistent.npz"
            except Exception:
                cfg["hdc"]["base_vectors_path"] = f"_profiling_bv_{n_features}d_nonexistent.npz"

    cfg.setdefault("data", {}).setdefault("n_features", n_features)
    cfg.setdefault("data", {}).setdefault("n_classes", 2)
    # feature_bounds est requis par HDCClassifier pour quantifier les features.
    # Si absent, on injecte des bornes standardisées [-3σ, +3σ] (valeurs typiques post-normalisation).
    if not cfg.get("feature_bounds"):
        cfg["feature_bounds"] = {
            f"feat_{i}": [-3.0, 3.0] for i in range(n_features)
        }

    hdc_cfg = cfg["hdc"]
    D: int = int(hdc_cfg.get("D", 1024))

    print(f"\n🔍 Profiling HDCClassifier (supervisé) — D={D}, n_features={n_features}")

    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, 200).astype(np.int64)
    x_single = X_train[:1]

    model = HDCClassifier(cfg)
    model.update(X_train, y_train)  # Construit les prototypes

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")

    # --- RAM inférence ---
    tracemalloc.start()
    for _ in range(n_runs_inf):
        _ = model.predict(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Latence inférence ---
    t0 = time.perf_counter()
    for _ in range(n_runs_inf):
        model.predict(x_single)
    inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    _print_model_block("HDC supervisé", n_features, n_params, ram_static, peak_inf)

    return {
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": None,  # HDC : pas de backprop
        "inference_latency_ms": round(inf_ms, 3),
        "update_latency_ms": None,
        "within_budget_64ko": peak_inf < BUDGET_BYTES,
        "within_budget_256ko": peak_inf < BUDGET_256KO_BYTES,
    }


def _profile_tinyol_cl(
    tinyol_config_path: Path,
    n_runs_inf: int = 100,
    n_runs_upd: int = 50,
) -> dict:
    """
    Profil RAM TinyOL (backbone gelé + tête OtO) en mode CL supervisé.

    Inférence : encode(x) → z [8D] + MSE → features [9D] → OtO → prob
    Update     : SGD sur OtO uniquement (backbone gelé)
    """
    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    from src.models.tinyol.autoencoder import TinyOLAutoencoder
    from src.models.tinyol.oto_head import OtOHead

    with open(tinyol_config_path) as f:
        cfg = yaml.safe_load(f)

    bb_cfg = cfg.get("backbone", {})
    oto_cfg = cfg.get("oto_head", {})
    input_dim: int = int(bb_cfg.get("input_dim", 25))
    encoder_dims: tuple = tuple(bb_cfg.get("encoder_dims", [32, 16, 8]))
    decoder_dims: tuple = tuple(bb_cfg.get("decoder_dims", [16, 32, 25]))
    oto_input_dim: int = int(oto_cfg.get("input_dim", 9))
    lr: float = float(oto_cfg.get("learning_rate", 0.01))

    print(f"\n🔍 Profiling TinyOL CL — input_dim={input_dim}, encoder={encoder_dims}, OtO={oto_input_dim}")

    autoencoder = TinyOLAutoencoder(input_dim=input_dim, encoder_dims=encoder_dims, decoder_dims=decoder_dims)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    oto = OtOHead(input_dim=oto_input_dim)

    rng = np.random.default_rng(42)
    x_single = torch.from_numpy(rng.standard_normal((1, input_dim)).astype(np.float32))
    x_batch = torch.from_numpy(rng.standard_normal((16, input_dim)).astype(np.float32))
    y_batch = torch.from_numpy(rng.integers(0, 2, (16, 1)).astype(np.float32))

    def _inference(x: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            z = autoencoder.encode(x)
            x_hat = autoencoder.decode(z)
            mse = ((x_hat - x) ** 2).mean(dim=1, keepdim=True)
            feats = torch.cat([z, mse], dim=1)
        return oto(feats)

    n_params_ae = sum(p.numel() for p in autoencoder.parameters())
    n_params_oto = oto.n_params()
    ram_static_ae = sum(p.numel() for p in autoencoder.parameters()) * 4
    ram_static_oto = n_params_oto * 4

    # --- RAM inférence ---
    tracemalloc.start()
    for _ in range(n_runs_inf):
        _ = _inference(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- RAM update (OtO SGD seulement) ---
    optimizer = optim.SGD(oto.parameters(), lr=lr, momentum=0.0)
    oto.train()
    tracemalloc.start()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        with torch.no_grad():
            z = autoencoder.encode(x_batch)
            x_hat = autoencoder.decode(z)
            mse = ((x_hat - x_batch) ** 2).mean(dim=1, keepdim=True)
            feats = torch.cat([z, mse], dim=1)
        pred = oto(feats)
        loss = F.binary_cross_entropy(pred, y_batch)
        loss.backward()
        optimizer.step()
    _, peak_upd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Latences ---
    t0 = time.perf_counter()
    for _ in range(n_runs_inf):
        _inference(x_single)
    inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        with torch.no_grad():
            z = autoencoder.encode(x_batch)
            x_hat = autoencoder.decode(z)
            mse = ((x_hat - x_batch) ** 2).mean(dim=1, keepdim=True)
            feats = torch.cat([z, mse], dim=1)
        pred = oto(feats)
        loss = F.binary_cross_entropy(pred, y_batch)
        loss.backward()
        optimizer.step()
    upd_ms = (time.perf_counter() - t0) / n_runs_upd * 1000

    _print_model_block("TinyOL FP32", input_dim, n_params_ae + n_params_oto, ram_static_ae, peak_inf)
    print(f"   │  RAM update OtO (tracemalloc) : {peak_upd:>8,} B  ({peak_upd/1024:.2f} Ko)")

    return {
        "n_params": n_params_ae + n_params_oto,
        "n_params_encoder": n_params_ae,
        "n_params_oto": n_params_oto,
        "ram_static_bytes": ram_static_ae + ram_static_oto,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": peak_upd,
        "inference_latency_ms": round(inf_ms, 3),
        "update_latency_ms": round(upd_ms, 3),
        "within_budget_64ko": peak_upd < BUDGET_BYTES,
        "within_budget_256ko": peak_upd < BUDGET_256KO_BYTES,
    }


# ---------------------------------------------------------------------------
# Sprint 25 — Profiling nouveaux modèles (EWCMlpRegressor, EWCMlpMulticlass, HDCRegressor)
# ---------------------------------------------------------------------------


def _profile_ewc_regression(
    config_path: Path,
    n_runs_inf: int = 100,
    n_runs_upd: int = 50,
) -> dict:
    """Profil RAM EWCMlpRegressor (régression RUL sur CMAPSS)."""
    import torch
    import torch.optim as optim

    from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dim: int = int(cfg.get("INPUT_DIM", 5))
    hidden_dims: list[int] = list(cfg.get("HIDDEN_DIMS", [32, 16]))
    ewc_lambda: float = float(cfg.get("EWC_LAMBDA", 400.0))
    lr: float = float(cfg.get("EWC_LR", 0.01))

    print(f"\n🔍 Profiling EWCMlpRegressor — input_dim={input_dim}, hidden={hidden_dims}")

    model = EWCMlpRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0, ewc_lambda=ewc_lambda)
    model.eval()

    rng = np.random.default_rng(42)
    x_single = torch.from_numpy(rng.standard_normal((1, input_dim)).astype(np.float32))
    x_batch = torch.from_numpy(rng.standard_normal((32, input_dim)).astype(np.float32))
    y_batch = torch.from_numpy(rng.standard_normal(32).astype(np.float32))

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")

    # RAM inférence
    tracemalloc.start()
    with torch.no_grad():
        for _ in range(n_runs_inf):
            _ = model(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # RAM update
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    tracemalloc.start()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        loss = criterion(model(x_batch).squeeze(), y_batch) + model.ewc_penalty()
        loss.backward()
        optimizer.step()
    _, peak_upd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Latence inférence
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs_inf):
            model(x_single)
        inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    _print_model_block("EWC Regression", input_dim, n_params, ram_static, peak_inf)
    print(f"   │  RAM update (tracemalloc) : {peak_upd:>8,} B  ({peak_upd/1024:.2f} Ko)")
    s = "✅" if peak_upd < BUDGET_256KO_BYTES else "❌"
    print(f"   │  NUCLEO-F439ZI 256Ko     : {s}")

    return {
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": peak_upd,
        "inference_latency_ms": round(inf_ms, 3),
        "within_budget_256ko": peak_upd < BUDGET_256KO_BYTES,
    }


def _profile_ewc_multiclass(
    config_path: Path,
    n_runs_inf: int = 100,
    n_runs_upd: int = 50,
) -> dict:
    """Profil RAM EWCMlpMulticlass (classification 10 classes CWRU)."""
    import torch
    import torch.optim as optim

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    input_dim: int = int(cfg.get("INPUT_DIM", 9))
    n_classes: int = int(cfg.get("N_CLASSES", 10))
    hidden_dims: list[int] = list(cfg.get("HIDDEN_DIMS", [32, 16]))
    ewc_lambda: float = float(cfg.get("EWC_LAMBDA", 400.0))
    lr: float = float(cfg.get("EWC_LR", 0.01))

    print(f"\n🔍 Profiling EWCMlpMulticlass — input_dim={input_dim}, n_classes={n_classes}, hidden={hidden_dims}")

    model = EWCMlpMulticlass(input_dim=input_dim, n_classes=n_classes, hidden_dims=hidden_dims, dropout=0.0, ewc_lambda=ewc_lambda)
    model.eval()

    rng = np.random.default_rng(42)
    x_single = torch.from_numpy(rng.standard_normal((1, input_dim)).astype(np.float32))
    x_batch = torch.from_numpy(rng.standard_normal((32, input_dim)).astype(np.float32))
    y_batch = torch.from_numpy(rng.integers(0, n_classes, 32).astype(np.int64))

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes("fp32")

    # RAM inférence
    tracemalloc.start()
    with torch.no_grad():
        for _ in range(n_runs_inf):
            _ = model(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # RAM update
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    tracemalloc.start()
    for _ in range(n_runs_upd):
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch) + model.ewc_penalty()
        loss.backward()
        optimizer.step()
    _, peak_upd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Latence inférence
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs_inf):
            model(x_single)
        inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    _print_model_block("EWC Multiclass", input_dim, n_params, ram_static, peak_inf)
    print(f"   │  RAM update (tracemalloc) : {peak_upd:>8,} B  ({peak_upd/1024:.2f} Ko)")
    s = "✅" if peak_upd < BUDGET_256KO_BYTES else "❌"
    print(f"   │  NUCLEO-F439ZI 256Ko     : {s}")

    return {
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": peak_upd,
        "inference_latency_ms": round(inf_ms, 3),
        "within_budget_256ko": peak_upd < BUDGET_256KO_BYTES,
    }


def _profile_hdc_regressor(
    config_path: Path,
    n_runs_inf: int = 100,
) -> dict:
    """Profil RAM HDCRegressor (régression RUL sur CMAPSS via HDC)."""
    from src.models.hdc.hdc_regressor import HDCRegressor

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_features: int = int(cfg.get("INPUT_DIM", 5))
    D: int = int(cfg.get("D", 1024))
    n_levels: int = int(cfg.get("N_LEVELS", 10))
    lr: float = float(cfg.get("HDC_LR", cfg.get("EWC_LR", 0.01)))

    print(f"\n🔍 Profiling HDCRegressor — D={D}, n_levels={n_levels}, n_features={n_features}")

    rng = np.random.default_rng(42)
    x_batch = rng.standard_normal((32, n_features)).astype(np.float32)
    x_single = x_batch[:1]

    model = HDCRegressor(D=D, n_levels=n_levels, n_features=n_features, lr=lr, seed=42)
    model.set_feature_bounds(x_batch)

    n_params = model.count_parameters()
    ram_static = model.estimate_ram_bytes()

    # RAM inférence
    tracemalloc.start()
    for _ in range(n_runs_inf):
        _ = model.predict(x_single)
    _, peak_inf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Latence inférence
    t0 = time.perf_counter()
    for _ in range(n_runs_inf):
        model.predict(x_single)
    inf_ms = (time.perf_counter() - t0) / n_runs_inf * 1000

    _print_model_block("HDC Regressor", n_features, n_params, ram_static, peak_inf)
    s = "✅" if peak_inf < BUDGET_256KO_BYTES else "❌"
    print(f"   │  NUCLEO-F439ZI 256Ko     : {s}")

    return {
        "n_params": n_params,
        "ram_static_bytes": ram_static,
        "ram_inference_bytes": peak_inf,
        "ram_update_bytes": None,
        "inference_latency_ms": round(inf_ms, 3),
        "within_budget_256ko": peak_inf < BUDGET_256KO_BYTES,
    }


def _run_sprint25_profiling(
    rul_config_path: Path,
    multiclass_config_path: Path,
    output_path: Path | None = None,
) -> dict:
    """
    Profiling complet Sprint 25 : EWCMlpRegressor + EWCMlpMulticlass + HDCRegressor.

    Produit experiments/exp_S25_05/results.json.
    """
    from datetime import date as _date

    print(f"\n{'=' * 72}")
    print("  Profiling RAM Sprint 25 — EWC Regression + EWC Multiclass + HDC Regressor")
    print(f"{'=' * 72}")

    ewc_reg = _profile_ewc_regression(rul_config_path)
    ewc_mc = _profile_ewc_multiclass(multiclass_config_path)
    hdc_reg = _profile_hdc_regressor(rul_config_path)

    # Référence binaire depuis exp_S23_01 si disponible
    ref_ram = None
    ref_path = Path("experiments/exp_S23_01/results.json")
    if ref_path.exists():
        import json as _json
        ref_data = _json.load(open(ref_path))
        ref_ram = ref_data.get("ram_peak_bytes")

    # Critère spec : ewc_regression ≤ ewc_binary × 1.20
    ewc_reg_ram = ewc_reg["ram_update_bytes"]
    budget_ok = (ref_ram is None) or (ewc_reg_ram is None) or (ewc_reg_ram <= ref_ram * 1.20)

    results = {
        "exp_id": "exp_S25_05",
        "sprint": 25,
        "generated_at": str(_date.today()),
        "models": {
            "ewc_regression": {
                "dataset": "cmapss",
                "input_dim": 5,
                "n_params": ewc_reg["n_params"],
                "ram_peak_bytes": ewc_reg["ram_update_bytes"],
                "inference_latency_ms": ewc_reg["inference_latency_ms"],
                "gap2_compliant": ewc_reg["within_budget_256ko"],
            },
            "ewc_multiclass": {
                "dataset": "cwru",
                "input_dim": 9,
                "n_classes": 10,
                "n_params": ewc_mc["n_params"],
                "ram_peak_bytes": ewc_mc["ram_update_bytes"],
                "inference_latency_ms": ewc_mc["inference_latency_ms"],
                "gap2_compliant": ewc_mc["within_budget_256ko"],
            },
            "hdc_regressor": {
                "dataset": "cmapss",
                "D": 1024,
                "n_features": 5,
                "n_params": hdc_reg["n_params"],
                "ram_peak_bytes": hdc_reg["ram_inference_bytes"],
                "inference_latency_ms": hdc_reg["inference_latency_ms"],
                "gap2_compliant": hdc_reg["within_budget_256ko"],
            },
        },
        "reference_ewc_binary": {
            "exp_id": "exp_S23_01",
            "ram_peak_bytes": ref_ram,
            "note": "référence mode binaire pour comparaison",
        },
        "criteria": {
            "ewc_regression_vs_binary_20pct": budget_ok,
            "all_under_256ko": all([
                ewc_reg["within_budget_256ko"],
                ewc_mc["within_budget_256ko"],
                hdc_reg["within_budget_256ko"],
            ]),
        },
    }

    print(f"\n{'=' * 72}")
    print("  Résultats Sprint 25")
    print(f"{'=' * 72}")
    for name, d in results["models"].items():
        ram = d["ram_peak_bytes"] or 0
        ok = "✅" if d["gap2_compliant"] else "❌"
        print(f"  {name:<20} RAM={ram:>8,} B ({ram/1024:.1f} Ko)  lat={d['inference_latency_ms']:.3f}ms  {ok}")
    if ref_ram:
        print(f"  Référence binaire    RAM={ref_ram:>8,} B ({ref_ram/1024:.1f} Ko)")
    print(f"  Critère 20% overhead : {'✅' if budget_ok else '❌'}")

    if output_path is not None:
        import json as _json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            _json.dump(results, fh, indent=2, default=str)
        # Config snapshot
        import yaml as _yaml
        snap_path = output_path.parent / "config_snapshot.yaml"
        with open(snap_path, "w") as fh:
            _yaml.dump({
                "exp_id": "exp_S25_05",
                "sprint": 25,
                "generated_at": str(_date.today()),
                "rul_config": str(rul_config_path),
                "multiclass_config": str(multiclass_config_path),
                "budget_256ko_bytes": BUDGET_256KO_BYTES,
            }, fh, default_flow_style=False, allow_unicode=True)
        print(f"\nRapport → {output_path}")

    return results


def _print_cl_summary_table(models_data: dict) -> None:
    """Tableau comparatif ASCII pour les 3 modèles CL supervisés Sprint 4."""
    BUDGET = BUDGET_256KO_BYTES
    print("\n" + "=" * 80)
    print("  TABLEAU COMPARATIF RAM — modèles CL supervisés Sprint 4 (NUCLEO-F439ZI 256Ko)")
    print("=" * 80)
    header = (
        f"{'Modèle':<16} {'RAM statique':>13} {'RAM inférence':>14} "
        f"{'RAM update':>12} {'Lat. inf':>9} {'≤ 256Ko':>8}"
    )
    print(header)
    print("-" * 80)
    for key, d in models_data.items():
        if d is None:
            print(f"{key:<16} {'—':>13} {'—':>14} {'—':>12} {'—':>9}  N/A")
            continue
        static = d.get("ram_static_bytes") or 0
        inf_pk = d.get("ram_inference_bytes") or 0
        upd_pk = d.get("ram_update_bytes")
        inf_ms = d.get("inference_latency_ms", 0)
        budget_key = upd_pk if upd_pk is not None else inf_pk
        ok = "✅" if budget_key < BUDGET else "❌"
        upd_str = f"{upd_pk/1024:.2f} Ko" if upd_pk is not None else "—"
        print(
            f"{key:<16} {static/1024:>11.2f}Ko {inf_pk/1024:>12.2f}Ko "
            f"{upd_str:>12} {inf_ms:>7.3f}ms  {ok}"
        )
    print("=" * 80)
    print("  Mesures PC (tracemalloc) — MCU validé en Phase 2 (portage NUCLEO-F439ZI).\n")


# ---------------------------------------------------------------------------
# Profiling systématique (S2404)
# ---------------------------------------------------------------------------

def run_all_profiles(output_path: Path | None = None) -> dict:
    """
    Profil RAM systématique de tous les combos _PROFILE_CONFIG_MAP.

    Génère un rapport unifié au format S2404 (20 entrées : 4 modèles × 5 datasets + skips).

    Parameters
    ----------
    output_path : Path, optional
        Chemin JSON de sortie. Pas de sauvegarde si None.

    Returns
    -------
    dict : résultats bruts (clé = "model_dataset").
    """
    import yaml as _yaml

    results: dict[str, dict] = {}

    print(f"\n{'=' * 72}")
    print("  Profiling RAM systématique (S2404) — 4 modèles × 6 datasets")
    print(f"{'=' * 72}")

    for model_id in _MODELS:
        for dataset_id in _DATASETS:
            key = (model_id, dataset_id)
            result_key = f"{model_id}_{dataset_id}"

            # Combos skippés
            if key in _SKIP_COMBOS:
                results[result_key] = {
                    "model":   model_id,
                    "dataset": dataset_id,
                    "status":  "skipped",
                    "reason":  _SKIP_COMBOS[key],
                }
                print(f"\n  ~ {model_id} × {dataset_id} → skip : {_SKIP_COMBOS[key]}")
                continue

            # Combos non définis (hors matrice supportée)
            if key not in _PROFILE_CONFIG_MAP:
                results[result_key] = {
                    "model":   model_id,
                    "dataset": dataset_id,
                    "status":  "skipped",
                    "reason":  "combo non défini dans _PROFILE_CONFIG_MAP",
                }
                continue

            ewc_cfg_path, hdc_cfg_path = _PROFILE_CONFIG_MAP[key]

            print(f"\n--- {model_id} × {dataset_id} ---")
            try:
                if model_id == "ewc":
                    raw = _profile_ewc_mlp(Path(ewc_cfg_path))
                elif model_id == "hdc":
                    raw = _profile_hdc_supervised(Path(hdc_cfg_path))
                elif model_id == "tinyol":
                    raw = _profile_tinyol_cl(Path(ewc_cfg_path))
                else:  # mahalanobis
                    # Lecture input_dim depuis la config EWC correspondante
                    with open(ewc_cfg_path) as fh:
                        cfg_yaml = _yaml.safe_load(fh)
                    data_cfg = cfg_yaml.get("data", {})
                    _input_dim = (
                        cfg_yaml.get("DATASETS", {}).get(dataset_id, {}).get("INPUT_DIM")
                        or cfg_yaml.get("model", {}).get("input_dim")
                        or data_cfg.get("n_features")
                        or data_cfg.get("n_features_selected")
                        or 4
                    )
                    raw = _profile_mahalanobis(Path(ewc_cfg_path), dataset_id)

                entry = _to_standard_entry(model_id, dataset_id, raw)
                results[result_key] = entry

                gap_ok = "✅" if entry["gap2_compliant"] else "❌ FIXME(gap2)"
                print(
                    f"  → inf RAM: {entry['inference_ram_peak_kb']:.2f} Ko "
                    f"| upd RAM: {entry.get('update_ram_peak_kb') or '—'} Ko "
                    f"| lat: {entry['inference_latency_ms_mean']:.3f} ms "
                    f"| {gap_ok}"
                )

            except Exception as exc:
                print(f"  ✗ Erreur : {exc}")
                results[result_key] = {
                    "model":   model_id,
                    "dataset": dataset_id,
                    "status":  "error",
                    "reason":  str(exc),
                }

    # Vérification Gap 2
    compliant = sum(1 for r in results.values() if r.get("gap2_compliant") is True)
    violations = [k for k, r in results.items() if r.get("gap2_compliant") is False]

    print(f"\n{'=' * 72}")
    print(f"  Gap 2 (256 Ko) : {compliant} combos conformes")
    if violations:
        print(f"  VIOLATIONS : {violations}")
        for v in violations:
            print(f"    FIXME(gap2) — {v}")
    else:
        print("  Tous les combos mesurés sont conformes ✅")
    print(f"{'=' * 72}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "generated_at": str(date.today()),
                    "sprint":       24,
                    "budget_256ko_bytes": BUDGET_256KO_BYTES,
                    "entries":      list(results.values()),
                },
                fh, indent=2, default=str,
            )
        print(f"\nRapport → {output_path}")

    return results


def _create_exp_s24_03(report_path: Path) -> None:
    """Crée experiments/exp_S24_03/ avec config_snapshot.yaml et results.json."""
    import yaml as _yaml

    exp_dir = Path("experiments/exp_S24_03")
    exp_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "exp_id":        "exp_S24_03",
        "description":   "Profiling RAM systématique — 4 modèles × 6 datasets (S2404)",
        "generated_at":  str(date.today()),
        "models":        _MODELS,
        "datasets":      _DATASETS,
        "budget_256ko":  BUDGET_256KO_BYTES,
        "report_path":   str(report_path),
    }
    with open(exp_dir / "config_snapshot.yaml", "w") as fh:
        _yaml.dump(snapshot, fh, default_flow_style=False, allow_unicode=True)

    with open(exp_dir / "results.json", "w") as fh:
        json.dump({"report_path": str(report_path)}, fh, indent=2)

    print(f"exp_S24_03 → {exp_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _run_autonomy_profiling(output_path: Path) -> None:
    """RAM profiling du module autonomy.py (S3307) + génération de autonomy.json.

    Règle CLAUDE.md : tout nouveau module mesuré doit être RAM-profilé. Le module
    autonomy.py n'est pas un nn.Module ; on profile son *calcul* via tracemalloc
    (pattern memory_profiler.py) plutôt qu'un forward torch.

    autonomy.json : pour chaque couple modèle×encodage de la campagne S3306, on
    tente de dériver I_moy + le balayage de capacités. Si les µJ/phase valent
    encore le placeholder « à mesurer » (LPM01A non exécuté), I_moy et l'autonomie
    portent « à mesurer » — mais la STRUCTURE de balayage (clés capacité) est réelle.
    """
    from src.evaluation import autonomy as autonomy_mod

    A_MESURER = autonomy_mod.A_MESURER
    energy_dir = Path("experiments/exp_S33_energy")
    capacites = autonomy_mod.load_battery_capacities("configs/hw_profile_f439zi.yaml")

    # --- RAM profiling du calcul (tracemalloc, pattern memory_profiler) ---
    sample_phases = {"startup": 50.0, "acquisition": 10.0, "inference": 100.0, "idle": 5.0}
    sample_durations = {"startup": 0.001, "acquisition": 0.0002, "inference": 0.0005, "idle": 0.01}
    tracemalloc.start()
    i_demo = autonomy_mod.average_current_ma(sample_phases, sample_durations)
    _ = autonomy_mod.sweep_capacities(i_demo, capacites)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Dérivation par modèle × encodage (réelle si mesurée, sinon placeholder) ---
    per_model: dict = {}
    for model in ("ewc", "hdc", "tinyol", "maha"):
        for enc in ("fp32", "int8"):
            jp = energy_dir / f"{model}_{enc}.json"
            if not jp.is_file():
                continue
            data = json.loads(jp.read_text(encoding="utf-8"))
            phases_uj = data.get("phases_uj", {})
            durations = data.get("phase_durations_s")  # absent tant que non mesuré
            measured = all(
                isinstance(phases_uj.get(p), (int, float)) for p in phases_uj
            ) and bool(durations)
            if measured:
                i_moy = autonomy_mod.average_current_ma(phases_uj, durations)
                sweep = {str(c): h for c, h in
                         autonomy_mod.sweep_capacities(i_moy, capacites).items()}
            else:
                i_moy = A_MESURER
                sweep = {str(c): A_MESURER for c in capacites}
            per_model[f"{model}_{enc}"] = {"i_moy_ma": i_moy, "autonomy_h_by_mah": sweep}

    report = {
        "description": "Autonomie estimée (S3307) — I_moy + balayage capacités batterie.",
        "capacites_mah": capacites,
        "tension_v": 3.3,
        "per_model": per_model,
        "ram_profiling": {
            "note": "RAM peak du calcul autonomy.py (tracemalloc PC, proxy).",
            "ram_peak_bytes": int(peak),
            "within_budget_64ko": bool(peak < 65_536),
        },
        "timestamp": date.today().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🔋 Autonomie — RAM peak calcul : {peak:,} B ({peak/1024:.2f} Ko)")
    print(f"   {len(per_model)} couples modèle×encodage → {output_path}")


def _run_condition_ram_profiling(
    condition: str,
    model: str,
    dataset: str,
    seed: int = 42,
) -> dict:
    """RAM profiling par condition de features (Sprint 35 / S3505).

    Mesure réelle (tracemalloc, pas d'estimation analytique) en réutilisant la
    boucle ``train_and_evaluate`` du moteur de sweep S3503 — la même qui produit
    ``ram_peak_bytes`` dans ``results.json``. Écrit ``ram.json`` à côté.

    Parameters
    ----------
    condition : {'5feat', 'all', 'best'}
    model     : {'mahalanobis', 'ewc', 'tinyol', 'hdc'}
    dataset   : {'cwru', 'monitoring', 'pronostia', 'cmapss', 'paderborn'}

    Returns
    -------
    dict : contenu écrit dans ram.json.
    """
    from scripts.run_feature_condition_sweep import resolve_feature_indices
    from src.evaluation.feature_conditions import (
        load_native_task_arrays,
        train_and_evaluate,
    )

    idx, note = resolve_feature_indices(condition, model, dataset)
    print(
        f"\n🔍 RAM profiling (S3505) — condition={condition} model={model} "
        f"dataset={dataset} | {len(idx)} features | {note}"
    )

    tasks = load_native_task_arrays(dataset, seed=seed)
    res = train_and_evaluate(model, tasks, idx, seed=seed)

    ram = {
        "exp_id": f"exp_S35_PC_{condition}_{model}_{dataset}",
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "platform": "pc",
        "sprint": 35,
        "n_features": int(res["n_features"]),
        "ram_peak_bytes": int(res["ram_peak_bytes"]),
        "n_params": int(res["n_params"]),
        "measure": "tracemalloc",  # mesure réelle (CLAUDE.md), pas analytique
    }

    out_dir = Path("experiments") / ram["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ram.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ram, f, indent=2)

    print(
        f"   ✅ ram_peak_bytes={ram['ram_peak_bytes']:,} B "
        f"({ram['ram_peak_bytes']/1024:.2f} Ko) | n_params={ram['n_params']:,} "
        f"→ {out_path}"
    )
    return ram


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profiling RAM des modèles CL embarqués (Gap 2)"
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        choices=[
            "ewc_oneclass", "hdc", "tinyol_ae", "kmeans", "mahalanobis", "dbscan",
            "ewc", "tinyol",  # noms du moteur S3503 (avec --condition, Sprint 35)
            "all",       # anomaly detection models (Phase 2 sprint)
            "cl_all",    # supervised CL models Sprint 4 (EWC MLP, HDC, TinyOL+OtO)
            "sprint25",  # Sprint 25 nouveaux modèles (EWCMlpRegressor, EWCMlpMulticlass, HDCRegressor)
            "autonomy",  # Sprint 33 — RAM profiling du module autonomy.py + autonomy.json
        ],
        help="Modèle à profiler. 'sprint25' → EWCMlpRegressor + EWCMlpMulticlass + HDCRegressor. "
             "Avec --condition (S3505) : ewc/hdc/tinyol/mahalanobis.",
    )
    parser.add_argument(
        "--condition",
        choices=["5feat", "all", "best"],
        default=None,
        help="Sprint 35 (S3505) : RAM profiling par condition de features → "
             "experiments/exp_S35_PC_{condition}_{model}_{dataset}/ram.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed de reproductibilité (S3505).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profiling systématique de tous les combos (S2404b) — génère sprint24_memory_report.json.",
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
        help="Config EWCOneClassDetector (ou EWCMlpClassifier pour --model cl_all).",
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
        help="Config TinyOL (anomaly detector ou backbone CL selon --model).",
    )
    parser.add_argument(
        "--multiclass_config",
        type=Path,
        default=Path("configs/cwru_multiclass_config.yaml"),
        help="Config EWCMlpMulticlass (pour --model sprint25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin JSON de sortie du rapport (ex. experiments/sprint24_memory_report.json).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Mode --condition : RAM profiling par condition de features (S3505)
    # ------------------------------------------------------------------
    if args.condition is not None:
        if args.model is None:
            print("Erreur : --model requis avec --condition "
                  "(ewc/hdc/tinyol/mahalanobis).", file=sys.stderr)
            sys.exit(1)
        from src.evaluation.feature_conditions import DATASETS as _DS
        from src.evaluation.feature_conditions import MODELS as _MD
        if args.model not in _MD:
            print(f"Erreur : --model {args.model} non supporté avec --condition. "
                  f"Attendu : {_MD}.", file=sys.stderr)
            sys.exit(1)
        if args.dataset not in _DS:
            print(f"Erreur : --dataset {args.dataset} inconnu. Attendu : {_DS}.",
                  file=sys.stderr)
            sys.exit(1)
        _run_condition_ram_profiling(args.condition, args.model, args.dataset, args.seed)
        return

    # ------------------------------------------------------------------
    # Mode --all : profiling systématique (S2404b)
    # ------------------------------------------------------------------
    if args.all:
        out = args.output or Path("experiments/sprint24_memory_report.json")
        run_all_profiles(out)
        _create_exp_s24_03(out)
        return

    if args.model is None:
        print("Erreur : --model requis (ou utiliser --all pour le mode systématique).", file=sys.stderr)
        sys.exit(1)

    # Validation configs requises
    if args.model == "cl_all":
        required = [
            (args.ewc_config, "--ewc_config"),
            (args.hdc_config, "--hdc_config"),
            (args.tinyol_config, "--tinyol_config"),
        ]
    else:
        required = [(args.config, "--config"), (args.ewc_config, "--ewc_config")]

    for p, name in required:
        if not p.exists():
            print(f"❌ Config introuvable : {p} ({name})", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  RAM Profiling — model={args.model}  dataset={args.dataset}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # cl_all — modèles CL supervisés Sprint 4
    # ------------------------------------------------------------------
    if args.model == "cl_all":
        ewc_data = _profile_ewc_mlp(args.ewc_config)
        hdc_data = _profile_hdc_supervised(args.hdc_config)
        tinyol_data = _profile_tinyol_cl(args.tinyol_config)

        models_data = {
            "ewc": ewc_data,
            "hdc": hdc_data,
            "tinyol_fp32": tinyol_data,
            "tinyol_uint8": None,  # exp_004 non disponible
        }

        _print_cl_summary_table(models_data)

        # --- Gap 2 summary ---
        all_within = all(
            (d.get("within_budget_256ko", False) if d else False)
            for d in models_data.values()
            if d is not None
        )
        budget_values = {
            k: (d.get("ram_update_bytes") or d.get("ram_inference_bytes") or 0)
            for k, d in models_data.items()
            if d is not None
        }
        tightest_model = max(budget_values, key=budget_values.get) if budget_values else None
        tightest_val = budget_values.get(tightest_model, 0) if tightest_model else 0
        margin_pct = (1 - tightest_val / BUDGET_256KO_BYTES) * 100 if tightest_val else None

        report = {
            "generated": str(date.today()),
            "budget_64ko_bytes": BUDGET_BYTES,
            "budget_256ko_bytes": BUDGET_256KO_BYTES,
            "models": models_data,
            "gap2_summary": {
                "all_within_256ko": all_within,
                "tightest_model": tightest_model,
                "margin_256ko_percent": round(margin_pct, 1) if margin_pct else None,
            },
        }

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📋 Rapport JSON → {args.output}")

        print(
            f"\n📋 Gap 2 — NUCLEO-F439ZI 256Ko :\n"
            f"   Tous dans le budget : {'✅' if all_within else '❌'}\n"
            f"   Modèle le plus serré : {tightest_model} "
            f"({tightest_val/1024:.1f} Ko / {BUDGET_256KO_BYTES/1024:.0f} Ko)\n"
            f"   Marge restante : {margin_pct:.1f}%" if margin_pct else ""
        )
        return

    # ------------------------------------------------------------------
    # sprint25 — nouveaux modèles Sprint 25
    # ------------------------------------------------------------------
    if args.model == "sprint25":
        multiclass_config = getattr(args, "multiclass_config", None) or Path("configs/cwru_multiclass_config.yaml")
        out = args.output or Path("experiments/exp_S25_05/results.json")
        _run_sprint25_profiling(
            rul_config_path=args.config,
            multiclass_config_path=multiclass_config,
            output_path=out,
        )
        return

    if args.model == "autonomy":
        out = args.output or Path("experiments/exp_S33_energy/autonomy.json")
        _run_autonomy_profiling(out)
        return

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
