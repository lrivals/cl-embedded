#!/usr/bin/env python3
"""run_s46_quant_moment.py — Harnais unifié des trois moments de quantification (S4602).

Trois moments, un seul harnais. Pour un couple (modèle, dataset) et un `after_scheme`, ce
script évalue les quatre variantes d'un même modèle :

    - fp32   : entraîne FP32, évalue (référence).
    - before : entraîne QAT (fake-quant dans la boucle), évalue AVEC fake-quant à
               l'inférence — **borne haute** (la carte n'exécute jamais de fake-quant).
    - after  : entraîne FP32, EXTRAIT les poids figés, les passe dans le noyau PTQ
               `int8_c_emulation` (calibré selon `after_scheme`).
    - both   : entraîne QAT, EXTRAIT les poids APPRIS sous fake-quant, les passe dans le
               MÊME noyau PTQ — **fidèle au déploiement** (noyau entier + poids appris
               quantif-conscient). C'est le seul maillon qui n'existait nulle part avant.

Taxonomie : `docs/context/quantization_moments.md` (S4601). Axe orthogonal (format) :
`docs/context/quantization_strategies.md` (S4202).

Réutilise (source unique, aucune duplication) :
    - scripts/benchmark_int8_fp32.py : `EWCAdapter` (câblage FP32/QAT réel), helpers AUROC.
    - src/utils/int8_c_emulation.py  : `EWCHeadWeights.from_state_dict`, `forward_quant`,
                                       presets `QuantConfig`.
    - scripts/run_s39_quant_sweep.py : `SCHEME_BITS`/`SCHEME_WEIGHT_BYTES`/`_proxies`
                                       (comptage RAM/BOPs/latence-proxy).

100 % PC (l'émulateur reproduit le chemin C bit-à-bit). La validation board du chemin
`both` est différée (S4608). Aucune valeur écrite à la main : chaque cellule sort d'une
exécution ; les cellules non mesurées restent `null` (règle « aucun chiffre inventé »).

Usage :
    python scripts/run_s46_quant_moment.py --model ewc --dataset monitoring \
        --moment all --after-scheme per_tensor_calib \
        --config configs/quant_moment/ewc_monitoring.yaml \
        --output experiments/exp_S46_ewc/monitoring_all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.benchmark_int8_fp32 import (  # noqa: E402
    EWCAdapter,
    HDCAdapter,
    MahalanobisAdapter,
    TinyOLAdapter,
    _first_task_train_X,
    _loader_to_numpy,
    _mean_auroc_over_tasks,
    _truncate_tasks,
)
from scripts.run_s39_quant_sweep import (  # noqa: E402
    SCHEME_WEIGHT_BYTES,
    _proxies,
)
from src.evaluation import compute_cost  # noqa: E402
from src.utils.config_loader import load_config_extends  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    QuantConfig,
    calibrate_activations,
    forward_quant,
)
from src.utils.reproducibility import set_seed  # noqa: E402

MOMENTS = ["fp32", "before", "after", "both"]

# after_scheme (CLI) → (preset QuantConfig, clé RAM/BOPs de run_s39_quant_sweep).
# Source unique des presets et du comptage RAM ; pas de redéfinition.
AFTER_SCHEMES: dict[str, tuple] = {
    "legacy_c": (QuantConfig.legacy_c, "int8_legacy"),
    "per_tensor_calib": (QuantConfig.per_tensor_calib, "int8"),
    "per_channel_int8": (QuantConfig.per_channel_int8, "int8_perchannel"),
    "q15": (QuantConfig.q15, "q15"),
}


# ── Extraction de poids + évaluation quantifiée (émulateur) ─────────────────

def _weights_from_model(model) -> EWCHeadWeights:
    """Lit fc1/fc2/fc3 d'un modèle EWC (FP32 ou QAT) → EWCHeadWeights.

    `EWCMlpClassifier` (FP32) et `EWCMlpInt8Classifier` (QAT) exposent tous deux les
    couches `fc1/fc2/fc3` ; `from_state_dict` ignore les buffers de fake-quant.
    """
    with torch.no_grad():
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return EWCHeadWeights.from_state_dict(state)


def _task_eval_xy(task) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) d'évaluation d'une tâche (test_loader sinon val_loader)."""
    loader = task.get("test_loader") or task["val_loader"]
    xs, ys = [], []
    for x, y in loader:
        xs.append(np.asarray(x).astype(np.float64))
        ys.append(np.asarray(y).ravel())
    if not xs:
        return np.empty((0, 0)), np.empty((0,))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _eval_quant_auroc(
    w: EWCHeadWeights, tasks: list[dict], cfg: QuantConfig, act_max: dict | None
) -> float:
    """AUROC de détection de panne via le noyau PTQ (logit binaire = score)."""
    per_task = []
    for task in tasks:
        X, y = _task_eval_xy(task)
        if X.size == 0:
            continue
        logits = forward_quant(w, X, cfg, act_max=act_max)
        score = np.asarray(logits)[:, 0]  # tête binaire : sortie unique = score d'anomalie
        per_task.append((y.tolist(), score.tolist()))
    return _mean_auroc_over_tasks(per_task)


# ── Coût RAM / latence-proxy par cellule ────────────────────────────────────

def _ewc_dims(cfg: dict) -> tuple[int, list[int]]:
    k = int(cfg["model"]["input_dim"])
    hidden = list(cfg["model"]["hidden_dims"])
    return k, hidden


def _weights_bytes(cfg: dict, weight_bytes: int) -> int:
    """RAM des poids purs (params × octets/poids). Tête EWC binaire : n_classes=1."""
    k, hidden = _ewc_dims(cfg)
    params = compute_cost.params_ewc_mlp(k, hidden, 1)
    return int(params) * int(weight_bytes)


def _lat_proxy_rel(cfg: dict, scheme_key: str) -> float | None:
    k, hidden = _ewc_dims(cfg)
    macs = compute_cost.macs_ewc_mlp(k, hidden, 1)
    return _proxies(macs, scheme_key)["lat_proxy_rel"]


# ── Exécution EWC : les quatre moments ──────────────────────────────────────

def run_ewc_moments(
    cfg: dict,
    config_path: str,
    dataset: str,
    moments: list[str],
    after_scheme: str,
    seed: int,
    n_samples: int | None,
    device: str = "cpu",
) -> dict:
    """Entraîne au plus 2 fois (FP32 + QAT) et dérive les moments demandés."""
    adapter = EWCAdapter()
    quant_cfg, scheme_key = AFTER_SCHEMES[after_scheme]
    quant_cfg = quant_cfg()  # instancie le preset

    # Tâches (une fois).
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)

    need_fp32 = any(m in ("fp32", "after") for m in moments)
    need_qat = any(m in ("before", "both") for m in moments)

    cells: dict[str, dict] = {}

    # --- FP32 (référence + source du chemin `after`) ---
    fp32_model = None
    if need_fp32:
        print("[fp32] entraînement FP32…")
        set_seed(seed)
        fp32_model = adapter.build_fp32(cfg)
        adapter.train(fp32_model, tasks, cfg, device)
        if "fp32" in moments:
            metric = adapter.evaluate(fp32_model, tasks, device)
            cells["fp32"] = {
                "metric": _round(metric),
                "ram_weights_bytes": _weights_bytes(cfg, SCHEME_WEIGHT_BYTES["fp32"]),
                "lat_proxy_rel": 1.0,
            }
            print(f"  auroc={metric:.4f}")

    # --- QAT (before) + source du chemin `both` ---
    qat_model = None
    if need_qat:
        print("[qat] entraînement QAT (fake-quant)…")
        set_seed(seed)
        qat_model = adapter.build_int8(cfg)
        adapter.train(qat_model, tasks, cfg, device)
        if "before" in moments:
            metric = adapter.evaluate(qat_model, tasks, device)  # fake-quant à l'inférence
            cells["before"] = {
                "metric": _round(metric),
                # cible INT8 (poids simulés int8) ; latence non pertinente (pas de noyau entier)
                "ram_weights_bytes": _weights_bytes(cfg, SCHEME_WEIGHT_BYTES["int8"]),
                "lat_proxy_rel": None,
                "note": "borne haute (fake-quant inférence)",
            }
            print(f"  auroc={metric:.4f}")

    # Calibration d'activations sur la 1re tâche (lot représentatif).
    def _cal(model):
        w = _weights_from_model(model)
        X_cal = _first_task_train_X(tasks).astype(np.float64)
        return w, calibrate_activations(w, X_cal)

    # --- after : PTQ sur poids FP32 figés ---
    if "after" in moments:
        print(f"[after] PTQ ({after_scheme}) sur poids FP32 figés…")
        w, act_max = _cal(fp32_model)
        metric = _eval_quant_auroc(w, tasks, quant_cfg, act_max)
        cells["after"] = {
            "metric": _round(metric),
            "ram_weights_bytes": _weights_bytes(cfg, SCHEME_WEIGHT_BYTES[scheme_key]),
            "lat_proxy_rel": _lat_proxy_rel(cfg, scheme_key),
            "after_scheme": after_scheme,
        }
        print(f"  auroc={metric:.4f}")

    # --- both : PTQ sur poids QAT appris (chemin de déploiement) ---
    if "both" in moments:
        print(f"[both] PTQ ({after_scheme}) sur poids QAT appris…")
        w, act_max = _cal(qat_model)
        metric = _eval_quant_auroc(w, tasks, quant_cfg, act_max)
        cells["both"] = {
            "metric": _round(metric),
            "ram_weights_bytes": _weights_bytes(cfg, SCHEME_WEIGHT_BYTES[scheme_key]),
            "lat_proxy_rel": _lat_proxy_rel(cfg, scheme_key),
            "after_scheme": after_scheme,
            "note": "fidèle au déploiement (noyau entier)",
        }
        print(f"  auroc={metric:.4f}")

    return _assemble("ewc", dataset, config_path, "auroc", seed, cfg, cells)


# ── Exécution TinyOL : les quatre moments (S4604) ───────────────────────────

# Nuance honnête (S4604) : TinyOL n'a ni noyau entier per-canal ni vraie boucle QAT sur
# l'autoencodeur. Sa métrique = AUROC sur l'erreur de reconstruction ; l'axe INT8 disponible
# est la fake-quantization par-tenseur (poids INT8 + activations UINT8) de tinyol_int8.py.
# Conséquence : sur l'erreur de reconstruction, `before` ≈ `after` ≈ `both` (les trois
# empruntent le MÊME forward INT8 appliqué au MÊME autoencodeur FP32 entraîné). Ce collapse
# est reporté tel quel (aucune cellule artificielle), avec `na_note` explicite.
_TINYOL_NA_NOTE = (
    "TinyOL : métrique = AUROC erreur de reconstruction. Pas de noyau per-canal ni de "
    "boucle QAT sur l'autoencodeur → before/after/both empruntent le même forward INT8 "
    "par-tenseur (fake-quant poids INT8 + activations UINT8) → valeurs proches par "
    "construction. Seul l'axe fake-quant réel serait la tête OtO (non retenu ici)."
)


def _tinyol_int8_auroc(model, tasks: list[dict]) -> float:
    """AUROC (erreur de reconstruction) via le chemin INT8 (fake-quant à l'inférence)."""
    per_task = []
    for task in tasks:
        loader = task.get("test_loader") or task["val_loader"]
        scores, labels = [], []
        for x, y in loader:
            xa = np.asarray(x).astype(np.float32)
            s = np.array([model._int8.reconstruction_error_int8(xi) for xi in xa])
            scores.extend(np.asarray(s).ravel().tolist())
            labels.extend(np.asarray(y).ravel().tolist())
        per_task.append((labels, scores))
    return _mean_auroc_over_tasks(per_task)


def run_tinyol_moments(
    cfg: dict,
    config_path: str,
    dataset: str,
    moments: list[str],
    after_scheme: str,
    seed: int,
    n_samples: int | None,
    device: str = "cpu",
) -> dict:
    """TinyOL 3-way sur l'erreur de reconstruction (S4604).

    Entraîne l'autoencodeur FP32 au plus une fois ; l'INT8 est une fake-quantization du
    même backbone (calibrée par-tenseur). `before` = fake-quant à l'inférence (borne
    haute) ; `after`/`both` = même calibration PTQ UINT8 par-tenseur (per-canal absent).
    """
    adapter = TinyOLAdapter()

    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)

    cells: dict[str, dict] = {}

    # --- FP32 (référence recon-error AUROC) ---
    fp32_model = None
    if "fp32" in moments:
        print("[fp32] entraînement autoencodeur FP32…")
        set_seed(seed)
        fp32_model = adapter.build_fp32(cfg)
        adapter.train(fp32_model, tasks, cfg, device)
        metric = adapter.evaluate(fp32_model, tasks, device)
        cells["fp32"] = {
            "metric": _round(metric),
            "ram_weights_bytes": adapter.ram_bytes(fp32_model, "fp32"),
            "lat_proxy_rel": 1.0,
        }
        print(f"  auroc(recon)={metric:.4f}")

    # --- INT8 : chemin fake-quant par-tenseur, partagé par before/after/both ---
    # Un seul entraînement + une seule calibration (économie de calcul) : les trois
    # moments décrivent le MÊME modèle INT8 sous trois éclairages (borne haute vs
    # déploiement) — le collapse est honnête, pas un doublon caché.
    int8_model = None
    if any(m in ("before", "after", "both") for m in moments):
        print(f"[int8] autoencodeur INT8 (fake-quant, calib {after_scheme})…")
        set_seed(seed)
        int8_model = adapter.build_int8(cfg)
        adapter.train(int8_model, tasks, cfg, device)  # entraîne FP32 puis wrap+calibrate
        int8_auroc = _round(_tinyol_int8_auroc(int8_model, tasks))
        int8_ram = adapter.ram_bytes(int8_model, "int8")
        print(f"  auroc(recon,int8)={int8_auroc}")

        if "before" in moments:
            cells["before"] = {
                "metric": int8_auroc,
                "ram_weights_bytes": int8_ram,
                "lat_proxy_rel": None,  # pas de noyau entier TinyOL → latence non pertinente
                "note": "borne haute (fake-quant inférence)",
            }
        if "after" in moments:
            cells["after"] = {
                "metric": int8_auroc,
                "ram_weights_bytes": int8_ram,
                "lat_proxy_rel": None,
                "after_scheme": after_scheme,
                "note": "PTQ activations UINT8 par-tenseur (per-canal absent)",
            }
        if "both" in moments:
            cells["both"] = {
                "metric": int8_auroc,
                "ram_weights_bytes": int8_ram,
                "lat_proxy_rel": None,
                "after_scheme": after_scheme,
                "note": "fidèle au déploiement (même forward INT8 par-tenseur)",
            }

    result = _assemble("tinyol", dataset, config_path, "auroc_recon_error", seed, cfg, cells)
    result["na_note"] = _TINYOL_NA_NOTE
    return result


# ── Exécution contexte HDC / Mahalanobis (S4605, mode --moment context) ──────

def run_hdc_context(
    cfg: dict, config_path: str, dataset: str, seed: int, n_samples: int | None,
    device: str = "cpu",
) -> dict:
    """HDC — quantification structurelle (pas d'axe moment).

    HDC est nativement entier (hypervecteurs int8 ±1, mémoire associative int16) : il n'y a
    ni fake-quant d'entraînement ni conversion post-hoc. La métrique INT8 native ≡ FP32
    hypothétique par construction (confirmé par run) ; on reporte le ratio RAM structurel.
    """
    adapter = HDCAdapter()
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)

    set_seed(seed)
    model = adapter.build_int8(cfg)  # HDCClassifier = structure entière native
    adapter.train(model, tasks, cfg, device)
    metric = _round(adapter.evaluate(model, tasks, device))

    ram_fp32 = adapter.ram_bytes(model, "fp32")  # FP32 hypothétique
    ram_int8 = adapter.ram_bytes(model, "int8")  # structure native (int8 HV + int16 AM)
    ram_ratio = round(ram_fp32 / ram_int8, 4) if ram_int8 else None

    return {
        "model": "hdc",
        "dataset": dataset,
        "axis": "structural",
        "moments_3way": "N/A",
        "na_reason": (
            "HDC natif entier (hypervecteurs int8 ±1, mémoire associative int16) : "
            "quantification structurelle, pas de moment avant/après."
        ),
        "metric_name": adapter.metric_name,
        "config_path": config_path,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # fp32 = hypothétique (≡ int8_native par construction), int8_native = mesuré.
        "values": {"fp32": metric, "int8_native": metric},
        "fp32_is_hypothetical": True,
        "ram_fp32_bytes": ram_fp32,
        "ram_int8_bytes": ram_int8,
        "ram_ratio": ram_ratio,
    }


def _maha_auroc(model, tasks: list[dict], score_fn_name: str) -> float:
    """AUROC de la distance de Mahalanobis via `score_fn_name` (fp32/int8/q15)."""
    per_task = []
    for task in tasks:
        loader = task.get("test_loader") or task["val_loader"]
        scores, labels = [], []
        score_fn = getattr(model, score_fn_name)
        for x, y in loader:
            xa = np.asarray(x).astype(np.float32)
            s = score_fn(xa)
            scores.extend(np.asarray(s).ravel().tolist())
            labels.extend(np.asarray(y).ravel().tolist())
        per_task.append((labels, scores))
    return _mean_auroc_over_tasks(per_task)


def run_maha_context(
    cfg: dict, config_path: str, dataset: str, seed: int, n_samples: int | None,
    device: str = "cpu",
) -> dict:
    """Mahalanobis — axe format INT8 vs Q15 (pas d'axe moment).

    Maha n'a pas d'entraînement par gradient (fit statistique) → `before` sans objet. Son
    axe pertinent est le format de Σ⁻¹ : `int8` (affine, casse sur grande dynamique, S28)
    vs `q15` (int16, récupère, S34). Reporte fp32/int8/q15 + deltas.
    """
    from src.models.unsupervised.mahalanobis_detector import MahalanobisDetector
    from src.models.unsupervised.mahalanobis_int8 import MahalanobisDetectorInt8

    adapter = MahalanobisAdapter()
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)
    maha_cfg = adapter._maha_cfg(cfg)

    def _fit(model):
        for i, task in enumerate(tasks):
            model.fit_task(_loader_to_numpy(task["train_loader"]), task_id=i)
        return model

    # fp32
    set_seed(seed)
    m_fp32 = _fit(MahalanobisDetector(maha_cfg))
    auroc_fp32 = _round(_maha_auroc(m_fp32, tasks, "anomaly_score"))

    # int8 (Σ⁻¹ affine)
    set_seed(seed)
    m_int8 = _fit(MahalanobisDetectorInt8({**maha_cfg, "quantization": "int8"}))
    m_int8.calibrate()
    auroc_int8 = _round(_maha_auroc(m_int8, tasks, "anomaly_score_int8"))

    # q15 (Σ⁻¹ int16 Q15)
    set_seed(seed)
    m_q15 = _fit(MahalanobisDetectorInt8({**maha_cfg, "quantization": "q15"}))
    m_q15.calibrate()
    auroc_q15 = _round(_maha_auroc(m_q15, tasks, "anomaly_score_q15"))

    d_int8 = None if (auroc_int8 is None or auroc_fp32 is None) else round(auroc_int8 - auroc_fp32, 6)
    d_q15 = None if (auroc_q15 is None or auroc_fp32 is None) else round(auroc_q15 - auroc_fp32, 6)

    return {
        "model": "mahalanobis",
        "dataset": dataset,
        "axis": "format_int8_q15",
        "moments_3way": "N/A",
        "na_reason": (
            "Maha PTQ-only (fit statistique, pas d'entraînement gradient) ; "
            "axe = format Sigma^-1 (int8 affine vs q15 int16)."
        ),
        "metric_name": "auroc",
        "config_path": config_path,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "values": {"fp32": auroc_fp32, "int8": auroc_int8, "q15": auroc_q15},
        "delta_int8": d_int8,
        "delta_q15": d_q15,
    }


# ── Assemblage du JSON de sortie ────────────────────────────────────────────

def _round(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


def _delta(cell: dict | None, ref: dict | None) -> float | None:
    if not cell or not ref:
        return None
    m, r = cell.get("metric"), ref.get("metric")
    if m is None or r is None:
        return None
    return round(m - r, 6)


def _assemble(
    model: str,
    dataset: str,
    config_path: str,
    metric_name: str,
    seed: int,
    cfg: dict,
    cells: dict,
) -> dict:
    """Construit le dict de sortie (schéma S4602), avec deltas et flags Gap 3."""
    # Squelette complet : les 4 moments présents, `null` si non calculé.
    moments = {m: cells.get(m, {"metric": None, "ram_weights_bytes": None,
                                "lat_proxy_rel": None}) for m in MOMENTS}
    fp32 = cells.get("fp32")
    d_before = _delta(cells.get("before"), fp32)
    d_after = _delta(cells.get("after"), fp32)
    d_both = _delta(cells.get("both"), fp32)

    gap3_ram_ok = None
    both_cell = cells.get("both")
    if fp32 and both_cell:
        r_fp32 = fp32.get("ram_weights_bytes")
        r_both = both_cell.get("ram_weights_bytes")
        if r_fp32 and r_both:
            gap3_ram_ok = bool((r_fp32 / r_both) > 1.0)

    gap3_metric_ok_both = None if d_both is None else bool(abs(d_both) < 0.02)

    return {
        "model": model,
        "dataset": dataset,
        "metric_name": metric_name,
        "seed": seed,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "moments": moments,
        "delta_before_vs_fp32": d_before,
        "delta_after_vs_fp32": d_after,
        "delta_both_vs_fp32": d_both,
        "gap3_metric_ok_both": gap3_metric_ok_both,
        "gap3_ram_ok": gap3_ram_ok,
    }


def _write_config_snapshot(cfg: dict, output: Path) -> None:
    """Dépose config_snapshot.yaml à côté du JSON (convention CLAUDE.md)."""
    snap = output.parent / "config_snapshot.yaml"
    with open(snap, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harnais 3-way des moments de quantification (S4602)")
    p.add_argument("--model", required=True,
                   choices=["ewc", "tinyol", "hdc", "mahalanobis"])
    p.add_argument("--dataset", required=True, choices=["monitoring", "pronostia"])
    p.add_argument("--moment", default="all",
                   choices=["fp32", "before", "after", "both", "all", "context"])
    p.add_argument("--after-scheme", default="per_tensor_calib",
                   choices=sorted(AFTER_SCHEMES))
    p.add_argument("--config", required=True, help="Config YAML (support extends:)")
    p.add_argument("--output", required=True, help="Chemin du JSON de sortie")
    p.add_argument("--seed", type=int, default=None,
                   help="Surcharge la seed de la config (défaut : config, sinon 42)")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Limite d'exemples par tâche (tests rapides)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_extends(args.config)
    dataset = cfg["data"].get("dataset", args.dataset)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    moments = MOMENTS if args.moment == "all" else [args.moment]

    print(f"\n{'=' * 64}")
    print(f"  Moments de quantification — {args.model} × {dataset}")
    print(f"  moments={moments} after_scheme={args.after_scheme} seed={seed}")
    print(f"{'=' * 64}")

    # Mode contexte (S4605) : HDC structurel / Maha INT8-vs-Q15 — pas de grille 3-way.
    if args.moment == "context" or args.model in ("hdc", "mahalanobis"):
        if args.model == "hdc":
            result = run_hdc_context(cfg, args.config, dataset, seed, args.n_samples, args.device)
        elif args.model == "mahalanobis":
            result = run_maha_context(cfg, args.config, dataset, seed, args.n_samples, args.device)
        else:
            raise SystemExit(
                f"--moment context ne s'applique qu'à hdc/mahalanobis (reçu : {args.model})"
            )
    elif args.model == "ewc":
        result = run_ewc_moments(
            cfg, args.config, dataset, moments, args.after_scheme,
            seed, args.n_samples, args.device,
        )
    elif args.model == "tinyol":
        result = run_tinyol_moments(
            cfg, args.config, dataset, moments, args.after_scheme,
            seed, args.n_samples, args.device,
        )
    else:  # pragma: no cover — garde-fou
        raise SystemExit(f"modèle non géré : {args.model}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _write_config_snapshot(cfg, out)

    if "moments" in result:
        print(f"\n  deltas vs fp32 : before={result['delta_before_vs_fp32']} "
              f"after={result['delta_after_vs_fp32']} both={result['delta_both_vs_fp32']}")
        print(f"  Gap 3 métrique (both) : {result['gap3_metric_ok_both']} | "
              f"Gap 3 RAM : {result['gap3_ram_ok']}")
    else:
        print(f"\n  contexte {result['model']} ({result['axis']}) : "
              f"values={result['values']}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
