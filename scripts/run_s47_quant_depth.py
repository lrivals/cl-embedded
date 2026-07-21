#!/usr/bin/env python3
"""run_s47_quant_depth.py — Harnais du sweep profondeur × granularité × symétrie (S4702).

Troisième axe de la quantification (après *moment* S46 et *format* S4202) : **à quelle
profondeur en bits (sub-INT8) et avec quelle calibration (granularité / symétrie)** la tête
EWC casse, et quel schéma rachète la métrique. Périmètre **EWC-only × {Monitoring, Pronostia}**,
**PC-only** (émulateur bit-exact `int8_c_emulation.py`) — le portage board est le Sprint 48.

Pour un couple (EWC, dataset), le harnais :
  1. entraîne la tête EWC FP32 **une fois** (voie AUROC binaire S28/S4601) ;
  2. extrait les poids (`EWCHeadWeights.from_state_dict`) ;
  3. construit `QuantConfig.subint8(weight_bits, granularity, symmetry, mode)` (S4701) ;
  4. calibre les activations sur le lot d'enrôlement ;
  5. évalue en AUROC (chemin PTQ émulé) + AUROC FP32 + accord + delta + RAM théorique ;
  6. écrit `experiments/exp_S47_depth/exp_S47_ewc_<dataset>_<bits>_<gran>.json`.

Réutilise (source unique, aucune duplication) :
  - scripts/benchmark_int8_fp32.py : `EWCAdapter`, `_first_task_train_X`, `_mean_auroc_over_tasks` ;
  - scripts/run_s46_quant_moment.py : `_weights_from_model`, `_task_eval_xy`, `_eval_quant_auroc` ;
  - src/utils/int8_c_emulation.py  : `forward_fp32`, `forward_quant`, `calibrate_activations`,
                                     `predict`, `QuantConfig.subint8`, `theoretical_weight_ram`.

Règle « aucun chiffre inventé » : chaque champ numérique sort d'un run ; tant que le harnais
n'a pas tourné, le JSON n'existe pas (pas de squelette à `0`).

Usage :
    # une cellule
    python scripts/run_s47_quant_depth.py --config configs/quant_depth/ewc_monitoring_int4_perchannel.yaml
    # tout le sweep
    python scripts/run_s47_quant_depth.py --sweep configs/quant_depth/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.benchmark_int8_fp32 import (  # noqa: E402
    EWCAdapter,
    _first_task_train_X,
    _mean_auroc_over_tasks,
    _truncate_tasks,
)
from scripts.run_s46_quant_moment import (  # noqa: E402
    _eval_quant_auroc,
    _task_eval_xy,
    _weights_from_model,
)
from src.utils.config_loader import load_config_extends  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    QuantConfig,
    calibrate_activations,
    forward_fp32,
    forward_quant,
    theoretical_weight_ram,
)
from src.utils.reproducibility import set_seed  # noqa: E402

OUT_DIR = _ROOT / "experiments" / "exp_S47_depth"
# Axe symétrie (S4704) : les configs `ewc_sym_*` sont routées vers un répertoire
# dédié, avec un tag de symétrie (au lieu de la granularité) dans le nom de fichier.
SYMMETRY_DIR = _ROOT / "experiments" / "exp_S47_symmetry"

# Mapping des profondeurs config → (bits nominaux, weight_mode) pour QuantConfig.subint8.
# Les valeurs entières restent linéaires ; ternaire/binaire portent leur schéma dédié.
DEPTH_MODES: dict[str, tuple[int, str]] = {
    "ternaire": (2, "ternary"),
    "ternary": (2, "ternary"),
    "binaire": (1, "binary"),
    "binary": (1, "binary"),
}


def _resolve_depth(weight_bits) -> tuple[int, str, str]:
    """(bits, weight_mode, tag fichier) depuis la clé config `weight_bits`."""
    if isinstance(weight_bits, str) and weight_bits.lower() in DEPTH_MODES:
        bits, mode = DEPTH_MODES[weight_bits.lower()]
        return bits, mode, weight_bits.lower()
    bits = int(weight_bits)
    return bits, "linear", f"int{bits}"


def _act_repr(act_bits: int) -> str:
    """8 → activations 8-bit calibrées ; 16 → Q15 (borne haute)."""
    return "q15" if int(act_bits) == 16 else "q7_calib"


def _eval_fp32_auroc(w: EWCHeadWeights, tasks: list[dict]) -> float:
    """AUROC FP32 de référence (miroir de `_eval_quant_auroc`, logit binaire = score)."""
    per_task = []
    for task in tasks:
        X, y = _task_eval_xy(task)
        if X.size == 0:
            continue
        logits = forward_fp32(w, X)
        score = np.asarray(logits)[:, 0]
        per_task.append((y.tolist(), score.tolist()))
    return _mean_auroc_over_tasks(per_task)


def _binary_decision(logits: np.ndarray) -> np.ndarray:
    """Décision de la tête EWC binaire : signe du logit d'anomalie (⟺ prob 0.5).

    La tête a une **sortie unique** (`logits[:, 0]` = score d'anomalie) : `predict`
    (argmax) serait dégénéré (toujours 0). L'accord se mesure donc sur le seuil de
    décision `logit > 0` — la vraie frontière binaire normal-vs-faute.
    """
    return (np.asarray(logits)[:, 0] > 0.0).astype(np.int64)


def _agreement_over_tasks(w: EWCHeadWeights, tasks: list[dict], cfg: QuantConfig,
                          act_max: dict) -> float:
    """Accord de décision binaire quant↔fp32, concaténé sur toutes les tâches d'éval."""
    dec_q, dec_f = [], []
    for task in tasks:
        X, _ = _task_eval_xy(task)
        if X.size == 0:
            continue
        dec_q.extend(_binary_decision(forward_quant(w, X, cfg, act_max=act_max)).tolist())
        dec_f.extend(_binary_decision(forward_fp32(w, X)).tolist())
    if not dec_q:
        return float("nan")
    return float(np.mean(np.asarray(dec_q) == np.asarray(dec_f)))


def _round(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


def run_cell(config_path: str, n_samples: int | None, device: str = "cpu") -> dict:
    """Exécute une cellule du sweep (une config = un point profondeur × granularité)."""
    cfg = load_config_extends(config_path)
    dataset = cfg.get("dataset") or cfg["data"].get("dataset")
    weight_bits_key = cfg["weight_bits"]
    granularity = cfg.get("granularity", "per_channel")
    symmetry = cfg.get("symmetry", "symmetric")
    act_bits = int(cfg.get("act_bits", 8))
    seed = int(cfg.get("seed", cfg.get("training", {}).get("seed", 42)))
    metric = cfg.get("metric", "auroc")

    bits, weight_mode, tag = _resolve_depth(weight_bits_key)
    quant_cfg = QuantConfig.subint8(
        bits=bits, granularity=granularity, symmetry=symmetry,
        mode=weight_mode, act_repr=_act_repr(act_bits),
    )

    print(f"\n{'=' * 64}")
    print(f"  S47 depth — ewc × {dataset} | weight_bits={weight_bits_key} "
          f"({weight_mode}) granularity={granularity} symmetry={symmetry} act={act_bits}b")
    print(f"{'=' * 64}")

    # --- Tête EWC FP32 de référence (une fois par cellule ; même seed → même tête/dataset) ---
    adapter = EWCAdapter()
    set_seed(seed)
    tasks = adapter.load_tasks(cfg, config_path)
    if n_samples is not None:
        tasks = _truncate_tasks(tasks, n_samples)

    set_seed(seed)
    model = adapter.build_fp32(cfg)
    adapter.train(model, tasks, cfg, device)
    w = _weights_from_model(model)

    # --- Calibration activations (lot d'enrôlement = 1re tâche) ---
    X_cal = _first_task_train_X(tasks).astype(np.float64)
    act_max = calibrate_activations(w, X_cal)

    # --- Métriques ---
    auroc_fp32 = _round(_eval_fp32_auroc(w, tasks))
    auroc_quant = _round(_eval_quant_auroc(w, tasks, quant_cfg, act_max))
    delta = None if (auroc_quant is None or auroc_fp32 is None) else round(
        auroc_quant - auroc_fp32, 6)
    agree = _round(_agreement_over_tasks(w, tasks, quant_cfg, act_max))
    ram_bytes, ram_ratio = theoretical_weight_ram(w, quant_cfg)

    print(f"  auroc_fp32={auroc_fp32} auroc_quant={auroc_quant} delta={delta} "
          f"agreement={agree} ram_ratio=×{ram_ratio}")

    return {
        "model": "ewc",
        "dataset": dataset,
        "weight_bits": weight_bits_key,
        "weight_mode": weight_mode,
        "granularity": granularity,
        "symmetry": symmetry,
        "act_bits": act_bits,
        "metric": metric,
        "auroc_fp32": auroc_fp32,
        "auroc_quant": auroc_quant,
        "delta_auroc": delta,
        "agreement_vs_fp32": agree,
        "ram_weight_bytes_theoretical": int(ram_bytes),
        "ram_ratio_vs_fp32": ram_ratio,
        "ram_note": "théorique (bit-packée) ; RAM .bss réelle = Sprint 48",
        "seed": seed,
        "config_path": config_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": {
            "weight_bits": weight_bits_key, "granularity": granularity,
            "symmetry": symmetry, "act_bits": act_bits, "seed": seed,
            "input_dim": cfg["model"]["input_dim"],
            "hidden_dims": list(cfg["model"]["hidden_dims"]),
        },
    }


def _is_symmetry_axis(config_path: str) -> bool:
    """Une cellule de l'axe symétrie (S4704) = config `ewc_sym_*` → répertoire dédié."""
    return "sym" in Path(config_path).name


def _out_path(result: dict) -> Path:
    _, _, tag = _resolve_depth(result["weight_bits"])
    if _is_symmetry_axis(result["config_path"]):
        # S4704 : tag de symétrie (au lieu de granularité) — la granularité est figée
        # à la gagnante S4703 (per_channel), la variable balayée est la symétrie.
        return SYMMETRY_DIR / f"exp_S47_ewc_{result['dataset']}_{tag}_{result['symmetry']}.json"
    return OUT_DIR / f"exp_S47_ewc_{result['dataset']}_{tag}_{result['granularity']}.json"


def _write(result: dict) -> Path:
    out = _out_path(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep profondeur/granularité/symétrie EWC (S4702)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="Config YAML d'une cellule (configs/quant_depth/*.yaml)")
    g.add_argument("--sweep", help="Répertoire de configs à balayer (boucle)")
    p.add_argument("--filter", default=None,
                   help="Sous-chaîne à matcher dans le nom des configs (--sweep), "
                        "ex. 'sym' pour l'axe symétrie S4704")
    p.add_argument("--n-samples", type=int, default=None,
                   help="Limite d'exemples par tâche (tests rapides)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = ([args.config] if args.config
               else sorted(str(p) for p in Path(args.sweep).glob("*.yaml")))
    if args.filter:
        configs = [c for c in configs if args.filter in Path(c).name]
    if not configs:
        raise SystemExit(
            f"aucune config trouvée dans {args.sweep}"
            + (f" (filtre '{args.filter}')" if args.filter else ""))

    written = []
    for cp in configs:
        result = run_cell(cp, args.n_samples, args.device)
        out = _write(result)
        written.append(out)
        print(f"  → {out}")

    dirs = sorted({str(p.parent) for p in written})
    print(f"\n{len(configs)} cellule(s) écrite(s) dans : {', '.join(dirs)}")


if __name__ == "__main__":
    main()
