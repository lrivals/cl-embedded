#!/usr/bin/env python3
"""run_sprint46_board.py — S4608 : colonne « both » sur carte réelle (NUCLEO-F439ZI).

Le chemin **both** (Sprint 46) = QAT → export PTQ : entraîner un head EWC **avec fake-quant**
puis exporter ses poids appris à travers le noyau INT8 v2 calibré du firmware. C'est le seul
axe **fidèle au déploiement** (le firmware exécute un noyau entier, jamais de fake-quant à
l'inférence). Cette tâche flashe et mesure `both` sur carte réelle, et le compare en A/B à la
colonne `after` (PTQ depuis des poids FP32-entraînés) déjà mesurée board au Sprint 40
(``experiments/exp_S40_board_v2/results_per_channel_{ds}_frozen.json``).

Réconciliation d'architecture (S4608) : le head firmware est **multiclasse 2 sorties**
(``EWCMlpMulticlass`` → ce que ``export_weights_c.py --int8-v2`` et le kernel v2 consomment).
Le QAT binaire de S28 (``EWCMlpInt8Classifier``, 1 sortie) est incompatible. On entraîne donc
un **head QAT multiclasse** (``EWCMlpMulticlassInt8``, S4608) — poids FP32 sous-jacents,
fake-quant au forward → ``state_dict`` (fc1/fc2/fc3) directement exportable, comme le head FP32.

Réutilise (source unique, 0 duplication) :
  - ``run_s40_board_v2.build_and_flash_s40`` (export --ewc-head + --int8-v2 → build -DEWC_INT8_V2
    → flash), ``_emu_pred``/``_fp32_pred``/``_ram_block``/``_load_head`` ;
  - ``run_sprint36_pc._temporal_tasks``/``_split_task`` (split identique à la réf PC/board S36) ;
  - ``sensor_stream`` (UART v3, flag ``FRAME_FLAGS_INT8_MODE`` 0x40 — aucun flag neuf).

Parité board↔émulateur frozen = 1.000 attendue **par construction** (le board et l'émulateur
quantifient le même checkpoint QAT à l'identique — précédent S40).

Règle « aucun chiffre inventé » : ``experiments/exp_S46_board/`` n'est écrit qu'au stream réussi.

Usage :
    python scripts/run_sprint46_board.py --dataset monitoring --proto frozen
    python scripts/run_sprint46_board.py                      # monitoring + pronostia, frozen
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.sensor_stream as ss  # noqa: E402
from scripts.run_s40_board_v2 import (  # noqa: E402
    SCHEMES,
    _emu_pred,
    _fp32_pred,
    _load_head,
    _ram_block,
    build_and_flash_s40,
)
from scripts.run_sprint36_board import GAP2_LATENCY_US  # noqa: E402
from scripts.run_sprint36_pc import _split_task, _temporal_tasks  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.models.ewc import EWCMlpMulticlassInt8  # noqa: E402
from src.utils.int8_c_emulation import calibrate_activations  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

EXPERIMENTS = Path("experiments")
OUT_DIR = EXPERIMENTS / "exp_S46_board"
AFTER_DIR = EXPERIMENTS / "exp_S40_board_v2"   # colonne `after` board (S40, source FP32)
DEFAULT_CONFIG = "configs/sprint36_ewc_comparison.yaml"
SCHEME = "per_channel"   # kernel v2 calibré (le chemin de déploiement) — cf. S40


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Entraînement QAT multiclasse (miroir strict de run_sprint36_pc.train_and_eval) ──────

def train_qat_checkpoint(X: np.ndarray, y: np.ndarray, k: int, tr_cfg: dict,
                         ewc_lr: float, ewc_lambda: float, ckpt_path: Path) -> Path:
    """Entraîne EWCMlpMulticlassInt8 (QAT) en CL séquentiel et sauve le checkpoint.

    Séquence, split et hyperparamètres identiques à la réf PC/board S36 (seul le head
    diffère : QAT au lieu de FP32) → le checkpoint est exportable comme ``ewc_head.pt``.
    """
    n_tasks = int(tr_cfg["n_tasks"])
    epochs = int(tr_cfg["epochs_per_task"])
    batch_size = int(tr_cfg["batch_size"])
    test_ratio = float(tr_cfg["test_ratio"])

    tasks = _temporal_tasks(X, y, n_tasks)
    splits = [_split_task(Xt, yt, test_ratio) for Xt, yt in tasks]

    model = EWCMlpMulticlassInt8(input_dim=k, n_classes=2, hidden_dims=[32, 16],
                                 dropout=0.2, ewc_lambda=ewc_lambda)
    optimizer = torch.optim.SGD(model.parameters(), lr=ewc_lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for i, (Xtr, ytr, _Xte, _yte) in enumerate(splits):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb) + model.ewc_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.consolidate(loader, n_samples=200)
        print(f"    QAT tâche {i + 1}/{n_tasks} consolidée")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"  checkpoint QAT → {ckpt_path}")
    return ckpt_path


# ── A/B vs `after` (board S40, source FP32) ──────────────────────────────────

def _after_f1(dataset: str) -> float | None:
    """F1 board `after` = kernel v2 même calibration, poids FP32 (exp_S40_board_v2)."""
    p = AFTER_DIR / f"results_{SCHEME}_{dataset}_frozen.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("f1_faulty")


# ── Passe FROZEN board `both` ────────────────────────────────────────────────

def run_both_frozen(dataset: str, condition: str, cfg: dict, args) -> dict:
    print(f"\n{'='*70}\n=== S46 BOTH FROZEN  dataset={dataset}  condition={condition}  "
          f"kernel=v2 (per-canal calibré)  ===\n{'='*70}")
    set_seed(int(cfg["seed"]))
    n_tasks = int(cfg["training"]["n_tasks"])

    X, y, idx, names = load_condition_arrays(dataset, condition, "ewc", seed=int(cfg["seed"]))
    k = len(idx)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = OUT_DIR / f"cell_both_{dataset}_frozen"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Entraîner le head QAT multiclasse (both = QAT → export PTQ).
    qat_ckpt = train_qat_checkpoint(
        X, y, k, cfg["training"],
        float(args.ewc_lr), float(args.ewc_lambda),
        OUT_DIR / "qat_ckpt" / f"{dataset}_ewc_head.pt")

    # 2. Export (poids QAT) → header v2 calibré → build -DEWC_INT8_V2 → flash.
    w = _load_head(qat_ckpt)
    act_max = calibrate_activations(w, X)     # mêmes bornes que l'export → parité frozen
    bss = build_and_flash_s40(SCHEME, dataset, condition, k, X, qat_ckpt, exp_dir)

    # 3. Stream gelé (flag 0x40 → kernel v2 sous -DEWC_INT8_V2).
    results = ss._stream_uart(
        args.port, args.baud, X, y,
        n_samples=len(X), n_tasks=n_tasks,
        rate_hz=float(cfg["uart"]["rate_hz"]), request_update=False, verbose=args.verbose,
        protocol_version=int(cfg["uart"]["proto"]), model_flags=ss.FRAME_FLAGS_INT8_MODE,
    )
    stats = ss._compute_stats(results)
    feats = np.array([r["features"] for r in results], dtype=np.float32)
    board_pred = np.array([int(r["pred"]) for r in results])

    emu_pred = _emu_pred(w, feats, SCHEME, act_max)   # référence bit-exacte du kernel v2
    fp32_pred = _fp32_pred(w, feats)                  # forward FP32 des mêmes poids QAT
    lat = stats.get("latency_p50_us")

    board_samples = [
        {"idx": i, "true": int(results[i]["true"]),
         "pred_board": int(results[i]["pred"]), "pred_pc": int(emu_pred[i]),
         "pred_fp32_ref": int(fp32_pred[i]), "conf_board": results[i].get("confidence")}
        for i in range(len(results))
    ]
    (exp_dir / "board_samples.json").write_text(json.dumps(board_samples))

    f1_both = stats.get("f1_faulty")
    f1_after = _after_f1(dataset)
    result = {
        "exp_id": f"exp_S46_board/{dataset}_both",
        "dataset": dataset, "moment": "both", "platform": "board",
        "model": "ewc", "kernel": "v2", "scheme": SCHEME, "protocol": "frozen",
        "condition": condition, "n_features": k, "feature_names": names,
        "date": datetime.now().isoformat(timespec="seconds"),
        "stream_mode": "frozen (sans --update)",
        # ── champs du schéma S4608 ──
        "latency_dwt_us_p50": lat,
        "bss_bytes": bss,
        "metric_board": f1_both,
        "parity_board_pc": float((board_pred == emu_pred).mean()),
        "ab_vs_after": (round(f1_both - f1_after, 6)
                        if (f1_both is not None and f1_after is not None) else None),
        # ── enrichissement style S40 ──
        "f1_faulty": f1_both, "f1_macro": stats.get("f1_macro"),
        "online_accuracy": stats.get("accuracy"),
        "latency_us_p99": stats.get("latency_p99_us"),
        "latency_us_mean": stats.get("latency_mean_us"),
        "n_streamed": len(results), "crc_errors": stats.get("crc_errors"),
        "gap2_latency_compliant": (lat is not None and lat < GAP2_LATENCY_US),
        "parity_class": "exact_vs_emulator",
        "parity_mismatch_count": int((board_pred != emu_pred).sum()),
        "agreement_int8_vs_fp32": float((board_pred == fp32_pred).mean()),
        "n_compared": len(results),
        "after_ref": str(AFTER_DIR / f"results_{SCHEME}_{dataset}_frozen.json"),
        "f1_after_board": f1_after,
        "note": "both = QAT (EWCMlpMulticlassInt8) → export PTQ → kernel v2 calibré "
                "(fidèle au déploiement) ; A/B vs after (S40, source FP32)",
    }
    result.update(_ram_block(SCHEME, k))
    (OUT_DIR / f"{dataset}_both.json").write_text(json.dumps(result, indent=2))
    print(f"  k={k} .bss={bss} lat_p50={lat}µs F1_both={f1_both} F1_after={f1_after} "
          f"A/B={result['ab_vs_after']} parity_emu={result['parity_board_pc']:.4f} "
          f"ramx={result['ram_ratio_fp32_over_quant']}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Board `both` (QAT→export PTQ) — Sprint 46 S4608")
    p.add_argument("--dataset", choices=["monitoring", "pronostia"], default=None,
                   help="défaut = les deux")
    p.add_argument("--proto", choices=["frozen"], default="frozen",
                   help="frozen (parité exacte + latence inférence) ; online = extension future")
    p.add_argument("--condition", default="5feat")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(cfg["ewc_base_config"]).read_text())
    args.ewc_lr, args.ewc_lambda = float(base["EWC_LR"]), float(base["EWC_LAMBDA"])

    datasets = [args.dataset] if args.dataset else ["monitoring", "pronostia"]
    rows = []
    for ds in datasets:
        try:
            rows.append(run_both_frozen(ds, args.condition, cfg, args))
        except Exception as exc:  # noqa: BLE001 — cellule robuste
            print(f"  [FAIL both/{ds}] {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}\nBoard S46 `both` : {len(rows)} cellules → {OUT_DIR}/")
    for r in rows:
        print(f"  {r['dataset']:10s} F1_both={r.get('metric_board')} "
              f"A/B_vs_after={r.get('ab_vs_after')} lat={r.get('latency_dwt_us_p50')}µs "
              f"parity={r.get('parity_board_pc')}")


if __name__ == "__main__":
    main()
