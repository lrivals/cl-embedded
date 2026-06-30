#!/usr/bin/env python3
"""run_s29_board_extend.py — Extension board INT8 5→20 (Sprint 29, S2913).

Complète les 15 cellules (modèle, dataset) manquantes du benchmark board INT8 pour obtenir
une grille 4 modèles × 5 datasets comparable au PC (Sprint 28). Chaque cellule produit un
JSON au schéma S2904 dans experiments/exp_S29_board_int8/, **uniquement** après mesure réelle
sur la NUCLEO-F439ZI (règle CLAUDE.md : aucun chiffre inventé).

Procédure par modèle :
  - ewc / hdc  : réutilisent le binaire de base (apprentissage en ligne) → stream direct
                 des deux flags FP32/INT8 (reset DTR entre runs).
  - tinyol     : export encodeur par dataset → 1 build/flash → stream FP32/INT8 (même binaire).
  - mahalanobis: train détecteur 5-feat sur le dataset → export poids (FP32 + INT8) →
                 2 builds (défaut FP32 / -DMAHA_INT8) → flash → stream (flag 0x00).

Gestion N/A honnête : si les tâches de test sont mono-classe (ex. Paderborn) → AUROC non
défini → metric_value=null + na_reason (la latence/RAM restent mesurées).

Usage :
  python scripts/run_s29_board_extend.py                       # 15 cellules manquantes
  python scripts/run_s29_board_extend.py --only mahalanobis:cmapss
  python scripts/run_s29_board_extend.py --models mahalanobis  # tous datasets d'un modèle
  python scripts/run_s29_board_extend.py --dry-run             # pas de flash/stream
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FW_DIR = ROOT / "firmware" / "stm32f4_blink"
OUT_DIR = ROOT / "experiments" / "exp_S29_board_int8"
INC_DIR = FW_DIR / "inc"

# Import dynamique de run_s29_board_int8 (réutilise _run_one, assemble_result, ss).
_spec = importlib.util.spec_from_file_location(
    "run_s29_board_int8", Path(__file__).parent / "run_s29_board_int8.py"
)
r29 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r29)
ss = r29.ss  # sensor_stream

# ── Matrice des 15 cellules manquantes (board 5→20) ───────────────────────────
MISSING = {
    "ewc":         ["cmapss", "monitoring", "paderborn"],
    "hdc":         ["cwru", "pronostia", "paderborn"],
    "tinyol":      ["cmapss", "monitoring", "pronostia", "paderborn"],
    "mahalanobis": ["cmapss", "cwru", "monitoring", "pronostia", "paderborn"],
}

# Échantillons par défaut (cohérents S2904 : ewc/tinyol 498, hdc/maha 300).
N_SAMPLES = {"ewc": 498, "hdc": 300, "tinyol": 498, "mahalanobis": 300}


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _build(extra_cflags: str = "") -> bool:
    """make clean && make all [EXTRA_CFLAGS=...]. Retourne True si OK."""
    subprocess.run(["make", "-C", str(FW_DIR), "clean"], capture_output=True)
    cmd = ["make", "-C", str(FW_DIR), "all"]
    if extra_cflags:
        cmd.append(f"EXTRA_CFLAGS={extra_cflags}")
    proc = _run(cmd)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    return proc.returncode == 0


def _flash() -> bool:
    proc = _run(["make", "-C", str(FW_DIR), "flash"])
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
    return proc.returncode == 0


def _mono_class(y: np.ndarray, n_samples: int, n_tasks: int) -> bool:
    """True si l'ensemble réellement streamé est mono-classe (AUROC non défini)."""
    size = len(y) // n_tasks
    streamed = np.concatenate([y[i * size:(i + 1) * size][:n_samples] for i in range(n_tasks)])
    return len(np.unique(streamed)) < 2


# ── Préparation firmware par modèle ───────────────────────────────────────────

def _train_maha(dataset: str, exp_dir: Path) -> Path:
    """Entraîne un MahalanobisDetector 5-feat sur les features réellement streamées."""
    from src.models.unsupervised import MahalanobisDetector

    X, y = ss._load_dataset(dataset)
    X = np.asarray(X, dtype=np.float32)
    normal = X[np.asarray(y) == 0]
    if len(normal) < 10:          # fallback : pas assez de classe normale → tout le set
        normal = X
    cfg = yaml.safe_load((ROOT / "configs" / "board_mahalanobis.yaml").read_text())
    maha_cfg = cfg.get("mahalanobis", {"anomaly_percentile": 95, "cl_strategy": "refit"})
    model = MahalanobisDetector(maha_cfg)
    model.fit_task(normal, task_id=0)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt = exp_dir / f"mahalanobis_{dataset}_task0.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(model, f)
    print(f"  [maha] μ={model.mu_.shape} Σ⁻¹={model.sigma_inv_.shape} "
          f"seuil={model.threshold_:.4f} → {ckpt}")
    return ckpt


def _export_maha(ckpt: Path) -> bool:
    """Exporte les poids Mahalanobis FP32 (--mahal) ET INT8 (--maha-int8) depuis le même pkl."""
    proc = _run([sys.executable, "scripts/export_weights_c.py",
                 "--mahal", str(ckpt), "--maha-int8", str(ckpt), "--out", str(INC_DIR)])
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
    return proc.returncode == 0


def _export_tinyol(dataset: str) -> bool:
    proc = _run([sys.executable, "scripts/export_weights_tinyol.py",
                 "--train-dataset", dataset, "--train-epochs", "150"])
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
    return proc.returncode == 0


# ── Mesure d'une cellule ──────────────────────────────────────────────────────

def measure_cell(model: str, dataset: str, args) -> dict | None:
    """Prépare le firmware, mesure FP32+INT8 sur board, retourne le JSON schéma S2904."""
    n = args.n_samples or N_SAMPLES[model]
    fp32_flag, int8_flag = r29.MODE_FLAGS[model]
    X, y = ss._load_dataset(dataset)
    X = np.asarray(X, dtype=np.float32)
    na_reason = ("tâches de test mono-classe (AUROC non défini)"
                 if _mono_class(np.asarray(y), n, args.n_tasks) else None)
    if na_reason:
        print(f"  [N/A] {model}×{dataset} : {na_reason}")

    if args.dry_run:
        print(f"  [dry-run] {model}×{dataset} (n={n}) — pas de flash/stream")
        return None

    if model == "mahalanobis":
        ckpt = _train_maha(dataset, OUT_DIR / "checkpoints")
        if not _export_maha(ckpt):
            print("  [FAIL export maha]"); return None
        # FP32 = build défaut (model_weights.h) ; INT8 = build -DMAHA_INT8.
        if not (_build() and _flash()):
            print("  [FAIL build/flash FP32]"); return None
        fp32 = r29._run_one(args.port, args.baud, X, y, n, args.n_tasks,
                            args.rate_hz, model, fp32_flag)
        if not (_build("-DMAHA_INT8") and _flash()):
            print("  [FAIL build/flash INT8]"); return None
        int8 = r29._run_one(args.port, args.baud, X, y, n, args.n_tasks,
                            args.rate_hz, model, int8_flag)
    else:
        if model == "tinyol":
            if not _export_tinyol(dataset):
                print("  [FAIL export tinyol]"); return None
            if not (_build() and _flash()):
                print("  [FAIL build/flash tinyol]"); return None
        elif args.flash_base:
            # ewc/hdc : (re)flash du binaire de base si demandé.
            if not (_build() and _flash()):
                print("  [FAIL build/flash base]"); return None
        # FP32 puis INT8 dans le même binaire (flags distincts, reset DTR entre runs).
        fp32 = r29._run_one(args.port, args.baud, X, y, n, args.n_tasks,
                            args.rate_hz, model, fp32_flag)
        int8 = r29._run_one(args.port, args.baud, X, y, n, args.n_tasks,
                            args.rate_hz, model, int8_flag)

    return r29.assemble_result(model, dataset, fp32, int8, na_reason=na_reason)


def _cells_from_args(args) -> list[tuple[str, str]]:
    if args.only:
        cells = []
        for tok in args.only.split(","):
            m, d = tok.split(":")
            cells.append((m.strip(), d.strip()))
        return cells
    models = args.models.split(",") if args.models else list(MISSING)
    return [(m, d) for m in models for d in MISSING[m]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="Cellules ciblées 'model:dataset,model:dataset'")
    p.add_argument("--models", help="Modèles ciblés (CSV) — tous datasets manquants")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--n-samples", type=int, default=0, help="0 = défaut par modèle")
    p.add_argument("--n-tasks", type=int, default=3)
    p.add_argument("--rate-hz", type=float, default=0.0)
    p.add_argument("--flash-base", action="store_true",
                   help="(re)build+flash le binaire de base avant ewc/hdc")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cells = _cells_from_args(args)
    print(f"Cellules à mesurer : {len(cells)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for model, dataset in cells:
        print(f"\n{'='*60}\n{model} × {dataset}\n{'='*60}")
        try:
            out = measure_cell(model, dataset, args)
        except Exception as exc:  # noqa: BLE001 — robustesse driver hardware
            print(f"  [ERREUR] {model}×{dataset} : {exc}")
            summary.append({"cell": f"{model}×{dataset}", "status": "error", "err": str(exc)})
            continue
        if out is None:
            summary.append({"cell": f"{model}×{dataset}", "status": "skipped"})
            continue
        import json
        out_path = OUT_DIR / f"results_{model}_int8_{dataset}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  ✅ Sauvegardé : {out_path}  (metric={out['metric_value']}, "
              f"lat_p50={out['int8_detail']['latency_p50_us']}µs, crc={out['crc_errors']})")
        summary.append({"cell": f"{model}×{dataset}", "status": "ok",
                        "metric": out["metric_value"]})

    print(f"\n{'='*60}\nRésumé\n{'='*60}")
    for s in summary:
        print(f"  {s}")


if __name__ == "__main__":
    main()
