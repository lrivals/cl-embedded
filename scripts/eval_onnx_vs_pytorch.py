"""
scripts/eval_onnx_vs_pytorch.py — Comparaison FP32 PyTorch vs ONNX FP32 vs ONNX INT8.

Mesure la dégradation AUROC entre les versions FP32 et INT8 post-quantification.
Seuil acceptable (S1002) : dégradation AUROC < 2 points.

Usage :
    python scripts/eval_onnx_vs_pytorch.py \\
        --onnx experiments/exp_160/ewc_backbone.onnx \\
        --config configs/ewc_config.yaml \\
        --dataset monitoring \\
        [--checkpoint experiments/exp_001_ewc_dataset2/model.pt] \\
        [--onnx-int8 experiments/exp_160/ewc_backbone_int8.onnx]

Références : S1002 — docs/sprints/sprint_phase2/S1002_onnx_export.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.ewc.ewc_mlp import EWCMlpClassifier
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Chargement dataset
# ---------------------------------------------------------------------------

def _load_monitoring_val_data(cfg: dict) -> list[dict]:
    """
    Charge les données de validation du dataset Monitoring (3 tâches).

    Returns
    -------
    list[dict] : chaque dict a "task_id", "domain", "x_val" (np.ndarray), "y_val" (np.ndarray).
    """
    from src.data.monitoring_dataset import get_cl_dataloaders

    csv_path = Path(cfg["data"]["csv_path"])
    normalizer_path = Path(cfg["data"]["normalizer_path"])
    seed = cfg.get("training", {}).get("seed", 42)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {csv_path}\n"
            "Vérifier que le CSV est bien dans data/raw/equipment_monitoring/"
        )

    tasks = get_cl_dataloaders(
        csv_path=csv_path,
        normalizer_path=normalizer_path,
        batch_size=256,
        seed=seed,
    )

    result = []
    for task in tasks:
        xs, ys = [], []
        for x_batch, y_batch in task["val_loader"]:
            xs.append(x_batch.numpy())
            ys.append(y_batch.numpy())
        result.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "x_val": np.concatenate(xs, axis=0),
            "y_val": np.concatenate(ys, axis=0),
        })
    return result


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

def _predict_pytorch(
    model: EWCMlpClassifier,
    x: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x)).numpy()
    return out.squeeze(-1)  # [N]


def _predict_onnx(
    sess: ort.InferenceSession,
    x: np.ndarray,
) -> np.ndarray:
    # Le modèle ONNX est exporté avec batch=1 fixe (contrainte MCU).
    # On itère sample par sample.
    input_name = sess.get_inputs()[0].name
    x_f32 = x.astype(np.float32)
    outs = [
        sess.run(None, {input_name: x_f32[i : i + 1]})[0].squeeze()
        for i in range(len(x_f32))
    ]
    return np.array(outs)  # [N]


# ---------------------------------------------------------------------------
# Validation de cohérence PyTorch ↔ ONNX FP32
# ---------------------------------------------------------------------------

def _check_ort_match(
    model: EWCMlpClassifier,
    sess_fp32: ort.InferenceSession,
    x: np.ndarray,
    atol: float = 1e-5,
) -> tuple[bool, float]:
    """Vérifie que les sorties PyTorch et ONNX FP32 sont identiques à atol près."""
    np.random.seed(0)
    idx = np.random.choice(len(x), size=min(100, len(x)), replace=False)
    x_sample = x[idx]

    pt_out = _predict_pytorch(model, x_sample)
    ort_out = _predict_onnx(sess_fp32, x_sample)
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    return max_diff < atol, max_diff


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Calcule accuracy et AUROC."""
    y_pred = (y_score >= 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": auroc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparaison FP32 PyTorch vs ONNX FP32 vs ONNX INT8 (S1002)"
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Chemin vers le fichier ONNX FP32 (ex. experiments/exp_160/ewc_backbone.onnx).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Chemin vers la config YAML du modèle (ex. configs/ewc_config.yaml).",
    )
    parser.add_argument(
        "--dataset",
        choices=["monitoring"],
        default="monitoring",
        help="Dataset d'évaluation. [défaut: monitoring]",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint PyTorch (.pt). Poids aléatoires si absent.",
    )
    parser.add_argument(
        "--onnx-int8",
        default=None,
        dest="onnx_int8",
        help="Chemin vers le fichier ONNX INT8 (ex. experiments/exp_160/ewc_backbone_int8.onnx).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie JSON pour les résultats. Défaut : répertoire du fichier ONNX.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"Erreur : fichier ONNX introuvable : {onnx_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    input_dim: int = cfg["model"]["input_dim"]
    hidden_dims: list[int] = cfg["model"]["hidden_dims"]

    # --- Modèle PyTorch ---
    model = EWCMlpClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)
    if args.checkpoint is not None:
        if not Path(args.checkpoint).exists():
            print(f"Erreur : checkpoint introuvable : {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        model.load_state(args.checkpoint)
        print(f"[PyTorch] Checkpoint chargé : {args.checkpoint}")
    else:
        warnings.warn(
            "Aucun checkpoint fourni — poids aléatoires. "
            "Les métriques PyTorch ne sont pas significatives.",
            stacklevel=2,
        )

    # --- Sessions ONNX ---
    providers = ["CPUExecutionProvider"]
    sess_fp32 = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"[ONNX FP32] Session chargée : {onnx_path}")

    sess_int8: ort.InferenceSession | None = None
    if args.onnx_int8 is not None:
        int8_path = Path(args.onnx_int8)
        if not int8_path.exists():
            # Tenter de trouver automatiquement le fichier _int8.onnx
            auto_int8 = onnx_path.parent / (onnx_path.stem + "_int8.onnx")
            if auto_int8.exists():
                int8_path = auto_int8
                print(f"[ONNX INT8] Fichier INT8 auto-détecté : {int8_path}")
            else:
                print(f"Avertissement : fichier ONNX INT8 introuvable : {args.onnx_int8}")
        if int8_path.exists():
            sess_int8 = ort.InferenceSession(str(int8_path), providers=providers)
            print(f"[ONNX INT8] Session chargée : {int8_path}")
    else:
        # Détecter automatiquement
        auto_int8 = onnx_path.parent / (onnx_path.stem + "_int8.onnx")
        if auto_int8.exists():
            sess_int8 = ort.InferenceSession(str(auto_int8), providers=providers)
            print(f"[ONNX INT8] Session chargée (auto) : {auto_int8}")

    # --- Données ---
    print(f"\n[Data] Chargement dataset {args.dataset}...")
    tasks = _load_monitoring_val_data(cfg)

    all_results: list[dict] = []

    print("\n" + "=" * 72)
    print(f"{'Tâche':<14} {'Mode':<14} {'Accuracy':>10} {'AUROC':>10}")
    print("-" * 72)

    for task in tasks:
        x_val = task["x_val"].astype(np.float32)
        y_val = task["y_val"].astype(np.float32).squeeze()
        domain = task["domain"]
        task_id = task["task_id"]

        # PyTorch FP32
        pt_scores = _predict_pytorch(model, x_val)
        pt_metrics = _compute_metrics(y_val, pt_scores)
        print(
            f"  Task {task_id} {domain:<10} {'PyTorch FP32':<14} "
            f"{pt_metrics['accuracy']:>10.4f} {pt_metrics['auroc']:>10.4f}"
        )

        # ONNX FP32
        ort_fp32_scores = _predict_onnx(sess_fp32, x_val)
        ort_fp32_metrics = _compute_metrics(y_val, ort_fp32_scores)
        ort_match, max_diff = _check_ort_match(model, sess_fp32, x_val)
        match_str = f"ok (max|Δ|={max_diff:.1e})" if ort_match else f"ATTENTION (max|Δ|={max_diff:.1e})"
        print(
            f"  Task {task_id} {domain:<10} {'ONNX FP32':<14} "
            f"{ort_fp32_metrics['accuracy']:>10.4f} {ort_fp32_metrics['auroc']:>10.4f}"
            f"  [{match_str}]"
        )

        task_result: dict = {
            "task_id": task_id,
            "domain": domain,
            "n_val": len(y_val),
            "pytorch_fp32": pt_metrics,
            "onnx_fp32": ort_fp32_metrics,
            "ort_match_atol_1e5": ort_match,
            "max_abs_diff_fp32": max_diff,
        }

        # ONNX INT8
        if sess_int8 is not None:
            ort_int8_scores = _predict_onnx(sess_int8, x_val)
            ort_int8_metrics = _compute_metrics(y_val, ort_int8_scores)
            auroc_drop = pt_metrics["auroc"] - ort_int8_metrics["auroc"]
            drop_str = f"Δ AUROC={auroc_drop:+.4f}"
            ok_drop = auroc_drop < 0.02  # seuil S1002 : < 2 points
            drop_flag = "ok" if ok_drop else "ATTENTION > 2pts"
            print(
                f"  Task {task_id} {domain:<10} {'ONNX INT8':<14} "
                f"{ort_int8_metrics['accuracy']:>10.4f} {ort_int8_metrics['auroc']:>10.4f}"
                f"  [{drop_str} — {drop_flag}]"
            )
            task_result["onnx_int8"] = ort_int8_metrics
            task_result["auroc_drop_fp32_to_int8"] = auroc_drop
            task_result["within_2pt_threshold"] = ok_drop

        all_results.append(task_result)

    print("=" * 72)

    # --- Agrégats ---
    if all_results:
        avg_auroc_pt = float(np.mean([r["pytorch_fp32"]["auroc"] for r in all_results]))
        avg_auroc_fp32 = float(np.mean([r["onnx_fp32"]["auroc"] for r in all_results]))
        print(f"\n  Avg AUROC PyTorch FP32 : {avg_auroc_pt:.4f}")
        print(f"  Avg AUROC ONNX FP32    : {avg_auroc_fp32:.4f}")
        if sess_int8 is not None and all("onnx_int8" in r for r in all_results):
            avg_auroc_int8 = float(np.mean([r["onnx_int8"]["auroc"] for r in all_results]))
            avg_drop = float(np.mean([r["auroc_drop_fp32_to_int8"] for r in all_results]))
            ok_global = avg_drop < 0.02
            print(f"  Avg AUROC ONNX INT8    : {avg_auroc_int8:.4f}  (Δ={avg_drop:+.4f})")
            print(f"  Critère S1002 (Δ < 2pts) : {'PASSÉ' if ok_global else 'ÉCHOUÉ'}")

    # --- Sauvegarde JSON ---
    output_path = Path(args.output) if args.output else onnx_path.parent / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_data = {
        "_onnx_fp32": str(onnx_path),
        "_onnx_int8": str(args.onnx_int8) if args.onnx_int8 else None,
        "_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "_dataset": args.dataset,
        "tasks": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nRésultats → {output_path}")


if __name__ == "__main__":
    main()
