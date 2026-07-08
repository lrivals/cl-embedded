#!/usr/bin/env python3
"""run_s39_int8_ablation.py — Ablation chiffrée de la perte F1 INT8 (S3904).

Décompose la perte de F1 de l'INT8 board (F1 ≈ 0.14 vs FP32 ≈ 0.92, Sprint 36) en
activant **un facteur à la fois** le long de ``ABLATION_LADDER`` de
``src/utils/int8_c_emulation.py`` :

    legacy_c → fix_acc32 → per_tensor_calib → per_channel_int8 → q15

Chaque marche n'active qu'un seul changement, ce qui isole sa contribution (``delta_prev``).
Le tout tourne **au PC, sans carte** (l'émulateur reproduit le forward C bit-à-bit) : on
diagnostique la dégradation observée sur board sans flasher.

Pour chaque dataset (cmapss, cwru, monitoring, pronostia, paderborn) en condition board
``5feat`` :
  1. Entraîne la tête EWC board (``EWCMlpMulticlass`` — mêmes hyperparamètres que
     ``scripts/train_board_reference.py``, dim d'entrée = dim de la condition).
  2. Calcule ``f1_faulty`` FP32 puis pour chaque marche de l'échelle.
  3. Reporte ``delta_prev`` (contribution du facteur isolé) et le ``dominant_factor``.

Aucune valeur écrite à la main : les JSON sortent de l'exécution.

Usage :
    python scripts/run_s39_int8_ablation.py                 # les 5 datasets
    python scripts/run_s39_int8_ablation.py --dataset pronostia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_board_reference import (  # noqa: E402
    EWC_EPOCHS_PER_TASK,
    EWC_LAMBDA,
    EWC_LR,
    N_TASKS_DEFAULT,
    _temporal_tasks,
)
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    ABLATION_LADDER,
    EWCHeadWeights,
    forward_fp32,
    forward_quant,
    predict,
)
from src.utils.reproducibility import set_seed  # noqa: E402

DATASETS: list[str] = ["cmapss", "cwru", "monitoring", "pronostia", "paderborn"]
CONDITION = "5feat"
OUT_DIR = Path("experiments/exp_S39_ablation")

# Étiquette lisible du facteur isolé par chaque marche de l'échelle.
FACTOR_LABELS: dict[str, str] = {
    "legacy_c": "firmware actuel (int16 wrap + 1/128 figé)",
    "fix_acc32": "accumulateur int32 (supprime l'overflow)",
    "per_tensor_calib": "scale calibré par-tenseur (vs 1/128)",
    "per_channel_int8": "scale par-canal (mirroir QAT PC)",
    "q15": "16-bit (fidélité 256×)",
}


def train_ewc_head(X: np.ndarray, y: np.ndarray, n_tasks: int = N_TASKS_DEFAULT,
                   seed: int = 42):
    """Entraîne la tête EWC board en mémoire (miroir de ``train_board_reference``).

    Retourne le modèle ``EWCMlpMulticlass`` entraîné (input_dim = ``X.shape[1]``,
    pas de dim board figée à 5 → gère monitoring 4-feat). Aucune écriture disque.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass

    set_seed(seed)
    input_dim = int(X.shape[1])
    model = EWCMlpMulticlass(input_dim=input_dim, n_classes=2, hidden_dims=[32, 16],
                             dropout=0.2, ewc_lambda=EWC_LAMBDA)
    optimizer = torch.optim.SGD(model.parameters(), lr=EWC_LR, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for _task_id, (Xt, yt) in enumerate(_temporal_tasks(X, y, n_tasks)):
        ds = TensorDataset(torch.tensor(Xt, dtype=torch.float32),
                           torch.tensor(yt, dtype=torch.long))
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model.train()
        for _ in range(EWC_EPOCHS_PER_TASK):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb) + model.ewc_penalty()
                loss.backward()
                # Clip de gradient : stabilise sur features brutes (Pronostia non normalisé).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.consolidate(loader, n_samples=200)  # Fisher + θ* pour la tâche suivante
    model.eval()
    return model


def _f1(y_true: np.ndarray, logits: np.ndarray) -> float:
    """F1 de la classe faulty depuis des logits [N, 2] (argmax → label)."""
    return float(compute_fault_f1(y_true, predict(logits))["f1_faulty"])


def run_ablation(dataset: str, seed: int = 42) -> dict:
    """Entraîne la tête board 5feat et évalue F1 le long de ``ABLATION_LADDER``."""
    X, y, indices, names = load_condition_arrays(dataset, CONDITION, "ewc", seed=seed)
    model = train_ewc_head(X, y, seed=seed)

    import torch  # local : ne charge torch que si l'entraînement a réussi

    with torch.no_grad():
        state = {k: v.cpu() for k, v in model.state_dict().items()}
    w = EWCHeadWeights.from_state_dict(state)

    f1_fp32 = _f1(y, forward_fp32(w, X))

    ladder: list[dict] = []
    prev_f1 = None
    for cfg in ABLATION_LADDER:
        f1 = _f1(y, forward_quant(w, X, cfg))
        delta = None if prev_f1 is None else round(f1 - prev_f1, 4)
        ladder.append({
            "scheme": cfg.name,
            "f1": round(f1, 4),
            "delta_prev": delta,
            "factor": FACTOR_LABELS.get(cfg.name, cfg.name),
        })
        prev_f1 = f1

    # Facteur dominant = marche au plus grand gain de F1 (delta_prev max positif).
    steps = [s for s in ladder if s["delta_prev"] is not None]
    dominant = max(steps, key=lambda s: s["delta_prev"]) if steps else ladder[0]

    return {
        "dataset": dataset,
        "condition": CONDITION,
        "n_features": int(X.shape[1]),
        "feature_indices": list(indices),
        "feature_names": list(names),
        "n_samples": int(len(y)),
        "positive_ratio": round(float(np.mean(y)), 4),
        "f1_fp32": round(f1_fp32, 4),
        "ladder": ladder,
        "dominant_factor": dominant["factor"],
        "dominant_scheme": dominant["scheme"],
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation chiffrée perte F1 INT8 (S3904)")
    parser.add_argument("--dataset", choices=DATASETS, default=None,
                        help="Un seul dataset (défaut : les 5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.dataset] if args.dataset else DATASETS

    for ds in targets:
        print(f"[S3904] ablation {ds} ({CONDITION}) …")
        try:
            result = run_ablation(ds, seed=args.seed)
        except Exception as exc:  # skip honnête : dataset absent / chargement KO
            print(f"  ⚠️  {ds} ignoré : {type(exc).__name__}: {exc}")
            continue
        out = OUT_DIR / f"{ds}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        deltas = " → ".join(f"{s['scheme']}={s['f1']}" for s in result["ladder"])
        print(f"  F1 fp32={result['f1_fp32']} | {deltas}")
        print(f"  dominant={result['dominant_scheme']} → {out}")


if __name__ == "__main__":
    main()
