#!/usr/bin/env python3
"""run_s39_matched_compare.py — Harnais de comparaison INT8 appariée PC↔board (S3918).

Produit, pour chaque ``(modèle, dataset, schéma, condition)``, un résultat **PC directement
comparable au board futur** parce qu'il exécute le **chemin board exact** — l'émulateur
bit-à-bit ``src/utils/int8_c_emulation.py`` faisant tourner le schéma du firmware
(``legacy_c`` = kernel v1, ``per_channel_int8``/``q15``/``mixed`` = kernel v2), **jamais** le
modèle QAT du Sprint 28 (``EWCMlpInt8Classifier``). C'est la règle qui rend la comparaison
PC↔board *pertinente* : mêmes poids, mêmes données, même quantification (cf. S3918).

Deux dérives à éviter (documentées S3918) :
  1. Schémas non appariés → le côté PC est l'émulateur du schéma board, pas le QAT S28.
  2. Pipeline de données non partagé → source unique ``load_condition_arrays`` (board & PC
     consomment les mêmes colonnes/ordre/normalisation, cf. S3508), même métrique
     ``compute_fault_f1`` (pas de redéfinition).

Le checkpoint FP32 entraîné est **dumpé** (``exp_S39_matched/checkpoints/``) pour que le
board (S3919, ``run_s39_board.py``) réutilise **exactement les mêmes poids** → parité gelée
bit-exacte attendue.

Sortie : ``experiments/exp_S39_matched/matched_{model}_{dataset}_{scheme}.json`` avec table
par échantillon ``[idx, y_true, pred_fp32, pred_int8_pc]``, F1, accord vs FP32, et les mêmes
clés que la confrontation de parité (``rows``/``pred_pc``/``parity_class``/``n_compared``)
pour que S3919 confronte la sortie board sans retraitement.

Usage :
    python scripts/run_s39_matched_compare.py --model ewc --dataset pronostia --condition 5feat
    python scripts/run_s39_matched_compare.py --dataset cmapss              # tous les schémas
    python scripts/run_s39_matched_compare.py --scheme per_channel_int8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_s39_int8_ablation import train_ewc_head  # noqa: E402
from src.evaluation.feature_conditions import load_condition_arrays  # noqa: E402
from src.evaluation.metrics import compute_fault_f1  # noqa: E402
from src.utils.int8_c_emulation import (  # noqa: E402
    EWCHeadWeights,
    QuantConfig,
    calibrate_activations,
    forward_fp32,
    forward_quant,
    predict,
)

MODELS: list[str] = ["ewc"]
DATASETS: list[str] = ["cmapss", "cwru", "monitoring", "pronostia", "paderborn"]
CONDITION_DEFAULT = "5feat"
OUT_DIR = Path("experiments/exp_S39_matched")

# Schémas board appariés → QuantConfig émulateur. ``fp32`` = référence (pas un schéma board).
#   legacy_c          = kernel firmware v1 (ewc_head_int8.c)
#   per_channel_int8  = kernel firmware v2 défaut (-DEWC_INT8_V2)
#   q15               = kernel firmware v2 -DEWC_INT8_Q15
#   mixed             = kernel firmware v2 -DEWC_INT8_MIXED
SCHEME_TO_QUANTCONFIG: dict[str, QuantConfig] = {
    "legacy_c": QuantConfig.legacy_c(),
    "per_channel_int8": QuantConfig.per_channel_int8(),
    "q15": QuantConfig.q15(),
    "mixed": QuantConfig.mixed_int8w_q15act(),
}
ALL_SCHEMES: list[str] = list(SCHEME_TO_QUANTCONFIG)

# Régime de parité attendu côté board (S3918) : gelé = bit-exact, online = approché.
# Ces schémas sont tous en inférence gelée → parité bit-exacte attendue.
PARITY_CLASS = "exact"


def _weights_from_model(model) -> EWCHeadWeights:
    """Extrait les poids FP32 (``EWCHeadWeights``) d'un ``EWCMlpMulticlass`` entraîné."""
    import torch

    with torch.no_grad():
        state = {k: v.cpu() for k, v in model.state_dict().items()}
    return EWCHeadWeights.from_state_dict(state)


def _save_checkpoint(model, dataset: str, condition: str) -> Path:
    """Dumpe le checkpoint FP32 (``model_state_dict``) réutilisé par le board (S3919).

    Format identique à ``train_board_reference`` (``{"model_state_dict": ...}``) pour que
    ``export_weights_c.py --ewc-head`` / ``--int8-v2`` le consomment tel quel.
    """
    import torch

    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"ewc_{dataset}_{condition}.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    return ckpt


def compare_scheme(
    w: EWCHeadWeights, X: np.ndarray, y: np.ndarray, scheme: str,
    act_max: dict[str, float], logits_fp32: np.ndarray,
) -> dict:
    """Compare un schéma board (émulateur) à la référence FP32 sur X/y.

    Retourne le dict résultat complet pour ce schéma (table par échantillon + F1 + accord).
    """
    cfg = SCHEME_TO_QUANTCONFIG[scheme]
    logits_q = forward_quant(w, X, cfg, act_max=act_max)
    pred_fp32 = predict(logits_fp32)
    pred_q = predict(logits_q)

    rows = [
        {"idx": int(i), "true": int(y[i]),
         "pred_fp32": int(pred_fp32[i]), "pred_int8_pc": int(pred_q[i]),
         # Alias 'pred_pc' = prédiction du chemin board émulé (ce que le board doit reproduire).
         "pred_pc": int(pred_q[i])}
        for i in range(len(y))
    ]
    f1_int8 = float(compute_fault_f1(y, pred_q)["f1_faulty"])
    f1_fp32 = float(compute_fault_f1(y, pred_fp32)["f1_faulty"])
    agreement = float(np.mean(pred_fp32 == pred_q))
    return {
        "scheme": scheme,
        "quant_name": cfg.name,
        "parity_class": PARITY_CLASS,
        "n_compared": len(y),
        "f1_int8_pc": round(f1_int8, 4),
        "f1_fp32": round(f1_fp32, 4),
        "agreement_vs_fp32": round(agreement, 4),
        "rows": rows,
    }


def run_matched(model_name: str, dataset: str, condition: str, schemes: list[str],
                seed: int) -> dict:
    """Entraîne la tête board, dumpe le checkpoint, compare chaque schéma à FP32."""
    X, y, indices, names = load_condition_arrays(dataset, condition, model_name, seed=seed)
    model = train_ewc_head(X, y, seed=seed)
    ckpt = _save_checkpoint(model, dataset, condition)

    w = _weights_from_model(model)
    act_max = calibrate_activations(w, X)  # bornes activation partagées C↔émulateur (S3908)
    logits_fp32 = forward_fp32(w, X)

    results = {s: compare_scheme(w, X, y, s, act_max, logits_fp32) for s in schemes}
    return {
        "model": model_name,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "n_features": int(X.shape[1]),
        "feature_indices": list(indices),
        "feature_names": list(names),
        "n_samples": int(len(y)),
        "positive_ratio": round(float(np.mean(y)), 4),
        "checkpoint": str(ckpt),
        "act_max": {k: float(v) for k, v in act_max.items()},
        "schemes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harnais comparaison INT8 appariée PC↔board (S3918)")
    parser.add_argument("--model", choices=MODELS, default="ewc")
    parser.add_argument("--dataset", choices=DATASETS, default=None,
                        help="Un seul dataset (défaut : les 5).")
    parser.add_argument("--condition", default=CONDITION_DEFAULT,
                        help="Condition de features (défaut : 5feat).")
    parser.add_argument("--scheme", choices=ALL_SCHEMES, default=None,
                        help="Un seul schéma (défaut : tous).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset else DATASETS
    schemes = [args.scheme] if args.scheme else ALL_SCHEMES

    for ds in datasets:
        print(f"[S3918] matched {args.model} × {ds} ({args.condition}) …")
        try:
            result = run_matched(args.model, ds, args.condition, schemes, args.seed)
        except Exception as exc:  # skip honnête : dataset absent / entraînement KO
            print(f"  ⚠️  {args.model}×{ds} ignoré : {type(exc).__name__}: {exc}")
            continue
        # Un fichier par schéma (le board S3919 confronte fichier par schéma).
        for scheme, cell in result["schemes"].items():
            out = OUT_DIR / f"matched_{args.model}_{ds}_{scheme}.json"
            payload = {k: v for k, v in result.items() if k != "schemes"}
            payload.update(cell)
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"  {scheme}: F1={cell['f1_int8_pc']} (fp32={cell['f1_fp32']}) "
                  f"accord={cell['agreement_vs_fp32']} → {out}")


if __name__ == "__main__":
    main()
