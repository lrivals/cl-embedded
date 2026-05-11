"""
check_onnx_compat.py — Vérifie la compatibilité d'un modèle ONNX avec STM32Cube.AI.

Vérifie :
    - Validité du modèle (onnx.checker)
    - Opset ≤ 17 (limite STM32Cube.AI)
    - Absence d'opérateurs custom ou non supportés
    - Batch size = 1 (pas de dynamic shapes)
    - Compatibilité des types (float32 uniquement en entrée)

Usage :
    python scripts/check_onnx_compat.py --onnx path/to/model.onnx
    python scripts/check_onnx_compat.py --model ewc  # cherche dans experiments/exp_160/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto

# Opérateurs ONNX non supportés par STM32Cube.AI (opset ≤ 17)
# Source : https://www.st.com/resource/en/user_manual/um2878-getting-started-with-neural-art-turbo.pdf
UNSUPPORTED_OPS: set[str] = {
    "LSTM",
    "GRU",
    "RNN",
    "BatchNormalization",
    "LayerNormalization",
    "GroupNormalization",
    "InstanceNormalization",
    "Einsum",
    "Loop",
    "Scan",
    "If",
    "DynamicQuantizeLinear",
}

MAX_OPSET = 17


def check_model(onnx_path: Path) -> list[str]:
    """
    Vérifie la compatibilité du modèle ONNX avec STM32Cube.AI.

    Parameters
    ----------
    onnx_path : Path
        Chemin vers le fichier .onnx.

    Returns
    -------
    list[str]
        Liste des erreurs/avertissements. Vide si le modèle est compatible.
    """
    errors: list[str] = []

    model = onnx.load(str(onnx_path))

    # 1. Validation structurelle ONNX
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        errors.append(f"[ERREUR] Modèle ONNX invalide : {e}")
        return errors  # Pas la peine de continuer

    # 2. Vérification opset
    for opset in model.opset_import:
        if opset.domain == "" and opset.version > MAX_OPSET:
            errors.append(
                f"[ERREUR] Opset {opset.version} > {MAX_OPSET} (max STM32Cube.AI)"
            )

    # 3. Opérateurs non supportés
    graph = model.graph
    for node in graph.node:
        if node.op_type in UNSUPPORTED_OPS:
            errors.append(
                f"[ERREUR] Opérateur non supporté : {node.op_type} (nœud '{node.name}')"
            )

    # 4. Dynamic shapes (batch size doit être 1)
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            errors.append(f"[AVERT] Entrée '{inp.name}' sans shape définie")
            continue
        for dim in shape.dim:
            if dim.dim_param and not dim.dim_value:
                errors.append(
                    f"[AVERT] Dimension dynamique '{dim.dim_param}' "
                    f"dans l'entrée '{inp.name}' — fixer batch_size=1"
                )

    # 5. Types de données (float32 uniquement en entrée)
    for inp in graph.input:
        dtype = inp.type.tensor_type.elem_type
        if dtype != TensorProto.FLOAT:
            errors.append(
                f"[ERREUR] Entrée '{inp.name}' de type {dtype} (attendu FLOAT=1)"
            )

    return errors


def resolve_onnx_path(args: argparse.Namespace) -> Path:
    if args.onnx:
        return Path(args.onnx)
    model_map = {
        "ewc": "experiments/exp_160/ewc_backbone.onnx",
        "tinyol": "experiments/exp_160/tinyol_encoder.onnx",
    }
    if args.model not in model_map:
        raise ValueError(f"Modèle inconnu : {args.model}. Choisir parmi {list(model_map)}")
    return Path(model_map[args.model])


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie la compatibilité ONNX / STM32Cube.AI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=str, help="Chemin direct vers le fichier .onnx")
    group.add_argument("--model", choices=["ewc", "tinyol"], help="Modèle prédéfini")
    args = parser.parse_args()

    onnx_path = resolve_onnx_path(args)
    if not onnx_path.exists():
        print(f"[ERREUR] Fichier introuvable : {onnx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Vérification : {onnx_path}")
    errors = check_model(onnx_path)

    if not errors:
        print("[OK] Modèle compatible STM32Cube.AI (opset OK, ops supportés, shapes statiques)")
        sys.exit(0)
    else:
        for msg in errors:
            print(msg)
        n_errors = sum(1 for m in errors if m.startswith("[ERREUR]"))
        n_warns = sum(1 for m in errors if m.startswith("[AVERT]"))
        print(f"\n{n_errors} erreur(s), {n_warns} avertissement(s)")
        if n_errors > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
