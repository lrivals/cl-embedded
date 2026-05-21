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
    python scripts/check_onnx_compat.py --model ewc --output report.json
"""

from __future__ import annotations

import argparse
import json
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

_DTYPE_NAMES: dict[int, str] = {
    TensorProto.FLOAT: "FLOAT32",
    TensorProto.UINT8: "UINT8",
    TensorProto.INT8: "INT8",
    TensorProto.INT32: "INT32",
    TensorProto.INT64: "INT64",
    TensorProto.DOUBLE: "FLOAT64",
    TensorProto.BOOL: "BOOL",
}


def _shape_to_list(shape) -> list | None:
    if shape is None:
        return None
    dims = []
    for dim in shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def collect_metadata(model: onnx.ModelProto, onnx_path: Path) -> dict:
    """
    Collecte les métadonnées du modèle : opset, opérateurs, entrées/sorties, taille.

    Parameters
    ----------
    model : onnx.ModelProto
        Modèle ONNX chargé.
    onnx_path : Path
        Chemin du fichier .onnx (pour lire la taille).

    Returns
    -------
    dict
        Métadonnées du modèle.
    """
    graph = model.graph

    opset = next(
        (o.version for o in model.opset_import if o.domain == ""),
        0,
    )

    seen: dict[str, bool] = {}
    operators: list[str] = []
    for node in graph.node:
        if node.op_type not in seen:
            seen[node.op_type] = True
            operators.append(node.op_type)

    inputs = [
        {
            "name": inp.name,
            "shape": _shape_to_list(inp.type.tensor_type.shape),
            "dtype": _DTYPE_NAMES.get(
                inp.type.tensor_type.elem_type,
                str(inp.type.tensor_type.elem_type),
            ),
        }
        for inp in graph.input
    ]
    outputs = [
        {
            "name": out.name,
            "shape": _shape_to_list(out.type.tensor_type.shape),
        }
        for out in graph.output
    ]

    return {
        "opset": opset,
        "operators": operators,
        "n_nodes": len(list(graph.node)),
        "inputs": inputs,
        "outputs": outputs,
        "file_size_bytes": onnx_path.stat().st_size,
    }


def check_model(onnx_path: Path) -> tuple[list[str], dict]:
    """
    Vérifie la compatibilité du modèle ONNX avec STM32Cube.AI.

    Parameters
    ----------
    onnx_path : Path
        Chemin vers le fichier .onnx.

    Returns
    -------
    tuple[list[str], dict]
        (errors, metadata) — errors vide si compatible ; metadata contient opset/ops/shapes.
    """
    errors: list[str] = []

    model = onnx.load(str(onnx_path))

    # 1. Validation structurelle ONNX
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        errors.append(f"[ERREUR] Modèle ONNX invalide : {e}")
        return errors, {}

    metadata = collect_metadata(model, onnx_path)

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

    return errors, metadata


def build_report(onnx_path: Path, errors: list[str], metadata: dict) -> dict:
    """
    Construit le rapport JSON de compatibilité (substitut à `stm32ai analyze`).

    Parameters
    ----------
    onnx_path : Path
        Chemin du fichier .onnx analysé.
    errors : list[str]
        Messages d'erreur/avertissement issus de check_model().
    metadata : dict
        Métadonnées issues de collect_metadata().

    Returns
    -------
    dict
        Rapport JSON sérialisable.
    """
    n_errors = sum(1 for m in errors if m.startswith("[ERREUR]"))
    return {
        "path": str(onnx_path),
        "valid": n_errors == 0,
        **metadata,
        "errors": [m for m in errors if m.startswith("[ERREUR]")],
        "warnings": [m for m in errors if m.startswith("[AVERT]")],
    }


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


def _print_summary(metadata: dict) -> None:
    ops_str = ", ".join(metadata.get("operators", []))
    n_nodes = metadata.get("n_nodes", "?")
    size_kb = metadata.get("file_size_bytes", 0) / 1024
    print(f"  Opset     : {metadata.get('opset', '?')}")
    print(f"  Opérateurs : {ops_str} ({n_nodes} nœuds)")
    for inp in metadata.get("inputs", []):
        shape_str = str(inp.get("shape", "?"))
        print(f"  Entrée    : {inp['name']} — {shape_str} {inp.get('dtype', '')}")
    for out in metadata.get("outputs", []):
        shape_str = str(out.get("shape", "?"))
        print(f"  Sortie    : {out['name']} — {shape_str}")
    print(f"  Taille    : {size_kb:.1f} Ko")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie la compatibilité ONNX / STM32Cube.AI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=str, help="Chemin direct vers le fichier .onnx")
    group.add_argument("--model", choices=["ewc", "tinyol"], help="Modèle prédéfini")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Chemin JSON pour le rapport de compatibilité (substitut à stm32ai analyze)",
    )
    args = parser.parse_args()

    onnx_path = resolve_onnx_path(args)
    if not onnx_path.exists():
        print(f"[ERREUR] Fichier introuvable : {onnx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Vérification : {onnx_path}")
    errors, metadata = check_model(onnx_path)

    if metadata:
        _print_summary(metadata)

    if args.output:
        report = build_report(onnx_path, errors, metadata)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"  Rapport   : {output_path}")

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
