"""
scripts/convert_tflite.py — Pipeline de repli ONNX → TFLite INT8.

À utiliser quand STM32Cube.AI CLI n'est pas disponible (licence MyST requise).

Pipeline :
    .onnx → onnx-tf backend → TF SavedModel (tmp) → TFLiteConverter INT8 → .tflite

Dépendances optionnelles (non incluses dans requirements.txt) :
    pip install onnx-tf tensorflow

Usage :
    python scripts/convert_tflite.py --model ewc
    python scripts/convert_tflite.py --model tinyol
    python scripts/convert_tflite.py --model all
    python scripts/convert_tflite.py --onnx path/to/model.onnx --output path/to/out.tflite

Références : S1005 — docs/sprints/sprint_phase2/S1005_stm32cubeai_setup.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path


MODEL_PRESETS: dict[str, tuple[str, str]] = {
    "ewc": (
        "experiments/exp_160/ewc_backbone.onnx",
        "experiments/exp_160/ewc_backbone_int8.tflite",
    ),
    "tinyol": (
        "experiments/exp_160/tinyol_encoder.onnx",
        "experiments/exp_160/tinyol_encoder_int8.tflite",
    ),
}


def check_deps() -> bool:
    """Vérifie que onnx-tf et tensorflow sont installés."""
    missing: list[str] = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    try:
        import onnx_tf  # noqa: F401
    except ImportError:
        missing.append("onnx-tf")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    if missing:
        print(
            "[ERREUR] Dépendances manquantes pour le pipeline TFLite :\n"
            f"  pip install {' '.join(missing)}\n"
            "Ces packages sont optionnels (non inclus dans requirements.txt).",
            file=sys.stderr,
        )
        return False
    return True


def convert_to_tflite(
    onnx_path: Path,
    output_path: Path,
    tmp_dir: Path | None = None,
) -> dict:
    """
    Convertit un modèle ONNX FP32 en TFLite INT8.

    Parameters
    ----------
    onnx_path : Path
        Chemin vers le fichier .onnx source.
    output_path : Path
        Chemin de sortie pour le fichier .tflite.
    tmp_dir : Path, optional
        Répertoire pour le SavedModel intermédiaire (nettoyé après conversion).
        Si absent, un répertoire temporaire est créé automatiquement.

    Returns
    -------
    dict
        Résumé de la conversion (chemins, tailles, statut).
    """
    import onnx
    from onnx_tf.backend import prepare

    output_path.parent.mkdir(parents=True, exist_ok=True)

    manage_tmp = tmp_dir is None
    if manage_tmp:
        tmp_path = Path(tempfile.mkdtemp(prefix="onnx2tflite_"))
    else:
        tmp_path = tmp_dir
        tmp_path.mkdir(parents=True, exist_ok=True)

    saved_model_dir = tmp_path / "saved_model"

    try:
        # --- 1. ONNX → TF SavedModel ---
        print(f"[{onnx_path.stem}] Chargement ONNX : {onnx_path}")
        model = onnx.load(str(onnx_path))

        print(f"[{onnx_path.stem}] Conversion ONNX → TF SavedModel : {saved_model_dir}")
        tf_rep = prepare(model)
        tf_rep.export_graph(str(saved_model_dir))

        # --- 2. TF SavedModel → TFLite INT8 ---
        import tensorflow as tf

        print(f"[{onnx_path.stem}] Conversion TF SavedModel → TFLite INT8 : {output_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        tflite_bytes = len(tflite_model)
        onnx_bytes = onnx_path.stat().st_size
        print(
            f"[{onnx_path.stem}] TFLite INT8 écrit : {output_path} "
            f"({tflite_bytes} B, ratio {tflite_bytes / onnx_bytes:.2f}x vs ONNX FP32)"
        )

        return {
            "onnx_path": str(onnx_path),
            "tflite_path": str(output_path),
            "onnx_bytes": onnx_bytes,
            "tflite_bytes": tflite_bytes,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        print(f"[{onnx_path.stem}] ERREUR lors de la conversion : {exc}", file=sys.stderr)
        return {
            "onnx_path": str(onnx_path),
            "tflite_path": str(output_path),
            "onnx_bytes": onnx_path.stat().st_size if onnx_path.exists() else 0,
            "tflite_bytes": 0,
            "success": False,
            "error": str(exc),
        }

    finally:
        if manage_tmp and tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)


def _write_summary(results: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "tflite_summary.json"
    data = {
        "_converted_at": datetime.now().isoformat(),
        "_pipeline": "onnx → onnx-tf SavedModel → TFLiteConverter INT8",
        "models": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nRésumé TFLite → {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de repli ONNX → TFLite INT8 (S1005, quand STM32Cube.AI indisponible)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", type=str, help="Chemin direct vers le fichier .onnx source")
    group.add_argument(
        "--model",
        choices=["ewc", "tinyol", "all"],
        help="Raccourci prédéfini (exp_160)",
    )
    parser.add_argument("--output", type=str, default=None, help="Chemin de sortie .tflite (avec --onnx)")
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Répertoire pour le SavedModel intermédiaire (nettoyé après). Défaut : tmpdir système.",
    )
    args = parser.parse_args()

    if not check_deps():
        sys.exit(1)

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else None
    results: list[dict] = []

    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            print(f"[ERREUR] Fichier introuvable : {onnx_path}", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.output) if args.output else onnx_path.parent / (onnx_path.stem + "_int8.tflite")
        results.append(convert_to_tflite(onnx_path, out_path, tmp_dir))
        _write_summary(results, out_path.parent)

    else:
        models = list(MODEL_PRESETS.keys()) if args.model == "all" else [args.model]
        for name in models:
            onnx_str, tflite_str = MODEL_PRESETS[name]
            onnx_path = Path(onnx_str)
            out_path = Path(tflite_str)
            if not onnx_path.exists():
                print(f"[ERREUR] Fichier introuvable : {onnx_path}", file=sys.stderr)
                results.append({"onnx_path": str(onnx_path), "success": False, "error": "file not found"})
                continue
            results.append(convert_to_tflite(onnx_path, out_path, tmp_dir))

        output_dir = Path(MODEL_PRESETS[models[0]][1]).parent
        _write_summary(results, output_dir)

    print("\n" + "=" * 60)
    print("SYNTHÈSE CONVERSION TFLITE (repli S1005)")
    print("=" * 60)
    all_ok = True
    for r in results:
        ok = r.get("success", False)
        all_ok &= ok
        status = "✓" if ok else "✗"
        tflite_kb = r.get("tflite_bytes", 0) / 1024
        onnx_kb = r.get("onnx_bytes", 0) / 1024
        name = Path(r["onnx_path"]).stem
        print(f"  {status} {name:25s} | ONNX={onnx_kb:.1f} Ko → TFLite={tflite_kb:.1f} Ko")
        if not ok:
            print(f"    Erreur : {r.get('error', '?')}")
    print("=" * 60)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
