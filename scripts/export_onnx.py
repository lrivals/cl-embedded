"""
scripts/export_onnx.py — Export ONNX + quantification PTQ INT8 pour EWC et TinyOL.

Pipeline :
    PyTorch .pt → torch.onnx.export → .onnx → onnxruntime quantize_dynamic → _int8.onnx
    (optionnel) → stm32ai generate → code C INT8

Usage :
    python scripts/export_onnx.py --model ewc --config configs/ewc_config.yaml \\
        --output experiments/exp_160/ewc_backbone.onnx

    python scripts/export_onnx.py --model tinyol --config configs/tinyol_config.yaml \\
        --output experiments/exp_160/tinyol_encoder.onnx

    python scripts/export_onnx.py --model all \\
        --ewc-config configs/ewc_config.yaml \\
        --tinyol-config configs/tinyol_config.yaml \\
        --output-dir experiments/exp_160/

Références : S1002 — docs/sprints/sprint_phase2/S1002_onnx_export.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models.ewc.ewc_mlp import EWCMlpClassifier
from src.models.tinyol.autoencoder import TinyOLAutoencoder
from src.utils.config_loader import load_config

# Opset ONNX compatible STM32Cube.AI NeuralART Turbo — TODO(dorra): confirmer opset optimal
OPSET_VERSION: int = 17


# ---------------------------------------------------------------------------
# Wrapper pour exporter uniquement l'encodeur TinyOL
# ---------------------------------------------------------------------------

class _TinyOLEncoderWrapper(nn.Module):
    """Expose encode() comme forward() pour l'export ONNX du backbone TinyOL."""

    def __init__(self, autoencoder: TinyOLAutoencoder) -> None:
        super().__init__()
        self.enc1 = autoencoder.enc1
        self.enc2 = autoencoder.enc2
        self.enc3 = autoencoder.enc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.enc1(x))  # MEM: 32 × 4 B = 128 B @ FP32 (batch=1)
        z = torch.relu(self.enc2(z))  # MEM: 16 × 4 B = 64 B @ FP32 (batch=1)
        z = self.enc3(z)              # MEM: 8 × 4 B = 32 B @ FP32 (batch=1)
        return z


# ---------------------------------------------------------------------------
# Fonctions d'export
# ---------------------------------------------------------------------------

def export_ewc_backbone(
    config_path: str,
    output_path: str,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Exporte le backbone EWC-MLP vers ONNX et le quantifie en INT8.

    Parameters
    ----------
    config_path : str
        Chemin vers ewc_config.yaml.
    output_path : str
        Chemin de sortie pour le fichier .onnx FP32.
    checkpoint_path : str, optional
        Chemin vers un checkpoint PyTorch (.pt). Si absent, poids aléatoires.

    Returns
    -------
    dict : résumé de l'export (chemins, métriques de validation).
    """
    cfg = load_config(config_path)
    input_dim: int = cfg["model"]["input_dim"]
    hidden_dims: list[int] = cfg["model"]["hidden_dims"]

    model = EWCMlpClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.0)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
        model.load_state(checkpoint_path)
        print(f"[EWC] Checkpoint chargé : {checkpoint_path}")
    else:
        warnings.warn(
            "Aucun checkpoint fourni — poids aléatoires. "
            "Le graphe ONNX est valide mais les sorties sont sans signification.",
            stacklevel=2,
        )

    return _export_and_quantize(
        model=model,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name="ewc_backbone",
        input_dim=input_dim,
    )


def export_tinyol_encoder(
    config_path: str,
    output_path: str,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Exporte l'encodeur TinyOL (backbone frozen) vers ONNX et le quantifie en INT8.

    Parameters
    ----------
    config_path : str
        Chemin vers tinyol_config.yaml.
    output_path : str
        Chemin de sortie pour le fichier .onnx FP32.
    checkpoint_path : str, optional
        Chemin vers un checkpoint PyTorch (.pt). Si absent, poids aléatoires.

    Returns
    -------
    dict : résumé de l'export.
    """
    cfg = load_config(config_path)
    input_dim: int = cfg["backbone"]["input_dim"]
    encoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["encoder_dims"])
    decoder_dims: tuple[int, ...] = tuple(cfg["backbone"]["decoder_dims"])

    autoencoder = TinyOLAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
    )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        autoencoder.load_state_dict(state_dict)
        print(f"[TinyOL] Checkpoint chargé : {checkpoint_path}")
    else:
        warnings.warn(
            "Aucun checkpoint fourni — poids aléatoires. "
            "Le graphe ONNX est valide mais les sorties sont sans signification.",
            stacklevel=2,
        )

    encoder_wrapper = _TinyOLEncoderWrapper(autoencoder)

    return _export_and_quantize(
        model=encoder_wrapper,
        dummy_input=torch.zeros(1, input_dim),
        output_path=output_path,
        model_name="tinyol_encoder",
        input_dim=input_dim,
    )


# ---------------------------------------------------------------------------
# Export commun : torch.onnx → checker → onnxruntime → quantize_dynamic
# ---------------------------------------------------------------------------

def _export_and_quantize(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    model_name: str,
    input_dim: int,
) -> dict:
    """
    Exporte un nn.Module vers ONNX, le valide, le compare avec OnnxRuntime
    et produit une version quantifiée INT8.

    Returns
    -------
    dict avec les clés :
        onnx_path, int8_path, n_params, onnx_valid, ort_match_atol,
        max_abs_diff, mean_abs_diff
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path = out_path.parent / (out_path.stem + "_int8.onnx")

    model.eval()

    # --- 1. Export torch → ONNX ---
    # On utilise l'exporteur legacy (dynamo=False) pour compatibilité avec
    # onnxruntime.quantization.quantize_dynamic (shape inference stable).
    print(f"\n[{model_name}] Export ONNX → {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=OPSET_VERSION,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,  # batch=1 fixe (MCU)
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[{model_name}] Fichier ONNX écrit : {out_path} ({out_path.stat().st_size} B)")

    # --- 2. Vérification onnx.checker ---
    onnx_model = onnx.load(str(out_path))
    try:
        onnx.checker.check_model(onnx_model)
        onnx_valid = True
        print(f"[{model_name}] onnx.checker.check_model → OK")
    except onnx.checker.ValidationError as e:
        onnx_valid = False
        print(f"[{model_name}] onnx.checker.check_model → ERREUR : {e}")

    # --- 3. Vérification onnxruntime vs PyTorch ---
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    np.random.seed(42)
    test_inputs = np.random.randn(100, input_dim).astype(np.float32)
    abs_diffs: list[float] = []

    with torch.no_grad():
        for x_np in test_inputs:
            x_t = torch.from_numpy(x_np[None])  # [1, input_dim]
            pt_out = model(x_t).numpy()
            ort_out = sess.run(None, {input_name: x_np[None]})[0]
            abs_diffs.append(float(np.max(np.abs(pt_out - ort_out))))

    max_diff = float(np.max(abs_diffs))
    mean_diff = float(np.mean(abs_diffs))
    ort_match = max_diff < 1e-5
    status = "OK" if ort_match else "ATTENTION"
    print(
        f"[{model_name}] PyTorch vs ONNX Runtime (100 inputs) → {status} "
        f"| max |Δ| = {max_diff:.2e} (seuil 1e-5)"
    )

    # --- 4. Quantification PTQ INT8 (poids uniquement, sans calibration) ---
    print(f"[{model_name}] Quantification dynamique INT8 → {int8_path}")
    quantize_dynamic(
        model_input=str(out_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[{model_name}] Fichier INT8 écrit : {int8_path} ({int8_path.stat().st_size} B)")

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "model_name": model_name,
        "onnx_path": str(out_path),
        "int8_path": str(int8_path),
        "opset_version": OPSET_VERSION,
        "input_dim": input_dim,
        "n_params": n_params,
        "onnx_fp32_bytes": out_path.stat().st_size,
        "onnx_int8_bytes": int8_path.stat().st_size,
        "onnx_valid": onnx_valid,
        "ort_match_atol_1e5": ort_match,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


# ---------------------------------------------------------------------------
# STM32Cube.AI (optionnel — dépend de S1001)
# ---------------------------------------------------------------------------

def try_stm32ai_generate(onnx_path: str, output_dir: str) -> bool:
    """
    Tente de générer du code C INT8 via STM32Cube.AI CLI.

    Ne bloque pas si stm32ai n'est pas dans le PATH.
    TODO(dorra): confirmer la version STM32Cube.AI pour compatibilité STM32N6.

    Returns
    -------
    bool : True si la génération a réussi, False sinon.
    """
    if shutil.which("stm32ai") is None:
        print("[STM32Cube.AI] stm32ai non trouvé dans le PATH — génération C ignorée.")
        print("  → Installez STM32Cube.AI CLI depuis https://www.st.com/stm32cubeai")
        return False

    stm32ai_out = Path(output_dir) / "stm32ai_output"
    stm32ai_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "stm32ai", "generate",
        "-m", onnx_path,
        "--type", "onnx",
        "-o", str(stm32ai_out),
    ]
    print(f"[STM32Cube.AI] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[STM32Cube.AI] Génération réussie → {stm32ai_out}")
        return True
    else:
        print(f"[STM32Cube.AI] Erreur (code {result.returncode}) :\n{result.stderr}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch → ONNX + quantification PTQ INT8 (S1002)"
    )
    parser.add_argument(
        "--model",
        choices=["ewc", "tinyol", "all"],
        required=True,
        help="Modèle à exporter : 'ewc', 'tinyol', ou 'all'.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Chemin vers le fichier de config YAML (obligatoire si --model != all).",
    )
    parser.add_argument(
        "--ewc-config",
        default="configs/ewc_config.yaml",
        help="Config EWC (utilisé avec --model all). [défaut: configs/ewc_config.yaml]",
    )
    parser.add_argument(
        "--tinyol-config",
        default="configs/tinyol_config.yaml",
        help="Config TinyOL (utilisé avec --model all). [défaut: configs/tinyol_config.yaml]",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie .onnx (pour --model ewc ou tinyol).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/exp_160/",
        help="Répertoire de sortie (pour --model all). [défaut: experiments/exp_160/]",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint PyTorch (.pt) à charger. Poids aléatoires si absent.",
    )
    parser.add_argument(
        "--stm32ai",
        action="store_true",
        help="Tenter la génération code C via STM32Cube.AI CLI (requiert S1001).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results: list[dict] = []

    if args.model == "ewc":
        if args.config is None:
            print("Erreur : --config requis avec --model ewc", file=sys.stderr)
            sys.exit(1)
        out = args.output or "experiments/exp_160/ewc_backbone.onnx"
        summary = export_ewc_backbone(args.config, out, args.checkpoint)
        results.append(summary)
        if args.stm32ai:
            try_stm32ai_generate(out, str(Path(out).parent))

    elif args.model == "tinyol":
        if args.config is None:
            print("Erreur : --config requis avec --model tinyol", file=sys.stderr)
            sys.exit(1)
        out = args.output or "experiments/exp_160/tinyol_encoder.onnx"
        summary = export_tinyol_encoder(args.config, out, args.checkpoint)
        results.append(summary)
        if args.stm32ai:
            try_stm32ai_generate(out, str(Path(out).parent))

    else:  # all
        out_dir = Path(args.output_dir)

        ewc_out = str(out_dir / "ewc_backbone.onnx")
        results.append(export_ewc_backbone(args.ewc_config, ewc_out, args.checkpoint))
        if args.stm32ai:
            try_stm32ai_generate(ewc_out, str(out_dir))

        tinyol_out = str(out_dir / "tinyol_encoder.onnx")
        results.append(export_tinyol_encoder(args.tinyol_config, tinyol_out, args.checkpoint))
        if args.stm32ai:
            try_stm32ai_generate(tinyol_out, str(out_dir))

    # --- Résumé JSON ---
    out_dir_for_summary = Path(results[0]["onnx_path"]).parent
    summary_path = out_dir_for_summary / "export_summary.json"
    summary_data = {
        "_exported_at": datetime.now().isoformat(),
        "_opset_version": OPSET_VERSION,
        "models": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nRésumé export → {summary_path}")

    # --- Affichage synthèse ---
    print("\n" + "=" * 60)
    print("SYNTHÈSE EXPORT ONNX (S1002)")
    print("=" * 60)
    all_ok = True
    for r in results:
        ok = r["onnx_valid"] and r["ort_match_atol_1e5"]
        all_ok &= ok
        status = "✓" if ok else "✗"
        fp32_kb = r["onnx_fp32_bytes"] / 1024
        int8_kb = r["onnx_int8_bytes"] / 1024
        print(
            f"  {status} {r['model_name']:20s} | params={r['n_params']:>5d} "
            f"| FP32={fp32_kb:.1f}Ko | INT8={int8_kb:.1f}Ko "
            f"| max|Δ|={r['max_abs_diff']:.1e}"
        )
    print("=" * 60)
    if not all_ok:
        print("ATTENTION : certains exports ont des problèmes de validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
