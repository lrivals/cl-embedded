#!/usr/bin/env python3
"""
export_weights_ewc_rul.py — Exporte EWCMlpRegressor entraîné → C header FP32.

Usage :
    python scripts/export_weights_ewc_rul.py \
        --checkpoint experiments/exp_S25_01/model_ewc_reg.pt \
        --output firmware/stm32f4_blink/inc/model_weights_rul.h \
        [--input-dim 5] [--hidden-dims 32 16]

Sortie : model_weights_rul.h avec :
    const float EWC_REG_W1[EWC_REG_H1][EWC_REG_IN]
    const float EWC_REG_B1[EWC_REG_H1]
    const float EWC_REG_W2[EWC_REG_H2][EWC_REG_H1]
    const float EWC_REG_B2[EWC_REG_H2]
    const float EWC_REG_W3[EWC_REG_OUT][EWC_REG_H2]
    const float EWC_REG_B3[EWC_REG_OUT]
"""

from __future__ import annotations
import argparse
import hashlib
import struct
from pathlib import Path

import torch

from src.models.ewc.ewc_mlp_regression import EWCMlpRegressor


def tensor_to_c_array(t: torch.Tensor, name: str, shape_str: str) -> str:
    """Convertit un tenseur PyTorch en déclaration C const float[][]."""
    flat = t.detach().cpu().float().numpy().flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}{shape_str} = {{\n    {vals}\n}};\n"


def export_ewc_reg(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int = 5,
    hidden_dims: list[int] | None = None,
) -> None:
    if hidden_dims is None:
        hidden_dims = [32, 16]

    model = EWCMlpRegressor(input_dim=input_dim, hidden_dims=hidden_dims)
    state = torch.load(checkpoint_path, map_location="cpu")

    # Accepte un dict state_dict ou un dict wrappé {"model_state_dict": ...}
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    h1, h2 = hidden_dims[0], hidden_dims[1]

    lines: list[str] = [
        f"/* model_weights_rul.h — AUTO-GÉNÉRÉ par export_weights_ewc_rul.py",
        f" * Source : {checkpoint_path}",
        f" * SHA256  : {hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}...",
        f" * Ne pas modifier manuellement — régénérer via le script.",
        f" * Référence : ewc_head_regression.h (EWC_REG_IN={input_dim}, H1={h1}, H2={h2})",
        f" */",
        f"#pragma once",
        f"",
        f"/* Inclure ewc_head_regression.h avant ce header pour les #define */",
        f"",
    ]

    w1 = model.fc1.weight.data   # shape [H1, IN]
    b1 = model.fc1.bias.data     # shape [H1]
    w2 = model.fc2.weight.data   # shape [H2, H1]
    b2 = model.fc2.bias.data     # shape [H2]
    w3 = model.fc3.weight.data   # shape [1, H2]
    b3 = model.fc3.bias.data     # shape [1]

    lines.append(tensor_to_c_array(w1, "EWC_REG_W1_INIT", f"[{h1}][{input_dim}]"))
    lines.append(tensor_to_c_array(b1, "EWC_REG_B1_INIT", f"[{h1}]"))
    lines.append(tensor_to_c_array(w2, "EWC_REG_W2_INIT", f"[{h2}][{h1}]"))
    lines.append(tensor_to_c_array(b2, "EWC_REG_B2_INIT", f"[{h2}]"))
    lines.append(tensor_to_c_array(w3, "EWC_REG_W3_INIT", f"[1][{h2}]"))
    lines.append(tensor_to_c_array(b3, "EWC_REG_B3_INIT", f"[1]"))

    output_path.write_text("\n".join(lines))
    print(f"Exporté {output_path} ({output_path.stat().st_size // 1024} Ko)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EWCMlpRegressor → C header")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output",     type=Path,
                        default=Path("firmware/stm32f4_blink/inc/model_weights_rul.h"))
    parser.add_argument("--input-dim",  type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[32, 16])
    args = parser.parse_args()

    export_ewc_reg(args.checkpoint, args.output, args.input_dim, args.hidden_dims)


if __name__ == "__main__":
    main()
