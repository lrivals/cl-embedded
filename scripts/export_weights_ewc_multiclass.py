#!/usr/bin/env python3
"""
export_weights_ewc_multiclass.py — Exporte EWCMlpMulticlass entraîné → C header FP32.

Usage :
    python scripts/export_weights_ewc_multiclass.py \
        --checkpoint experiments/exp_S25_03/model_ewc_mc.pt \
        --output firmware/stm32f4_blink/inc/model_weights_multiclass.h \
        [--input-dim 9] [--n-classes 10] [--hidden-dims 32 16]
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import torch

from src.models.ewc.ewc_mlp_multiclass import EWCMlpMulticlass


def tensor_to_c_array(t: torch.Tensor, name: str, shape_str: str) -> str:
    flat = t.detach().cpu().float().numpy().flatten()
    vals = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}{shape_str} = {{\n    {vals}\n}};\n"


def export_ewc_mc(
    checkpoint_path: Path,
    output_path: Path,
    input_dim: int = 9,
    n_classes: int = 10,
    hidden_dims: list[int] | None = None,
) -> None:
    if hidden_dims is None:
        hidden_dims = [32, 16]

    model = EWCMlpMulticlass(input_dim=input_dim, n_classes=n_classes, hidden_dims=hidden_dims)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    h1, h2 = hidden_dims[0], hidden_dims[1]

    lines: list[str] = [
        f"/* model_weights_multiclass.h — AUTO-GÉNÉRÉ par export_weights_ewc_multiclass.py",
        f" * Source    : {checkpoint_path}",
        f" * SHA256    : {hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}...",
        f" * N_CLASSES : {n_classes} (CWRU=10, Paderborn=3)",
        f" * Ne pas modifier manuellement.",
        f" */",
        f"#pragma once",
        f"",
    ]

    w1 = model.fc1.weight.data   # [H1, IN]
    b1 = model.fc1.bias.data
    w2 = model.fc2.weight.data   # [H2, H1]
    b2 = model.fc2.bias.data
    w3 = model.fc3.weight.data   # [N_CLASSES, H2]
    b3 = model.fc3.bias.data

    lines.append(tensor_to_c_array(w1, "EWC_MC_W1_INIT", f"[{h1}][{input_dim}]"))
    lines.append(tensor_to_c_array(b1, "EWC_MC_B1_INIT", f"[{h1}]"))
    lines.append(tensor_to_c_array(w2, "EWC_MC_W2_INIT", f"[{h2}][{h1}]"))
    lines.append(tensor_to_c_array(b2, "EWC_MC_B2_INIT", f"[{h2}]"))
    lines.append(tensor_to_c_array(w3, "EWC_MC_W3_INIT", f"[{n_classes}][{h2}]"))
    lines.append(tensor_to_c_array(b3, "EWC_MC_B3_INIT", f"[{n_classes}]"))

    output_path.write_text("\n".join(lines))
    print(f"Exporté {output_path} (N_CLASSES={n_classes}, {output_path.stat().st_size // 1024} Ko)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EWCMlpMulticlass → C header")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output",     type=Path,
                        default=Path("firmware/stm32f4_blink/inc/model_weights_multiclass.h"))
    parser.add_argument("--input-dim",  type=int, default=9)
    parser.add_argument("--n-classes",  type=int, default=10)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[32, 16])
    args = parser.parse_args()

    export_ewc_mc(args.checkpoint, args.output, args.input_dim, args.n_classes, args.hidden_dims)


if __name__ == "__main__":
    main()
