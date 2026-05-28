#!/usr/bin/env python3
"""
scripts/export_weights_tinyol.py — Export poids TinyOL board (5→32→16→5) vers model_weights.h.

Architecture board (NUCLEO-F439ZI) :
  Encoder : Linear(5→32) + ReLU → Linear(32→16) + ReLU  [poids en Flash]
  Decoder : Linear(16→32) + ReLU → Linear(32→5)          [poids en Flash]

Sans checkpoint : init aléatoire reproductible (seed=42).
Avec --checkpoint : charge un TinyOLBoard sauvé via torch.save(model.state_dict(), ...).

Usage :
    python scripts/export_weights_tinyol.py [--checkpoint PATH] [--seed 42]

Sorties :
  - firmware/stm32f4_blink/inc/model_weights.h  (section TinyOL mise à jour)
  - stdout : embeddings de référence pour MOCK_NORMAL_T0[0] (à copier dans test_tinyol.c)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constantes board (conformes à tinyol.h)
# ---------------------------------------------------------------------------
BOARD_IN  = 5
BOARD_H1  = 32
BOARD_EMB = 16
BOARD_OUT = 5

MODEL_WEIGHTS_H = Path("firmware/stm32f4_blink/inc/model_weights.h")

# MOCK_NORMAL_T0[0] — identique à mock_data.h pour la validation delta
MOCK_SAMPLE = [0.10, 0.05, 0.08, -0.03, 0.12]


# ---------------------------------------------------------------------------
# Modèle TinyOL board (2-layer encoder, compatible avec tinyol.h)
# ---------------------------------------------------------------------------

class TinyOLBoard(nn.Module):
    """
    Autoencoder minimal conforme à l'architecture board (tinyol.h) :
      encoder[0] : Linear(5→32)
      encoder[2] : Linear(32→16)
      decoder[0] : Linear(16→32)
      decoder[2] : Linear(32→5)

    Distinct de TinyOLAutoencoder (src/models/tinyol/autoencoder.py) qui a
    INPUT_DIM=25 et un encoder 3 couches — incompatible avec le board.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(BOARD_IN, BOARD_H1),  # [0]
            nn.ReLU(),                       # [1]
            nn.Linear(BOARD_H1, BOARD_EMB), # [2]
            nn.ReLU(),                       # [3]
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOARD_EMB, BOARD_H1), # [0]
            nn.ReLU(),                       # [1]
            nn.Linear(BOARD_H1, BOARD_OUT), # [2]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb  = self.encoder(x)
        recon = self.decoder(emb)
        return emb, recon


# ---------------------------------------------------------------------------
# Formatage C
# ---------------------------------------------------------------------------

def _c_matrix(name: str, arr: np.ndarray, rows: int, cols: int) -> str:
    """Génère un tableau 2D C constant (style model_weights.h)."""
    lines = [f"static const float {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        row_vals = ", ".join(f"{arr[r, c]:.8f}f" for c in range(cols))
        lines.append(f"    {{{row_vals}}},")
    lines.append("};")
    return "\n".join(lines)


def _c_vector(name: str, arr: np.ndarray, n: int) -> str:
    """Génère un tableau 1D C constant (style model_weights.h)."""
    vals = ", ".join(f"{arr[i]:.8f}f" for i in range(n))
    return f"static const float {name}[{n}] = {{{vals}}};"


def build_tinyol_section(model: TinyOLBoard) -> str:
    """Construit la section TinyOL de model_weights.h à partir des poids du modèle."""
    enc_w1 = model.encoder[0].weight.detach().numpy()   # [32, 5]
    enc_b1 = model.encoder[0].bias.detach().numpy()     # [32]
    enc_w2 = model.encoder[2].weight.detach().numpy()   # [16, 32]
    enc_b2 = model.encoder[2].bias.detach().numpy()     # [16]
    dec_w1 = model.decoder[0].weight.detach().numpy()   # [32, 16]
    dec_b1 = model.decoder[0].bias.detach().numpy()     # [32]
    dec_w2 = model.decoder[2].weight.detach().numpy()   # [5, 32]
    dec_b2 = model.decoder[2].bias.detach().numpy()     # [5]

    lines = [
        "/* ── TinyOL encoder weights — MEM: ~2.8 Ko @ FP32 en Flash ──────────── */",
        f"/* Généré par scripts/export_weights_tinyol.py                          */",
        "#include \"tinyol.h\"",
        "",
        _c_matrix("TINYOL_W_ENC1", enc_w1, BOARD_H1,  BOARD_IN),
        f"  /* MEM: {BOARD_H1 * BOARD_IN * 4} B @ FP32 */",
        _c_vector("TINYOL_B_ENC1", enc_b1, BOARD_H1),
        f"  /* MEM: {BOARD_H1 * 4} B @ FP32 */",
        _c_matrix("TINYOL_W_ENC2", enc_w2, BOARD_EMB, BOARD_H1),
        f"  /* MEM: {BOARD_EMB * BOARD_H1 * 4} B @ FP32 */",
        _c_vector("TINYOL_B_ENC2", enc_b2, BOARD_EMB),
        f"  /* MEM: {BOARD_EMB * 4} B @ FP32 */",
        "",
        "/* ── TinyOL decoder weights — MEM: ~2.8 Ko @ FP32 en Flash ──────────── */",
        _c_matrix("TINYOL_W_DEC1", dec_w1, BOARD_H1,  BOARD_EMB),
        f"  /* MEM: {BOARD_H1 * BOARD_EMB * 4} B @ FP32 */",
        _c_vector("TINYOL_B_DEC1", dec_b1, BOARD_H1),
        f"  /* MEM: {BOARD_H1 * 4} B @ FP32 */",
        _c_matrix("TINYOL_W_DEC2", dec_w2, BOARD_OUT, BOARD_H1),
        f"  /* MEM: {BOARD_OUT * BOARD_H1 * 4} B @ FP32 */",
        _c_vector("TINYOL_B_DEC2", dec_b2, BOARD_OUT),
        f"  /* MEM: {BOARD_OUT * 4} B @ FP32 */",
        "",
        "static const float TINYOL_THRESHOLD = 0.05f;  /* depuis configs/board_tinyol.yaml */",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mise à jour de model_weights.h
# ---------------------------------------------------------------------------

_TINYOL_SECTION_RE = re.compile(
    r"/\* ── TinyOL encoder weights.*?static const float TINYOL_THRESHOLD[^\n]*\n",
    re.DOTALL,
)


def update_model_weights_h(new_section: str, path: Path) -> None:
    """Remplace la section TinyOL dans model_weights.h par regex."""
    content = path.read_text(encoding="utf-8")

    replacement = new_section + "\n"

    if _TINYOL_SECTION_RE.search(content):
        updated = _TINYOL_SECTION_RE.sub(replacement, content)
    else:
        # Bloc non trouvé → append à la fin (premier run après un header minimal)
        updated = content.rstrip("\n") + "\n\n" + replacement

    path.write_text(updated, encoding="utf-8")
    print(f"[export] model_weights.h mis à jour : {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Calcul de référence pour les tests
# ---------------------------------------------------------------------------

def compute_reference(model: TinyOLBoard) -> None:
    """Affiche les sorties de référence pour MOCK_NORMAL_T0[0]."""
    model.eval()
    with torch.no_grad():
        x   = torch.tensor([MOCK_SAMPLE], dtype=torch.float32)
        emb, recon = model(x)
        emb_np   = emb.numpy().flatten()
        recon_np = recon.numpy().flatten()
        x_np     = np.array(MOCK_SAMPLE, dtype=np.float32)
        mse      = float(np.mean((x_np - recon_np) ** 2))

    print("\n[référence Python — à copier dans test_tinyol.c si nécessaire]")
    print(f"  input    : {MOCK_SAMPLE}")
    emb_str = ", ".join(f"{v:.8f}f" for v in emb_np)
    print(f"  emb      : {{{emb_str}}}")
    recon_str = ", ".join(f"{v:.8f}f" for v in recon_np)
    print(f"  recon    : {{{recon_str}}}")
    print(f"  MSE      : {mse:.8f}f")
    print(f"  predict  : {'anomalie' if mse > 0.05 else 'normal'} (seuil=0.05)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Chemin vers un state_dict TinyOLBoard (.pt)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed aléatoire si pas de checkpoint (défaut: 42)")
    p.add_argument("--output", type=Path, default=MODEL_WEIGHTS_H,
                   help=f"Chemin model_weights.h (défaut: {MODEL_WEIGHTS_H})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = TinyOLBoard()

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(f"[erreur] Checkpoint introuvable : {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"[export] Poids chargés depuis {args.checkpoint}", file=sys.stderr)
    else:
        torch.manual_seed(args.seed)
        # Réinitialiser avec Kaiming uniform (défaut PyTorch) après seed
        for layer in [model.encoder[0], model.encoder[2],
                      model.decoder[0], model.decoder[2]]:
            nn.init.kaiming_uniform_(layer.weight, a=0.0)
            nn.init.zeros_(layer.bias)
        print(f"[export] Aucun checkpoint — init aléatoire seed={args.seed}", file=sys.stderr)

    model.eval()

    section = build_tinyol_section(model)
    update_model_weights_h(section, args.output)
    compute_reference(model)

    # Résumé des tailles
    enc_bytes = (BOARD_H1 * BOARD_IN + BOARD_H1 + BOARD_EMB * BOARD_H1 + BOARD_EMB) * 4
    dec_bytes = (BOARD_H1 * BOARD_EMB + BOARD_H1 + BOARD_OUT * BOARD_H1 + BOARD_OUT) * 4
    print(f"\n[bilan] Flash encoder : {enc_bytes} B ({enc_bytes/1024:.1f} Ko) @ FP32",
          file=sys.stderr)
    print(f"[bilan] Flash decoder : {dec_bytes} B ({dec_bytes/1024:.1f} Ko) @ FP32",
          file=sys.stderr)
    print(f"[bilan] Flash total   : {enc_bytes + dec_bytes} B "
          f"({(enc_bytes + dec_bytes)/1024:.1f} Ko) @ FP32", file=sys.stderr)


if __name__ == "__main__":
    main()
