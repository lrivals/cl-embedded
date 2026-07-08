#!/usr/bin/env python3
"""ram_breakdown.py — Décomposition de la RAM firmware, par modèle.

Source unique et *pure* des chiffres RAM utilisés par le notebook
`notebooks/cl_eval/ram_measurement/ram_explained.ipynb`. Aucune visualisation ici.

Pour chaque modèle du firmware NUCLEO-F439ZI on distingue :

  • STATIQUE   : poids figés après l'init (chargés de la Flash → copie de travail
                 en `.bss`). Ne bougent pas pendant l'apprentissage en ligne.
  • MODULABLE  : état de continual learning mis à jour à bord (SGD, Fisher, θ*,
                 mémoire associative HDC, moyenne EMA Mahalanobis, tête OtO…).
                 Vit dans la MÊME struct pré-allouée en `.bss`.
  • PILE       : tableaux temporaires alloués pendant forward / update. Ne sont
                 PAS dans `.bss` ; comptés à part (estimation analytique ici, pic
                 réel mesuré par scripts/measure_stack_watermark.py).

Le total `.bss` de chaque modèle est lu **réellement** depuis l'ELF via
`arm-none-eabi-nm --print-size` (source de vérité). Le split statique/modulable
est calculé analytiquement depuis les `#define` de dimension des headers, puis
**vérifié** contre la taille nm (garde-fou anti-dérive).

CLI :
    python scripts/ram_breakdown.py \
        --elf firmware/stm32f4_blink/build/stm32f4_blink.elf
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

FLOAT = 4  # octets d'un float32
ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware" / "stm32f4_blink"
DEFAULT_ELF = FW / "build" / "stm32f4_blink.elf"

# Headers d'où sont lus les #define de dimension (restent synchro configs/board).
HEADER_FILES = [
    FW / "inc" / "ewc_head.h",
    FW / "inc" / "hdc.h",
    FW / "inc" / "mahalanobis.h",
    FW / "inc" / "tinyol.h",
]

# #define clés à extraire (dimensions qui pilotent la taille des structs).
DEFINE_NAMES = [
    "EWC_IN", "EWC_H1", "EWC_H2", "EWC_OUT",
    "HDC_DIM", "HDC_N_FEATURES", "HDC_N_CLASSES", "HDC_RETRAIN_BUF",
    "MAHA_DIM",
    "TINYOL_IN", "TINYOL_H1", "TINYOL_EMB",
]


# ── Lecture des sources de vérité ─────────────────────────────────────────────
def read_bss_symbols(elf: Path) -> dict[str, int]:
    """Retourne {symbole: taille_octets} pour les symboles `.bss` via nm.

    Réutilise le même outil que scripts/measure_stack_watermark.py.
    """
    out = subprocess.check_output(
        ["arm-none-eabi-nm", "--print-size", "--size-sort", "--radix=d", str(elf)],
        text=True,
    )
    sizes: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        # format : "<addr> <size> <type> <name>"
        if len(parts) == 4 and parts[2] in ("b", "B"):
            sizes[parts[3]] = int(parts[1])
    return sizes


def read_defines(headers: list[Path] | None = None) -> dict[str, int]:
    """Extrait les #define entiers listés dans DEFINE_NAMES depuis les headers."""
    headers = headers or HEADER_FILES
    values: dict[str, int] = {}
    pat = re.compile(r"^#define\s+(\w+)\s+(\d+)")
    for h in headers:
        for line in h.read_text().splitlines():
            m = pat.match(line.strip())
            if m and m.group(1) in DEFINE_NAMES and m.group(1) not in values:
                values[m.group(1)] = int(m.group(2))
    missing = [n for n in DEFINE_NAMES if n not in values]
    if missing:
        raise ValueError(f"#define manquants dans les headers : {missing}")
    return values


# ── Décomposition analytique par modèle (dataset Monitoring, dims 5-feat) ─────
def monitoring_layout(defines: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Décompose statique/modulable/pile pour les 4 modèles Monitoring.

    Les formules suivent exactement les définitions de struct des headers
    firmware ; elles sont recoupées avec les tailles nm par ``monitoring_breakdown``.
    """
    d = defines
    IN, H1, H2, OUT = d["EWC_IN"], d["EWC_H1"], d["EWC_H2"], d["EWC_OUT"]
    HDIM, HF, HC = d["HDC_DIM"], d["HDC_N_FEATURES"], d["HDC_N_CLASSES"]
    HBUF = d["HDC_RETRAIN_BUF"]
    MD = d["MAHA_DIM"]
    TIN, TH1, TEMB = d["TINYOL_IN"], d["TINYOL_H1"], d["TINYOL_EMB"]

    # — EWC (EWCHead) : poids | Fisher+θ*+lambda —
    ewc_weights = (H1 * IN + H1 + H2 * H1 + H2 + OUT * H2 + OUT) * FLOAT
    ewc_fisher = (H1 * IN + H2 * H1 + OUT * H2) * FLOAT
    ewc_star = ewc_fisher
    ewc_lambda = FLOAT
    ewc = {
        "symbols": ["g_ewc_head"],
        "static": ewc_weights,
        "modular": ewc_fisher + ewc_star + ewc_lambda,
        "static_desc": "poids w1/w2/w3 + biais (MLP 5→32→16→2)",
        "modular_desc": "Fisher diag. + θ* de référence + λ (régularisation EWC)",
        # forward: h1[H1]+h2[H2]+logits[OUT] ; sgd ajoute dout/dh2/dh1
        "stack_infer": (H1 + H2 + OUT) * FLOAT,
        "stack_train": (H1 + H2 + OUT + OUT + H2 + H1) * FLOAT,
        "update_fn": "ewc_sgd_step() / ewc_consolidate()",
    }

    # — HDC (HDCClassifier) : proj (fixe) | mémoire associative + buffer —
    hdc_proj = HDIM * HF * FLOAT
    hdc_am = HC * HDIM * FLOAT
    hdc_buf = HBUF * (HF + 1)          # buf_storage uint8
    hdc_state = 20 + 3 * 4             # RingBuffer (~20 B) + 3 compteurs int
    hdc = {
        "symbols": ["g_hdc"],
        "static": hdc_proj,
        "modular": hdc_am + hdc_buf + hdc_state,
        "static_desc": "projection aléatoire proj[1000][5] (fixée à l'init)",
        "modular_desc": "mémoire associative am[2][1000] + buffer retrain + ring",
        "stack_infer": HDIM * FLOAT,                       # hv[HDC_DIM] = 4 Ko
        "stack_train": HDIM * FLOAT + HBUF * (HF + 1),     # + fenêtre retrain
        "update_fn": "hdc_update() / hdc_binarize()",
    }

    # — Mahalanobis (MahalanobisDetector) : Σ⁻¹ (fixe) | moyenne EMA —
    maha_prec = MD * MD * FLOAT
    maha_params = 2 * FLOAT            # threshold + ema_alpha
    maha_mean = MD * FLOAT
    maha = {
        "symbols": ["g_detector"],
        "static": maha_prec + maha_params,
        "modular": maha_mean,
        "static_desc": "matrice de précision Σ⁻¹[5][5] + seuil + alpha EMA",
        "modular_desc": "vecteur moyenne mean[5] (mis à jour par EMA)",
        "stack_infer": 2 * MD * FLOAT,   # diff[d] + left[d]
        "stack_train": 2 * MD * FLOAT,
        "update_fn": "maha_update() (EMA)",
    }

    # — TinyOL : auto-encodeur gelé | tête OtO entraînée —
    tin_enc = (TH1 * TIN + TH1 + TEMB * TH1 + TEMB) * FLOAT       # TinyOLEncoder
    tin_dec = (TH1 * TEMB + TH1 + TIN * TH1 + TIN) * FLOAT        # TinyOLDecoder
    oto = 96                                                       # OtOHeadInt8
    tinyol = {
        "symbols": ["g_tinyol_enc", "g_tinyol_dec", "g_oto_int8"],
        "static": tin_enc + tin_dec,
        "modular": oto,
        "static_desc": "encodeur + décodeur (auto-encodeur gelé)",
        "modular_desc": "tête OtO w_master/b_master (seule partie entraînée)",
        "stack_infer": (TH1 + TEMB + TIN) * FLOAT,   # h1 + emb + recon
        "stack_train": (TH1 + TEMB + TIN) * FLOAT,
        "update_fn": "oto_int8_update() (BCE/SGD)",
    }

    return {"EWC": ewc, "HDC": hdc, "Mahalanobis": maha, "TinyOL": tinyol}


def monitoring_breakdown(
    elf: Path = DEFAULT_ELF,
    headers: list[Path] | None = None,
    tol_bytes: int = 8,
) -> dict[str, dict[str, Any]]:
    """Assemble le bilan RAM des 4 modèles Monitoring.

    Attache la taille `.bss` réelle (nm) et vérifie qu'elle égale (± tol) la somme
    analytique statique+modulable.
    """
    defines = read_defines(headers)
    layout = monitoring_layout(defines)
    nm = read_bss_symbols(elf)

    for name, m in layout.items():
        bss_real = sum(nm.get(sym, 0) for sym in m["symbols"])
        analytic = m["static"] + m["modular"]
        m["bss_real_nm"] = bss_real
        m["bss_analytic"] = analytic
        m["bss_total"] = m["static"] + m["modular"]
        if bss_real and abs(analytic - bss_real) > tol_bytes:
            raise AssertionError(
                f"{name}: somme analytique {analytic} B ≠ nm réel {bss_real} B "
                f"(écart {analytic - bss_real} B > {tol_bytes})"
            )
    return layout


def _fmt(n: int) -> str:
    return f"{n:,} B".replace(",", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    args = ap.parse_args()

    if not args.elf.exists():
        raise SystemExit(f"ELF introuvable : {args.elf} (lancer `make all`)")

    bd = monitoring_breakdown(args.elf)
    defines = read_defines()
    print(f"Dimensions firmware (Monitoring 5-feat) : "
          f"EWC_IN={defines['EWC_IN']}  MAHA_DIM={defines['MAHA_DIM']}  "
          f"HDC_DIM={defines['HDC_DIM']}  TINYOL_IN={defines['TINYOL_IN']}\n")

    hdr = f"{'Modèle':<12}{'statique':>12}{'modulable':>12}{'.bss nm':>12}" \
          f"{'pile inf.':>12}{'pile entr.':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in bd.items():
        print(f"{name:<12}{_fmt(m['static']):>12}{_fmt(m['modular']):>12}"
              f"{_fmt(m['bss_real_nm']):>12}{_fmt(m['stack_infer']):>12}"
              f"{_fmt(m['stack_train']):>12}")
    print("\nNote : `.bss` identique en inférence et en entraînement (aucun malloc ;")
    print("l'état modulable est pré-alloué). Seule la pile transitoire varie.")
    print("Pic réel de pile → scripts/measure_stack_watermark.py (carte requise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
