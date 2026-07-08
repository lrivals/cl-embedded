"""Style commun des figures du projet (S4201).

Deux presets rcParams (`slide` / `manuscript`), la palette stable par stratégie
de quantification (mêmes couleurs dans toutes les figures des catalogues
`quantization/*` et dans `docs/context/quantization_strategies.md`), et l'export
PNG normalisé ``docs/figures/<catalogue>/<nom>.png``.

Contrainte : ce module ne doit être importé que depuis src/figures/, scripts/ et
notebooks/ — jamais depuis les modules modèles ou d'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (serveurs, CI)
import matplotlib.pyplot as plt

# DPI d'export aligné sur la convention du projet (src/evaluation/plots.py)
from src.evaluation.plots import FIGURE_DPI

ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT: Path = ROOT / "docs" / "figures"

# ── Palette stable par stratégie de quantification ───────────────────────────
# Famille Material déjà utilisée dans le projet (plots.py, feature_space_plots.py).
# Ces clés sont la nomenclature de référence — identiques dans le doc S4202 et
# toutes les figures des catalogues quantization/*.
STRATEGY_COLORS: dict[str, str] = {
    "fp32": "#2196F3",             # bleu — référence float32
    "int8_qat": "#4CAF50",         # vert — fake-quant PC, métrique préservée
    "int8_ptq_legacy": "#F44336",  # rouge — PTQ board échelle figée 1/128 (défaillant)
    "int8_v2": "#FF9800",          # orange — INT8 v2 per-tensor/per-channel calibré
    "q15": "#9C27B0",              # violet — grille 16 bits (grande dynamique)
    "int16_am": "#795548",         # brun — HDC, mémoire associative int16
}

STRATEGY_LABELS_FR: dict[str, str] = {
    "fp32": "FP32 (référence)",
    "int8_qat": "INT8 QAT (fake-quant PC)",
    "int8_ptq_legacy": "INT8 PTQ legacy (échelle 1/128)",
    "int8_v2": "INT8 v2 (scale calibré)",
    "q15": "Q15 (16 bits)",
    "int16_am": "HDC int16-AM",
}

# Ordre canonique d'affichage (légendes, barres)
STRATEGY_ORDER: list[str] = [
    "fp32", "int8_qat", "int8_ptq_legacy", "int8_v2", "q15", "int16_am",
]

# ── Presets rcParams ─────────────────────────────────────────────────────────

_COMMON: dict = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "font.family": "DejaVu Sans",
}

_PRESETS: dict[str, dict] = {
    # Slides : grandes polices, format 16:9-friendly
    "slide": {
        **_COMMON,
        "figure.figsize": (12.8, 7.2),
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
    },
    # Manuscrit : tailles compatibles LaTeX, sobre
    "manuscript": {
        **_COMMON,
        "figure.figsize": (6.4, 4.2),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.4,
    },
}


def apply_style(target: str = "slide") -> None:
    """Applique le preset rcParams du projet (``slide`` ou ``manuscript``)."""
    if target not in _PRESETS:
        raise ValueError(f"Preset de style inconnu : {target!r} (attendu : {sorted(_PRESETS)})")
    plt.rcdefaults()
    plt.rcParams.update(_PRESETS[target])


def savefig_png(
    fig: plt.Figure,
    catalog: str,
    name: str,
    out_root: Path = DEFAULT_OUT_ROOT,
) -> Path:
    """Exporte ``docs/figures/<catalog>/<name>.png`` (dpi fixe, bbox tight).

    Crée le dossier si besoin, ferme la figure, retourne le chemin produit.
    """
    path = Path(out_root) / catalog / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {path}")
    return path
