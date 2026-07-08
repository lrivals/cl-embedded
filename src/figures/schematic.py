"""Primitives de schéma partagées (S4204).

Boîtes arrondies, flèches et note de bas de figure — les briques communes aux
diagrammes de flux (`quantization/pipeline`) et aux schémas de mécanisme
(`quantization/pedagogy`). Regroupées ici pour éviter la duplication entre
catalogues ; matplotlib pur (patches/annotate), aucune dépendance nouvelle.

Toutes les positions sont en coordonnées **axes** (``transform=ax.transAxes``),
ce qui rend les schémas indépendants des données tracées.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    text: str,
    color: str,
    w: float = 0.17,
    h: float = 0.16,
    fontsize: int = 11,
) -> None:
    """Boîte arrondie centrée en ``(cx, cy)`` (coordonnées axes), bord ``color``."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=mcolors.to_rgba(color, 0.12), edgecolor=color, linewidth=2,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            transform=ax.transAxes, wrap=True)


def arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#555555",
) -> None:
    """Flèche ``->`` de ``(x1, y1)`` à ``(x2, y2)`` (coordonnées axes)."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=18, color=color, linewidth=1.8,
    ))


def footnote(fig: plt.Figure, text: str) -> None:
    """Note grise discrète en bas à gauche de la figure (source/illustration)."""
    fig.text(0.01, 0.005, text, fontsize=8, color="#666666", ha="left")
