"""Primitives de schéma partagées (S4204).

Boîtes arrondies, flèches, rangées d'étapes et note de bas de figure — les briques
communes aux diagrammes de flux (`quantization/pipeline`), aux schémas de mécanisme
(`quantization/pedagogy`) et aux schémas de soutenance (`soutenance`). Regroupées ici pour éviter la duplication entre
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


def base_fig() -> tuple[plt.Figure, plt.Axes]:
    """Figure vierge en coordonnées axes ``[0, 1]²``, sans cadre ni graduations.

    Base de tout schéma : les positions sont alors indépendantes de toute donnée.
    """
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def stage_row(
    ax: plt.Axes,
    y: float,
    stages: list[str],
    colors: list[str],
    x0: float = 0.06,
    x1: float = 0.94,
    box_h: float = 0.11,
    fontsize: int = 9,
    box_w: float | None = None,
) -> list[float]:
    """Rangée horizontale de boîtes reliées par des flèches ; retourne les x des centres.

    ``box_w`` force la largeur des boîtes ; par défaut elle s'ajuste au nombre
    d'étapes. Une rangée d'une seule étape est acceptée (pas de division par zéro).
    """
    n = len(stages)
    if n == 0:
        return []
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)] if n > 1 else [(x0 + x1) / 2]
    bw = box_w if box_w is not None else min(0.135, (x1 - x0) / n * 0.92)
    for i, (cx, txt, col) in enumerate(zip(xs, stages, colors)):
        box(ax, cx, y, txt, col, w=bw, h=box_h, fontsize=fontsize)
        if i:
            arrow(ax, xs[i - 1] + bw / 2 + 0.004, y, cx - bw / 2 - 0.004, y, color="#777777")
    return xs
