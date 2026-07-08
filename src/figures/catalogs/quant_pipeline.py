"""Catalogue `quantization/pipeline` — *où* la quantification s'applique (S4204).

5 diagrammes de flux de données montrant, pour chaque stratégie, le point exact
de la chaîne réelle du projet où la transformation FP32→quantifié intervient :

    données → entraînement PC → checkpoint → export_weights_c.py → header .h
            → firmware (RAM) → forward (FPU) → MAJ CL

C'est la question récurrente en présentation (« l'INT8, il est appliqué où ? »)
et la clé des résultats opposés (QAT PC ≠ PTQ board alors que le format stocké
est identique). Schémas **matplotlib purs** (aucune donnée d'expérience) : seules
les annotations structurelles de format (RAM ÷4, ÷2) — issues de la *définition*
des formats, pas d'une mesure — sont permises. Les étapes reprennent les vrais
noms du dépôt : le diagramme sert aussi de carte du code.

Couleurs par stratégie : ``src.figures.style.STRATEGY_COLORS`` (cohérence avec
`quantization/pedagogy` et `quantization/impact`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from src.figures.registry import register_catalog
from src.figures.schematic import arrow, box, footnote
from src.figures.style import STRATEGY_COLORS, STRATEGY_LABELS_FR, savefig_png

CATALOG = "quantization/pipeline"

# Couleur des étapes hors périmètre d'une stratégie (grisées)
C_OFF = "#B0BEC5"
C_NEUTRAL = "#607D8B"

# Étapes canoniques de la chaîne (nom court affiché, vrai artefact du dépôt)
STAGES: list[str] = [
    "Données\n(capteurs)",
    "Entraînement\nPC (PyTorch)",
    "Checkpoint\n.pt / .pkl",
    "export_weights_c.py",
    "header .h\n(model_weights*.h)",
    "Firmware RAM\n(pipeline.c)",
    "Forward\nFPU (Cortex-M4)",
    "MAJ CL\n(online)",
]

LABELS: dict[str, str] = {
    "f1_title": "Chaîne de référence FP32 : float32 de bout en bout",
    "f1_note": "float32 partout (poids, activations, MAJ CL) — RAM ×1, forward sur FPU Cortex-M4. "
               "Aucune quantification : c'est la base de comparaison des 4 schémas suivants.",
    "f2_title": "INT8 QAT (fake-quant PC) : la quantification vit DANS la boucle d'entraînement",
    "f2_box": "pourquoi la métrique tient",
    "f2_why": "Le fake-quant (quant→déquant) est inséré dans le forward d'entraînement : le gradient\n"
              "« voit » l'erreur de quantification (STE) et le modèle s'y adapte. Évaluation PC —\n"
              "l'export et le firmware ne sont pas concernés (Δmétrique ≤ 0.006, Sprint 28).",
    "f3_title": "INT8 PTQ legacy (board) : quantification one-shot au boot du firmware",
    "f3_collapse": "pourquoi la métrique s'effondre",
    "f3_collapse_txt": "ewc_int8_from_fp32 convertit les poids FP32 déjà entraînés avec une échelle\n"
                       "FIGÉE (1/128) et un accumulateur int16 : aucune calibration, l'erreur\n"
                       "n'a jamais été vue à l'entraînement → F1 0.07–0.15 (Sprints 29/36).",
    "f3_slow": "pourquoi pas plus rapide",
    "f3_slow_txt": "poids stockés int8 (RAM ÷4) mais déquantifiés en float dans la boucle :\n"
                   "le MAC reste sur la FPU → latence ≥ FP32 (paradoxe latence).",
    "f4_title": "INT8 v2 / Q15 : scale calibré CÔTÉ PC, puis exporté dans le header",
    "f4_delta": "delta clé vs PTQ legacy",
    "f4_delta_txt": "Le scale est calibré sur les données (per-tensor / per-channel, ou Q15 16 bits)\n"
                    "AVANT l'export : export_weights_c.py --int8-v2 / --maha-q15 écrit un header\n"
                    "déjà correct. Même firmware, même déquant FPU — mais métrique récupérée\n"
                    "(F1 ≈ FP32, Sprint 39 ; AUROC Q15 recouvrée, Sprint 34).",
    "f5_title": "Où chaque stratégie applique la quantification (◆) sur la même chaîne",
    "applied": "◆ quantification appliquée ici",
    "ram4": "RAM ÷4 (int8)",
    "ram2": "RAM ÷2 (Q15)",
    "note_names": "Étapes = vrais artefacts du dépôt (export_weights_c.py, model_weights*.h, "
                  "ewc_int8_from_fp32, pipeline.c). Schéma structurel, aucune donnée d'expérience.",
    "todo_dorra": "TODO(dorra) : valider le point d'application des scales per-channel du kernel v2.",
}


# ── Helpers de disposition ───────────────────────────────────────────────────

def _stage_row(
    ax: plt.Axes,
    y: float,
    stages: list[str],
    colors: list[str],
    x0: float = 0.06,
    x1: float = 0.94,
    box_h: float = 0.11,
    fontsize: int = 9,
) -> list[float]:
    """Trace une rangée horizontale de boîtes reliées ; retourne les x des centres."""
    n = len(stages)
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    bw = min(0.135, (x1 - x0) / n * 0.92)
    for i, (cx, txt, col) in enumerate(zip(xs, stages, colors)):
        box(ax, cx, y, txt, col, w=bw, h=box_h, fontsize=fontsize)
        if i:
            arrow(ax, xs[i - 1] + bw / 2 + 0.004, y, cx - bw / 2 - 0.004, y, color="#777777")
    return xs


def _spark(ax: plt.Axes, x: float, y: float, color: str, text: str) -> None:
    """Marqueur « ◆ quantification appliquée ici » sous une étape."""
    ax.annotate(
        text, xy=(x, y - 0.055), xytext=(x, y - 0.16),
        ha="center", va="top", fontsize=10, color=color, fontweight="bold",
        transform=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.2),
    )


def _callout(ax: plt.Axes, cx: float, cy: float, title: str, body: str,
             color: str, w: float = 0.42) -> None:
    """Encadré explicatif (titre coloré + corps) en coordonnées axes."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - 0.11), w, 0.22, boxstyle="round,pad=0.014",
        facecolor=mcolors.to_rgba(color, 0.08), edgecolor=color, linewidth=1.6,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(cx, cy + 0.075, title, ha="center", va="center", fontsize=11,
            color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(cx, cy - 0.025, body, ha="center", va="center", fontsize=9.5,
            color="#333333", transform=ax.transAxes)


def _base_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


# ── F1 — chaîne FP32 de référence ────────────────────────────────────────────

def _fig_f1() -> plt.Figure:
    c = STRATEGY_COLORS["fp32"]
    fig, ax = _base_fig()
    colors = [C_NEUTRAL] + [c] * (len(STAGES) - 1)
    xs = _stage_row(ax, 0.62, STAGES, colors)
    ax.text(xs[5], 0.62 + 0.10, "RAM ×1", ha="center", fontsize=10, color=c)
    ax.text(xs[6], 0.62 + 0.10, "float32 · FPU", ha="center", fontsize=10, color=c)
    _callout(ax, 0.5, 0.28, STRATEGY_LABELS_FR["fp32"], LABELS["f1_note"], c, w=0.82)
    ax.set_title(LABELS["f1_title"])
    footnote(fig, LABELS["note_names"])
    return fig


# ── F2 — QAT PC (quantif dans l'entraînement) ────────────────────────────────

def _fig_f2() -> plt.Figure:
    c = STRATEGY_COLORS["int8_qat"]
    fig, ax = _base_fig()
    # Étapes 3..7 (export→firmware→MAJ) grisées : hors périmètre (évaluation PC)
    colors = [C_NEUTRAL, c, c] + [C_OFF] * 5
    xs = _stage_row(ax, 0.66, STAGES, colors)
    _spark(ax, xs[1], 0.66, c, LABELS["applied"] + "\n(fake-quant, STE)")
    ax.text((xs[3] + xs[7]) / 2, 0.66 + 0.10,
            "hors périmètre (évaluation PC)", ha="center", fontsize=10,
            color="#90A4AE", style="italic")
    _callout(ax, 0.5, 0.24, LABELS["f2_box"], LABELS["f2_why"], c, w=0.86)
    ax.set_title(LABELS["f2_title"])
    footnote(fig, LABELS["note_names"])
    return fig


# ── F3 — PTQ legacy board ────────────────────────────────────────────────────

def _fig_f3() -> plt.Figure:
    c = STRATEGY_COLORS["int8_ptq_legacy"]
    fig, ax = _base_fig()
    # Entraînement FP32 classique, quantif au boot firmware (étape 5)
    colors = [C_NEUTRAL, C_NEUTRAL, C_NEUTRAL, C_NEUTRAL, C_NEUTRAL, c, c, c]
    xs = _stage_row(ax, 0.70, STAGES, colors)
    _spark(ax, xs[5], 0.70, c,
           LABELS["applied"] + "\newc_int8_from_fp32\n(échelle 1/128, acc int16)")
    ax.text(xs[5], 0.70 + 0.10, LABELS["ram4"], ha="center", fontsize=10, color=c)
    _callout(ax, 0.28, 0.24, LABELS["f3_collapse"], LABELS["f3_collapse_txt"], c, w=0.52)
    _callout(ax, 0.75, 0.24, LABELS["f3_slow"], LABELS["f3_slow_txt"],
             STRATEGY_COLORS["fp32"], w=0.42)
    ax.set_title(LABELS["f3_title"])
    footnote(fig, LABELS["note_names"])
    return fig


# ── F4 — v2 / Q15 (scale calibré côté PC) ────────────────────────────────────

def _fig_f4() -> plt.Figure:
    c = STRATEGY_COLORS["int8_v2"]
    c15 = STRATEGY_COLORS["q15"]
    fig, ax = _base_fig()
    # Calibration à l'export (étape 3), poids quantifiés dès le header (étape 4+)
    colors = [C_NEUTRAL, C_NEUTRAL, C_NEUTRAL, c, c, c, c, c]
    xs = _stage_row(ax, 0.70, STAGES, colors)
    _spark(ax, xs[3], 0.70, c,
           LABELS["applied"] + "\n--int8-v2 / --maha-q15\n(scale calibré per-tensor/-channel)")
    ax.text(xs[4], 0.70 + 0.10, f"{LABELS['ram4']} · {LABELS['ram2']}",
            ha="center", fontsize=9.5, color=c15)
    _callout(ax, 0.5, 0.24, LABELS["f4_delta"], LABELS["f4_delta_txt"], c, w=0.86)
    ax.set_title(LABELS["f4_title"])
    footnote(fig, LABELS["note_names"] + "  " + LABELS["todo_dorra"])
    return fig


# ── F5 — vue comparative 4 lignes ────────────────────────────────────────────

def _fig_f5() -> plt.Figure:
    fig, ax = _base_fig()
    # Étapes compactes pour la vue multi-lignes
    stages = ["Données", "Entraîn.\nPC", "Checkpt", "export_\nweights_c", "header\n.h",
              "Firmware\nRAM", "Forward\nFPU", "MAJ CL"]
    rows = [
        ("fp32", None, "float32 partout — référence"),
        ("int8_qat", 1, "fake-quant dans l'entraînement (métrique préservée ✓)"),
        ("int8_ptq_legacy", 5, "conversion figée au boot (métrique effondrée ✗)"),
        ("int8_v2", 3, "scale calibré à l'export (métrique récupérée ✓ ; Q15 idem)"),
    ]
    ys = [0.80, 0.585, 0.37, 0.155]
    for (strat, spark_i, caption), y in zip(rows, ys):
        c = STRATEGY_COLORS[strat]
        ax.text(0.005, y + 0.075, STRATEGY_LABELS_FR[strat], fontsize=11,
                color=c, fontweight="bold", transform=ax.transAxes)
        colors = [C_NEUTRAL if i == 0 else c for i in range(len(stages))]
        xs = _stage_row(ax, y, stages, colors, x0=0.20, x1=0.985,
                        box_h=0.075, fontsize=7.5)
        if spark_i is not None:
            ax.annotate("◆", xy=(xs[spark_i], y + 0.045), xytext=(xs[spark_i], y + 0.075),
                        ha="center", fontsize=15, color=c, transform=ax.transAxes)
        ax.text(0.20, y - 0.058, caption, fontsize=8.5, style="italic",
                color="#444444", transform=ax.transAxes)
    ax.text(0.20, 0.055, LABELS["applied"] + " · même chaîne, même firmware, même déquant FPU — "
            "seul le POINT d'application change.", fontsize=9.5, color="#333333",
            transform=ax.transAxes)
    ax.set_title(LABELS["f5_title"])
    footnote(fig, LABELS["note_names"])
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère F1–F5 sous ``out_root/quantization/pipeline/`` ; retourne les chemins."""
    return [
        savefig_png(_fig_f1(), CATALOG, "pipeline_fp32", out_root),
        savefig_png(_fig_f2(), CATALOG, "pipeline_int8_qat_pc", out_root),
        savefig_png(_fig_f3(), CATALOG, "pipeline_int8_ptq_board", out_root),
        savefig_png(_fig_f4(), CATALOG, "pipeline_int8_v2_q15", out_root),
        savefig_png(_fig_f5(), CATALOG, "pipeline_comparatif", out_root),
    ]
