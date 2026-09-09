"""Catalogue `quant_depth_board` — portage board sub-INT8 de la tête EWC (S4806).

Restitue les mesures **board réelles** (Sprint 48, NUCLEO-F439ZI) face à la RAM **théorique**
(Sprint 47, émulateur) — l'écart théorie↔matériel que l'émulateur ne pouvait pas donner.
5 figures régénérables, symétriques au catalogue PC ``quant_depth`` :

    board_auroc_vs_bits.png       AUROC board vs profondeur, 1 panneau/dataset (symétrique au PC).
    bss_packed_vs_unpacked.png    `.bss` mesuré non-packé (≈INT8) vs packé (÷2/÷4/÷8) — LE résultat clé.
    latency_vs_bits.png           latence DWT P50/P99 vs profondeur (coût du dépacking) ; ligne Gap 2.
    parity_board_pc.png           parité pred board↔émulateur par cellule (attendu 1.000).
    heatmap_board_bits_dataset.png heatmap AUROC board (dataset × profondeur) ; N/A gris.

**Toute valeur tracée provient d'un ``load_experiment``** (``experiments/exp_S48_summary.json``) —
aucun littéral numérique de résultat (garde AST ``test_no_hardcoded_results``). Cellules non
flashées → gris (jamais 0). Badges plateforme **mesuré board** vs **théorique PC**.

Source (lecture seule) : ``exp_S48_summary.json`` (S4805), lui-même agrégé de ``exp_S48_board/``
(mesuré) et ``exp_S47_depth/`` (théorique).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import A_MESURER, load_experiment
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, savefig_png

CATALOG = "quant_depth_board"
OUT_SUBDIR = "quant_depth_board"

DATASETS: list[str] = ["monitoring", "pronostia"]
GRANULARITY = "per_channel"
# Profondeurs portées board (S4801) : bits effectifs (entiers, structurels) + libellé FR.
BIT_ORDER: list[int] = [4, 2, 1]
BIT_LABEL_FR: dict[int, str] = {4: "4 b (INT4)", 2: "ternaire (2 b)", 1: "binaire (1 b)"}

COLOR_NONPACKED = STRATEGY_COLORS["int8_ptq_legacy"]   # conteneur int8 — pas de gain
COLOR_PACKED = STRATEGY_COLORS["int8_v2"]              # bit-packé — gain matérialisé
COLOR_P99 = STRATEGY_COLORS["q15"]
BADGE_BOARD = "mesuré board (NUCLEO-F439ZI)"
BADGE_PC = "théorique PC (émulateur S47)"
GAP2_US = 100_000

NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"


# ── Chargement de l'agrégat (unique source) ──────────────────────────────────

def _summary() -> dict | None:
    try:
        data, _ = load_experiment("experiments/exp_S48_summary.json")
    except FileNotFoundError:
        return None
    return data


def _cell(summary: dict | None, ds: str, bits: int) -> dict | None:
    """Cellule ``[dataset][bits][granularity]`` de l'agrégat, ou None."""
    if summary is None:
        return None
    node = summary.get("results_by_condition", {}).get(ds, {}).get(str(bits), {})
    return node.get(GRANULARITY)


def _num(value) -> float:
    """Valeur numérique ou NaN (sentinelle « à mesurer »/None → NaN, jamais 0)."""
    if value is None or value == A_MESURER:
        return float("nan")
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _board_leaf(cell: dict | None, packing: str, key: str) -> float:
    if cell is None:
        return float("nan")
    sub = (cell.get("board") or {}).get(packing) or {}
    return _num(sub.get(key))


def _pc_leaf(cell: dict | None, key: str) -> float:
    if cell is None:
        return float("nan")
    return _num((cell.get("pc") or {}).get(key))


# ── board_auroc_vs_bits — AUROC board (packé) vs profondeur ───────────────────

def _fig_auroc_vs_bits(summary: dict | None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11.0, 4.5), sharey=True)
    x = np.arange(len(BIT_ORDER))
    for ax, ds in zip(axes, DATASETS):
        board = [_board_leaf(_cell(summary, ds, b), "packed", "auroc_board") for b in BIT_ORDER]
        pc = [_pc_leaf(_cell(summary, ds, b), "auroc_quant") for b in BIT_ORDER]
        ax.plot(x, board, "-o", color=COLOR_PACKED, ms=8, label=BADGE_BOARD)
        ax.plot(x, pc, "--s", color=MUTED, ms=6, label=BADGE_PC)
        ax.set_xticks(x)
        ax.set_xticklabels([BIT_LABEL_FR[b] for b in BIT_ORDER], rotation=20, fontsize=8)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("profondeur des poids")
    axes[0].set_ylabel("AUROC")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("AUROC board vs profondeur — sub-INT8 EWC (mesuré board ∥ émulé PC)")
    fig.text(0.01, 0.005,
             "Source : exp_S48_summary.json. Board = NUCLEO-F439ZI (frozen) ; "
             "PC = émulateur subint8 (parité par construction). N/A = non flashé.",
             fontsize=8, color=MUTED)
    return fig


# ── bss_packed_vs_unpacked — LE résultat clé (théorie↔matériel) ──────────────

def _fig_bss(summary: dict | None) -> plt.Figure:
    cols = [(ds, b) for ds in DATASETS for b in BIT_ORDER]
    x = np.arange(len(cols))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    npk = [_board_leaf(_cell(summary, ds, b), "nonpacked", "bss_bytes") for ds, b in cols]
    pk = [_board_leaf(_cell(summary, ds, b), "packed", "bss_bytes") for ds, b in cols]
    ax.bar(x - width / 2, npk, width, color=COLOR_NONPACKED, edgecolor="white",
           label="non-packé (conteneur int8 ≈ INT8)")
    ax.bar(x + width / 2, pk, width, color=COLOR_PACKED, edgecolor="white",
           label="packé (bit-packé)")
    for xi, (ds, b) in zip(x, cols):
        for off, ys in ((-width / 2, npk), (width / 2, pk)):
            v = ys[cols.index((ds, b))]
            if np.isnan(v):
                ax.text(xi + off, 0.0, "N/A", ha="center", va="bottom", rotation=90,
                        fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds.capitalize()}\n{BIT_LABEL_FR[b]}" for ds, b in cols], fontsize=8)
    ax.set_ylabel(".bss firmware (octets)")
    ax.set_title("RAM .bss mesurée : non-packé (≈ INT8) vs packé (÷2/÷4/÷8) — le gain devient réel")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.005,
             "Source : exp_S48_summary.json (mesuré board). Le packing ne réduit que les "
             "matrices de poids ; l'overhead .bss fixe est partagé (écart théorie↔matériel).",
             fontsize=8, color=MUTED)
    return fig


# ── latency_vs_bits — DWT P50/P99 vs profondeur (coût du dépacking) ──────────

def _fig_latency(summary: dict | None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11.0, 4.5), sharey=True)
    x = np.arange(len(BIT_ORDER))
    for ax, ds in zip(axes, DATASETS):
        p50 = [_board_leaf(_cell(summary, ds, b), "packed", "latency_dwt_p50_us") for b in BIT_ORDER]
        p99 = [_board_leaf(_cell(summary, ds, b), "packed", "latency_dwt_p99_us") for b in BIT_ORDER]
        ax.plot(x, p50, "-o", color=COLOR_PACKED, ms=8, label="DWT P50 (packé)")
        ax.plot(x, p99, "--s", color=COLOR_P99, ms=6, label="DWT P99 (packé)")
        ax.set_xticks(x)
        ax.set_xticklabels([BIT_LABEL_FR[b] for b in BIT_ORDER], rotation=20, fontsize=8)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("profondeur des poids")
    axes[0].set_ylabel("latence DWT (µs)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Latence de dépacking sub-INT8 vs profondeur — board (Gap 2 : ≪ 100 ms)")
    fig.text(0.01, 0.005,
             f"Source : exp_S48_summary.json (DWT board). Budget Gap 2 = {GAP2_US} µs "
             "(100 ms) — les latences board sont deux ordres de grandeur en dessous.",
             fontsize=8, color=MUTED)
    return fig


# ── parity_board_pc — parité pred board↔émulateur par cellule ────────────────

def _fig_parity(summary: dict | None) -> plt.Figure:
    cols = [(ds, b, pk) for ds in DATASETS for b in BIT_ORDER for pk in ("nonpacked", "packed")]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    vals = [_board_leaf(_cell(summary, ds, b), pk, "parity_pred") for ds, b, pk in cols]
    colors = [COLOR_PACKED if pk == "packed" else COLOR_NONPACKED for _, _, pk in cols]
    ax.bar(x, vals, 0.6, color=colors, edgecolor="white")
    for xi, v in zip(x, vals):
        if np.isnan(v):
            ax.text(xi, 0.0, "N/A", ha="center", va="bottom", rotation=90, fontsize=8, color=MUTED)
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.0)
    ax.set_ylim(0.9, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds[:4].capitalize()}\n{BIT_LABEL_FR[b].split()[0]}\n{pk[:2]}"
                        for ds, b, pk in cols], fontsize=7)
    ax.set_ylabel("parité pred board↔PC")
    ax.set_title("Parité board↔émulateur par cellule sub-INT8 (attendu 1.000, schéma bit-identique)")
    fig.text(0.01, 0.005,
             "Source : exp_S48_summary.json. Le kernel firmware et l'émulateur partagent le "
             "schéma de quantification ⇒ parité exacte par construction.",
             fontsize=8, color=MUTED)
    return fig


# ── heatmap_board_bits_dataset — AUROC board (dataset × profondeur) ──────────

def _fig_heatmap(summary: dict | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    mat = np.full((len(DATASETS), len(BIT_ORDER)), np.nan)
    for i, ds in enumerate(DATASETS):
        for j, b in enumerate(BIT_ORDER):
            mat[i, j] = _board_leaf(_cell(summary, ds, b), "packed", "auroc_board")
    im = ax.imshow(mat, aspect="auto", cmap=plt.cm.RdYlGn, vmin=0.5, vmax=1.0)
    for i in range(len(DATASETS)):
        for j in range(len(BIT_ORDER)):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=NA_GRAY, zorder=2))
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color=MUTED, zorder=3)
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=INK, zorder=3)
    ax.set_xticks(np.arange(len(BIT_ORDER)))
    ax.set_xticklabels([BIT_LABEL_FR[b] for b in BIT_ORDER], fontsize=9)
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels([d.capitalize() for d in DATASETS])
    ax.set_title("AUROC board (packé) — dataset × profondeur (mesuré board)")
    fig.colorbar(im, ax=ax, label="AUROC board", fraction=0.05)
    fig.text(0.01, 0.005, "Source : exp_S48_summary.json. Cellules non flashées en gris (N/A).",
             fontsize=8, color=MUTED)
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les figures board sous ``out_root/quantization_depth_board/``.

    Skip gracieux si l'agrégat est absent (board pas encore streamée) — aucune valeur
    inventée, la galerie se remplit dès que ``exp_S48_summary.json`` existe.
    """
    summary = _summary()
    if summary is None:
        print("[figures] quant_depth_board : exp_S48_summary.json absent — skip gracieux.")
        return []
    return [
        savefig_png(_fig_auroc_vs_bits(summary), OUT_SUBDIR, "board_auroc_vs_bits", out_root),
        savefig_png(_fig_bss(summary), OUT_SUBDIR, "bss_packed_vs_unpacked", out_root),
        savefig_png(_fig_latency(summary), OUT_SUBDIR, "latency_vs_bits", out_root),
        savefig_png(_fig_parity(summary), OUT_SUBDIR, "parity_board_pc", out_root),
        savefig_png(_fig_heatmap(summary), OUT_SUBDIR, "heatmap_board_bits_dataset", out_root),
    ]
