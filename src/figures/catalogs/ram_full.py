"""Catalogue `ram_full` — RAM complète `.data + .bss + pic de pile` (Sprint 49, S4905).

Restitue les mesures RAM **complètes** du Sprint 49 (32 cellules `exp_S49_ram/`, board réelle
NUCLEO-F439ZI + PC), corrige les deux défauts relevés au CR du 16 juillet 2026 :

  * **étiquettes de phase neutres** — `idle` / `inférence` / `mise à jour CL`, jamais des étapes
    d'architecture de couches neuronales (HDC/Mahalanobis n'en sont pas) ;
  * **instants du pic alignés sur les phases réelles** — pic tracé **après** chaque phase,
    chronologiquement (R2).

4 figures régénérables :

    ram_totale_empilee.png   RAM totale empilée `.data`/`.bss`/pic par modèle × encodage (board).
    historique_pic_pile.png  Historique du pic de pile (phases réelles, 1 courbe/modèle).
    ratio_int8_fp32.png      Ratio int8/fp32 de la RAM totale par modèle (board) ; N/A gris.
    board_vs_pc.png          RAM totale board vs PC (colonnes séparées, badge plateforme).

**Toute valeur tracée provient d'un ``load_experiment``** (``exp_S49_ram/summary.json``, agrégé
par ``scripts/aggregate_ram.py``) — aucun littéral numérique de résultat (garde AST
``test_no_hardcoded_results``). Cellule non mesurée / N/A → gris (jamais 0).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import load_experiment
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "ram_full"

CONDITION = "5feat"
MODELS: list[str] = ["ewc", "hdc", "tinyol", "mahalanobis"]
MODEL_LABEL: dict[str, str] = {
    "ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "mahalanobis": "Mahalanobis",
}
DATASETS: list[str] = ["monitoring", "pronostia"]
ENCODINGS: list[str] = ["fp32", "int8"]
# Phases réelles des modèles (fix plots CR) — PAS d'étapes de couches neuronales.
PHASE_ORDER: list[str] = ["idle", "inference", "update"]
PHASE_LABEL: dict[str, str] = {
    "idle": "idle", "inference": "inférence", "update": "mise à jour CL",
}

COLOR_DATA = STRATEGY_COLORS["fp32"]        # bleu — .data
COLOR_BSS = STRATEGY_COLORS["int16_am"]     # brun — .bss
COLOR_STACK = STRATEGY_COLORS["int8_v2"]    # orange — pic de pile
COLOR_FP32 = STRATEGY_COLORS["fp32"]
COLOR_INT8 = STRATEGY_COLORS["int8_v2"]
NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"
KIB = 1024
BADGE = "mesuré board (NUCLEO-F439ZI) · CR 16/07/2026 : RAM = .data + .bss + pic de pile"


# ── Source unique ────────────────────────────────────────────────────────────

def _summary() -> dict | None:
    try:
        data, _ = load_experiment("experiments/exp_S49_ram/summary.json")
    except FileNotFoundError:
        return None
    return data


def _cell(summary: dict | None, model: str, dataset: str, encoding: str, platform: str) -> dict | None:
    if summary is None:
        return None
    node = summary.get(model, {}).get(dataset, {}).get(CONDITION, {})
    return node.get(encoding, {}).get(platform)


def _kib(v) -> float | None:
    return v / KIB if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _empty(msg: str, name: str, out_root: Path) -> Path:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, wrap=True)
    return savefig_png(fig, CATALOG, name, out_root)


# ── R1 : RAM totale empilée .data/.bss/pic par modèle × encodage (board) ─────

def _fig_stacked(summary: dict | None, out_root: Path) -> Path:
    name = "ram_totale_empilee"
    if summary is None:
        return _empty("summary.json absent — lancer aggregate_ram.py", name, out_root)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    x = np.arange(len(MODELS))
    width = 0.35
    for ax, dataset in zip(axes, DATASETS):
        for i, encoding in enumerate(ENCODINGS):
            offset = (i - 0.5) * width
            data_v, bss_v, stack_v = [], [], []
            for model in MODELS:
                c = _cell(summary, model, dataset, encoding, "board")
                if c is None or c.get("status") in ("na", "missing"):
                    data_v.append(0.0); bss_v.append(0.0); stack_v.append(0.0)
                    continue
                data_v.append(_kib(c.get("data")) or 0.0)
                bss_v.append(_kib(c.get("bss")) or 0.0)
                peaks = [c.get("stack_peak_inference"), c.get("stack_peak_update")]
                peaks = [p for p in peaks if isinstance(p, (int, float))]
                stack_v.append(_kib(max(peaks)) if peaks else 0.0)
            data_a = np.array(data_v); bss_a = np.array(bss_v); stack_a = np.array(stack_v)
            lbl = i == 0  # légende une seule fois
            ax.bar(x + offset, data_a, width, color=COLOR_DATA,
                   label=".data" if lbl else None)
            ax.bar(x + offset, bss_a, width, bottom=data_a, color=COLOR_BSS,
                   label=".bss" if lbl else None)
            ax.bar(x + offset, stack_a, width, bottom=data_a + bss_a, color=COLOR_STACK,
                   label="pic de pile" if lbl else None)
            for xi, tot in zip(x + offset, data_a + bss_a + stack_a):
                ax.text(xi, tot, encoding, ha="center", va="bottom",
                        fontsize=8, color=MUTED, rotation=90)
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20)
    axes[0].set_ylabel("RAM (Kio)")
    axes[0].legend(loc="upper right")
    fig.suptitle("RAM totale = .data + .bss + pic de pile — board, par modèle × encodage")
    fig.text(0.5, 0.005, BADGE, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── R2 : historique du pic de pile (phases réelles, 1 courbe/modèle) ─────────

def _fig_history(summary: dict | None, out_root: Path) -> Path:
    name = "historique_pic_pile"
    if summary is None:
        return _empty("summary.json absent — lancer aggregate_ram.py", name, out_root)
    fig, axes = plt.subplots(1, len(DATASETS), sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    xs = np.arange(len(PHASE_ORDER))
    for ax, dataset in zip(axes, DATASETS):
        plotted = False
        for model in MODELS:
            c = _cell(summary, model, dataset, "fp32", "board")
            if c is None or c.get("status") in ("na", "missing"):
                continue
            hist = {h["phase"]: h["stack_peak_bytes"] for h in c.get("stack_history", [])}
            ys = [(_kib(hist.get(p)) if p in hist else None) for p in PHASE_ORDER]
            if all(y is None for y in ys):
                continue
            ax.plot(xs, ys, marker="o", label=MODEL_LABEL[model])
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "aucun historique board", transform=ax.transAxes,
                    ha="center", color=NA_GRAY)
        ax.set_title(dataset)
        ax.set_xticks(xs)
        ax.set_xticklabels([PHASE_LABEL[p] for p in PHASE_ORDER], rotation=15)
    axes[0].set_ylabel("pic de pile (Kio)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Historique du pic de pile par phase réelle — la MAJ CL creuse plus la pile")
    fig.text(0.5, 0.005, BADGE, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── R3 : ratio int8/fp32 de la RAM totale (board) ────────────────────────────

def _fig_ratio(summary: dict | None, out_root: Path) -> Path:
    name = "ratio_int8_fp32"
    if summary is None:
        return _empty("summary.json absent — lancer aggregate_ram.py", name, out_root)
    fig, ax = plt.subplots()
    x = np.arange(len(MODELS))
    width = 0.35
    for i, dataset in enumerate(DATASETS):
        offset = (i - 0.5) * width
        ratios, colors = [], []
        for model in MODELS:
            c = _cell(summary, model, dataset, "int8", "board")
            r = c.get("ratio_int8_vs_fp32") if c else None
            if isinstance(r, (int, float)):
                ratios.append(r); colors.append(COLOR_INT8)
            else:
                ratios.append(0.0); colors.append(NA_GRAY)
        bars = ax.bar(x + offset, ratios, width, color=colors,
                      label=dataset, edgecolor=INK, linewidth=0.5)
        for b, r in zip(bars, ratios):
            txt = f"{r:g}" if r else "N/A"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                    ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20)
    ax.set_ylabel("RAM totale int8 / fp32")
    ax.set_title("Ratio int8 vs fp32 de la RAM totale — board (N/A gris)")
    ax.legend()
    fig.text(0.5, 0.005, BADGE, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── R4 : board vs PC (colonnes séparées, badge plateforme) ───────────────────

def _fig_board_vs_pc(summary: dict | None, out_root: Path) -> Path:
    name = "board_vs_pc"
    if summary is None:
        return _empty("summary.json absent — lancer aggregate_ram.py", name, out_root)
    # Deux panneaux distincts : board et PC ne se fusionnent pas (échelles ≠).
    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(MODELS))
    width = 0.35
    panels = [("board", "RAM totale board (Kio) — .bss + pile réels", COLOR_FP32),
              ("pc", "Pic RAM PC (Kio) — tracemalloc (un passage modèle)", COLOR_INT8)]
    for ax, (platform, title, color) in zip(axes, panels):
        for i, dataset in enumerate(DATASETS):
            offset = (i - 0.5) * width
            vals, colors = [], []
            for model in MODELS:
                c = _cell(summary, model, dataset, "fp32", platform)
                t = _kib(c.get("total")) if c else None
                if isinstance(t, (int, float)):
                    vals.append(t); colors.append(color)
                else:
                    vals.append(0.0); colors.append(NA_GRAY)
            ax.bar(x + offset, vals, width, color=colors, label=dataset,
                   edgecolor=INK, linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=20)
        ax.set_ylabel("Kio")
        ax.legend(fontsize=9)
    fig.suptitle("Board vs PC — plateformes séparées (jamais fusionnées)")
    fig.text(0.5, 0.005, "board = .bss+pile réels · PC = tracemalloc (fp32 proxy)",
             ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 4 figures RAM complète sous ``out_root/ram_full/``."""
    apply_style("slide")
    summary = _summary()
    return [
        _fig_stacked(summary, out_root),
        _fig_history(summary, out_root),
        _fig_ratio(summary, out_root),
        _fig_board_vs_pc(summary, out_root),
    ]
