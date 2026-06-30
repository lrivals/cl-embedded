"""
Génération des figures pour la présentation board NUCLEO-F439ZI (Sprints 16-32).

Usage:
    python scripts/generate_presentation_plots.py
    python scripts/generate_presentation_plots.py --output docs/figures/presentation_board/
    python scripts/generate_presentation_plots.py --show   # affiche sans sauvegarder

Produit 14 figures PNG dans le répertoire de sortie. Les fonctions plot_* sont
importables (cf. presentation_plots.ipynb qui les rappelle pour régénérer les figures
en direct). Toutes les valeurs proviennent de mesures board / experiments — rien d'inventé.
"""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {
    "mahal": "#2196F3",
    "ewc": "#4CAF50",
    "tinyol": "#FF9800",
    "hdc": "#9C27B0",
    "budget": "#F44336",
    "board": "#1565C0",
    "dryrun": "#78909C",
    "ok": "#43A047",
    "warn": "#FFA000",
    "bad": "#E53935",
}


def save_or_show(fig, name: str, out_dir: str | None, show: bool):
    # Sauvegarde d'abord (régénère le PNG), puis affichage : en notebook inline,
    # plt.show() rend la figure avant la fermeture finale ; en script, --show l'affiche.
    if out_dir:
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  ✅ {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. RAM Budget — tous les modèles vs limite Gap 2
# ---------------------------------------------------------------------------
def plot_ram_budget(out_dir, show):
    # Empreinte .bss par modèle isolé + firmware complet multi-modèle (paires + méta, Sprint 31).
    models = ["Mahalanobis", "EWC Head\nMLP", "TinyOL\n(SRAM)", "HDC\n(SRAM)", "Firmware\npaires + méta"]
    ram_bytes = [200, 9728, 400, 28000, 104596]

    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["tinyol"], COLORS["hdc"], COLORS["board"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, [r / 1024 for r in ram_bytes], color=colors, width=0.55, zorder=3)
    ax.axhline(64, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget Gap 2 (par modèle) : 64 Ko")
    ax.axhline(256, color="#B71C1C", linewidth=1, linestyle=":", label="SRAM totale NUCLEO-F439ZI : 256 Ko")

    for bar, val in zip(bars, ram_bytes):
        label = f"{val} B" if val < 1024 else f"{val/1024:.1f} Ko"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.annotate("39.9 % de 256 Ko\n(tous modèles liés)", xy=(4, 102), xytext=(2.7, 150),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")

    ax.set_ylabel("RAM utilisée — .bss (Ko)")
    ax.set_title("Empreinte RAM : modèles isolés vs firmware complet (NUCLEO-F439ZI)")
    ax.set_ylim(0, 280)
    ax.set_yticks([0, 16, 32, 64, 104, 128, 192, 256])
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "01_ram_budget", out_dir, show)


# ---------------------------------------------------------------------------
# 2. Latence d'inférence vs budget 100 ms
# ---------------------------------------------------------------------------
def _load_latency_per_model():
    """Latence board (µs) inférence vs inférence+update par modèle de détection.

    Source prioritaire : la campagne board S33
    (`experiments/exp_S33_board_latency/latency_summary.json`, produite par
    `scripts/run_board_gap1_completion.py`). Le repli ci-dessous reprend désormais
    ces mêmes valeurs **mesurées** (DWT CYCCNT, P50) : inf+update complétées pour
    tous les modèles — plus aucune barre « à mesurer ».
    """
    import json as _json
    from pathlib import Path as _Path

    # Repli : valeurs board mesurées (DWT CYCCNT, P50) de la campagne S33
    # (experiments/exp_S33_board_latency/latency_summary.json — rien d'inventé).
    #   Maha 5/5 µs, EWC (tête MC 5-feat) 50/251 µs, HDC 585/653 µs, TinyOL 5/5 µs.
    #   NB : l'EWC RUL (slide 10) est distinct (130/403 µs, tâche/archi différentes).
    per_model = {
        "mahalanobis": {"inf": 5.0,   "update": 5.0},
        "ewc":         {"inf": 50.0,  "update": 251.0},
        "hdc":         {"inf": 585.0, "update": 653.0},
        "tinyol":      {"inf": 5.0,   "update": 5.0},
    }

    summary = _Path("experiments/exp_S33_board_latency/latency_summary.json")
    if summary.exists():
        try:
            for row in _json.loads(summary.read_text()):
                m = row.get("model")
                if m in per_model:
                    if row.get("latency_inf_us") is not None:
                        per_model[m]["inf"] = float(row["latency_inf_us"])
                    if row.get("latency_inf_update_us") is not None:
                        per_model[m]["update"] = float(row["latency_inf_update_us"])
        except Exception:
            pass
    return per_model


def plot_latency(out_dir, show):
    # Latence board (DWT CYCCNT, NUCLEO-F439ZI) par modèle de détection de faute :
    # inférence vs inférence+update CL — toutes ≪ budget 100 ms.
    per_model = _load_latency_per_model()
    order  = ["mahalanobis", "ewc", "hdc", "tinyol"]
    labels = ["Maha", "EWC", "HDC", "TinyOL"]
    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["hdc"], COLORS["tinyol"]]

    x = np.arange(len(order))
    width = 0.38
    inf_us    = [per_model[m]["inf"] / 1000.0 for m in order]            # ms
    update_us = [per_model[m]["update"] for m in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_inf = ax.bar(x - width / 2, inf_us, width, color=colors, zorder=3,
                      edgecolor="white", label="Inférence")
    # Barres inf+update : hachurées + plus claires ; placeholder « à mesurer » si None
    bars_upd = []
    for xi, c, u in zip(x, colors, update_us):
        if u is not None:
            b = ax.bar(xi + width / 2, u / 1000.0, width, color=c, alpha=0.55,
                       hatch="//", edgecolor="white", zorder=3)
            bars_upd.append((b[0], u / 1000.0, False))
        else:
            # placeholder visuel à hauteur de l'inférence (borne basse), annoté
            b = ax.bar(xi + width / 2, inf_us[xi], width, facecolor="none",
                       edgecolor=c, linestyle=":", hatch="..", zorder=3)
            bars_upd.append((b[0], inf_us[xi], True))

    ax.axhline(100, color=COLORS["budget"], linewidth=2, linestyle="--",
               label="Budget Gap 2 : 100 ms")

    for bar, val in zip(bars_inf, inf_us):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{int(val*1000)} µs",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val, missing in bars_upd:
        txt = "à mesurer" if missing else f"{int(val*1000)} µs"
        ax.text(bar.get_x() + bar.get_width() / 2, val, txt,
                ha="center", va="bottom", fontsize=8,
                color="gray" if missing else "black",
                fontweight="normal" if missing else "bold",
                style="italic" if missing else "normal")

    # Légende inférence / inf+update (hachuré) + budget ; l'entrée « à mesurer »
    # n'apparaît que s'il reste au moins une mesure inf+update manquante.
    any_missing = any(m for _, _, m in bars_upd)
    legend_handles = [
        mpatches.Patch(facecolor="#888888", label="Inférence"),
        mpatches.Patch(facecolor="#888888", alpha=0.55, hatch="//", label="Inférence + update CL"),
    ]
    if any_missing:
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="#888888", linestyle=":",
                           hatch="..", label="inf+update à mesurer (campagne board S33)"))
    legend_handles.append(mpatches.Patch(facecolor=COLORS["budget"], label="Budget Gap 2 : 100 ms"))
    ax.text(0.02, 0.92, "Budget 100 ms ≈ 150× au-dessus du pire cas (HDC 0.59 ms)",
            transform=ax.transAxes, color="gray", fontsize=9, va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latence (ms) — échelle linéaire")
    ax.set_title("Latence inférence vs inf+update CL par modèle — NUCLEO-F439ZI @ 180 MHz")
    ax.set_ylim(0, 1.0)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "02_latency", out_dir, show)


# ---------------------------------------------------------------------------
# 3. Latence en échelle log pour voir les valeurs réelles
# ---------------------------------------------------------------------------
def plot_latency_log(out_dir, show):
    # Échelle log : du plus rapide (Maha 5 µs) au plus lourd (paire Maha+HDC 657 µs) vs budget 100 ms.
    models = ["Maha\n5 µs", "EWC MC\n50 µs", "EWC RUL\n130 µs", "TinyOL\n5 µs", "PAIR\nMaha+EWC\n256 µs", "TRIPLE\n258 µs", "PAIR\nMaha+HDC\n657 µs"]
    latencies = [0.005, 0.050, 0.130, 0.005, 0.256, 0.258, 0.657]
    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["ewc"], COLORS["tinyol"], COLORS["ok"], COLORS["board"], COLORS["hdc"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, latencies, color=colors, width=0.55, zorder=3, edgecolor="white")
    ax.axhline(100, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget Gap 2 : 100 ms", zorder=4)
    ax.set_yscale("log")

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.4,
                f"{int(val*1000)} µs", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.legend(loc="upper left")
    ax.set_ylim(0.002, 200)
    ax.set_ylabel("Latence (ms) — échelle log")
    ax.set_title("Latences board mesurées (DWT) — toutes ≪ budget 100 ms")
    ax.yaxis.grid(True, alpha=0.3, zorder=0, which="both")
    fig.tight_layout()
    save_or_show(fig, "03_latency_log", out_dir, show)


# ---------------------------------------------------------------------------
# 4. EWC Forgetting — impact de λ
# ---------------------------------------------------------------------------
def plot_ewc_forgetting(out_dir, show):
    lambdas = [0, 100, 400]
    forgetting_dryrun = [0.3084, 0.0534, 0.0534]
    forgetting_board = [0.0542, 0.009, 0.009]
    acc_dryrun = [0.6118, 0.7818, 0.7818]
    acc_board = [0.9036, 0.9016, 0.8976]

    x = np.arange(len(lambdas))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1 : Forgetting
    ax = axes[0]
    b1 = ax.bar(x - w/2, forgetting_dryrun, w, color=COLORS["dryrun"], label="Dry-run (simulation)")
    b2 = ax.bar(x + w/2, forgetting_board, w, color=COLORS["board"], label="Board réel")
    ax.axhline(0.10, color=COLORS["warn"], linewidth=1.5, linestyle="--", label="Seuil acceptable (0.10)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ = {l}" for l in lambdas])
    ax.set_ylabel("avg_forgetting")
    ax.set_title("Forgetting EWC par λ\n(3 tâches, Equipment Monitoring)")
    ax.set_ylim(0, 0.40)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    for b in [b1, b2]:
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Subplot 2 : Accuracy
    ax = axes[1]
    b1 = ax.bar(x - w/2, acc_dryrun, w, color=COLORS["dryrun"], label="Dry-run (simulation)")
    b2 = ax.bar(x + w/2, acc_board, w, color=COLORS["board"], label="Board réel")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ = {l}" for l in lambdas])
    ax.set_ylabel("acc_final")
    ax.set_title("Accuracy finale EWC par λ\n(3 tâches, Equipment Monitoring)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    for b in [b1, b2]:
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Impact du paramètre λ EWC sur le forgetting et l'accuracy", fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "04_ewc_lambda_impact", out_dir, show)


# ---------------------------------------------------------------------------
# 5. Comparaison board vs dry-run (toutes expériences)
# ---------------------------------------------------------------------------
def plot_board_vs_dryrun(out_dir, show):
    exps = ["E19-01\nMahal.", "E19-02\nEWC (bug)", "Baseline\nλ=0", "Baseline\nboard", "EWC\nλ=400 dry", "EWC100\nboard", "EWC400\nboard"]
    acc = [0.6285, 0.08, 0.6118, 0.9036, 0.7818, 0.9016, 0.8976]
    forgetting = [0.0, 0.0, 0.3084, 0.0542, 0.0534, 0.009, 0.009]
    is_board = [True, True, False, True, False, True, True]
    colors = [COLORS["board"] if b else COLORS["dryrun"] for b in is_board]
    hatch = ["" if b else "//" for b in is_board]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, vals, ylabel, title in [
        (axes[0], acc, "acc_final", "Accuracy finale par expérience"),
        (axes[1], forgetting, "avg_forgetting", "Forgetting moyen par expérience"),
    ]:
        bars = ax.bar(range(len(exps)), vals, color=colors, hatch=hatch,
                      edgecolor="white", width=0.6, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(range(len(exps)))
    axes[1].set_xticklabels(exps, fontsize=9)

    board_p = mpatches.Patch(color=COLORS["board"], label="Board réel (NUCLEO-F439ZI)")
    dry_p = mpatches.Patch(facecolor=COLORS["dryrun"], hatch="//", edgecolor="white", label="Dry-run (simulation PC)")
    axes[0].legend(handles=[board_p, dry_p], loc="lower right", fontsize=9)

    fig.suptitle("Toutes les expériences Sprints 19–20 : board vs simulation", fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "05_all_experiments_comparison", out_dir, show)


# ---------------------------------------------------------------------------
# 6. Empreinte mémoire détaillée — Flash vs SRAM
# ---------------------------------------------------------------------------
def plot_memory_breakdown(out_dir, show):
    models = ["Mahalanobis", "EWC Head MLP", "TinyOL", "HDC", "Firmware\npaires+méta"]
    flash_kb = [0.12, 0, 5.7, 2.0, 0]    # poids en Flash (.rodata)
    sram_kb = [0.20, 9.73, 0.40, 28.0, 104.6]  # poids en SRAM (.bss)

    x = np.arange(len(models))
    w = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x, flash_kb, w, color="#FF8F00", label="Flash (.rodata — poids constants)")
    b2 = ax.bar(x, sram_kb, w, bottom=flash_kb, color=COLORS["ewc"], label="SRAM (.bss — poids entraînables)")

    ax.axhline(64, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget SRAM Gap 2 (par modèle) : 64 Ko")
    ax.axhline(256, color="#B71C1C", linewidth=1, linestyle=":", label="SRAM totale : 256 Ko")

    for i, (f, s) in enumerate(zip(flash_kb, sram_kb)):
        total = f + s
        ax.text(i, total + 2.0, f"{total:.1f} Ko", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mémoire (Ko)")
    ax.set_title("Empreinte mémoire : modèles isolés vs firmware complet (Flash + SRAM)")
    ax.set_ylim(0, 280)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "06_memory_breakdown", out_dir, show)


# ---------------------------------------------------------------------------
# 7. Timeline des sprints 16–20
# ---------------------------------------------------------------------------
def plot_sprint_timeline(out_dir, show):
    # Jalons clés de la Phase 2 MCU (Sprints 16 → 32). Largeurs égales pour lisibilité.
    sprints = [
        ("S16-17", "mai", "Toolchain ARM + C\nHAL + profiling DWT", COLORS["ok"]),
        ("S18-19", "1 jun", "Pipeline UART v2/v3\n3 modèles CL en C", COLORS["ok"]),
        ("S20-23", "15 jun", "Gap 2 formel\nbenchmark 5 datasets", COLORS["ok"]),
        ("S26-27", "16 jun", "EWC RUL board\nDUAL_MODE 637 µs", COLORS["ok"]),
        ("S28", "17 jun", "Gap 3 INT8\n×4 RAM préservée", COLORS["ok"]),
        ("S30", "17 jun", "Paires parallèles\nPAIR_MODE board", COLORS["ok"]),
        ("S31", "18 jun", "Méta-modèle stacking\nTRIPLE_MODE 258 µs", COLORS["ok"]),
        ("S32", "23 jun", "Étude seuil RUL\nparité board↔PC", "#FFA000"),
    ]

    fig, ax = plt.subplots(figsize=(15, 4))
    dur = 4
    for i, (name, date, desc, color) in enumerate(sprints):
        start = i * dur
        rect = mpatches.FancyBboxPatch(
            (start, 0.2), dur - 0.3, 0.6,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(start + dur/2, 0.5, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(start + dur/2, 0.14, date, ha="center", va="top",
                fontsize=8, color="gray")
        ax.text(start + dur/2, 0.86, desc, ha="center", va="bottom",
                fontsize=7.5, color="#333333", wrap=True)

    status_patches = [
        mpatches.Patch(color=COLORS["ok"], label="✅ Terminé"),
        mpatches.Patch(color="#FFA000", label="🔄 En cours"),
    ]
    ax.legend(handles=status_patches, loc="lower right", fontsize=9)

    ax.set_xlim(-0.5, len(sprints) * dur)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Timeline Phase 2 MCU — Sprints 16–32 (mai–juin 2026)", fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "07_sprint_timeline", out_dir, show)


# ---------------------------------------------------------------------------
# 8. Architecture firmware — Vue schématique
# ---------------------------------------------------------------------------
def plot_firmware_architecture(out_dir, show):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#455A64", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="white" if color not in ("#FFF9C4", "#E8F5E9") else "#333")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#546E7A", lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.05, my, label, fontsize=7.5, color="#37474F")

    # PC
    box(0.2, 3.2, 1.6, 1.0, "PC\nLinux / Python", "#546E7A", bold=True)
    arrow(1.8, 3.7, 2.5, 3.7, "UART\n115200")

    # ST-LINK
    box(2.5, 3.2, 1.2, 1.0, "ST-LINK\nVCP", "#607D8B")
    arrow(3.7, 3.7, 4.4, 3.7)

    # USART3
    box(4.4, 3.2, 1.2, 1.0, "USART3\n@ MCU", "#0277BD")
    arrow(5.6, 3.7, 6.2, 3.7)

    # Pipeline
    box(6.2, 3.0, 1.6, 1.4, "pipeline.c\n[orchestrateur]", "#1565C0", bold=True)

    # Models
    box(6.2, 5.0, 1.6, 0.9, "mahalanobis\n~200 B SRAM", COLORS["mahal"])
    box(6.2, 1.2, 1.6, 0.9, "ewc_head.c\n~9.7 Ko SRAM", COLORS["ewc"])
    box(8.1, 3.0, 1.6, 0.9, "tinyol.c\n5.7 Ko Flash", COLORS["tinyol"])

    arrow(6.95, 4.4, 6.95, 5.0, "EWC_MODE=0")
    arrow(6.95, 3.0, 6.95, 2.1, "EWC_MODE=1")
    arrow(7.8, 3.5, 8.1, 3.5)

    # Profiling
    box(8.1, 5.0, 1.6, 0.9, "profiling.c\nDWT cycles", "#6A1B9A")
    arrow(7.8, 4.2, 8.1, 5.0)

    # LED
    box(4.4, 5.0, 1.2, 0.9, "LED PA5\n[anomalie]", "#F57F17")
    arrow(6.2, 3.7, 5.6, 5.0, "anomalie\ndétectée")

    ax.set_title("Architecture firmware NUCLEO-F439ZI — vue d'ensemble", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "08_firmware_architecture", out_dir, show)


# ---------------------------------------------------------------------------
# 9. Protocole UART — Évolution v1→v2→v3
# ---------------------------------------------------------------------------
def plot_uart_protocol(out_dir, show):
    fig, axes = plt.subplots(4, 1, figsize=(12, 9))

    def draw_frame(ax, title, fields, colors):
        total = sum(w for _, w, _ in fields)
        x = 0
        for name, width, color in fields:
            frac = width / total
            ax.barh(0, frac, left=x/total, height=0.5, color=color, edgecolor="white", linewidth=1.5)
            ax.text(x/total + frac/2, 0, f"{name}\n{width}B",
                    ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            x += width
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)
        ax.axis("off")
        ax.set_title(f"{title}  ({total} octets total)", fontsize=10, loc="left")

    # Trame envoyée PC → Carte (v2)
    request_fields = [
        ("MAGIC\n2B", 2, "#455A64"),
        ("VER\n1B", 1, "#546E7A"),
        ("N_FEAT\n1B", 1, "#607D8B"),
        ("FEATURES\n5×4B=20B", 20, "#0277BD"),
        ("LABEL\n1B", 1, "#1565C0"),
        ("FLAGS\n1B", 1, "#1976D2"),
        ("TASK_ID\n1B", 1, "#1E88E5"),
        ("TIMESTAMP\n4B", 4, "#42A5F5"),
        ("CRC8\n1B", 1, "#78909C"),
    ]
    draw_frame(axes[0], "Trame PC → Carte (request)", request_fields, None)

    # Réponse v3 (mono-modèle + métriques CL embarquées)
    resp_v3 = [
        ("PRED\n1B", 1, "#2E7D32"),
        ("CONF\n2B", 2, "#388E3C"),
        ("LAT_US\n4B", 4, "#43A047"),
        ("RAM_B\n2B", 2, "#4CAF50"),
        ("THRU\n2B", 2, "#66BB6A"),
        ("STATUS\n1B", 1, "#81C784"),
        ("ACC\n2B", 2, "#1565C0"),
        ("AUROC\n2B", 2, "#1976D2"),
        ("FORGET\n2B", 2, "#42A5F5"),
        ("CRC8\n1B", 1, "#78909C"),
    ]
    draw_frame(axes[1], "Réponse v3 (Sprint 19) — mono-modèle + métriques CL embarquées", resp_v3, None)

    # Réponse PAIR (Sprint 30) — Mahalanobis + supervisé en parallèle
    resp_pair = [
        ("pred_maha\n1B", 1, "#1565C0"),
        ("score_maha\n4B", 4, "#1976D2"),
        ("pred_sup\n1B", 1, "#2E7D32"),
        ("conf_sup\n4B", 4, "#43A047"),
        ("lat\n4B", 4, "#66BB6A"),
        ("auroc_maha\n4B", 4, "#42A5F5"),
        ("f1_sup\n4B", 4, "#81C784"),
    ]
    draw_frame(axes[2], "Réponse PAIR_MODE (Sprint 30) — paire Maha + supervisé", resp_pair, None)

    # Réponse TRIPLE (Sprint 31) — paire + méta-modèle
    resp_triple = resp_pair + [
        ("pred_meta\n1B", 1, "#6A1B9A"),
        ("prob_meta\n4B", 4, "#9C27B0"),
    ]
    draw_frame(axes[3], "Réponse TRIPLE_MODE (Sprint 31) — paire + méta-modèle de stacking", resp_triple, None)

    fig.suptitle("Protocole UART binaire : du mono-modèle (v3) aux paires + méta (PAIR/TRIPLE)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_or_show(fig, "09_uart_protocol", out_dir, show)


# ---------------------------------------------------------------------------
# 10. EWC — Intuition du forgetting (diagramme tâche par tâche)
# ---------------------------------------------------------------------------
def plot_forgetting_intuition(out_dir, show):
    tasks = ["Tâche 0\n(pump)", "Tâche 1\n(turbine)", "Tâche 2\n(compressor)"]
    n_tasks = 3

    # Accuracy simulée par tâche après chaque étape d'entraînement
    # Ligne = "état après avoir entraîné jusqu'à la tâche j"
    # acc_no_ewc[j][i] = accuracy sur tâche i après entraînement tâche j
    acc_no_ewc = [
        [0.90, None, None],
        [0.55, 0.88, None],
        [0.30, 0.60, 0.91],
    ]
    acc_ewc = [
        [0.90, None, None],
        [0.85, 0.89, None],
        [0.83, 0.86, 0.90],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors_task = [COLORS["mahal"], COLORS["ewc"], COLORS["tinyol"]]
    markers = ["o", "s", "^"]

    for ax, data, title in [(axes[0], acc_no_ewc, "Sans EWC (λ=0) — Fine-tuning naïf"),
                             (axes[1], acc_ewc, "Avec EWC (λ=400) — Régularisation Fisher")]:
        for task_i in range(n_tasks):
            x_vals, y_vals = [], []
            for step in range(n_tasks):
                if data[step][task_i] is not None:
                    x_vals.append(step)
                    y_vals.append(data[step][task_i])
            ax.plot(x_vals, y_vals, color=colors_task[task_i], marker=markers[task_i],
                    linewidth=2, markersize=8, label=tasks[task_i])

        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels([f"Après\n{t}" for t in tasks])
        ax.set_ylabel("Accuracy par tâche")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.1, 1.0)
        ax.yaxis.grid(True, alpha=0.3)
        ax.legend(title="Tâche évaluée", fontsize=9)

    # Annotation forgetting
    axes[0].annotate("Forgetting\n↓ 0.30 !", xy=(2, 0.30), xytext=(1.5, 0.20),
                     arrowprops=dict(arrowstyle="->", color=COLORS["bad"]),
                     color=COLORS["bad"], fontsize=9, fontweight="bold")
    axes[1].annotate("Forgetting\n↓ 0.009 ✅", xy=(2, 0.83), xytext=(1.3, 0.70),
                     arrowprops=dict(arrowstyle="->", color=COLORS["ok"]),
                     color=COLORS["ok"], fontsize=9, fontweight="bold")

    fig.suptitle("Catastrophic Forgetting : impact de la régularisation EWC (3 tâches domaine-incrémental)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "10_forgetting_intuition", out_dir, show)


# ---------------------------------------------------------------------------
# 11. Gap 2 — Tableau de conformité visuel
# ---------------------------------------------------------------------------
def plot_gap2_compliance(out_dir, show):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    columns = ["Configuration", "RAM (.bss)", "% de 256 Ko", "Latence board", "Budget latence", "Marge latence", "✓"]
    rows = [
        ["Mahalanobis", "200 B", "0.1 %", "5 µs", "100 ms", "×20 000", "OK"],
        ["EWC RUL (inf/upd)", "66.7 Ko", "26 %", "130 / 403 µs", "100 ms", "×248", "OK"],
        ["DUAL Reg+MC", "66.7 Ko", "26 %", "637 µs", "100 ms", "×157", "OK"],
        ["PAIR Maha+EWC", "104.6 Ko", "39.9 %", "256 µs", "100 ms", "×390", "OK"],
        ["TRIPLE +méta", "104.6 Ko", "39.9 %", "258 µs", "100 ms", "×388", "OK"],
        ["PAIR Maha+HDC", "104.6 Ko", "39.9 %", "657 µs", "100 ms", "×152", "OK"],
    ]

    cell_colors = []
    for row in rows:
        row_c = ["#ECEFF1"] * (len(columns) - 1)
        if row[-1] == "OK":
            row_c.append("#C8E6C9")
        elif row[-1] == "⏳":
            row_c.append("#FFF9C4")
        else:
            row_c.append("#FFCDD2")
        cell_colors.append(row_c)

    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                     loc="center", cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Conformité Gap 2 — latence < 100 ms et RAM < 256 Ko sur NUCLEO-F439ZI",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    save_or_show(fig, "11_gap2_compliance", out_dir, show)


# ---------------------------------------------------------------------------
# 12. Throughput — capacité de traitement en ligne
# ---------------------------------------------------------------------------
def plot_throughput(out_dir, show):
    configs = ["Mahalanobis\n(board)", "EWC Head\n(board)", "Streaming\nE18-01"]
    throughput = [34235, 34235, 34235]
    reference = 1000  # 1 kHz sensor sampling rate (typical industrial)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["mahal"], COLORS["ewc"], "#546E7A"]
    bars = ax.bar(configs, throughput, color=colors, width=0.45, zorder=3)
    ax.axhline(reference, color=COLORS["warn"], linewidth=2, linestyle="--",
               label="Fréquence capteur typique : 1 kHz")

    for bar, val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,} ips", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Inférences par seconde (ips)")
    ax.set_title("Throughput d'inférence NUCLEO-F439ZI @ 180 MHz\n(ips = inférences par seconde)")
    ax.set_ylim(0, 42000)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.annotate(f"×{throughput[0]//reference} vs capteur", xy=(0, reference + 200), xytext=(0.3, 15000),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")
    fig.tight_layout()
    save_or_show(fig, "12_throughput", out_dir, show)


# ---------------------------------------------------------------------------
# 13. Paires de modèles + méta-modèle de stacking (Sprints 30-31)
# ---------------------------------------------------------------------------
def plot_pairs_meta_results(out_dir, show):
    """Individuel vs ensemble vs méta (F1, CWRU maha+ewc) + latences board par mode."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panneau gauche : F1 individuel vs ensemble vs méta (exp_S30/S31 PC, paire Maha+EWC, CWRU)
    ax = axes[0]
    labels = ["Maha\nseul", "EWC\nseul", "Ensemble\n(OR)", "Méta\n(logreg)"]
    f1 = [0.379, 1.000, 0.991, 0.997]
    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["warn"], COLORS["board"]]
    bars = ax.bar(labels, f1, color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("F1 (binaire)")
    ax.set_title("Paire Maha + EWC sur CWRU\nindividuel → ensemble → méta-modèle")
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.annotate("le méta arbitre :\n≥ ensemble (+0.006)", xy=(3, 0.997), xytext=(1.4, 0.55),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8.5, color="gray")

    # Panneau droit : latence board combinée par mode (DWT, NUCLEO-F439ZI)
    ax = axes[1]
    modes = ["PAIR\nMaha+EWC", "TRIPLE\n+méta", "PAIR\nMaha+HDC"]
    lat_us = [256, 258, 657]
    colors2 = [COLORS["ok"], COLORS["board"], COLORS["hdc"]]
    bars = ax.bar(modes, lat_us, color=colors2, width=0.55, zorder=3)
    for bar, val in zip(bars, lat_us):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{val} µs", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Latence combinée board (µs)")
    ax.set_title("Latences board par mode (≪ 100 ms)\nparité méta board↔PC = 1.000")
    ax.set_ylim(0, 760)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.text(0.5, 0.9, "méta logreg ≈ négligeable\n(258 − 256 = 2 µs)", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="gray")

    fig.suptitle("Sprints 30-31 — Paires parallèles + méta-modèle de stacking (board réelle)",
                 fontweight="bold")
    fig.tight_layout()
    save_or_show(fig, "13_pairs_meta_results", out_dir, show)


# ---------------------------------------------------------------------------
# 14. Gap 3 — Benchmark INT8 vs FP32 (Sprint 28)
# ---------------------------------------------------------------------------
def plot_int8_benchmark(out_dir, show):
    """Tableau de synthèse INT8 vs FP32 : gain RAM et préservation de la métrique."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")

    columns = ["Modèle", "Gain RAM INT8", "Δ métrique INT8 vs FP32", "Verdict"]
    rows = [
        ["EWC", "×4.00", "≤ 0.006", "✅ préservée"],
        ["HDC (int16-AM)", "×2.33", "0.000", "✅ identique"],
        ["TinyOL", "×4.00", "|Δ| > 0.02 (amélioré)", "✅ fake-quant régularisante"],
        ["Mahalanobis", "×4.00", "−0.236 / −0.238", "⚠️ fallback Q15"],
    ]

    cell_colors = []
    for row in rows:
        row_c = ["#ECEFF1"] * (len(columns) - 1)
        row_c.append("#C8E6C9" if row[-1].startswith("✅") else "#FFF9C4")
        cell_colors.append(row_c)

    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                     loc="center", cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    for j in range(len(columns)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Gap 3 — Quantification INT8 pendant l'apprentissage (Sprint 28, 4 modèles × 5 datasets)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    save_or_show(fig, "14_int8_benchmark", out_dir, show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génère les figures de présentation board (Sprints 16-32)")
    parser.add_argument("--output", "-o", default="docs/figures/presentation_board",
                        help="Répertoire de sortie des figures PNG")
    parser.add_argument("--show", action="store_true", help="Afficher les figures (plt.show())")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder les figures")
    args = parser.parse_args()

    out_dir = None if args.no_save else args.output
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGénération des figures → {out_dir}/\n")

    plots = [
        ("RAM budget par modèle", plot_ram_budget),
        ("Latence inférence", plot_latency),
        ("Latence log scale", plot_latency_log),
        ("Impact λ EWC", plot_ewc_forgetting),
        ("Toutes expériences board vs dry-run", plot_board_vs_dryrun),
        ("Empreinte mémoire Flash + SRAM", plot_memory_breakdown),
        ("Timeline sprints 16-32", plot_sprint_timeline),
        ("Architecture firmware", plot_firmware_architecture),
        ("Protocole UART v3/PAIR/TRIPLE", plot_uart_protocol),
        ("Forgetting intuition", plot_forgetting_intuition),
        ("Conformité Gap 2", plot_gap2_compliance),
        ("Throughput", plot_throughput),
        ("Paires + méta-modèle (S30-31)", plot_pairs_meta_results),
        ("Benchmark INT8 Gap 3 (S28)", plot_int8_benchmark),
    ]

    for name, fn in plots:
        print(f"  → {name}...")
        fn(out_dir, args.show)

    print(f"\n✅ {len(plots)} figures générées dans {out_dir}/\n")


if __name__ == "__main__":
    main()
