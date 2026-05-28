"""
Génération des figures pour la présentation board NUCLEO-F439ZI (Sprints 16-20).

Usage:
    python scripts/generate_presentation_plots.py
    python scripts/generate_presentation_plots.py --output docs/figures/presentation_board/
    python scripts/generate_presentation_plots.py --show   # affiche sans sauvegarder

Produit ~10 figures PNG dans le répertoire de sortie.
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
    if show:
        plt.show()
    if out_dir:
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  ✅ {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. RAM Budget — tous les modèles vs limite Gap 2
# ---------------------------------------------------------------------------
def plot_ram_budget(out_dir, show):
    models = ["Mahalanobis", "EWC Head\nMLP", "TinyOL\n(SRAM)", "HDC\n(SRAM)", "3 modèles\nsimultanés"]
    ram_bytes = [200, 9728, 400, 28000, 10328]
    budget_bytes = 64 * 1024  # 64 Ko

    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["tinyol"], COLORS["hdc"], COLORS["ok"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, [r / 1024 for r in ram_bytes], color=colors, width=0.55, zorder=3)
    ax.axhline(64, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget Gap 2 : 64 Ko")
    ax.axhline(192, color="#B71C1C", linewidth=1, linestyle=":", label="SRAM totale NUCLEO : 192 Ko")

    for bar, val in zip(bars, ram_bytes):
        label = f"{val} B" if val < 1024 else f"{val/1024:.1f} Ko"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("RAM utilisée (Ko)")
    ax.set_title("Empreinte RAM par modèle vs budget Gap 2 (NUCLEO-F439ZI)")
    ax.set_ylim(0, 220)
    ax.set_yticks([0, 16, 32, 64, 96, 128, 192])
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "01_ram_budget", out_dir, show)


# ---------------------------------------------------------------------------
# 2. Latence d'inférence vs budget 100 ms
# ---------------------------------------------------------------------------
def plot_latency(out_dir, show):
    models = ["Mahalanobis\n(board)", "EWC Head\n(board)", "Budget\nGap 2"]
    latencies = [0.004, 0.004, 100.0]
    colors = [COLORS["mahal"], COLORS["ewc"], COLORS["budget"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models[:2], latencies[:2], color=colors[:2], width=0.4, zorder=3)
    ax.axhline(100, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget : 100 ms")

    for bar, val in zip(bars, latencies[:2]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotation marge
    ax.annotate("", xy=(0, 100), xytext=(0, 0.004),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(0.15, 50, "×25 000\nde marge", color="gray", fontsize=9, va="center")

    ax.set_ylabel("Latence (ms) — échelle linéaire")
    ax.set_title("Latence d'inférence sur NUCLEO-F439ZI @ 180 MHz")
    ax.set_ylim(0, 115)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "02_latency", out_dir, show)


# ---------------------------------------------------------------------------
# 3. Latence en échelle log pour voir les valeurs réelles
# ---------------------------------------------------------------------------
def plot_latency_log(out_dir, show):
    models = ["Mahal.\n(board)", "EWC\n(board)", "Mahal.\n(dry-run)", "EWC\n(dry-run)"]
    latencies = [0.004, 0.004, 0.004, 5.44]
    colors = [COLORS["board"], COLORS["board"], COLORS["dryrun"], COLORS["dryrun"]]
    hatches = ["", "//", "", "//"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, latencies, color=colors, hatch=hatches, width=0.5, zorder=3, edgecolor="white")
    ax.axhline(100, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget Gap 2 : 100 ms", zorder=4)
    ax.set_yscale("log")

    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.4,
                f"{val} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    board_patch = mpatches.Patch(color=COLORS["board"], label="Board réel")
    dry_patch = mpatches.Patch(color=COLORS["dryrun"], label="Dry-run (simulation)")
    ax.legend(handles=[board_patch, dry_patch,
                        mpatches.Patch(color=COLORS["budget"], label="Budget 100 ms")],
              loc="upper left")
    ax.set_ylabel("Latence (ms) — échelle log")
    ax.set_title("Latence board vs simulation (échelle logarithmique)")
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
    models = ["Mahalanobis", "EWC Head MLP", "TinyOL", "HDC"]
    flash_kb = [0.12, 0, 5.7, 2.0]    # poids en Flash (.rodata)
    sram_kb = [0.20, 9.73, 0.40, 28.0]  # poids en SRAM (.bss)

    x = np.arange(len(models))
    w = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x, flash_kb, w, color="#FF8F00", label="Flash (.rodata — poids constants)")
    b2 = ax.bar(x, sram_kb, w, bottom=flash_kb, color=COLORS["ewc"], label="SRAM (.bss — poids entraînables)")

    ax.axhline(64, color=COLORS["budget"], linewidth=2, linestyle="--", label="Budget SRAM Gap 2 : 64 Ko")

    for i, (f, s) in enumerate(zip(flash_kb, sram_kb)):
        total = f + s
        ax.text(i, total + 0.3, f"{total:.1f} Ko", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mémoire (Ko)")
    ax.set_title("Empreinte mémoire par modèle (Flash + SRAM)")
    ax.set_ylim(0, 75)
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    save_or_show(fig, "06_memory_breakdown", out_dir, show)


# ---------------------------------------------------------------------------
# 7. Timeline des sprints 16–20
# ---------------------------------------------------------------------------
def plot_sprint_timeline(out_dir, show):
    sprints = [
        ("Sprint 16", "20 mai", 0, 5, "Toolchain ARM + C + profiling DWT", COLORS["ok"]),
        ("Sprint 17", "25 mai", 5, 5, "HAL GPIO/UART/PWM + Renode CI", COLORS["ok"]),
        ("Sprint 18", "1 jun", 10, 7, "Pipeline UART v2 + profiling + E18-01 board", COLORS["ok"]),
        ("Sprint 19", "8 jun", 17, 7, "3 modèles CL en C + protocol v3 + E19-01/02", "#FFA000"),
        ("Sprint 20", "15 jun", 24, 7, "Fix EWC + TinyOL weights + Gap 2 formel", "#78909C"),
    ]

    fig, ax = plt.subplots(figsize=(13, 4))

    for name, date, start, dur, desc, color in sprints:
        rect = mpatches.FancyBboxPatch(
            (start, 0.2), dur - 0.2, 0.6,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
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
        mpatches.Patch(color="#FFA000", label="🔄 En cours (12/13 tâches)"),
        mpatches.Patch(color="#78909C", label="⬜ À faire"),
    ]
    ax.legend(handles=status_patches, loc="lower right", fontsize=9)

    ax.set_xlim(-0.5, 32)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Timeline Sprints 16–20 — CL-Embedded Phase 2 (mai–juin 2026)", fontweight="bold")
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
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))

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

    # Réponse v2
    resp_v2 = [
        ("PRED\n1B", 1, "#2E7D32"),
        ("CONF\n2B", 2, "#388E3C"),
        ("LAT_US\n4B", 4, "#43A047"),
        ("RAM_B\n2B", 2, "#4CAF50"),
        ("THRU\n2B", 2, "#66BB6A"),
        ("STATUS\n1B", 1, "#81C784"),
        ("CRC16\n2B", 2, "#A5D6A7"),
    ]
    draw_frame(axes[1], "Réponse Carte → PC — Protocol v2 (Sprint 18)", resp_v2, None)

    # Réponse v3
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
    draw_frame(axes[2], "Réponse Carte → PC — Protocol v3 (Sprint 19, +métriques CL)", resp_v3, None)

    fig.suptitle("Évolution du protocole UART binaire v2→v3", fontsize=13, fontweight="bold", y=1.01)
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    columns = ["Modèle", "RAM utilisée", "Budget Gap 2", "Marge RAM", "Latence", "Budget latence", "Marge latence", "✓"]
    rows = [
        ["Mahalanobis", "200 B", "64 Ko", "×328", "3–4 µs", "100 ms", "×25 000", "✅"],
        ["EWC Head MLP", "9.7 Ko", "64 Ko", "×6.8", "3–4 µs", "100 ms", "×25 000", "✅"],
        ["TinyOL", "400 B SRAM\n+5.7 Ko Flash", "64 Ko", "×164", "TBD", "100 ms", "TBD", "⏳"],
        ["3 modèles simultanés", "~11 Ko", "64 Ko", "×5.8", "~4 µs", "100 ms", "×25 000", "✅"],
    ]

    cell_colors = []
    for row in rows:
        row_c = ["#ECEFF1"] * 7
        if row[-1] == "✅":
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

    ax.set_title("Conformité Gap 2 — RAM < 64 Ko et latence < 100 ms sur NUCLEO-F439ZI",
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génère les figures de présentation board (Sprints 16-20)")
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
        ("Timeline sprints 16-20", plot_sprint_timeline),
        ("Architecture firmware", plot_firmware_architecture),
        ("Protocole UART v2→v3", plot_uart_protocol),
        ("Forgetting intuition", plot_forgetting_intuition),
        ("Conformité Gap 2", plot_gap2_compliance),
        ("Throughput", plot_throughput),
    ]

    for name, fn in plots:
        print(f"  → {name}...")
        fn(out_dir, args.show)

    print(f"\n✅ {len(plots)} figures générées dans {out_dir}/\n")


if __name__ == "__main__":
    main()
