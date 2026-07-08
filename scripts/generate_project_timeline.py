"""
Timeline Gantt de l'évolution du projet CL-Embedded — Sprints 1 → 40.

Usage:
    python scripts/generate_project_timeline.py
    python scripts/generate_project_timeline.py --output docs/figures/
    python scripts/generate_project_timeline.py --show

Produit `project_timeline.png` : une barre par sprint positionnée sur l'axe
calendaire réel (avril–août 2026), couleur par phase (Phase 1 PC / Phase 2 MCU),
jalons verticaux (réunions CR, gaps, bornes du stage, deadline manuscrit).

Sources des dates — AUCUNE date inventée, chaque fenêtre est traçable :
- `docs/roadmap_phase1.md` (en-têtes « Sprint N — Semaine (…) — exécuté le … »)
- `docs/roadmap_phase2.md` (« Vue macro Phase 2 » + sections détaillées)
- `docs/sprints/sprint_NN/SNN00_*.md` (champs Semaine/Statut, dates « Implémenté (…) »)
- `git log` du dépôt (dates de commit des sprints)
- `scripts/generate_presentation_plots.py::plot_sprint_timeline` (jalons S16–S32)

Convention : les fenêtres marquées `approx=True` (« ≈ » sur la figure) sont
encadrées par recoupement de ces sources (les champs « Semaine » des docs
sprint ≥ 25 sont des semaines PLANIFIÉES postérieures à l'exécution réelle,
attestée par git log / CLAUDE.md — on privilégie l'exécution réelle).
"""

import argparse
import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Palette validée (dataviz : catégoriel 2 slots, ΔE CVD 73.6, PASS)
COLORS = {
    "phase1": "#2a78d6",   # Phase 1 — PC Python
    "phase2": "#1baf7a",   # Phase 2 — MCU NUCLEO-F439ZI
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}

# (id, début, fin, thème, phase, statut, approx)
# statut : done | in_progress | planned — approx : fenêtre estimée par recoupement
SPRINTS = [
    # ---- Phase 1 — PC Python (avril–mai 2026) ----
    ("S1",  date(2026, 4, 2),  date(2026, 4, 4),  "Infrastructure + EWC/MLP (M2, Monitoring)", "phase1", "done", False),
    ("S2",  date(2026, 4, 5),  date(2026, 4, 6),  "HDC (M3) + comparaison EWC vs HDC", "phase1", "done", False),
    ("S3",  date(2026, 4, 7),  date(2026, 4, 10), "TinyOL (M1) : autoencodeur + tête OtO", "phase1", "done", False),
    ("S5",  date(2026, 4, 7),  date(2026, 4, 12), "Baselines non supervisées (KMeans, PCA, Mahalanobis…)", "phase1", "done", False),
    ("S6",  date(2026, 4, 13), date(2026, 4, 15), "Infrastructure notebooks + 18 expériences", "phase1", "done", True),
    ("S7",  date(2026, 4, 15), date(2026, 4, 21), "Notebooks Monitoring (D2) — 14 notebooks", "phase1", "done", True),
    ("S8",  date(2026, 4, 15), date(2026, 4, 21), "Notebooks Pump (D1) — 14 notebooks", "phase1", "done", True),
    ("S9",  date(2026, 4, 14), date(2026, 4, 22), "Use case déploiement embarqué + clôture", "phase1", "done", True),
    ("S10", date(2026, 4, 21), date(2026, 4, 24), "Dataset Pronostia (D4) — Gap 1", "phase1", "done", False),
    ("S11", date(2026, 4, 24), date(2026, 5, 5),  "Single-task online + feature importance", "phase1", "done", False),
    ("S12", date(2026, 4, 25), date(2026, 4, 26), "Dataset CWRU (D3) by-fault / by-severity", "phase1", "done", True),
    ("S13", date(2026, 4, 27), date(2026, 5, 4),  "Détection d'anomalies one-class", "phase1", "done", False),
    ("S14", date(2026, 5, 5),  date(2026, 5, 9),  "Anomaly detection Monitoring — clôture 6 modèles", "phase1", "done", False),
    ("S15", date(2026, 5, 12), date(2026, 5, 15), "Anomaly detection Pronostia", "phase1", "done", False),
    ("S4",  date(2026, 5, 30), date(2026, 6, 1),  "UINT8 + ONNX + comparaison 3 modèles (hors séquence)", "phase1", "done", True),
    # ---- Phase 2 — MCU NUCLEO-F439ZI (mai–juillet 2026) ----
    ("S16", date(2026, 5, 6),  date(2026, 5, 11), "Portage MCU : toolchain ARM + C embarqué + profiling", "phase2", "done", True),
    ("S17", date(2026, 5, 20), date(2026, 5, 25), "HAL GPIO/UART/PWM + Renode CI", "phase2", "done", False),
    ("S18", date(2026, 5, 25), date(2026, 5, 28), "Pipeline UART v2 + profiling DWT board", "phase2", "done", False),
    ("S19", date(2026, 5, 28), date(2026, 6, 1),  "3 modèles CL en C sur board", "phase2", "done", True),
    ("S21", date(2026, 5, 27), date(2026, 6, 6),  "Tests multi-datasets board (Monitoring + Pronostia)", "phase2", "done", False),
    ("S20", date(2026, 6, 8),  date(2026, 6, 15), "Gap 2 formel : RAM + latence mesurées (3 modèles)", "phase2", "done", False),
    ("S22", date(2026, 6, 7),  date(2026, 6, 14), "CMAPSS + Paderborn + Gap 3 INT8 Python+C", "phase2", "done", True),
    ("S25", date(2026, 6, 10), date(2026, 6, 12), "Tâches natives : RUL régression + multi-classe (PC)", "phase2", "done", True),
    ("S23", date(2026, 6, 14), date(2026, 6, 15), "Board 5 datasets + HDC C + benchmark", "phase2", "done", True),
    ("S24", date(2026, 6, 15), date(2026, 6, 16), "Rétro-application S4 (UINT8, ONNX, profiling unifié)", "phase2", "done", True),
    ("S26", date(2026, 6, 13), date(2026, 6, 16), "Board RUL régression + multi-classe C", "phase2", "done", True),
    ("S27", date(2026, 6, 16), date(2026, 6, 17), "DUAL_MODE : RUL + faute en séquence (637 µs)", "phase2", "done", True),
    ("S28", date(2026, 6, 16), date(2026, 6, 17), "Benchmark INT8 vs FP32 PC (4 modèles × 5 datasets)", "phase2", "done", False),
    ("S29", date(2026, 6, 15), date(2026, 6, 28), "INT8 firmware board — grille 20 cellules", "phase2", "done", False),
    ("S30", date(2026, 6, 17), date(2026, 6, 18), "PAIR_MODE : paires Mahalanobis + supervisé", "phase2", "done", True),
    ("S31", date(2026, 6, 18), date(2026, 6, 19), "TRIPLE_MODE : méta-modèle de stacking", "phase2", "done", True),
    ("S32", date(2026, 6, 22), date(2026, 6, 23), "Étude seuil RUL→faulty + parité board↔PC", "phase2", "done", False),
    ("S33", date(2026, 6, 23), date(2026, 6, 24), "Profilage énergétique + coût (FLOPs/BOPs)", "phase2", "done", False),
    ("S34", date(2026, 6, 24), date(2026, 6, 25), "Streaming/buffer + Q15 Mahalanobis", "phase2", "done", True),
    ("S35", date(2026, 6, 25), date(2026, 6, 27), "Impact nb features (5feat / all / best)", "phase2", "done", True),
    ("S36", date(2026, 6, 27), date(2026, 6, 28), "Comparaison appariée PC↔board EWC + INT8 vs FP32", "phase2", "done", True),
    ("S37", date(2026, 6, 28), date(2026, 6, 29), "Pipeline publication GitLab (export sanitisé)", "phase2", "done", True),
    ("S38", date(2026, 6, 29), date(2026, 6, 30), "EWC autonome : gate de nouveauté embarqué", "phase2", "done", False),
    ("S39", date(2026, 6, 30), date(2026, 7, 4),  "Diagnostic INT8 : émulateur bit-exact + kernel v2", "phase2", "in_progress", False),
    ("S40", date(2026, 7, 5),  date(2026, 7, 11), "Article standalone EWC PC↔board & INT8 vs FP32", "phase2", "planned", False),
]

# (date, label, style) — style : "bound" bornes stage · "cr" réunion · "gap" jalon scientifique
MILESTONES = [
    (date(2026, 3, 16), "Début stage", "bound"),
    (date(2026, 3, 17), "CR réunion", "cr"),
    (date(2026, 4, 8),  "CR réunion", "cr"),
    (date(2026, 4, 15), "Deadline manuscrit prélim.", "bound"),
    (date(2026, 4, 22), "CR réunion", "cr"),
    (date(2026, 4, 23), "Gap 1 ✓ données industrielles", "gap"),
    (date(2026, 5, 19), "CR réunion", "cr"),
    (date(2026, 6, 9),  "CR réunion", "cr"),
    (date(2026, 6, 15), "Gap 2 ✓ RAM/latence board", "gap"),
    (date(2026, 6, 21), "Gap 3 ✓ INT8 en ligne", "gap"),
    (date(2026, 8, 6),  "Fin stage", "bound"),
]


def plot_project_timeline(out_dir: str | None, show: bool) -> None:
    """Gantt calendaire des sprints 1–40 avec jalons projet."""
    fig, ax = plt.subplots(figsize=(15, 13.5))
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])

    n = len(SPRINTS)
    label_flip = date(2026, 6, 20)  # au-delà, le libellé passe à gauche de la barre

    for i, (sid, start, end, theme, phase, status, approx) in enumerate(SPRINTS):
        y = n - 1 - i
        x0 = mdates.date2num(start)
        width = max(mdates.date2num(end) - x0, 0.8)  # barre visible même pour ~1 jour
        color = COLORS[phase]
        if status == "done":
            face, hatch, edge = color, None, "none"
        elif status == "in_progress":
            face, hatch, edge = color, "///", "white"
        else:  # planned
            face, hatch, edge = "none", None, color
        ax.barh(y, width, left=x0, height=0.62, color=face, hatch=hatch,
                edgecolor=edge if edge != "none" else None,
                linewidth=1.4 if status == "planned" else 0.8, zorder=3)

        sid_txt = f"{sid} ≈" if approx else sid
        if end >= label_flip and start > date(2026, 5, 20):
            ax.text(x0 - 0.6, y, f"{theme}  ·  {sid_txt}", ha="right", va="center",
                    fontsize=8, color=COLORS["ink2"], zorder=4)
        else:
            ax.text(x0 + width + 0.6, y, f"{sid_txt}  ·  {theme}", ha="left", va="center",
                    fontsize=8, color=COLORS["ink2"], zorder=4)

    # Jalons verticaux
    y_top = n + 0.2
    for d, label, style in MILESTONES:
        x = mdates.date2num(d)
        if style == "bound":
            ax.axvline(x, color=COLORS["ink2"], linewidth=1.2, linestyle="-", alpha=0.6, zorder=2)
            ax.text(x, y_top + 2.4, label, rotation=45, ha="left", va="bottom",
                    fontsize=8, color=COLORS["ink"], fontweight="bold")
        elif style == "gap":
            ax.axvline(x, color=COLORS["muted"], linewidth=1.0, linestyle="--", alpha=0.8, zorder=2)
            ax.text(x, y_top + 2.4, label, rotation=45, ha="left", va="bottom",
                    fontsize=8, color=COLORS["ink"])
        else:  # cr
            ax.axvline(x, color=COLORS["baseline"], linewidth=0.8, linestyle=":", alpha=0.9, zorder=1)
            ax.plot(x, y_top + 1.6, marker="v", markersize=5, color=COLORS["muted"], clip_on=False)

    # Axe X calendaire
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b 2026"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.set_xlim(mdates.date2num(date(2026, 3, 12)), mdates.date2num(date(2026, 8, 12)))
    ax.grid(axis="x", which="major", color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="x", which="minor", color=COLORS["grid"], linewidth=0.4, alpha=0.5, zorder=0)
    ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=9)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["baseline"])
    ax.set_ylim(-1, y_top + 1.2)

    legend_items = [
        mpatches.Patch(color=COLORS["phase1"], label="Phase 1 — PC Python (S1–S15)"),
        mpatches.Patch(color=COLORS["phase2"], label="Phase 2 — MCU NUCLEO-F439ZI (S16–S40)"),
        mpatches.Patch(facecolor=COLORS["phase2"], hatch="///", edgecolor="white", label="En cours"),
        mpatches.Patch(facecolor="none", edgecolor=COLORS["phase2"], label="Planifié"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9, framealpha=0.95)

    fig.suptitle("CL-Embedded — Timeline des sprints 1–40 (stage 16 mars – 6 août 2026)",
                 fontweight="bold", color=COLORS["ink"], y=0.995, fontsize=14)
    fig.text(0.01, 0.005,
             "≈ : fenêtre d'exécution estimée par recoupement (roadmaps, git log, docs sprint) — "
             "les « semaines » planifiées des docs sprint ≥ 25 sont postérieures à l'exécution réelle. "
             "▾ : réunions CR (17 mars, 8 avr, 22 avr, 19 mai, 9 juin).",
             fontsize=7.5, color=COLORS["muted"])
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "project_timeline.png")
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=COLORS["surface"])
        print(f"  ✅ {path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline Gantt du projet CL-Embedded (sprints 1–40)")
    parser.add_argument("--output", default="docs/figures/", help="Répertoire de sortie PNG")
    parser.add_argument("--show", action="store_true", help="Afficher sans sauvegarder")
    args = parser.parse_args()
    plot_project_timeline(None if args.show and not args.output else args.output, args.show)


if __name__ == "__main__":
    main()
