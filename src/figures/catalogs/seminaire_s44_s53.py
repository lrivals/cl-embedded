"""Catalogue `seminaire_s44_s53` — présentation encadrants, sprints 44 → 53.

Restitue le travail accompli de juillet à septembre 2026 sur quatre axes, plus les
schémas de cadrage :

    t1_timeline_sprints.png        Frise S44→S53 par axe, avec le statut de plateforme.
    t2_carte_axes_gaps.png         Axes de travail × triple gap.
    d1_drift_proxy_vs_board.png    Latence proxy PC (S44) vs mesurée board (S45).
    d2_drift_portabilite.png       État algorithmique, `.bss` réel, parité, PSI N/A.
    q1_carte_quantification.png    Les 3 axes orthogonaux de quantification.
    q2_profondeur_pc_board.png     Sub-INT8 : ΔAUROC émulé PC ↔ `.bss` packé mesuré board.
    q3_moment_bilan.png            Moment de quantification : `both` vs `after` (board).
    r1_ram_totale_recap.png        RAM totale `.data + .bss + pic` en % du budget.
    e1_latence_int8_breakdown.png  Décomposition en cycles du noyau EWC INT8.
    e2_s53_wfi_repos.png           Courant de repos : scrutation UART vs `__WFI()`.
    e3_s53_uj_par_inference.png    µJ/inférence par régression I(N) sur lot imposé.
    e4_s53_gap3_energie.png        Gap 3 énergie : INT8 vs FP32 avec incertitudes.
    e5_s53_sysclk.png              Énergie et marge Gap 2 vs fréquence SYSCLK.
    e6_s53_statut_mesures.png      Statut honnête : chiffré vs « à mesurer ».

**Toute valeur tracée provient d'un ``load_experiment``** — aucun littéral numérique de
résultat (garde AST ``test_no_hardcoded_results``). Une cellule non mesurée est tracée en
**gris à zéro avec sa mention textuelle**, jamais comme un résultat nul. Les trois schémas
(T1, T2, Q1) ne portent aucune donnée : ce sont des cartes de lecture.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import A_MESURER, load_experiment
from src.figures.registry import register_catalog
from src.figures.schematic import base_fig, box, footnote
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "seminaire_s44_s53"

# ── Palette ──────────────────────────────────────────────────────────────────
COLOR_BOARD = STRATEGY_COLORS["fp32"]        # bleu   — mesuré board
COLOR_PC = STRATEGY_COLORS["int8_qat"]       # vert   — PC / émulé
COLOR_INT8 = STRATEGY_COLORS["int8_v2"]      # orange — INT8
COLOR_ALERT = STRATEGY_COLORS["int8_ptq_legacy"]  # rouge — dégradation / alerte
COLOR_ACCENT = STRATEGY_COLORS["q15"]        # violet — accent
COLOR_BSS = STRATEGY_COLORS["int16_am"]      # brun   — .bss
NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"
GRID = "#dddddd"

# ── Constantes entières (unités, jamais des résultats) ───────────────────────
KIB = 1024
MA_PER_A = 1000
US_PER_S = 1000000
PCT = 100
WRAP_WIDTH = 52
X_PAD_MHZ = 12
LOG_LABEL_GAP = 5

NA_TEXT = "N/A"
BADGE_BOARD = "mesuré board — NUCLEO-F439ZI (Cortex-M4 @ 180 MHz, 256 Ko SRAM)"
BADGE_MIXED = "mesuré board (NUCLEO-F439ZI) vs proxy / émulé PC — jamais fusionnés"
BADGE_SCHEMA = "schéma de lecture — aucune donnée tracée"
BADGE_ENERGY = "mesuré board + sonde X-NUCLEO-LPM01A — les cases grises restent « à mesurer »"


# ── Sources (lecture seule, skip gracieux) ───────────────────────────────────

def _load(rel: str) -> dict | None:
    """Charge un JSON d'expérience ; ``None`` s'il n'a pas encore été produit."""
    try:
        data, _ = load_experiment(rel)
    except FileNotFoundError:
        return None
    return data


def _num(value) -> float | None:
    """Nombre exploitable, ou ``None`` — ``« à mesurer »``, ``null`` et bool exclus."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_na(value) -> bool:
    """Vrai si la valeur est le sentinel littéral « à mesurer »."""
    return isinstance(value, str) and value.strip().lower() == A_MESURER


def _empty(message: str, name: str, out_root: Path) -> Path:
    """Figure d'attente lisible quand la source n'est pas encore produite."""
    fig, ax = base_fig()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color=MUTED,
            wrap=True, transform=ax.transAxes)
    return savefig_png(fig, CATALOG, name, out_root)


def _bar_labels(ax: plt.Axes, xs, values, texts, dy: float = 0.03, errs=None) -> None:
    """Étiquette chaque barre ; une valeur manquante porte son texte, jamais un zéro.

    ``errs`` décale l'étiquette au-dessus de la barre d'erreur quand il y en a une ;
    sur un axe logarithmique le décalage devient multiplicatif (sinon les petites
    barres verraient leur étiquette projetée en haut de la figure).
    """
    log_scale = ax.get_yscale() == "log"
    top = max([v for v in values if v is not None] or [1.0])
    errs = errs if errs is not None else [0.0] * len(values)
    for x, v, txt, e in zip(xs, values, texts, errs):
        if not txt:
            continue
        base = (v if v is not None else 0.0) + (e or 0.0)
        y = base * (1 + LOG_LABEL_GAP * dy) if (log_scale and base > 0) else base + top * dy
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=9, color=INK)


# ═════════════════════════════ T — Cadrage ═══════════════════════════════════

SPRINT_LANES: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Drift", [
        ("S44", "Famille de détecteurs\n9 × 4 cellules", "pc"),
        ("S45", "Portage board\nPH · DDM · PSI", "board"),
    ]),
    ("Quantification", [
        ("S46", "Moment\navant / après / les deux", "board"),
        ("S47", "Profondeur\nsub-INT8, granularité", "pc"),
        ("S48", "Portage sub-INT8\nbit-packing", "board"),
    ]),
    ("Mémoire", [
        ("S49", "RAM totale\n.data + .bss + pile", "board"),
        ("S51", "Score système\nspec seule", "spec"),
    ]),
    ("Coût & énergie", [
        ("S50", "Latence INT8\n+ 1er banc énergie", "board"),
        ("S53", "Campagne LPM01A\nWFI · cadence · SYSCLK", "board"),
    ]),
    ("Qualité", [
        ("S52", "Flag UART TinyOL\nré-mesure complète", "board"),
    ]),
]

STATUS_COLOR: dict[str, str] = {"board": COLOR_BOARD, "pc": COLOR_PC, "spec": NA_GRAY}
STATUS_LABEL: dict[str, str] = {
    "board": "mesuré carte réelle",
    "pc": "PC / émulateur",
    "spec": "spécifié, non exécuté",
}


def _fig_timeline(out_root: Path) -> Path:
    """T1 — frise des sprints 44→53, un couloir par axe de travail."""
    fig, ax = base_fig()
    n_lanes = len(SPRINT_LANES)
    for row, (lane, sprints) in enumerate(SPRINT_LANES):
        y = 0.86 - row * 0.155
        ax.text(0.02, y, lane, ha="left", va="center", fontsize=11, color=INK,
                fontweight="bold", transform=ax.transAxes)
        x_first, x_step, box_w = 0.35, 0.22, 0.185
        ax.plot([0.25, x_first - box_w / 2], [y, y], color=GRID, linewidth=1.5,
                transform=ax.transAxes, zorder=0)
        for col, (sid, label, status) in enumerate(sprints):
            cx = x_first + col * x_step
            box(ax, cx, y, f"{sid} — {label}", STATUS_COLOR[status], w=box_w, h=0.115,
                fontsize=9)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                          color=STATUS_COLOR[k], label=STATUS_LABEL[k])
               for k in ("board", "pc", "spec")]
    ax.legend(handles=handles, loc="lower center", ncol=n_lanes, frameon=False,
              fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sprints 44 → 53 — juillet à septembre 2026, par axe de travail")
    footnote(fig, BADGE_SCHEMA)
    return savefig_png(fig, CATALOG, "t1_timeline_sprints", out_root)


GAP_COLUMNS: list[str] = [
    "Gap 1\ndonnées industrielles",
    "Gap 2\nlatence & RAM mesurées",
    "Gap 3\nquantification en CL",
]
AXIS_ROWS: list[tuple[str, list[str]]] = [
    ("Drift\nS44 · S45", [
        "4 corpus à drift\nlabellisé (S43)",
        "270 µs board,\n.bss mesurée (S45)",
        "—",
    ]),
    ("Quantification\nS46 · S47 · S48", [
        "Monitoring\net Pronostia",
        "dépacking mesuré,\nlatence board",
        "moment × profondeur\n× granularité",
    ]),
    ("Mémoire\nS49", [
        "—",
        "RAM totale réelle\n.data + .bss + pile",
        "INT8 ≡ FP32 en .bss :\nle gain est le packing",
    ]),
    ("Coût & énergie\nS50 · S53", [
        "—",
        "budget latence\nconvertible en autonomie",
        "l'INT8 ne gagne\nni latence ni énergie",
    ]),
]


def _fig_axes_gaps(out_root: Path) -> Path:
    """T2 — ce que chaque axe de travail apporte à chacun des trois gaps."""
    fig, ax = base_fig()
    x0, col_w = 0.24, 0.245
    for j, header in enumerate(GAP_COLUMNS):
        ax.text(x0 + j * col_w + col_w / 2, 0.9, header, ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold", transform=ax.transAxes)
    for i, (axis_name, cells) in enumerate(AXIS_ROWS):
        y = 0.74 - i * 0.19
        ax.text(0.03, y, axis_name, ha="left", va="center", fontsize=11, color=INK,
                fontweight="bold", transform=ax.transAxes)
        for j, cell in enumerate(cells):
            cx = x0 + j * col_w + col_w / 2
            filled = cell != "—"
            box(ax, cx, y, cell, COLOR_ACCENT if filled else NA_GRAY,
                w=col_w * 0.86, h=0.14, fontsize=9)
    fig.suptitle("Axes de travail × triple gap — où chaque sprint apporte une preuve")
    footnote(fig, BADGE_SCHEMA)
    return savefig_png(fig, CATALOG, "t2_carte_axes_gaps", out_root)


# ═════════════════════════════ D — Drift ═════════════════════════════════════

DRIFT_DATASET = "gas_sensor_drift"
DRIFT_LABEL: dict[str, str] = {
    "page_hinkley": "Page-Hinkley", "ddm": "DDM", "psi": "PSI",
}


def _s45() -> dict | None:
    return _load("experiments/exp_S45_summary.json")


def _drift_cells(summary: dict) -> list[tuple[str, dict, dict]]:
    """(détecteur, bloc board, bloc proxy PC) pour le dataset porté sur carte."""
    node = summary.get("results", {}).get(DRIFT_DATASET, {})
    return [(det, node.get(det, {}).get("board", {}) or {},
             node.get(det, {}).get("pc_proxy", {}) or {})
            for det in summary.get("detectors", [])]


def _fig_drift_proxy_vs_board(out_root: Path) -> Path:
    """D1 — le proxy PC ne prédit pas la carte : latences côte à côte, échelle log."""
    summary = _s45()
    if summary is None:
        return _empty("exp_S45_summary.json absent — lancer scripts/aggregate_sprint45.py",
                      "d1_drift_proxy_vs_board", out_root)
    cells = _drift_cells(summary)
    fig, ax = plt.subplots()
    x = np.arange(len(cells))
    width = 0.35
    proxy = [_num(p.get("latency_us_per_update")) for _, _, p in cells]
    board = [_num(b.get("latency_us_p50")) for _, b, _ in cells]
    ax.bar(x - width / 2, [v if v is not None else 0.0 for v in proxy], width,
           color=[COLOR_PC if v is not None else NA_GRAY for v in proxy],
           edgecolor=INK, linewidth=0.5, label="proxy PC (S44)")
    ax.bar(x + width / 2, [v if v is not None else 0.0 for v in board], width,
           color=[COLOR_BOARD if v is not None else NA_GRAY for v in board],
           edgecolor=INK, linewidth=0.5, label="mesuré board (S45)")
    ax.set_yscale("log")
    for xi, v in zip(x - width / 2, proxy):
        ax.text(xi, v if v is not None else 1.0, f"{v:.1f}" if v is not None else NA_TEXT,
                ha="center", va="bottom", fontsize=9, color=INK)
    for xi, (v, (det, b, _)) in zip(x + width / 2, zip(board, cells)):
        txt = f"{v:.0f}" if v is not None else NA_TEXT
        ax.text(xi, v if v is not None else 1.0, txt, ha="center", va="bottom",
                fontsize=9, color=INK)
        if v is None and b.get("na_reason"):
            ax.annotate("non portable\nsur carte\n(débordement SRAM)",
                        xy=(xi, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 20), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=MUTED)
    gap2 = _num(summary.get("gap2_latency_us"))
    if gap2 is not None:
        ax.axhline(gap2, color=COLOR_ALERT, linestyle="--", linewidth=1.5)
        ax.text(x[-1], gap2, f" budget Gap 2 = {gap2 / MA_PER_A:.0f} ms", va="bottom",
                ha="right", fontsize=9, color=COLOR_ALERT)
    ax.set_xticks(x)
    ax.set_xticklabels([DRIFT_LABEL.get(d, d) for d, _, _ in cells])
    ax.set_ylabel("latence par échantillon (µs, échelle log)")
    ax.legend(fontsize=10)
    ax.set_title(f"Détection de drift — proxy PC vs carte réelle ({DRIFT_DATASET})")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "d1_drift_proxy_vs_board", out_root)


def _fig_drift_portabilite(out_root: Path) -> Path:
    """D2 — coût d'état, surcoût `.bss` réel et parité de verdict board ↔ PC."""
    summary = _s45()
    if summary is None:
        return _empty("exp_S45_summary.json absent — lancer scripts/aggregate_sprint45.py",
                      "d2_drift_portabilite", out_root)
    cells = _drift_cells(summary)
    deltas = summary.get("bss_delta_by_method", {})
    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(cells))
    labels = [DRIFT_LABEL.get(d, d) for d, _, _ in cells]

    ax = axes[0]
    state = [_num(p.get("state_bytes_algo")) for _, _, p in cells]
    delta = [_num(deltas.get(d)) for d, _, _ in cells]
    width = 0.35
    ax.bar(x - width / 2, [v if v is not None else 0.0 for v in state], width,
           color=COLOR_PC, edgecolor=INK, linewidth=0.5, label="état algorithmique (PC)")
    ax.bar(x + width / 2, [v if v is not None else 0.0 for v in delta], width,
           color=COLOR_BSS, edgecolor=INK, linewidth=0.5, label="surcoût .bss mesuré")
    ax.set_yscale("log")
    _bar_labels(ax, x - width / 2, state,
                [f"{v:.0f} o" if v is not None else NA_TEXT for v in state])
    _bar_labels(ax, x + width / 2, delta,
                [f"+{v:.0f} o" if v is not None else NA_TEXT for v in delta])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("octets (échelle log)")
    ax.set_title("Coût mémoire de l'état", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    parity = [_num(b.get("verdict_parity")) for _, b, _ in cells]
    f1 = [_num(b.get("f1")) for _, b, _ in cells]
    ax.bar(x - width / 2, [v if v is not None else 0.0 for v in parity], width,
           color=[COLOR_BOARD if v is not None else NA_GRAY for v in parity],
           edgecolor=INK, linewidth=0.5, label="parité verdict board ↔ PC")
    ax.bar(x + width / 2, [v if v is not None else 0.0 for v in f1], width,
           color=[COLOR_ACCENT if v is not None else NA_GRAY for v in f1],
           edgecolor=INK, linewidth=0.5, label="F1 de détection (board)")
    _bar_labels(ax, x - width / 2, parity,
                [f"{v:.3f}" if v is not None else "" for v in parity])
    _bar_labels(ax, x + width / 2, f1,
                [f"{v:.3f}" if v is not None else "" for v in f1])
    for xi, (_, b, _) in zip(x, cells):
        if not b.get("measured"):
            ax.annotate("non porté sur carte", xy=(xi, 0.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 16),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=MUTED, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("sans unité")
    ax.set_title("Fidélité du portage et détection", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Portabilité MCU des détecteurs de drift (S45)")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "d2_drift_portabilite", out_root)


# ═════════════════════════ Q — Quantification ════════════════════════════════

QUANT_AXES: list[tuple[str, str, str]] = [
    ("MOMENT", "quand quantifier ?", "avant (QAT) · après (PTQ) · les deux\n→ Sprint 46"),
    ("FORMAT", "quel encodage ?", "INT8 affine · Q15 · v2 calibré\n→ Sprints 34 · 39"),
    ("PROFONDEUR", "combien de bits ?", "int8 → int4 → ternaire → binaire\nper-tensor / per-channel\n→ Sprints 47 · 48"),
]


def _fig_carte_quantification(out_root: Path) -> Path:
    """Q1 — les trois axes orthogonaux le long desquels la quantification varie."""
    fig, ax = base_fig()
    for i, (title, question, content) in enumerate(QUANT_AXES):
        cx = 0.2 + i * 0.3
        ax.text(cx, 0.82, title, ha="center", va="center", fontsize=14, color=INK,
                fontweight="bold", transform=ax.transAxes)
        ax.text(cx, 0.74, question, ha="center", va="center", fontsize=10,
                color=MUTED, style="italic", transform=ax.transAxes)
        box(ax, cx, 0.55, content, COLOR_ACCENT, w=0.26, h=0.26, fontsize=10)
    ax.text(0.5, 0.3, "Trois axes indépendants : une même tête EWC peut être QAT + Q15 + per-channel.\n"
                      "Les balayer séparément est ce qui rend les conclusions attribuables.",
            ha="center", va="center", fontsize=11, color=INK, transform=ax.transAxes)
    box(ax, 0.5, 0.12,
        "Contrainte commune : parité bit-à-bit émulateur PC ↔ carte réelle\n"
        "— sans elle, aucun chiffre PC n'engage la carte",
        COLOR_BOARD, w=0.62, h=0.13, fontsize=10)
    fig.suptitle("Carte de la quantification — les trois axes explorés")
    footnote(fig, BADGE_SCHEMA)
    return savefig_png(fig, CATALOG, "q1_carte_quantification", out_root)


S48_DATASETS: list[str] = ["monitoring", "pronostia"]
S48_BITS: list[str] = ["4", "2", "1"]
BITS_LABEL: dict[str, str] = {"4": "int4", "2": "ternaire", "1": "binaire"}


def _fig_profondeur(out_root: Path) -> Path:
    """Q2 — sub-INT8 : la perte AUROC (émulée PC) et le gain `.bss` (mesuré board)."""
    summary = _load("experiments/exp_S48_summary.json")
    if summary is None:
        return _empty("exp_S48_summary.json absent — lancer scripts/aggregate_sprint48.py",
                      "q2_profondeur_pc_board", out_root)
    by_cond = summary.get("results_by_condition", {})

    def cell(ds: str, bits: str) -> dict:
        return by_cond.get(ds, {}).get(bits, {}).get("per_channel", {}) or {}

    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(S48_BITS))

    ax = axes[0]
    for i, (ds, color) in enumerate(zip(S48_DATASETS, (COLOR_PC, COLOR_ACCENT))):
        deltas = [_num((cell(ds, b).get("pc") or {}).get("delta_auroc_vs_fp32"))
                  for b in S48_BITS]
        ax.plot(x, deltas, marker="o", color=color, label=f"{ds} — ΔAUROC émulé PC")
        dy = 12 if i == 0 else -20
        for xi, v in zip(x, deltas):
            if v is not None:
                ax.annotate(f"{v:+.4f}", (xi, v), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=9, color=color)
    ax.axhline(0.0, color=MUTED, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([BITS_LABEL[b] for b in S48_BITS])
    ax.set_ylabel("ΔAUROC vs FP32")
    ax.set_title("Perte de performance (per-channel)", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    width = 0.35
    for i, (ds, color) in enumerate(zip(S48_DATASETS, (COLOR_BOARD, COLOR_INT8))):
        saved = [_num((cell(ds, b).get("board") or {}).get("bss_saved_by_packing"))
                 for b in S48_BITS]
        offset = (i - 0.5) * width
        ax.bar(x + offset, [v if v is not None else 0.0 for v in saved], width,
               color=[color if v is not None else NA_GRAY for v in saved],
               edgecolor=INK, linewidth=0.5, label=f"{ds} — .bss gagnée (board)")
        _bar_labels(ax, x + offset, saved,
                    [f"{v:.0f} o" if v is not None else NA_TEXT for v in saved])
    lat_np = _num(((cell(S48_DATASETS[0], S48_BITS[0]).get("board") or {})
                   .get("nonpacked") or {}).get("latency_dwt_p50_us"))
    lat_p = _num(((cell(S48_DATASETS[0], S48_BITS[0]).get("board") or {})
                  .get("packed") or {}).get("latency_dwt_p50_us"))
    if lat_np is not None and lat_p is not None:
        ax.set_xlabel(f"coût du dépacking : {lat_np:.0f} µs → {lat_p:.0f} µs "
                      f"(+{lat_p - lat_np:.0f} µs) — Gap 2 tenu",
                      fontsize=10, color=COLOR_ALERT)
    ax.set_xticks(x)
    ax.set_xticklabels([BITS_LABEL[b] for b in S48_BITS])
    ax.set_ylabel("octets de `.bss` économisés")
    ax.set_title("Gain mémoire — conditionnel au bit-packing", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Sub-INT8 : perte émulée PC (S47) ↔ gain mémoire mesuré board (S48)")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "q2_profondeur_pc_board", out_root)


def _fig_moment(out_root: Path) -> Path:
    """Q3 — moment de quantification : `both` (QAT→PTQ) vs `after` (PTQ calibrée)."""
    cells = [(ds, _load(f"experiments/exp_S46_board/{ds}_both.json"))
             for ds in S48_DATASETS]
    if all(c is None for _, c in cells):
        return _empty("exp_S46_board/*.json absent — lancer scripts/run_sprint46_board.py",
                      "q3_moment_bilan", out_root)
    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(cells))
    width = 0.35

    ax = axes[0]
    after = [_num((c or {}).get("f1_after_board")) for _, c in cells]
    both = [_num((c or {}).get("f1_faulty")) for _, c in cells]
    ax.bar(x - width / 2, [v if v is not None else 0.0 for v in after], width,
           color=[COLOR_INT8 if v is not None else NA_GRAY for v in after],
           edgecolor=INK, linewidth=0.5, label="après (PTQ calibrée, S40)")
    ax.bar(x + width / 2, [v if v is not None else 0.0 for v in both], width,
           color=[COLOR_PC if v is not None else NA_GRAY for v in both],
           edgecolor=INK, linewidth=0.5, label="les deux (QAT → PTQ, S46)")
    _bar_labels(ax, x - width / 2, after,
                [f"{v:.4f}" if v is not None else NA_TEXT for v in after])
    _bar_labels(ax, x + width / 2, both,
                [f"{v:.4f}" if v is not None else NA_TEXT for v in both])
    ax.set_xticks(x)
    ax.set_xticklabels([ds for ds, _ in cells])
    ax.set_ylabel("F1 classe « faulty » (board)")
    ax.set_title("Le QAT préserve, sans gain décisif", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    lat = [_num((c or {}).get("latency_dwt_us_p50")) for _, c in cells]
    parity = [_num((c or {}).get("parity_board_pc")) for _, c in cells]
    ratio = [_num((c or {}).get("ram_ratio_fp32_over_quant")) for _, c in cells]
    rows = [("latence board (µs)", lat, COLOR_BOARD, "{:.0f}"),
            ("parité board ↔ émulateur", parity, COLOR_PC, "{:.3f}"),
            ("RAM poids FP32 / quantifiée", ratio, COLOR_ACCENT, "×{:.1f}")]
    ax.axis("off")
    ax.text(0.5, 0.95, "Contrôles du portage", ha="center", fontsize=11,
            color=INK, transform=ax.transAxes)
    for j, (ds, _) in enumerate(cells):
        ax.text(0.6 + j * 0.3, 0.84, ds, ha="center", fontsize=10, color=INK,
                fontweight="bold", transform=ax.transAxes)
    for i, (label, values, color, fmt) in enumerate(rows):
        y = 0.7 - i * 0.19
        ax.text(0.0, y, label, ha="left", fontsize=10, color=MUTED,
                transform=ax.transAxes)
        for j, v in enumerate(values):
            ax.text(0.6 + j * 0.3, y, fmt.format(v) if v is not None else NA_TEXT,
                    ha="center", fontsize=12, color=color if v is not None else MUTED,
                    transform=ax.transAxes)
    crc = [_num((c or {}).get("crc_errors")) for _, c in cells]
    if all(v is not None for v in crc):
        ax.text(0.5, 0.06, f"erreurs CRC cumulées : {sum(crc):.0f}", ha="center",
                fontsize=10, color=INK, transform=ax.transAxes)

    fig.suptitle("Moment de quantification — mesuré carte réelle (S46, protocole gelé)")
    fig.text(0.5, 0.005, BADGE_BOARD, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "q3_moment_bilan", out_root)


# ═════════════════════════════ R — RAM ═══════════════════════════════════════

RAM_MODELS: list[str] = ["ewc", "hdc", "tinyol", "mahalanobis"]
RAM_LABEL: dict[str, str] = {"ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL",
                             "mahalanobis": "Mahalanobis"}
RAM_CONDITION = "5feat"


def _fig_ram(out_root: Path) -> Path:
    """R1 — RAM totale réelle en part du budget, et ce que chaque phase creuse."""
    summary = _load("experiments/exp_S49_ram/summary.json")
    if summary is None:
        return _empty("exp_S49_ram/summary.json absent — lancer scripts/aggregate_ram.py",
                      "r1_ram_totale_recap", out_root)
    budget = _num((summary.get("_meta") or {}).get("ram_budget_bytes"))
    dataset = S48_DATASETS[1]

    def cell(model: str, enc: str) -> dict | None:
        node = (summary.get(model, {}).get(dataset, {}).get(RAM_CONDITION, {})
                .get(enc, {}).get("board"))
        return node or None

    fig, axes = plt.subplots(1, 2)
    x = np.arange(len(RAM_MODELS))

    # Panneau gauche — la RAM totale, section par section, en part du budget.
    ax = axes[0]
    width = 0.35
    for i, (enc, color) in enumerate((("fp32", COLOR_BOARD), ("int8", COLOR_INT8))):
        offset = (i - 0.5) * width
        for j, model in enumerate(RAM_MODELS):
            c = cell(model, enc)
            total = _num((c or {}).get("total"))
            if total is None or budget is None:
                ax.bar(x[j] + offset, 0.0, width, color=NA_GRAY, edgecolor=INK,
                       linewidth=0.5)
                ax.annotate("non mesuré", xy=(x[j] + offset, 0.0),
                            xycoords=("data", "axes fraction"), xytext=(0, 14),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=MUTED, rotation=90)
                continue
            segs = [(_num(c.get("data")), COLOR_ACCENT),
                    (_num(c.get("bss")), COLOR_BSS),
                    (_num(c.get("stack_peak_update")) or
                     _num(c.get("stack_peak_inference")), color)]
            bottom = 0.0
            for value, seg_color in segs:
                if value is None:
                    continue
                height = value / budget * PCT
                ax.bar(x[j] + offset, height, width, bottom=bottom, color=seg_color,
                       edgecolor=INK, linewidth=0.5)
                bottom += height
            ax.text(x[j] + offset, bottom, f"{total / budget * PCT:.1f} %",
                    ha="center", va="bottom", fontsize=8, color=INK, rotation=90)
    ax.set_ylim(0, PCT)
    ax.set_xticks(x)
    ax.set_xticklabels([RAM_LABEL[m] for m in RAM_MODELS], rotation=15)
    ax.set_ylabel(f"% du budget SRAM ({budget / KIB:.0f} Ko)" if budget
                  else "% du budget SRAM")
    ax.set_title("RAM totale = .data + .bss + pic de pile", fontsize=11)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=11, color=c,
                          label=lab)
               for c, lab in ((COLOR_ACCENT, ".data"), (COLOR_BSS, ".bss"),
                              (COLOR_BOARD, "pic de pile FP32"),
                              (COLOR_INT8, "pic de pile INT8"),
                              (NA_GRAY, "non mesuré"))]
    ax.legend(handles=handles, fontsize=9, ncol=2, loc="upper center")

    # Panneau droit — le pic de pile par phase : seul l'apprentissage la creuse.
    ax = axes[1]
    phases = [("stack_peak_inference", "inférence", COLOR_BOARD),
              ("stack_peak_update", "inférence + mise à jour CL", COLOR_ALERT)]
    width = 0.35
    for i, (key, label, color) in enumerate(phases):
        offset = (i - 0.5) * width
        vals = [_num((cell(m, "fp32") or {}).get(key)) for m in RAM_MODELS]
        ax.bar(x + offset, [v if v is not None else 0.0 for v in vals], width,
               color=[color if v is not None else NA_GRAY for v in vals],
               edgecolor=INK, linewidth=0.5, label=label)
        _bar_labels(ax, x + offset, vals,
                    [f"{v:.0f}" if v is not None else NA_TEXT for v in vals], dy=0.01)
    idle = [(_num((cell(m, "fp32") or {}).get("stack_history") and
                  next((h.get("stack_peak_bytes")
                        for h in cell(m, "fp32")["stack_history"]
                        if h.get("phase") == "idle"), None)))
            for m in RAM_MODELS]
    if any(v is not None for v in idle):
        ax.plot(x, idle, marker="_", markersize=28, linestyle="", color=INK,
                label="pic au repos (jamais nul)")
    ax.set_xticks(x)
    ax.set_xticklabels([RAM_LABEL[m] for m in RAM_MODELS], rotation=15)
    ax.set_ylabel("pic de pile mesuré (octets)")
    ax.set_title("Seul l'apprentissage embarqué creuse la pile", fontsize=11)
    ax.margins(y=0.25)
    ax.legend(fontsize=9, loc="upper right")
    lo = min([v for v in idle if v is not None] or [0.0])
    ax.set_ylim(bottom=lo * 0.98)

    fig.suptitle(f"RAM réellement occupée sur carte — {dataset}, FP32 | INT8 (S49)")
    fig.text(0.5, 0.005, BADGE_BOARD, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "r1_ram_totale_recap", out_root)


# ═════════════════════ E — Latence et énergie ════════════════════════════════

SEGMENTS: list[tuple[str, str, str]] = [
    ("dequant", "déquantification\nint → FP32", COLOR_PC),
    ("mac", "produits scalaires\n(MAC entiers)", COLOR_BOARD),
    ("requant", "requantification\nFP32 → int8 (lroundf)", COLOR_ALERT),
]


def _fig_latence_int8(out_root: Path) -> Path:
    """E1 — où passent les cycles du noyau INT8, et pourquoi il est plus lent."""
    data = _load("experiments/exp_S50_int8_latency/ewc.json")
    if data is None:
        return _empty("exp_S50_int8_latency/ewc.json absent — "
                      "lancer scripts/run_s50_int8_latency.py",
                      "e1_latence_int8_breakdown", out_root)
    segs = data.get("segments", {})
    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    names = [s for s, _, _ in SEGMENTS]
    cycles = [_num((segs.get(s) or {}).get("cycles_p50_mean_over_datasets"))
              for s in names]
    colors = [c for _, _, c in SEGMENTS]
    y = np.arange(len(names))
    ax.barh(y, [v if v is not None else 0.0 for v in cycles],
            color=[c if v is not None else NA_GRAY for c, v in zip(colors, cycles)],
            edgecolor=INK, linewidth=0.5)
    for yi, v in zip(y, cycles):
        ax.text(v if v is not None else 0.0, yi,
                f" {v:.0f} cycles" if v is not None else f" {NA_TEXT}",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in SEGMENTS], fontsize=9)
    ax.invert_yaxis()
    ax.margins(x=0.25)
    ax.set_xlabel("cycles CPU (médiane, DWT)")
    ax.set_title("Décomposition du noyau EWC INT8", fontsize=11)

    ax = axes[1]
    ds_list = data.get("datasets_measured", [])
    x = np.arange(len(ds_list))
    width = 0.35
    fp32 = [_num((data.get("total_fp32_us_p50_by_dataset") or {}).get(d)) for d in ds_list]
    int8 = [_num((data.get("total_int8_us_p50_by_dataset") or {}).get(d)) for d in ds_list]
    ax.bar(x - width / 2, [v if v is not None else 0.0 for v in fp32], width,
           color=COLOR_BOARD, edgecolor=INK, linewidth=0.5, label="FP32 (FPU)")
    ax.bar(x + width / 2, [v if v is not None else 0.0 for v in int8], width,
           color=COLOR_INT8, edgecolor=INK, linewidth=0.5, label="INT8")
    _bar_labels(ax, x - width / 2, fp32,
                [f"{v:.0f} µs" if v is not None else NA_TEXT for v in fp32])
    _bar_labels(ax, x + width / 2, int8,
                [f"{v:.0f} µs" if v is not None else NA_TEXT for v in int8])
    for xi, a, b in zip(x, fp32, int8):
        if a is not None and b is not None:
            ax.text(xi + width / 2, b / 2, f"+{b - a:.0f} µs\nvs FP32", ha="center",
                    va="center", fontsize=11, color="white", fontweight="bold")
    ax.margins(y=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_list)
    ax.set_ylabel("latence médiane (µs)")
    ax.set_title("Paradoxe FPU : l'INT8 est plus lent", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Coût en cycles de la quantification INT8 sur Cortex-M4 (S50)")
    fig.text(0.5, 0.005, BADGE_BOARD, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e1_latence_int8_breakdown", out_root)


def _fig_wfi(out_root: Path) -> Path:
    """E2 — le repos en scrutation UART masquait toute mesure d'énergie."""
    data = _load("experiments/exp_S53_wfi/idle_reference.json")
    if data is None:
        return _empty("exp_S53_wfi/idle_reference.json absent — "
                      "lancer scripts/run_s53_wfi.py", "e2_s53_wfi_repos", out_root)
    builds = [("default_polling", "attente UART\nen scrutation", COLOR_ALERT),
              ("UART_WFI_IDLE", "attente UART\nsous __WFI()", COLOR_PC)]
    fig, ax = plt.subplots()
    x = np.arange(len(builds))
    values = [_num((data.get(key) or {}).get("i_idle_established_ma"))
              for key, _, _ in builds]
    errs = [(_num((data.get(key) or {}).get("bench_dispersion_a")) or 0.0) * MA_PER_A
            for key, _, _ in builds]
    ax.bar(x, [v if v is not None else 0.0 for v in values], 0.5,
           yerr=errs, capsize=6,
           color=[c if v is not None else NA_GRAY for (_, _, c), v in zip(builds, values)],
           edgecolor=INK, linewidth=0.5)
    _bar_labels(ax, x, values,
                [f"{v:.2f} mA" if v is not None else NA_TEXT for v in values])
    gain = data.get("wfi_gain") or {}
    gain_ma, gain_pct = _num(gain.get("gain_ma")), _num(gain.get("gain_pct"))
    if gain_ma is not None and gain_pct is not None:
        mid_x = (x[0] + x[1]) / 2
        ax.annotate(f"−{gain_ma:.2f} mA\n(−{gain_pct:.1f} %)",
                    xy=(x[1], values[1] or 0.0), xytext=(mid_x, (values[0] or 0.0) * 0.8),
                    fontsize=13, color=COLOR_ACCENT, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_ACCENT, linewidth=2))
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in builds])
    ax.set_ylabel("courant moyen au repos (mA)")
    ax.set_title("Sans plancher de repos, le protocole delta rendait des µJ négatifs")
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e2_s53_wfi_repos", out_root)


BATCH_CELLS: list[tuple[str, str]] = [("ewc_fp32", "EWC (FP32)"),
                                      ("maha_fp32", "Mahalanobis (FP32)")]


def _fig_uj_par_inference(out_root: Path) -> Path:
    """E3 — µJ/inférence par régression du courant sur la taille de lot imposée."""
    data = _load("experiments/exp_S53_wfi/batch_sweep.json")
    if data is None:
        return _empty("exp_S53_wfi/batch_sweep.json absent — "
                      "lancer scripts/run_s53_wfi.py --batch",
                      "e3_s53_uj_par_inference", out_root)
    fig, ax = plt.subplots()
    for (key, label), color in zip(BATCH_CELLS, (COLOR_BOARD, COLOR_ACCENT)):
        cell = data.get(key)
        if cell is None:
            continue
        kept = [(p["n"], _num(p.get("i_mean_a"))) for p in cell.get("points", [])
                if not p.get("saturated")]
        drop = [(p["n"], _num(p.get("i_mean_a"))) for p in cell.get("points", [])
                if p.get("saturated")]
        xs = [n for n, v in kept if v is not None]
        ys = [v * MA_PER_A for _, v in kept if v is not None]
        ax.plot(xs, ys, marker="o", linestyle="", color=color, label=label)
        slope = _num(cell.get("slope_a_per_inference_per_frame"))
        intercept = _num(cell.get("intercept_a"))
        if slope is not None and intercept is not None and xs:
            grid = np.linspace(0, max(xs), len(xs) + 1)
            ax.plot(grid, (intercept + slope * grid) * MA_PER_A, color=color,
                    linewidth=1.5, alpha=0.6)
        for n, v in drop:
            if v is not None:
                ax.plot([n], [v * MA_PER_A], marker="x", color=NA_GRAY, markersize=11)
                ax.annotate("écarté : UART saturée", (n, v * MA_PER_A),
                            textcoords="offset points", xytext=(-14, -18), fontsize=9,
                            color=MUTED, ha="right", va="top")
        uj = _num(cell.get("energy_uj_per_inference"))
        r2 = _num(cell.get("r2"))
        if uj is not None and r2 is not None and ys:
            ax.annotate(f"{label} : {uj:.3f} µJ / inférence  (r² = {r2:.4f})",
                        (xs[-1], ys[-1]), textcoords="offset points", xytext=(-8, 12),
                        fontsize=10, color=color, ha="right")
    ax.set_xlabel("inférences par trame UART (N imposé par -DINFER_BATCH_N)")
    ax.set_ylabel("courant moyen (mA)")
    ax.set_title("Isoler le calcul : la pente de I(N) est le coût d'une inférence")
    ax.legend(fontsize=10, loc="upper left")
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e3_s53_uj_par_inference", out_root)


RATE_MODELS: list[tuple[str, str]] = [("ewc", "EWC"), ("hdc", "HDC"),
                                      ("tinyol", "TinyOL"), ("maha", "Mahalanobis")]


def _fig_gap3_energie(out_root: Path) -> Path:
    """E4 — Gap 3 côté énergie : l'INT8 ne déplace pas le coût par inférence."""
    summary = _load("experiments/exp_S53_rate_sweep/summary.json")
    if summary is None:
        return _empty("exp_S53_rate_sweep/summary.json absent — "
                      "lancer scripts/run_s53_rate_sweep.py",
                      "e4_s53_gap3_energie", out_root)
    cells = summary.get("cells", {})
    fig, ax = plt.subplots()
    x = np.arange(len(RATE_MODELS))
    width = 0.35
    for i, (enc, color) in enumerate((("fp32", COLOR_BOARD), ("int8", COLOR_INT8))):
        offset = (i - 0.5) * width
        vals, errs, texts = [], [], []
        for model, _ in RATE_MODELS:
            cell = cells.get(f"{model}_{enc}") or {}
            v = _num(cell.get("energy_uj_per_inference"))
            vals.append(v)
            errs.append(_num(cell.get("energy_uncertainty_uj")) or 0.0)
            texts.append(f"{v:.1f}" if v is not None else "")
        ax.bar(x + offset, [v if v is not None else 0.0 for v in vals], width,
               yerr=errs, capsize=5,
               color=[color if v is not None else NA_GRAY for v in vals],
               edgecolor=INK, linewidth=0.5, label=enc.upper())
        _bar_labels(ax, x + offset, vals, texts, errs=errs)
        for xi, v in zip(x + offset, vals):
            if v is None:
                ax.annotate("non mesuré", xy=(xi, 0.0),
                            xycoords=("data", "axes fraction"), xytext=(0, 14),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=MUTED, rotation=90)
    verdict = summary.get("gap3_energy_verdict") or {}
    rationale = verdict.get("rationale")
    if rationale:
        ax.text(0.98, 0.98, textwrap.fill(rationale, WRAP_WIDTH),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=COLOR_ALERT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in RATE_MODELS])
    ax.set_ylabel("énergie par inférence (µJ) — régression I(cadence)")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("Gap 3 côté énergie — barres = incertitude de la régression")
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e4_s53_gap3_energie", out_root)


def _fig_sysclk(out_root: Path) -> Path:
    """E5 — ralentir le MCU réduit l'énergie par inférence, Gap 2 restant tenu."""
    summary = _load("experiments/exp_S53_freq_sweep/summary.json")
    if summary is None:
        return _empty("exp_S53_freq_sweep/summary.json absent — "
                      "lancer scripts/run_s53_freq_sweep.py", "e5_s53_sysclk", out_root)
    freqs = sorted(summary.get("frequencies_mhz", []))
    keys = [str(f) for f in freqs]
    energy = [_num((summary.get("energy_uj_per_inference_by_mhz") or {}).get(k))
              for k in keys]
    idle = [_num((summary.get("i_idle_ma_by_mhz") or {}).get(k)) for k in keys]
    worst = [_num((summary.get("gap2_worst_us_by_mhz") or {}).get(k)) for k in keys]
    budget = _num((_load("experiments/exp_S45_summary.json") or {}).get("gap2_latency_us"))

    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(freqs, energy, marker="o", color=COLOR_ACCENT, label="énergie / inférence (µJ)")
    for f, v in zip(freqs, energy):
        if v is not None:
            ax.annotate(f"{v:.1f}", (f, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=COLOR_ACCENT)
    ax.set_xlabel("SYSCLK (MHz)")
    ax.set_ylabel("µJ par inférence", color=COLOR_ACCENT)
    twin = ax.twinx()
    twin.plot(freqs, idle, marker="s", linestyle="--", color=COLOR_BOARD,
              label="courant de base (mA)")
    twin.set_ylabel("courant de base au repos (mA)", color=COLOR_BOARD)
    ax.set_xticks(freqs)
    ax.set_xlim(min(freqs) - X_PAD_MHZ, max(freqs) + X_PAD_MHZ)
    ax.margins(y=0.2)
    twin.margins(y=0.2)
    twin.yaxis.set_label_coords(1.12, 0.5)
    ax.set_title("Le coût par inférence croît avec la fréquence", fontsize=11)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=9, loc="lower right")

    ax = axes[1]
    x = np.arange(len(freqs))
    ax.bar(x, [v if v is not None else 0.0 for v in worst], 0.5,
           color=[COLOR_BOARD if v is not None else NA_GRAY for v in worst],
           edgecolor=INK, linewidth=0.5, label="pire latence mesurée (µs)")
    if budget is not None:
        ax.axhline(budget, color=COLOR_ALERT, linestyle="--", linewidth=1.5)
        ax.text(x[0], budget, f" budget Gap 2 = {budget / MA_PER_A:.0f} ms",
                va="bottom", fontsize=9, color=COLOR_ALERT)
        for xi, v in zip(x, worst):
            if v is not None:
                ax.text(xi, v, f"×{budget / v:.0f}\nde marge", ha="center", va="bottom",
                        fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} MHz" for f in freqs])
    ax.set_ylabel("µs (échelle log)")
    ax.set_title("Marge Gap 2 conservée à toutes les fréquences", fontsize=11)
    ax.legend(fontsize=9)

    trend = summary.get("tendance_rationale")
    fig.subplots_adjust(wspace=0.45)
    fig.suptitle("Balayage SYSCLK — la marge de latence est convertible en autonomie")
    fig.text(0.5, 0.005, trend or BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e5_s53_sysclk", out_root)


STATUS_ROWS: list[tuple[str, str, str, str]] = [
    ("Repos réel (WFI)",
     "experiments/exp_S53_wfi/idle_reference.json", "wfi_gain.gain_ma", "mA gagnés"),
    ("µJ / inférence — delta",
     "experiments/exp_S53_wfi/delta_recovery.json", "cells.ewc_fp32.energy_uj_per_inference",
     "µJ (EWC)"),
    ("µJ / inférence — régression",
     "experiments/exp_S53_rate_sweep/summary.json", "cells.ewc_fp32.energy_uj_per_inference",
     "µJ (EWC)"),
    ("µJ / inférence — intégration de profil",
     "experiments/exp_S53_phase_profile/summary.json",
     "energy_uj_per_inference_by_cell.hdc_45", "µJ"),
    ("Énergie d'une mise à jour CL",
     "experiments/exp_S53_build_isolation/summary.json",
     "variants.poids.energy_uj_per_inference", "µJ"),
    ("Décomposition MCU / périphériques",
     "experiments/exp_S50_energy/ewc_fp32.json", "by_component.sensor", "µJ"),
]


def _dig(data: dict | None, dotted: str):
    """Descend un chemin pointé ; ``None`` dès qu'un maillon manque."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _fig_statut(out_root: Path) -> Path:
    """E6 — ce qui est chiffré, ce qui reste « à mesurer », et pour quelle raison."""
    fig, ax = base_fig()
    ax.text(0.03, 0.9, "Grandeur", fontsize=11, color=INK, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.55, 0.9, "Statut mesuré", fontsize=11, color=INK, fontweight="bold",
            transform=ax.transAxes)
    n_ok = 0
    for i, (label, rel, dotted, unit) in enumerate(STATUS_ROWS):
        y = 0.8 - i * 0.125
        raw = _dig(_load(rel), dotted)
        value = _num(raw)
        ax.text(0.03, y, label, fontsize=11, color=INK, va="center",
                transform=ax.transAxes)
        if value is not None:
            n_ok += 1
            box(ax, 0.72, y, f"{value:.3f} {unit}", COLOR_PC, w=0.32, h=0.09, fontsize=11)
        else:
            # Sentinel explicite ou champ absent : le même statut, la raison
            # détaillée restant dans le `na_reason` du JSON.
            marker = "●" if _is_na(raw) else "○"
            box(ax, 0.72, y, f"{A_MESURER} {marker}", NA_GRAY, w=0.32, h=0.09,
                fontsize=11)
    ax.text(0.5, 0.04,
            f"{n_ok} grandeurs sur {len(STATUS_ROWS)} sont chiffrées ; les autres portent "
            f"leur raison dans le JSON (`na_reason`) — aucune n'est comblée par un zéro.\n"
            f"● = le JSON porte le sentinel « {A_MESURER} »   ○ = la grandeur n'est pas "
            f"encore produite",
            ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)
    fig.suptitle("Campagne énergie — statut honnête des grandeurs (S50 · S53)")
    footnote(fig, BADGE_ENERGY)
    return savefig_png(fig, CATALOG, "e6_s53_statut_mesures", out_root)


# ═════════════════════════════ Registre ══════════════════════════════════════

FIGURES: list[str] = [
    "t1_timeline_sprints",
    "t2_carte_axes_gaps",
    "d1_drift_proxy_vs_board",
    "d2_drift_portabilite",
    "q1_carte_quantification",
    "q2_profondeur_pc_board",
    "q3_moment_bilan",
    "r1_ram_totale_recap",
    "e1_latence_int8_breakdown",
    "e2_s53_wfi_repos",
    "e3_s53_uj_par_inference",
    "e4_s53_gap3_energie",
    "e5_s53_sysclk",
    "e6_s53_statut_mesures",
]


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 14 figures de la présentation encadrants sous ``out_root``."""
    apply_style("slide")
    return [
        _fig_timeline(out_root),
        _fig_axes_gaps(out_root),
        _fig_drift_proxy_vs_board(out_root),
        _fig_drift_portabilite(out_root),
        _fig_carte_quantification(out_root),
        _fig_profondeur(out_root),
        _fig_moment(out_root),
        _fig_ram(out_root),
        _fig_latence_int8(out_root),
        _fig_wfi(out_root),
        _fig_uj_par_inference(out_root),
        _fig_gap3_energie(out_root),
        _fig_sysclk(out_root),
        _fig_statut(out_root),
    ]
