"""Catalogue `article_ewc_int8` — figures de l'article standalone « EWC INT8 sur MCU » (S4009).

Source **unique** : ``experiments/exp_S40_article_metrics/summary.json``, produit en lecture
seule par ``scripts/aggregate_article_ewc.py`` (S4008). Chaque cellule y porte déjà
``value``/``source_json``/``platform``/``na_reason`` : le catalogue ne fait que tracer, il
n'ouvre aucun JSON de sprint directement et **n'écrit aucun littéral numérique de résultat**
(garde AST ``test_no_hardcoded_results``).

Dix figures, la numérotation ``figN_`` étant celle attendue par l'article et les tests :

    fig1_parity_fp32_pc_board     parité FP32 PC↔carte (gelé / en ligne)
    fig2_latency_gap2             latences Gap 2 + décomposition du kernel INT8
    fig3_ablation_ladder          échelle d'ablation legacy_c → … → q15 (émulé PC)
    fig4_int8_recovery_board      effondrement puis récupération mesurée carte
    fig5_pareto_ram_f1_latency    Pareto RAM des poids × F1 × latence
    fig6_quant_moment             moment de quantification : avant / après / les deux
    fig7_quant_depth_packing      profondeur sub-INT8 : AUROC vs bits, effet du bit-packing
    fig8_ram_total                RAM totale .data + .bss + pic de pile, % du budget
    fig9_energy_estimators        énergie : estimateurs côte à côte, jamais fusionnés
    fig10_compute_cost            MACs/FLOPs/BOPs théoriques vs latence mesurée

Conventions d'honnêteté : **plein = mesuré carte**, **hachuré = émulé PC / théorique**, gris =
non mesuré (``None`` ou sentinel ``"à mesurer"``) — jamais 0. En fin de ``build()``, les PNG
sont copiés à l'identique vers ``docs/article/ewc_int8_mcu/figures/`` et
``docs/figures/sprint40_article/`` : une seule source de vérité, plus de dérive de copie.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import A_MESURER, load_experiment
from src.figures.registry import register_catalog
from src.figures.style import (
    DEFAULT_OUT_ROOT,
    STRATEGY_COLORS,
    apply_style,
    savefig_png,
)

CATALOG = "article_ewc_int8"
SUMMARY = "experiments/exp_S40_article_metrics/summary.json"

ROOT = Path(__file__).resolve().parents[3]
#: Répertoires miroirs — copie bit-identique en fin de build (anti-dérive).
SYNC_DIRS: list[Path] = [
    ROOT / "docs" / "article" / "ewc_int8_mcu" / "figures",
    ROOT / "docs" / "figures" / "sprint40_article",
]

DATASETS: list[str] = ["monitoring", "pronostia"]
DS_LABEL: dict[str, str] = {"monitoring": "Monitoring (D2)", "pronostia": "Pronostia (D4)"}

C_FP32 = STRATEGY_COLORS["fp32"]
C_LEGACY = STRATEGY_COLORS["int8_ptq_legacy"]
C_V2 = STRATEGY_COLORS["int8_v2"]
C_Q15 = STRATEGY_COLORS["q15"]
C_QAT = STRATEGY_COLORS["int8_qat"]
NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"

#: Style « mesuré carte » vs « émulé PC / théorique » — repris des conventions du dépôt.
BOARD_KW: dict = {"hatch": None, "edgecolor": INK, "linewidth": 0.8}
EMU_KW: dict = {"hatch": "///", "edgecolor": INK, "linewidth": 0.8}

CYCLES_PER_US = 1e6  # conversion Hz → MHz pour l'axe des cycles (mise en page, pas un résultat)


# ── Accès traçable aux cellules de l'agrégat ─────────────────────────────────

def _summary() -> dict | None:
    """Agrégat S4008 ou ``None`` s'il n'a pas encore été généré."""
    try:
        data, _ = load_experiment(SUMMARY)
    except FileNotFoundError:
        return None
    return data


def val(summary: dict | None, *path: str):
    """Valeur d'une cellule ``[dataset][axe][clé]`` de l'agrégat.

    Retourne ``None`` si la cellule est absente ou non mesurée, et conserve le sentinel
    ``"à mesurer"`` tel quel — jamais 0 par défaut (convention Sprints 33/40).
    """
    node = summary
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def num(v) -> float | None:
    """Valeur numérique traçable, ou ``None`` (le sentinel « à mesurer » n'est pas un nombre)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _bar(ax, xs, vals, color, label, width, kw, fmt="{:.3f}"):
    """Barres avec cellules non mesurées en gris + étiquette « à mesurer »."""
    for x, v in zip(xs, vals):
        n = num(v)
        if n is None:
            ax.bar(x, 0, width=width, color=NA_GRAY, **kw)
            ax.text(x, 0, f"\n{A_MESURER}", ha="center", va="bottom",
                    fontsize=7, color=MUTED, rotation=90)
            continue
        ax.bar(x, n, width=width, color=color, label=label, **kw)
        ax.text(x, n, fmt.format(n), ha="center", va="bottom", fontsize=7, color=INK)
        label = None  # une seule entrée de légende par série
    return ax


def _dedup_legend(ax, **kwargs) -> None:
    """Légende sans doublons (les barres réutilisent le même label par série)."""
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    if seen:
        ax.legend(seen.values(), seen.keys(), **kwargs)


def _no_data(ax, message: str) -> None:
    """Axe neutre quand l'agrégat manque — aucune valeur inventée pour combler."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_axis_off()


# ── fig1 — Parité FP32 PC ↔ carte ────────────────────────────────────────────

def fig1_parity(s: dict | None, out_root: Path) -> Path:
    fig, ax = plt.subplots()
    if s is None:
        _no_data(ax, "agrégat absent : lancer scripts/aggregate_article_ewc.py")
        return savefig_png(fig, CATALOG, "fig1_parity_fp32_pc_board", out_root)
    x = np.arange(len(DATASETS))
    w = 0.35
    _bar(ax, x - w / 2, [val(s, d, "performance", "board_frozen_parity_rate") for d in DATASETS],
         C_FP32, "gelé (poids figés)", w, BOARD_KW, fmt="{:.4f}")
    _bar(ax, x + w / 2, [val(s, d, "performance", "board_online_parity_rate") for d in DATASETS],
         C_V2, "en ligne (avec MAJ CL)", w, BOARD_KW, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("taux de parité PC ↔ carte")
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Parité FP32 PC ↔ NUCLEO-F439ZI (mesuré carte)")
    _dedup_legend(ax, loc="lower right")
    return savefig_png(fig, CATALOG, "fig1_parity_fp32_pc_board", out_root)


# ── fig2 — Latences Gap 2 + décomposition du kernel INT8 ─────────────────────

def fig2_latency(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if s is None:
        for a in axes:
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig2_latency_gap2", out_root)

    ax = axes[0]
    x = np.arange(len(DATASETS))
    w = 0.27
    _bar(ax, x - w, [val(s, d, "performance", "board_frozen_latency_us_p50") for d in DATASETS],
         C_FP32, "inférence FP32 (carte)", w, BOARD_KW, fmt="{:.0f}")
    _bar(ax, x, [val(s, d, "latency", "total_int8_us_p50") for d in DATASETS],
         C_V2, "inférence INT8 v2 (carte)", w, BOARD_KW, fmt="{:.0f}")
    _bar(ax, x + w, [val(s, d, "performance", "board_online_latency_us_p50") for d in DATASETS],
         C_QAT, "inférence + MAJ CL (carte)", w, BOARD_KW, fmt="{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("latence P50 (µs)")
    ax.set_title("Latences mesurées — budget Gap 2 = 100 ms")
    _dedup_legend(ax, fontsize=8)

    ax = axes[1]
    segments = ["dequant", "mac", "requant"]
    seg_label = {"dequant": "déquantification", "mac": "MAC entier", "requant": "requantification"}
    seg_color = {"dequant": C_FP32, "mac": C_V2, "requant": C_LEGACY}
    bottom = np.zeros(len(DATASETS))
    for seg in segments:
        vals = [num(val(s, d, "latency", f"segment_{seg}_cycles_p50")) for d in DATASETS]
        heights = np.array([v if v is not None else 0.0 for v in vals])
        ax.bar(np.arange(len(DATASETS)), heights, bottom=bottom, width=0.5,
               color=seg_color[seg], label=seg_label[seg], **BOARD_KW)
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="white")
        bottom = bottom + heights
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("cycles DWT (P50)")
    ax.set_title("Décomposition du kernel INT8 v2 (mesuré carte)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig2_latency_gap2", out_root)


# ── fig3 — Échelle d'ablation (émulé PC) ─────────────────────────────────────

LADDER = ["legacy_c", "fix_acc32", "per_tensor_calib", "per_channel", "q15"]


def fig3_ablation(s: dict | None, out_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if s is None:
        _no_data(ax, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig3_ablation_ladder", out_root)
    x = np.arange(len(LADDER))
    for d, color in zip(DATASETS, (C_V2, C_QAT)):
        ys = [num(val(s, d, "ablation_int8", f"ladder_{step}_f1")) for step in LADDER]
        mask = [i for i, y in enumerate(ys) if y is not None]
        ax.plot([x[i] for i in mask], [ys[i] for i in mask], marker="o",
                color=color, label=f"{DS_LABEL[d]} (émulé PC)", linestyle="--")
        ref = num(val(s, d, "ablation_int8", "f1_fp32_reference"))
        if ref is not None:
            ax.axhline(ref, color=color, alpha=0.3, linewidth=1.0)
            ax.text(x[-1], ref, f" FP32 {ref:.4f}", fontsize=7, color=color, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER, rotation=20, ha="right")
    ax.set_ylabel("F1 (classe faulty)")
    ax.set_title("Ablation du kernel INT8 — émulateur bit-exact (PC)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig3_ablation_ladder", out_root)


# ── fig4 — Effondrement puis récupération mesurée carte ──────────────────────

def fig4_recovery(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(10.0, 4.2), sharey=True)
    if s is None:
        for a in np.atleast_1d(axes):
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig4_int8_recovery_board", out_root)
    series = [
        ("FP32 (carte)", "performance", "board_frozen_f1_faulty", C_FP32, BOARD_KW),
        ("INT8 legacy (carte)", "performance", "board_frozen_int8_legacy_f1_faulty",
         C_LEGACY, BOARD_KW),
        ("INT8 per-channel (émulé)", "ablation_int8", "scheme_int8_perchannel_metric",
         C_V2, EMU_KW),
        ("INT8 per-channel v2 (carte)", "recovery_board", "per_channel_frozen_f1_faulty",
         C_V2, BOARD_KW),
        ("Q15 (carte)", "recovery_board", "q15_frozen_f1_faulty", C_Q15, BOARD_KW),
    ]
    for ax, d in zip(np.atleast_1d(axes), DATASETS):
        for i, (label, axis, key, color, kw) in enumerate(series):
            _bar(ax, [i], [val(s, d, axis, key)], color, label, 0.6, kw, fmt="{:.4f}")
        ax.set_xticks(np.arange(len(series)))
        ax.set_xticklabels([lbl for lbl, *_ in series], rotation=30, ha="right", fontsize=8)
        ax.set_title(DS_LABEL[d])
    np.atleast_1d(axes)[0].set_ylabel("F1 (classe faulty)")
    fig.suptitle("Effondrement de la PTQ naïve, puis récupération — plein : carte · hachuré : émulé")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig4_int8_recovery_board", out_root)


# ── fig5 — Pareto RAM des poids × F1 × latence ───────────────────────────────

def fig5_pareto(s: dict | None, out_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if s is None:
        _no_data(ax, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig5_pareto_ram_f1_latency", out_root)
    points = [
        ("FP32", "scheme_fp32_ram_weights_bytes", "scheme_fp32_metric", C_FP32, "o"),
        ("INT8 legacy", "scheme_int8_legacy_ram_weights_bytes", "scheme_int8_legacy_metric",
         C_LEGACY, "s"),
        ("INT8 per-channel", "scheme_int8_perchannel_ram_weights_bytes",
         "scheme_int8_perchannel_metric", C_V2, "^"),
        ("Q15", "scheme_q15_ram_weights_bytes", "scheme_q15_metric", C_Q15, "D"),
    ]
    for d, marker_edge in zip(DATASETS, (INK, MUTED)):
        for label, ram_key, f1_key, color, marker in points:
            ram = num(val(s, d, "ablation_int8", ram_key))
            f1 = num(val(s, d, "ablation_int8", f1_key))
            if ram is None or f1 is None:
                continue
            ax.scatter(ram, f1, color=color, marker=marker, s=70, edgecolor=marker_edge,
                       label=f"{label} (émulé PC)", zorder=3)
        # Le point réellement flashé : kernel v2 calibré, RAM et F1 mesurés sur la carte.
        ram_b = num(val(s, d, "recovery_board", "per_channel_frozen_ram_weights_quant_bytes"))
        f1_b = num(val(s, d, "recovery_board", "per_channel_frozen_f1_faulty"))
        if ram_b is not None and f1_b is not None:
            ax.scatter(ram_b, f1_b, color=C_V2, marker="*", s=190, edgecolor=INK,
                       label="INT8 v2 (mesuré carte)", zorder=4)
            ax.annotate(DS_LABEL[d], (ram_b, f1_b), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=MUTED)
    ax.set_xlabel("RAM des poids (octets)")
    ax.set_ylabel("F1 (classe faulty)")
    ax.set_title("Compromis RAM × métrique — étoile : point réellement flashé")
    _dedup_legend(ax, fontsize=7, loc="lower right")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig5_pareto_ram_f1_latency", out_root)


# ── fig6 — Moment de quantification ──────────────────────────────────────────

MOMENTS = ["fp32", "before", "after", "both"]
MOMENT_LABEL = {"fp32": "FP32", "before": "avant (QAT)", "after": "après (PTQ calibrée)",
                "both": "les deux"}
MOMENT_COLOR = {"fp32": C_FP32, "before": C_QAT, "after": C_V2, "both": C_Q15}


def fig6_moment(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if s is None:
        for a in axes:
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig6_quant_moment", out_root)

    ax = axes[0]
    x = np.arange(len(DATASETS))
    w = 0.2
    for i, m in enumerate(MOMENTS):
        _bar(ax, x + (i - 1.5) * w, [val(s, d, "moment", f"{m}_auroc") for d in DATASETS],
             MOMENT_COLOR[m], MOMENT_LABEL[m], w, EMU_KW, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("AUROC")
    ax.set_title("Moment de quantification — émulé PC")
    _dedup_legend(ax, fontsize=8, loc="lower right")

    ax = axes[1]
    w = 0.35
    _bar(ax, x - w / 2, [val(s, d, "moment", "board_both_f1_after_board") for d in DATASETS],
         C_V2, "« après » seule (carte)", w, BOARD_KW, fmt="{:.4f}")
    _bar(ax, x + w / 2, [val(s, d, "moment", "board_both_f1_faulty") for d in DATASETS],
         C_Q15, "« les deux » QAT→PTQ (carte)", w, BOARD_KW, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("F1 (classe faulty)")
    ax.set_title("A/B mesuré sur carte")
    _dedup_legend(ax, fontsize=8, loc="lower right")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig6_quant_moment", out_root)


# ── fig7 — Profondeur sub-INT8 et bit-packing ────────────────────────────────

#: Profondeurs du balayage S47, dans l'ordre décroissant de bits. `ternaire` et `binaire`
#: partagent l'axe avec les grilles linéaires mais sont des modes distincts ({-1,0,1}, {-1,1}) :
#: l'axe est donc catégoriel, pas numérique.
DEPTH_TAGS = ["int8", "int6", "int4", "int3", "int2", "ternaire", "binaire"]
DEPTH_LABEL = {"int8": "8 b", "int6": "6 b", "int4": "4 b", "int3": "3 b", "int2": "2 b",
               "ternaire": "ternaire", "binaire": "binaire"}


def fig7_depth(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if s is None:
        for a in axes:
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig7_quant_depth_packing", out_root)

    ax = axes[0]
    xd = np.arange(len(DEPTH_TAGS))
    for d, color in zip(DATASETS, (C_V2, C_QAT)):
        for gran, style in (("per_tensor", ":"), ("per_channel", "-")):
            ys = [num(val(s, d, "depth_pc", f"{tag}_{gran}_delta_auroc")) for tag in DEPTH_TAGS]
            idx = [i for i, y in enumerate(ys) if y is not None]
            if not idx:
                continue
            ax.plot([xd[i] for i in idx], [ys[i] for i in idx], marker="o", linestyle=style,
                    color=color, label=f"{DS_LABEL[d]} · {gran}")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(xd)
    ax.set_xticklabels([DEPTH_LABEL[t] for t in DEPTH_TAGS], fontsize=8, rotation=20, ha="right")
    ax.set_xlabel("profondeur des poids (grilles linéaires, puis modes ternaire/binaire)")
    ax.set_ylabel("Δ AUROC vs FP32")
    ax.set_title("Profondeur × granularité — émulé PC")
    ax.legend(fontsize=7)

    # Le `.bss` total masque l'effet (le gain se compte en centaines d'octets sur ~100 Ko) :
    # on trace donc directement les octets ÉCONOMISÉS par le bit-packing, qui sont le résultat.
    ax = axes[1]
    bits_axis = ["4", "2", "1"]
    x = np.arange(len(bits_axis))
    w = 0.35
    for i, d in enumerate(DATASETS):
        _bar(ax, x + (i - 0.5) * w,
             [val(s, d, "depth_board", f"bits{b}_per_channel_bss_saved_by_packing")
              for b in bits_axis],
             (C_V2 if i == 0 else C_QAT), DS_LABEL[d], w, BOARD_KW, fmt="{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} bit(s)" for b in bits_axis])
    ax.set_ylabel("octets de `.bss` économisés par le packing")
    ax.set_title("Le gain RAM sub-INT8 n'existe que packé (mesuré carte)")
    _dedup_legend(ax, fontsize=8)
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig7_quant_depth_packing", out_root)


# ── fig8 — RAM totale décomposée ─────────────────────────────────────────────

def fig8_ram(s: dict | None, out_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    if s is None:
        _no_data(ax, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig8_ram_total", out_root)
    labels, comps = [], []
    for d in DATASETS:
        for enc in ("fp32", "int8"):
            labels.append(f"{DS_LABEL[d]}\n{enc}")
            comps.append([
                num(val(s, d, "ram", f"{enc}_board_data")),
                num(val(s, d, "ram", f"{enc}_board_bss")),
                num(val(s, d, "ram", f"{enc}_board_stack_peak_update")),
            ])
    x = np.arange(len(labels))
    parts = [("`.data`", C_FP32), ("`.bss`", C_V2), ("pic de pile (MAJ)", C_QAT)]
    bottom = np.zeros(len(labels))
    for i, (label, color) in enumerate(parts):
        h = np.array([c[i] if c[i] is not None else 0.0 for c in comps])
        ax.bar(x, h, bottom=bottom, width=0.55, color=color, label=label, **BOARD_KW)
        bottom = bottom + h
    budget = num(val(s, DATASETS[0], "ram", "ram_budget_bytes"))
    for i, d in enumerate(DATASETS):
        for j, enc in enumerate(("fp32", "int8")):
            pct = num(val(s, d, "ram", f"{enc}_board_total_pct_budget"))
            if pct is not None:
                ax.text(2 * i + j, bottom[2 * i + j], f"{pct:.1f} % du budget",
                        ha="center", va="bottom", fontsize=7, color=INK)
    if budget is not None:
        ax.axhline(budget, color=C_LEGACY, linestyle="--", linewidth=1.0,
                   label="budget SRAM (256 Ko)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("octets")
    ax.set_title("RAM totale mesurée = .data + .bss + pic de pile (carte)")
    _dedup_legend(ax, fontsize=8, loc="upper left")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig8_ram_total", out_root)


# ── fig9 — Énergie : estimateurs côte à côte, jamais fusionnés ───────────────

def fig9_energy(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if s is None:
        for a in axes:
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig9_energy_estimators", out_root)

    ax = axes[0]
    estimators = [
        ("régression I(cadence)", "regression", "ewc_fp32_energy_uj_per_inference",
         "ewc_int8_energy_uj_per_inference"),
        ("delta / repos WFI", "delta_wfi", "ewc_fp32_energy_uj_per_inference",
         "ewc_int8_energy_uj_per_inference"),
        ("lot INFER_BATCH_N", "batch", "ewc_fp32_energy_uj_per_inference", None),
    ]
    x = np.arange(len(estimators))
    w = 0.35
    _bar(ax, x - w / 2,
         [val(s, "energy", "estimators", e[1], "cells", e[2]) for e in estimators],
         C_FP32, "EWC FP32", w, BOARD_KW, fmt="{:.2f}")
    _bar(ax, x + w / 2,
         [val(s, "energy", "estimators", e[1], "cells", e[3]) if e[3] else None
          for e in estimators],
         C_V2, "EWC INT8", w, BOARD_KW, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in estimators], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("µJ par inférence")
    ax.set_title("Trois estimateurs — jamais fusionnés")
    _dedup_legend(ax, fontsize=8)

    ax = axes[1]
    freq_cells = (val(s, "energy", "estimators", "freq_sweep", "cells") or {})
    mhz = sorted({k.split("MHz")[0] for k in freq_cells if "MHz" in k}, key=float)
    energies, lats = [], []
    for m in mhz:
        energies.append(num(val(s, "energy", "estimators", "freq_sweep", "cells",
                                f"{m}MHz_energy_uj_per_inference")))
        lats.append(num(val(s, "energy", "estimators", "freq_sweep", "cells",
                            f"{m}MHz_ewc_latency_us_p50")))
    if any(e is not None for e in energies):
        xf = np.arange(len(mhz))
        ax.bar(xf, [e if e is not None else 0.0 for e in energies], width=0.5,
               color=C_QAT, label="énergie par inférence", **BOARD_KW)
        for i, e in enumerate(energies):
            if e is not None:
                ax.text(i, e, f"{e:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax2 = ax.twinx()
        ax2.plot(xf, [l if l is not None else np.nan for l in lats], marker="o",
                 color=C_LEGACY, label="latence EWC")
        ax2.set_ylabel("latence P50 (µs)")
        ax2.grid(False)
        ax.set_xticks(xf)
        ax.set_xticklabels([f"{m} MHz" for m in mhz])
        ax.set_ylabel("µJ par inférence")
        ax.set_title("Arbitrage fréquence : latence contre énergie")
        # Légende commune aux deux axes (la latence vit sur l'axe jumeau).
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(handles, labels, fontsize=8, loc="upper left")
    else:
        _no_data(ax, f"balayage de fréquence : {A_MESURER}")
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig9_energy_estimators", out_root)


# ── fig10 — Coût de calcul théorique vs latence mesurée ──────────────────────

def fig10_compute_cost(s: dict | None, out_root: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    if s is None:
        for a in axes:
            _no_data(a, "agrégat absent")
        return savefig_png(fig, CATALOG, "fig10_compute_cost", out_root)

    ax = axes[0]
    x = np.arange(len(DATASETS))
    w = 0.35
    _bar(ax, x - w / 2, [val(s, d, "compute_cost", "bops_fp32") for d in DATASETS],
         C_FP32, "BOPs FP32", w, EMU_KW, fmt="{:.0f}")
    _bar(ax, x + w / 2, [val(s, d, "compute_cost", "bops_int8") for d in DATASETS],
         C_V2, "BOPs INT8", w, EMU_KW, fmt="{:.0f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("BOPs par inférence (échelle log)")
    ratio = num(val(s, DATASETS[0], "compute_cost", "bops_ratio_fp32_over_int8"))
    suffix = f" — ratio théorique ×{ratio:.0f}" if ratio is not None else ""
    ax.set_title(f"Coût de calcul théorique{suffix}")
    _dedup_legend(ax, fontsize=8)

    ax = axes[1]
    _bar(ax, x - w / 2, [val(s, d, "latency", "total_fp32_us_p50") for d in DATASETS],
         C_FP32, "latence FP32 (carte)", w, BOARD_KW, fmt="{:.0f}")
    _bar(ax, x + w / 2, [val(s, d, "latency", "total_int8_us_p50") for d in DATASETS],
         C_V2, "latence INT8 (carte)", w, BOARD_KW, fmt="{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("latence P50 (µs)")
    ax.set_title("Latence réellement mesurée — l'INT8 coûte plus cher (FPU)")
    _dedup_legend(ax, fontsize=8)
    fig.tight_layout()
    return savefig_png(fig, CATALOG, "fig10_compute_cost", out_root)


# ── Build + synchronisation anti-dérive ──────────────────────────────────────

def _sync(paths: list[Path], out_root: Path) -> None:
    """Copie bit-identique des PNG vers l'article et le dossier historique du sprint.

    C'est ce qui supprime la dérive constatée (fig2/fig4/fig5 divergeaient entre le dossier
    de figures et celui de l'article, faute de copie automatisée).

    La synchronisation n'a lieu que pour une génération canonique (``out_root`` = le dossier
    de figures du dépôt) : une exécution vers un répertoire temporaire (tests, essais) ne
    doit jamais écraser les figures publiées.
    """
    if Path(out_root).resolve() != DEFAULT_OUT_ROOT.resolve():
        print("[figures] out_root non canonique → synchronisation ignorée")
        return
    for target in SYNC_DIRS:
        target.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, target / p.name)
    print(f"[figures] synchronisé vers {len(SYNC_DIRS)} répertoire(s) : "
          + ", ".join(str(t.relative_to(ROOT)) for t in SYNC_DIRS))


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 10 figures de l'article puis les synchronise (source unique)."""
    apply_style("manuscript")
    s = _summary()
    paths = [
        fig1_parity(s, out_root),
        fig2_latency(s, out_root),
        fig3_ablation(s, out_root),
        fig4_recovery(s, out_root),
        fig5_pareto(s, out_root),
        fig6_moment(s, out_root),
        fig7_depth(s, out_root),
        fig8_ram(s, out_root),
        fig9_energy(s, out_root),
        fig10_compute_cost(s, out_root),
    ]
    _sync(paths, out_root)
    return paths
