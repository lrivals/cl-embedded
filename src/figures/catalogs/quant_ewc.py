"""Catalogue `quantization_ewc` — variante *présentation, EWC seul, sans Q15*.

Retravail des figures de quantification pour le fil de présentation FP32/INT8
(cf. ``docs/context/quantization_presentation.md``) : on **isole la tête EWC** et
on **retire les résidus Q15** des figures qui en contenaient. Sortie dans un
dossier **séparé** ``docs/figures/quantization_ewc/`` — les catalogues
``quantization/{pedagogy,impact,pipeline}`` d'origine ne sont pas modifiés.

Aucune donnée n'est recalculée ni écrite en dur : ce module **réutilise les
loaders** de :mod:`quant_impact` et :mod:`quant_pedagogy` (mêmes JSON sources,
mêmes checkpoints), et ne fait que retracer un sous-ensemble EWC sans la
stratégie ``q15``. Deux figures déjà EWC-pures et sans Q15 sont réexportées
telles quelles (``paradoxe_latence``, ``mapping_affine_int8``).

Sources (lecture seule, via les loaders réutilisés) : ``exp_S28_PC_ewc_hdc/``,
``exp_S29_board_int8/``, ``exp_S36_summary.json``, ``exp_S39_ablation/``,
``exp_S39_quant_sweep/``, ``exp_S40_board_v2/``, checkpoint EWC S39-matched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.catalogs import quant_impact as qi
from src.figures.catalogs import quant_pedagogy as qp
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, STRATEGY_LABELS_FR, savefig_png

CATALOG = "quantization_ewc"

# Réutilise la liste canonique de datasets et les badges de plateforme.
DATASETS = qi.DATASETS
BADGE_PC, BADGE_BOARD, BADGE_EMU = qi.BADGE_PC, qi.BADGE_BOARD, qi.BADGE_EMU


# ── E1 — Δmétrique par stratégie (EWC), sans Q15 ─────────────────────────────

def _fig_e1() -> plt.Figure:
    """I1 restreinte à la tête EWC et aux 3 stratégies INT8 (q15 retirée)."""
    strategies = [
        ("int8_qat", BADGE_PC, "AUROC — EWC S28"),
        ("int8_ptq_legacy", BADGE_BOARD, "F1 — board S36"),
        ("int8_v2", BADGE_EMU, "F1 — émulateur S39"),
    ]
    vals: dict[str, list[float | None]] = {s: [] for s, _, _ in strategies}
    for ds in DATASETS:
        qat, _ = qi._qat_delta(ds)
        abl = qi._ablation(ds)
        pc_v2 = None
        if abl and isinstance(abl["f1_fp32"], (int, float)):
            f0 = float(abl["f1_fp32"])
            pcc = abl["ladder"].get("per_channel_int8")
            pc_v2 = float(pcc) - f0 if isinstance(pcc, (int, float)) else None
        vals["int8_qat"].append(qat)
        vals["int8_ptq_legacy"].append(qi._ptq_board_delta(ds))
        vals["int8_v2"].append(pc_v2)

    fig, ax = plt.subplots()
    x = np.arange(len(DATASETS))
    width = 0.25
    for k, (strat, badge, metric) in enumerate(strategies):
        offs = (k - 1.0) * width
        ys = [v if isinstance(v, (int, float)) else np.nan for v in vals[strat]]
        ax.bar(x + offs, ys, width, color=STRATEGY_COLORS[strat],
               label=f"{STRATEGY_LABELS_FR[strat]} · {badge} · {metric}")
        for xi, v in zip(x + offs, vals[strat]):
            if not isinstance(v, (int, float)):
                ax.text(xi, 0.0, "n/a", ha="center", va="bottom", fontsize=8,
                        color="#999999", rotation=90)
    ax.axhline(0.0, color="#555555", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.set_ylabel("Δ (métrique quantifiée − FP32 de référence)")
    ax.set_title("Tête EWC — Δmétrique vs FP32 par stratégie INT8 (Q15 hors périmètre)")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.01, 0.005,
             "Δ = métrique_quantifiée − métrique_FP32 (chargée depuis les JSON sources). "
             "Modèle : tête EWC uniquement ; badge = plateforme.",
             fontsize=8, color="#666666")
    return fig


# ── E2 — escalier d'ablation S39 (EWC), sans l'échelon Q15 ───────────────────

def _fig_e2() -> plt.Figure:
    """I2 sans l'échelon q15 : la démonstration s'arrête au per-channel INT8."""
    order = ["legacy_c", "fix_acc32", "per_tensor_calib", "per_channel_int8"]
    rung_labels = ["legacy C\n(1/128, int16)", "+ acc int32", "+ scale\ncalibré",
                   "+ per-channel"]
    ds_colors = {"pronostia": STRATEGY_COLORS["int8_v2"],
                 "monitoring": STRATEGY_COLORS["int8_qat"]}

    fig, ax = plt.subplots()
    xr = np.arange(len(order))
    for ds, color in ds_colors.items():
        abl = qi._ablation(ds)
        if abl is None:
            continue
        ys = [abl["ladder"].get(s) for s in order]
        ys = [float(v) if isinstance(v, (int, float)) else np.nan for v in ys]
        ax.plot(xr, ys, "-o", color=color, ms=9, label=f"{ds.upper()} (F1)")
        f0 = abl["f1_fp32"]
        if isinstance(f0, (int, float)):
            ax.axhline(float(f0), color=color, linestyle="--", linewidth=1.4, alpha=0.6)
            ax.text(xr[-1], float(f0), f"  FP32 {ds.upper()}", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(xr)
    ax.set_xticklabels(rung_labels)
    ax.set_ylabel("F1_faulty")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Ablation Sprint 39 (tête EWC) : à quel facteur attribuer la perte de F1 ?")
    ax.annotate("le scale calibré (vs 1/128 figé)\nrécupère l'essentiel de la perte",
                xy=(2.0, 0.9), xytext=(0.15, 0.55), textcoords="axes fraction",
                fontsize=10, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.legend(loc="center right")
    fig.text(0.01, 0.005,
             "Source : exp_S39_ablation/ (émulateur bit-exact, badge « émulé PC »). "
             "Q15 retiré : le fil s'arrête au per-channel INT8.",
             fontsize=8, color="#666666")
    return fig


# ── E3 — RAM Gap 3 isolée sur EWC ────────────────────────────────────────────

def _fig_e3() -> plt.Figure:
    """I4 isolée : gain RAM INT8 de la SEULE tête EWC, par dataset (plus de HDC/TinyOL/Maha)."""
    labels, ratios = [], []
    for ds in DATASETS:
        r = qi._ram_ratio("ewc", "exp_S28_PC_ewc_hdc", ds)
        if r is None:
            continue
        labels.append(ds.upper())
        ratios.append(r)

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x, ratios, 0.55, color=STRATEGY_COLORS["int8_v2"])
    ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--")
    ax.text(len(labels) - 0.5, 1.0, " FP32 ×1", va="bottom", ha="right",
            fontsize=10, color="#555555")
    for xi, r in zip(x, ratios):
        ax.text(xi, r, f"×{r:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gain RAM poids EWC (FP32 / INT8)")
    ax.set_ylim(0.0, 4.5)
    ax.set_title("Gap 3 — gain RAM INT8 de la tête EWC, invariant au dataset")
    ax.text(0.5, 0.92,
            f"Budget board NUCLEO-F439ZI : {qi.RAM_BUDGET_BYTES // 1024} Ko SRAM "
            "(.bss mesurée ≪ budget, Sprints 29/36)",
            transform=ax.transAxes, ha="center", fontsize=10, color="#333333")
    fig.text(0.01, 0.005,
             "Modèle : tête EWC uniquement (INT8 ×4). Source : exp_S28_PC_ewc_hdc "
             "(champ ram_ratio). Variante EWC de la figure multi-modèle ram_gap3.",
             fontsize=8, color="#666666")
    return fig


# ── E4 — QAT vs PTQ vs v2 (EWC), résidu couleur Q15 retiré ───────────────────

def _fig_e4() -> plt.Figure:
    """I6 sur EWC : la série board réel v2 reprend la couleur int8_v2 (plus de violet Q15)."""
    ds_list = ["pronostia", "monitoring"]
    series = [
        ("FP32 (référence)", "fp32", BADGE_EMU),
        ("PTQ legacy (board)", "int8_ptq_legacy", BADGE_BOARD),
        ("v2 per-channel calibré (émulé)", "int8_v2", BADGE_EMU),
        ("v2 per-channel (board réel)", "int8_v2", BADGE_BOARD),
    ]

    def _values(ds: str) -> list[float | str | None]:
        fp = qi._quant_sweep_metric("ewc", ds, "fp32")
        try:
            summ, _ = qi.load_experiment("experiments/exp_S36_summary.json")
            cell = summ["results"].get(ds, {}).get("5feat", {})
            ptq = qi.metric_or_na(cell.get("board_frozen_int8", {}), "metric_value")
        except (FileNotFoundError, KeyError):
            ptq = None
        v2_emu = qi._quant_sweep_metric("ewc", ds, "int8_perchannel")
        v2_board = qi._board_v2_f1(ds, "frozen")
        return [fp, ptq, v2_emu, v2_board]

    # Hachure la 4e série (board réel) pour la distinguer de la 3e (émulé), même couleur.
    hatches = [None, None, None, "//"]

    fig, ax = plt.subplots()
    x = np.arange(len(ds_list))
    width = 0.2
    for k, (label, strat, badge) in enumerate(series):
        offs = (k - 1.5) * width
        vals = [_values(ds)[k] for ds in ds_list]
        ys = [v if isinstance(v, (int, float)) else np.nan for v in vals]
        ax.bar(x + offs, ys, width, color=STRATEGY_COLORS[strat], hatch=hatches[k],
               edgecolor="white", label=f"{label} · {badge}")
        for xi, v in zip(x + offs, vals):
            if v == qi.A_MESURER:
                ax.text(xi, 0.05, "à mesurer", ha="center", va="bottom", fontsize=8.5,
                        color="#777777", rotation=90)
            elif not isinstance(v, (int, float)):
                ax.text(xi, 0.05, "n/a", ha="center", va="bottom", fontsize=8,
                        color="#999999", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in ds_list])
    ax.set_ylabel("F1_faulty")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Tête EWC — même format INT8, résultats opposés : PTQ legacy ✗  vs  v2 calibré ✓")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.005,
             "Sources : exp_S39_quant_sweep/ (émulé), exp_S36_summary.json (PTQ board), "
             "exp_S40_board_v2/ (v2 board réel ; « à mesurer » si non flashé). "
             "Hachure = mesure board réelle. Q15 hors périmètre.",
             fontsize=8, color="#666666")
    return fig


# ── E5 — erreur de quantification sur les poids EWC, sans Q15 ─────────────────

def _fig_e5(w, note: str) -> plt.Figure:
    """P4 restreinte à legacy vs v2 (courbe q15 retirée)."""
    layers = [w.w1, w.w2, w.w3]
    keep = ["int8_ptq_legacy", "int8_v2"]
    errs: dict[str, np.ndarray] = {}
    for wmat in layers:
        paths = qp._dequant_paths(wmat)
        for strat in keep:
            e = np.abs(wmat - paths[strat]).ravel()
            errs[strat] = np.concatenate([errs.get(strat, np.empty(0)), e])

    means = {s: float(np.mean(e)) for s, e in errs.items()}
    if not (means["int8_ptq_legacy"] > means["int8_v2"]):
        print("[figures][ATTENTION] E5 : ordre erreur legacy > v2 NON reproduit "
              f"({means}) — investiguer (constat S3904).")

    floor = 1e-9
    all_pos = np.concatenate([e[e > 0] for e in errs.values()])
    bins = np.logspace(np.log10(max(all_pos.min(), floor)), np.log10(all_pos.max() * 1.5), 45)

    fig, ax = plt.subplots()
    for strat in keep:
        e = np.clip(errs[strat], floor, None)
        ax.hist(e, bins=bins, histtype="stepfilled", alpha=0.45,
                color=STRATEGY_COLORS[strat],
                label=f"{STRATEGY_LABELS_FR[strat]} — |erreur| moyenne {means[strat]:.2e}")
        ax.axvline(max(means[strat], floor), color=STRATEGY_COLORS[strat],
                   linestyle="--", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Erreur absolue |w − déquant(quant(w))|  (échelle log)")
    ax.set_ylabel("Nombre de poids")
    ax.set_title("Erreur de quantification sur les poids EWC réels : legacy figé vs v2 calibré")
    n_clamp = int(sum(np.sum(np.abs(m) > 127.0 / 128.0) for m in layers))
    plur = "s" if n_clamp > 1 else ""
    ax.annotate(
        f"queue du legacy = clamp :\n{n_clamp} poids |w| > 127/128\nsaturé{plur} à ±127 (erreur ≫ pas de grille)",
        xy=(float(errs["int8_ptq_legacy"].max()), 1.0),
        xytext=(0.62, 0.55), textcoords="axes fraction",
        fontsize=11, color=STRATEGY_COLORS["int8_ptq_legacy"],
        arrowprops=dict(arrowstyle="->", color=STRATEGY_COLORS["int8_ptq_legacy"]),
    )
    ax.legend(loc="upper left")
    fig.text(0.01, 0.005,
             f"{note} — quantification via l'émulateur bit-exact S39 (int8_c_emulation.py). "
             "Q15 hors périmètre.",
             fontsize=8, color="#666666")
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère la variante EWC-only (sans Q15) sous ``out_root/quantization_ewc/``.

    Réexporte P1 (``mapping_affine_int8``) et I5 (``paradoxe_latence``) tels
    quels (déjà EWC-purs, sans Q15) pour former un jeu de présentation cohérent.
    """
    w, ewc_note = qp._load_ewc_weights()
    return [
        savefig_png(_fig_e1(), CATALOG, "metrique_par_strategie_ewc", out_root),
        savefig_png(_fig_e2(), CATALOG, "ablation_perte_f1_ewc", out_root),
        savefig_png(_fig_e3(), CATALOG, "ram_gap3_ewc", out_root),
        savefig_png(qi._fig_i5(), CATALOG, "paradoxe_latence_ewc", out_root),
        savefig_png(_fig_e4(), CATALOG, "qat_vs_ptq_resultats_ewc", out_root),
        savefig_png(_fig_e5(w, ewc_note), CATALOG, "erreur_quantification_poids_ewc", out_root),
        savefig_png(qp._fig_p1(w, ewc_note), CATALOG, "mapping_affine_int8_ewc", out_root),
    ]
