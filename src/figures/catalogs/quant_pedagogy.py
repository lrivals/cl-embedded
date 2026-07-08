"""Catalogue `quantization/pedagogy` — ce que la quantification fait aux données (S4203).

6 figures conceptuelles (P1–P6) expliquant le *mécanisme* de chaque stratégie de
quantification, pour un public non spécialiste. P1/P2/P4/P5 utilisent de **vrais
tenseurs** du projet (checkpoints EWC + Mahalanobis), quantifiés via l'émulateur
bit-exact ``src/utils/int8_c_emulation.py`` (S3902) — aucune réimplémentation.
P3/P6 sont des schémas de mécanisme, étiquetés « illustration ».

Couleurs par stratégie : ``src.figures.style.STRATEGY_COLORS`` (cohérence avec
S4204/S4205 et ``docs/context/quantization_strategies.md``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.figures.loaders import ROOT, record_source
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, STRATEGY_LABELS_FR, savefig_png
from src.utils.int8_c_emulation import (
    EWCHeadWeights,
    _quant_weight,
    _sat8,
    _trunc_to_int,
    _weight_scales,
)

CATALOG = "quantization/pedagogy"

# Tenseurs réels du projet (provenance affichée en note de bas de figure)
EWC_CKPT = ROOT / "experiments/exp_S39_matched/checkpoints/ewc_pronostia_5feat.pt"
MAHA_CKPT = (
    ROOT / "experiments/exp_S35_board_5feat_mahalanobis_paderborn/checkpoints/mahalanobis_task0.pkl"
)

# Labels FR regroupés (rend une future option --lang en mécanique, cf. S4201)
LABELS: dict[str, str] = {
    "p1_title": "Mapping affine FP32 → INT8 : 256 niveaux sur un axe continu",
    "p1_hist": "Poids FP32 réels (tête EWC, toutes couches)",
    "p1_grid": "Niveaux représentables INT8 (échelle legacy s = 1/128)",
    "p1_clamp": "Zone de clamp : |w| > 127·s",
    "p1_zero": "zero-point z = 0\n(quantification symétrique)",
    "p2_title": "Grilles de quantification : 256 niveaux (INT8) vs 65 536 (Q15)\nsur la dynamique réelle de Σ⁻¹ (Mahalanobis, Paderborn)",
    "p2_lane_int8": "INT8 — 256 niveaux",
    "p2_lane_q15": "Q15 — 65 536 niveaux",
    "p2_vals": "Coefficients |Σ⁻¹| réels",
    "p2_zoom": "Zoom près de zéro : les petites valeurs sont écrasées en INT8",
    "p3_title": "QAT vs PTQ : quand la quantification rencontre l'entraînement",
    "p4_title": "Erreur de quantification  w − déquant(quant(w))  sur les poids EWC réels",
    "p4_xlabel": "Erreur absolue |w − déquant(quant(w))|  (échelle log)",
    "p4_ylabel": "Nombre de poids",
    "p5_title": "Le cas « grande dynamique » : Σ⁻¹ face aux grilles INT8 et Q15",
    "p5_xlabel": "Coefficients de Σ⁻¹ triés par magnitude (rang)",
    "p5_ylabel": "|coefficient|  (échelle log)",
    "p6_title": "Trois forwards d'un même neurone : FP32, fake-quant (QAT), firmware INT8",
    "illustration": "Illustration — schéma de principe, aucune donnée expérimentale.",
}


# ── Chargement des tenseurs réels ────────────────────────────────────────────

def _load_ewc_weights() -> tuple[EWCHeadWeights, str]:
    """Poids FP32 réels de la tête EWC (checkpoint S39-matched, Pronostia 5 features)."""
    import torch

    record_source(EWC_CKPT)
    ck = torch.load(EWC_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    note = (
        "Source : poids réels — tête EWC 5→32→16→2, Pronostia 5 features "
        f"({EWC_CKPT.relative_to(ROOT)})"
    )
    return EWCHeadWeights.from_state_dict(sd), note


def _load_sigma_inv() -> tuple[np.ndarray, str]:
    """Σ⁻¹ réelle du détecteur Mahalanobis (Paderborn, grande dynamique)."""
    record_source(MAHA_CKPT)
    with open(MAHA_CKPT, "rb") as f:
        det = pickle.load(f)
    si = np.asarray(det.sigma_inv_, dtype=np.float64)
    note = (
        "Source : Σ⁻¹ réelle — Mahalanobis 5 features, Paderborn "
        f"({MAHA_CKPT.relative_to(ROOT)})"
    )
    return si, note


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=8, color="#666666", ha="left")


def _dequant_paths(wmat: np.ndarray) -> dict[str, np.ndarray]:
    """Poids déquantifiés par les 3 chemins de l'émulateur S39 (aucune réimpl.).

    - ``int8_ptq_legacy`` : ``SAT8(trunc(w·128))/128`` (cf. ``ewc_int8_from_fp32``)
    - ``int8_v2``        : per-channel calibré, round symétrique, 8 bits
    - ``q15``            : per-channel calibré, 16 bits
    """
    legacy = _sat8(_trunc_to_int(wmat * 128.0)) / 128.0
    s8 = _weight_scales(wmat, "per_channel", 8)
    v2 = _quant_weight(wmat, s8, 8) * s8[:, None]
    s15 = _weight_scales(wmat, "per_channel", 16)
    q15 = _quant_weight(wmat, s15, 16) * s15[:, None]
    return {"int8_ptq_legacy": legacy, "int8_v2": v2, "q15": q15}


# ── Petits helpers de schéma (P3/P6) ─────────────────────────────────────────

def _box(ax: plt.Axes, cx: float, cy: float, text: str, color: str,
         w: float = 0.17, h: float = 0.16, fontsize: int = 11) -> None:
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=mcolors.to_rgba(color, 0.12), edgecolor=color, linewidth=2,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            transform=ax.transAxes, wrap=True)


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
           color: str = "#555555") -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=18, color=color, linewidth=1.8,
    ))


# ── P1 — mapping affine FP32→INT8 ────────────────────────────────────────────

def _fig_p1(w: EWCHeadWeights, note: str) -> plt.Figure:
    all_w = np.concatenate([w.w1.ravel(), w.w2.ravel(), w.w3.ravel()])
    s = 1.0 / 128.0  # échelle legacy du firmware (ewc_int8_from_fp32)
    qmax = 127
    bound = qmax * s
    n_clamp = int(np.sum(np.abs(all_w) > bound))
    c_grid = STRATEGY_COLORS["int8_ptq_legacy"]

    fig, ax = plt.subplots()
    ax.hist(all_w, bins=70, color="#B0BEC5", edgecolor="white", zorder=2,
            label=LABELS["p1_hist"])
    # Grille des 256 niveaux représentables (k·s, k ∈ [−127, 127])
    levels = np.arange(-qmax, qmax + 1) * s
    ax.vlines(levels, 0, ax.get_ylim()[1] * 0.35, color=c_grid, linewidth=0.4,
              alpha=0.35, zorder=1)
    ax.vlines([-bound, bound], 0, ax.get_ylim()[1], color=c_grid, linewidth=2,
              linestyle="--", zorder=3)
    xmax = float(np.abs(all_w).max()) * 1.12
    ax.set_xlim(-xmax, xmax)
    # Zones de clamp au-delà de ±127·s
    ymax = ax.get_ylim()[1]
    for lo, hi in ((bound, xmax), (-xmax, -bound)):
        ax.axvspan(lo, hi, color=c_grid, alpha=0.10, zorder=0)
    plur = "s" if n_clamp > 1 else ""
    ax.text(0.985, 0.32,
            f"{LABELS['p1_clamp']}\n→ saturé{plur} à ±127\n({n_clamp} poids réel{plur} clampé{plur})",
            ha="right", fontsize=11, color=c_grid, transform=ax.transAxes)
    # zero-point
    ax.annotate(LABELS["p1_zero"], xy=(0.0, ymax * 0.82), xytext=(-xmax * 0.62, ymax * 0.86),
                fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.set_xlabel("Valeur du poids (axe réel continu)")
    ax.set_ylabel("Nombre de poids")
    ax.set_title(LABELS["p1_title"])
    ax.legend(loc="upper left")

    # Inset : zoom sur quelques niveaux, pas de quantification s annoté
    axins = fig.add_axes([0.60, 0.52, 0.27, 0.26])
    zmax = 6 * s
    axins.hist(all_w[(all_w >= 0) & (all_w <= zmax)], bins=24, color="#B0BEC5",
               edgecolor="white")
    zlevels = np.arange(0, 7) * s
    axins.vlines(zlevels, 0, axins.get_ylim()[1], color=c_grid, linewidth=1.2, alpha=0.8)
    y_arrow = axins.get_ylim()[1] * 0.85
    axins.annotate("", xy=(2 * s, y_arrow), xytext=(3 * s, y_arrow),
                   arrowprops=dict(arrowstyle="<->", color="#333333"))
    axins.text(2.5 * s, y_arrow * 1.03, f"s = 1/128 ≈ {s:.4f}", ha="center", fontsize=10)
    axins.set_xlim(0, zmax)
    axins.set_title("Zoom : pas de la grille", fontsize=11)
    axins.tick_params(labelsize=8)

    _footnote(fig, f"{note} — grille : {STRATEGY_LABELS_FR['int8_ptq_legacy']}.")
    return fig


# ── P2 — grilles INT8 vs Q15 sur grande dynamique ────────────────────────────

def _fig_p2(si: np.ndarray, note: str) -> plt.Figure:
    vals = np.abs(si.ravel())
    vals_nz = vals[vals > 0]
    m = float(vals.max())
    # Pas des grilles = échelle per-tensor de l'émulateur (max|·|/qmax)
    s8 = float(_weight_scales(si, "per_tensor", 8)[0])
    s15 = float(_weight_scales(si, "per_tensor", 16)[0])
    q8 = _quant_weight(si, _weight_scales(si, "per_tensor", 8), 8)
    n_zero8 = int(np.sum((q8 == 0) & (si != 0)))
    c8 = STRATEGY_COLORS["int8_ptq_legacy"]
    c15 = STRATEGY_COLORS["q15"]

    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1.0, 1.15])
    fig.subplots_adjust(hspace=0.45)

    # Panneau 1 : pleine dynamique, deux « couloirs » de niveaux
    int8_levels = np.arange(0, 128) * s8
    ax1.vlines(int8_levels, 0.72, 1.28, color=c8, linewidth=0.6, alpha=0.7)
    # Q15 : 32 768 niveaux positifs — indiqués par une bande (indessinables un à un)
    ax1.axhspan(-0.28, 0.28, xmin=0, xmax=1, color=mcolors.to_rgba(c15, 0.12))
    ax1.text(m * 0.5, -0.45, f"grille quasi continue — pas s₁₅ ≈ {s15:.1f}",
             ha="center", va="top", color=c15, fontsize=11)
    ax1.plot(vals_nz, np.full_like(vals_nz, 1.0), "o", color="#212121", ms=6,
             zorder=5, label=LABELS["p2_vals"])
    ax1.plot(vals_nz, np.full_like(vals_nz, 0.0), "o", color="#212121", ms=6, zorder=5)
    ax1.set_yticks([1.0, 0.0])
    ax1.set_yticklabels([LABELS["p2_lane_int8"], LABELS["p2_lane_q15"]])
    ax1.set_ylim(-0.85, 1.55)
    ax1.set_xlim(0, m * 1.03)
    ax1.set_xlabel("|coefficient|  (même dynamique pour les deux grilles)")
    ax1.legend(loc="upper center")
    ax1.grid(False)

    # Panneau 2 : zoom près de zéro — l'écrasement INT8
    zmax = 3.2 * s8
    q15_levels = np.arange(0, int(zmax / s15) + 1) * s15
    ax2.vlines(q15_levels, -0.28, 0.28, color=c15, linewidth=0.3, alpha=0.5)
    ax2.vlines(np.arange(0, 4) * s8, 0.72, 1.28, color=c8, linewidth=2.0)
    in_zoom = vals_nz[vals_nz <= zmax]
    ax2.plot(in_zoom, np.full_like(in_zoom, 1.0), "o", color="#212121", ms=6, zorder=5)
    ax2.plot(in_zoom, np.full_like(in_zoom, 0.0), "o", color="#212121", ms=6, zorder=5)
    ax2.axvspan(0, s8 / 2, color=c8, alpha=0.12)
    ax2.text(s8 / 2, 1.45,
             f"← |x| < s₈/2 : arrondis à 0 en INT8 ({n_zero8}/{vals_nz.size} coefficients réels)",
             fontsize=11, color=c8)
    ax2.set_yticks([1.0, 0.0])
    ax2.set_yticklabels([f"INT8 (pas s₈ ≈ {s8:.0f})", f"Q15 (pas s₁₅ ≈ {s15:.1f})"])
    ax2.set_ylim(-0.55, 1.75)
    ax2.set_xlim(0, zmax)
    ax2.set_xlabel("|coefficient| — zoom sur [0, 3·s₈]")
    ax2.set_title(LABELS["p2_zoom"], fontsize=12)
    ax2.grid(False)

    fig.suptitle(LABELS["p2_title"])
    _footnote(fig, f"{note} — pas de grille : échelle per-tensor max|Σ⁻¹|/qmax (émulateur S39).")
    return fig


# ── P3 — QAT vs PTQ (schéma) ─────────────────────────────────────────────────

def _fig_p3() -> plt.Figure:
    c_qat = STRATEGY_COLORS["int8_qat"]
    c_ptq = STRATEGY_COLORS["int8_ptq_legacy"]
    c_neutral = "#607D8B"
    fig, ax = plt.subplots()
    ax.axis("off")

    xs = [0.10, 0.37, 0.64, 0.90]
    y_qat, y_ptq = 0.74, 0.26

    ax.text(0.02, y_qat + 0.17, "QAT — quantification pendant l'entraînement",
            fontsize=14, color=c_qat, fontweight="bold", transform=ax.transAxes)
    _box(ax, xs[0], y_qat, "Données\nd'entraînement", c_neutral)
    _box(ax, xs[1], y_qat, "Entraînement avec\nfake-quant :\nquant → déquant\ninséré dans le forward", c_qat, w=0.20)
    _box(ax, xs[2], y_qat, "Export des poids\nINT8 (scales appris)", c_qat)
    _box(ax, xs[3], y_qat, "Déploiement :\nmodèle déjà adapté\nà l'erreur ✓", c_qat)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + 0.09, y_qat, b - 0.10, y_qat)
    ax.text(xs[1], y_qat - 0.155, "le gradient « voit » l'erreur de quantification\n(straight-through estimator, STE)",
            ha="center", fontsize=11, color=c_qat, style="italic", transform=ax.transAxes)

    ax.text(0.02, y_ptq + 0.17, "PTQ — quantification après l'entraînement",
            fontsize=14, color=c_ptq, fontweight="bold", transform=ax.transAxes)
    _box(ax, xs[0], y_ptq, "Données\nd'entraînement", c_neutral)
    _box(ax, xs[1], y_ptq, "Entraînement\nFP32 classique\n(aucune erreur vue)", c_neutral, w=0.20)
    _box(ax, xs[2], y_ptq, "Conversion INT8\naprès coup\n(± calibration)", c_ptq)
    _box(ax, xs[3], y_ptq, "Déploiement :\nle modèle découvre\nl'erreur ✗", c_ptq)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + 0.09, y_ptq, b - 0.10, y_ptq)
    ax.text(xs[2], y_ptq - 0.155, "sans calibration du scale (legacy 1/128),\nl'erreur peut être destructrice (Sprints 29/36/39)",
            ha="center", fontsize=11, color=c_ptq, style="italic", transform=ax.transAxes)

    ax.set_title(LABELS["p3_title"])
    _footnote(fig, LABELS["illustration"])
    return fig


# ── P4 — distribution de l'erreur de quantification ──────────────────────────

def _fig_p4(w: EWCHeadWeights, note: str) -> plt.Figure:
    layers = [w.w1, w.w2, w.w3]
    errs: dict[str, np.ndarray] = {}
    for wmat in layers:
        for strat, dq in _dequant_paths(wmat).items():
            e = np.abs(wmat - dq).ravel()
            errs[strat] = np.concatenate([errs.get(strat, np.empty(0)), e])

    means = {s: float(np.mean(e)) for s, e in errs.items()}
    # Garde-fou S3904 : legacy ≫ calibré ≫ Q15 — ne pas maquiller si non reproduit
    if not (means["int8_ptq_legacy"] > means["int8_v2"] > means["q15"]):
        print("[figures][ATTENTION] P4 : ordre erreur legacy > v2 > q15 NON reproduit "
              f"({means}) — investiguer (constat S3904).")

    floor = 1e-9
    all_pos = np.concatenate([e[e > 0] for e in errs.values()])
    bins = np.logspace(np.log10(max(all_pos.min(), floor)), np.log10(all_pos.max() * 1.5), 45)
    order = ["int8_ptq_legacy", "int8_v2", "q15"]

    fig, ax = plt.subplots()
    for strat in order:
        e = np.clip(errs[strat], floor, None)
        ax.hist(e, bins=bins, histtype="stepfilled", alpha=0.45,
                color=STRATEGY_COLORS[strat],
                label=f"{STRATEGY_LABELS_FR[strat]} — |erreur| moyenne {means[strat]:.2e}")
        ax.axvline(max(means[strat], floor), color=STRATEGY_COLORS[strat],
                   linestyle="--", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(LABELS["p4_xlabel"])
    ax.set_ylabel(LABELS["p4_ylabel"])
    ax.set_title(LABELS["p4_title"])
    # Pourquoi le legacy clampe : poids hors de la plage représentable ±127/128
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
    _footnote(fig, f"{note} — quantification via l'émulateur bit-exact S39 (int8_c_emulation.py).")
    return fig


# ── P5 — grande dynamique de Σ⁻¹ vs niveaux représentables ───────────────────

def _fig_p5(si: np.ndarray, note: str) -> plt.Figure:
    s8v = _weight_scales(si, "per_tensor", 8)
    s15v = _weight_scales(si, "per_tensor", 16)
    s8, s15 = float(s8v[0]), float(s15v[0])
    dq8 = np.abs((_quant_weight(si, s8v, 8) * s8v[:, None]).ravel())
    dq15 = np.abs((_quant_weight(si, s15v, 16) * s15v[:, None]).ravel())
    vals = np.abs(si.ravel())
    keep = vals > 0
    order = np.argsort(vals[keep])
    v_sorted = vals[keep][order]
    dq8_s, dq15_s = dq8[keep][order], dq15[keep][order]
    rank = np.arange(1, v_sorted.size + 1)
    c8 = STRATEGY_COLORS["int8_ptq_legacy"]
    c15 = STRATEGY_COLORS["q15"]

    y_floor = s15 / 5.0
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(rank, v_sorted, "o", color="#212121", ms=8, zorder=5,
            label="Valeurs réelles de |Σ⁻¹| (FP32)")
    ax.plot(rank, np.where(dq15_s > 0, dq15_s, y_floor), "P", color=c15, ms=9,
            alpha=0.85, zorder=4, label=f"{STRATEGY_LABELS_FR['q15']} — déquantifié")
    zero8 = dq8_s == 0
    ax.plot(rank[~zero8], dq8_s[~zero8], "x", color=c8, ms=10, mew=2.5, zorder=4,
            label="INT8 (per-tensor) — déquantifié")
    if zero8.any():
        ax.plot(rank[zero8], np.full(zero8.sum(), y_floor), "x", color=c8, ms=10,
                mew=2.5, zorder=4)
        ax.annotate(
            f"{int(zero8.sum())}/{v_sorted.size} coefficients → 0 en INT8\n(non représentables : |x| < s₈/2)\n→ distances de Mahalanobis collapsées (Sprint 34)",
            xy=(float(rank[zero8].mean()), y_floor), xytext=(0.08, 0.42),
            textcoords="axes fraction", fontsize=12, color=c8,
            arrowprops=dict(arrowstyle="->", color=c8),
        )
    ax.axhline(s8, color=c8, linestyle="--", linewidth=1.6,
               label=f"Premier niveau INT8 : s₈ = max|Σ⁻¹|/127 ≈ {s8:.0f}")
    ax.axhline(s15, color=c15, linestyle="--", linewidth=1.6,
               label=f"Premier niveau Q15 : s₁₅ = max|Σ⁻¹|/32767 ≈ {s15:.1f}")
    ax.axhspan(y_floor / 2, s8 / 2, color=c8, alpha=0.08)
    ax.set_ylim(y_floor / 2, float(v_sorted.max()) * 3)
    ax.set_xlabel(LABELS["p5_xlabel"])
    ax.set_ylabel(LABELS["p5_ylabel"])
    ax.set_title(LABELS["p5_title"])
    ax.legend(loc="lower right", fontsize=11)
    _footnote(fig, f"{note} — grilles per-tensor de l'émulateur S39 ; dynamique ≈ "
                   f"{v_sorted.max() / v_sorted.min():.0e} entre plus petit et plus grand coefficient.")
    return fig


# ── P6 — trois forwards d'un neurone (schéma) ────────────────────────────────

def _fig_p6() -> plt.Figure:
    c_fp = STRATEGY_COLORS["fp32"]
    c_qat = STRATEGY_COLORS["int8_qat"]
    c_fw = STRATEGY_COLORS["int8_v2"]
    fig, ax = plt.subplots()
    ax.axis("off")

    lanes = [
        (0.82, c_fp, STRATEGY_LABELS_FR["fp32"], [
            ("x\n(float32)", 0.10, 0.10),
            ("w·x + b\n(float32)", 0.34, 0.14),
            ("ReLU", 0.58, 0.10),
            ("h (float32)", 0.85, 0.12),
        ], "tout le chemin en float32 — la référence"),
        (0.50, c_qat, STRATEGY_LABELS_FR["int8_qat"], [
            ("x", 0.07, 0.07),
            ("quant → déquant\n(fake-quant)", 0.24, 0.15),
            ("w̃·x + b\navec w̃ = déquant(quant(w))", 0.51, 0.20),
            ("ReLU", 0.73, 0.09),
            ("h (float32)", 0.90, 0.11),
        ], "l'erreur de quantification est simulée en float — le gradient la voit (STE)"),
        (0.18, c_fw, "Firmware board (poids INT8 stockés)", [
            ("x → quant Q7/Q15\n(entier)", 0.13, 0.15),
            ("MAC entier\nw_int8 · x_q\n(acc int16/int32)", 0.38, 0.16),
            ("déquant\nsur FPU\n(× scale)", 0.61, 0.12),
            ("ReLU", 0.77, 0.08),
            ("h", 0.91, 0.06),
        ], "poids stockés int8 (RAM ÷4) ; la fidélité dépend du scale (1/128 figé vs calibré)"),
    ]

    for y, color, label, boxes, caption in lanes:
        ax.text(0.01, y + 0.12, label, fontsize=13, color=color, fontweight="bold",
                transform=ax.transAxes)
        for i, (text, cx, bw) in enumerate(boxes):
            _box(ax, cx, y, text, color, w=bw, h=0.13, fontsize=10)
            if i:
                px, pw = boxes[i - 1][1], boxes[i - 1][2]
                _arrow(ax, px + pw / 2 + 0.005, y, cx - bw / 2 - 0.005, y, color=color)
        ax.text(0.5, y - 0.115, caption, ha="center", fontsize=10.5, style="italic",
                color="#444444", transform=ax.transAxes)

    ax.set_title(LABELS["p6_title"])
    _footnote(fig, LABELS["illustration"])
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère P1–P6 sous ``out_root/quantization/pedagogy/`` ; retourne les chemins."""
    w, ewc_note = _load_ewc_weights()
    si, maha_note = _load_sigma_inv()
    return [
        savefig_png(_fig_p1(w, ewc_note), CATALOG, "mapping_affine_int8", out_root),
        savefig_png(_fig_p2(si, maha_note), CATALOG, "grilles_int8_vs_q15", out_root),
        savefig_png(_fig_p3(), CATALOG, "qat_vs_ptq", out_root),
        savefig_png(_fig_p4(w, ewc_note), CATALOG, "erreur_quantification_poids", out_root),
        savefig_png(_fig_p5(si, maha_note), CATALOG, "dynamique_sigma_inv", out_root),
        savefig_png(_fig_p6(), CATALOG, "fakequant_forward", out_root),
    ]
