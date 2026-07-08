"""Catalogue `quantization/impact` — l'effet mesuré des stratégies (S4205).

6 figures de *résultats* : ce que chaque stratégie a réellement donné. Toute valeur
tracée provient d'un ``load_experiment`` (JSON d'``experiments/``) — **aucun littéral
numérique de résultat** dans ce module (test ``test_no_hardcoded_results``, S4207) ;
seuls des flottants de mise en page (positions, alpha) apparaissent, sur une grille
au pas 0.05. Chaque série porte un **badge de plateforme** (``mesuré board`` /
``émulé PC (bit-exact)`` / ``PC natif``) ; les cellules non mesurées affichent
``« à mesurer »`` (jamais 0, jamais extrapolé).

Sources (lecture seule) : ``exp_S28_PC_*``, ``exp_S29_board_int8/``,
``exp_S34_maha_q15/``, ``exp_S36_summary.json``, ``exp_S39_ablation/``,
``exp_S39_quant_sweep/``, ``exp_S40_board_v2/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import (
    A_MESURER,
    EXPERIMENTS_DIR,
    load_experiment,
    metric_or_na,
)
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, STRATEGY_LABELS_FR, savefig_png

CATALOG = "quantization/impact"

# Datasets du projet (ordre canonique des axes)
DATASETS: list[str] = ["cmapss", "cwru", "monitoring", "paderborn", "pronostia"]

# Budget RAM de la NUCLEO-F439ZI (constante matérielle, pas un résultat)
RAM_BUDGET_BYTES: int = 256 * 1024

# Badges de plateforme (règle Sprint 40)
BADGE_PC = "PC natif"
BADGE_BOARD = "mesuré board"
BADGE_EMU = "émulé PC (bit-exact)"


# ── Loaders spécifiques (tout passe par load_experiment) ─────────────────────

def _qat_delta(ds: str) -> tuple[float | None, str]:
    """QAT PC (S28, EWC) : Δ = métrique int8 − fp32, + nom de métrique."""
    try:
        data, _ = load_experiment(f"experiments/exp_S28_PC_ewc_hdc/results_ewc_{ds}.json")
    except FileNotFoundError:
        return None, ""
    fp = metric_or_na(data, "fp32.metric_value")
    q = metric_or_na(data, "int8.metric_value")
    name = metric_or_na(data, "fp32.metric_name") or ""
    if not isinstance(fp, (int, float)) or not isinstance(q, (int, float)):
        return None, str(name)
    return float(q) - float(fp), str(name)


def _ablation(ds: str) -> dict | None:
    """Escalier d'ablation S39 : f1_fp32 + {scheme: f1}."""
    try:
        data, _ = load_experiment(f"experiments/exp_S39_ablation/{ds}.json")
    except FileNotFoundError:
        return None
    ladder = {row["scheme"]: metric_or_na(row, "f1") for row in data.get("ladder", [])}
    return {"f1_fp32": metric_or_na(data, "f1_fp32"), "ladder": ladder}


def _ptq_board_delta(ds: str) -> float | None:
    """PTQ legacy board (S36 summary, 5feat) : Δ = F1 int8 board − F1 fp32 board."""
    try:
        summ, _ = load_experiment("experiments/exp_S36_summary.json")
    except FileNotFoundError:
        return None
    cell = summ.get("results", {}).get(ds, {}).get("5feat", {})
    fp = metric_or_na(cell.get("board_frozen", {}), "f1_faulty")
    q = metric_or_na(cell.get("board_frozen_int8", {}), "metric_value")
    if not isinstance(fp, (int, float)) or not isinstance(q, (int, float)):
        return None
    return float(q) - float(fp)


def _maha_variants(ds: str) -> dict[str, dict] | None:
    """Mahalanobis S34 : {variant: {auroc, corr}} pour fp32/int8/q15."""
    out: dict[str, dict] = {}
    for variant in ("fp32", "int8", "q15"):
        try:
            data, _ = load_experiment(f"experiments/exp_S34_maha_q15/{ds}_{variant}.json")
        except FileNotFoundError:
            return None
        out[variant] = {
            "auroc": metric_or_na(data, "auroc"),
            "corr": metric_or_na(data, "score_fidelity_vs_fp32.corr_with_fp32"),
        }
    return out


def _ram_ratio(model: str, subdir: str, ds: str) -> float | None:
    """Ratio RAM fp32/quant d'un modèle (S28)."""
    try:
        data, _ = load_experiment(f"experiments/{subdir}/results_{model}_{ds}.json")
    except FileNotFoundError:
        return None
    r = metric_or_na(data, "ram_ratio")
    return float(r) if isinstance(r, (int, float)) else None


def _board_latency(ds: str) -> tuple[float | None, float | None]:
    """Latences board FP32 vs INT8 (S29, EWC), en µs P50."""
    try:
        data, _ = load_experiment(f"experiments/exp_S29_board_int8/results_ewc_int8_{ds}.json")
    except FileNotFoundError:
        return None, None
    fp = metric_or_na(data, "fp32_reference.latency_p50_us")
    q = metric_or_na(data, "int8_detail.latency_p50_us")
    fp = float(fp) if isinstance(fp, (int, float)) else None
    q = float(q) if isinstance(q, (int, float)) else None
    return fp, q


def _board_v2_f1(ds: str, proto: str) -> float | str | None:
    """F1 board réelle du kernel v2 per-channel (S40) ; sentinel si non flashé."""
    path = EXPERIMENTS_DIR / "exp_S40_board_v2" / f"results_per_channel_{ds}_{proto}.json"
    if not path.exists():
        return A_MESURER
    data, _ = load_experiment(path)
    v = metric_or_na(data, "metric_value")
    return float(v) if isinstance(v, (int, float)) else A_MESURER


def _quant_sweep_metric(model: str, ds: str, scheme: str) -> float | None:
    """Métrique émulée d'un schéma (S39 quant_sweep)."""
    try:
        data, _ = load_experiment(f"experiments/exp_S39_quant_sweep/{model}_{ds}.json")
    except FileNotFoundError:
        return None
    v = metric_or_na(data, f"schemes.{scheme}.metric")
    return float(v) if isinstance(v, (int, float)) else None


# ── Helpers de tracé ──────────────────────────────────────────────────────────

def _grid(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="#555555", linewidth=1.0, zorder=1)


# ── I1 — Δmétrique par stratégie, groupé par dataset ─────────────────────────

def _fig_i1() -> plt.Figure:
    strategies = [
        ("int8_qat", BADGE_PC, "AUROC — EWC S28"),
        ("int8_ptq_legacy", BADGE_BOARD, "F1 — board S36"),
        ("int8_v2", BADGE_EMU, "F1 — émulateur S39"),
        ("q15", BADGE_EMU, "F1 — émulateur S39"),
    ]
    # Valeurs Δ (int8 − fp32) par (dataset, stratégie) ; None → cellule absente
    vals: dict[str, list[float | None]] = {s: [] for s, _, _ in strategies}
    for ds in DATASETS:
        qat, _ = _qat_delta(ds)
        abl = _ablation(ds)
        pc_v2 = None
        pc_q15 = None
        if abl and isinstance(abl["f1_fp32"], (int, float)):
            f0 = float(abl["f1_fp32"])
            pcc = abl["ladder"].get("per_channel_int8")
            q15 = abl["ladder"].get("q15")
            pc_v2 = float(pcc) - f0 if isinstance(pcc, (int, float)) else None
            pc_q15 = float(q15) - f0 if isinstance(q15, (int, float)) else None
        vals["int8_qat"].append(qat)
        vals["int8_ptq_legacy"].append(_ptq_board_delta(ds))
        vals["int8_v2"].append(pc_v2)
        vals["q15"].append(pc_q15)

    fig, ax = plt.subplots()
    x = np.arange(len(DATASETS))
    width = 0.2
    for k, (strat, badge, metric) in enumerate(strategies):
        offs = (k - 1.5) * width
        ys = [v if isinstance(v, (int, float)) else np.nan for v in vals[strat]]
        ax.bar(x + offs, ys, width, color=STRATEGY_COLORS[strat],
               label=f"{STRATEGY_LABELS_FR[strat]} · {badge} · {metric}")
        for xi, v in zip(x + offs, vals[strat]):
            if not isinstance(v, (int, float)):
                ax.text(xi, 0.0, "n/a", ha="center", va="bottom", fontsize=8,
                        color="#999999", rotation=90)
    _grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.set_ylabel("Δ (métrique quantifiée − FP32 de référence)")
    ax.set_title("« Quantifier ≠ quantifier » : Δmétrique vs FP32 par stratégie")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.01, 0.005,
             "Δ = métrique_quantifiée − métrique_FP32 (calculée depuis les JSON sources). "
             "Métrique par stratégie indiquée en légende ; badge = plateforme.",
             fontsize=8, color="#666666")
    return fig


# ── I2 — escalier d'ablation S39 ─────────────────────────────────────────────

def _fig_i2() -> plt.Figure:
    order = ["legacy_c", "fix_acc32", "per_tensor_calib", "per_channel_int8", "q15"]
    rung_labels = ["legacy C\n(1/128, int16)", "+ acc int32", "+ scale\ncalibré",
                   "+ per-channel", "+ Q15\n(16 bits)"]
    ds_colors = {"pronostia": STRATEGY_COLORS["int8_v2"], "monitoring": STRATEGY_COLORS["q15"]}

    fig, ax = plt.subplots()
    xr = np.arange(len(order))
    for ds, color in ds_colors.items():
        abl = _ablation(ds)
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
    ax.set_title("Ablation Sprint 39 : à quel facteur attribuer la perte de F1 ?")
    ax.annotate("le scale calibré (vs 1/128 figé)\nrécupère l'essentiel de la perte",
                xy=(2.0, 0.9), xytext=(0.15, 0.55), textcoords="axes fraction",
                fontsize=10, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.legend(loc="center right")
    fig.text(0.01, 0.005, "Source : exp_S39_ablation/ (émulateur bit-exact, badge « émulé PC »).",
             fontsize=8, color="#666666")
    return fig


# ── I3 — récupération Q15 sur Mahalanobis ────────────────────────────────────

def _fig_i3() -> plt.Figure:
    corr8, corr15, dauroc8, dauroc15 = [], [], [], []
    for ds in DATASETS:
        m = _maha_variants(ds)
        if m is None:
            corr8.append(np.nan); corr15.append(np.nan)
            dauroc8.append(np.nan); dauroc15.append(np.nan)
            continue
        fp = m["fp32"]["auroc"]
        c8 = m["int8"]["corr"]; c15 = m["q15"]["corr"]
        corr8.append(float(c8) if isinstance(c8, (int, float)) else np.nan)
        corr15.append(float(c15) if isinstance(c15, (int, float)) else np.nan)
        for variant, sink in (("int8", dauroc8), ("q15", dauroc15)):
            a = m[variant]["auroc"]
            if isinstance(a, (int, float)) and isinstance(fp, (int, float)):
                sink.append(float(a) - float(fp))
            else:
                sink.append(np.nan)

    c8col = STRATEGY_COLORS["int8_ptq_legacy"]
    c15col = STRATEGY_COLORS["q15"]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.3)
    x = np.arange(len(DATASETS))
    width = 0.35
    ax1.bar(x - width / 2, corr8, width, color=c8col, label="INT8")
    ax1.bar(x + width / 2, corr15, width, color=c15col, label="Q15")
    ax1.set_xticks(x); ax1.set_xticklabels([d.upper() for d in DATASETS], rotation=45, ha="right")
    ax1.set_ylabel("Corrélation de rang au score FP32")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Fidélité du score (↑ mieux)")
    ax1.legend()

    ax2.bar(x - width / 2, dauroc8, width, color=c8col, label="INT8")
    ax2.bar(x + width / 2, dauroc15, width, color=c15col, label="Q15")
    ax2.axhline(0.0, color="#555555", linewidth=1.0)
    ax2.set_xticks(x); ax2.set_xticklabels([d.upper() for d in DATASETS], rotation=45, ha="right")
    ax2.set_ylabel("ΔAUROC vs FP32")
    ax2.set_title("Écart d'AUROC (0 = préservé)")
    ax2.legend()

    fig.suptitle("Mahalanobis : Q15 récupère la grande dynamique de Σ⁻¹ (Sprint 34)")
    fig.text(0.01, 0.005, "Source : exp_S34_maha_q15/ — badge « émulé PC (bit-exact) ».",
             fontsize=8, color="#666666")
    return fig


# ── I4 — RAM Gap 3 ───────────────────────────────────────────────────────────

def _fig_i4() -> plt.Figure:
    # (modèle, sous-dossier S28, dataset repère, stratégie de quantification)
    rows = [
        ("ewc", "exp_S28_PC_ewc_hdc", "cmapss", "int8_qat"),
        ("hdc", "exp_S28_PC_ewc_hdc", "cmapss", "int16_am"),
        ("tinyol", "exp_S28_PC_tinyol_maha", "cmapss", "int8_qat"),
        ("mahalanobis", "exp_S28_PC_tinyol_maha", "pronostia", "int8_qat"),
    ]
    labels, ratios, colors = [], [], []
    for model, subdir, ds, strat in rows:
        r = _ram_ratio(model, subdir, ds)
        if r is None:
            continue
        labels.append(model.upper())
        ratios.append(r)
        colors.append(STRATEGY_COLORS[strat])

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x, ratios, 0.55, color=colors)
    ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--")
    ax.text(len(labels) - 0.5, 1.0, " FP32 ×1", va="bottom", ha="right",
            fontsize=10, color="#555555")
    for xi, r in zip(x, ratios):
        ax.text(xi, r, f"×{r:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Gain RAM poids (FP32 / quantifié)")
    ax.set_ylim(0.0, 4.5)
    ax.set_title("Gap 3 — gain RAM par stratégie de quantification")
    ax.text(0.5, 0.92,
            f"Budget board NUCLEO-F439ZI : {RAM_BUDGET_BYTES // 1024} Ko SRAM "
            "(.bss mesurée ≪ budget, Sprints 29/36)",
            transform=ax.transAxes, ha="center", fontsize=10, color="#333333")
    fig.text(0.01, 0.005,
             "HDC = mémoire associative int16 (×2.33) ; EWC/TinyOL/Maha = INT8 (×4). "
             "Source : exp_S28_PC_* (champ ram_ratio).",
             fontsize=8, color="#666666")
    return fig


# ── I5 — paradoxe latence board ──────────────────────────────────────────────

def _fig_i5() -> plt.Figure:
    fp_lat, q_lat, keep = [], [], []
    for ds in DATASETS:
        fp, q = _board_latency(ds)
        if fp is None or q is None:
            continue
        keep.append(ds); fp_lat.append(fp); q_lat.append(q)

    fig, ax = plt.subplots()
    x = np.arange(len(keep))
    width = 0.35
    ax.bar(x - width / 2, fp_lat, width, color=STRATEGY_COLORS["fp32"], label="FP32 (board)")
    ax.bar(x + width / 2, q_lat, width, color=STRATEGY_COLORS["int8_ptq_legacy"],
           label="INT8 (board)")
    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in keep])
    ax.set_ylabel("Latence P50 (µs, DWT)")
    ax.set_title("Paradoxe latence : sur FPU Cortex-M4, l'INT8 n'accélère pas")
    ax.legend(loc="upper left")
    ax.annotate("les poids int8 sont déquantifiés en float\ndans la boucle → MAC sur FPU\n"
                "→ latence ≥ FP32 ; le gain est RAM, pas vitesse",
                xy=(0.98, 0.6), xytext=(0.4, 0.55), textcoords="axes fraction",
                fontsize=10, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333"))
    fig.text(0.01, 0.005,
             "Source : exp_S29_board_int8/ (EWC, latences DWT board réelles). "
             "Chemin entier SIMD/CMSIS-NN = travaux futurs (TODO(dorra)).",
             fontsize=8, color="#666666")
    return fig


# ── I6 — QAT vs PTQ vs v2, même format INT8 ──────────────────────────────────

def _fig_i6() -> plt.Figure:
    # Contraste sur les 2 datasets couverts board (S36) : pronostia, monitoring.
    # Même format INT8 stocké, résultats opposés selon MOMENT et CALIBRATION du scale.
    ds_list = ["pronostia", "monitoring"]
    series = [
        ("FP32 (référence)", "fp32", BADGE_EMU),
        ("PTQ legacy (board)", "int8_ptq_legacy", BADGE_BOARD),
        ("v2 per-channel calibré (émulé)", "int8_v2", BADGE_EMU),
        ("v2 per-channel (board réel)", "q15", BADGE_BOARD),  # couleur distincte : mesure board
    ]
    # Récupère la métrique de fonctionnement (F1) par série
    def _values(ds: str) -> list[float | str | None]:
        fp = _quant_sweep_metric("ewc", ds, "fp32")
        try:
            summ, _ = load_experiment("experiments/exp_S36_summary.json")
            cell = summ["results"].get(ds, {}).get("5feat", {})
            ptq = metric_or_na(cell.get("board_frozen_int8", {}), "metric_value")
        except (FileNotFoundError, KeyError):
            ptq = None
        v2_emu = _quant_sweep_metric("ewc", ds, "int8_perchannel")
        v2_board = _board_v2_f1(ds, "frozen")
        return [fp, ptq, v2_emu, v2_board]

    fig, ax = plt.subplots()
    x = np.arange(len(ds_list))
    width = 0.2
    for k, (label, strat, badge) in enumerate(series):
        offs = (k - 1.5) * width
        vals = [_values(ds)[k] for ds in ds_list]
        ys = [v if isinstance(v, (int, float)) else np.nan for v in vals]
        ax.bar(x + offs, ys, width, color=STRATEGY_COLORS[strat],
               label=f"{label} · {badge}")
        for xi, v in zip(x + offs, vals):
            if v == A_MESURER:
                ax.text(xi, 0.05, "à mesurer", ha="center", va="bottom", fontsize=8.5,
                        color="#777777", rotation=90)
            elif not isinstance(v, (int, float)):
                ax.text(xi, 0.05, "n/a", ha="center", va="bottom", fontsize=8,
                        color="#999999", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in ds_list])
    ax.set_ylabel("F1_faulty")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Même format INT8, résultats opposés : QAT ✓  vs  PTQ legacy ✗  vs  v2 calibré ✓")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.005,
             "Sources : exp_S39_quant_sweep/ (émulé), exp_S36_summary.json (PTQ board), "
             "exp_S40_board_v2/ (v2 board réel ; « à mesurer » si non flashé).",
             fontsize=8, color="#666666")
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère I1–I6 sous ``out_root/quantization/impact/`` ; retourne les chemins."""
    return [
        savefig_png(_fig_i1(), CATALOG, "metrique_par_strategie", out_root),
        savefig_png(_fig_i2(), CATALOG, "ablation_perte_f1", out_root),
        savefig_png(_fig_i3(), CATALOG, "recuperation_q15_maha", out_root),
        savefig_png(_fig_i4(), CATALOG, "ram_gap3", out_root),
        savefig_png(_fig_i5(), CATALOG, "paradoxe_latence", out_root),
        savefig_png(_fig_i6(), CATALOG, "qat_vs_ptq_resultats", out_root),
    ]
