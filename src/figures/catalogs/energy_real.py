"""Catalogue `energy_real` — énergie réelle + breakdown latence INT8 (Sprint 50, S5006).

Restitue les livrables du Sprint 50 avec **honnêteté matérielle**. Le banc
**X-NUCLEO-LPM01A a tourné** (S5008) : il faut donc distinguer trois statuts, et non
plus deux —

  * **mesuré** : courant moyen par modèle × encodage (8/8 cellules), référence au repos,
    autonomie qui en découle, et breakdown **latence INT8** (DWT, S5004) ;
  * **non séparable, mesuré comme tel** : les µJ/inférence. La référence au repos du banc
    est plus haute que tous les régimes de flux (le firmware attend la trame UART par
    scrutation active), donc l'énergie marginale par inférence ressort négative. Ces
    figures restent **grises**, mais avec une raison mesurée — pas « pas encore fait » ;
  * **non câblé** : la décomposition MCU / périphériques (rail VDD_MCU global) et le
    capteur (simulé par UART, S5001).

6 figures :

    e1_energie_par_inference.png   µJ/inférence par modèle × encodage (gris = non séparable).
    e2_par_composant.png           décomposition MCU / périph / capteur (capteur = N/A UART).
    e3_autonomie.png               autonomie (h) par configuration × capacité batterie.
    e4_latence_int8_breakdown.png  déquant/MAC/requant vs total FP32 — **mesuré board**.
    e5_cout_benefice.png           RAM ÷ (gain) vs latence INT8/FP32 vs énergie.
    e6_courant_moyen_mesure.png    courant moyen par cellule + repos — **mesuré board**.

**Toute valeur provient d'un ``load_experiment``** (``exp_S50_energy/*``, ``exp_S50_int8_latency/*``,
``exp_S49_ram/summary.json``) — aucun littéral de résultat (garde AST ``test_no_hardcoded_results``).
Cellule ``« à mesurer »`` / N/A / absente → **gris**, jamais 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import A_MESURER, load_experiment
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "energy_real"

MODELS: list[str] = ["ewc", "hdc", "tinyol", "maha"]
MODEL_LABEL: dict[str, str] = {"ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "maha": "Maha"}
ENCODINGS: list[str] = ["fp32", "int8"]
SEGMENTS: list[str] = ["dequant", "mac", "requant"]
SEGMENT_LABEL: dict[str, str] = {
    "dequant": "déquant (int→FP32)", "mac": "MAC (entier)", "requant": "requant (FP32→int)",
}

COLOR_FP32 = STRATEGY_COLORS["fp32"]
COLOR_INT8 = STRATEGY_COLORS["int8_v2"]
COLOR_SEG = {
    "dequant": STRATEGY_COLORS["fp32"],
    "mac": STRATEGY_COLORS["int16_am"],
    "requant": STRATEGY_COLORS["int8_v2"],
}
#: Conversions et marges d'affichage (mise en page, pas des résultats).
MA_PER_A = 1000
YLIM_MARGIN_MA = 2

NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"
BADGE_ENERGY = (
    "courant : mesuré board réelle NUCLEO-F439ZI (LPM01A, S5008) — "
    "µJ/inférence non séparables sur ce firmware, cf. « energy_na_reason »"
)
#: Raccourci de la raison mesurée du N/A, pour les figures qui restent grises.
NA_ENERGY_TEXT = (
    "non séparable\n(référence de scrutation active —\nvoir energy_na_reason)"
)
BADGE_LAT = "latence : mesurée board réelle NUCLEO-F439ZI (DWT, S5004)"


# ── Sources ──────────────────────────────────────────────────────────────────

def _load(path: str) -> dict | None:
    try:
        data, _ = load_experiment(path)
    except FileNotFoundError:
        return None
    return data


def _energy_summary() -> dict | None:
    return _load("experiments/exp_S50_energy/summary.json")


def _autonomy() -> dict | None:
    return _load("experiments/exp_S50_energy/autonomy.json")


def _latency() -> dict | None:
    return _load("experiments/exp_S50_int8_latency/ewc.json")


def _ram_summary() -> dict | None:
    return _load("experiments/exp_S49_ram/summary.json")


def _num(value) -> float | None:
    """« à mesurer » / None / non numérique → None (jamais 0). Sinon float."""
    if value is None or value == A_MESURER:
        return None
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _empty(msg: str, name: str, out_root: Path) -> Path:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, wrap=True)
    return savefig_png(fig, CATALOG, name, out_root)


def _grid(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=NA_GRAY, linewidth=0.5, alpha=0.5)


# ── E1 : µJ/inférence par modèle × encodage ──────────────────────────────────

def _fig_energy_per_inference(summary: dict | None, out_root: Path) -> Path:
    name = "e1_energie_par_inference"
    if summary is None:
        return _empty("exp_S50_energy/summary.json absent", name, out_root)
    per = summary.get("per_model", {})
    fig, ax = plt.subplots()
    x = np.arange(len(MODELS))
    width = 0.35
    any_measured = False
    for i, enc in enumerate(ENCODINGS):
        offset = (i - 0.5) * width
        vals, colors = [], []
        for m in MODELS:
            v = _num(per.get(m, {}).get(enc, {}).get("energy_uj_per_inference"))
            if v is None:
                vals.append(0.0); colors.append(NA_GRAY)
            else:
                vals.append(v); colors.append(COLOR_FP32 if enc == "fp32" else COLOR_INT8)
                any_measured = True
        bars = ax.bar(x + offset, vals, width, color=colors, label=enc,
                      edgecolor=INK, linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    (f"{v:g}" if v else "à mesurer"), ha="center", va="bottom",
                    fontsize=8, color=MUTED, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("µJ / inférence")
    ax.set_title("Énergie par inférence — modèle × encodage"
                 + ("" if any_measured else " (non séparable)"))
    ax.legend()
    _grid(ax)
    if not any_measured:
        # Le banc A tourné : ce N/A est un résultat mesuré, pas une case vide.
        ax.text(0.5, 0.55, NA_ENERGY_TEXT, transform=ax.transAxes,
                ha="center", va="center", color=NA_GRAY, fontsize=11)
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── E2 : décomposition par composant (MCU / périph / capteur) ────────────────

def _fig_by_component(out_root: Path) -> Path:
    name = "e2_par_composant"
    comps = ["mcu", "periph", "sensor"]
    comp_label = {"mcu": "MCU", "periph": "périph.", "sensor": "capteur (UART = N/A)"}
    cols = [f"{m}_{e}" for m in MODELS for e in ENCODINGS]
    fig, ax = plt.subplots()
    x = np.arange(len(cols))
    bottom = np.zeros(len(cols))
    plotted = False
    for comp in comps:
        vals, colors = [], []
        for m in MODELS:
            for e in ENCODINGS:
                cell = _load(f"experiments/exp_S50_energy/{m}_{e}.json")
                v = _num(cell.get("by_component", {}).get(comp)) if cell else None
                if v is None:
                    vals.append(0.0); colors.append(NA_GRAY)
                else:
                    vals.append(v); plotted = True
                    colors.append(COLOR_SEG["dequant"] if comp == "mcu"
                                  else COLOR_SEG["mac"] if comp == "periph" else NA_GRAY)
        ax.bar(x, vals, 0.6, bottom=bottom, color=colors,
               label=comp_label[comp], edgecolor=INK, linewidth=0.5)
        bottom = bottom + np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_ylabel("µJ / inférence")
    ax.set_title("Décomposition énergie par composant"
                 + ("" if plotted else " (à mesurer — capteur = N/A, simulé UART)"))
    ax.legend()
    if not plotted:
        # Le capteur est simulé par UART et le rail mesuré est global : la
        # décomposition par composant exige un câblage séparé, pas une campagne.
        ax.text(0.5, 0.5, "à mesurer\n(rail VDD_MCU global — décomposition\n"
                          "MCU/périph. non câblée ; capteur simulé UART)",
                transform=ax.transAxes, ha="center", va="center",
                color=NA_GRAY, fontsize=11)
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── E3 : autonomie (h) par configuration × capacité ──────────────────────────

def _fig_autonomy(auto: dict | None, out_root: Path) -> Path:
    name = "e3_autonomie"
    if auto is None:
        return _empty("exp_S50_energy/autonomy.json absent", name, out_root)
    per = auto.get("per_model", {})
    caps = auto.get("capacites_mah", [])
    cols = [f"{m}_{e}" for m in MODELS for e in ENCODINGS]
    fig, ax = plt.subplots()
    x = np.arange(len(cols))
    width = 0.8 / max(len(caps), 1)
    plotted = False
    for ci, cap in enumerate(caps):
        vals, colors = [], []
        for col in cols:
            by = per.get(col, {}).get("autonomy_h_by_mah", {})
            v = _num(by.get(str(cap)))
            if v is None:
                vals.append(0.0); colors.append(NA_GRAY)
            else:
                vals.append(v); colors.append(COLOR_INT8); plotted = True
        ax.bar(x + (ci - len(caps) / 2) * width, vals, width, color=colors,
               label=f"{cap:g} mAh", edgecolor=INK, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_ylabel("autonomie (h)")
    ax.set_title("Autonomie par configuration × capacité batterie"
                 + ("" if plotted else " (à mesurer)"))
    ax.legend(fontsize=8, ncol=2)
    if not plotted:
        ax.text(0.5, 0.5, "à mesurer\n(dépend des µJ réels — S5002)", transform=ax.transAxes,
                ha="center", va="center", color=NA_GRAY, fontsize=13)
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── E6 : courant moyen mesuré par modèle × encodage — MESURÉ board ───────────

def _fig_mean_current(out_root: Path) -> Path:
    """Courant moyen par cellule, à cadence imposée, avec la référence au repos.

    C'est la grandeur que la campagne S5008 produit réellement. Les barres
    d'erreur sont l'écart-type des répétitions : elles disent si un écart entre
    deux cellules dépasse la dispersion du banc.
    """
    name = "e6_courant_moyen_mesure"
    cells = {(m, e): _load(f"experiments/exp_S50_energy/{m}_{e}.json")
             for m in MODELS for e in ENCODINGS}
    if not any(cells.values()):
        return _empty("exp_S50_energy/*.json absents", name, out_root)

    fig, ax = plt.subplots()
    x = np.arange(len(MODELS))
    width = 0.35
    idle_vals: list[float] = []
    measured_ma: list[float] = []
    for i, enc in enumerate(ENCODINGS):
        vals, errs, colors = [], [], []
        for m in MODELS:
            mes = (cells.get((m, enc)) or {}).get("current_measurement") or {}
            v = _num(mes.get("i_mean_a"))
            s = _num(mes.get("i_std_a"))
            idle = _num(mes.get("i_idle_a"))
            if idle is not None:
                idle_vals.append(idle * MA_PER_A)
            if v is None:
                vals.append(0.0); errs.append(0.0); colors.append(NA_GRAY)
            else:
                vals.append(v * MA_PER_A)
                errs.append((s or 0.0) * MA_PER_A)
                measured_ma.append(v * MA_PER_A)
                colors.append(COLOR_FP32 if enc == "fp32" else COLOR_INT8)
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
               color=colors, label=enc, edgecolor=INK, linewidth=0.5)

    if idle_vals:
        idle_ma = float(np.mean(idle_vals))
        ax.axhline(idle_ma, color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(len(MODELS) - 0.5, idle_ma, " repos (scrutation UART)",
                va="bottom", ha="right", fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("courant moyen (mA)")
    ax.set_title("Courant moyen sous charge d'inférence — cadence imposée")
    # Échelle resserrée : les écarts utiles sont de l'ordre du mA sur ~50 mA ;
    # partir de zéro les écraserait jusqu'à les rendre invisibles.
    if measured_ma:
        ax.set_ylim(min(measured_ma) - YLIM_MARGIN_MA,
                    max(measured_ma + idle_vals) + YLIM_MARGIN_MA)
    ax.legend()
    _grid(ax)
    fig.text(0.5, 0.005, BADGE_ENERGY, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── E4 : breakdown latence INT8 (déquant/MAC/requant) — MESURÉ board ─────────

def _fig_int8_latency_breakdown(lat: dict | None, out_root: Path) -> Path:
    name = "e4_latence_int8_breakdown"
    if lat is None:
        return _empty("exp_S50_int8_latency/ewc.json absent — lancer run_s50_int8_latency.py",
                      name, out_root)
    seg = lat.get("segments", {})
    datasets = lat.get("datasets_measured", [])
    if not datasets:
        return _empty("aucun dataset mesuré dans ewc.json", name, out_root)
    fig, ax = plt.subplots()
    x = np.arange(len(datasets))
    width = 0.5
    bottom = np.zeros(len(datasets))
    for s in SEGMENTS:
        by_ds = seg.get(s, {}).get("by_dataset_cycles_p50", {})
        vals = [(_num(by_ds.get(ds)) or 0.0) for ds in datasets]
        ax.bar(x, vals, width, bottom=bottom, color=COLOR_SEG[s],
               label=SEGMENT_LABEL[s], edgecolor=INK, linewidth=0.5)
        bottom = bottom + np.array(vals)
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot, f"{tot:g} cyc\n(INT8 forward)", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("cycles DWT (p50, 180 MHz)")
    ax.set_title("Breakdown latence INT8 : déquant + MAC + requant (cycles bruts)")
    ax.legend()
    _grid(ax)
    fig.text(0.5, 0.005, BADGE_LAT, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


# ── E5 : coût/bénéfice — RAM (gain) vs latence INT8/FP32 vs énergie ──────────

def _weights_ram_ratio(dataset: str) -> float | None:
    """Gain RAM sur les POIDS (÷4 int8), depuis exp_S40_board_v2 — pas la RAM système totale
    (dominée par des buffers fixes → ratio total ≈ 1.0, cf. S49). Le gain porte sur le modèle."""
    d = _load(f"experiments/exp_S40_board_v2/results_per_channel_{dataset}_frozen.json")
    return _num(d.get("ram_ratio_fp32_over_quant")) if d else None


def _fig_cost_benefit(ram: dict | None, lat: dict | None, summary: dict | None,
                      out_root: Path) -> Path:
    name = "e5_cout_benefice"
    fig, ax = plt.subplots()
    dims = ["RAM poids\n(fp32/int8, gain)", "Latence\n(int8/fp32)", "Énergie\n(int8/fp32)"]
    x = np.arange(len(dims))

    # RAM : ratio fp32/int8 des POIDS (>1 = gain) — moyenne board sur les datasets EWC mesurés.
    ram_ratios = [r for ds in ("monitoring", "pronostia")
                  if (r := _weights_ram_ratio(ds)) is not None]
    ram_val = float(np.mean(ram_ratios)) if ram_ratios else None

    # Latence : ratio int8/fp32 (>1 = surcoût FPU) — depuis ewc.json.
    lat_val = None
    if lat is not None:
        t_int8 = [v for v in (lat.get("total_int8_us_p50_by_dataset") or {}).values()
                  if isinstance(v, (int, float))]
        t_fp32 = [v for v in (lat.get("total_fp32_us_p50_by_dataset") or {}).values()
                  if isinstance(v, (int, float))]
        if t_int8 and t_fp32 and float(np.mean(t_fp32)) > 0:
            lat_val = float(np.mean(t_int8)) / float(np.mean(t_fp32))

    # Énergie : ratio int8/fp32 — « à mesurer » tant que LPM01A non posé.
    ene_val = None
    if summary is not None:
        ene_val = _num(summary.get("per_model", {}).get("ewc", {}).get("ratio_int8_fp32"))

    vals = [ram_val, lat_val, ene_val]
    colors = [(COLOR_INT8 if v is not None else NA_GRAY) for v in vals]
    heights = [(v if v is not None else 0.0) for v in vals]
    bars = ax.bar(x, heights, 0.5, color=colors, edgecolor=INK, linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                (f"×{v:.2g}" if v is not None else "à mesurer"),
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("ratio (référence 1.0)")
    ax.set_title("Coût/bénéfice INT8 : gain RAM vs surcoût latence FPU vs énergie")
    _grid(ax)
    fig.text(0.5, 0.005,
             "RAM poids ÷4 (S40) = gain · latence ≥ 1 = paradoxe FPU (S29/S5004) · énergie à mesurer (S50)",
             ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, name, out_root)


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 6 figures énergie réelle + latence INT8 sous ``out_root/energy_real/``."""
    apply_style("slide")
    summary = _energy_summary()
    auto = _autonomy()
    lat = _latency()
    ram = _ram_summary()
    return [
        _fig_energy_per_inference(summary, out_root),
        _fig_by_component(out_root),
        _fig_autonomy(auto, out_root),
        _fig_int8_latency_breakdown(lat, out_root),
        _fig_cost_benefit(ram, lat, summary, out_root),
        _fig_mean_current(out_root),
    ]
