"""Catalogue `quantization/moment` — les trois moments de quantification (S4606).

4 figures qui rendent lisible le message Sprint 46 « quantifier ≠ quantifier : le
*moment* (QAT avant / PTQ après / les-deux) et la *calibration* dominent » :

    M1  barres groupées fp32 / before / after / both par (modèle × dataset) — EWC + TinyOL.
    M2  heatmap métrique = f(moment, modèle×dataset) ; HDC/Maha en gris N/A (hors-axe).
    M3  effet calibration sur after/both (legacy → per-tensor → per-canal) — lien S39.
    M4  contexte HDC (INT8 ≡ FP32 structurel) + Maha (INT8 vs Q15) — hors grille 3-way.

Toute valeur tracée provient d'un ``load_experiment`` (JSON d'``experiments/exp_S46_*``) —
**aucun littéral numérique de résultat** dans ce module (garde AST ``test_no_hardcoded_results``,
S4207/S4606) ; seuls des flottants de mise en page apparaissent.

Distinction honnête portée par les figures : ``before`` = fake-quant à l'inférence (**borne
haute** que la carte n'exécute pas ; hachures) ; ``both`` = QAT → export PTQ (**fidèle au
déploiement**, noyau entier ; plein). HDC/Maha ne sont **pas** sur l'axe des moments (gris N/A).

Sources (lecture seule) : ``exp_S46_ewc/``, ``exp_S46_tinyol/``, ``exp_S46_context/``.
Board différée (S4608) → non tracée ici (aucun chiffre inventé).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import load_experiment, metric_or_na
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, savefig_png

CATALOG = "quantization/moment"
# Répertoire de sortie (aligné sur la spec S4606 : docs/figures/quantization_moment/).
OUT_SUBDIR = "quantization_moment"

MOMENT_ORDER: list[str] = ["fp32", "before", "after", "both"]
MODELS_3WAY: list[str] = ["ewc", "tinyol"]
DATASETS: list[str] = ["monitoring", "pronostia"]

# Libellés FR + habillage visuel par moment (borne haute vs déploiement).
MOMENT_LABEL_FR: dict[str, str] = {
    "fp32": "FP32 (réf.)",
    "before": "before — QAT (borne haute)",
    "after": "after — PTQ",
    "both": "both — déploiement",
}
MOMENT_COLOR: dict[str, str] = {
    "fp32": STRATEGY_COLORS["fp32"],
    "before": STRATEGY_COLORS["int8_qat"],
    "after": STRATEGY_COLORS["int8_v2"],
    "both": STRATEGY_COLORS["int8_v2"],
}
MOMENT_HATCH: dict[str, str] = {"fp32": "", "before": "//", "after": "..", "both": ""}

NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"

# Balayage de calibration EWC (S4603) : fichier JSON par schéma after_scheme.
CALIB_FILES: list[tuple[str, str]] = [
    ("legacy_c", "legacy C\n(1/128 figé)"),
    ("all", "per-tensor\ncalibré"),          # {ds}_all.json = per_tensor_calib canonique
    ("per_channel_int8", "per-canal\nINT8"),
]


# ── Loaders (tout passe par load_experiment) ─────────────────────────────────

def _moment_metrics(model: str, ds: str) -> dict[str, float | None]:
    """{moment: métrique} d'un couple (modèle 3-way, dataset), None si absent/non mesuré."""
    try:
        data, _ = load_experiment(f"experiments/exp_S46_{model}/{ds}_all.json")
    except FileNotFoundError:
        return {m: None for m in MOMENT_ORDER}
    out: dict[str, float | None] = {}
    for m in MOMENT_ORDER:
        v = metric_or_na(data, f"moments.{m}.metric")
        out[m] = float(v) if isinstance(v, (int, float)) else None
    return out


def _calib_metrics(ds: str, scheme_file: str) -> tuple[float | None, float | None]:
    """(after, both) AUROC EWC pour un schéma de calibration donné."""
    try:
        data, _ = load_experiment(f"experiments/exp_S46_ewc/{ds}_{scheme_file}.json")
    except FileNotFoundError:
        return None, None
    a = metric_or_na(data, "moments.after.metric")
    b = metric_or_na(data, "moments.both.metric")
    return (
        float(a) if isinstance(a, (int, float)) else None,
        float(b) if isinstance(b, (int, float)) else None,
    )


def _ewc_fp32(ds: str) -> float | None:
    return _moment_metrics("ewc", ds)["fp32"]


def _hdc_context(ds: str) -> dict | None:
    try:
        data, _ = load_experiment(f"experiments/exp_S46_context/hdc_{ds}.json")
    except FileNotFoundError:
        return None
    return {
        "fp32": metric_or_na(data, "values.fp32"),
        "int8_native": metric_or_na(data, "values.int8_native"),
        "ram_ratio": metric_or_na(data, "ram_ratio"),
    }


def _maha_context(ds: str) -> dict | None:
    try:
        data, _ = load_experiment(f"experiments/exp_S46_context/maha_{ds}.json")
    except FileNotFoundError:
        return None
    return {k: metric_or_na(data, f"values.{k}") for k in ("fp32", "int8", "q15")}


def _num(v) -> float:
    """Valeur numérique ou NaN (pour tracé ; jamais 0 substitué)."""
    return float(v) if isinstance(v, (int, float)) else float("nan")


# ── M1 — barres groupées des 4 moments par (modèle × dataset) ────────────────

def _fig_m1() -> plt.Figure:
    cols = [(m, ds) for m in MODELS_3WAY for ds in DATASETS]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(cols))
    width = 0.2
    for k, moment in enumerate(MOMENT_ORDER):
        offs = (k - 1.5) * width
        ys, na_x = [], []
        for xi, (model, ds) in zip(x + offs, cols):
            v = _moment_metrics(model, ds)[moment]
            ys.append(_num(v))
            if v is None:
                na_x.append(xi)
        ax.bar(x + offs, ys, width, color=MOMENT_COLOR[moment],
               hatch=MOMENT_HATCH[moment], edgecolor="white",
               label=MOMENT_LABEL_FR[moment])
        for xi in na_x:
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=8,
                    color="#999999", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.upper()}\n{d.capitalize()}" for m, d in cols])
    ax.set_ylabel("AUROC (détection de panne)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Les quatre moments de quantification (EWC + TinyOL · PC/émulé)")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.text(0.01, 0.005,
             "before = fake-quant à l'inférence (borne haute, hachures) ; "
             "both = QAT→export PTQ (déploiement, plein). "
             "Source : exp_S46_{ewc,tinyol}/ · badge « émulé PC (bit-exact) ».",
             fontsize=8, color=MUTED)
    return fig


# ── M2 — heatmap métrique = f(moment, modèle×dataset) ────────────────────────

def _fig_m2() -> plt.Figure:
    # Colonnes : les 2 modèles 3-way (mesurés) + HDC/Maha (gris N/A, hors-axe).
    three_way = [(m, ds) for m in MODELS_3WAY for ds in DATASETS]
    na_cols = [(m, ds) for m in ("hdc", "mahalanobis") for ds in DATASETS]
    cols = three_way + na_cols

    mat = np.full((len(MOMENT_ORDER), len(cols)), np.nan)
    for j, (model, ds) in enumerate(three_way):
        metrics = _moment_metrics(model, ds)
        for i, moment in enumerate(MOMENT_ORDER):
            mat[i, j] = _num(metrics[moment])

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    cmap = plt.cm.viridis.copy()
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)

    # Grise + annote toutes les cellules N/A (HDC/Maha hors-axe, et trous éventuels).
    for i in range(len(MOMENT_ORDER)):
        for j, (model, _ds) in enumerate(cols):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color=NA_GRAY, zorder=2))
                if model in ("hdc", "mahalanobis"):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=8, color=MUTED, zorder=3)
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.9 else INK, zorder=3)

    def _col_label(model: str, ds: str) -> str:
        short = "MAHA" if model == "mahalanobis" else model.upper()
        return f"{short}\n{ds.capitalize()}"

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([_col_label(m, d) for m, d in cols], fontsize=8)
    ax.set_yticks(np.arange(len(MOMENT_ORDER)))
    ax.set_yticklabels([MOMENT_LABEL_FR[m] for m in MOMENT_ORDER])
    ax.set_title("Métrique par moment × (modèle × dataset)")
    fig.colorbar(im, ax=ax, label="AUROC", fraction=0.03)
    fig.text(0.01, 0.005,
             "HDC (natif entier) et Mahalanobis (PTQ-only) ne sont pas sur l'axe des "
             "moments → gris N/A (cf. M4). Source : exp_S46_*.",
             fontsize=8, color=MUTED)
    return fig


# ── M3 — effet calibration sur after/both (lien ablation S39) ────────────────

def _fig_m3() -> plt.Figure:
    ds_colors = {"pronostia": STRATEGY_COLORS["int8_v2"], "monitoring": STRATEGY_COLORS["q15"]}
    xr = np.arange(len(CALIB_FILES))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for ds, color in ds_colors.items():
        after = [_calib_metrics(ds, f)[0] for f, _ in CALIB_FILES]
        both = [_calib_metrics(ds, f)[1] for f, _ in CALIB_FILES]
        ax.plot(xr, [_num(v) for v in after], "-o", color=color, ms=9,
                label=f"{ds.capitalize()} · after (PTQ FP32)")
        ax.plot(xr, [_num(v) for v in both], "--s", color=color, ms=8, alpha=0.6,
                label=f"{ds.capitalize()} · both (QAT→PTQ)")
        fp = _ewc_fp32(ds)
        if isinstance(fp, (int, float)):
            ax.axhline(float(fp), color=color, linestyle=":", linewidth=1.2, alpha=0.5)
            ax.text(xr[-1], float(fp), f"  FP32 {ds.capitalize()}", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(xr)
    ax.set_xticklabels([lbl for _, lbl in CALIB_FILES])
    ax.set_ylabel("AUROC (EWC, détection de panne)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Effet de la calibration : PTQ naïf s'effondre, calibré récupère")
    ax.annotate("le scale calibré (vs 1/128 figé)\nrécupère tout l'AUROC",
                xy=(1.0, 0.9), xytext=(0.2, 0.4), textcoords="axes fraction",
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.legend(loc="center right", fontsize=9)
    fig.text(0.01, 0.005,
             "Source : exp_S46_ewc/{ds}_{legacy_c,all,per_channel_int8}.json — "
             "balayage after_scheme (lien direct ablation S39). Badge « émulé PC ».",
             fontsize=8, color=MUTED)
    return fig


# ── M4 — contexte HDC (structurel) + Maha (INT8 vs Q15) ──────────────────────

def _fig_m4() -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.subplots_adjust(wspace=0.3, top=0.82)

    # Gauche : HDC — INT8 natif ≡ FP32 hypothétique (barres égales + ratio RAM annoté).
    x = np.arange(len(DATASETS))
    width = 0.35
    fp_vals, int8_vals = [], []
    for ds in DATASETS:
        ctx = _hdc_context(ds) or {}
        fp_vals.append(_num(ctx.get("fp32")))
        int8_vals.append(_num(ctx.get("int8_native")))
    ax1.bar(x - width / 2, fp_vals, width, color=STRATEGY_COLORS["fp32"],
            label="FP32 (hypothétique)")
    ax1.bar(x + width / 2, int8_vals, width, color=STRATEGY_COLORS["int16_am"],
            label="INT8 natif (int8 HV + int16 AM)")
    for xi, ds in zip(x, DATASETS):
        ctx = _hdc_context(ds) or {}
        r = ctx.get("ram_ratio")
        if isinstance(r, (int, float)):
            ax1.text(xi, 0.02, f"RAM ×{float(r):.2f}", ha="center", va="bottom",
                     fontsize=9, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in DATASETS])
    ax1.set_ylabel("F1 macro (HDC)")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("HDC — INT8 ≡ FP32 (structurel)")
    ax1.legend(loc="lower right", fontsize=9)

    # Droite : Maha — fp32 / int8 / q15 (int8 casse, q15 récupère).
    variants = [("fp32", STRATEGY_COLORS["fp32"]),
                ("int8", STRATEGY_COLORS["int8_ptq_legacy"]),
                ("q15", STRATEGY_COLORS["q15"])]
    xw = 0.25
    for k, (var, color) in enumerate(variants):
        offs = (k - 1.0) * xw
        ys = [_num((_maha_context(ds) or {}).get(var)) for ds in DATASETS]
        ax2.bar(x + offs, ys, xw, color=color, label=var.upper())
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in DATASETS])
    ax2.set_ylabel("AUROC (Mahalanobis)")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Maha — INT8 casse, Q15 récupère")
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle("Contexte hors-axe : HDC structurel · Maha INT8-vs-Q15 (pas de 3-way)")
    fig.text(0.01, 0.005,
             "Source : exp_S46_context/{hdc,maha}_{ds}.json — "
             "ces deux modèles ne sont pas sur l'axe avant/après (cf. M2 gris N/A).",
             fontsize=8, color=MUTED)
    return fig


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère M1–M4 sous ``out_root/quantization_moment/`` ; retourne les chemins."""
    return [
        savefig_png(_fig_m1(), OUT_SUBDIR, "M1_moments_bars", out_root),
        savefig_png(_fig_m2(), OUT_SUBDIR, "M2_moment_heatmap", out_root),
        savefig_png(_fig_m3(), OUT_SUBDIR, "M3_calibration_effect", out_root),
        savefig_png(_fig_m4(), OUT_SUBDIR, "M4_hdc_maha_context", out_root),
    ]
