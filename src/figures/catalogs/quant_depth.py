"""Catalogue `quant_depth` — profondeur × granularité × symétrie de la tête EWC (S4706).

Restitue le sweep Sprint 47 (second axe de la quantification, orthogonal au *moment*
S46) en 5 figures régénérables :

    auroc_vs_bits.png            delta_auroc vs weight_bits, 1 ligne/granularité, 1 panneau/dataset ; cliff annoté.
    heatmap_bits_granularity.png heatmap delta_auroc (bits × granularité) par dataset ; N/A gris.
    ram_vs_bits.png              ratio RAM théorique (bit-packée) vs weight_bits, axe log ; badge d'honnêteté.
    symmetry_gain.png            barres symmetric vs affine aux bits critiques (gain zero-point).
    scope_context.png            EWC balayé ∥ HDC/Maha/TinyOL N/A (cartouches justifiés).

**Toute valeur tracée provient d'un ``load_experiment``** (JSON d'``experiments/exp_S47_*``) —
aucun littéral numérique de résultat dans ce module (garde AST ``test_no_hardcoded_results``,
S4707) ; seuls des flottants de mise en page apparaissent. N/A en gris (jamais 0). La RAM
est étiquetée **théorique (bit-packée)** : le gain réel exige un kernel bit-packé (Sprint 48).

Sources (lecture seule) : ``exp_S47_depth/`` (S4703), ``exp_S47_symmetry/`` (S4704),
``exp_S47_context/context.json`` (S4705).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import load_experiment, metric_or_na
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, savefig_png

CATALOG = "quant_depth"
OUT_SUBDIR = "quantization_depth"

DATASETS: list[str] = ["monitoring", "pronostia"]
GRANULARITIES: list[str] = ["per_tensor", "per_channel"]

# Ordre canonique de profondeur : tags de fichier + libellés FR + bits effectifs (entiers,
# structurels — pas des résultats) pour l'axe RAM log. ternaire → 2 bits stockés, binaire → 1.
BIT_TAGS: list[str] = ["int8", "int6", "int4", "int3", "int2", "ternaire", "binaire"]
BIT_LABEL_FR: dict[str, str] = {
    "int8": "8 b", "int6": "6 b", "int4": "4 b", "int3": "3 b", "int2": "2 b",
    "ternaire": "ternaire", "binaire": "binaire",
}
EFFECTIVE_BITS: dict[str, int] = {
    "int8": 8, "int6": 6, "int4": 4, "int3": 3, "int2": 2, "ternaire": 2, "binaire": 1,
}
CRITICAL_TAGS: list[str] = ["int2", "int3", "int4"]  # bits critiques S4704

GRAN_COLOR: dict[str, str] = {
    "per_tensor": STRATEGY_COLORS["int8_ptq_legacy"],   # rouge — plus fragile aux bas bits
    "per_channel": STRATEGY_COLORS["int8_v2"],          # orange — repousse le cliff
}
GRAN_LABEL_FR: dict[str, str] = {"per_tensor": "per-tensor", "per_channel": "per-canal"}
SYM_COLOR: dict[str, str] = {
    "symmetric": STRATEGY_COLORS["int8_v2"],
    "affine": STRATEGY_COLORS["q15"],
}
SYM_LABEL_FR: dict[str, str] = {"symmetric": "symétrique", "affine": "affine (zero-point)"}

NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"


# ── Loaders (tout passe par load_experiment) ─────────────────────────────────

def _depth(ds: str, tag: str, gran: str) -> dict | None:
    """Cellule du sweep profondeur (S4703) ou None si absente."""
    try:
        data, _ = load_experiment(f"experiments/exp_S47_depth/exp_S47_ewc_{ds}_{tag}_{gran}.json")
    except FileNotFoundError:
        return None
    return data


def _sym(ds: str, tag: str, symmetry: str) -> dict | None:
    """Cellule de l'axe symétrie (S4704) ou None si absente."""
    try:
        data, _ = load_experiment(f"experiments/exp_S47_symmetry/exp_S47_ewc_{ds}_{tag}_{symmetry}.json")
    except FileNotFoundError:
        return None
    return data


def _val(data: dict | None, key: str) -> float:
    """Métrique numérique d'une cellule ou NaN (jamais 0 substitué)."""
    if data is None:
        return float("nan")
    v = metric_or_na(data, key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


# ── auroc_vs_bits — delta_auroc vs profondeur, par granularité et dataset ────

def _fig_auroc_vs_bits() -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11.0, 4.5), sharey=True)
    x = np.arange(len(BIT_TAGS))
    for ax, ds in zip(axes, DATASETS):
        for gran in GRANULARITIES:
            ys = [_val(_depth(ds, t, gran), "delta_auroc") for t in BIT_TAGS]
            ax.plot(x, ys, "-o", color=GRAN_COLOR[gran], ms=7, label=GRAN_LABEL_FR[gran])
        # Repère du seuil de dégradation (−0,02) — annotation, pas une donnée tracée.
        ax.axhline(-0.02, color=MUTED, linestyle=":", linewidth=1.0)
        ax.text(0.0, -0.02, " seuil cliff (−0,02)", va="bottom", ha="left",
                fontsize=8, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([BIT_LABEL_FR[t] for t in BIT_TAGS], rotation=30, fontsize=8)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("profondeur des poids")
    axes[0].set_ylabel("Δ AUROC (quant − FP32)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Δ AUROC vs profondeur — la per-canal repousse le « cliff » (EWC · émulé PC)")
    fig.text(0.01, 0.005,
             "Source : exp_S47_depth/. Le cliff = profondeur où Δ AUROC franchit −0,02. "
             "Badge « émulé PC (bit-exact) ».", fontsize=8, color=MUTED)
    return fig


# ── heatmap_bits_granularity — delta_auroc (bits × granularité) par dataset ──

def _fig_heatmap() -> plt.Figure:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11.0, 4.5))
    for ax, ds in zip(axes, DATASETS):
        mat = np.full((len(GRANULARITIES), len(BIT_TAGS)), np.nan)
        for i, gran in enumerate(GRANULARITIES):
            for j, tag in enumerate(BIT_TAGS):
                mat[i, j] = _val(_depth(ds, tag, gran), "delta_auroc")
        im = ax.imshow(mat, aspect="auto", cmap=plt.cm.RdYlGn, vmin=-0.05, vmax=0.0)
        for i in range(len(GRANULARITIES)):
            for j in range(len(BIT_TAGS)):
                v = mat[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color=NA_GRAY, zorder=2))
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8,
                            color=MUTED, zorder=3)
                else:
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                            color=INK, zorder=3)
        ax.set_xticks(np.arange(len(BIT_TAGS)))
        ax.set_xticklabels([BIT_LABEL_FR[t] for t in BIT_TAGS], rotation=30, fontsize=8)
        ax.set_yticks(np.arange(len(GRANULARITIES)))
        ax.set_yticklabels([GRAN_LABEL_FR[g] for g in GRANULARITIES])
        ax.set_title(ds.capitalize())
        fig.colorbar(im, ax=ax, label="Δ AUROC", fraction=0.05)
    fig.suptitle("Δ AUROC par (profondeur × granularité) — EWC (émulé PC)")
    fig.text(0.01, 0.005, "Source : exp_S47_depth/. Cellules manquantes en gris (N/A).",
             fontsize=8, color=MUTED)
    return fig


# ── ram_vs_bits — ratio RAM théorique (bit-packée) vs profondeur ─────────────

def _fig_ram_vs_bits() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # La RAM théorique ne dépend que de la profondeur/granularité (mêmes poids) : on trace
    # Pronostia (les deux datasets donnent le même ratio pour une profondeur donnée).
    ds = DATASETS[-1]
    xb = [EFFECTIVE_BITS[t] for t in BIT_TAGS]
    for gran in GRANULARITIES:
        ratios = [_val(_depth(ds, t, gran), "ram_ratio_vs_fp32") for t in BIT_TAGS]
        ax.plot(xb, ratios, "-o", color=GRAN_COLOR[gran], ms=7, label=GRAN_LABEL_FR[gran])
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(xb)))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("bits effectifs stockés (échelle log₂)")
    ax.set_ylabel("ratio RAM poids vs FP32 (×)")
    ax.set_title("Gain RAM théorique vs profondeur — EWC")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.005,
             "RAM théorique — gain réel sous réserve d'un kernel bit-packé (Sprint 48). "
             "Source : exp_S47_depth/ (ram_ratio_vs_fp32).", fontsize=8, color=MUTED)
    return fig


# ── symmetry_gain — symmetric vs affine aux bits critiques (zero-point) ──────

def _fig_symmetry_gain() -> plt.Figure:
    cols = [(ds, tag) for ds in DATASETS for tag in CRITICAL_TAGS]
    x = np.arange(len(cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    for k, sym in enumerate(["symmetric", "affine"]):
        offs = (k - 0.5) * width
        ys, na_x = [], []
        for xi, (ds, tag) in zip(x + offs, cols):
            v = _val(_sym(ds, tag, sym), "delta_auroc")
            ys.append(v)
            if np.isnan(v):
                na_x.append(xi)
        ax.bar(x + offs, ys, width, color=SYM_COLOR[sym], edgecolor="white",
               label=SYM_LABEL_FR[sym])
        for xi in na_x:
            ax.text(xi, 0.0, "n/a", ha="center", va="bottom", fontsize=8,
                    color="#999999", rotation=90)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds.capitalize()}\n{BIT_LABEL_FR[t]}" for ds, t in cols],
                       fontsize=8)
    ax.set_ylabel("Δ AUROC (quant − FP32)")
    ax.set_title("Zero-point affine vs symétrique aux bits critiques (activations post-ReLU)")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.01, 0.005,
             "Source : exp_S47_symmetry/ (granularité per-canal, gagnante S4703). "
             "Gain zero-point = Δ(affine) − Δ(symétrique). Badge « émulé PC ».",
             fontsize=8, color=MUTED)
    return fig


# ── scope_context — EWC balayé ∥ HDC/Maha/TinyOL N/A ─────────────────────────

def _fig_scope_context() -> plt.Figure:
    try:
        ctx, _ = load_experiment("experiments/exp_S47_context/context.json")
    except FileNotFoundError:
        ctx = {"swept_models": [], "context_models": {}}
    swept = ctx.get("swept_models", [])
    context_models = ctx.get("context_models", {})

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axis("off")
    ax.set_title("Périmètre du sweep profondeur/schéma — EWC balayé, autres N/A justifiés")

    # Cartouche EWC (balayé).
    ax.add_patch(plt.Rectangle((0.03, 0.72), 0.94, 0.2, color=STRATEGY_COLORS["fp32"],
                               alpha=0.15, zorder=1))
    swept_txt = ", ".join(m.upper() for m in swept) if swept else "(aucun)"
    ax.text(0.06, 0.86, f"Balayé (profondeur × granularité × symétrie) : {swept_txt}",
            fontsize=11, color=INK, fontweight="bold", zorder=2)
    ax.text(0.06, 0.78, "sub-INT8 {8,6,4,3,2,ternaire,binaire} × {per-tensor, per-canal} "
                        "× {symétrique, affine}", fontsize=9, color=MUTED, zorder=2)

    # Cartouches N/A justifiés (un par modèle contexte).
    y = 0.60
    labels = {"na_structural": "N/A structurel", "na_format_only": "N/A format-only",
              "na_out_of_scope": "N/A hors-périmètre"}
    for name, info in context_models.items():
        ax.add_patch(plt.Rectangle((0.03, y - 0.12), 0.94, 0.15, color=NA_GRAY,
                                   alpha=0.5, zorder=1))
        status = info.get("status", "")
        ax.text(0.06, y, f"{name.upper()} — {labels.get(status, status)}",
                fontsize=10, color=INK, fontweight="bold", zorder=2)
        reason = info.get("reason", "")
        ref = info.get("ref", "")
        ax.text(0.06, y - 0.06, _wrap(reason, 95) + f"  [{ref}]",
                fontsize=8, color=MUTED, zorder=2, va="top")
        y -= 0.19
    fig.text(0.01, 0.005,
             "Source : exp_S47_context/context.json — cadrage pur (aucun champ métrique).",
             fontsize=8, color=MUTED)
    return fig


def _wrap(text: str, width: int) -> str:
    """Enveloppe simple pour les cartouches (mise en page ; aucune donnée)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ── Build du catalogue ───────────────────────────────────────────────────────

@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 5 figures sous ``out_root/quantization_depth/`` ; retourne les chemins."""
    return [
        savefig_png(_fig_auroc_vs_bits(), OUT_SUBDIR, "auroc_vs_bits", out_root),
        savefig_png(_fig_heatmap(), OUT_SUBDIR, "heatmap_bits_granularity", out_root),
        savefig_png(_fig_ram_vs_bits(), OUT_SUBDIR, "ram_vs_bits", out_root),
        savefig_png(_fig_symmetry_gain(), OUT_SUBDIR, "symmetry_gain", out_root),
        savefig_png(_fig_scope_context(), OUT_SUBDIR, "scope_context", out_root),
    ]
