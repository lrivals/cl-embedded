"""Catalogue `manuscrit_final` — figures du manuscrit M2 (Sprint 41, S4109).

Isole en un seul dossier (``docs/figures/manuscrit_final/``) **toutes** les figures
destinées au manuscrit final, régénérables d'une commande :

    python scripts/generate_figures.py --catalog manuscrit_final --style manuscript

Principe : **chaque tableau de valeurs des chapitres 5, 6 et 7 possède ici sa version
tracée**, plus les figures de synthèse. Les figures reprennent des matériaux déjà
produits par d'autres catalogues/notebooks mais sont **régénérées ici** — jamais
copiées — pour garantir un style homogène et l'absence de dérive quand une
expérience est relancée.

Périmètre (décision utilisateur, 30 juillet 2026) :
  * ch. 5 — grille **complète** 4 modèles × 3 datasets : l'argument « accuracy
    trompeuse » repose sur HDC (F1 nul) et sur Mahalanobis × CMAPSS ;
  * ch. 6 et 7 — resserrés sur **EWC + Mahalanobis × Monitoring + Pronostia**,
    les jeux et modèles focus du manuscrit ; annotations réduites au strict utile.

Nomenclature : ``ch<N>_<sujet>.png`` — le préfixe donne le chapitre de destination.
Le descriptif complet (emplacement + légende) est dans ``LEGENDES_FIGURES.txt``,
généré par ce module.

**Toute valeur tracée provient d'un ``load_experiment``** — aucun littéral numérique
de résultat (garde AST ``test_no_hardcoded_results``). Donnée absente → panneau
explicite ou barre grise, **jamais 0**.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures import sources
from src.figures.registry import register_catalog
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "manuscrit_final"

# ── Périmètre ────────────────────────────────────────────────────────────────
CONDITION = "5feat"
# ch. 5 : grille complète (l'argument « accuracy trompeuse » a besoin de HDC/CMAPSS)
GRID_MODELS: list[str] = ["ewc", "hdc", "tinyol", "mahalanobis"]
GRID_DATASETS: list[str] = ["monitoring", "cmapss", "pronostia"]
# ch. 6 / 7 : jeux et modèles focus
FOCUS_DATASETS: list[str] = ["monitoring", "pronostia"]

MODEL_LABEL: dict[str, str] = {
    "ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "mahalanobis": "Mahalanobis",
}

#: Modèles dont les cellules **carte** sont invalides et ne doivent pas être tracées.
#: Vidé au Sprint 52 (S5201) : les cellules carte de TinyOL, jusque-là écartées parce que
#: la campagne empruntait le chemin d'inférence par défaut du firmware (flag UART absent)
#: et dupliquait donc Mahalanobis, ont été **re-mesurées** sur NUCLEO-F439ZI avec la route
#: TinyOL dédiée et des poids exportés à la dim de la condition — parité board↔PC vérifiée.
#: Conservé (vide) comme point d'entrée si une autre cellule devait être écartée.
INVALID_BOARD_MODELS: frozenset[str] = frozenset()
INVALID_BOARD_NOTE = "toutes les cellules carte tracées ici sont mesurées"
DATASET_LABEL: dict[str, str] = {
    "monitoring": "Monitoring (D2)",
    "cmapss": "CMAPSS (D5)",
    "pronostia": "Pronostia (D4)",
}

# ── Couleurs (réutilise la palette projet) ───────────────────────────────────
C_PC = STRATEGY_COLORS["fp32"]          # bleu — référence PC
C_BOARD = STRATEGY_COLORS["int8_v2"]    # orange — carte
C_FP32 = STRATEGY_COLORS["fp32"]
C_LEGACY = STRATEGY_COLORS["int8_ptq_legacy"]  # rouge — PTQ naïve
C_V2 = STRATEGY_COLORS["int8_v2"]
C_QAT = STRATEGY_COLORS["int8_qat"]     # vert
C_Q15 = STRATEGY_COLORS["q15"]
NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"

KIB = 1024
BADGE_BOARD = "mesuré sur carte réelle NUCLEO-F439ZI"
BADGE_PC = "référence PC (Python)"


# ── Helpers de chargement ────────────────────────────────────────────────────

# Les chargeurs vivent dans src/figures/sources.py : le manuscrit et la soutenance
# doivent lire chaque grandeur par le MEME chemin de code, sans quoi une expérience
# relancée pourrait les faire diverger en silence.
_try = sources.try_load
_num = sources.num


def _empty(msg: str, name: str, out_root: Path) -> Path:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, wrap=True)
    return savefig_png(fig, CATALOG, name, out_root)


def _footer(fig: plt.Figure, text: str) -> None:
    """Pied de figure + réservation de l'espace correspondant.

    ``bbox_inches="tight"`` étend la boîte pour inclure le texte mais n'empêche pas
    le chevauchement avec les étiquettes d'axe : on réserve donc explicitement une
    bande basse (et une bande haute quand un ``suptitle`` est présent).
    """
    top = 0.93 if fig._suptitle is not None else 0.97
    fig.tight_layout(rect=(0.0, 0.09, 1.0, top))
    fig.text(0.5, 0.015, text, ha="center", fontsize=7, color=MUTED, wrap=True)


def _annotate(ax, bars, values, fmt="{:.3f}", na_label="N/A") -> None:
    """Étiquette chaque barre ; les cellules non mesurées portent ``N/A``, pas 0."""
    for bar, val in zip(bars, values):
        label = fmt.format(val) if val is not None else na_label
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=7, color=MUTED)


def _bar_values(values: list[float | None]) -> tuple[list[float], list[str]]:
    """Hauteurs traçables + couleur grise pour les cellules non mesurées."""
    heights = [v if v is not None else 0.0 for v in values]
    return heights, [NA_GRAY if v is None else "" for v in values]


# ── Sources agrégées ─────────────────────────────────────────────────────────

def _s35_cell(model: str, dataset: str, platform: str) -> dict | None:
    """Cellule S35 condition 5feat — ``platform`` ∈ {``PC``, ``board``}.

    Renvoie ``None`` pour les cellules carte connues invalides
    (cf. :data:`INVALID_BOARD_MODELS`) : elles seront tracées en gris « N/A »
    au lieu d'être présentées comme des mesures.
    """
    return sources.s35_cell(
        model, dataset, platform, condition=CONDITION, exclude=INVALID_BOARD_MODELS
    )


def _s35_f1(model: str, dataset: str, platform: str) -> float | None:
    cell = _s35_cell(model, dataset, platform)
    return _num(cell.get("f1_faulty")) if cell else None


def _s35_acc(model: str, dataset: str, platform: str) -> float | None:
    cell = _s35_cell(model, dataset, platform)
    if not cell:
        return None
    for key in ("acc_final", "online_accuracy"):
        val = _num(cell.get(key))
        if val is not None:
            return val
    return None


def _s36_node(dataset: str, platform: str) -> dict | None:
    return sources.s36_node(dataset, platform, condition=CONDITION)


def _ewc_bss_split(n_features) -> tuple[float, float] | None:
    """Part `.bss` de la tête EWC à ``k`` entrées : ``(statique, non constante)``.

    Le découpage vient de :mod:`scripts.ram_breakdown`, qui l'établit depuis les
    ``#define`` de dimension des headers firmware et le **recoupe contre la taille
    ``nm`` réelle** de l'ELF (garde-fou anti-dérive). Seul ``EWC_IN`` est substitué :
    la condition ``5feat`` vaut k=4 sur Monitoring et k=5 sur Pronostia, alors que
    l'ELF présent n'est bâti que pour l'une des deux dimensions.

    ``None`` si les headers sont hors d'atteinte — la figure le signale, elle
    n'invente pas de découpage.
    """
    if not isinstance(n_features, int) or isinstance(n_features, bool):
        return None
    try:
        from scripts import ram_breakdown as rb

        defines = dict(rb.read_defines())
        defines["EWC_IN"] = n_features
        ewc = rb.monitoring_layout(defines)["EWC"]
    except Exception:  # headers absents, API déplacée → pas de découpage inventé
        return None
    return float(ewc["static"]), float(ewc["modular"])


def _s49_board(model: str, dataset: str, encoding: str) -> dict | None:
    return sources.s49_board(model, dataset, encoding, condition=CONDITION)


# =============================================================================
# CHAPITRE 5 — Gap 1 : validation sur données industrielles
# =============================================================================

def _fig_ch5_grille(out_root: Path) -> Path:
    """Grille F1 4 modèles × 3 datasets, PC vs carte — version tracée du tableau du ch. 5."""
    name = "ch5_f1_grille_pc_board"
    fig, axes = plt.subplots(1, len(GRID_DATASETS), sharey=True, figsize=(9.6, 3.4))
    x = np.arange(len(GRID_MODELS))
    width = 0.38
    any_data = False
    for ax, dataset in zip(axes, GRID_DATASETS):
        for i, (platform, color, label) in enumerate(
            [("PC", C_PC, "PC"), ("board", C_BOARD, "carte")]
        ):
            vals = [_s35_f1(m, dataset, platform) for m in GRID_MODELS]
            any_data = any_data or any(v is not None for v in vals)
            heights, na = _bar_values(vals)
            colors = [g if g else color for g in na]
            bars = ax.bar(x + (i - 0.5) * width, heights, width, color=colors,
                          edgecolor=INK, linewidth=0.4,
                          label=label if dataset == GRID_DATASETS[0] else None)
            _annotate(ax, bars, vals, fmt="{:.2f}")
        ax.set_title(DATASET_LABEL[dataset], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in GRID_MODELS], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
    if not any_data:
        return _empty("Grille S35 absente — relancer le sweep S3503/S3508", name, out_root)
    axes[0].set_ylabel("F1 classe « fautif »")
    # Légende hors des axes : sur ce jeu, une légende interne masque les barres EWC.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Gap 1 — F1 par modèle et par jeu de données (condition 5feat)", fontsize=11)
    _footer(fig, f"{BADGE_PC} vs {BADGE_BOARD} · 5feat · gris = cellule non mesurée")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch5_accuracy_trompeuse(out_root: Path) -> Path:
    """Accuracy vs F1 sur carte : le cœur du message « l'accuracy est trompeuse »."""
    name = "ch5_accuracy_vs_f1"
    pairs: list[tuple[str, float, float]] = []
    for model in GRID_MODELS:
        for dataset in GRID_DATASETS:
            acc = _s35_acc(model, dataset, "board")
            f1 = _s35_f1(model, dataset, "board")
            if acc is not None and f1 is not None:
                pairs.append((f"{MODEL_LABEL[model]}\n{dataset}", acc, f1))
    if not pairs:
        return _empty("Grille S35 board absente", name, out_root)
    labels = [p[0] for p in pairs]
    accs = [p[1] for p in pairs]
    f1s = [p[2] for p in pairs]
    order = np.argsort(accs)[::-1]
    labels = [labels[i] for i in order]
    accs = [accs[i] for i in order]
    f1s = [f1s[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, accs, width, color=C_PC, edgecolor=INK, linewidth=0.4,
           label="accuracy")
    ax.bar(x + width / 2, f1s, width, color=C_LEGACY, edgecolor=INK, linewidth=0.4,
           label="F1 « fautif »")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Une accuracy élevée peut masquer un modèle qui ne détecte aucune panne",
                 fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · 5feat · TinyOL carte écarté")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch5_oubli(out_root: Path) -> Path:
    """Oubli catastrophique EWC multiclasse : la moyenne post-tâche masque l'effondrement."""
    name = "ch5_oubli_catastrophique"
    data = _try("experiments/exp_S26_02/results.json")
    if data is None:
        return _empty("exp_S26_02 absent", name, out_root)
    series = [
        ("Moyenne des F1\nrelevés après\nchaque tâche", _num(data.get("f1_macro_pc_per_task_mean")), C_QAT),
        ("Modèle final,\ntoutes tâches\n(PC)", _num(data.get("f1_macro_pc_final_all_tasks")), C_LEGACY),
        ("Carte,\ninférence pure", _num(data.get("f1_macro_board_inference")), C_BOARD),
        ("Carte,\nen ligne", _num(data.get("f1_macro_board_online")), C_V2),
    ]
    labels = [s[0] for s in series]
    values = [s[1] for s in series]
    colors = [s[2] for s in series]
    if all(v is None for v in values):
        return _empty("champs F1 absents de exp_S26_02", name, out_root)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    heights, na = _bar_values(values)
    bars = ax.bar(labels, heights, color=[g if g else c for g, c in zip(na, colors)],
                  edgecolor=INK, linewidth=0.5, width=0.6)
    _annotate(ax, bars, values)
    forgetting = _num(data.get("avg_forgetting_f1_pc"))
    if forgetting is not None:
        ax.annotate(f"oubli moyen $AF_{{F1}}$ = {forgetting:.3f}",
                    xy=(0.5, 0.88), xycoords="axes fraction", ha="center",
                    fontsize=9, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6))
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("L'oubli catastrophique n'apparaît qu'avec la bonne métrique", fontsize=11)
    _footer(fig, "EWC multiclasse · CWRU, 3 tâches · parité carte↔PC exacte")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch5_paderborn(out_root: Path) -> Path:
    """Paderborn (mono-classe par tâche) : EWC seul préserve un F1 utile."""
    name = "ch5_paderborn_ewc_seul"
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(GRID_MODELS))
    width = 0.38
    any_data = False
    for i, (platform, color, label) in enumerate(
        [("PC", C_PC, "PC"), ("board", C_BOARD, "carte")]
    ):
        vals = [_s35_f1(m, "paderborn", platform) for m in GRID_MODELS]
        any_data = any_data or any(v is not None for v in vals)
        heights, na = _bar_values(vals)
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=label)
        _annotate(ax, bars, vals)
    if not any_data:
        return _empty("cellules Paderborn S35 absentes", name, out_root)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in GRID_MODELS])
    ax.set_ylabel("F1 classe « fautif »")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("Paderborn — scénario mono-classe par tâche (condition 5feat)", fontsize=11)
    _footer(fig, f"{BADGE_PC} vs {BADGE_BOARD} · 5feat · TinyOL carte écarté")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# CHAPITRE 6 — Gap 2 : RAM, latence, parité
# =============================================================================

def _fig_ch6_ram_niveaux(out_root: Path) -> Path:
    """Les trois niveaux de claim RAM, rapportés au budget de la carte."""
    name = "ch6_ram_trois_niveaux"
    minimal = _try("experiments/exp_S18_01_board/results.json")
    s48 = _try("experiments/exp_S48_summary.json")
    sweep = _try("experiments/exp_S35_board_sweep_summary.json")

    v_min = _num(minimal.get("ram_peak_bytes")) if minimal else None
    v_default = _num(s48.get("bss_default_invariant")) if s48 else None
    v_worst = None
    if isinstance(sweep, list):
        cand = [_num(c.get("bss_bytes")) for c in sweep if isinstance(c, dict)]
        cand = [c for c in cand if c is not None]
        v_worst = max(cand) if cand else None
    budget = _num(s48.get("ram_budget_bytes")) if s48 else None
    if all(v is None for v in (v_min, v_default, v_worst)):
        return _empty("sources RAM absentes (S18/S48/S35)", name, out_root)

    labels = ["Noyau minimal\n(tête EWC seule)",
              "Système multi-modèle\n(build par défaut)",
              "Pire cas mesuré\n(toutes variables)"]
    values = [v_min, v_default, v_worst]
    colors = [C_QAT, C_PC, C_LEGACY]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    heights = [(v / KIB) if v is not None else 0.0 for v in values]
    na = [v is None for v in values]
    bars = ax.bar(labels, heights, color=[NA_GRAY if n else c for n, c in zip(na, colors)],
                  edgecolor=INK, linewidth=0.5, width=0.55)
    for bar, val in zip(bars, values):
        txt = f"{val / KIB:.1f} Kio" if val is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), txt,
                ha="center", va="bottom", fontsize=8, color=MUTED)
    if budget is not None:
        ax.axhline(budget / KIB, color=INK, linestyle="--", linewidth=1.0)
        ax.text(0.99, budget / KIB, " budget carte", ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), fontsize=8, color=INK)
        ax.set_ylim(0, budget / KIB * 1.12)
    ax.set_ylabel("RAM statique (Kio)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Gap 2 — trois niveaux de claim RAM, tous dans le budget", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · `.bss` seul (pile comptée séparément)")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch6_ram_totale(out_root: Path) -> Path:
    """RAM totale décomposée en barres horizontales, façon « plancher RAM ».

    Reprend la lecture du panneau ① de ``fig_lifecycle_ewc.png`` (notebook
    ``ram_measurement``) — segments empilés horizontalement, du permanent vers le
    transitoire — mais **à l'échelle du firmware entier mesuré** (S49), afin que le
    total tracé reste celui qui porte la revendication Gap 2.

    Une seule barre par jeu : FP32 et INT8 partagent le même binaire (les deux têtes
    coexistent en ``.bss``, la précision est choisie à l'exécution), leurs ``.data`` et
    ``.bss`` sont donc byte-identiques — deux barres jumelles n'apprendraient rien.
    """
    name = "ch6_ram_totale_decomposee"
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    summary = _try("experiments/exp_S49_ram/summary.json")
    budget = _num((summary or {}).get("_meta", {}).get("ram_budget_bytes"))

    rows: list[tuple[str, list[tuple[str, float, str, str]]]] = []
    for dataset in FOCUS_DATASETS:
        cell = _s49_board("ewc", dataset, "fp32")
        if cell is None:
            continue
        data_b = _num(cell.get("data"))
        bss_b = _num(cell.get("bss"))
        peaks = [_num(cell.get("stack_peak_inference")), _num(cell.get("stack_peak_update"))]
        peaks = [p for p in peaks if p is not None]
        if data_b is None or bss_b is None or not peaks:
            continue
        # `n_features` n'est pas remonté dans le résumé agrégé : il vit dans la
        # cellule brute, seule source de la dimension d'entrée réellement flashée.
        raw = _try(f"experiments/exp_S49_ram/ewc_{dataset}_{CONDITION}_fp32_board.json")
        split = _ewc_bss_split((raw or {}).get("n_features"))
        segs: list[tuple[str, float, str, str]] = [(".data", data_b, C_LEGACY, "")]
        if split is None:
            segs.append((".bss", bss_b, STRATEGY_COLORS["int16_am"], ""))
        else:
            static_b, modular_b = split
            segs += [
                ("tête EWC — statique\n(poids figés)", static_b, C_PC, ""),
                ("tête EWC — non constante\n(Fisher, θ*, λ)", modular_b, C_V2, ""),
                ("reste du firmware\n(autres modèles, tampons)",
                 bss_b - static_b - modular_b, STRATEGY_COLORS["int16_am"], ""),
            ]
        segs.append(("pic de pile\n(mise à jour)", max(peaks), C_QAT, "//"))
        rows.append((DATASET_LABEL[dataset], segs))
    if not rows:
        return _empty("exp_S49_ram absent — lancer aggregate_ram.py", name, out_root)

    def _o(val: float) -> str:
        return f"{val:,.0f} o".replace(",", " ")

    totals = [sum(v for _, v, _, _ in segs) for _, segs in rows]
    span = max(totals)
    ys = np.arange(len(rows))[::-1]
    seen: set[str] = set()
    for y, (_, segs) in zip(ys, rows):
        left = 0.0
        run: list[tuple[str, float, float]] = []  # segments trop fins, consécutifs
        runs: list[list[tuple[str, float, float]]] = []
        for label, val, color, hatch in segs:
            ax.barh(y, val, left=left, height=0.4, color=color, edgecolor=INK,
                    linewidth=0.5, hatch=hatch,
                    label=label if label not in seen else None)
            seen.add(label)
            if val > span * 0.12:  # assez large pour porter son étiquette à l'intérieur
                ax.text(left + val / 2, y, _o(val), ha="center", va="center",
                        fontsize=8, color="white")
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append((label.split("\n")[0], left + val / 2, val))
            left += val
        if run:
            runs.append(run)
        pct = f" — {100 * left / budget:.1f} % du budget" if budget else ""
        ax.text(left + span * 0.01, y, _o(left) + pct,
                ha="left", va="center", fontsize=8, color=INK, weight="bold")

        # Les segments fins forment deux amas (les composants de la tête EWC à
        # gauche, la pile à droite) : un renvoi groupé par amas reste lisible là où
        # une étiquette par segment se chevaucherait — `.data` pèse moins de 1 % du
        # total et resterait sinon muet.
        for group in runs:
            xc = sum(x for _, x, _ in group) / len(group)
            txt = "  ·  ".join(f"{lbl} {_o(v)}" for lbl, _, v in group)
            above = xc > span / 2
            # Ancrage rabattu vers l'intérieur près des bords : un renvoi centré sur
            # un amas proche de x=0 déborderait hors du cadre.
            ha = "left" if xc < span * 0.25 else ("right" if xc > span * 0.8 else "center")
            ax.annotate(txt, xy=(xc, y + (0.2 if above else -0.2)),
                        xytext=(xc, y + (0.5 if above else -0.5)),
                        ha=ha, va="bottom" if above else "top",
                        fontsize=7, color=MUTED,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))

    ax.set_yticks(ys)
    ax.set_yticklabels([lbl for lbl, _ in rows], fontsize=9)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xlim(0, span * 1.2)
    ax.set_xlabel("RAM (octets)")
    ax.legend(fontsize=7, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_title("RAM totale mesurée sur carte — la tête EWC dans le plancher firmware",
                 fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · plancher `.data` + `.bss` relevé à la compilation, "
                 "pic de pile par stack painting · découpage EWC recoupé sur `nm`")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch6_pic_pile(out_root: Path) -> Path:
    """Le pic de pile par phase : la mise à jour continue creuse plus que l'inférence."""
    name = "ch6_pic_pile_par_phase"
    phases = ["idle", "inference", "update"]
    phase_label = {"idle": "idle", "inference": "inférence", "update": "mise à jour CL"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    plotted = False
    for dataset, marker in zip(FOCUS_DATASETS, ("o", "s")):
        cell = _s49_board("ewc", dataset, "fp32")
        if cell is None:
            continue
        hist = {h.get("phase"): _num(h.get("stack_peak_bytes"))
                for h in cell.get("stack_history", [])}
        ys = [(hist.get(p) / KIB if hist.get(p) is not None else None) for p in phases]
        if all(y is None for y in ys):
            continue
        # Les deux courbes sont quasi confondues : étiqueter chaque point les ferait
        # se chevaucher sans rien apprendre — l'axe porte déjà la valeur.
        ax.plot(range(len(phases)), ys, marker=marker, color=C_BOARD if plotted else C_PC,
                label=DATASET_LABEL[dataset])
        plotted = True
    if not plotted:
        return _empty("historique de pile absent (exp_S49_ram)", name, out_root)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([phase_label[p] for p in phases])
    ax.set_ylabel("pic de pile (Kio)")
    ax.legend(fontsize=8)
    ax.set_title("Le coût mémoire de l'adaptation en ligne — EWC", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · pic relevé après chaque phase")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch6_latence_inf_maj(out_root: Path) -> Path:
    """Inférence seule vs inférence + mise à jour CL, rapporté au budget 100 ms."""
    name = "ch6_latence_inference_vs_maj"
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(FOCUS_DATASETS))
    width = 0.35
    inf_v, upd_v = [], []
    for dataset in FOCUS_DATASETS:
        frozen = _s36_node(dataset, "board_frozen")
        online = _s36_node(dataset, "board_online")
        inf_v.append(_num(frozen.get("latency_us_p50")) if frozen else None)
        upd_v.append(_num(online.get("latency_us_p50")) if online else None)
    if all(v is None for v in inf_v + upd_v):
        return _empty("exp_S36_summary absent", name, out_root)
    for i, (vals, color, label) in enumerate(
        [(inf_v, C_PC, "inférence seule (gelé)"),
         (upd_v, C_BOARD, "inférence + mise à jour CL (en ligne)")]
    ):
        heights, na = _bar_values(vals)
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=label)
        _annotate(ax, bars, vals, fmt="{:.0f} µs")
    for xi, (a, b) in enumerate(zip(inf_v, upd_v)):
        if a and b:
            ax.annotate(f"×{b / a:.1f}", xy=(xi, max(a, b)), textcoords="offset points",
                        xytext=(0, 18), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in FOCUS_DATASETS])
    ax.set_ylabel("latence DWT P50 (µs)")
    finite = [v for v in inf_v + upd_v if v is not None]
    if finite:
        ax.set_ylim(0, max(finite) * 1.45)  # marge pour le ratio et la légende
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("EWC — le surcoût de l'apprentissage en ligne", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · budget Gap 2 = 100 ms")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch6_latence_modeles(out_root: Path) -> Path:
    """Dispersion des latences d'inférence entre modèles, échelle logarithmique."""
    name = "ch6_latence_par_modele"
    sweep = _try("experiments/exp_S35_board_sweep_summary.json")
    if not isinstance(sweep, list):
        return _empty("exp_S35_board_sweep_summary absent", name, out_root)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(GRID_MODELS))
    width = 0.38
    any_data = False
    for i, (dataset, color) in enumerate(zip(FOCUS_DATASETS, (C_PC, C_BOARD))):
        vals: list[float | None] = []
        for model in GRID_MODELS:
            if model in INVALID_BOARD_MODELS:
                vals.append(None)  # même chemin d'inférence que Mahalanobis : non mesuré
                continue
            match = [c for c in sweep if isinstance(c, dict)
                     and c.get("model") == model and c.get("dataset") == dataset
                     and c.get("condition") == CONDITION]
            vals.append(_num(match[0].get("latency_us_p50")) if match else None)
        any_data = any_data or any(v is not None for v in vals)
        heights, na = _bar_values(vals)
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=DATASET_LABEL[dataset])
        _annotate(ax, bars, vals, fmt="{:.0f}")
    if not any_data:
        return _empty("aucune latence board S35 exploitable", name, out_root)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in GRID_MODELS], rotation=15)
    ax.set_ylabel("latence DWT P50 (µs, échelle log)")
    ax.legend(fontsize=8)
    ax.set_title("Latence d'inférence par modèle — deux ordres de grandeur d'écart", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · 5feat · TinyOL carte écarté")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch6_parite(out_root: Path) -> Path:
    """Parité PC ↔ carte : exacte à poids figés, approchée et expliquée en ligne."""
    name = "ch6_parite_gele_vs_online"
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(FOCUS_DATASETS))
    width = 0.35
    frozen_v, online_v = [], []
    for dataset in FOCUS_DATASETS:
        fr = _s36_node(dataset, "board_frozen")
        on = _s36_node(dataset, "board_online")
        frozen_v.append(_num(fr.get("parity_rate")) if fr else None)
        online_v.append(_num(on.get("parity_rate")) if on else None)
    if all(v is None for v in frozen_v + online_v):
        return _empty("exp_S36_summary absent", name, out_root)
    for i, (vals, color, label) in enumerate(
        [(frozen_v, C_QAT, "régime gelé"), (online_v, C_V2, "régime en ligne")]
    ):
        heights, na = _bar_values(vals)
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=label)
        _annotate(ax, bars, vals, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in FOCUS_DATASETS])
    ax.set_ylabel("parité PC ↔ carte")
    ax.set_ylim(0.9, 1.01)
    # Les barres occupent toute la hauteur : légende hors des axes.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, bbox_to_anchor=(0.99, 0.99))
    ax.set_title("Validation du portage — parité prédiction à prédiction", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · axe tronqué")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# CHAPITRE 7 — Gap 3 : quantification
# =============================================================================

#: Étiquettes lisibles des étages d'ablation (les JSON portent les noms techniques).
ABLATION_LABEL: dict[str, str] = {
    "legacy_c": "firmware\nhistorique",
    "fix_acc32": "+ accumulateur\nint32",
    "per_tensor_calib": "+ échelle\ncalibrée",
    "per_channel_int8": "+ échelle\npar canal",
    "q15": "Q15\n(16 bits)",
}


def _fig_ch7_ablation(out_root: Path) -> Path:
    """Échelle d'ablation : une seule cause domine, la calibration de l'échelle."""
    name = "ch7_ablation_echelle"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    plotted = False
    names: list[str] = []
    for idx, (dataset, color, marker) in enumerate(
        zip(FOCUS_DATASETS, (C_PC, C_BOARD), ("o", "s"))
    ):
        data = _try(f"experiments/exp_S39_ablation/{dataset}.json")
        if data is None:
            continue
        ladder = data.get("ladder", [])
        names = [str(step.get("scheme")) for step in ladder]
        vals = [_num(step.get("f1")) for step in ladder]
        if not names:
            continue
        ax.plot(range(len(names)), vals, marker=marker, color=color,
                label=DATASET_LABEL[dataset])
        ref = _num(data.get("f1_fp32"))
        if ref is not None:
            ax.axhline(ref, color=color, linestyle=":", linewidth=1.0, alpha=0.6)
        # met en évidence le seul saut décisif
        best = None
        for idx in range(1, len(vals)):
            if vals[idx] is not None and vals[idx - 1] is not None:
                gain = vals[idx] - vals[idx - 1]
                if best is None or gain > best[1]:
                    best = (idx, gain)
        if best is not None:
            # décalage vertical par série : les deux gains sont quasi identiques
            ax.annotate(f"+{best[1]:.3f}", xy=(best[0], vals[best[0]]),
                        textcoords="offset points", xytext=(12, -6 - idx * 26),
                        fontsize=8, color=color)
        plotted = True
    if not plotted:
        return _empty("exp_S39_ablation absent", name, out_root)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([ABLATION_LABEL.get(n, n) for n in names], fontsize=8)
    ax.set_ylabel("F1 classe « fautif »")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("Diagnostic de l'effondrement INT8 — le facteur décisif est la calibration",
                 fontsize=11)
    _footer(fig, "émulateur bit-exact du noyau embarqué (PC)")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch7_recuperation(out_root: Path) -> Path:
    """FP32 → PTQ naïve → noyau v2 calibré : la métrique est récupérée sur carte."""
    name = "ch7_noyau_v2_recuperation"
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(FOCUS_DATASETS))
    width = 0.27
    series: list[tuple[str, list[float | None], str]] = []
    fp32_v, legacy_v, v2_v = [], [], []
    for dataset in FOCUS_DATASETS:
        frozen = _s36_node(dataset, "board_frozen")
        legacy = _s36_node(dataset, "board_frozen_int8")
        v2 = _try(f"experiments/exp_S40_board_v2/results_per_channel_{dataset}_frozen.json")
        fp32_v.append(_num(frozen.get("f1_faulty")) if frozen else None)
        legacy_v.append(_num(legacy.get("f1_faulty")) if legacy else None)
        v2_v.append(_num(v2.get("f1_faulty")) if v2 else None)
    series = [("FP32 (référence)", fp32_v, C_FP32),
              ("INT8 PTQ naïve", legacy_v, C_LEGACY),
              ("INT8 v2 calibré", v2_v, C_V2)]
    if all(v is None for _, vals, _ in series for v in vals):
        return _empty("sources S36/S40 absentes", name, out_root)
    for i, (label, vals, color) in enumerate(series):
        heights, na = _bar_values(vals)
        bars = ax.bar(x + (i - 1) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=label)
        _annotate(ax, bars, vals)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d] for d in FOCUS_DATASETS])
    ax.set_ylabel("F1 classe « fautif »")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Gap 3 — la perte venait de l'implémentation, pas du principe INT8",
                 fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · régime gelé")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch7_moment(out_root: Path) -> Path:
    """Moment de la quantification : avant / après / les deux."""
    name = "ch7_moment_quantification"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    moments = ["fp32", "before", "after", "both"]
    moment_label = {"fp32": "FP32", "before": "avant\n(QAT)",
                    "after": "après\n(PTQ calibrée)", "both": "les deux\n(déploiement)"}
    colors = [C_FP32, C_QAT, C_V2, C_Q15]
    plotted = False
    for ax, dataset in zip(axes, FOCUS_DATASETS):
        data = _try(f"experiments/exp_S46_ewc/{dataset}_all.json")
        if data is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "exp_S46_ewc absent", ha="center", color=MUTED)
            continue
        node = data.get("moments", {})
        vals = [_num(node.get(m, {}).get("metric")) for m in moments]
        heights, na = _bar_values(vals)
        bars = ax.bar([moment_label[m] for m in moments], heights,
                      color=[g if g else c for g, c in zip(na, colors)],
                      edgecolor=INK, linewidth=0.4, width=0.6)
        _annotate(ax, bars, vals, fmt="{:.4f}")
        finite = [v for v in vals if v is not None]
        if finite:
            span = max(finite) - min(finite)
            ax.set_ylim(min(finite) - max(span, 1e-3) * 2, max(finite) + max(span, 1e-3) * 2)
        ax.set_title(DATASET_LABEL[dataset], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        plotted = True
    if not plotted:
        return _empty("exp_S46_ewc absent", name, out_root)
    axes[0].set_ylabel(f"{'AUROC'}")
    fig.suptitle("Le moment de la quantification pèse peu — la calibration décide", fontsize=11)
    _footer(fig, "émulateur bit-exact (PC) · échelle verticale très resserrée")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch7_profondeur(out_root: Path) -> Path:
    """Profondeur sub-INT8 : métrique préservée jusqu'où, et gain RAM conditionnel au packing."""
    name = "ch7_profondeur_bits"
    s48 = _try("experiments/exp_S48_summary.json")
    if s48 is None:
        return _empty("exp_S48_summary absent", name, out_root)
    grid = s48.get("results_by_condition", {})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # Panneau A — écart de métrique vs FP32 en fonction de la profondeur (émulateur PC)
    ax = axes[0]
    plotted = False
    for dataset, color, marker in zip(FOCUS_DATASETS, (C_PC, C_BOARD), ("o", "s")):
        node = grid.get(dataset, {})
        bits, deltas = [], []
        for b in sorted(node, key=lambda k: -int(k)):
            pc = node[b].get("per_channel", {}).get("pc", {})
            d = _num(pc.get("delta_auroc_vs_fp32"))
            if d is not None:
                bits.append(int(b))
                deltas.append(d)
        if bits:
            ax.plot(bits, deltas, marker=marker, color=color, label=DATASET_LABEL[dataset])
            plotted = True
    if plotted:
        ax.invert_xaxis()
        ax.set_xticks(sorted({int(b) for b in grid.get(FOCUS_DATASETS[0], {})}))
        ax.set_xlabel("bits de poids")
        ax.set_ylabel("écart d'AUROC vs FP32")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Métrique : jusqu'où descendre ?", fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pas de delta PC exploitable", ha="center", color=MUTED)

    # Panneau B — .bss packé vs non packé (le gain n'existe qu'avec bit-packing)
    ax = axes[1]
    cats, np_v, p_v = [], [], []
    for dataset in FOCUS_DATASETS:
        node = grid.get(dataset, {})
        for b in sorted(node, key=lambda k: -int(k)):
            board = node[b].get("per_channel", {}).get("board", {})
            npb = _num(board.get("nonpacked", {}).get("bss_bytes"))
            pb = _num(board.get("packed", {}).get("bss_bytes"))
            if npb is None and pb is None:
                continue
            mode = node[b].get("per_channel", {}).get("mode", b)
            cats.append(f"{DATASET_LABEL[dataset].split(' ')[0][:4]}\n{mode}")
            np_v.append((npb / KIB) if npb is not None else 0.0)
            p_v.append((pb / KIB) if pb is not None else 0.0)
    if cats:
        x = np.arange(len(cats))
        width = 0.38
        ax.bar(x - width / 2, np_v, width, color=C_LEGACY, label="non packé",
               edgecolor=INK, linewidth=0.4)
        ax.bar(x + width / 2, p_v, width, color=C_QAT, label="bit-packé",
               edgecolor=INK, linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=7)
        ax.set_ylabel(".bss (Kio)")
        lo = min(v for v in np_v + p_v if v)
        hi = max(np_v + p_v)
        ax.set_ylim(lo - (hi - lo) * 2 - 0.1, hi + (hi - lo) * 0.5 + 0.1)
        ax.legend(fontsize=8)
        ax.set_title("RAM : le gain exige le bit-packing", fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pas de .bss board exploitable", ha="center", color=MUTED)

    fig.suptitle("Profondeur sub-INT8 — métrique et RAM réelle", fontsize=11)
    _footer(fig, "gauche : émulateur bit-exact (PC) · droite : " f"{BADGE_BOARD}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch7_latence_breakdown(out_root: Path) -> Path:
    """Décomposition cycle par cycle du noyau INT8 : le paradoxe de latence expliqué."""
    name = "ch7_latence_int8_breakdown"
    data = _try("experiments/exp_S50_int8_latency/ewc.json")
    if data is None:
        return _empty("exp_S50_int8_latency absent", name, out_root)
    segs = data.get("segments", {})
    order = ["dequant", "mac", "requant"]
    seg_label = {"dequant": "déquantification\nentier → FP32",
                 "mac": "produits scalaires\n(MAC entier)",
                 "requant": "requantification\nFP32 → INT8"}
    colors = [C_QAT, C_PC, C_LEGACY]
    vals = [_num(segs.get(s, {}).get("cycles_p50_mean_over_datasets")) for s in order]
    if all(v is None for v in vals):
        return _empty("segments absents de exp_S50_int8_latency", name, out_root)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    heights, na = _bar_values(vals)
    bars = ax.bar([seg_label[s] for s in order], heights,
                  color=[g if g else c for g, c in zip(na, colors)],
                  edgecolor=INK, linewidth=0.4, width=0.6)
    _annotate(ax, bars, vals, fmt="{:.0f}")
    ax.set_ylabel("cycles DWT (P50)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Où passent les cycles du noyau INT8", fontsize=10)

    ax = axes[1]
    int8 = data.get("total_int8_us_p50_by_dataset", {})
    fp32 = data.get("total_fp32_us_p50_by_dataset", {})
    x = np.arange(len(FOCUS_DATASETS))
    width = 0.35
    for i, (src, color, label) in enumerate(
        [(fp32, C_FP32, "FP32"), (int8, C_V2, "INT8")]
    ):
        vv = [_num(src.get(d)) for d in FOCUS_DATASETS]
        heights, na = _bar_values(vv)
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=[g if g else color for g in na],
                      edgecolor=INK, linewidth=0.4, label=label)
        _annotate(ax, bars, vv, fmt="{:.0f} µs")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL[d].split(" ")[0] for d in FOCUS_DATASETS], fontsize=8)
    ax.set_ylabel("latence totale P50 (µs)")
    ax.legend(fontsize=8)
    ax.set_title("L'INT8 est plus lent que le FP32", fontsize=10)

    fig.suptitle("Le paradoxe de latence, chiffré poste par poste", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} à 180 MHz · cycles DWT bruts")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_ch7_ram_poids_vs_systeme(out_root: Path) -> Path:
    """Le gain RAM ÷4 porte sur les poids, pas sur le firmware complet."""
    name = "ch7_ram_poids_vs_systeme"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # Panneau A — RAM des poids FP32 vs quantifiée
    ax = axes[0]
    cats, fp_v, q_v = [], [], []
    for dataset in FOCUS_DATASETS:
        v2 = _try(f"experiments/exp_S40_board_v2/results_per_channel_{dataset}_frozen.json")
        if v2 is None:
            continue
        fp = _num(v2.get("ram_weights_fp32_bytes"))
        qq = _num(v2.get("ram_weights_quant_bytes"))
        if fp is None and qq is None:
            continue
        cats.append(DATASET_LABEL[dataset].split(" ")[0])
        fp_v.append(fp or 0.0)
        q_v.append(qq or 0.0)
    if cats:
        x = np.arange(len(cats))
        width = 0.35
        b1 = ax.bar(x - width / 2, fp_v, width, color=C_FP32, label="FP32",
                    edgecolor=INK, linewidth=0.4)
        b2 = ax.bar(x + width / 2, q_v, width, color=C_V2, label="INT8",
                    edgecolor=INK, linewidth=0.4)
        for bars, vals in ((b1, fp_v), (b2, q_v)):
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f} o", ha="center", va="bottom", fontsize=7, color=MUTED)
        for xi, (a, b) in enumerate(zip(fp_v, q_v)):
            if a and b:
                ax.annotate(f"÷{a / b:.0f}", xy=(xi, a), textcoords="offset points",
                            xytext=(0, 16), ha="center", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel("RAM des poids (octets)")
        ax.set_ylim(0, max(fp_v) * 1.3)  # marge pour l'annotation de ratio
        ax.legend(fontsize=8, loc="center right")
        ax.set_title("À l'échelle du modèle", fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "exp_S40_board_v2 absent", ha="center", color=MUTED)

    # Panneau B — RAM totale système FP32 vs INT8
    ax = axes[1]
    cats, tot_fp, tot_int = [], [], []
    for dataset in FOCUS_DATASETS:
        c_fp = _s49_board("ewc", dataset, "fp32")
        c_i8 = _s49_board("ewc", dataset, "int8")
        if c_fp is None and c_i8 is None:
            continue
        cats.append(DATASET_LABEL[dataset].split(" ")[0])
        tot_fp.append(((_num(c_fp.get("total")) or 0.0) / KIB) if c_fp else 0.0)
        tot_int.append(((_num(c_i8.get("total")) or 0.0) / KIB) if c_i8 else 0.0)
    if cats:
        x = np.arange(len(cats))
        width = 0.35
        ax.bar(x - width / 2, tot_fp, width, color=C_FP32, label="FP32",
               edgecolor=INK, linewidth=0.4)
        ax.bar(x + width / 2, tot_int, width, color=C_V2, label="INT8",
               edgecolor=INK, linewidth=0.4)
        for xi, (a, b) in enumerate(zip(tot_fp, tot_int)):
            if a and b:
                ax.annotate(f"×{b / a:.4f}", xy=(xi, max(a, b)), textcoords="offset points",
                            xytext=(0, 14), ha="center", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel("RAM totale système (Kio)")
        ax.set_ylim(0, max(tot_fp + tot_int) * 1.3)  # marge pour l'annotation de ratio
        ax.legend(fontsize=8, loc="center right")
        ax.set_title("À l'échelle du firmware", fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "exp_S49_ram absent", ha="center", color=MUTED)

    fig.suptitle("Le gain INT8 est réel sur les poids, marginal sur le système", fontsize=11)
    _footer(fig, f"{BADGE_BOARD} · RAM des poids vs RAM système")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================

FIGURES = [
    ("ch5", _fig_ch5_grille),
    ("ch5", _fig_ch5_accuracy_trompeuse),
    ("ch5", _fig_ch5_oubli),
    ("ch5", _fig_ch5_paderborn),
    ("ch6", _fig_ch6_ram_niveaux),
    ("ch6", _fig_ch6_ram_totale),
    ("ch6", _fig_ch6_pic_pile),
    ("ch6", _fig_ch6_latence_inf_maj),
    ("ch6", _fig_ch6_latence_modeles),
    ("ch6", _fig_ch6_parite),
    ("ch7", _fig_ch7_ablation),
    ("ch7", _fig_ch7_recuperation),
    ("ch7", _fig_ch7_moment),
    ("ch7", _fig_ch7_profondeur),
    ("ch7", _fig_ch7_latence_breakdown),
    ("ch7", _fig_ch7_ram_poids_vs_systeme),
]


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les figures du manuscrit sous ``out_root/manuscrit_final/``."""
    apply_style("manuscript")
    return [fn(out_root) for _chapter, fn in FIGURES]
