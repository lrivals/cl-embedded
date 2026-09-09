"""Catalogue `soutenance` — les figures projetées le jour de la soutenance.

    python scripts/generate_figures.py --catalog soutenance --style slide

Ce catalogue **ne duplique pas** ``manuscrit_final`` : il lit les mêmes mesures, par les
mêmes chargeurs (:mod:`src.figures.sources`), mais les met en forme pour la projection.
Deux règles gouvernent chaque figure, et expliquent les écarts avec la version manuscrit :

1. **Une figure = une comparaison.** Là où le manuscrit empile des barres groupées et une
   étiquette par barre, la version projetée encode la *relation* — pente, longueur
   d'haltère, marche de cascade — et n'écrit le chiffre que là où l'orateur le prononce.
   Le jury a le manuscrit pour les tableaux ; en salle il a dix secondes par figure.
2. **Aucun axe tronqué.** Le manuscrit en compte deux (parité 0,9–1,01 ; pic de pile), et
   ils exagèrent visuellement des écarts que l'orateur doit ensuite relativiser à l'oral.
   Les grandeurs à faible contraste sont retracées **en écart depuis un vrai zéro**.

Toute valeur tracée provient d'un chargeur de :mod:`src.figures.sources` — aucun littéral
numérique de résultat (garde AST ``test_no_hardcoded_results``). Donnée absente → panneau
explicite, point creux ou mention « N/A », **jamais 0**.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures import sources as src
from src.figures.registry import register_catalog
from src.figures.schematic import arrow, base_fig, box, stage_row
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "soutenance"
CONDITION = "5feat"

# ── Périmètre ────────────────────────────────────────────────────────────────
GRID_MODELS: list[str] = ["ewc", "hdc", "tinyol", "mahalanobis"]
GRID_DATASETS: list[str] = ["monitoring", "cmapss", "pronostia"]
FOCUS_DATASETS: list[str] = ["monitoring", "pronostia"]
PARITY_PROTOCOLS: list[str] = ["frozen", "online"]

#: Libellés **courts** : en projection, « Monitoring » suffit, « Monitoring (D2) » encombre.
MODEL_LABEL: dict[str, str] = {
    "ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "mahalanobis": "Mahalanobis",
}
DATASET_LABEL: dict[str, str] = {
    "monitoring": "Monitoring", "cmapss": "CMAPSS", "pronostia": "Pronostia",
    "paderborn": "Paderborn", "cwru": "CWRU",
}
PROTOCOL_LABEL: dict[str, str] = {"frozen": "régime gelé", "online": "régime en ligne"}

# ── Couleurs ─────────────────────────────────────────────────────────────────
C_PC = STRATEGY_COLORS["fp32"]              # bleu — référence PC
C_BOARD = STRATEGY_COLORS["int8_v2"]        # orange — mesuré sur carte
C_OK = STRATEGY_COLORS["int8_qat"]          # vert — préservé / conforme
C_BAD = STRATEGY_COLORS["int8_ptq_legacy"]  # rouge — effondrement
C_ALT = STRATEGY_COLORS["q15"]              # violet — seconde série

INK = "#333333"
MUTED = "#666666"
FAINT = "#bdbdbd"     # séries mises en retrait
NA_GRAY = "#cccccc"   # non mesuré
TRACK = "#eeeeee"     # fond de piste (budget, enveloppe)

KIB = 1024
BADGE_BOARD = "mesuré sur carte réelle NUCLEO-F439ZI"


# ── Helpers de rendu ─────────────────────────────────────────────────────────

def _empty(msg: str, name: str, out_root: Path) -> Path:
    """Panneau explicite : la figure dit ce qui manque, elle n'invente pas de valeur."""
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, wrap=True)
    return savefig_png(fig, CATALOG, name, out_root)


def _footer(fig: plt.Figure, text: str) -> None:
    """Une seule ligne de pied — en projection, deux lignes ne sont jamais lues."""
    top = 0.93 if fig._suptitle is not None else 0.97
    fig.tight_layout(rect=(0.0, 0.08, 1.0, top))
    fig.text(0.5, 0.02, text, ha="center", fontsize=9, color=MUTED, wrap=True)


def _clean(ax) -> None:
    """Cadre minimal : la donnée porte le message, pas la grille."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(False)


def _bytes(value: float | None) -> str:
    """Octets formatés à la française (espace fine insécable), ou « N/A »."""
    if value is None:
        return "N/A"
    return f"{int(round(value)):,}".replace(",", " ") + " o"


def _pct(value: float | None, total: float | None) -> str:
    """Part d'un budget, ou chaîne vide si l'un des deux termes manque."""
    if value is None or not total:
        return ""
    return f"{100 * value / total:.1f} %".replace(".", ",")


def _int_fr(value: float | None) -> str:
    """Entier à la française — « 7 672 », pas « 7672 » : lisible en projection."""
    if value is None:
        return "N/A"
    return f"{int(round(value)):,}".replace(",", " ")


def _fr(value: float | None, digits: int = 3) -> str:
    """Nombre à la française, ou « N/A » — jamais un 0 de remplacement."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}".replace(".", ",")


def _dumbbell(ax, y: float, left: float | None, right: float | None,
              c_left: str, c_right: str, *, dim: bool = False) -> None:
    """Deux points reliés : l'écart se lit comme une longueur, sans échelle à décoder.

    Une extrémité manquante est tracée en point **creux** gris à l'emplacement de
    l'autre, jamais à 0 — un zéro serait lu comme une mesure.
    """
    alpha = 0.45 if dim else 1.0
    if left is not None and right is not None:
        ax.plot([left, right], [y, y], color=FAINT, lw=2, zorder=1, alpha=alpha)
    for value, color in ((left, c_left), (right, c_right)):
        if value is None:
            continue
        ax.plot(value, y, "o", ms=11, color=color, zorder=3, alpha=alpha,
                markeredgecolor="white", markeredgewidth=1)
    if left is None or right is None:
        anchor = right if left is None else left
        if anchor is not None:
            ax.plot(anchor, y, "o", ms=13, mfc="none", mec=NA_GRAY, mew=2, zorder=2)
            ax.text(anchor, y + 0.28, "N/A", ha="center", fontsize=8, color=MUTED)


def _stacked_row(ax, y: float, segments: list[tuple[float, str, str]],
                 height: float = 0.55) -> float:
    """Barre horizontale empilée ; retourne le total cumulé.

    ``segments`` = liste de ``(largeur, couleur, hachure)``. Les segments dont la
    largeur est inconnue sont simplement absents : la barre est plus courte, ce qui
    est honnête, plutôt que complétée par un bloc de longueur inventée.
    """
    cursor = 0.0
    for width, color, hatch in segments:
        if width is None:
            continue
        ax.barh(y, width, left=cursor, height=height, color=color,
                edgecolor="white", linewidth=1, hatch=hatch or None, zorder=2)
        cursor += width
    return cursor


def _booktabs(ax, headers: list[str], rows: list[list[str]], *,
              bold_cells: set[tuple[int, int]] | None = None,
              col_x: list[float] | None = None) -> None:
    """Tableau au style ``booktabs`` du manuscrit : filets horizontaux, zéro trait vertical.

    Reproduit la grammaire de ``\\toprule`` / ``\\midrule`` / ``\\bottomrule`` (règles
    d'épaisseurs différentes, en-tête en gras, pas de fond alterné) pour que les slides
    de tableau et le rapport se lisent comme un même document. ``ax.table`` ne convient
    pas : il dessine une grille complète et ne sait pas varier l'épaisseur des filets.
    """
    bold_cells = bold_cells or set()
    n_cols = len(headers)
    if col_x is None:
        col_x = [i / n_cols for i in range(n_cols)]

    ax.axis("off")
    ax.set_xlim(0, 1)
    n_rows = len(rows)
    top, bottom = 1.0, 0.0
    row_h = (top - bottom) / (n_rows + 2)
    y_header = top - row_h

    for i, (text, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y_header, text, fontsize=13, fontweight="bold", color=INK,
                va="center", ha="left")

    for r, row in enumerate(rows):
        y = y_header - (r + 1.4) * row_h
        for c, (text, x) in enumerate(zip(row, col_x)):
            weight = "bold" if (r, c) in bold_cells else "normal"
            ax.text(x, y, text, fontsize=12, color=INK, va="center", ha="left",
                    fontweight=weight)

    y_mid = y_header - 0.6 * row_h
    y_bot = y_header - (n_rows + 0.9) * row_h
    ax.plot([0, 1], [top - 0.2 * row_h] * 2, color=INK, lw=2, clip_on=False)   # \toprule
    ax.plot([0, 1], [y_mid] * 2, color=INK, lw=1, clip_on=False)               # \midrule
    ax.plot([0, 1], [y_bot] * 2, color=INK, lw=2, clip_on=False)               # \bottomrule
    ax.set_ylim(y_bot - row_h, top)


# =============================================================================
# BLOC 1 — Ouverture : l'obstacle
# =============================================================================

def _fig_s4_oubli_mesure(out_root: Path) -> Path:
    """Slide 4 — l'oubli est *mécanique* : le F1 de la tâche 0 décroche à la bascule.

    Remplace un schéma dont les accuracies étaient écrites à la main. L'axe des
    époques est ce qui porte le message : on voit la chute se produire à l'instant
    précis où l'optimiseur change de tâche, pas « après la tâche k ».
    """
    name = "s4_oubli_mesure"
    arms = [("naive", "sans EWC (λ = 0)", C_BAD), ("ewc", "avec EWC (λ = 400)", C_PC)]
    loaded = [(label, color, src.s54_forgetting(arm)) for arm, label, color in arms]
    if all(data is None for _, _, data in loaded):
        return _empty("exp_S54_forgetting_* absent — lancer train_ewc_multiclass",
                      name, out_root)

    fig, ax = plt.subplots(figsize=(13, 6))
    n_epochs = None
    for label, color, data in loaded:
        if data is None:
            continue
        matrix = data.get("task_f1_per_epoch") or []
        serie = [src.num(row[0]) if row else None for row in matrix]
        xs = [i for i, v in enumerate(serie) if v is not None]
        ys = [serie[i] for i in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, lw=3, label=label, zorder=3)
        n_epochs = src.num(data.get("n_epochs_per_task")) or n_epochs

    # Frontières de tâche : c'est là que se produit la chute, il faut les voir.
    n_tasks = next((d.get("n_tasks") for _, _, d in loaded if d), None)
    if n_epochs and n_tasks:
        for task in range(1, int(n_tasks)):
            x = task * n_epochs
            ax.axvline(x, color=INK, lw=1, ls="--", alpha=0.6, zorder=1)
            ax.text(x + 1, 0.97, f"bascule sur la tâche {task}", fontsize=12,
                    color=INK, ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2))

    ax.set_xlabel("époques d'entraînement (les tâches s'enchaînent)")
    ax.set_ylabel("F1-macro sur la tâche 0")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=12)
    ax.set_title("Ce que la tâche 0 devient pendant qu'on apprend les suivantes")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, "EWC multiclasse · CWRU, 3 tâches · F1 relevé sur la tâche 0 après chaque époque")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 3 — Méthodologie
# =============================================================================

def _fig_s12_gele_vs_en_ligne(out_root: Path) -> Path:
    """Slide 12 — les deux régimes, et l'attendu de parité de chacun.

    Remplace le schéma de trame UART, qui décrivait le transport et non la
    comparaison. Ici : deux voies appariées, ce qu'on attend de chacune, et le taux
    réellement obtenu — de sorte que la promesse soit immédiatement gagée sur du mesuré.
    """
    name = "s12_gele_vs_en_ligne"
    rates = {
        proto: [src.num((src.s36_node(ds, f"board_{proto}", condition=CONDITION) or {})
                        .get("parity_rate")) for ds in FOCUS_DATASETS]
        for proto in PARITY_PROTOCOLS
    }
    if all(v is None for vals in rates.values() for v in vals):
        return _empty("exp_S36_summary absent", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(5, 5.6, "Mêmes poids · mêmes échantillons, même ordre, même graine",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    lanes = [
        ("frozen", 3.9, C_OK, "poids figés des deux côtés",
         "parité attendue : EXACTE\nun écart = bug de portage"),
        ("online", 1.7, C_BOARD, "les deux côtés apprennent en ligne",
         "parité attendue : APPROCHÉE\ncarte en float32, PC en float64"),
    ]
    for proto, y, color, sub, expect in lanes:
        ax.add_patch(plt.Rectangle((0.3, y - 0.75), 4, 1.5, facecolor=TRACK,
                                   edgecolor=color, lw=2, zorder=1))
        ax.text(2.4, y + 0.3, PROTOCOL_LABEL[proto], ha="center", fontsize=15,
                fontweight="bold", color=color)
        ax.text(2.4, y - 0.32, sub, ha="center", fontsize=11, color=MUTED)
        ax.annotate("", xy=(5.1, y), xytext=(4.4, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
        ax.text(5.3, y + 0.34, expect, ha="left", va="center", fontsize=12, color=INK)
        measured = " · ".join(
            f"{DATASET_LABEL[ds]} {_fr(v, 4)}"
            for ds, v in zip(FOCUS_DATASETS, rates[proto]) if v is not None
        )
        ax.text(5.3, y - 0.52, f"mesuré : {measured}" if measured else "mesuré : N/A",
                ha="left", va="center", fontsize=12, color=color, fontweight="bold")

    ax.text(5, 0.4, "comparaison prédiction à prédiction, échantillon par échantillon",
            ha="center", fontsize=12, color=MUTED, style="italic")
    _footer(fig, BADGE_BOARD)
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 4 — Gap 1 : validation sur données industrielles
# =============================================================================

def _fig_s13_accuracy_trompeuse(out_root: Path) -> Path:
    """Slide 13 — la même cellule vue en accuracy puis en F1 : la chute est le dessin.

    Graphe de pentes plutôt que barres groupées : ce qui compte n'est aucune des deux
    valeurs, c'est l'effondrement de l'une à l'autre. Seuls les couples effondrés
    portent un nom, et il est posé **à droite**, du côté où les valeurs se séparent —
    à gauche, toutes les accuracies sont voisines et les étiquettes se chevauchent.
    """
    name = "s13_accuracy_trompeuse"
    couples: list[tuple[str, float, float]] = []
    for model in GRID_MODELS:
        for dataset in GRID_DATASETS:
            acc = src.s35_acc(model, dataset, "board", condition=CONDITION)
            f1 = src.s35_f1(model, dataset, "board", condition=CONDITION)
            if acc is None or f1 is None:
                continue
            couples.append((f"{MODEL_LABEL[model]} × {DATASET_LABEL[dataset]}", acc, f1))
    if not couples:
        return _empty("grille S35 carte absente", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 7))
    collapsed = sorted((c for c in couples if c[2] < c[1] / 2), key=lambda c: c[2])
    for label, acc, f1 in couples:
        is_bad = (label, acc, f1) in collapsed
        color = C_BAD if is_bad else FAINT
        ax.plot([0, 1], [acc, f1], color=color, lw=3 if is_bad else 1,
                zorder=3 if is_bad else 2)
        ax.plot([0, 1], [acc, f1], "o", color=color, ms=9 if is_bad else 5, zorder=3)

    # Plusieurs couples s'effondrent à la même valeur (trois F1 nuls) : les étiquettes
    # sont écartées **vers le haut** en partant du bas, ce qui les garde toutes dans
    # les axes — les empiler vers le bas les ferait sortir sous l'axe des abscisses.
    gap, previous = 0.055, None
    for label, acc, f1 in collapsed:
        y = max(f1, 0.03) if previous is None else max(f1, previous + gap)
        previous = y
        ax.plot([1, 1.06], [f1, y], color=C_BAD, lw=1, alpha=0.5, zorder=2)
        ax.text(1.08, y, f"{label} — {_fr(f1, 3)}", ha="left", va="center",
                fontsize=13, color=C_BAD, fontweight="bold")

    ax.set_xlim(-0.12, 1.75)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["accuracy", "F1 « fautif »"], fontsize=16, fontweight="bold")
    ax.set_ylabel("score")
    ax.set_title("Le même modèle, mesuré deux fois")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, f"{BADGE_BOARD} · en rouge, les couples dont le F1 tombe sous la moitié "
                 "de leur accuracy")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s14_oubli_bilan(out_root: Path) -> Path:
    """Slide 14 — la chute annotée, et la carte qui retombe exactement dessus.

    Deux stations et une flèche, au lieu de quatre barres de même largeur qui donnaient
    le même poids visuel à des grandeurs non commensurables. Le marqueur creux « carte »
    superposé au marqueur « modèle final » *est* l'argument de parité.
    """
    name = "s14_oubli_bilan"
    data = src.s26_multiclass()
    if data is None:
        return _empty("exp_S26_02 absent", name, out_root)
    peak = src.num(data.get("f1_macro_pc_per_task_mean"))
    final = src.num(data.get("f1_macro_pc_final_all_tasks"))
    board = src.num(data.get("f1_macro_board_inference"))
    af = src.num(data.get("avg_forgetting_f1_pc"))
    if peak is None or final is None:
        return _empty("F1 d'oubli absents de exp_S26_02", name, out_root)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = 0.36
    ax.plot(x, peak, "o", ms=20, color=C_OK, zorder=3)
    ax.plot(x, final, "o", ms=20, color=C_BAD, zorder=3)
    ax.annotate("", xy=(x, final), xytext=(x, peak),
                arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=4))
    if af is not None:
        ax.text(x - 0.05, (peak + final) / 2, f"oubli moyen\nAF = {_fr(af, 3)}",
                ha="right", va="center", fontsize=16, color=C_BAD, fontweight="bold")

    ax.text(x + 0.06, peak, f"moyenne des F1 relevés\naprès chaque tâche — {_fr(peak, 3)}",
            ha="left", va="center", fontsize=13, color=INK)
    ax.text(x + 0.06, final, f"modèle final, toutes tâches — {_fr(final, 3)}",
            ha="left", va="center", fontsize=13, color=INK)

    if board is not None:
        # Anneau tracé AUTOUR du point PC final : la superposition est l'argument de
        # parité. Le décaler sur le côté suggérerait deux mesures distinctes.
        ax.plot(x, board, "o", ms=34, mfc="none", mec=C_BOARD, mew=3, zorder=4)
        ax.text(x - 0.05, board - 0.09, f"carte — {_fr(board, 3)}", ha="right", va="top",
                fontsize=14, color=C_BOARD, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)   # l'étiquette du point haut déborderait sur le titre à 1.0
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([])
    ax.set_ylabel("F1-macro")
    ax.set_title("Le même modèle, avant et après avoir appris la suite")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, "EWC multiclasse · CWRU, 3 tâches · la carte retombe sur la valeur PC : parité exacte")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s15_grille_classement(out_root: Path) -> Path:
    """Slide 15 — la grille en haltères : niveau *et* parité d'un seul regard.

    Remplace la heatmap et sa barre de couleur : douze lignes, deux points chacune.
    Le classement se lit sans décoder une échelle, et l'écart PC↔carte est la longueur
    du connecteur. Valeurs affichées seulement sur les lignes EWC — celles que
    l'orateur prononce.
    """
    name = "s15_grille_classement"
    rows: list[tuple[str, str, float | None, float | None]] = []
    for dataset in GRID_DATASETS:
        for model in GRID_MODELS:
            rows.append((dataset, model,
                         src.s35_f1(model, dataset, "PC", condition=CONDITION),
                         src.s35_f1(model, dataset, "board", condition=CONDITION)))
    if all(pc is None and bd is None for _, _, pc, bd in rows):
        return _empty("grille S35 absente", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 8))
    labels: list[str] = []
    for i, (dataset, model, pc, bd) in enumerate(rows):
        y = len(rows) - 1 - i
        is_ewc = model == "ewc"
        if i % len(GRID_MODELS) == 0:
            ax.axhspan(y + 0.5, y - len(GRID_MODELS) + 0.5,
                       color=TRACK if (i // len(GRID_MODELS)) % 2 == 0 else "white", zorder=0)
        _dumbbell(ax, y, pc, bd, C_PC, C_BOARD, dim=not is_ewc)
        labels.append(f"{DATASET_LABEL[dataset]} · {MODEL_LABEL[model]}")
        if is_ewc and pc is not None and bd is not None:
            ax.text(max(pc, bd) + 0.03, y, f"{_fr(pc, 2)} / {_fr(bd, 2)}",
                    va="center", fontsize=11, color=INK, fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=12)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("F1 « fautif »")
    ax.plot([], [], "o", color=C_PC, ms=10, label="PC")
    ax.plot([], [], "o", color=C_BOARD, ms=10, label="carte")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_title("Quatre familles de modèles sur trois jeux industriels")
    _clean(ax)
    _footer(fig, "deux protocoles distincts — ce graphe établit un classement, pas une comparaison au centième")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s16_paderborn(out_root: Path) -> Path:
    """Slide 16 — même grammaire que la grille, réduite à un seul jeu et une seule couleur."""
    name = "s16_paderborn_ewc_seul"
    dataset = "paderborn"
    rows = [(m, src.s35_f1(m, dataset, "PC", condition=CONDITION),
             src.s35_f1(m, dataset, "board", condition=CONDITION)) for m in GRID_MODELS]
    if all(pc is None and bd is None for _, pc, bd in rows):
        return _empty("cellules Paderborn absentes", name, out_root)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, pc, bd) in enumerate(rows):
        y = len(rows) - 1 - i
        is_ewc = model == "ewc"
        _dumbbell(ax, y, pc, bd,
                  C_PC if is_ewc else FAINT, C_BOARD if is_ewc else FAINT, dim=not is_ewc)
        if is_ewc and pc is not None and bd is not None:
            # PC et carte sont ici confondus : sans un anneau, un seul point serait
            # visible et on croirait une mesure manquante.
            ax.plot(pc, y, "o", ms=24, mfc="none", mec=C_PC, mew=2, zorder=4)
            ax.text(max(pc, bd), y - 0.42, f"{_fr(pc, 3)} sur PC comme sur carte",
                    ha="center", va="top", fontsize=14, color=INK, fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MODEL_LABEL[m] for m, _, _ in rows][::-1], fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xlabel("F1 « fautif »")
    ax.set_title("Paderborn — une seule famille tient le régime")
    _clean(ax)
    _footer(fig, "class-incremental, une classe par tâche · les autres modèles prédisent la classe majoritaire")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 5 — Gap 2 : la contrainte mémoire et temporelle
# =============================================================================

def _fig_s17_ram_trois_niveaux(out_root: Path) -> Path:
    """Slide 17 — le budget devient l'axe, pas une ligne pointillée à interpréter.

    Chart « bullet » : une piste = la carte entière. Le niveau revendiqué est la seule
    barre pleine ; plancher et pire cas sont des repères. La version manuscrit donnait
    trois barres de largeur égale, ce qui promouvait le plancher au rang du chiffre
    du travail.
    """
    name = "s17_ram_trois_niveaux"
    s48 = src.s48_summary()
    minimal = src.try_load("experiments/exp_S18_01_board/results.json")
    sweep = src.s35_sweep()

    budget = src.num((s48 or {}).get("ram_budget_bytes"))
    default = src.num((s48 or {}).get("bss_default_invariant"))
    floor = src.num((minimal or {}).get("ram_peak_bytes"))
    worst, worst_k = None, None
    for cell in sweep or []:
        value = src.num(cell.get("bss_bytes"))
        if value is not None and (worst is None or value > worst):
            worst, worst_k = value, cell.get("n_features")
    if budget is None or default is None:
        return _empty("exp_S48_summary absent (budget / .bss par défaut)", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.barh(0, budget, height=0.5, color=TRACK, zorder=1)
    ax.barh(0, default, height=0.28, color=C_PC, zorder=3)
    ax.text(default, 0.42, f"{_bytes(default)} — {_pct(default, budget)} du budget",
            ha="center", fontsize=15, color=C_PC, fontweight="bold")

    for value, label in ((floor, "plancher"),
                         (worst, f"pire cas mesuré (k = {worst_k})" if worst_k else "pire cas mesuré")):
        if value is None:
            continue
        ax.plot([value, value], [-0.3, 0.3], color=INK, lw=2, zorder=4)
        ax.text(value, -0.42, f"{label}\n{_bytes(value)}", ha="center", va="top",
                fontsize=11, color=MUTED)

    ax.text(budget, 0.42, f"budget carte {_bytes(budget)}", ha="right", fontsize=12, color=MUTED)
    ax.set_xlim(0, budget * 1.02)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Tout tient dans l'enveloppe de la carte")
    _clean(ax)
    ax.spines["bottom"].set_visible(False)
    _footer(fig, f"{BADGE_BOARD} · `.bss` seul — la pile est comptée à part")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s18_ram_totale_cascade(out_root: Path) -> Path:
    """Slide 18 — la formule du titre devient l'axe de lecture.

    Cascade : `.data` → `+ .bss` → `+ pic de pile` → total. Un repère au niveau
    `.bss` seul matérialise l'ancienne mesure, et l'écart est le seul incrément annoté.
    """
    name = "s18_ram_totale_cascade"
    dataset = FOCUS_DATASETS[0]
    cell = src.s49_board("ewc", dataset, "fp32", condition=CONDITION)
    if cell is None:
        return _empty("cellule RAM S49 absente", name, out_root)
    data = src.num(cell.get("data"))
    bss = src.num(cell.get("bss"))
    stack = src.num(cell.get("stack_peak_update"))
    total = src.num(cell.get("total"))
    budget = src.num((src.s48_summary() or {}).get("ram_budget_bytes"))
    if None in (data, bss, stack, total):
        return _empty("composantes RAM incomplètes", name, out_root)

    steps = [("`.data`", data, C_ALT), ("`.bss`", bss, C_PC), ("pic de pile", stack, C_BOARD)]
    fig, ax = plt.subplots(figsize=(13, 6))
    cursor = 0.0
    for i, (label, value, color) in enumerate(steps):
        ax.bar(i, value / KIB, bottom=cursor / KIB, color=color, width=0.55,
               edgecolor="white", linewidth=1, zorder=3)
        cursor += value
        if label == "pic de pile":
            # Annoté au-dessus de sa propre marche : à droite, le texte chevauchait
            # la barre du total et la légende du repère `.bss`.
            ax.text(i, cursor / KIB + 1, f"+ {_bytes(value)}", ha="center", va="bottom",
                    fontsize=15, color=C_BOARD, fontweight="bold")
    ax.bar(len(steps), total / KIB, color=INK, width=0.55, zorder=3)
    ax.axhline((data + bss) / KIB, color=MUTED, ls="--", lw=1, zorder=1)
    ax.text(-0.3, (data + bss) / KIB - 2, "ce que `.bss` seul laissait croire",
            fontsize=12, color=MUTED, va="top", ha="left")

    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels([s[0] for s in steps] + ["RAM totale"], fontsize=14)
    ax.set_ylim(0, total / KIB * 1.15)
    ax.set_ylabel("Kio")
    part = f" · {_pct(total, budget)} du budget carte" if budget else ""
    ax.set_title(f"RAM totale mesurée : {_bytes(total)}{part}")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, f"{BADGE_BOARD} · EWC × {DATASET_LABEL[dataset]} · pic relevé pendant la mise à jour")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s19_pile_par_phase(out_root: Path) -> Path:
    """Slide 19 — le creusement, mesuré depuis la base au repos, à vrai zéro.

    La version manuscrit trace deux courbes quasi confondues (4 416 vs 4 688 sur un axe
    partant de ~4 200) : visuellement, il ne se passe rien. En traçant l'écart au repos
    depuis 0, le rapport entre les deux phases devient la longueur des barres.
    """
    name = "s19_pile_par_phase"
    fig, ax = plt.subplots(figsize=(12, 5))
    drawn, idles = 0, []
    for i, dataset in enumerate(FOCUS_DATASETS):
        cell = src.s49_board("ewc", dataset, "fp32", condition=CONDITION)
        if cell is None:
            continue
        history = {h.get("phase"): src.num(h.get("stack_peak_bytes"))
                   for h in cell.get("stack_history", [])}
        base = history.get("idle")
        if base is None:
            continue
        idles.append(base)
        for k, (phase, label, color) in enumerate(
            [("inference", "inférence", C_PC), ("update", "mise à jour CL", C_BOARD)]
        ):
            peak = history.get(phase)
            if peak is None:
                continue
            y = i * 3 + (1 - k)
            ax.barh(y, peak - base, height=0.6, color=color, zorder=3)
            ax.text(peak - base + 12, y, f"+ {_bytes(peak - base)}", va="center",
                    fontsize=13, color=color, fontweight="bold")
            ax.text(-14, y, f"{DATASET_LABEL[dataset]} · {label}", ha="right",
                    va="center", fontsize=12, color=INK)
            drawn += 1
    if not drawn:
        return _empty("historique de pile S49 absent", name, out_root)

    ax.set_yticks([])
    ax.set_xlabel("pile creusée au-delà du repos (octets)")
    ax.set_xlim(0, None)
    ax.set_title("La mise à jour creuse la pile plus que l'inférence")
    _clean(ax)
    base_txt = " · ".join(f"{DATASET_LABEL[d]} {_bytes(b)}"
                          for d, b in zip(FOCUS_DATASETS, idles))
    _footer(fig, f"{BADGE_BOARD} · base au repos : {base_txt}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s20_latence_surcout(out_root: Path) -> Path:
    """Slide 20 — le surcoût devient l'objet visuel dominant.

    Barre empilée « inférence + mise à jour », prise dans le **même** nœud
    ``board_online`` : les deux composantes somment donc exactement à leur total, ce
    que la version manuscrit ne garantissait pas (elle prenait l'inférence dans le run
    gelé et la mise à jour dans le run en ligne).
    """
    name = "s20_latence_surcout"
    fig, ax = plt.subplots(figsize=(13, 5))
    drawn = 0
    for i, dataset in enumerate(FOCUS_DATASETS):
        node = src.s36_node(dataset, "board_online", condition=CONDITION) or {}
        inference = src.num(node.get("latency_inference_only_us_p50"))
        overhead = src.num(node.get("latency_update_overhead_us_p50"))
        if inference is None and overhead is None:
            continue
        y = len(FOCUS_DATASETS) - 1 - i
        total = _stacked_row(ax, y, [(inference, C_PC, ""), (overhead, C_BOARD, "//")])
        if inference:
            ax.text(inference / 2, y, "inférence", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")
        if overhead:
            ax.text(inference + overhead / 2, y, "mise à jour CL", ha="center",
                    va="center", fontsize=13, color=INK, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=3, alpha=0.85))
            ax.text(total + 8, y, f"{_int_fr(total)} µs   ×{_fr(total / inference, 1)}",
                    va="center", fontsize=14, color=INK, fontweight="bold")
        drawn += 1
    if not drawn:
        return _empty("latences S36 absentes", name, out_root)

    budget = src.num((src.s48_summary() or {}).get("gap2_latency_us_threshold"))
    ax.set_yticks(range(len(FOCUS_DATASETS)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in FOCUS_DATASETS][::-1], fontsize=14)
    ax.set_xlabel("latence P50 (µs)")
    ax.set_title("Apprendre en ligne coûte cinq fois une inférence")
    _clean(ax)
    margin = ""
    if budget:
        margin = f" · budget Gap 2 = {_fr(budget / 1000, 0)} ms, soit plus de deux ordres de grandeur de marge"
    _footer(fig, f"{BADGE_BOARD}{margin}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s21_parite_desaccords(out_root: Path) -> Path:
    """Slide 21 — chaque désaccord tracé à sa position réelle dans le flux.

    C'est la refonte la plus importante des douze. La version manuscrit borne l'axe à
    0,9–1,01, ce qui **exagère** visuellement un écart de quelques pour mille et oblige
    l'orateur à prévenir le jury. Ici, une bande par cellule à l'échelle réelle de
    l'échantillon : les bandes du régime gelé sont vierges, et c'est l'argument.
    """
    name = "s21_parite_desaccords"
    cells = []
    for protocol in PARITY_PROTOCOLS:
        for dataset in FOCUS_DATASETS:
            data = src.s36_parity(dataset, protocol, condition=CONDITION)
            if data:
                cells.append((protocol, dataset, data))
    if not cells:
        return _empty("fichiers de parité S36 absents", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (protocol, dataset, data) in enumerate(cells):
        y = len(cells) - 1 - i
        n = src.num(data.get("n_compared"))
        mismatches = src.num(data.get("mismatch_count"))
        if n is None:
            continue
        ax.barh(y, n, height=0.55, color=TRACK, zorder=1)
        bad = [r.get("idx") for r in data.get("rows", []) if r.get("match") is False]
        for idx in bad:
            ax.plot([idx, idx], [y - 0.28, y + 0.28], color=C_BAD, lw=1, alpha=0.7, zorder=3)
        ax.text(-n * 0.015, y, f"{DATASET_LABEL[dataset]}\n{PROTOCOL_LABEL[protocol]}",
                ha="right", va="center", fontsize=12, color=INK)
        count = "0 désaccord" if mismatches == 0 else f"{_int_fr(mismatches)} désaccords"
        ax.text(n * 1.02, y, f"{count} / {_int_fr(n)} comparaisons", va="center",
                fontsize=13, color=C_OK if mismatches == 0 else C_BAD,
                fontweight="bold")

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, max(src.num(d.get("n_compared")) or 0 for _, _, d in cells) * 1.42)
    ax.set_title("Chaque trait rouge est un échantillon où la carte et le PC divergent")
    _clean(ax)
    ax.spines["bottom"].set_visible(False)
    _footer(fig, f"{BADGE_BOARD} · chaque bande couvre la totalité du flux, à l'échelle")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 6 — Gap 3 : quantification
# =============================================================================

def _fig_s22_ram_deux_echelles(out_root: Path) -> Path:
    """Slide 22 — un seul axe partagé : « la bonne échelle » devient démontrée.

    La version manuscrit met les poids et le système dans deux panneaux à échelles
    indépendantes — ce découpage est précisément ce qui rend le ÷4 faussement
    rassurant. Sur un axe commun, le ÷4 des poids et l'invariance du système se
    voient ensemble.
    """
    name = "s22_ram_deux_echelles"
    dataset = FOCUS_DATASETS[0]
    v2 = src.s40_board_v2(dataset) or {}
    w_fp32 = src.num(v2.get("ram_weights_fp32_bytes"))
    w_int8 = src.num(v2.get("ram_weights_quant_bytes"))
    sys_fp32 = src.num((src.s49_board("ewc", dataset, "fp32", condition=CONDITION) or {}).get("total"))
    sys_int8 = src.num((src.s49_board("ewc", dataset, "int8", condition=CONDITION) or {}).get("total"))
    if w_fp32 is None or sys_fp32 is None:
        return _empty("sources RAM poids / système absentes", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 5))
    rows = [("poids du modèle", w_fp32, w_int8, 1), ("RAM système", sys_fp32, sys_int8, 0)]
    for label, fp32, int8, y in rows:
        _dumbbell(ax, y, fp32, int8, C_PC, C_BOARD)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=14,
                color=INK, transform=ax.get_yaxis_transform())
        if fp32 and int8:
            ratio = fp32 / int8
            mid = (fp32 * int8) ** 0.5
            if ratio > 1.5:
                ax.text(mid, y + 0.22, f"÷{_fr(ratio, 0)}", ha="center", fontsize=16,
                        color=C_BOARD, fontweight="bold")
                ax.text(mid, y - 0.26, f"{_bytes(fp32)} → {_bytes(int8)}", ha="center",
                        fontsize=12, color=MUTED)
            else:
                # Les deux points sont confondus : sans les deux valeurs, on croirait
                # qu'une seule mesure a été tracée.
                ax.text(fp32, y + 0.24, "inchangé", ha="center", fontsize=16,
                        color=MUTED, fontweight="bold")
                ax.text(fp32, y - 0.26, f"{_bytes(fp32)} → {_bytes(int8)}", ha="center",
                        fontsize=12, color=MUTED)

    ax.set_xscale("log")
    ax.set_xlim(w_int8 / 3 if w_int8 else None, (sys_fp32 or 1) * 6)
    ax.set_ylim(-1, 2)
    ax.set_yticks([])
    ax.set_xlabel("octets (échelle logarithmique, commune aux deux lignes)")
    ax.plot([], [], "o", color=C_PC, ms=10, label="FP32")
    ax.plot([], [], "o", color=C_BOARD, ms=10, label="INT8")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_title("L'INT8 divise les poids par quatre — et ne bouge pas la RAM système")
    _clean(ax)
    _footer(fig, f"{BADGE_BOARD} · EWC × {DATASET_LABEL[dataset]}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s23_effondrement_recuperation(out_root: Path) -> Path:
    """Slide 23 — effondrement puis récupération en un seul geste.

    Trois stations dans l'ordre du récit de l'orateur, une ligne par jeu : la courbe
    en V remplace six barres groupées que le jury devait rapprocher mentalement.
    """
    name = "s23_effondrement_recuperation"
    stations = ["FP32\n(référence)", "INT8 PTQ naïve\n(embarquée)", "INT8 v2\n(échelle calibrée)"]
    fig, ax = plt.subplots(figsize=(12, 6))
    drawn, parity_class = 0, None
    for dataset, color in zip(FOCUS_DATASETS, (C_PC, C_ALT)):
        fp32 = src.num((src.s36_node(dataset, "board_frozen", condition=CONDITION) or {}).get("f1_faulty"))
        naive = src.num((src.s36_node(dataset, "board_frozen_int8", condition=CONDITION) or {}).get("f1_faulty"))
        v2 = src.s40_board_v2(dataset) or {}
        calib = src.num(v2.get("f1_faulty"))
        parity_class = v2.get("parity_class") or parity_class
        values = [fp32, naive, calib]
        xs = [i for i, v in enumerate(values) if v is not None]
        if len(xs) < 2:
            continue
        ax.plot(xs, [values[i] for i in xs], "-o", color=color, lw=3, ms=12,
                label=DATASET_LABEL[dataset], zorder=3)
        if fp32 is not None:
            ax.axhline(fp32, color=color, ls=":", lw=1, alpha=0.5, zorder=1)
        drawn += 1
    if not drawn:
        return _empty("sources INT8 (S36 / S40) absentes", name, out_root)

    ax.annotate("effondrement", xy=(1, 0.12), xytext=(1, 0.35), ha="center",
                fontsize=15, color=C_BAD, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=2))
    recovered = "récupéré sur la carte"
    if parity_class:
        recovered += "\nparité exacte vs émulateur"
    ax.text(2, 0.62, recovered, ha="center", fontsize=13, color=C_OK, fontweight="bold")

    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(stations, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 « fautif »")
    ax.legend(loc="center left", fontsize=12)
    ax.set_title("Ce n'est pas la quantification qui échouait, c'est l'échelle")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, BADGE_BOARD)
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s24_paradoxe_latence(out_root: Path) -> Path:
    """Slide 24 — deux barres sur un axe µs commun, l'INT8 décomposée par poste.

    Ce que les deux panneaux du manuscrit obligeaient à reconstituer mentalement
    devient visible : le MAC entier seul tient sous le total FP32 ; c'est la
    requantification qui fait basculer. Le reliquat (total − Σ segments) est tracé et
    nommé, jamais absorbé dans un segment.
    """
    name = "s24_paradoxe_latence"
    data = src.s50_int8_latency()
    if data is None:
        return _empty("exp_S50_int8_latency absent", name, out_root)
    cpu_hz = src.num(data.get("cpu_hz"))
    dataset = FOCUS_DATASETS[0]
    total_int8 = src.num((data.get("total_int8_us_p50_by_dataset") or {}).get(dataset))
    total_fp32 = src.num((data.get("total_fp32_us_p50_by_dataset") or {}).get(dataset))
    if not cpu_hz or total_int8 is None or total_fp32 is None:
        return _empty("totaux / fréquence absents de exp_S50", name, out_root)

    order = [("dequant", "déquantification", C_OK),
             ("mac", "MAC entier", C_PC),
             ("requant", "requantification", C_BAD)]
    segs, cycles = [], {}
    for key, _, color in order:
        cyc = src.num(((data.get("segments") or {}).get(key, {})
                       .get("by_dataset_cycles_p50") or {}).get(dataset))
        cycles[key] = cyc
        segs.append((None if cyc is None else cyc / cpu_hz * 1_000_000, color, ""))
    accounted = sum(w for w, _, _ in segs if w is not None)
    segs.append((max(total_int8 - accounted, 0), NA_GRAY, "//"))

    fig, ax = plt.subplots(figsize=(13, 5))
    _stacked_row(ax, 1, [(total_fp32, C_PC, "")], height=0.5)
    ax.text(total_fp32 + 2, 1, f"FP32 — {_int_fr(total_fp32)} µs", va="center",
            fontsize=14, color=INK, fontweight="bold")
    _stacked_row(ax, 0, segs, height=0.5)
    ax.text(total_int8 + 2, 0, f"INT8 — {_int_fr(total_int8)} µs", va="center",
            fontsize=14, color=INK, fontweight="bold")

    cursor = 0.0
    for (key, label, color), (width, _, _) in zip(order, segs):
        if width is None:
            continue
        if key != "dequant":
            ax.text(cursor + width / 2, -0.42,
                    f"{label}\n{_int_fr(cycles[key])} cyc", ha="center", va="top",
                    fontsize=11, color=color, fontweight="bold")
        cursor += width
    if total_int8 - accounted > 0:
        ax.text(cursor + (total_int8 - accounted) / 2, -0.42, "reste\ndu chemin",
                ha="center", va="top", fontsize=11, color=MUTED)

    ax.annotate("", xy=(total_int8, 0.62), xytext=(total_fp32, 0.62),
                arrowprops=dict(arrowstyle="<|-|>", color=C_BAD, lw=2))
    ax.text((total_fp32 + total_int8) / 2, 0.75, f"+ {_int_fr(total_int8 - total_fp32)} µs",
            ha="center", fontsize=15, color=C_BAD, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("latence P50 (µs)")
    ax.set_ylim(-1.5, 1.6)
    ax.set_title("Quantifier n'accélère pas : ce sont les conversions qui coûtent")
    _clean(ax)
    _footer(fig, f"{BADGE_BOARD} · EWC × {DATASET_LABEL[dataset]} · déquantification trop courte pour être étiquetée")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# Figures de secours
# =============================================================================

def _fig_s25_gate_economie(out_root: Path) -> Path:
    """Slide 25 — ce que le gate économise, et ce qu'il coûte en F1.

    Chaque politique est un point : taux de mise à jour en abscisse, F1 en ordonnée.
    Le gate se lit comme le point qui reste **haut** tout en étant **très à gauche** —
    la lecture « économie sans perte » est une position, pas deux barres à rapprocher.
    """
    name = "s25_gate_economie"
    summary = src.try_load("experiments/exp_S38_summary.json")
    if summary is None:
        return _empty("exp_S38_summary absent", name, out_root)
    results = summary.get("results", {})
    init = "pretrained"   # le régime recommandé ; « scratch » est traité en backup
    policy_style = {
        "frozen": ("gelée", C_PC, "o"),
        "always": ("systématique", C_BAD, "s"),
        "gated_truelabel": ("gate + étiquette vraie", C_OK, "D"),
        "gated_pseudolabel": ("gate + pseudo-étiquette", C_ALT, "^"),
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = False
    for dataset, alpha in zip(summary.get("datasets", []), (1.0, 0.45)):
        for policy, (label, color, marker) in policy_style.items():
            cell = results.get(dataset, {}).get(init, {}).get(policy, {}).get("board", {})
            rate = src.num(cell.get("update_rate"))
            f1 = src.num(cell.get("f1_faulty"))
            if rate is None or f1 is None:
                continue
            ax.plot(rate, f1, marker, ms=17, color=color, alpha=alpha, zorder=3,
                    markeredgecolor="white", markeredgewidth=1,
                    label=label if alpha == 1.0 else None)
            plotted = True
        ax.text(1.02, 0, "", transform=ax.transAxes)
    if not plotted:
        return _empty("cellules S38 absentes", name, out_root)

    ax.set_xlabel("part des échantillons déclenchant une mise à jour")
    ax.set_ylabel("F1 « fautif »")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12, loc="lower center", ncol=2)
    ax.set_title("Mettre à jour 2,5 % du temps suffit")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    datasets = " et ".join(DATASET_LABEL.get(d, d) for d in summary.get("datasets", []))
    _footer(fig, f"{BADGE_BOARD} · {datasets} (le second en transparence) · init pré-entraînée")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_perspectives_trois_axes(out_root: Path) -> Path:
    """Slide 15 — les trois prolongements, et **où chacun s'arrête**.

    Le piège d'une slide de perspectives est de présenter trois souhaits sur le même
    plan : le jury ne peut alors pas distinguer ce qui est déjà mesuré de ce qui est
    une intention. Ce schéma encode l'écart. Chaque axe est une rangée de quatre
    jalons — *mesuré sur carte*, *verrou identifié*, *verrou levé*, *déployable* —
    pleins jusqu'où le travail est allé, creux ensuite. La marche entre le plein et
    le creux **est** le message : deux axes sont instrumentés et butent sur un verrou
    nommé, le troisième n'a pas commencé à être mesuré.

    Les jalons sont un statut de projet, pas une grandeur mesurée : ils sont donc
    tracés en **discret** (franchi / non franchi), jamais en fraction continue, qui
    inventerait une progression. Toute valeur chiffrée de la colonne de droite vient
    d'un chargeur de :mod:`src.figures.sources`.
    """
    name = "perspectives_trois_axes"
    gate = src.try_load("experiments/exp_S38_summary.json")
    sweep = src.s35_sweep()
    energy = src.try_load("experiments/exp_S33_energy/summary.json")
    if gate is None or sweep is None:
        return _empty("exp_S38_summary ou grille S35 absents", name, out_root)

    # ── Axe 1 : décision de mise à jour autonome ─────────────────────────────
    # Économie et verrou lus sur le jeu le plus sévère (class-incremental).
    gate_ds = "pronostia" if "pronostia" in gate.get("datasets", []) else None
    gate_cells = gate.get("results", {}).get(gate_ds, {}).get("pretrained", {}) if gate_ds else {}

    def _gate(policy: str, key: str) -> float | None:
        return src.num(gate_cells.get(policy, {}).get("board", {}).get(key))

    rate_gated = _gate("gated_truelabel", "update_rate")
    rate_always = _gate("always", "update_rate")
    saved = (1 - rate_gated / rate_always) if (rate_gated and rate_always) else None
    verdict = _gate("gated_truelabel", "verdict_parity_rate")
    f1_true = _gate("gated_truelabel", "f1_faulty")
    f1_pseudo = _gate("gated_pseudolabel", "f1_faulty")

    # ── Axe 2 : sélection des variables ──────────────────────────────────────
    feat = {
        r["condition"]: r
        for r in sweep
        if r["model"] == "ewc" and r["dataset"] == "cmapss"
        and r["platform"] == "nucleo_f439zi" and r["condition"] in ("5feat", "all")
    }
    f1_min = src.num(feat.get("5feat", {}).get("f1_faulty"))
    f1_max = src.num(feat.get("all", {}).get("f1_faulty"))
    bss_min = src.num(feat.get("5feat", {}).get("bss_bytes"))
    bss_max = src.num(feat.get("all", {}).get("bss_bytes"))

    # ── Axe 3 : énergie ──────────────────────────────────────────────────────
    # Le producteur écrit littéralement « à mesurer » : src.num renvoie None, et la
    # rangée reste grise. Aucune valeur de repli n'est fabriquée.
    uj = src.num((energy or {}).get("per_model", {}).get("ewc", {}).get("fp32"))

    #: (titre, jalons franchis, acquis mesuré, verrou) — `None` en acquis ⇒ rangée grise.
    axes_rows = [
        (
            "Décider seul\nquand se mettre à jour",
            2,
            # Tronqué à l'entier inférieur — c'est l'arrondi du manuscrit et de `S01`
            # (« 97 % »), et il minore le bénéfice plutôt que de le flatter. Afficher
            # 97,5 % ferait diverger la figure projetée du chiffre prononcé.
            f"{int(100 * saved)} % des mises à jour économisées"
            if saved is not None else "N/A",
            f"parité de verdict {_fr(verdict)} · +{_int_fr(_gate('gated_truelabel', 'bss_delta_vs_default'))} o",
            f"en auto-étiquetage, F1 {_fr(f1_pseudo, 3)}\nau lieu de {_fr(f1_true, 3)}",
        ),
        (
            "Choisir seul\nles variables d'entrée",
            2,
            f"F1 {_fr(f1_min, 3)} → {_fr(f1_max, 3)}",
            f"mais {_bytes(bss_min)} → {_bytes(bss_max)}",
            "arbitrage encore manuel,\ncouple par couple",
        ),
        (
            "Mesurer\nl'énergie",
            0 if uj is None else 1,
            None,
            "chaîne d'instrumentation prête",
            "sonde non posée :\naucun chiffre publié",
        ),
    ]

    milestones = ["mesuré\nsur carte", "verrou\nidentifié", "verrou\nlevé", "déployable"]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    # Recadré sur le contenu : les rangées occupent [0,17 ; 0,93], et une bande vide
    # sous la dernière rangée coûterait de la hauteur de projection pour rien.
    ax.set_ylim(0.13, 1.0)

    # Les jalons sont tracés en **marqueurs** (taille en points) et non en patches
    # `Circle` : en coordonnées données, un cercle d'une figure non carrée sort en
    # ellipse. `pad` est le demi-espacement en x qui dégage les flèches des jalons.
    x0, step, pad = 0.26, 0.1, 0.032
    ys = [0.76, 0.5, 0.24]
    xs = [x0 + i * step for i in range(len(milestones))]

    for x, label in zip(xs, milestones):
        ax.text(x, 0.93, label, ha="center", va="center", fontsize=10, color=MUTED)

    for y, (title, reached, gain, detail, lock) in zip(ys, axes_rows):
        grey = gain is None
        color = NA_GRAY if grey else C_BOARD
        ax.text(0.02, y, title, ha="left", va="center", fontsize=13, color=INK)

        ax.plot([xs[0], xs[-1]], [y, y], color=TRACK, lw=5, zorder=1,
                solid_capstyle="round")
        if reached > 1:
            ax.plot([xs[0], xs[reached - 1]], [y, y], color=color, lw=5, zorder=2,
                    solid_capstyle="round")
        for i, x in enumerate(xs):
            done = i < reached
            ax.plot(x, y, "o", ms=22, zorder=3,
                    color=color if done else "white",
                    markeredgecolor=color if done else FAINT, markeredgewidth=2)

        # La marche : ce qui reste à faire commence là où le plein s'arrête.
        if 0 < reached < len(xs):
            ax.annotate("", xy=(xs[reached] - pad, y), xytext=(xs[reached - 1] + pad, y),
                        arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.8,
                                        linestyle="--"))

        tx = xs[-1] + 0.055
        if grey:
            ax.text(tx, y, f"{detail}\n{lock}", ha="left", va="center",
                    fontsize=11, color=MUTED, style="italic")
        else:
            ax.text(tx, y + 0.06, gain, ha="left", va="center", fontsize=12,
                    color=INK, weight="bold")
            ax.text(tx, y + 0.005, detail, ha="left", va="center", fontsize=10,
                    color=MUTED)
            ax.text(tx, y - 0.065, lock, ha="left", va="center", fontsize=11,
                    color=C_BAD)

    ax.set_title("Deux axes butent sur un verrou nommé, le troisième n'est pas mesuré",
                 pad=18)
    _footer(fig, f"{BADGE_BOARD} pour les deux premiers axes · "
                 "jalons = statut de projet, non mesurés · énergie : « à mesurer »")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b9_economie(out_root: Path) -> Path:
    """B9 — le bilan du gate : ce qu'il économise, ce qu'il coûte.

    Trois grandeurs de natures différentes (mises à jour évitées, latence, RAM) que des
    barres côte à côte rendraient incomparables : chacune a sa propre ligne et sa propre
    unité, lues comme un bilan, pas comme un graphe.
    """
    name = "b9_economie"
    summary = src.try_load("experiments/exp_S38_summary.json")
    if summary is None:
        return _empty("exp_S38_summary absent", name, out_root)
    results = summary.get("results", {})
    init = "pretrained"
    gated = [p for p in summary.get("policies", []) if p.startswith("gated")]

    rows = []
    for dataset in summary.get("datasets", []):
        table = (results.get(dataset, {}).get(init, {}) or {}).get("economy_table", {})
        for policy in gated:
            cell = table.get(policy, {})
            rows.append((
                f"{DATASET_LABEL.get(dataset, dataset)}\n{policy.replace('gated_', '')}",
                src.num(cell.get("updates_saved_pct")),
                src.num(cell.get("latency_saved_us")),
                src.num(cell.get("ram_added_bytes")),
                src.num(cell.get("f1_lost")),
            ))
    if not rows or all(r[1] is None for r in rows):
        return _empty("economy_table absente de exp_S38_summary", name, out_root)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    panels = [
        (1, "mises à jour évitées", lambda v: f"{100 * v:.0f} %", C_OK, 1),
        (2, "latence économisée", lambda v: f"{v:.0f} µs", C_PC, 1),
        (4, "F1 perdu", lambda v: ("+" if v >= 0 else "") + _fr(v, 3), C_BAD, 1),
    ]
    labels = [r[0] for r in rows]
    for ax, (idx, title, fmt, color, _) in zip(axes, panels):
        values = [r[idx] for r in rows]
        ys = np.arange(len(rows))
        ax.barh(ys, [v if v is not None else 0 for v in values],
                color=[NA_GRAY if v is None else color for v in values], height=0.55)
        for y, value in zip(ys, values):
            ax.text(value if value is not None else 0, y,
                    f"  {fmt(value)}" if value is not None else "  N/A",
                    va="center", fontsize=12, color=INK, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=11)
        ax.set_xticks([])
        ax.set_title(title, fontsize=14)
        _clean(ax)
        ax.spines["bottom"].set_visible(False)
        ax.margins(x=0.3)

    ram = next((r[3] for r in rows if r[3] is not None), None)
    _footer(fig, f"{BADGE_BOARD} · deltas mesurés face à la mise à jour systématique · "
                 f"coût mémoire du gate : {_bytes(ram)}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b4_ablation(out_root: Path) -> Path:
    """B4 — l'échelle d'ablation : un seul barreau explique l'effondrement.

    Le manuscrit superpose deux courbes, deux références pointillées et deux annotations
    de gain. Ici, une bande grise donne la référence FP32 commune, et **seul le barreau
    décisif** est annoté : c'est la réponse à la question posée.
    """
    name = "b4_ablation_echelle"
    label = {"legacy_c": "firmware\nhistorique", "fix_acc32": "+ accumulateur\nint32",
             "per_tensor_calib": "+ échelle\ncalibrée", "per_channel_int8": "+ échelle\npar canal",
             "q15": "Q15\n(16 bits)"}
    fig, ax = plt.subplots(figsize=(13, 6))
    names, plotted, refs = [], False, []
    for dataset, color, marker in zip(FOCUS_DATASETS, (C_PC, C_ALT), ("o", "s")):
        data = src.try_load(f"experiments/exp_S39_ablation/{dataset}.json")
        if data is None:
            continue
        ladder = data.get("ladder", [])
        names = [str(step.get("scheme")) for step in ladder]
        values = [src.num(step.get("f1")) for step in ladder]
        if not names:
            continue
        ax.plot(range(len(names)), values, marker=marker, ms=13, lw=3, color=color,
                label=DATASET_LABEL[dataset], zorder=3)
        refs.append(src.num(data.get("f1_fp32")))
        plotted = True
    if not plotted:
        return _empty("exp_S39_ablation absent", name, out_root)

    finite = [r for r in refs if r is not None]
    if finite:
        ax.axhspan(min(finite), max(finite), color=C_OK, alpha=0.15, zorder=1)
        ax.text(0, max(finite) + 0.03, "niveau FP32 à retrouver", fontsize=12, color=C_OK,
                fontweight="bold")
    if "per_tensor_calib" in names:
        step = names.index("per_tensor_calib")
        ax.annotate("le barreau décisif :\nl'échelle calibrée", xy=(step, 0.55),
                    xytext=(step + 0.35, 0.4), fontsize=14, color=C_BAD, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=2))

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label.get(n, n) for n in names], fontsize=12)
    ax.set_ylabel("F1 « fautif »")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12, loc="center right")
    ax.set_title("Un seul facteur explique l'effondrement INT8")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, "émulateur bit-exact du noyau embarqué (PC)")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b5_moment(out_root: Path) -> Path:
    """B5 — le moment de la quantification, lu en **écart** à la référence FP32.

    Le manuscrit trace les valeurs absolues sur une échelle si resserrée qu'elle donne
    du relief à des écarts de l'ordre de 1e-4. En traçant l'écart au FP32 autour d'un
    vrai zéro, le message « le moment ne change presque rien » devient la figure.
    """
    name = "b5_moment_quantification"
    moments = ["before", "after", "both"]
    label = {"before": "avant (QAT)", "after": "après (PTQ calibrée)",
             "both": "les deux (déploiement)"}
    colors = {"before": C_OK, "after": C_BOARD, "both": C_ALT}

    rows: list[tuple[str, float, str]] = []
    metric = None
    for dataset in FOCUS_DATASETS:
        data = src.try_load(f"experiments/exp_S46_ewc/{dataset}_all.json")
        if data is None:
            continue
        node = data.get("moments", {})
        metric = data.get("metric_name") or metric
        ref = src.num(node.get("fp32", {}).get("metric"))
        if ref is None:
            continue
        for moment in moments:
            value = src.num(node.get(moment, {}).get("metric"))
            if value is None:
                continue
            rows.append((f"{DATASET_LABEL[dataset]} · {label[moment]}",
                         value - ref, colors[moment]))
    if not rows:
        return _empty("exp_S46_ewc absent", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 6))
    ys = np.arange(len(rows))[::-1]
    for y, (_, delta, color) in zip(ys, rows):
        ax.barh(y, delta, height=0.55, color=color, zorder=3)
        ax.text(delta + (0.00002 if delta >= 0 else -0.00002), y, _fr(delta, 5),
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=12, color=INK, fontweight="bold")
    ax.axvline(0, color=INK, lw=2, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlabel(f"écart d'{metric or 'métrique'} par rapport au FP32")
    ax.margins(x=0.22)
    ax.set_title("Le moment de la quantification ne décide de rien")
    _clean(ax)
    _footer(fig, "émulateur bit-exact (PC) · écarts de l'ordre du dix-millième — "
                 "c'est la calibration qui décide, pas le moment")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b6_profondeur(out_root: Path) -> Path:
    """B6 — jusqu'où descendre en bits sans perdre la métrique.

    Un seul panneau : l'écart d'AUROC en fonction de la profondeur, avec la limite de
    tolérance tracée. Le gain RAM, second message, passe en étiquette sur chaque point
    plutôt qu'en second panneau.
    """
    name = "b6_profondeur_bits"
    s48 = src.s48_summary()
    if s48 is None:
        return _empty("exp_S48_summary absent", name, out_root)
    grid = s48.get("results_by_condition", {})

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = False
    for dataset, color, marker in zip(FOCUS_DATASETS, (C_PC, C_ALT), ("o", "s")):
        node = grid.get(dataset, {})
        bits = sorted(node, key=lambda k: -int(k))
        xs, ys, notes = [], [], []
        for b in bits:
            per_channel = node[b].get("per_channel", {})
            delta = src.num(per_channel.get("pc", {}).get("delta_auroc_vs_fp32"))
            ratio = src.num(per_channel.get("pc", {}).get("ram_ratio_theoretical_vs_fp32"))
            if delta is None:
                continue
            xs.append(int(b))
            ys.append(delta)
            notes.append(ratio)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, ms=13, lw=3, color=color,
                label=DATASET_LABEL[dataset], zorder=3)
        # Décalage vertical par jeu : à 4 bits les deux séries sont quasi confondues
        # et les deux étiquettes de gain se superposeraient exactement.
        offset = 0.0015 if marker == "o" else -0.0025
        for x, y, ratio in zip(xs, ys, notes):
            if ratio is not None:
                ax.text(x, y + offset, f"÷{_fr(ratio, 0)}", ha="center", fontsize=11,
                        color=color)
        plotted = True
    if not plotted:
        return _empty("results_by_condition absent de exp_S48_summary", name, out_root)

    ax.axhline(0, color=INK, lw=1, zorder=1)
    ax.set_xlabel("bits de poids")
    ax.set_ylabel("écart d'AUROC au FP32")
    ax.set_xticks([int(b) for b in sorted(grid.get(FOCUS_DATASETS[0], {}), key=lambda k: -int(k))])
    ax.margins(y=0.18)
    ax.invert_xaxis()
    ax.legend(fontsize=12)
    ax.set_title("Jusqu'où descendre sans perdre la métrique")
    _clean(ax)
    ax.spines["left"].set_visible(True)
    _footer(fig, "émulateur bit-exact (PC), échelle par canal · "
                 "les ÷N sont le gain RAM théorique bit-packé — la RAM `.bss` réelle est mesurée B?")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b7_latence_modeles(out_root: Path) -> Path:
    """B7 — latence par famille, sur échelle logarithmique.

    Sucettes horizontales triées plutôt que barres groupées : deux ordres de grandeur
    séparent Mahalanobis de HDC, et le classement doit se lire sans compter les barres.
    """
    name = "b7_latence_par_modele"
    sweep = src.s35_sweep()
    if sweep is None:
        return _empty("exp_S35_board_sweep_summary absent", name, out_root)

    fig, ax = plt.subplots(figsize=(13, 6))
    rows = []
    for model in GRID_MODELS:
        for dataset, color in zip(FOCUS_DATASETS, (C_PC, C_BOARD)):
            match = [c for c in sweep if isinstance(c, dict)
                     and c.get("model") == model and c.get("dataset") == dataset
                     and c.get("condition") == CONDITION]
            value = src.num(match[0].get("latency_us_p50")) if match else None
            if value is not None:
                rows.append((f"{MODEL_LABEL[model]} · {DATASET_LABEL[dataset]}", value, color))
    if not rows:
        return _empty("aucune latence dans le balayage S35", name, out_root)
    rows.sort(key=lambda r: r[1])

    for i, (label, value, color) in enumerate(rows):
        ax.plot([0.5, value], [i, i], color=FAINT, lw=2, zorder=1)
        ax.plot(value, i, "o", ms=14, color=color, zorder=3)
        ax.text(value * 1.15, i, f"{_int_fr(value)} µs", va="center", fontsize=12,
                color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xscale("log")
    ax.set_xlabel("latence P50 (µs, échelle logarithmique)")
    budget = src.num((src.s48_summary() or {}).get("gap2_latency_us_threshold"))
    ax.set_title("Deux ordres de grandeur séparent les familles")
    _clean(ax)
    margin = f" · budget Gap 2 = {_fr(budget / 1000, 0)} ms" if budget else ""
    _footer(fig, f"{BADGE_BOARD} · condition 5 variables{margin}")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b9_parite_gate(out_root: Path) -> Path:
    """B9 — parité du gate autonome, restreinte aux politiques qui ont un gate.

    ``frozen`` et ``always`` n'ont pas de gate : leur ``verdict_parity_rate`` est
    ``null`` **par construction**, soit huit barres vides sur seize. Les retirer des
    deux panneaux allège la figure et rend le titre « parité verdict du gate » exact.
    """
    name = "b9_parite_gate"
    summary = src.try_load("experiments/exp_S38_summary.json")
    if summary is None:
        return _empty("exp_S38_summary absent", name, out_root)
    gated = [p for p in summary.get("policies", []) if p.startswith("gated")]
    datasets = summary.get("datasets", [])
    inits = summary.get("init_modes", [])
    results = summary.get("results", {})

    panels = [("prediction_parity_rate", "parité prédiction"),
              ("verdict_parity_rate", "parité verdict du gate")]
    fig, axes = plt.subplots(len(datasets), len(panels),
                             figsize=(13, 4 * max(len(datasets), 1)), squeeze=False)
    labels = [f"{p.replace('gated_', '')}\n{i}" for i in inits for p in gated]
    for r, dataset in enumerate(datasets):
        for c, (key, title) in enumerate(panels):
            ax = axes[r][c]
            values = [src.num(results.get(dataset, {}).get(i, {}).get(p, {})
                              .get("board", {}).get(key)) for i in inits for p in gated]
            xs = np.arange(len(values))
            ax.bar(xs, [v if v is not None else 0 for v in values],
                   color=[NA_GRAY if v is None else C_OK for v in values],
                   edgecolor="white", linewidth=1)
            for x, value in zip(xs, values):
                ax.text(x, (value or 0) + 0.05, _fr(value, 3), ha="center", fontsize=11)
            ax.set_ylim(0, 1.2)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=10)
            ax.axhline(1, color=INK, lw=1, ls="--", zorder=0)
            ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)} — {title}", fontsize=13)
            _clean(ax)
            ax.spines["left"].set_visible(True)
    _footer(fig, f"{BADGE_BOARD} · seules les politiques à gate sont représentées : "
                 "sans gate, il n'y a pas de verdict à comparer")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 2 / 3 — Tableaux au style du manuscrit
# =============================================================================

def _table_figure(name: str, title: str, headers: list[str], rows: list[list[str]],
                  bold_cells: set[tuple[int, int]], col_x: list[float],
                  footer: str, out_root: Path, height: int = 5) -> Path:
    """Une slide-tableau : même grammaire booktabs que le rapport, taille projection."""
    fig, ax = plt.subplots(figsize=(13, height))
    _booktabs(ax, headers, rows, bold_cells=bold_cells, col_x=col_x)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    _footer(fig, footer)
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s5_etat_de_lart(out_root: Path) -> Path:
    """Slide 5 — quatre travaux, et surtout la colonne de droite."""
    rows = [
        ["TinyOL (Ren 2021)", "apprentissage en ligne sur Cortex-M4", "pas de décomposition mémoire"],
        ["QLR-CL (Ravaglia 2021)", "rejeu latent quantifié 8 bits, ÷4", "entraînement resté flottant"],
        ["HDC (Benatti 2019)", "apprentissage sans gradient, très basse conso.", "pas de cadre PdM industriel"],
        ["LifeLearner (Kwon 2023)", "212 Ko sur Cortex-M7", "surcoût d'adaptation non mesuré"],
    ]
    return _table_figure(
        "s5_etat_de_lart", "Quatre travaux tracent le champ",
        ["Travail", "Ce qu'il démontre", "Ce qu'il laisse ouvert"], rows,
        bold_cells={(r, 2) for r in range(len(rows))},
        col_x=[0.0, 0.26, 0.68],
        footer="c'est la colonne de droite qui construit la slide suivante",
        out_root=out_root, height=5,
    )


def _fig_s8_jeux_donnees(out_root: Path) -> Path:
    """Slide 8 — tab. 4.1 condensé : la colonne « type de signal » sort, le scénario reste."""
    rows = [
        ["D1", "Pump Maintenance", "maintenance (binaire)", "Domain, dérive temporelle", "Annexe"],
        ["D2", "Equipment Monitoring", "faulty (0/1)", "Domain, par type d'équipement", "Focus"],
        ["D3", "CWRU Bearing", "Type de défaut", "Domain, par défaut ou sévérité", "Annexe"],
        ["D4", "Pronostia (FEMTO)", "Condition (normal / faute)", "Class, par condition", "Focus"],
        ["D5", "CMAPSS (NASA)", "RUL", "Domain, par sous-jeu", "Focus (RUL)"],
        ["D6", "Paderborn", "Niveau de dommage", "Domain, par degré", "Annexe"],
    ]
    bold = {(1, 4), (3, 4), (3, 3), (1, 1), (3, 1)}
    return _table_figure(
        "s8_jeux_donnees", "Six jeux industriels, deux scénarios d'oubli",
        ["ID", "Jeu de données", "Label", "Scénario CL", "Rôle"], rows,
        bold_cells=bold, col_x=[0.0, 0.06, 0.32, 0.55, 0.87],
        footer="le découpage en tâches est fixé par configuration, donc reproductible",
        out_root=out_root, height=6,
    )


def _fig_s9_modeles(out_root: Path) -> Path:
    """Slide 9 — tab. 4.2 : les quatre familles, deux au focus."""
    rows = [
        ["M1 — TinyOL + tête OtO", "Architecture (backbone gelé)", "Supervisé", "Comparaison"],
        ["M2 — EWC Online + MLP", "Régularisation (Fisher)", "Supervisé", "Focus (+ INT8)"],
        ["M3 — HDC", "Architecture (non neuronale)", "Supervisé", "Comparaison"],
        ["M4 — Mahalanobis", "Baseline anomalie", "Non supervisé", "Focus"],
    ]
    bold = {(1, 0), (1, 3), (3, 0), (3, 3), (3, 2)}
    return _table_figure(
        "s9_modeles", "Quatre familles, deux mises au focus",
        ["Modèle", "Stratégie CL", "Nature", "Rôle"], rows,
        bold_cells=bold, col_x=[0.0, 0.30, 0.58, 0.78],
        footer="tête EWC embarquée : k → 32 → 16 → 2, k = nombre de variables d'entrée",
        out_root=out_root, height=5,
    )


# =============================================================================
# BLOC 2 — Cible matérielle
# =============================================================================

def _fig_s7_fiche_carte(out_root: Path) -> Path:
    """Slide 7 — la fiche de la carte, avec les deux traits qui reviennent plus tard.

    La FPU simple précision prépare le paradoxe de latence (slide 24) et l'absence
    d'accélérateur entier explique pourquoi l'INT8 ne gagne pas de temps : ces deux
    lignes sont mises en avant, le reste est du contexte.
    """
    name = "s7_fiche_carte"
    hw = src.hw_profile()
    ram_total = src.firmware_ram_total_bytes()
    if hw is None:
        return _empty("configs/hw_profile_f439zi.yaml absent", name, out_root)
    hardware = hw.get("hardware", {})
    sysclk = src.num(hardware.get("sysclk_hz"))
    fp32_peak = src.num(hardware.get("flops_peak_fp32"))
    int8_peak = src.num(hardware.get("flops_peak_int8"))
    budget_us = src.num((src.s48_summary() or {}).get("gap2_latency_us_threshold"))

    specs = [
        ("Cœur", "Cortex-M4", False),
        ("Horloge", f"{_fr(sysclk / 1_000_000, 0)} MHz" if sysclk else "N/A", False),
        ("SRAM", f"{_fr(ram_total / KIB, 0)} Ko (192 + 64 CCM)" if ram_total else "N/A", True),
        ("Flash", "2 Mo", False),
        ("FPU", "simple précision (FP32)", True),
        ("Accélérateur entier / NPU", "aucun" if fp32_peak == int8_peak else "présent", True),
        ("Budget de latence", f"{_fr(budget_us / 1000, 0)} ms par inférence + mise à jour"
         if budget_us else "N/A", False),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(specs) + 1)
    for i, (label, value, highlight) in enumerate(specs):
        y = len(specs) - i
        color = INK if highlight else MUTED
        if highlight:
            ax.add_patch(plt.Rectangle((0, y - 0.38), 1, 0.76, facecolor=TRACK, zorder=0))
        ax.text(0.02, y, label, fontsize=14, color=color, va="center")
        ax.text(0.52, y, value, fontsize=15, color=color, va="center",
                fontweight="bold" if highlight else "normal")
    ax.set_title("NUCLEO-F439ZI — l'enveloppe dans laquelle tout doit tenir",
                 fontsize=17, fontweight="bold", pad=16)
    _footer(fig, "en gris, les trois traits qui reviendront : la mémoire, la FPU, "
                 "et l'absence d'accélérateur entier")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s7_budget_ram(out_root: Path) -> Path:
    """Slide 7 — l'occupation réelle, décomposée, à la place d'un camembert placeholder.

    Le camembert précédent affichait une valeur écrite à la main et annonçait dans son
    titre qu'elle restait à remplacer. Ici, les trois composantes mesurées, dans
    l'enveloppe de la carte, avec la même grammaire que les slides 17 et 18.
    """
    name = "s7_budget_ram"
    dataset = FOCUS_DATASETS[0]
    cell = src.s49_board("ewc", dataset, "fp32", condition=CONDITION)
    budget = src.num((src.s48_summary() or {}).get("ram_budget_bytes"))
    if cell is None or budget is None:
        return _empty("mesure RAM S49 ou budget S48 absent", name, out_root)
    parts = [("`.data`", src.num(cell.get("data")), C_ALT),
             ("`.bss`", src.num(cell.get("bss")), C_PC),
             ("pic de pile", src.num(cell.get("stack_peak_update")), C_BOARD)]
    total = src.num(cell.get("total"))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.barh(0, budget, height=0.5, color=TRACK, zorder=1)
    cursor = 0.0
    for label, value, color in parts:
        if value is None:
            continue
        ax.barh(0, value, left=cursor, height=0.5, color=color,
                edgecolor="white", linewidth=1, zorder=3)
        cursor += value
    ax.text(cursor + budget * 0.01, 0, f"utilisé : {_bytes(total)}  ({_pct(total, budget)})",
            va="center", fontsize=15, color=INK, fontweight="bold")
    legend = "   ".join(f"{label} {_bytes(value)}" for label, value, _ in parts if value is not None)
    ax.text(0, -0.55, legend, fontsize=12, color=MUTED, va="top")
    ax.text(budget, 0.42, f"256 Ko de SRAM — {_bytes(budget)}", ha="right",
            fontsize=12, color=MUTED)

    ax.set_xlim(0, budget * 1.02)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Ce que le système occupe réellement sur la carte")
    _clean(ax)
    ax.spines["bottom"].set_visible(False)
    _footer(fig, f"{BADGE_BOARD} · EWC × {DATASET_LABEL[dataset]} · "
                 "RAM totale = `.data` + `.bss` + pic de pile")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 1 / 3 — Schémas de contexte et de méthode
# =============================================================================

def _fig_s2_cas_usage(out_root: Path) -> Path:
    """Slide 2 — qui fait quoi, et ce que la carte décide toute seule.

    Le diagramme UML d'origine mêlait six ovales, trois acteurs et des flèches
    `«include»` qui se croisent : lisible à l'écran, illisible en projection. Ici, une
    seule lecture gauche→droite, et la frontière de la carte devient le message.
    """
    name = "s2_cas_usage"
    fig, ax = base_fig()

    ax.add_patch(plt.Rectangle((0.26, 0.16), 0.46, 0.68, facecolor=TRACK,
                               edgecolor=INK, lw=2, zorder=0))
    ax.text(0.49, 0.88, "Carte embarquée — NUCLEO-F439ZI", ha="center", fontsize=15,
            fontweight="bold", color=INK)

    box(ax, 0.12, 0.62, "Équipement\nindustriel", C_PC, w=0.17, h=0.16, fontsize=13)
    box(ax, 0.12, 0.30, "Technicien", MUTED, w=0.17, h=0.16, fontsize=13)
    box(ax, 0.88, 0.46, "Supervision", C_BAD, w=0.17, h=0.16, fontsize=13)

    stage_row(ax, 0.62, ["Mesurer", "Détecter\nla faute", "Alerter"],
              [C_PC, C_OK, C_BAD], x0=0.35, x1=0.63, box_h=0.14, fontsize=12)
    stage_row(ax, 0.30, ["Détecter\nla dérive", "Mettre à jour\nle modèle"],
              [C_BOARD, C_BOARD], x0=0.38, x1=0.60, box_h=0.14, fontsize=12)

    arrow(ax, 0.21, 0.62, 0.275, 0.62)
    arrow(ax, 0.21, 0.34, 0.30, 0.40)
    arrow(ax, 0.72, 0.58, 0.795, 0.50)
    # Le même flux de mesures alimente les deux voies : la voie basse part de « Mesurer »,
    # pas de « Détecter la faute » — ce n'est pas la faute qui déclenche l'adaptation.
    arrow(ax, 0.35, 0.55, 0.38, 0.37, color=C_BOARD)

    ax.text(0.243, 0.665, "mesures", ha="center", fontsize=11, color=MUTED, style="italic")
    ax.text(0.235, 0.44, "étiquette\n(optionnelle)", ha="center", va="bottom", fontsize=11,
            color=MUTED, style="italic")
    ax.text(0.775, 0.565, "alerte", ha="center", fontsize=11, color=C_BAD, style="italic")
    ax.text(0.62, 0.30, "aucun réseau requis", ha="center", fontsize=13, color=C_BOARD,
            fontweight="bold")

    ax.set_title("Détecter la panne — et s'adapter — sans quitter la machine")
    _footer(fig, "la boucle du bas est ce que le reste de l'exposé mesure : "
                 "elle s'exécute entièrement sur la carte")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s3_cycle_de_vie(out_root: Path) -> Path:
    """Slide 3 — le cycle sur site, et pourquoi le réentraînement complet ne tient pas.

    Le logigramme d'origine superposait son bloc de tête et son libellé. Il est ici
    déroulé en une ligne, avec la boucle de retour tracée sous les étapes.
    """
    name = "s3_cycle_de_vie"
    fig, ax = base_fig()

    xs = stage_row(
        ax, 0.62,
        ["① Déploiement\nsur site", "② Récolte\ndonnées normales",
         "③ Calibration\ndu modèle", "④ Surveillance\ncontinue"],
        [C_PC, C_BOARD, C_ALT, C_OK], x0=0.13, x1=0.87, box_h=0.20, fontsize=12,
    )

    # Boucle de retour : c'est elle qui pose le problème de la slide suivante.
    y = 0.30
    ax.plot([xs[-1], xs[-1], xs[1], xs[1]], [0.52, y, y, 0.52],
            color=C_BAD, lw=2.5, zorder=2)
    arrow(ax, xs[1] + 0.001, y + 0.02, xs[1], 0.51, color=C_BAD)
    ax.text((xs[1] + xs[-1]) / 2, y - 0.05, "la distribution dérive — il faut réapprendre",
            ha="center", va="top", fontsize=14, color=C_BAD, fontweight="bold")

    ax.text(0.5, 0.10, "réentraîner depuis zéro suppose de tout garder en mémoire, "
                       "et de tout réenvoyer : c'est ce qu'on ne peut pas faire ici",
            ha="center", fontsize=12, color=MUTED, style="italic")

    ax.set_title("Le cycle de vie sur site — et là où il se bloque")
    _footer(fig, "schéma de contexte")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s10_chaine_portage(out_root: Path) -> Path:
    """Slide 10 ★ — la parité n'est pas espérée, elle est garantie par construction."""
    name = "s10_chaine_portage"
    fig, ax = base_fig()

    stage_row(ax, 0.66,
              ["Entraînement\nPyTorch", "export_weights_c.py",
               "model_weights*.h\n(généré)", "Flash\n(.rodata)"],
              [C_PC, C_BOARD, C_ALT, MUTED], x0=0.13, x1=0.87, box_h=0.17, fontsize=12)
    stage_row(ax, 0.32,
              ["pipeline_init()\nmemcpy → .bss", "poids vivants\n(SGD en ligne)"],
              [C_OK, C_OK], x0=0.30, x1=0.62, box_h=0.17, fontsize=12)
    # Retour Flash → RAM : contourné par la droite plutôt que tracé en diagonale
    # à travers le schéma, où il coupait les deux rangées.
    ax.plot([0.87, 0.94, 0.94, 0.36], [0.575, 0.575, 0.47, 0.47], color="#777777", lw=1.8)
    arrow(ax, 0.37, 0.47, 0.30, 0.41)

    ax.text(0.5, 0.12, "Le header C est GÉNÉRÉ, jamais édité à la main —\n"
                       "c'est ce qui rend la parité carte ↔ PC vérifiable, et non espérée",
            ha="center", fontsize=14, color=C_BAD, fontweight="bold")
    ax.text(0.5, 0.02, "Flash = référence immuable  ·  `.bss` = copie que l'apprentissage embarqué modifie",
            ha="center", fontsize=11, color=MUTED)

    ax.set_title("Du modèle entraîné aux poids qui vivent sur la carte")
    _footer(fig, "étapes = artefacts réels du dépôt")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s11_protocole_mesure(out_root: Path) -> Path:
    """Slide 11 — comment latence et RAM sont mesurées, avec les chiffres **rechargés**.

    La figure héritée annonçait « 130 µs / 403 µs » et « `.bss` 66,7 → 104,6 Ko » :
    des valeurs de l'époque du Sprint 26, écrites en dur, que la slide 20 contredit.
    Ici les mêmes grandeurs sont chargées depuis les campagnes courantes, donc
    identiques à celles des slides 18 et 20 par construction.
    """
    name = "s11_protocole_mesure"
    node = src.s36_node(FOCUS_DATASETS[0], "board_online", condition=CONDITION) or {}
    inference = src.num(node.get("latency_inference_only_us_p50"))
    total = src.num(node.get("latency_us_p50"))
    ram = src.s49_board("ewc", FOCUS_DATASETS[0], "fp32", condition=CONDITION) or {}
    bss, data_seg = src.num(ram.get("bss")), src.num(ram.get("data"))
    stack = src.num(ram.get("stack_peak_update"))
    sysclk = src.num((src.hw_profile() or {}).get("hardware", {}).get("sysclk_hz"))

    fig, ax = base_fig()
    ax.text(0.02, 0.93, "Latence — compteur de cycles DWT", fontsize=15,
            fontweight="bold", color=C_PC)
    stage_row(ax, 0.72, ["t0 = CYCCNT", "inférence\n(+ mise à jour)", "Δ = CYCCNT − t0"],
              [C_OK, C_PC, C_OK], x0=0.16, x1=0.84, box_h=0.15, fontsize=12, box_w=0.22)
    if sysclk:
        ax.text(0.5, 0.575, f"latence = Δcycles ÷ {_fr(sysclk / 1_000_000, 0)}"
                            f"  ·  résolution ≈ 1 cycle",
                ha="center", fontsize=12, color=MUTED, family="monospace")
    measured = []
    if inference is not None:
        measured.append(f"inférence seule {_int_fr(inference)} µs")
    if total is not None:
        measured.append(f"inférence + mise à jour {_int_fr(total)} µs")
    if measured:
        ax.text(0.5, 0.50, "mesuré : " + "  ·  ".join(measured), ha="center",
                fontsize=13, color=C_PC, fontweight="bold")

    ax.text(0.02, 0.36, "RAM — symboles du linker + sonde de pile", fontsize=15,
            fontweight="bold", color=C_BOARD)
    for cx, label, color in ((0.24, "`.data`", C_ALT), (0.50, "`.bss`", C_PC),
                             (0.76, "pic de pile", C_BOARD)):
        box(ax, cx, 0.19, label, color, w=0.20, h=0.13, fontsize=12)
    for cx in (0.37, 0.63):
        ax.text(cx, 0.19, "+", ha="center", va="center", fontsize=20, color=INK,
                fontweight="bold")
    parts = [(data_seg, "`.data`"), (bss, "`.bss`"), (stack, "pic")]
    shown = "  +  ".join(f"{lbl} {_bytes(v)}" for v, lbl in parts if v is not None)
    if shown:
        ax.text(0.5, 0.05, f"mesuré : {shown}", ha="center", fontsize=13,
                color=C_BOARD, fontweight="bold")

    ax.set_title("Mesurer sur la carte, sans instrument externe")
    _footer(fig, f"{BADGE_BOARD} · EWC × {DATASET_LABEL[FOCUS_DATASETS[0]]} · "
                 "chiffres identiques à ceux des slides 18 et 20 (même source)")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# BLOC 2 / 7 — Le triple gap, à l'ouverture puis au bilan
# =============================================================================

#: Les trois gaps, dans l'ordre de l'exposé. Le libellé de Gap 2 dit « RAM totale »
#: et non « `.bss` » : tout le bloc 5 démontre justement que `.bss` seul sous-estime.
GAPS: list[tuple[str, str, str]] = [
    ("Gap 1", "Données industrielles\nréelles",
     "séries temporelles,\nscénarios incrémentaux,\nprotocole reproductible"),
    ("Gap 2", "RAM totale mesurée\nsur carte",
     "`.data` + `.bss` + pic de pile,\nlatence sous le budget"),
    ("Gap 3", "Quantification INT8\npendant l'apprentissage",
     "et non seulement\nà l'inférence"),
]
GAP_COLORS = [C_ALT, C_BOARD, C_PC]


def _draw_gaps(ax, status: list[str] | None = None) -> None:
    """Les trois gaps en colonnes ; ``status`` ajoute un verdict sous chacun."""
    for i, ((tag, title, body), color) in enumerate(zip(GAPS, GAP_COLORS)):
        cx = 0.19 + i * 0.31
        box(ax, cx, 0.60, "", color, w=0.27, h=0.44, fontsize=1)
        ax.text(cx, 0.77, tag, ha="center", fontsize=17, color=color, fontweight="bold")
        ax.text(cx, 0.69, title, ha="center", va="center", fontsize=14, color=INK,
                fontweight="bold", linespacing=1.4)
        ax.text(cx, 0.53, body, ha="center", va="center", fontsize=11, color=MUTED,
                linespacing=1.5)
        if status:
            ax.text(cx, 0.28, status[i], ha="center", va="center", fontsize=13,
                    color=color, fontweight="bold", linespacing=1.5)


def _fig_s6_triple_gap(out_root: Path) -> Path:
    """Slide 6 ★ — les trois lacunes, et l'annonce du plan.

    Redessinée : la version héritée était un PNG **sans générateur**, et son libellé
    Gap 2 parlait d'« empreinte mémoire (`.bss`) » — ce que les slides 17-18
    contredisent explicitement. Le libellé est corrigé ici.
    """
    name = "s6_triple_gap"
    fig, ax = base_fig()
    _draw_gaps(ax)
    ax.text(0.5, 0.20, "Aucun travail publié ne les satisfait tous les trois",
            ha="center", fontsize=16, color=INK, fontweight="bold")
    ax.text(0.5, 0.09, "les trois blocs qui suivent traitent chacun l'un de ces gaps, dans cet ordre",
            ha="center", fontsize=13, color=MUTED, style="italic")
    ax.set_title("Trois lacunes, une seule démonstration")
    _footer(fig, "plan de l'exposé")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_s26_bilan_gaps(out_root: Path) -> Path:
    """Slide 26 — le même schéma, avec le statut réellement obtenu par gap.

    Le troisième statut est volontairement nuancé : c'est la promesse d'honnêteté
    faite au jury dès la slide 6.
    """
    name = "s26_bilan_gaps"
    fig, ax = base_fig()
    _draw_gaps(ax, status=[
        "✔  démontré\n4 familles · PC et carte",
        "✔  démontré\ncomposant par composant",
        "~  bilan nuancé\nRAM ÷4, pas la latence",
    ])
    ax.text(0.5, 0.10, "la rétropropagation quantifiée sur carte reste à faire",
            ha="center", fontsize=14, color=MUTED, style="italic")
    ax.set_title("Ce qui est démontré, et ce qui reste ouvert")
    _footer(fig, "reprise de la slide 6 — le jury doit reconnaître le schéma")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# Figures de secours (suite)
# =============================================================================

def _fig_b2_grille_complete(out_root: Path) -> Path:
    """B2 — la grille complète 5 jeux × 4 modèles, PC et carte sur la même ligne.

    Remplace **deux** heatmaps (une par plateforme) que le jury devait comparer de
    tête. Surtout : les heatmaps héritées dataient d'avant le Sprint 52 et
    affichaient, pour TinyOL carte, les valeurs de Mahalanobis — un bug de drapeau
    UART depuis corrigé. Les valeurs sont ici rechargées depuis la comparaison
    consolidée régénérée.
    """
    name = "b2_grille_complete"
    datasets = ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"]
    rows: list[tuple[str, str, float | None, float | None]] = []
    for dataset in datasets:
        for model in GRID_MODELS:
            rows.append((
                dataset, model,
                src.s35_condition_f1(CONDITION, dataset, model, "pc"),
                src.s35_condition_f1(CONDITION, dataset, model, "nucleo_f439zi"),
            ))
    if all(pc is None and bd is None for _, _, pc, bd in rows):
        return _empty("comparison_sprint23 absent — lancer generate_comparison_sprint23.py",
                      name, out_root)

    fig, ax = plt.subplots(figsize=(13, 9))
    labels = []
    for i, (dataset, model, pc, bd) in enumerate(rows):
        y = len(rows) - 1 - i
        if i % len(GRID_MODELS) == 0:
            ax.axhspan(y + 0.5, y - len(GRID_MODELS) + 0.5,
                       color=TRACK if (i // len(GRID_MODELS)) % 2 == 0 else "white", zorder=0)
        _dumbbell(ax, y, pc, bd, C_PC, C_BOARD, dim=model != "ewc")
        labels.append(f"{DATASET_LABEL.get(dataset, dataset)} · {MODEL_LABEL[model]}")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 « fautif »")
    ax.plot([], [], "o", color=C_PC, ms=10, label="PC")
    ax.plot([], [], "o", color=C_BOARD, ms=10, label="carte")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_title("Grille complète — cinq jeux, quatre familles")
    _clean(ax)
    _footer(fig, f"{BADGE_BOARD} · condition 5 variables · "
                 "cellule non mesurée = point creux « N/A », jamais 0")
    return savefig_png(fig, CATALOG, name, out_root)


def _fig_b3_architecture_firmware(out_root: Path) -> Path:
    """B3 — architecture firmware et flux de données, en un seul schéma.

    Fusionne les deux figures héritées. Les tailles mémoire, qui étaient **écrites en
    dur** dans le script d'origine (`~200 B`, `~9,7 Ko`…), sont chargées ou omises.
    """
    name = "b3_architecture_firmware"
    fig, ax = base_fig()

    ax.text(0.02, 0.93, "PC (Python)", fontsize=14, fontweight="bold", color=C_PC)
    stage_row(ax, 0.78, ["Jeu de données", "Variables\n+ normalisation", "Trame\n+ CRC8"],
              [MUTED, C_PC, C_PC], x0=0.16, x1=0.62, box_h=0.13, fontsize=11)
    arrow(ax, 0.70, 0.78, 0.86, 0.78)
    ax.text(0.78, 0.83, "UART", ha="center", fontsize=11, color=C_BAD, fontweight="bold")

    ax.text(0.02, 0.57, "Carte (C)", fontsize=14, fontweight="bold", color=C_BOARD)
    stage_row(ax, 0.42,
              ["RX + CRC", "Routage\n(nibble)", "Inférence", "Mise à jour\nCL", "TX + profilage"],
              [C_BOARD, C_BOARD, C_OK, C_OK, MUTED], x0=0.12, x1=0.88, box_h=0.13, fontsize=11)

    bss = src.num((src.s49_board("ewc", FOCUS_DATASETS[0], "fp32",
                                 condition=CONDITION) or {}).get("bss"))
    budget = src.num((src.s48_summary() or {}).get("ram_budget_bytes"))
    ax.text(0.5, 0.22, "Un seul binaire porte les quatre modèles ; le nibble haut de la trame "
                       "choisit le chemin d'inférence.",
            ha="center", fontsize=13, color=INK)
    if bss is not None:
        detail = f"`.bss` du build par défaut : {_bytes(bss)}"
        if budget:
            detail += f"  ({_pct(bss, budget)} du budget)"
        ax.text(0.5, 0.12, detail, ha="center", fontsize=13, color=C_BOARD,
                fontweight="bold")
    ax.text(0.5, 0.03, "chaque échantillon ressort en `experiments/exp_*/` : "
                       "résultats, snapshot de config, profilage",
            ha="center", fontsize=11, color=MUTED)

    ax.set_title("Un binaire, quatre modèles, un flux mesuré de bout en bout")
    _footer(fig, f"{BADGE_BOARD} · étapes = fonctions réelles de `pipeline.c`")
    return savefig_png(fig, CATALOG, name, out_root)


# =============================================================================
# Registre
# =============================================================================

#: Ordre des slides — c'est aussi l'ordre de génération et de relecture.
FIGURES = [
    _fig_s2_cas_usage,
    _fig_s3_cycle_de_vie,
    _fig_s4_oubli_mesure,
    _fig_s5_etat_de_lart,
    _fig_s7_fiche_carte,
    _fig_s6_triple_gap,
    _fig_s7_budget_ram,
    _fig_s8_jeux_donnees,
    _fig_s9_modeles,
    _fig_s10_chaine_portage,
    _fig_s11_protocole_mesure,
    _fig_s12_gele_vs_en_ligne,
    _fig_s13_accuracy_trompeuse,
    _fig_s14_oubli_bilan,
    _fig_s15_grille_classement,
    _fig_s16_paderborn,
    _fig_s17_ram_trois_niveaux,
    _fig_s18_ram_totale_cascade,
    _fig_s19_pile_par_phase,
    _fig_s20_latence_surcout,
    _fig_s21_parite_desaccords,
    _fig_s22_ram_deux_echelles,
    _fig_s23_effondrement_recuperation,
    _fig_s24_paradoxe_latence,
    _fig_s25_gate_economie,
    _fig_perspectives_trois_axes,
    _fig_s26_bilan_gaps,
    _fig_b2_grille_complete,
    _fig_b3_architecture_firmware,
    _fig_b4_ablation,
    _fig_b5_moment,
    _fig_b6_profondeur,
    _fig_b7_latence_modeles,
    _fig_b9_parite_gate,
    _fig_b9_economie,
]


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les figures de soutenance sous ``out_root/soutenance/``."""
    apply_style("slide")
    paths = []
    for fn in FIGURES:
        paths.append(fn(out_root))
        plt.close("all")
    return paths
