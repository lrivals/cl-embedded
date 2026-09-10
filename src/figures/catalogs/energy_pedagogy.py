"""Catalogue `energy_pedagogy` — expliquer la mesure d'énergie : quoi, comment, pourquoi.

Dix figures qui déroulent la campagne LPM01A comme un raisonnement, et non comme une
liste de résultats. La question de fond n'est pas « combien de µJ ? » mais **comment
obtenir un µJ auquel on puisse croire**, quand la sonde ne sait rendre qu'un courant
moyen sur dix secondes.

    a1_chaine_de_mesure.png       Ce que la sonde mesure physiquement, et ce qu'elle ignore.
    a2_statique_vs_dynamique.png  Pourquoi le profil temporel nous est refusé (plafond mA).
    b1_probleme_isoler.png        Le signal cherché occupe 0,5 % de la fenêtre de mesure.
    b2_trois_estimateurs.png      Trois estimateurs, trois valeurs — et l'écart est un résultat.
    c1_methode_delta.png          Soustraire un repos : simple, et suspendu à sa référence.
    c2_methode_regression.png     Lire une pente : le coût fixe s'élimine de lui-même.
    c3_methode_lot.png            Grouper les inférences : isoler le calcul de la trame.
    d1_pieges_du_banc.png         Quatre pièges mesurés, et la parade de chacun.
    d2_repos_nest_pas_repos.png   « Repos » est une définition, pas un état.
    e1_de_la_mesure_a_lautonomie.png  Ce que la chaîne permet de conclure.

**Toute valeur chiffrée provient d'un ``load_experiment``** sur ``experiments/`` — les
schémas de principe (A1, et les volets de mécanisme de B1/B2/C1) ne portent aucune donnée
et le disent en note de bas de figure. Une grandeur non mesurée reste grise avec sa
raison, jamais un zéro (garde AST ``test_no_hardcoded_results``).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.loaders import load_experiment
from src.figures.registry import register_catalog
from src.figures.schematic import arrow, base_fig, box, footnote
from src.figures.style import STRATEGY_COLORS, apply_style, savefig_png

CATALOG = "energy_pedagogy"

# ── Palette ──────────────────────────────────────────────────────────────────
C_PROBE = STRATEGY_COLORS["fp32"]         # bleu   — la sonde, le mesuré
C_OK = STRATEGY_COLORS["int8_qat"]        # vert   — ce qui marche
C_ALERT = STRATEGY_COLORS["int8_ptq_legacy"]  # rouge — le piège, le refus
C_WARN = STRATEGY_COLORS["int8_v2"]       # orange — la nuance
C_ACCENT = STRATEGY_COLORS["q15"]         # violet — l'accent
C_BROWN = STRATEGY_COLORS["int16_am"]     # brun   — le coût fixe
NA_GRAY = "#cccccc"
INK = "#333333"
MUTED = "#666666"

# ── Constantes d'unités (jamais des résultats) ───────────────────────────────
MA_PER_A = 1000
UA_PER_A = 1000000
US_PER_S = 1000000
UJ_PER_J = 1000000
PCT = 100
SEC_PER_H = 3600
WRAP = 58
REF_CAPACITY_MAH = 2000
IDLE_PAD_MA = 2

BADGE_SCHEMA = "schéma de principe — aucune donnée tracée"
BADGE_MEASURED = "mesuré board réelle NUCLEO-F439ZI + sonde X-NUCLEO-LPM01A"
BADGE_MIXED = "schéma de principe à gauche · valeurs mesurées à droite (LPM01A)"
NA_TEXT = "à mesurer"


# ── Sources (lecture seule, skip gracieux) ───────────────────────────────────

def _load(rel: str) -> dict | None:
    try:
        data, _ = load_experiment(rel)
    except FileNotFoundError:
        return None
    return data


def _num(value) -> float | None:
    """Nombre exploitable ; ``None`` sur booléen, chaîne (« à mesurer ») ou absence."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _empty(message: str, name: str, out_root: Path) -> Path:
    fig, ax = base_fig()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color=MUTED,
            wrap=True, transform=ax.transAxes)
    return savefig_png(fig, CATALOG, name, out_root)


def _grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, color="#dddddd")


# ═══════════════ A — Ce que la sonde mesure physiquement ═════════════════════

def _fig_a1_chaine(out_root: Path) -> Path:
    """A1 — le chemin du courant, et la frontière de ce qui est compté."""
    fig, ax = base_fig()

    box(ax, 0.13, 0.72, "USB ST-LINK\n(alimentation carte)", MUTED, w=0.19, h=0.15,
        fontsize=10)
    box(ax, 0.38, 0.72, "régulateur 3,3 V\nde la Nucleo", MUTED, w=0.19, h=0.15,
        fontsize=10)
    box(ax, 0.63, 0.72, "cavalier JP5\nRETIRÉ", C_ALERT, w=0.17, h=0.15, fontsize=10)
    arrow(ax, 0.225, 0.72, 0.283, 0.72)
    arrow(ax, 0.475, 0.72, 0.543, 0.72)

    box(ax, 0.63, 0.42, "sonde X-NUCLEO-LPM01A\nCN14 : VOUT(+) et GND", C_PROBE,
        w=0.28, h=0.17, fontsize=11)
    arrow(ax, 0.63, 0.645, 0.63, 0.512, color=C_ALERT)
    arrow(ax, 0.77, 0.42, 0.845, 0.42, color=C_PROBE)

    box(ax, 0.9, 0.42, "VDD_MCU\nSTM32F439", C_OK, w=0.15, h=0.17, fontsize=11)

    ax.text(0.5, 0.24,
            "Le courant du MCU ne peut plus passer par le cavalier : il traverse la sonde.\n"
            "C'est le seul montage qui mesure la carte et non l'alimentation USB.",
            ha="center", va="center", fontsize=11, color=INK, transform=ax.transAxes)

    box(ax, 0.24, 0.42, "DANS le périmètre\n\nSTM32 · PHY Ethernet\ntout ce que porte VDD_MCU",
        C_OK, w=0.3, h=0.22, fontsize=10)
    box(ax, 0.24, 0.1, "HORS périmètre\n\nST-LINK · LED · régulateur",
        NA_GRAY, w=0.3, h=0.16, fontsize=10)

    ax.text(0.78, 0.13, "E  =  V × ∫ I dt", ha="center", va="center", fontsize=17,
            color=C_ACCENT, transform=ax.transAxes)
    ax.text(0.78, 0.05,
            "V est fixée et connue ; toute la difficulté\nest dans la mesure de I au bon instant.",
            ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    fig.suptitle("Ce que la sonde mesure — et ce qu'elle ignore")
    footnote(fig, BADGE_SCHEMA + " · câblage : docs/context/lpm01a_setup.md")
    return savefig_png(fig, CATALOG, "a1_chaine_de_mesure", out_root)


def _fig_a2_modes(out_root: Path) -> Path:
    """A2 — le mode dynamique est refusé par le courant de la carte, pas par un choix."""
    summary = _load("experiments/exp_S53_freq_sweep/summary.json")
    if summary is None:
        return _empty("exp_S53_freq_sweep/summary.json absent", "a2_statique_vs_dynamique",
                      out_root)
    dyn = summary.get("acqmode_dyn_by_mhz", {})
    freqs = sorted(int(k) for k in dyn)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.axis("off")
    box(ax, 0.5, 0.82, "MODE DYNAMIQUE\n\néchantillonnage rapide\n→ profil temporel, phase par phase",
        C_OK, w=0.86, h=0.26, fontsize=11)
    box(ax, 0.5, 0.48, "MODE STATIQUE\n\nun seul courant moyen par fenêtre\n→ aucune structure temporelle",
        C_PROBE, w=0.86, h=0.26, fontsize=11)
    ax.text(0.5, 0.22,
            "Le mode dynamique impose un plafond de courant.\n"
            "La carte le dépasse : le profil temporel nous est\n"
            "REFUSÉ par le matériel, pas écarté par confort.",
            ha="center", va="center", fontsize=11, color=C_ALERT, transform=ax.transAxes)
    ax.text(0.5, 0.04, "→ tout le reste de la campagne consiste à retrouver,\n"
                       "par l'inférence statistique, ce que le mode dynamique aurait donné",
            ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax = axes[1]
    x = np.arange(len(freqs))
    peaks = [_num(dyn[str(f)].get("i_max_ma")) for f in freqs]
    ok = [bool(dyn[str(f)].get("succeeded")) for f in freqs]
    colors = [C_OK if o else C_ALERT for o in ok]
    ax.bar(x, [v if v is not None else 0.0 for v in peaks], 0.55, color=colors,
           edgecolor=INK, linewidth=0.5)
    for xi, v, o in zip(x, peaks, ok):
        if v is None:
            continue
        ax.text(xi, v, f"{v:.1f} mA\n{'accepté' if o else 'refusé'}", ha="center",
                va="bottom", fontsize=10, color=INK)
    ceiling = max(v for v in peaks if v is not None)
    ax.axhline(ceiling, color=NA_GRAY, linewidth=0)  # ancre l'échelle sans tracer de seuil
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} MHz" for f in freqs])
    ax.set_ylabel("courant crête relevé par la sonde (mA)")
    ax.set_title("Tentatives d'acquisition dynamique", fontsize=11)
    ax.margins(y=0.2)
    _grid(ax)
    ax.text(0.02, 0.97,
            "vert = le mode dynamique a démarré\n"
            "rouge = refusé, le courant de la carte\ndépasse le plafond de la sonde",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, color=MUTED)

    fig.suptitle("Deux modes d'acquisition — et pourquoi il ne nous en reste qu'un")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "a2_statique_vs_dynamique", out_root)


# ═══════ B — Le problème : on ne peut pas mesurer une inférence seule ════════

def _fig_b1_probleme(out_root: Path) -> Path:
    """B1 — le calcul cherché occupe une fraction infime de la fenêtre de mesure."""
    cell = _load("experiments/exp_S53_rate_sweep/ewc_fp32.json")
    if cell is None:
        return _empty("exp_S53_rate_sweep/ewc_fp32.json absent", "b1_probleme_isoler",
                      out_root)
    latency_us = _num(cell.get("dwt_latency_us_p50"))
    window_s = _num(cell.get("window_s"))
    rate_hz = _num((cell.get("protocol_check") or {}).get("check_rate_hz"))
    duty = _num(cell.get("duty_cycle_at_max_rate"))

    fig, axes = plt.subplots(2, 1, height_ratios=[3, 2])

    # Volet haut — chronogramme de principe d'une période.
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.06, 0.3), 0.88, 0.34, facecolor="#eef4fb",
                               edgecolor=INK, linewidth=1))
    ax.add_patch(plt.Rectangle((0.06, 0.3), 0.1, 0.34, facecolor=C_BROWN,
                               edgecolor=INK, linewidth=1))
    ax.add_patch(plt.Rectangle((0.16, 0.3), 0.018, 0.34, facecolor=C_ALERT,
                               edgecolor=INK, linewidth=1))
    ax.text(0.11, 0.72, "réception de la trame UART", ha="center", fontsize=10,
            color=C_BROWN)
    ax.text(0.30, 0.78, "inférence  ← c'est CE QUE L'ON CHERCHE", ha="center",
            fontsize=11, color=C_ALERT)
    arrow(ax, 0.28, 0.74, 0.18, 0.62, color=C_ALERT)
    ax.text(0.56, 0.47, "attente de la trame suivante", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.text(0.5, 0.13, "une période du flux — le mode statique n'en voit que la MOYENNE",
            ha="center", fontsize=11, color=INK)

    # Volet bas — les proportions réelles.
    ax = axes[1]
    if latency_us and rate_hz:
        period_us = US_PER_S / rate_hz
        share = latency_us / period_us * PCT
        parts = [share, PCT - share]
        ax.barh([0], [parts[0]], height=0.35, color=C_ALERT, edgecolor=INK,
                linewidth=0.5, label=f"calcul ({latency_us:.0f} µs)")
        ax.barh([0], [parts[1]], left=[parts[0]], height=0.35, color=NA_GRAY,
                edgecolor=INK, linewidth=0.5, label="trame UART + attente")
        ax.set_xlim(0, PCT)
        ax.set_ylim(-0.6, 0.9)
        ax.set_yticks([])
        ax.legend(fontsize=10, loc="lower center", ncol=2)
        lines = [
            f"À {rate_hz:.0f} Hz, une période dure {period_us:.0f} µs "
            f"et le calcul en occupe {latency_us:.0f} µs, soit {share:.1f} %.",
            f"La sonde intègre sur une fenêtre de {window_s:.0f} s : "
            f"le calcul y est dilué dans {PCT - share:.1f} % d'autre chose.",
        ]
        if duty is not None:
            lines.append(
                f"Rapport cyclique de calcul relevé à la cadence maximale : "
                f"{duty * PCT:.2f} %."
            )
        ax.set_xlabel("part de la période, en %")
        ax.text(0.5, 0.95, textwrap.fill(" ".join(lines), 96), transform=ax.transAxes,
                ha="center", va="top", fontsize=11, color=INK)

    fig.suptitle("Pourquoi une inférence ne se mesure pas directement")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "b1_probleme_isoler", out_root)


ESTIMATORS: list[tuple[str, str, str, str, str]] = [
    ("delta", "experiments/exp_S53_wfi/delta_recovery.json",
     "cells.ewc_fp32.energy_uj_per_inference", "protocole delta",
     "I(charge) − I(repos)\n÷ nombre d'inférences"),
    ("regression", "experiments/exp_S53_rate_sweep/summary.json",
     "cells.ewc_fp32.energy_uj_per_inference", "régression de cadence",
     "pente de I(cadence)\nl'ordonnée à l'origine s'élimine"),
    ("lot", "experiments/exp_S53_wfi/batch_sweep.json",
     "ewc_fp32.energy_uj_per_inference", "régression par lot",
     "pente de I(N par trame)\nla trame devient constante"),
]


def _dig(data: dict | None, dotted: str):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _fig_b2_estimateurs(out_root: Path) -> Path:
    """B2 — la même grandeur, trois estimateurs, trois valeurs : l'écart est informatif."""
    values, labels, formulas = [], [], []
    for _, rel, dotted, label, formula in ESTIMATORS:
        values.append(_num(_dig(_load(rel), dotted)))
        labels.append(label)
        formulas.append(formula)
    if all(v is None for v in values):
        return _empty("agrégats S53 absents", "b2_trois_estimateurs", out_root)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.95, "Ce que chaque estimateur laisse entrer", ha="center",
            fontsize=12, color=INK, transform=ax.transAxes)
    contents = [
        ("protocole delta", ["repos soustrait ✓", "trame UART COMPTÉE ✗", "calcul ✓"], C_WARN),
        ("régression de cadence", ["repos éliminé ✓", "trame UART COMPTÉE ✗", "calcul ✓"], C_PROBE),
        ("régression par lot", ["repos éliminé ✓", "trame UART éliminée ✓", "calcul ✓"], C_OK),
    ]
    for i, (name, items, color) in enumerate(contents):
        y = 0.72 - i * 0.28
        box(ax, 0.5, y, f"{name}\n" + " · ".join(items), color, w=0.94, h=0.2,
            fontsize=10)
    ax.text(0.5, 0.03,
            "Plus on descend, moins il reste de coût étranger dans le chiffre.",
            ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax = axes[1]
    x = np.arange(len(values))
    colors = [C_WARN, C_PROBE, C_OK]
    ax.bar(x, [v if v is not None else 0.0 for v in values], 0.55,
           color=[c if v is not None else NA_GRAY for c, v in zip(colors, values)],
           edgecolor=INK, linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v if v is not None else 0.0,
                f"{v:.2f} µJ" if v is not None else NA_TEXT,
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(lab, 16) for lab in labels], fontsize=10)
    ax.set_ylabel("énergie par inférence (µJ, échelle log)")
    ax.set_title("Même modèle, même carte, même jour", fontsize=11)
    ax.margins(y=0.35)
    _grid(ax)

    known = [v for v in values if v is not None]
    if len(known) >= 2:
        ax.text(0.98, 0.97,
                textwrap.fill(
                    f"Écart du premier au dernier : un facteur "
                    f"{max(known) / min(known):.0f}.\nCe n'est pas une contradiction — "
                    f"c'est la mesure de ce que coûte la communication autour du calcul. "
                    f"Les trois chiffres sont conservés séparément, jamais moyennés.", 40),
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color=C_ACCENT)

    fig.suptitle("Trois estimateurs de la même grandeur — et pourquoi ils diffèrent")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "b2_trois_estimateurs", out_root)


# ═══════════════════ C — Les trois méthodes en détail ════════════════════════

CELL_LABEL: dict[str, str] = {
    "ewc_fp32": "EWC FP32", "ewc_int8": "EWC INT8",
    "hdc_fp32": "HDC FP32", "hdc_int8": "HDC INT8",
    "tinyol_fp32": "TinyOL FP32", "tinyol_int8": "TinyOL INT8",
    "maha_fp32": "Maha FP32", "maha_int8": "Maha INT8",
}


def _fig_c1_delta(out_root: Path) -> Path:
    """C1 — soustraire un repos : le principe, et sa dépendance à la référence."""
    data = _load("experiments/exp_S53_wfi/delta_recovery.json")
    if data is None:
        return _empty("exp_S53_wfi/delta_recovery.json absent", "c1_methode_delta",
                      out_root)
    idle_a = _num(data.get("i_idle_a"))
    cells = data.get("cells", {})
    keys = [k for k in CELL_LABEL if k in cells]

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.93, "Le principe", ha="center", fontsize=12, color=INK,
            transform=ax.transAxes)
    ax.text(0.5, 0.76,
            "E par inférence  =  (I_charge − I_repos) × V × T\n÷ nombre d'inférences",
            ha="center", va="center", fontsize=13, color=C_ACCENT,
            transform=ax.transAxes)
    box(ax, 0.5, 0.5,
        "Tout repose sur I_repos.\n"
        "Si la référence est prise trop tôt dans la session,\n"
        "ou si le firmware « au repos » scrute activement l'UART,\n"
        "la soustraction rend un résultat faux — parfois négatif.",
        C_ALERT, w=0.94, h=0.28, fontsize=11)
    ax.text(0.5, 0.22,
            "C'est exactement ce qui s'est produit au Sprint 50 :\n"
            "le « repos » consommait plus que la charge.",
            ha="center", va="center", fontsize=11, color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.04, "→ parade : sommeil UART (__WFI) + référence de session établie",
            ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax = axes[1]
    y = np.arange(len(keys))
    means = [_num(cells[k].get("i_mean_a")) for k in keys]
    if idle_a is not None:
        ax.axvline(idle_a * MA_PER_A, color=C_OK, linestyle="--", linewidth=2)
        right = max(v for v in means if v is not None) * MA_PER_A
        ax.set_xlim(idle_a * MA_PER_A - IDLE_PAD_MA, right + IDLE_PAD_MA * 3)
        ax.text(idle_a * MA_PER_A, 0.5, f"I_repos = {idle_a * MA_PER_A:.2f} mA",
                transform=ax.get_xaxis_transform(), rotation=90, ha="right",
                va="center", color=C_OK, fontsize=11)
    ax.barh(y, [(v - (idle_a or 0.0)) * MA_PER_A if v is not None else 0.0 for v in means],
            left=[(idle_a or 0.0) * MA_PER_A] * len(keys), height=0.6,
            color=C_PROBE, edgecolor=INK, linewidth=0.5)
    for yi, k, v in zip(y, keys, means):
        uj = _num(cells[k].get("energy_uj_per_inference"))
        if v is None:
            continue
        ax.text(v * MA_PER_A, yi, f"  {uj:.1f} µJ" if uj is not None else f"  {NA_TEXT}",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([CELL_LABEL[k] for k in keys], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("courant moyen mesuré (mA)")
    ax.set_title("Huit cellules, une seule référence de repos", fontsize=11)

    fig.suptitle("Méthode 1 — le protocole delta")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "c1_methode_delta", out_root)


def _fig_c2_regression(out_root: Path) -> Path:
    """C2 — lire une pente : le coût fixe s'élimine, et la saturation se voit."""
    cell = _load("experiments/exp_S53_rate_sweep/ewc_fp32.json")
    if cell is None:
        return _empty("exp_S53_rate_sweep/ewc_fp32.json absent", "c2_methode_regression",
                      out_root)
    points = cell.get("points", [])
    slope_ua = _num(cell.get("slope_ua_per_hz"))
    intercept_ma = _num(cell.get("intercept_ma"))
    r2 = _num(cell.get("r2"))
    uj = _num(cell.get("energy_uj_per_inference"))
    unc = _num(cell.get("energy_uncertainty_uj"))
    sat_hz = _num(cell.get("saturation_rate_hz"))
    max_fit = _num(cell.get("max_rate_fitted_hz"))

    fig, ax = plt.subplots()
    kept = [p for p in points
            if max_fit is None or _num(p.get("rate_hz")) <= max_fit]
    drop = [p for p in points if p not in kept]

    xs = [_num(p["rate_hz"]) for p in kept]
    ys = [_num(p["i_mean_a"]) * MA_PER_A for p in kept]
    es = [(_num(p.get("i_std_a")) or 0.0) * MA_PER_A for p in kept]
    ax.errorbar(xs, ys, yerr=es, fmt="o", capsize=5, color=C_PROBE, markersize=8,
                label="points retenus (3 répétitions, barres = écart-type)")

    if slope_ua is not None and intercept_ma is not None and xs:
        grid = np.linspace(0, max(xs), len(xs) + 1)
        ax.plot(grid, intercept_ma + slope_ua / MA_PER_A * grid, color=C_ACCENT,
                linewidth=2, label="ajustement pondéré par 1/σ²")
        ax.axhline(intercept_ma, color=C_BROWN, linestyle=":", linewidth=2)
        ax.annotate("ordonnée à l'origine = coût fixe (repos + hôte)\n"
                    "la pente s'en affranchit par construction",
                    xy=(0.98, intercept_ma), xycoords=("axes fraction", "data"),
                    xytext=(0, -10), textcoords="offset points", ha="right", va="top",
                    fontsize=10, color=C_BROWN)

    for p in drop:
        rx, ry = _num(p["rate_hz"]), _num(p["i_mean_a"]) * MA_PER_A
        ach = _num(p.get("achieved_rate_hz"))
        ax.plot([rx], [ry], marker="x", markersize=14, color=C_ALERT,
                markeredgewidth=3)
        note = "écarté — saturation"
        if ach is not None and rx is not None:
            note += f"\n{rx:.0f} Hz demandés, {ach:.0f} Hz atteints"
        ax.annotate(note, (rx, ry), textcoords="offset points", xytext=(-16, 4),
                    ha="right", fontsize=10, color=C_ALERT)

    ax.set_xlabel("cadence d'inférence demandée (Hz)")
    ax.set_ylabel("courant moyen mesuré (mA)")
    ax.legend(fontsize=10, loc="upper left")
    _grid(ax)
    ax.margins(x=0.15, y=0.15)

    caption = []
    if slope_ua is not None and uj is not None:
        caption.append(f"pente = {slope_ua:.2f} µA/Hz  →  {uj:.2f} µJ par inférence")
    if unc is not None and r2 is not None:
        caption.append(f"incertitude ± {unc:.2f} µJ · r² = {r2:.4f}")
    if sat_hz is not None:
        caption.append(f"seuil de saturation détecté à {sat_hz:.0f} Hz")
    ax.set_title("Méthode 2 — la pente de I(cadence)\n" + "  ·  ".join(caption),
                 fontsize=12)
    fig.text(0.5, 0.005,
             BADGE_MEASURED + " · ordre des acquisitions randomisé (graine 42) "
             "pour que la dérive de session ne se confonde pas avec la cadence",
             ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "c2_methode_regression", out_root)


def _fig_c3_lot(out_root: Path) -> Path:
    """C3 — grouper les inférences dans une trame : la trame devient une constante."""
    data = _load("experiments/exp_S53_wfi/batch_sweep.json")
    if data is None:
        return _empty("exp_S53_wfi/batch_sweep.json absent", "c3_methode_lot", out_root)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.axis("off")
    ax.text(0.5, 0.94, "L'idée", ha="center", fontsize=12, color=INK,
            transform=ax.transAxes)
    for i, (n_label, n_boxes, color) in enumerate(
            [("N = 1 inférence par trame", 1, C_PROBE),
             ("N = 4 inférences par trame", 4, C_OK)]):
        y = 0.68 - i * 0.28
        ax.text(0.5, y + 0.13, n_label, ha="center", fontsize=11, color=INK,
                transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0.08, y - 0.05), 0.16, 0.1, facecolor=C_BROWN,
                                   edgecolor=INK, transform=ax.transAxes))
        ax.text(0.16, y, "trame", ha="center", va="center", fontsize=9, color="white",
                transform=ax.transAxes)
        for j in range(n_boxes):
            ax.add_patch(plt.Rectangle((0.27 + j * 0.09, y - 0.05), 0.07, 0.1,
                                       facecolor=color, edgecolor=INK,
                                       transform=ax.transAxes))
    ax.text(0.5, 0.2,
            "Le coût de la trame ne bouge pas ; seul le nombre\n"
            "d'inférences change. La pente ne peut donc porter\n"
            "que le calcul — c'est l'estimateur le plus propre.",
            ha="center", va="center", fontsize=11, color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.03,
            "contrôle : les prédictions doivent rester identiques d'un N à l'autre",
            ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax = axes[1]
    checks: list[str] = []
    for key, color in (("ewc_fp32", C_PROBE), ("maha_fp32", C_ACCENT)):
        cell = data.get(key)
        if cell is None:
            continue
        pts = cell.get("points", [])
        kept = [p for p in pts if not p.get("saturated")]
        xs = [p["n"] for p in kept]
        ys = [_num(p["i_mean_a"]) * MA_PER_A for p in kept]
        ax.plot(xs, ys, marker="o", linestyle="", color=color,
                label=CELL_LABEL.get(key, key))
        slope = _num(cell.get("slope_a_per_inference_per_frame"))
        icept = _num(cell.get("intercept_a"))
        if slope is not None and icept is not None and xs:
            grid = np.linspace(0, max(xs), len(xs) + 1)
            ax.plot(grid, (icept + slope * grid) * MA_PER_A, color=color, linewidth=1.6,
                    alpha=0.7)
        uj, r2 = _num(cell.get("energy_uj_per_inference")), _num(cell.get("r2"))
        if uj is not None and r2 is not None and ys:
            ax.annotate(f"{uj:.3f} µJ / inférence (r² = {r2:.4f})", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(10, -6), ha="left",
                        va="top", fontsize=10, color=color)
        parity = {p.get("pred_parity_vs_n1") for p in pts} - {None}
        if parity == {1.0}:
            checks.append(CELL_LABEL.get(key, key))
    ax.set_xlabel("inférences par trame (N imposé à la compilation)")
    ax.set_ylabel("courant moyen mesuré (mA)")
    ax.margins(x=0.2, y=0.15)
    ax.legend(fontsize=10, loc="upper left")
    if checks:
        ax.text(0.98, 0.06,
                "contrôle : parité de prédiction 1.000 entre lots — "
                + " et ".join(checks),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_title("Le courant croît avec N, la trame reste constante", fontsize=11)
    _grid(ax)

    fig.suptitle("Méthode 3 — la régression par lot")
    fig.text(0.5, 0.005, BADGE_MIXED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "c3_methode_lot", out_root)


# ═════════════════════ D — Les pièges du banc ════════════════════════════════

def _fig_d1_pieges(out_root: Path) -> Path:
    """D1 — quatre pièges, chacun montré sur ses propres données de banc."""
    idle = _load("experiments/exp_S53_wfi/idle_reference.json")
    counter = _load("experiments/exp_S53_counterbalance/counterbalance.json")
    rate = _load("experiments/exp_S53_rate_sweep/ewc_fp32.json")
    if idle is None or counter is None or rate is None:
        return _empty("agrégats S53 absents", "d1_pieges_du_banc", out_root)

    fig, axes = plt.subplots(2, 2)

    # (a) Biais de première acquisition.
    ax = axes[0][0]
    node = idle.get("default_polling", {})
    warm = [(_num(v) or 0.0) * MA_PER_A for v in node.get("warmup_discarded_a", [])]
    runs = [(_num(r["i_a"]) or 0.0) * MA_PER_A for r in node.get("i_idle_runs_a", [])]
    ax.plot(range(len(warm)), warm, marker="X", linestyle="", markersize=13,
            color=C_ALERT, label="acquisitions de préchauffage (rebutées)")
    ax.plot(range(len(warm), len(warm) + len(runs)), runs, marker="o", linestyle="",
            color=C_OK, label="acquisitions retenues")
    ax.set_title("(a) La première acquisition ment", fontsize=11)
    ax.set_xlabel("rang dans la session")
    ax.set_ylabel("courant (mA)")
    ax.legend(fontsize=8)
    _grid(ax)

    # (b) Dérive d'établissement de session.
    ax = axes[0][1]
    ranks = counter.get("idle_by_rank", [])
    xs = [r["session_index"] for r in ranks]
    ys = [(_num(r["i_a"]) or 0.0) * MA_PER_A for r in ranks]
    ax.plot(xs, ys, marker="o", color=C_PROBE)
    if ys:
        ax.annotate("premier repos\nde la session", (xs[0], ys[0]),
                    textcoords="offset points", xytext=(18, -6), fontsize=9,
                    color=C_ALERT, va="center")
    slope = _num(counter.get("drift_slope_ma_per_acquisition"))
    ax.set_title("(b) Le repos dérive tant que la session s'établit"
                 + (f"\npente {slope:+.3f} mA / acquisition" if slope is not None else ""),
                 fontsize=11)
    ax.set_xlabel("rang dans la session")
    ax.set_ylabel("courant au repos (mA)")
    _grid(ax)

    # (c) Randomisation de l'ordre.
    ax = axes[1][0]
    seq = (rate.get("session") or {}).get("sequence", [])
    if seq:
        ax.plot([s["session_index"] for s in seq], [_num(s["rate_hz"]) for s in seq],
                marker="o", linestyle="", color=C_ACCENT)
    ax.set_title("(c) L'ordre des cadences est tiré au sort\n"
                 "sans cela, la dérive (b) se lirait comme un effet de la cadence",
                 fontsize=11)
    ax.set_xlabel("rang dans la session")
    ax.set_ylabel("cadence mesurée à ce rang (Hz)")
    _grid(ax)

    # (d) Saturation silencieuse — visible sur le balayage de lot, à cadence fixe.
    ax = axes[1][1]
    batch = _load("experiments/exp_S53_wfi/batch_sweep.json")
    cell = (batch or {}).get("ewc_fp32") or {}
    pts = cell.get("points", [])
    asked = _num(cell.get("rate_hz"))
    if pts and asked is not None:
        x = np.arange(len(pts))
        got = [_num(p.get("achieved_rate_hz")) for p in pts]
        sat = [bool(p.get("saturated")) for p in pts]
        ax.axhline(asked, color=C_PROBE, linestyle="--", linewidth=2,
                   label=f"cadence demandée ({asked:.0f} Hz)")
        ax.bar(x, [v if v is not None else 0.0 for v in got], 0.55,
               color=[C_ALERT if sv else C_OK for sv in sat], edgecolor=INK,
               linewidth=0.5, label="cadence réellement atteinte")
        for xi, v, sv in zip(x, got, sat):
            if v is None:
                continue
            ax.text(xi, v, f"{v:.0f}" + ("\nécarté" if sv else ""), ha="center",
                    va="bottom", fontsize=9, color=C_ALERT if sv else INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={p['n']}" for p in pts])
        ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(d) L'UART sature sans rien dire\n"
                 "un point sous 95 % de la consigne est écarté", fontsize=11)
    ax.set_ylabel("cadence de trames (Hz)")
    ax.margins(y=0.25)
    _grid(ax)

    fig.subplots_adjust(hspace=0.55, wspace=0.28)
    fig.suptitle("Quatre pièges du banc — mesurés, puis neutralisés")
    fig.text(0.5, 0.005, BADGE_MEASURED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "d1_pieges_du_banc", out_root)


STATE_LABEL: dict[str, str] = {
    "port_ferme": "port hôte fermé",
    "port_ouvert_inactif": "port ouvert,\naucune trame",
    "flux": "flux d'inférence",
    "apres_flux": "après le flux,\nport refermé",
}


def _fig_d2_repos(out_root: Path) -> Path:
    """D2 — « repos » recouvre plusieurs états, et le firmware en définit un de plus."""
    states = _load("experiments/exp_S53_counterbalance/idle_states_power_cycle.json")
    idle = _load("experiments/exp_S53_wfi/idle_reference.json")
    if states is None or idle is None:
        return _empty("agrégats S53 absents", "d2_repos_nest_pas_repos", out_root)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    means = states.get("state_mean_a", {})
    stds = states.get("state_std_a", {})
    keys = [k for k in STATE_LABEL if k in means]
    x = np.arange(len(keys))
    vals = [(_num(means[k]) or 0.0) * MA_PER_A for k in keys]
    errs = [(_num(stds.get(k)) or 0.0) * MA_PER_A for k in keys]
    colors = [C_PROBE if k != "flux" else C_WARN for k in keys]
    ax.bar(x, vals, 0.6, yerr=errs, capsize=5, color=colors, edgecolor=INK,
           linewidth=0.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABEL[k] for k in keys], fontsize=9)
    ax.set_ylabel("courant moyen (mA)")
    ax.set_ylim(bottom=min(vals) * 0.99, top=max(vals) * 1.01)
    ax.set_title("Quatre états de l'hôte, firmware identique", fontsize=11)
    _grid(ax)
    ax.text(0.02, 0.97,
            textwrap.fill("l'écart entre « port fermé » et « après le flux » tient dans "
                          "la dispersion : ce n'est pas l'hôte qui expliquait l'anomalie "
                          "du Sprint 50", 44),
            transform=ax.transAxes, ha="left", va="top", fontsize=10, color=MUTED)

    ax = axes[1]
    builds = [("default_polling", "attente UART\nen scrutation", C_ALERT),
              ("UART_WFI_IDLE", "attente UART\nsous __WFI()", C_OK)]
    x = np.arange(len(builds))
    vals = [_num((idle.get(k) or {}).get("i_idle_established_ma")) for k, _, _ in builds]
    ax.bar(x, [v if v is not None else 0.0 for v in vals], 0.5,
           color=[c if v is not None else NA_GRAY for (_, _, c), v in zip(builds, vals)],
           edgecolor=INK, linewidth=0.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v if v is not None else 0.0,
                f"{v:.2f} mA" if v is not None else NA_TEXT, ha="center", va="bottom",
                fontsize=11, color=INK)
    gain = idle.get("wfi_gain") or {}
    g_ma, g_pct = _num(gain.get("gain_ma")), _num(gain.get("gain_pct"))
    if g_ma is not None and g_pct is not None:
        ax.text(0.5, 0.55, f"−{g_ma:.2f} mA\n(−{g_pct:.1f} %)", transform=ax.transAxes,
                ha="center", fontsize=14, color=C_ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in builds], fontsize=10)
    ax.set_ylabel("courant au repos (mA)")
    ax.set_title("Le firmware définit ce qu'est « ne rien faire »", fontsize=11)
    _grid(ax)

    fig.suptitle("« Repos » est une définition, pas un état")
    fig.text(0.5, 0.005, BADGE_MEASURED, ha="center", fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "d2_repos_nest_pas_repos", out_root)


# ══════════════════ E — Ce que la mesure permet de conclure ══════════════════

def _fig_e1_autonomie(out_root: Path) -> Path:
    """E1 — de la grandeur mesurée à la grandeur qui intéresse l'industriel."""
    auto = _load("experiments/exp_S53_wfi/autonomy_delta.json")
    if auto is None:
        return _empty("exp_S53_wfi/autonomy_delta.json absent",
                      "e1_de_la_mesure_a_lautonomie", out_root)
    idle_ma = _num(auto.get("i_idle_ma"))
    per_model = auto.get("per_model", {})

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.axis("off")
    stages = [
        ("courant mesuré\npar la sonde", C_PROBE),
        ("µJ par inférence\n(3 estimateurs)", C_ACCENT),
        ("courant moyen\nselon la période", C_WARN),
        ("autonomie\nselon la batterie", C_OK),
    ]
    for i, (text, color) in enumerate(stages):
        y = 0.85 - i * 0.22
        box(ax, 0.5, y, text, color, w=0.7, h=0.15, fontsize=11)
        if i:
            arrow(ax, 0.5, y + 0.135, 0.5, y + 0.078)
    ax.text(0.5, 0.02,
            "chaque flèche ajoute une hypothèse déclarée (période, capacité) —\n"
            "aucune n'est cachée dans le chiffre final",
            ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax = axes[1]
    periods = sorted(float(p) for p in auto.get("scenario_periods_s", []))
    caps = sorted(float(c) for c in auto.get("capacites_mah", []))
    # Capacité de référence du projet : la plus proche de la batterie citée en roadmap.
    capacity = min(caps, key=lambda c: abs(c - REF_CAPACITY_MAH)) if caps else None
    for key, color in (("ewc_fp32", C_PROBE), ("maha_fp32", C_ACCENT)):
        node = per_model.get(key)
        if node is None or capacity is None:
            continue
        by_period = node.get("autonomy_h_by_period", {})
        ys = [_num((by_period.get(str(p), {}).get("autonomy_h_by_mah", {}))
                   .get(str(capacity))) for p in periods]
        ax.plot(periods, ys, marker="o", color=color, label=CELL_LABEL.get(key, key))
        for idx in (0, len(periods) - 1):
            if ys[idx] is not None:
                ax.annotate(f"{ys[idx]:.0f} h", (periods[idx], ys[idx]),
                            textcoords="offset points", xytext=(0, 11), ha="center",
                            fontsize=10, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("période entre deux inférences (s) — échelle log")
    ax.set_ylabel(f"autonomie estimée (h) — batterie {capacity:.0f} mAh"
                  if capacity else "autonomie estimée (h)")
    ax.legend(fontsize=10, loc="upper left")
    ax.margins(y=0.25)
    ax.set_title("Plus on espace les inférences,\nplus le repos gouverne l'autonomie",
                 fontsize=11)
    _grid(ax)
    if idle_ma is not None:
        ax.text(0.97, 0.05,
                textwrap.fill(
                    f"Le plateau est fixé par le courant de repos ({idle_ma:.2f} mA) : "
                    f"au-delà d'une certaine période, optimiser le modèle ne change plus "
                    f"rien — c'est le sommeil qu'il faut travailler.", 44),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK)

    fig.suptitle("De la mesure à la décision — ce que la chaîne permet de conclure")
    fig.text(0.5, 0.005,
             BADGE_MIXED + " · " + str(auto.get("scope_mesure", "")), ha="center",
             fontsize=8, color=MUTED)
    return savefig_png(fig, CATALOG, "e1_de_la_mesure_a_lautonomie", out_root)


# ═════════════════════════════ Registre ══════════════════════════════════════

FIGURES: list[str] = [
    "a1_chaine_de_mesure",
    "a2_statique_vs_dynamique",
    "b1_probleme_isoler",
    "b2_trois_estimateurs",
    "c1_methode_delta",
    "c2_methode_regression",
    "c3_methode_lot",
    "d1_pieges_du_banc",
    "d2_repos_nest_pas_repos",
    "e1_de_la_mesure_a_lautonomie",
]


@register_catalog(CATALOG)
def build(out_root: Path) -> list[Path]:
    """Génère les 10 figures pédagogiques de la mesure d'énergie."""
    apply_style("slide")
    return [
        _fig_a1_chaine(out_root),
        _fig_a2_modes(out_root),
        _fig_b1_probleme(out_root),
        _fig_b2_estimateurs(out_root),
        _fig_c1_delta(out_root),
        _fig_c2_regression(out_root),
        _fig_c3_lot(out_root),
        _fig_d1_pieges(out_root),
        _fig_d2_repos(out_root),
        _fig_e1_autonomie(out_root),
    ]
