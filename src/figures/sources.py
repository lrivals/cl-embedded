"""Accès aux sources d'expériences pour les catalogues de figures.

Ce module ne dessine rien : il **localise et lit** les JSON d'expérience, et c'est
tout. Il existe pour qu'une même grandeur ne soit jamais lue par deux chemins de code
différents. Le manuscrit et la soutenance annoncent les mêmes chiffres ; s'ils les
chargeaient chacun de leur côté, une expérience relancée pourrait les faire diverger
sans que rien ne le signale. Un chargeur unique rend cette divergence impossible.

Règle d'honnêteté héritée de :mod:`src.figures.loaders` : une donnée absente renvoie
``None``, jamais ``0``. C'est à la figure de rendre le ``None`` visible (case grise,
mention « N/A », panneau explicite).

Les accesseurs sont **paramétrés** (condition, plateforme, jeu) plutôt que câblés sur
le périmètre d'un catalogue : chaque catalogue garde ses propres listes de modèles,
de jeux et d'exclusions.
"""

from __future__ import annotations

import yaml

from src.figures.loaders import ROOT, load_experiment, record_source

# Condition de features par défaut des campagnes board (Sprint 35).
DEFAULT_CONDITION = "5feat"


# ── Primitives ───────────────────────────────────────────────────────────────

def try_load(path: str) -> dict | None:
    """Charge un JSON d'expérience, ou ``None`` s'il n'existe pas (jamais de défaut)."""
    try:
        data, _ = load_experiment(path)
        return data
    except FileNotFoundError:
        return None


def num(value) -> float | None:
    """Valeur numérique exploitable, sinon ``None`` (jamais 0 par défaut).

    Les booléens sont rejetés : en Python ``True`` est un ``int``, et un drapeau
    tracé comme une hauteur de barre serait une valeur inventée.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def first_num(data: dict | None, *keys: str) -> float | None:
    """Premier des ``keys`` portant une valeur numérique, sinon ``None``.

    Utile là où deux campagnes nomment différemment la même grandeur (par exemple
    ``acc_final`` côté PC et ``online_accuracy`` côté carte).
    """
    if not data:
        return None
    for key in keys:
        value = num(data.get(key))
        if value is not None:
            return value
    return None


# ── Sprint 35 — grille modèles × jeux × plateformes ──────────────────────────

def s35_cell(
    model: str,
    dataset: str,
    platform: str,
    condition: str = DEFAULT_CONDITION,
    exclude: frozenset[str] = frozenset(),
) -> dict | None:
    """Cellule de la grille S35 — ``platform`` ∈ {``PC``, ``board``}.

    ``exclude`` liste les modèles dont les cellules **carte** sont connues invalides :
    elles renvoient ``None`` (donc « N/A » à l'affichage) au lieu d'être présentées
    comme des mesures.
    """
    if platform == "board" and model in exclude:
        return None
    return try_load(f"experiments/exp_S35_{platform}_{condition}_{model}_{dataset}/results.json")


def s35_f1(model: str, dataset: str, platform: str, **kwargs) -> float | None:
    """F1 de la classe « fautif » pour une cellule de la grille S35."""
    return first_num(s35_cell(model, dataset, platform, **kwargs), "f1_faulty")


def s35_acc(model: str, dataset: str, platform: str, **kwargs) -> float | None:
    """Accuracy d'une cellule S35 (``acc_final`` PC, ``online_accuracy`` carte)."""
    return first_num(
        s35_cell(model, dataset, platform, **kwargs), "acc_final", "online_accuracy"
    )


def s35_sweep() -> list | None:
    """Résumé du balayage de features S35 (liste de cellules carte)."""
    data = try_load("experiments/exp_S35_board_sweep_summary.json")
    return data if isinstance(data, list) else None


# ── Sprint 36 — comparaison appariée PC ↔ carte, gelé vs en ligne ────────────

def s36_node(
    dataset: str, platform: str, condition: str = DEFAULT_CONDITION
) -> dict | None:
    """Nœud ``results.<jeu>.<condition>.<plateforme>`` du résumé S36.

    ``platform`` ∈ {``pc``, ``board_frozen``, ``board_online``,
    ``board_frozen_int8``, ``board_online_int8``, ``delta_pc_board``}.
    """
    summary = try_load("experiments/exp_S36_summary.json")
    if summary is None:
        return None
    return summary.get("results", {}).get(dataset, {}).get(condition, {}).get(platform)


def s36_parity(
    dataset: str, protocol: str, condition: str = DEFAULT_CONDITION
) -> dict | None:
    """Fichier de parité appariée S36 — ``protocol`` ∈ {``frozen``, ``online``}.

    Porte le détail par échantillon (``rows``) qui permet de tracer les désaccords
    à leur position réelle plutôt qu'un taux agrégé.
    """
    return try_load(
        f"experiments/exp_S36_parity_{condition}_{protocol}_{dataset}.json"
    )


# ── Sprint 49 — RAM totale mesurée (.data + .bss + pic de pile) ──────────────

def s49_board(
    model: str,
    dataset: str,
    encoding: str = "fp32",
    condition: str = DEFAULT_CONDITION,
) -> dict | None:
    """Cellule RAM carte S49, uniquement si la mesure a abouti (``status == "done"``)."""
    summary = try_load("experiments/exp_S49_ram/summary.json")
    if summary is None:
        return None
    cell = (
        summary.get(model, {})
        .get(dataset, {})
        .get(condition, {})
        .get(encoding, {})
        .get("board")
    )
    return cell if cell and cell.get("status") == "done" else None


# ── Autres campagnes ─────────────────────────────────────────────────────────

def s48_summary() -> dict | None:
    """Résumé sub-INT8 S48 — porte aussi ``bss_default_invariant`` et le budget carte."""
    return try_load("experiments/exp_S48_summary.json")


def s40_board_v2(dataset: str) -> dict | None:
    """Cellule carte du noyau INT8 v2 calibré (S40), régime gelé."""
    return try_load(
        f"experiments/exp_S40_board_v2/results_per_channel_{dataset}_frozen.json"
    )


def s50_int8_latency() -> dict | None:
    """Décomposition cycle par cycle du noyau INT8 mesurée sur carte (S50)."""
    return try_load("experiments/exp_S50_int8_latency/ewc.json")


def s26_multiclass() -> dict | None:
    """Bilan EWC multiclasse CWRU (S26) — F1 post-tâche vs F1 du modèle final."""
    return try_load("experiments/exp_S26_02/results.json")


def s54_forgetting(arm: str) -> dict | None:
    """Trajectoire d'oubli mesurée — ``arm`` ∈ {``ewc``, ``naive``}.

    Porte ``task_f1_per_epoch[epoch][task]`` : le F1-macro relevé sur **chaque tâche
    déjà vue** après chaque époque, ``None`` tant que la tâche n'a pas été rencontrée.
    """
    return try_load(f"experiments/exp_S54_forgetting_{arm}/results.json")


# ── Configuration matérielle (YAML, pas une expérience) ─────────────────────

def hw_profile() -> dict | None:
    """Profil matériel de la carte (``configs/hw_profile_f439zi.yaml``), ou ``None``.

    Passe par le même enregistrement de provenance que les JSON d'expérience : la
    fiche technique de la carte est une source citable au même titre qu'une mesure.
    """
    path = ROOT / "configs" / "hw_profile_f439zi.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    record_source(path)
    return data if isinstance(data, dict) else None


def firmware_ram_total_bytes() -> float | None:
    """Taille SRAM totale déclarée par le firmware (``inc/hw_info.h``), en octets.

    Lue depuis le commentaire de constante du header plutôt que réécrite ici : la
    valeur qui fait foi est celle que la carte utilise réellement.
    """
    header = ROOT / "firmware" / "stm32f4_blink" / "inc" / "hw_info.h"
    if not header.exists():
        return None
    import re

    match = re.search(r"(\d+)\s*B\s*\*/", header.read_text(encoding="utf-8"))
    if match:
        return float(match.group(1))
    match = re.search(r"=\s*(\d{6,})", header.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


# ── Comparaison consolidée Gap 1 (grille complète 5 jeux × 4 modèles) ────────

def comparison_sprint23() -> dict | None:
    """Comparaison consolidée `experiments/comparison_sprint23.json`, ou ``None``.

    Produite par ``scripts/generate_comparison_sprint23.py`` **à partir des JSON
    d'expérience** : ne jamais l'éditer à la main, la régénérer. Elle porte
    ``results_by_condition[condition][jeu][modèle][plateforme]``, la seule source qui
    couvre les 5 jeux × 4 modèles × 2 plateformes sous les 3 conditions de variables.
    """
    return try_load("experiments/comparison_sprint23.json")


def s35_condition_cell(
    condition: str, dataset: str, model: str, platform: str
) -> dict | None:
    """Cellule de la grille consolidée — ``platform`` ∈ {``pc``, ``nucleo_f439zi``}.

    Renvoie ``None`` si la cellule n'a pas été mesurée : le producteur y laisse
    ``{"acc_final": None, "f1_faulty": None, "note": "pending"}``, jamais un 0.
    """
    data = comparison_sprint23()
    if data is None:
        return None
    cell = (
        data.get("results_by_condition", {})
        .get(condition, {})
        .get(dataset, {})
        .get(model, {})
        .get(platform)
    )
    if not cell or cell.get("note") == "pending":
        return None
    return cell


def s35_condition_f1(condition: str, dataset: str, model: str, platform: str) -> float | None:
    """F1 « fautif » d'une cellule de la grille consolidée."""
    return first_num(s35_condition_cell(condition, dataset, model, platform), "f1_faulty")
