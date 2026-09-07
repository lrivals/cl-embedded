"""policy_energy.py — Énergie des politiques de mise à jour P0–P3 (S5306).

POURQUOI CE MODULE EXISTE, ET POURQUOI IL EST SÉPARÉ DU PILOTE :

Le Sprint 38 a mesuré sur carte réelle que le gate de nouveauté embarqué économise ~97 %
des mises à jour et 159–169 µs par échantillon, pour +300 B de RAM, à F1 préservé en mode
`pretrained`. C'est le résultat le plus « système » du projet — mais il n'existe **qu'en
latence et en RAM**. Or l'argument de déploiement d'un gate autonome est fondamentalement
énergétique : à quoi bon éviter 97 % des mises à jour si on ne sait pas ce qu'elles coûtent ?

Ce module porte la **règle de calcul et de décision** appliquée aux régressions de cadence
(S5304, `rate_regression`) mesurées politique par politique. Il ne mesure rien lui-même.

Le nœud méthodologique, et la raison pour laquelle cette tâche produit `energy_uj_per_update`
là où le Sprint 50 ne le pouvait pas :

    P0 (`frozen`) et P1 (`always`) partagent le MÊME binaire, le MÊME flux, la MÊME trame
    UART et la MÊME séance. Tout ce qui les sépare est le drapeau `PROTO_FLAG_UPDATE`,
    c'est-à-dire l'exécution — ou non — de la mise à jour CL. Leur différence de pente
    isole donc EXACTEMENT le coût d'une mise à jour, sans référence de repos (l'anomalie
    S5301 est absorbée par l'ordonnée à l'origine, commune aux deux) et sans instrumentation
    nouvelle. C'est le chemin le plus propre vers un champ N/A depuis le Sprint 50.

Trois issues, toutes publiables (spec S5306) :
    1. le gate économise significativement → argument de déploiement complet ;
    2. le gate est neutre → son surcoût permanent compense les mises à jour évitées ;
    3. le gate coûte plus qu'il ne rapporte → **résultat honnête**, qui borne le domaine de
       validité de la contribution du Sprint 38.

Le module est séparé de `scripts/run_s53_policy_energy.py` pour les deux raisons des
précédents `counterbalance.py` (S5301) et `rate_regression.py` (S5304) :
    1. la garde AST des tests interdit tout littéral flottant nouveau dans le pilote — les
       seuils de décision vivent donc ici, testables hors banc ;
    2. le verdict est ainsi **calculé et testé**, jamais saisi à la main dans un JSON.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : toute grandeur non séparable du bruit sort
``"à mesurer"`` avec sa raison CHIFFRÉE (la valeur relevée et sa barre d'erreur), jamais un
zéro, jamais une différence négative présentée comme une énergie. Les grandeurs du Sprint 38
ne sont JAMAIS recopiées : elles sont chargées depuis `experiments/exp_S38_summary.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation import autonomy as autonomy_mod
from src.evaluation.rate_regression import (
    A_MESURER,
    MA_PER_A,
    R2_MIN,
    SLOPE_SIGMA,
    UA_PER_A,
    UJ_PER_J,
)

# ── Cadrage des cellules (miroir strict du Sprint 38) ────────────────────────

#: Les quatre politiques, dans l'ordre de la spec. Miroir de
#: `scripts/run_sprint38_board.py:POLICIES`.
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")

#: Politiques dont la décision de mise à jour est prise À BORD par le gate.
GATED_POLICIES = ("gated_truelabel", "gated_pseudolabel")

#: Référence de la comparaison : la politique qui met à jour à CHAQUE échantillon.
REFERENCE_POLICY = "always"

#: Politique de base : aucune mise à jour. C'est le plancher énergétique du flux.
BASELINE_POLICY = "frozen"

#: Datasets du cadrage S38, et mode d'initialisation retenu (celui où le F1 est préservé —
#: c'est la configuration défendable, cf. spec S5306 §1).
DATASETS = ("monitoring", "pronostia")
INIT_MODE = "pretrained"
CONDITION = "5feat"

#: Drapeaux de compilation par politique — miroir de
#: `run_sprint38_board.build_and_flash_gated:127`. P0 et P1 partagent le build par DÉFAUT
#: (chaîne vide) : ils ne diffèrent que par le drapeau UART.
BUILD_BY_POLICY = {
    "frozen": "",
    "always": "",
    "gated_truelabel": "-DEWC_AUTO_UPDATE",
    "gated_pseudolabel": "-DEWC_AUTO_UPDATE -DGATE_PSEUDO_LABEL",
}

#: Nom court du binaire, tel que déclaré au banc (`--build`) et vérifié contre le manifeste
#: de flash. Deux politiques peuvent partager un binaire ; aucune ne peut être mesurée sur
#: un autre que le sien (leçon du Sprint 52 : un drapeau manquant fait mesurer, en silence,
#: un tout autre chemin d'exécution).
BUILD_NAME_BY_POLICY = {
    "frozen": "default",
    "always": "default",
    "gated_truelabel": "gate",
    "gated_pseudolabel": "gate_pseudo",
}

#: Seule `always` porte le drapeau UART `PROTO_FLAG_UPDATE` — miroir de
#: `run_sprint38_board.py:209` (`request_update = (policy == "always")`). Les politiques
#: gated streament SANS ce drapeau : le firmware décide seul.
UPDATE_FLAG_BY_POLICY = {policy: (policy == REFERENCE_POLICY) for policy in POLICIES}

# ── Seuils de la règle de décision (documentés, pas arbitraires) ─────────────

#: Nombre d'erreurs-types au-dessus de zéro exigé d'une DIFFÉRENCE de pentes pour qu'elle
#: soit publiée comme une énergie. Repris de `rate_regression.SLOPE_SIGMA` : la règle de
#: publication d'une différence ne doit pas être plus laxiste que celle d'une pente seule.
#: C'est ce test qui écarte, par construction, toute différence négative — le protocole
#: delta du Sprint 50 rendait des µJ négatifs, ce module refuse de les rapporter.
DELTA_SIGMA = SLOPE_SIGMA

#: Cadence de référence pour traduire une régression en courant moyen, donc en autonomie.
#: 100 Hz est la cadence de la campagne du Sprint 50 : l'autonomie de S5306 est ainsi
#: DIRECTEMENT comparable aux 8 cellules déjà mesurées, au lieu d'être une grandeur isolée.
AUTONOMY_RATE_HZ = 100.0

#: Facteur de tolérance du témoin de session : la dérive de la pente du témoin entre deux
#: séances est jugée acceptable si elle tient dans ce nombre d'écarts-types de sa propre
#: erreur-type. Au-delà, toute comparaison INTER-BUILD (P2/P3 contre P0) porte cette dérive,
#: et le JSON le dit. Même valeur que les autres tests de significativité du sprint, pour
#: qu'un seul seuil gouverne « séparable du bruit » dans toute la campagne.
ANCHOR_SIGMA = SLOPE_SIGMA

#: Écart relatif toléré entre le taux de mise à jour mesuré en séance et celui du
#: Sprint 38. 10 % : le gate est déterministe à seuils exportés identiques (parité verdict
#: 1.000 en S38), mais le flux de S5306 est rejoué à d'autres cadences et sur un nombre
#: d'échantillons différent — la queue du flux peut décaler le compteur de quelques unités.
UPDATE_RATE_TOLERANCE = 0.10

#: Nom du protocole. Il DOIT rester distinct de celui du delta (S5302) et de celui de
#: l'intégration par phase (S5305) : ce sont des estimateurs différents, dont la
#: comparaison est elle-même un résultat.
METHOD = "différence de pentes I(rate) entre politiques"

#: Réserve reconduite du Sprint 50 : le flux est CONTINU, la carte n'est jamais endormie.
#: L'autonomie publiée n'est donc pas une autonomie duty-cyclée, et ne doit pas être lue
#: comme telle.
REGIME_MESURE = ("flux continu à cadence imposée, carte jamais endormie — "
                 "ce n'est PAS une autonomie duty-cyclée")


# ── Accès aux cellules (tolérant : une cellule peut être N/A) ────────────────

def _slope_a_per_hz(cell: Mapping[str, Any]) -> float:
    """Pente d'une cellule, ramenée en A/Hz depuis les µA/Hz du JSON."""
    return float(cell["slope_ua_per_hz"]) / UA_PER_A


def _slope_std_a_per_hz(cell: Mapping[str, Any]) -> float:
    return float(cell["slope_std_ua_per_hz"]) / UA_PER_A


def _is_number(value: Any) -> bool:
    """Vrai si la valeur est un nombre exploitable (ni ``"à mesurer"``, ni None, ni bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: Alias public : les pilotes ont besoin de distinguer « mesuré » de ``"à mesurer"`` sans
#: réimplémenter le test (et sans importer un nom privé).
is_measured = _is_number


def _publiable(cell: Mapping[str, Any] | None) -> bool:
    """Une cellule est exploitable EN ABSOLU si sa régression a produit une énergie chiffrée.

    C'est la règle de `rate_regression.energy_uj_per_inference` : linéarité suffisante ET
    pente séparable de zéro à 2σ. Elle gouverne toute grandeur publiée pour une cellule
    prise SEULE ; pour une DIFFÉRENCE, voir :func:`_lineaire`.
    """
    return (cell is not None
            and _is_number(cell.get("slope_ua_per_hz"))
            and _is_number(cell.get("energy_uj_per_inference")))


def _lineaire(cell: Mapping[str, Any] | None) -> bool:
    """Condition d'entrée d'une cellule dans une DIFFÉRENCE de pentes (règle arrêtée A4).

    Exiger que chaque régression soit publiable INDIVIDUELLEMENT est plus strict que
    nécessaire ici : deux cellules d'une même paire partagent le binaire, la trame UART, la
    séance et l'ordonnée à l'origine, et le bruit commun s'annule dans l'écart. Une
    différence peut donc être séparable du bruit là où aucune des deux pentes ne l'est prise
    seule ; c'est précisément le régime où vit le coût d'une mise à jour CL.

    Ce qui reste exigé, en revanche, c'est la **linéarité** (`r² ≥ R2_MIN`) : sans elle une
    pente ne décrit aucun coût marginal, et la différence de deux non-coûts n'en décrit pas
    davantage. Le seul juge de la publication devient alors le test `Δ > DELTA_SIGMA · σ`,
    qui écarte toujours, par construction, les différences négatives.

    Règle arrêtée le 2026-09-07, AVANT la reprise des séances 2 et 3 de S5306 — jamais après
    avoir vu un résultat.
    """
    return (cell is not None
            and _is_number(cell.get("slope_ua_per_hz"))
            and _is_number(cell.get("slope_std_ua_per_hz"))
            and _is_number(cell.get("r2"))
            and float(cell["r2"]) >= R2_MIN)


def _raison_non_lineaire(cell: Mapping[str, Any] | None) -> str:
    """Raison CHIFFRÉE du refus d'une cellule dans une différence."""
    if cell is None:
        return "cellule absente"
    if not _is_number(cell.get("r2")):
        return "aucun r² relevé (régression absente)"
    return (f"r²={float(cell['r2']):.3f} < {R2_MIN} — la pente ne décrit pas un coût "
            f"marginal, sa différence non plus")


def _na(reason: str, **extra: Any) -> dict:
    """Bloc N/A honnête : la valeur littérale du projet ET sa raison chiffrée."""
    return {"value_uj": A_MESURER, "na_reason": reason, **extra}


def _tension(*cells: Mapping[str, Any] | None) -> float:
    """Tension d'alimentation commune aux cellules comparées.

    Lève `ValueError` si les cellules n'ont pas été mesurées à la même tension : une
    différence de pentes converties avec deux tensions différentes n'a pas de sens.
    """
    tensions = {round(float(c["tension_v"]), 6) for c in cells if c is not None}
    if len(tensions) != 1:
        raise ValueError(
            f"tensions d'alimentation hétérogènes entre les cellules comparées : "
            f"{sorted(tensions)} V — la différence de pentes n'est pas convertible."
        )
    return tensions.pop()


# ── 1. Énergie d'une mise à jour CL : la différence P1 − P0 ──────────────────

def update_energy_uj(cell_always: Mapping[str, Any] | None,
                     cell_frozen: Mapping[str, Any] | None) -> dict:
    """`E_update = (pente(always) − pente(frozen)) × V`, en µJ — ou N/A honnête.

    Pourquoi cette différence isole la mise à jour, et rien d'autre : les deux cellules
    partagent le binaire, le dataset, la trame UART (55 B), la séance et l'ordonnée à
    l'origine. Le coût de trame — 60,6 µJ au pilote S5304, soit l'essentiel de la pente
    d'une cellule rapide — se soustrait exactement. Ce qui reste est la mise à jour.

    Trois refus de publication, tous chiffrés :
        - une des deux régressions n'est pas LINÉAIRE (r² insuffisant) — une pente prise
          seule n'a en revanche pas à être significative : le bruit commun aux deux
          cellules s'annule dans l'écart (règle A4, cf. :func:`_lineaire`) ;
        - la différence ne dépasse pas `DELTA_SIGMA` fois son incertitude propagée
          (ce qui écarte aussi toute différence négative) ;
        - les tensions d'alimentation diffèrent.

    Returns
    -------
    dict
        ``{"value_uj", "uncertainty_uj", "delta_slope_ua_per_hz", "significant",
        "method", "tension_v"}`` — `value_uj` valant ``"à mesurer"`` + `na_reason` si la
        différence n'est pas séparable du bruit.
    """
    manquantes = [name for name, cell in (("always", cell_always), ("frozen", cell_frozen))
                  if not _lineaire(cell)]
    if manquantes:
        raisons = "; ".join(
            f"{name} : {_raison_non_lineaire(cell)}"
            for name, cell in (("always", cell_always), ("frozen", cell_frozen))
            if name in manquantes
        )
        return _na(
            f"différence non calculable : la régression de {', '.join(manquantes)} n'est "
            f"pas linéaire ({raisons}).",
            method=METHOD,
        )

    tension_v = _tension(cell_always, cell_frozen)
    delta = _slope_a_per_hz(cell_always) - _slope_a_per_hz(cell_frozen)
    sigma = math.hypot(_slope_std_a_per_hz(cell_always), _slope_std_a_per_hz(cell_frozen))
    significant = delta > DELTA_SIGMA * sigma

    bloc = {
        "delta_slope_ua_per_hz": delta * UA_PER_A,
        "delta_slope_std_ua_per_hz": sigma * UA_PER_A,
        "uncertainty_uj": sigma * tension_v * UJ_PER_J,
        "significant": significant,
        "tension_v": tension_v,
        "method": METHOD,
    }
    if not significant:
        bloc.update(_na(
            f"différence non significative : Δpente={delta * UA_PER_A:+.3f} ± "
            f"{sigma * UA_PER_A:.3f} µA/Hz, soit moins de {DELTA_SIGMA:g} σ au-dessus de "
            f"zéro — le coût d'une mise à jour n'est pas séparable du bruit du banc "
            f"(valeur relevée : {delta * tension_v * UJ_PER_J:+.2f} µJ).",
            method=METHOD,
        ))
        return bloc
    bloc["value_uj"] = delta * tension_v * UJ_PER_J
    return bloc


# ── 2. Surcoût PERMANENT du gate (critère d'acceptation n° 5) ────────────────

def gate_energy_overhead_uj(cell_gated: Mapping[str, Any] | None,
                            cell_frozen: Mapping[str, Any] | None,
                            update_rate: float | None,
                            update_energy: Mapping[str, Any] | None) -> dict:
    """Coût énergétique du gate LUI-MÊME, par échantillon — ou N/A honnête.

    Le gate coûte ~27 µs par échantillon (S38) sur **tous** les échantillons, alors qu'il
    n'évite une mise à jour que sur les ~97,5 % qu'il ne déclenche pas. La question n'est
    donc pas « le gate économise-t-il des mises à jour » (S38 l'a mesuré) mais « son
    surcoût permanent mange-t-il l'économie qu'il procure ». D'où la décomposition :

        (pente(gated) − pente(frozen)) × V  =  surcoût_gate  +  update_rate × E_update
                                               ^^^^^^^^^^^^
                                               ce qu'on isole ici

    L'incertitude est propagée sur les trois termes (les deux pentes et `E_update`).
    Le `update_rate` doit être MESURÉ (compteur du gate en séance ou S38 rechargé), jamais
    supposé.
    """
    if not (_lineaire(cell_gated) and _lineaire(cell_frozen)):
        raisons = "; ".join(f"{nom} : {_raison_non_lineaire(cell)}"
                            for nom, cell in (("gated", cell_gated), ("frozen", cell_frozen))
                            if not _lineaire(cell))
        return _na(f"surcoût du gate non calculable : une des deux régressions "
                   f"(gated, frozen) n'est pas linéaire ({raisons}).", method=METHOD)
    if not _is_number(update_rate):
        return _na("surcoût du gate non calculable : taux de mise à jour non mesuré.",
                   method=METHOD)
    if update_energy is None or not _is_number(update_energy.get("value_uj")):
        return _na(
            "surcoût du gate non calculable : l'énergie d'une mise à jour est "
            f"« {A_MESURER} » — la part imputable aux mises à jour réellement effectuées "
            "ne peut pas être retranchée.",
            method=METHOD,
        )

    tension_v = _tension(cell_gated, cell_frozen)
    delta = _slope_a_per_hz(cell_gated) - _slope_a_per_hz(cell_frozen)
    delta_uj = delta * tension_v * UJ_PER_J
    part_maj_uj = float(update_rate) * float(update_energy["value_uj"])
    surcout_uj = delta_uj - part_maj_uj

    sigma_delta_uj = math.hypot(_slope_std_a_per_hz(cell_gated),
                                _slope_std_a_per_hz(cell_frozen)) * tension_v * UJ_PER_J
    sigma_part_uj = float(update_rate) * float(update_energy.get("uncertainty_uj", 0.0))
    sigma_uj = math.hypot(sigma_delta_uj, sigma_part_uj)

    return {
        "value_uj": surcout_uj,
        "uncertainty_uj": sigma_uj,
        "significant": abs(surcout_uj) > DELTA_SIGMA * sigma_uj,
        "delta_vs_frozen_uj": delta_uj,
        "updates_share_uj": part_maj_uj,
        "update_rate": float(update_rate),
        "tension_v": tension_v,
        "method": METHOD,
    }


# ── 3. Économie par rapport à `always` ───────────────────────────────────────

def energy_saved_vs_always(cell_policy: Mapping[str, Any] | None,
                           cell_always: Mapping[str, Any] | None,
                           cell_frozen: Mapping[str, Any] | None) -> dict:
    """Économie d'une politique face à `always`, en µJ par échantillon ET en %.

    Le pourcentage est rapporté à l'énergie **marginale** de `always`, c'est-à-dire à
    `pente(always) − pente(frozen)` : le dénominateur est ce que la mise à jour ajoute au
    flux, pas la consommation totale de la carte (dont l'essentiel — trame UART, scrutation
    active, PHY — ne dépend pas de la politique). Le dénominateur est écrit dans le JSON
    pour que le pourcentage soit lisible sans ambiguïté.
    """
    if not all(_lineaire(c) for c in (cell_policy, cell_always, cell_frozen)):
        raisons = "; ".join(
            f"{nom} : {_raison_non_lineaire(cell)}"
            for nom, cell in (("politique", cell_policy), ("always", cell_always),
                              ("frozen", cell_frozen))
            if not _lineaire(cell))
        return {"saved_uj_per_sample": A_MESURER,
                "saved_pct_of_always_marginal": A_MESURER,
                "na_reason": f"économie non calculable : une des trois régressions "
                             f"(politique, always, frozen) n'est pas linéaire ({raisons}).",
                "method": METHOD}

    tension_v = _tension(cell_policy, cell_always, cell_frozen)
    marginal_always = (_slope_a_per_hz(cell_always) - _slope_a_per_hz(cell_frozen))
    marginal_policy = (_slope_a_per_hz(cell_policy) - _slope_a_per_hz(cell_frozen))
    saved_uj = (marginal_always - marginal_policy) * tension_v * UJ_PER_J
    sigma_uj = math.hypot(_slope_std_a_per_hz(cell_always),
                          _slope_std_a_per_hz(cell_policy)) * tension_v * UJ_PER_J

    denominateur_uj = marginal_always * tension_v * UJ_PER_J
    if denominateur_uj == 0.0:
        pct: float | str = A_MESURER
    else:
        pct = saved_uj / denominateur_uj * 100.0

    return {
        "saved_uj_per_sample": saved_uj,
        "uncertainty_uj": sigma_uj,
        "saved_pct_of_always_marginal": pct,
        "denominator_uj_always_marginal": denominateur_uj,
        "denominator_note": ("pourcentage rapporté à l'énergie MARGINALE de `always` "
                             "(pente(always) − pente(frozen)), pas à la consommation "
                             "totale de la carte"),
        "significant": abs(saved_uj) > DELTA_SIGMA * sigma_uj,
        "method": METHOD,
    }


# ── 4. Verdict : les trois issues de la spec ─────────────────────────────────

def gate_verdict(overhead: Mapping[str, Any] | None,
                 saved: Mapping[str, Any] | None) -> dict:
    """Le gate économise-t-il, est-il neutre, ou coûte-t-il plus qu'il ne rapporte ?

    Les trois issues sont publiables (spec S5306). La troisième borne le domaine de
    validité de la contribution du Sprint 38 : à faible taux de dérive, un gate dont le
    surcoût est permanent peut coûter plus que les mises à jour qu'il évite.
    """
    if (overhead is None or saved is None
            or not _is_number(overhead.get("value_uj"))
            or not _is_number(saved.get("saved_uj_per_sample"))):
        raison = ((overhead or {}).get("na_reason")
                  or (saved or {}).get("na_reason")
                  or "grandeurs manquantes")
        return {"verdict": A_MESURER,
                "rationale": f"verdict non prononçable : {raison}",
                "method": METHOD}

    net_uj = float(saved["saved_uj_per_sample"]) - float(overhead["value_uj"])
    sigma_uj = math.hypot(float(saved.get("uncertainty_uj", 0.0)),
                          float(overhead.get("uncertainty_uj", 0.0)))
    if abs(net_uj) <= DELTA_SIGMA * sigma_uj:
        verdict = "gate_neutre"
        rationale = (
            f"bilan net {net_uj:+.2f} ± {sigma_uj:.2f} µJ/échantillon : le surcoût "
            f"permanent du gate ({float(overhead['value_uj']):+.2f} µJ) compense, au bruit "
            f"près, l'économie de mises à jour ({float(saved['saved_uj_per_sample']):+.2f} "
            f"µJ). La justification du gate reste la latence pire-cas et l'autonomie de "
            f"décision, pas l'énergie."
        )
    elif net_uj > 0:
        verdict = "gate_economise"
        rationale = (
            f"bilan net {net_uj:+.2f} ± {sigma_uj:.2f} µJ/échantillon en faveur du gate : "
            f"l'économie de mises à jour ({float(saved['saved_uj_per_sample']):+.2f} µJ) "
            f"dépasse son surcoût permanent ({float(overhead['value_uj']):+.2f} µJ)."
        )
    else:
        verdict = "gate_plus_couteux"
        rationale = (
            f"bilan net {net_uj:+.2f} ± {sigma_uj:.2f} µJ/échantillon en défaveur du "
            f"gate : son surcoût permanent ({float(overhead['value_uj']):+.2f} µJ), payé "
            f"sur TOUS les échantillons, dépasse l'économie procurée "
            f"({float(saved['saved_uj_per_sample']):+.2f} µJ) à ce taux de dérive. "
            f"Résultat honnête : il borne le domaine de validité du gate du Sprint 38."
        )
    return {"verdict": verdict, "rationale": rationale,
            "net_uj_per_sample": net_uj, "uncertainty_uj": sigma_uj, "method": METHOD}


# ── 5. Autonomie par politique ───────────────────────────────────────────────

def policy_autonomy(cell: Mapping[str, Any] | None,
                    capacites_mah: Sequence[float],
                    rate_hz: float = AUTONOMY_RATE_HZ) -> dict:
    """Courant moyen prédit par la régression, puis autonomie sur les capacités FOURNIES.

    `I_moy = I_base + pente · rate` : c'est la traduction en heures qui parle à un
    industriel, et elle est directement comparable aux 8 cellules du Sprint 50 quand
    `rate_hz` vaut la cadence de cette campagne.

    Les capacités ne sont JAMAIS en dur : elles viennent de
    `configs/hw_profile_f439zi.yaml:batterie` via :func:`load_capacities`.

    `duty_cycle` est explicitement `None` et `regime_mesure` documente pourquoi : le flux
    est continu, la carte n'est jamais endormie (réserve du Sprint 50, reconduite).
    """
    if cell is None or not _is_number(cell.get("intercept_ma")) \
            or not _is_number(cell.get("slope_ua_per_hz")):
        return {"i_moy_ma": A_MESURER,
                "na_reason": "autonomie non calculable : régression absente ou non chiffrée.",
                "duty_cycle": None, "regime_mesure": REGIME_MESURE}

    i_moy_ma = float(cell["intercept_ma"]) + _slope_a_per_hz(cell) * float(rate_hz) * MA_PER_A
    if i_moy_ma <= 0:
        return {"i_moy_ma": A_MESURER,
                "na_reason": f"autonomie non calculable : courant moyen prédit "
                             f"{i_moy_ma:.3f} mA ≤ 0 à {rate_hz:g} Hz.",
                "duty_cycle": None, "regime_mesure": REGIME_MESURE}

    heures = autonomy_mod.sweep_capacities(i_moy_ma, list(capacites_mah))
    return {
        "i_moy_ma": i_moy_ma,
        "rate_hz": float(rate_hz),
        "autonomy_hours": {str(cap): h for cap, h in heures.items()},
        "duty_cycle": None,
        "regime_mesure": REGIME_MESURE,
    }


def load_capacities(hw_profile: str | Path) -> list[float]:
    """Capacités batterie du profil HW — jamais en dur (délègue à `autonomy`)."""
    return autonomy_mod.load_battery_capacities(hw_profile)


# ── 6. Témoin de session (dérive inter-builds) ───────────────────────────────

def anchor_drift(anchors: Mapping[str, Mapping[str, Any]]) -> dict:
    """La pente du témoin varie-t-elle entre les séances ? (constat, pas supposition)

    P0/P1 (binaire par défaut) et P2/P3 (binaires gate) sont nécessairement mesurés dans
    des séances séparées par un reflash. Une dérive de banc entre séances se confondrait
    alors avec l'effet du gate. Le témoin — une cellule dont le chemin d'exécution est
    inchangé par `-DEWC_AUTO_UPDATE` — est rebalayé à chaque séance : si sa pente bouge,
    l'écart P2−P0 porte cette dérive, et le JSON le dit.

    `energy_uj_per_update` (P1 − P0) est INTRA-séance et n'est donc pas concerné.
    """
    mesures = {name: a for name, a in anchors.items()
               if _is_number((a or {}).get("slope_ua_per_hz"))}
    if len(mesures) < 2:
        return {"comparable": None,
                "na_reason": "dérive non évaluable : moins de deux séances portent un "
                             "témoin mesuré.",
                "sessions": sorted(anchors)}

    pentes = {name: float(a["slope_ua_per_hz"]) for name, a in mesures.items()}
    sigmas = {name: float(a.get("slope_std_ua_per_hz") or 0.0) for name, a in mesures.items()}
    lo, hi = min(pentes, key=pentes.get), max(pentes, key=pentes.get)
    spread = pentes[hi] - pentes[lo]
    sigma = math.hypot(sigmas[hi], sigmas[lo])
    comparable = spread <= ANCHOR_SIGMA * sigma

    return {
        "comparable": comparable,
        "spread_ua_per_hz": spread,
        "spread_std_ua_per_hz": sigma,
        "slopes_ua_per_hz": pentes,
        "sessions": sorted(anchors),
        "rationale": (
            f"pente du témoin : {pentes[lo]:.3f} ({lo}) → {pentes[hi]:.3f} ({hi}) µA/Hz, "
            f"écart {spread:+.3f} ± {sigma:.3f}. "
            + ("Les séances sont comparables au bruit près : les écarts inter-builds "
               "(gated contre frozen) sont interprétables."
               if comparable else
               "Les séances NE sont PAS comparables au bruit près : tout écart "
               "inter-builds (gated contre frozen) porte cette dérive de banc, qui doit "
               "être retranchée de son interprétation.")
        ),
    }


# ── 7. Croisement avec les mesures du Sprint 38 (lecture seule) ──────────────

def load_s38_summary(path: str | Path) -> dict | None:
    """Charge `exp_S38_summary.json` en LECTURE SEULE, ou `None` s'il est absent."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cross_table(cells: Mapping[str, Mapping[str, Any]],
                s38: Mapping[str, Any] | None,
                energies: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict]:
    """Le tableau à quatre colonnes que le mémoire n'a pas (spec S5306 §3).

    Une ligne par (politique, dataset) : ce que le Sprint 38 a mesuré (mises à jour
    économisées, µs économisées, F1, RAM) confronté à ce que le Sprint 53 mesure (mA,
    µJ, autonomie).

    **Aucune valeur S38 n'est recopiée** : elles sont lues dans `exp_S38_summary.json`
    (`results[dataset][init][policy]["board"]` et `results[dataset][init]["economy_table"]`).
    Sans ce fichier, les colonnes S38 sortent ``"à mesurer"`` avec leur raison.
    """
    energies = energies or {}
    rows: list[dict] = []
    for dataset in DATASETS:
        for policy in POLICIES:
            key = f"{policy}_{dataset}"
            cell = cells.get(key)
            row: dict[str, Any] = {
                "policy": policy,
                "dataset": dataset,
                "init_mode": INIT_MODE,
                # ── colonnes Sprint 53 (mesurées ici) ──
                "slope_ua_per_hz": (cell or {}).get("slope_ua_per_hz", A_MESURER),
                "energy_uj_per_sample": (cell or {}).get("energy_uj_per_inference", A_MESURER),
                "i_moy_ma": (energies.get(key, {}).get("autonomy", {})
                             .get("i_moy_ma", A_MESURER)),
                "autonomy_hours": (energies.get(key, {}).get("autonomy", {})
                                   .get("autonomy_hours", A_MESURER)),
            }
            row.update(_s38_columns(s38, dataset, policy))
            rows.append(row)
    return rows


def _s38_columns(s38: Mapping[str, Any] | None, dataset: str, policy: str) -> dict:
    """Colonnes chargées du Sprint 38 — jamais saisies."""
    absent = {
        "updates_saved_pct_s38": A_MESURER,
        "latency_saved_us_s38": A_MESURER,
        "update_rate_s38": A_MESURER,
        "f1_faulty_s38": A_MESURER,
        "ram_added_bytes_s38": A_MESURER,
        "s38_na_reason": "experiments/exp_S38_summary.json absent ou illisible — les "
                         "colonnes du Sprint 38 sont chargées, jamais recopiées.",
    }
    if s38 is None:
        return absent
    try:
        bloc = s38["results"][dataset][INIT_MODE]
        board = bloc[policy]["board"]
        economy = bloc["economy_table"][policy]
    except (KeyError, TypeError):
        return {**absent,
                "s38_na_reason": f"cellule ({dataset}, {INIT_MODE}, {policy}) absente de "
                                 f"exp_S38_summary.json."}
    return {
        "updates_saved_pct_s38": economy.get("updates_saved_pct"),
        "latency_saved_us_s38": economy.get("latency_saved_us"),
        "update_rate_s38": board.get("update_rate"),
        "f1_faulty_s38": board.get("f1_faulty"),
        "ram_added_bytes_s38": economy.get("ram_added_bytes"),
    }


def s38_update_rate(s38: Mapping[str, Any] | None, dataset: str, policy: str) -> float | None:
    """Taux de mise à jour du Sprint 38, chargé — sert de dénominateur au surcoût du gate."""
    if s38 is None:
        return None
    try:
        rate = s38["results"][dataset][INIT_MODE][policy]["board"]["update_rate"]
    except (KeyError, TypeError):
        return None
    return float(rate) if _is_number(rate) else None


def update_rate_agreement(measured: float | None, s38_rate: float | None) -> dict:
    """Le taux mesuré en séance concorde-t-il avec celui du Sprint 38 ?

    Critère d'acceptation n° 1 de la spec. La concordance n'est pas décrétée : elle est
    calculée, et son écart relatif est écrit. Un désaccord n'invalide pas la mesure
    d'énergie — il signale que le flux rejoué n'a pas déclenché le gate au même rythme,
    ce qui doit être visible.
    """
    if not _is_number(measured) or not _is_number(s38_rate):
        return {"agreement": None,
                "na_reason": "concordance non évaluable : un des deux taux de mise à jour "
                             "n'est pas disponible.",
                "measured": measured, "s38": s38_rate}
    ecart = abs(float(measured) - float(s38_rate))
    relatif = (ecart / float(s38_rate)) if float(s38_rate) != 0 else None
    return {"agreement": relatif is not None and relatif <= UPDATE_RATE_TOLERANCE,
            "measured": float(measured), "s38": float(s38_rate),
            "absolute_gap": ecart, "relative_gap": relatif,
            "tolerance": UPDATE_RATE_TOLERANCE}


def ordering_check(cells: Mapping[str, Mapping[str, Any]], dataset: str) -> dict:
    """Contrôle d'ordre du critère d'acceptation n° 1 : frozen < gated < always.

    Sur les taux de mise à jour, l'ordre est structurel (0 < ~0,025 < 1). Sur les pentes
    énergétiques, il ne l'est PAS : c'est justement la question de la tâche. Ce contrôle
    rapporte donc l'ordre CONSTATÉ, sans le postuler.
    """
    pentes = {}
    for policy in POLICIES:
        cell = cells.get(f"{policy}_{dataset}")
        if _publiable(cell):
            pentes[policy] = float(cell["slope_ua_per_hz"])
    if len(pentes) < 2:
        return {"ordered": None,
                "na_reason": "ordre non évaluable : moins de deux politiques publiables "
                             f"sur {dataset}.",
                "slopes_ua_per_hz": pentes}
    attendu = [p for p in (BASELINE_POLICY, *GATED_POLICIES, REFERENCE_POLICY) if p in pentes]
    constate = sorted(pentes, key=pentes.get)
    return {"ordered": attendu == constate,
            "expected_order": attendu,
            "observed_order": constate,
            "slopes_ua_per_hz": pentes,
            "note": ("l'ordre frozen < gated < always est structurel sur le NOMBRE de "
                     "mises à jour ; sur l'ÉNERGIE il est constaté, jamais postulé.")}
