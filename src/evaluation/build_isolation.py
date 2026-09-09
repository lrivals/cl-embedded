"""build_isolation.py — Isolation à variable unique de l'effet « build » (S5306, B3/B8).

POURQUOI CE MODULE EXISTE

Le balayage de cadence (S5304) mesure un coût marginal par inférence en ajustant
`I_moy = I_base + pente · cadence`. Sur le build stock (dimensions par défaut, `EWC_IN=5`)
la droite est nette (pente +20,73 ± 0,21 µA/Hz, r² 0,9996). Sur le build aux dimensions du
Sprint 38 (`EWC_IN=MAHA_DIM=TINYOL_IN=HDC_N_FEATURES=4`), elle **disparaît** : la
consommation cesse de dépendre de la cadence, et toute grandeur qui se déduit d'une pente
— dont `energy_uj_per_update` (S5306) — devient incalculable.

B3 (2026-09-07) a tenu la dimension à k=4 et fait varier les seuls poids de la tête EWC :
les deux variantes sont plates. **Les poids sont écartés comme cause.** Le candidat restant
est le jeu de DIMENSIONS du binaire, que B8 met à l'épreuve dans une séance unique, à poids
non chargés des deux côtés (repli Xavier), en ne faisant varier que `k`.

Ce que la paire isole réellement, et pourquoi elle est propre : `sensor_stream.py --dataset
monitoring` envoie **4 features quelle que soit la dimension compilée** (`sensor_sim.py`,
`_load_monitoring`). Les deux binaires reçoivent donc la MÊME trame de 27 octets, à la même
cadence, avec la même tête non chargée. La longueur de trame et le contenu des poids sont
écartés **par construction**, pas par hypothèse : seule la dimension compilée varie.

CE QUE CE MODULE NE FAIT PAS : il ne mesure rien et ne publie pas la différence de pentes
comme une ÉNERGIE. La règle A4 (`policy_energy._lineaire`) exige la linéarité de chaque
cellule d'une différence, et c'est précisément ce qui manque ici — une cellule plate est le
PHÉNOMÈNE cherché, pas un défaut de mesure. Le verdict porté est donc qualitatif et
calculé : la dépendance à la cadence est-elle présente d'un côté et absente de l'autre ?

Séparé du pilote pour les deux raisons de `rate_regression.py` (S5304) et
`policy_energy.py` (S5306) : la garde AST interdit tout littéral flottant neuf dans un
pilote, et un verdict doit être calculé et testé, jamais saisi dans un JSON.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.evaluation.policy_energy import _lineaire, _raison_non_lineaire
from src.evaluation.rate_regression import (
    A_MESURER,
    R2_MIN,
    SLOPE_SIGMA,
    UA_PER_A,
)

#: Modèle mesuré par toutes les variantes : c'est la cellule qui a fait apparaître
#: l'anomalie (S5304 `ewc_fp32`, pente +20,73 µA/Hz sur le build stock).
CELL = "ewc_fp32"

#: Dataset du flux — 4 features envoyées quelle que soit la dimension compilée.
DATASET = "monitoring"

#: Nom de l'estimateur. Identique à celui de S5304 puisque c'est le MÊME ajustement ;
#: ce qui change est l'objet comparé (deux binaires), pas la méthode.
METHOD = "régression I(rate)"

#: Variantes de binaire, chacune décrite par ce qui la distingue et par rien d'autre.
#: `weights_state` documente l'état ATTENDU de la tête EWC : « exportés » (la garde
#: `pipeline.c` charge les poids) ou « repli Xavier » (`EWC_IN` compilé ≠
#: `EWC_HEAD_NATIVE_DIM` de l'en-tête → la garde refuse le chargement, sans qu'aucun
#: en-tête n'ait été édité à la main).
VARIANTS: dict[str, dict[str, Any]] = {
    # ── B3 (2026-09-07, deux séances) : dimension tenue, poids variables ──────
    "poids": {
        "ewc_in": 4, "extra_cflags": "", "weights_state": "exportés",
        "measured_by": "B3", "paire": "B3",
        "description": "k=4, tête EWC chargée depuis les poids exportés",
    },
    "xavier": {
        "ewc_in": 4, "extra_cflags": "", "weights_state": "repli Xavier",
        "measured_by": "B3", "paire": "B3",
        "description": "k=4, tête EWC non chargée (garde de dimension refusée)",
    },
    # ── B8 (paire ii) : poids non chargés des deux côtés, dimension variable ──
    "paire2_k4": {
        "ewc_in": 4, "extra_cflags": "", "weights_state": "repli Xavier",
        "measured_by": "B8", "paire": "B8",
        "description": "k=4, tête EWC non chargée — arm de référence de la paire (ii)",
    },
    "paire2_k6": {
        "ewc_in": 6, "extra_cflags": "", "weights_state": "repli Xavier",
        "measured_by": "B8", "paire": "B8",
        "description": "k=6, tête EWC non chargée — seule la dimension change",
    },
}

#: Paires appariées : les deux cellules comparées, et LA variable qui les sépare. Le
#: verdict est formulé à partir de cette variable et jamais d'une autre — B3 fait varier
#: les poids, B8 la dimension, et une paire ne conclut que sur ce qu'elle a fait varier.
PAIRS: dict[str, dict[str, Any]] = {
    "B3": {"cells": ("poids", "xavier"), "varied": "les poids de la tête EWC",
           "held": "dimension k=4, .bss au bit près, trame UART, grille, graine et ordre",
           "not_held": "les deux cellules sont deux SÉANCES distinctes : seules les "
                       "pentes, qui absorbent un offset constant, se comparent"},
    "B8": {"cells": ("paire2_k6", "paire2_k4"), "varied": "la dimension compilée",
           "held": "poids non chargés des deux côtés (mêmes en-têtes, empreintes "
                   "vérifiées), même trame de 4 features, même séance, même grille, "
                   "même graine, même ordre",
           "not_held": "la RAM statique suit la dimension et ne peut pas être tenue "
                       "(.bss croît avec k : les matrices Mahalanobis sont en k²) — "
                       "c'est une CONSÉQUENCE de la variable comparée, pas une seconde "
                       "variable, mais elle interdit d'attribuer un écart à la seule "
                       "arithmétique du calcul"},
}

#: Verdicts possibles d'une paire, formatés avec la variable réellement comparée. Aucun
#: n'est un échec : « ne suit pas » et « indéterminé » bornent la conclusion au lieu de la
#: fabriquer.
VERDICT_SUIT = "l'effet suit {varied}"
VERDICT_NE_SUIT_PAS = "l'effet ne suit pas {varied}"
VERDICT_INDETERMINE = "indéterminé"

#: États de la dépendance à la cadence pour une cellule prise seule.
PRESENTE = "présente"
ABSENTE = "absente"
INDETERMINEE = "indéterminée"


def make_dims(ewc_in: int) -> list[str]:
    """Arguments `make` d'une variante — miroir de `run_sprint38_board.build_and_flash`.

    Les quatre dimensions bougent ensemble : c'est le jeu de dimensions du build S38 qui
    est mis en cause, pas `EWC_IN` isolément. `PROTO_MAX_N` n'est ajouté qu'au-delà de 16,
    comme dans le pilote S38.
    """
    dims = [f"EWC_IN={ewc_in}", f"MAHA_DIM={ewc_in}",
            f"TINYOL_IN={ewc_in}", f"HDC_N_FEATURES={ewc_in}"]
    if ewc_in > 16:
        dims.append(f"PROTO_MAX_N={ewc_in}")
    return dims


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def slope_presence(cell: Mapping[str, Any] | None) -> dict:
    """La consommation dépend-elle de la cadence dans cette cellule ? — verdict calculé.

    La question posée est celle qui compte pour S5306 : **un coût marginal se lit-il dans
    cette cellule ?** Deux états, plus un refus de conclure quand la cellule manque :

    * ``présente`` — la régression est publiable au sens de S5304 (`r² ≥ R2_MIN` **et**
      pente `> SLOPE_SIGMA·σ`) : le coût marginal existe et est chiffré ;
    * ``absente`` — la régression n'est pas publiable : aucun coût marginal ne se lit. Le
      `sub_state` dit POURQUOI (pente non séparable de zéro, pente négative, ou nuage non
      linéaire), car ces trois régimes ne se confondent pas — mais aucun ne permet de lire
      un coût marginal, et c'est cela que la paire compare ;
    * ``indéterminée`` — cellule absente ou régression incomplète : on ne conclut pas.

    Le seuil de publication n'est pas redéfini ici : c'est celui de
    `rate_regression.energy_uj_per_inference`, pour qu'une seule règle gouverne « une pente
    décrit-elle un coût marginal ? » dans toute la campagne.
    """
    if cell is None:
        return {"state": INDETERMINEE, "rationale": "cellule absente"}
    if not (_is_number(cell.get("slope_ua_per_hz"))
            and _is_number(cell.get("slope_std_ua_per_hz"))
            and _is_number(cell.get("r2"))):
        return {"state": INDETERMINEE,
                "rationale": "régression absente ou incomplète dans la cellule"}

    pente = float(cell["slope_ua_per_hz"])
    sigma = float(cell["slope_std_ua_per_hz"])
    r2 = float(cell["r2"])
    chiffre = _is_number(cell.get("energy_uj_per_inference"))
    bloc = {"slope_ua_per_hz": pente, "slope_std_ua_per_hz": sigma, "r2": r2}

    if chiffre and r2 >= R2_MIN:
        bloc.update({
            "state": PRESENTE,
            "sub_state": None,
            "rationale": (f"pente {pente:+.3f} ± {sigma:.3f} µA/Hz à r²={r2:.3f} : "
                          f"la consommation suit la cadence et le coût marginal est "
                          f"chiffré ({float(cell['energy_uj_per_inference']):.1f} µJ)."),
        })
        return bloc

    if abs(pente) <= SLOPE_SIGMA * sigma:
        sous = "pente non séparable de zéro"
    elif pente < 0:
        sous = "pente négative"
    else:
        sous = "nuage non linéaire"
    bloc.update({
        "state": ABSENTE,
        "sub_state": sous,
        "rationale": (f"pente {pente:+.3f} ± {sigma:.3f} µA/Hz, r²={r2:.3f} "
                      f"({sous}) : aucun coût marginal ne se lit dans cette cellule."),
    })
    return bloc


def paired_verdict(cell_a: Mapping[str, Any] | None,
                   cell_b: Mapping[str, Any] | None,
                   name_a: str, name_b: str,
                   varied: str = "la variable comparée",
                   same_session: bool | None = None) -> dict:
    """Verdict d'une paire appariée `a` contre `b` — calculé, jamais saisi.

    La différence de pentes est reportée avec son incertitude propagée **en tant que
    diagnostic**, et explicitement PAS comme une énergie : la règle A4
    (`policy_energy._lineaire`) exige la linéarité de chaque cellule, et une cellule plate
    est ici le phénomène cherché. Le champ `delta_publishable_as_energy` porte ce refus et
    sa raison, pour qu'aucun lecteur ne prenne un µA/Hz de diagnostic pour un µJ.
    """
    a = slope_presence(cell_a)
    b = slope_presence(cell_b)
    bloc: dict[str, Any] = {
        "pair": [name_a, name_b],
        "varied": varied,
        "presence": {name_a: a, name_b: b},
        "method": METHOD,
        "same_session": same_session,
    }

    if (_is_number(a.get("slope_ua_per_hz")) and _is_number(b.get("slope_ua_per_hz"))):
        delta = float(a["slope_ua_per_hz"]) - float(b["slope_ua_per_hz"])
        sigma = math.hypot(float(a["slope_std_ua_per_hz"]),
                           float(b["slope_std_ua_per_hz"]))
        bloc["delta_slope_ua_per_hz"] = delta
        bloc["delta_slope_std_ua_per_hz"] = sigma
        bloc["delta_sigma"] = abs(delta) / sigma if sigma > 0 else None
        lineaires = all(_lineaire(c) for c in (cell_a, cell_b))
        bloc["delta_publishable_as_energy"] = False
        bloc["delta_na_reason"] = (
            "diagnostic de dimension, pas une énergie : la différence de pentes n'est "
            "convertible en µJ que si chaque régression est linéaire (règle A4), or "
            + ("les deux le sont ici mais l'objet comparé est un jeu de dimensions, pas "
               "une politique de mise à jour — la convertir en µJ nommerait une grandeur "
               "qui n'existe pas." if lineaires else
               "; ".join(f"{n} : {_raison_non_lineaire(c)}"
                         for n, c in ((name_a, cell_a), (name_b, cell_b))
                         if not _lineaire(c)) + ".")
        )
        bloc["delta_uj_equivalent"] = A_MESURER

    etats = (a["state"], b["state"])
    if etats in ((PRESENTE, ABSENTE), (ABSENTE, PRESENTE)):
        premier, second = ("PRÉSENTE", "ABSENTE") if etats[0] == PRESENTE else ("ABSENTE", "PRÉSENTE")
        bloc["verdict"] = VERDICT_SUIT.format(varied=varied)
        bloc["rationale"] = (
            f"la dépendance à la cadence est {premier} sur {name_a} et {second} sur "
            f"{name_b}, seul(e)s {varied} les séparant. {name_a} — {a['rationale']} "
            f"{name_b} — {b['rationale']}")
    elif etats in ((PRESENTE, PRESENTE), (ABSENTE, ABSENTE)):
        bloc["verdict"] = VERDICT_NE_SUIT_PAS.format(varied=varied)
        bloc["rationale"] = (
            f"les deux variantes sont dans le même état ({a['state']}) alors que "
            f"{varied} les sépare : elle n'explique pas l'anomalie. "
            f"{name_a} — {a['rationale']} {name_b} — {b['rationale']}")
    else:
        bloc["verdict"] = VERDICT_INDETERMINE
        bloc["rationale"] = (
            f"au moins une des deux cellules ne tranche pas. {name_a} — {a['rationale']} "
            f"{name_b} — {b['rationale']}")

    if same_session is False and bloc["verdict"] != VERDICT_INDETERMINE:
        bloc["reserve"] = (
            "les deux cellules proviennent de séances distinctes : la dérive de repos "
            "inter-séance mesurée en S5301 s'ajoute à l'écart. Seules les PENTES, qui "
            "absorbent un offset constant, se comparent — ce que fait ce verdict.")
    return bloc
