"""rate_regression.py — µJ par inférence par régression de cadence (S5304).

POURQUOI CETTE MÉTHODE, ET POURQUOI ELLE EST SÉPARÉE DU PILOTE :

Deux verrous bloquent les µJ par inférence du projet, et la régression de cadence les
contourne tous les deux **sans toucher au firmware ni au câblage** :

    1. **Pas de référence de repos fiable.** Le Sprint 50 a mesuré un repos (54,8 mA) PLUS
       HAUT que tous les régimes de flux : le protocole delta sort des µJ négatifs. Ici,
       aucun repos n'est soustrait — `I_base` (l'ordonnée à l'origine) absorbe
       INTÉGRALEMENT le repos, la scrutation active de l'attente UART, le trafic de
       l'hôte et l'anomalie de S5301. Seule la **pente** porte le coût marginal d'une
       inférence. La méthode reste donc valide quelles que soient les conclusions de
       S5301, S5302 et S5303.
    2. **Le signal sous le bruit.** À cadence unique, l'écart EWC INT8 ↔ FP32 vaut
       −0,003 mA pour une dispersion de ±0,03 mA. Faire varier la cadence transforme un
       écart ponctuel en une PENTE ajustée sur plusieurs points, dont l'erreur-type est
       elle-même mesurée.

Ce que la pente contient, et comment on l'en sépare (raffinement de la spec §2a) : le coût
d'une trame UART (identique pour toutes les cellules) PLUS le coût de calcul. La régression
de second niveau `pente vs latence DWT` (cf. :func:`slope_vs_latency`) donne alors deux
grandeurs que le projet n'a jamais mesurées — le coût énergétique par µs de calcul (sa
pente) et le coût énergétique d'une trame UART (son ordonnée).

Le module est séparé de `scripts/run_s53_rate_sweep.py` pour les deux raisons du précédent
`counterbalance.py` (S5301) :
    1. la garde AST des tests interdit tout littéral flottant nouveau dans le pilote — les
       seuils de décision vivent donc ici, testables hors banc ;
    2. la décision « énergie chiffrée ou N/A honnête » est ainsi **calculée et testée**,
       jamais saisie à la main dans un JSON.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : ce module ne produit que des grandeurs dérivées
des courants qu'on lui passe. Une régression non concluante rend ``"à mesurer"`` et sa
raison chiffrée, jamais un zéro ni une pente négative présentée comme une énergie.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, MutableMapping, NamedTuple, Sequence

#: Valeur littérale des grandeurs non mesurables (convention du projet, cf. energy_capture).
A_MESURER = "à mesurer"

# ── Seuils de la règle de décision (documentés, pas arbitraires) ─────────────

#: Coefficient de détermination minimal pour qu'une régression soit publiable (spec S5304
#: §5). En deçà, le nuage n'est pas une droite : la pente ne décrit alors pas un coût
#: marginal par inférence, quel que soit son signe.
R2_MIN = 0.9

#: Nombre d'erreurs-types au-dessus de zéro exigé de la pente (spec S5304 §5). Nuance
#: honnête : pour une régression simple, `r² = t²/(t² + n − 2)` avec `t = pente/σ_pente`,
#: si bien qu'un `r² ≥ 0,9` impose déjà `t ≥ 3√(n−2) > 2`. Ce seuil n'ajoute donc pas de
#: sévérité au critère de linéarité — ce qu'il attrape réellement, c'est le **signe** :
#: une droite décroissante parfaitement linéaire passe le test du r² et doit malgré tout
#: être refusée. C'est précisément le cas où le protocole delta du Sprint 50 rendait des
#: µJ négatifs. Il sert aussi de garde si `R2_MIN` était un jour assoupli.
SLOPE_SIGMA = 2.0

#: Écart relatif toléré entre la cadence COMMANDÉE et la cadence ATTEINTE avant de déclarer
#: la saturation. Le plafond n'est pas le calcul mais l'UART : 32 B de requête + 23 B de
#: réponse V3 à 115200 bauds ≈ 209 inférences/s. Au-delà, le flux tourne moins vite que la
#: consigne SANS perdre de trame ni lever de CRC — l'axe des cadences sature en silence et
#: la pente est sous-estimée. 5 % : au-dessus de la gigue d'ordonnancement de l'hôte,
#: très en dessous d'un vrai décrochage.
SATURATION_TOLERANCE = 0.05

#: Facteur de tolérance du contrôle de cohérence croisée (critère d'acceptation S5304) :
#: `uj_per_us_compute × Δlatence` doit être « du même ordre » que l'écart de pente mesuré.
#: Un facteur 3 encadre « même ordre de grandeur » sans prétendre à une égalité que deux
#: estimateurs indépendants n'ont aucune raison d'atteindre.
COHERENCE_FACTOR = 3.0

#: Conversions d'unités (pas des mesures).
UA_PER_A = 1e6
MA_PER_A = 1e3
UJ_PER_J = 1e6
US_PER_S = 1e6

#: Nom du protocole, recopié dans le champ `method` des cellules. Il DOIT rester distinct
#: de celui du protocole delta (S5302) et de celui de l'intégration par phase (S5305) :
#: ce sont trois estimateurs différents, dont la comparaison est elle-même un résultat.
METHOD = "régression I(rate)"


class Fit(NamedTuple):
    """Ajustement `y = intercept + slope · x` et son incertitude.

    `NamedTuple` et non `dataclass` : les pilotes de banc chargent leurs dépendances par
    `importlib.util.spec_from_file_location` sans les enregistrer dans `sys.modules`, ce
    qui casse la construction d'un dataclass.

    `slope_std` est l'erreur-type RÉSIDUELLE de la pente : elle est estimée sur la
    dispersion des points autour de la droite, et non postulée. C'est elle qui décide,
    dans :func:`energy_uj_per_inference`, si l'énergie est publiable.
    """

    slope: float
    slope_std: float
    intercept: float
    r2: float
    n_points: int
    weighted: bool


def _extract_points(points: Iterable[Any]) -> list[tuple[float, float, float]]:
    """Normalise les points en `(rate_hz, i_mean_a, i_std_a)`.

    Accepte le schéma JSON du sprint (`{"rate_hz", "i_mean_a", "i_std_a"}`) comme les
    couples bruts `(rate, courant)` du pilote du 2026-08-05, pour que les points déjà
    mesurés restent réanalysables sans être ressaisis.
    """
    out: list[tuple[float, float, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            out.append((float(point["rate_hz"]), float(point["i_mean_a"]),
                        float(point.get("i_std_a") or 0.0)))
        else:
            rate, current, *rest = point
            out.append((float(rate), float(current), float(rest[0]) if rest else 0.0))
    return out


def weighted_linear_fit(points: Iterable[Any]) -> Fit:
    """Moindres carrés de `I(rate)`, pondérés par `1/σ²` quand les σ sont disponibles.

    Pondérer par `1/σ²` est ce qui donne son sens aux répétitions : un point dont les
    répétitions divergent (une acquisition perturbée) pèse moins qu'un point stable. La
    pondération est ABANDONNÉE dès qu'un `i_std_a` vaut 0 — un poids infini ferait passer
    la droite par ce seul point. C'est le cas `n_repeats = 1`, et c'est aussi celui des
    points bruts du pilote, dont chaque répétition est un point à part entière.

    Lève `ValueError` s'il y a moins de trois points distincts en cadence : deux points
    passent toujours par une droite et n'ont ni résidu ni erreur-type — publier une
    « énergie » depuis un tel ajustement serait inventer une certitude.
    """
    data = _extract_points(points)
    if len({x for x, _, _ in data}) < 3:
        raise ValueError(
            "au moins trois cadences DISTINCTES sont requises : avec deux points la droite "
            "est exacte par construction, son r² vaut 1 et son erreur-type n'existe pas."
        )
    weighted = all(sigma > 0.0 for _, _, sigma in data)
    weights = [1.0 / sigma ** 2 if weighted else 1.0 for _, _, sigma in data]

    total = sum(weights)
    mx = sum(w * x for w, (x, _, _) in zip(weights, data)) / total
    my = sum(w * y for w, (_, y, _) in zip(weights, data)) / total
    sxx = sum(w * (x - mx) ** 2 for w, (x, _, _) in zip(weights, data))
    sxy = sum(w * (x - mx) * (y - my) for w, (x, y, _) in zip(weights, data))
    slope = sxy / sxx
    intercept = my - slope * mx

    residuals = [y - (intercept + slope * x) for x, y, _ in data]
    ss_res = sum(w * r ** 2 for w, r in zip(weights, residuals))
    ss_tot = sum(w * (y - my) ** 2 for w, (_, y, _) in zip(weights, data))
    dof = len(data) - 2
    slope_std = math.sqrt(ss_res / dof / sxx) if dof > 0 else float("inf")
    return Fit(
        slope=slope,
        slope_std=slope_std,
        intercept=intercept,
        r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        n_points=len(data),
        weighted=weighted,
    )


def energy_uj_per_inference(fit: Fit, tension_v: float) -> tuple[float | str, str | None]:
    """`E = pente · V`, en µJ — ou ``"à mesurer"`` avec sa raison CHIFFRÉE.

    Deux conditions de publication, et aucune n'est négociable (spec S5304 §5) :
        - `r² ≥ R2_MIN` : sans linéarité, la pente ne décrit pas un coût marginal ;
        - `pente > SLOPE_SIGMA × erreur-type` : sans quoi l'énergie n'est pas séparable de
          zéro. Ce test écarte AUSSI, par construction, toute pente négative — une pente
          négative n'est jamais rapportée comme une énergie.

    Retourne `(valeur, None)` si publiable, `(A_MESURER, raison)` sinon.
    """
    if fit.r2 < R2_MIN:
        return A_MESURER, (
            f"régression non significative : r²={fit.r2:.3f} < {R2_MIN} — le courant ne "
            f"suit pas une droite en cadence sur ce balayage, la pente ne mesure donc pas "
            f"un coût marginal par inférence."
        )
    if fit.slope <= SLOPE_SIGMA * fit.slope_std:
        return A_MESURER, (
            f"régression non significative : pente={fit.slope * UA_PER_A:+.3f} ± "
            f"{fit.slope_std * UA_PER_A:.3f} µA/Hz, soit moins de {SLOPE_SIGMA:g} σ "
            f"au-dessus de zéro (r²={fit.r2:.3f}) — l'énergie par inférence n'est pas "
            f"séparable du bruit du banc à cette cadence."
        )
    return fit.slope * float(tension_v) * UJ_PER_J, None


def energy_uncertainty_uj(fit: Fit, tension_v: float) -> float:
    """Incertitude sur l'énergie, propagée depuis l'erreur-type de la pente."""
    return fit.slope_std * float(tension_v) * UJ_PER_J


def duty_cycle(latency_us: float | None, rate_hz: float) -> float | None:
    """Taux d'occupation du calcul à une cadence donnée (sans unité).

    C'est la grandeur qui dit si une cellule est mesurable : à 209 Hz, HDC occupe 41 % du
    temps (mesurable), Mahalanobis 0,1 % (noyé). Retourne `None` sans latence mesurée —
    le taux d'occupation ne se déduit pas de la cadence seule.
    """
    if latency_us is None or rate_hz <= 0:
        return None
    return float(latency_us) / US_PER_S * float(rate_hz)


def saturation_rate_hz(points: Iterable[Any]) -> float | None:
    """Plus petite cadence COMMANDÉE dont la cadence ATTEINTE décroche, ou `None`.

    Sans ce contrôle, le balayage sature en silence : au-delà du plafond de transport
    (~209 Hz), `sensor_stream.py` ne perd aucune trame et ne lève aucun CRC, il émet
    simplement moins vite que la consigne. Les points saturés se tassent alors sur l'axe
    des abscisses et **sous-estiment la pente**.

    Un point sans cadence atteinte relevée est ignoré (il ne prouve rien), pas compté comme
    non saturé.
    """
    candidats: list[float] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        commanded = float(point["rate_hz"])
        achieved = point.get("achieved_rate_hz")
        if achieved is None or commanded <= 0:
            continue
        if float(achieved) < commanded * (1.0 - SATURATION_TOLERANCE):
            candidats.append(commanded)
    return min(candidats) if candidats else None


def normalize_achieved_source(points: Sequence[Any]) -> None:
    """Rétablit, EN PLACE, la provenance de `achieved_rate_hz` (règle A7).

    `achieved_rate_hz` est un champ de MESURE. Les pilotes antérieurs au 2026-09-07 y
    recopiaient la CONSIGNE pour toutes les cadences non re-streamées, ce qui présentait une
    hypothèse comme un relevé. La provenance reste reconstructible sans ambiguïté :

      * borne haute                 → point réellement re-streamé, donc MESURÉ — y compris
        (et surtout) quand la cadence atteinte décroche de la consigne : c'est là que le
        plafond de transport se relève ;
      * ailleurs, atteinte ≠ consigne → propagée depuis ce plafond mesuré (inférence) ;
      * ailleurs, atteinte = consigne → jamais mesurée : le champ repasse à `None`.

    Sans effet sur l'ajustement : :func:`saturation_rate_hz` ne retient que les points dont
    la cadence atteinte DÉCROCHE et ignore ceux qui n'en portent aucune.
    """
    cadences = [float(p["rate_hz"]) for p in points
                if isinstance(p, MutableMapping) and float(p["rate_hz"]) > 0]
    if not cadences:
        return
    rate_max = max(cadences)
    for point in points:
        if not isinstance(point, MutableMapping):
            continue
        rate = float(point["rate_hz"])
        atteint = point.get("achieved_rate_hz")
        if rate <= 0 or atteint is None:
            continue
        if rate == rate_max:
            point["achieved_rate_source"] = "mesuré"
        elif float(atteint) != rate:
            point["achieved_rate_source"] = "inféré du plafond mesuré"
        else:
            point["achieved_rate_hz"] = None
            point["achieved_rate_source"] = "non mesuré"


def usable_points(points: Iterable[Any]) -> list[dict]:
    """Points retenus pour l'ajustement : ceux qui ne saturent pas.

    Le repos (`rate = 0`) est conservé — c'est l'ancrage de `I_base`. Les points au-delà
    de la saturation sont ÉCARTÉS de l'ajustement mais restent écrits dans le JSON : une
    mesure écartée se montre, elle ne se supprime pas.
    """
    seuil = saturation_rate_hz(points)
    kept = [dict(p) for p in points if isinstance(p, Mapping)]
    if seuil is None:
        return kept
    return [p for p in kept if float(p["rate_hz"]) < seuil]


def fit_cell(points: Sequence[Any], tension_v: float,
             latency_us: float | None = None) -> dict:
    """Bloc de grandeurs dérivées d'une cellule — tout ce que le JSON ne mesure pas lui-même.

    Aucune valeur n'y est saisie : pente, énergie, saturation et taux d'occupation sont
    calculés depuis `points`, et l'énergie sort ``"à mesurer"`` + `na_reason` dès que la
    règle de publication n'est pas satisfaite.
    """
    seuil = saturation_rate_hz(points)
    retenus = usable_points(points)
    fit = weighted_linear_fit(retenus)
    energie, raison = energy_uj_per_inference(fit, tension_v)
    rate_max = max((float(p["rate_hz"]) for p in retenus), default=0.0)
    bloc = {
        "method": METHOD,
        "slope_ua_per_hz": fit.slope * UA_PER_A,
        "slope_std_ua_per_hz": fit.slope_std * UA_PER_A,
        "intercept_ma": fit.intercept * MA_PER_A,
        "r2": fit.r2,
        "n_points_fitted": fit.n_points,
        "weighted_by_inverse_variance": fit.weighted,
        "energy_uj_per_inference": energie,
        "energy_uncertainty_uj": energy_uncertainty_uj(fit, tension_v),
        "saturation_rate_hz": seuil,
        "duty_cycle_at_max_rate": duty_cycle(latency_us, rate_max),
        "max_rate_fitted_hz": rate_max,
        "tension_v": float(tension_v),
        "dwt_latency_us_p50": latency_us,
    }
    if raison is not None:
        bloc["energy_na_reason"] = raison
    return bloc


def slope_vs_latency(cells: Mapping[str, Mapping]) -> dict:
    """Régression de second niveau : énergie par inférence CONTRE latence DWT.

    C'est le raffinement qui donne sa valeur scientifique au balayage (spec §2a). Chaque
    cellule apporte un point `(latence P50, µJ par inférence)` ; la droite ajustée sépare
    ce que la pente d'une cellule mélangeait :

        - sa **pente** = coût énergétique par µs de CALCUL (`uj_per_us_compute`) ;
        - son **ordonnée** = coût énergétique d'une TRAME UART (`uj_per_uart_frame`),
          commun à toutes les cellules et jamais mesuré jusqu'ici.

    Les cellules dont l'énergie est ``"à mesurer"`` sont exclues et NOMMÉES : une droite
    ajustée sur des cellules non publiables serait une droite inventée.
    """
    retenues: list[tuple[str, float, float]] = []
    exclues: dict[str, str] = {}
    for name, cell in cells.items():
        energie = cell.get("energy_uj_per_inference")
        latence = cell.get("dwt_latency_us_p50")
        if not isinstance(energie, (int, float)) or latence is None:
            exclues[name] = (
                cell.get("energy_na_reason")
                or "latence DWT non relevée pour cette cellule"
            )
            continue
        retenues.append((name, float(latence), float(energie)))

    if len({lat for _, lat, _ in retenues}) < 3:
        return {
            "uj_per_us_compute": A_MESURER,
            "uj_per_uart_frame": A_MESURER,
            "r2": None,
            "cells_used": [name for name, _, _ in retenues],
            "cells_excluded": exclues,
            "na_reason": (
                f"{len(retenues)} cellule(s) publiable(s) à latences distinctes : la "
                f"régression de second niveau en exige au moins trois."
            ),
            "method": "régression µJ(latence DWT) sur les cellules du balayage",
        }

    fit = weighted_linear_fit([(lat, uj) for _, lat, uj in retenues])
    return {
        "uj_per_us_compute": fit.slope,
        "uj_per_us_compute_std": fit.slope_std,
        "uj_per_uart_frame": fit.intercept,
        "r2": fit.r2,
        "points": [{"cell": name, "dwt_latency_us_p50": lat, "energy_uj_per_inference": uj}
                   for name, lat, uj in retenues],
        "cells_used": [name for name, _, _ in retenues],
        "cells_excluded": exclues,
        "method": "régression µJ(latence DWT) sur les cellules du balayage",
    }


def _pair_gap(cells: Mapping[str, Mapping], model: str) -> dict | None:
    """Écart INT8 ↔ FP32 d'un modèle, en µJ, avec son incertitude propagée."""
    int8, fp32 = cells.get(f"{model}_int8"), cells.get(f"{model}_fp32")
    if int8 is None or fp32 is None:
        return None
    e_int8, e_fp32 = (int8.get("energy_uj_per_inference"),
                      fp32.get("energy_uj_per_inference"))
    if not isinstance(e_int8, (int, float)) or not isinstance(e_fp32, (int, float)):
        return None
    sigma = math.hypot(float(int8.get("energy_uncertainty_uj") or 0.0),
                       float(fp32.get("energy_uncertainty_uj") or 0.0))
    return {
        "model": model,
        "energy_uj_int8": float(e_int8),
        "energy_uj_fp32": float(e_fp32),
        "delta_uj": float(e_int8) - float(e_fp32),
        "delta_sigma_uj": sigma,
        "significant": sigma > 0 and abs(float(e_int8) - float(e_fp32)) > SLOPE_SIGMA * sigma,
        "latency_us_int8": int8.get("dwt_latency_us_p50"),
        "latency_us_fp32": fp32.get("dwt_latency_us_p50"),
    }


def gap3_energy_verdict(cells: Mapping[str, Mapping], model: str = "ewc") -> dict:
    """Verdict Gap 3 énergie CALCULÉ : l'INT8 change-t-il les µJ, et dans quel sens ?

    Le Sprint 50 a répondu « non » sur le COURANT MOYEN (écarts dans le bruit, HDC INT8
    même +7,1 %) ; cette tâche répond sur les µJ par inférence, qui est la grandeur que le
    mémoire discute. Les deux réponses sont indépendantes et se citent séparément.

    Le verdict n'est jamais une opinion : il compare l'écart à `SLOPE_SIGMA` fois son
    incertitude propagée, et dit « non concluant » quand les deux cellules ne sont pas
    toutes deux publiables.
    """
    ecart = _pair_gap(cells, model)
    if ecart is None:
        return {
            "verdict": A_MESURER,
            "rationale": (
                f"les deux cellules {model}_int8 et {model}_fp32 ne sont pas toutes deux "
                f"publiables : l'écart énergétique INT8↔FP32 n'est pas constatable par ce "
                f"balayage."
            ),
        }
    signe = "supérieure" if ecart["delta_uj"] > 0 else "inférieure"
    if not ecart["significant"]:
        verdict, raison = "non_significatif", (
            f"l'énergie par inférence de {model} INT8 ({ecart['energy_uj_int8']:.1f} µJ) "
            f"et FP32 ({ecart['energy_uj_fp32']:.1f} µJ) diffèrent de "
            f"{ecart['delta_uj']:+.1f} µJ pour une incertitude combinée de "
            f"±{ecart['delta_sigma_uj']:.1f} µJ (< {SLOPE_SIGMA:g} σ) : l'INT8 ne change "
            f"pas mesurablement l'énergie par inférence. Le gain INT8 reste la RAM."
        )
    else:
        verdict = "int8_plus_couteux" if ecart["delta_uj"] > 0 else "int8_moins_couteux"
        raison = (
            f"l'énergie par inférence de {model} INT8 ({ecart['energy_uj_int8']:.1f} µJ) est "
            f"{signe} à celle du FP32 ({ecart['energy_uj_fp32']:.1f} µJ) de "
            f"{abs(ecart['delta_uj']):.1f} ± {ecart['delta_sigma_uj']:.1f} µJ "
            f"(> {SLOPE_SIGMA:g} σ)."
        )
    return {"verdict": verdict, "rationale": raison, **ecart}


def coherence_check(cells: Mapping[str, Mapping], second_level: Mapping,
                    model: str = "ewc") -> dict:
    """Contrôle de cohérence croisée du critère d'acceptation S5304.

    Deux chemins indépendants mènent au surcoût énergétique de l'INT8 :
        - la **différence de pentes** mesurée entre les deux cellules ;
        - `uj_per_us_compute` × la **différence de latences DWT** relevée.
    S'ils divergent de plus d'un facteur `COHERENCE_FACTOR`, la régression de second niveau
    ne décrit pas ce que les cellules mesurent, et c'est un résultat à consigner — pas une
    erreur à masquer.
    """
    ecart = _pair_gap(cells, model)
    par_us = second_level.get("uj_per_us_compute")
    if ecart is None or not isinstance(par_us, (int, float)):
        return {
            "coherent": None,
            "na_reason": (
                "contrôle impossible : il exige les deux cellules du modèle publiables ET "
                "un `uj_per_us_compute` chiffré."
            ),
        }
    if ecart["latency_us_int8"] is None or ecart["latency_us_fp32"] is None:
        return {"coherent": None,
                "na_reason": "latences DWT manquantes pour au moins une des deux cellules"}
    delta_lat = float(ecart["latency_us_int8"]) - float(ecart["latency_us_fp32"])
    attendu = float(par_us) * delta_lat
    mesure = ecart["delta_uj"]
    if attendu == 0.0 or mesure == 0.0:
        rapport = None
        coherent = None
    else:
        rapport = mesure / attendu
        coherent = bool(0 < rapport and 1.0 / COHERENCE_FACTOR <= rapport <= COHERENCE_FACTOR)
    return {
        "model": model,
        "delta_latency_us": delta_lat,
        "expected_delta_uj_from_second_level": attendu,
        "measured_delta_uj": mesure,
        "ratio_measured_over_expected": rapport,
        "tolerance_factor": COHERENCE_FACTOR,
        "coherent": coherent,
        "rationale": (
            f"le second niveau prédit {attendu:+.1f} µJ d'écart INT8↔FP32 "
            f"({par_us:.3f} µJ/µs × {delta_lat:+.1f} µs) ; le balayage en mesure "
            f"{mesure:+.1f} µJ."
        ),
    }
