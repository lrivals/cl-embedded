"""
counterbalance.py — Verdict du protocole contre-balancé repos/charge (S5301).

POURQUOI CE MODULE EXISTE, ET POURQUOI IL EST SÉPARÉ DU PILOTE :

Le Sprint 50 a mesuré, et documenté honnêtement, que la référence « au repos » de la
carte (54,8 mA) est PLUS HAUTE que tous les régimes de flux (46,4 à 51,2 mA) — y compris
Mahalanobis, dont l'inférence n'occupe que ~0,05 % du temps. Un écart de −8 mA que le taux
d'occupation ne peut pas expliquer, et dont la cause n'a pas été établie.

Ce que le Sprint 50 n'a PAS testé : dans sa séquence, le repos est systématiquement mesuré
en tête de session et les cellules ensuite. **L'ordre est donc confondu avec la condition.**
Or le préchauffage relevé décroît d'acquisition en acquisition : rien ne prouve que
l'établissement soit terminé quand le repos est pris.

Le protocole contre-balancé (repos intercalé AVANT et APRÈS chaque cellule, plusieurs
répétitions, session unique) sépare les deux hypothèses. Ce module porte la **règle de
décision**, appliquée aux courants mesurés :

    - le repos rejoint le niveau des cellules      → artefact d'ordre (dérive d'établissement)
    - le repos reste stable, au-dessus des cellules → effet réel, cause non établie
    - le repos dérive sans converger               → dérive partielle (préchauffage à rallonger)

Aucun des trois n'est un échec : les trois sont des conclusions publiables sur la
méthodologie de banc.

Le module est séparé de `scripts/run_s50_board_current.py` pour deux raisons :
    1. la garde AST `tests/test_board_current.py::test_pilote_sans_resultat_en_dur`
       interdit tout littéral flottant nouveau dans le pilote — les seuils de la règle
       vivent donc ici, testables hors banc ;
    2. le verdict est ainsi **calculé et testé**, jamais saisi à la main dans un JSON.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : ce module ne produit qu'un verdict dérivé des
courants qu'on lui passe. Sans mesure, pas de verdict (`ValueError`).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

# ── Seuils de la règle de décision (documentés, pas arbitraires) ─────────────

#: Seuil de non-significativité, en écarts-types de la dispersion du RÉGIME ÉTABLI.
#: Choix à 3σ : seuil usuel, et très en deçà des écarts en jeu (le Sprint 50 constatait
#: −8 mA pour une dispersion de ±0,06 mA, soit plus de 100σ).
CONVERGENCE_SIGMA = 3.0

#: Une pente de dérive est jugée significative si, sur l'ensemble de la session, elle
#: déplace le repos de plus de ce nombre d'écarts-types. En deçà, le repos est « plat ».
DRIFT_SIGMA = 3.0

#: Un point de repos appartient au régime établi s'il tient dans ce nombre d'écarts-types
#: du plateau de fin de session. Sert à écarter la queue d'établissement AVANT d'estimer
#: la dispersion — sinon la dérive gonfle sa propre barre d'erreur et rend le test
#: trivialement concluant (défaut relevé à la première exécution du protocole).
PLATEAU_SIGMA = 5.0

#: Dispersion de repli (A) si la session ne fournit pas de quoi l'estimer. Garde-fou
#: contre la division par zéro, JAMAIS une mesure : ordre de grandeur du bruit de
#: quantification de la sonde, sans effet quand un plateau réel existe.
FALLBACK_DISPERSION_A = 1e-6

VERDICT_ARTEFACT = "artefact_ordre"
VERDICT_EFFET_REEL = "effet_reel"
VERDICT_DERIVE = "derive_partielle"

#: Seuil de significativité (en écarts-types du banc) des comparaisons entre ÉTATS de
#: repos (S5301b, cf. `idle_state_verdict`). Même choix qu'ailleurs dans ce module :
#: 3σ, usuel et très en deçà des écarts en jeu (l'ambiguïté à trancher vaut ~10 mA
#: pour une dispersion de banc de l'ordre de ±0,06 mA).
IDLE_STATE_SIGMA = 3.0

#: Clés des états de repos comparés par `idle_state_verdict`. Elles nomment ce qui
#: distingue les états côté HÔTE, seule variable manipulée : le port série de la carte
#: est fermé, ouvert et maintenu sans trafic, parcouru par un flux, ou refermé après
#: interruption d'un flux.
STATE_PORT_CLOSED = "port_ferme"
STATE_PORT_OPEN = "port_ouvert_inactif"
STATE_STREAM = "flux"
STATE_POST_STREAM = "apres_flux"

VERDICT_POST_FLUX = "repos_post_flux_different"
VERDICT_PORT_STATE = "etat_port_significatif"
VERDICT_REPOS_COHERENT = "repos_coherent"
#: Un écart entre états de repos n'est retenu que s'il pèse au moins cette fraction de
#: l'effet que la référence sert à mesurer — le surcoût de charge (état « flux ») mesuré
#: dans la MÊME session. Sur un banc très stable, un biais de référence de 0,1 mA ressort
#: significatif (> 3σ) tout en étant négligeable devant un surcoût de 2 mA : il ne change
#: aucune conclusion. Le seuil est relatif à une grandeur MESURÉE, jamais absolu.
MARGINAL_FRACTION = 0.1

#: Aucun état n'a été répété : la dispersion n'est pas estimable, donc aucun écart n'est
#: interprétable. Ce n'est pas un échec de mesure mais un refus de conclure — sans lui,
#: la dispersion de repli (`FALLBACK_DISPERSION_A`) rendrait n'importe quel écart
#: « significatif » à des centaines de σ (défaut relevé à la première exécution).
VERDICT_INDETERMINE = "dispersion_non_estimee"


def _ranked_currents(points: Iterable[Any]) -> list[tuple[int, float]]:
    """Normalise une liste de points en couples ``(session_index, i_a)`` triés.

    Accepte les deux formes rencontrées : dicts ``{"session_index": .., "i_a": ..}``
    (schéma JSON) ou couples déjà formés.
    """
    out: list[tuple[int, float]] = []
    for p in points:
        if isinstance(p, dict):
            out.append((int(p["session_index"]), float(p["i_a"])))
        else:
            idx, value = p
            out.append((int(idx), float(value)))
    return sorted(out, key=lambda t: t[0])


def idle_drift_slope(points: Iterable[Any]) -> float:
    """Pente de dérive du repos en fonction de son rang dans la session (A/acquisition).

    Régression linéaire par moindres carrés du courant sur le `session_index`. C'est la
    variable explicative testée par S5301 : si l'écart repos↔charge est une queue
    d'établissement, cette pente est négative et non nulle.

    Parameters
    ----------
    points : iterable
        Points de repos, sous forme ``{"session_index": int, "i_a": float}`` ou
        ``(session_index, i_a)``.

    Returns
    -------
    float
        Pente en ampères par acquisition. ``0.0`` si moins de deux points, ou si tous
        les points partagent le même rang (pente indéfinie, jamais extrapolée).
    """
    ranked = _ranked_currents(points)
    n = len(ranked)
    if n < 2:
        return 0.0

    mean_x = sum(x for x, _ in ranked) / n
    mean_y = sum(y for _, y in ranked) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in ranked)
    den = sum((x - mean_x) ** 2 for x, _ in ranked)
    if den == 0.0:
        return 0.0
    return num / den


def bench_dispersion_a(runs: Sequence[float]) -> float:
    """Dispersion du banc (écart-type, A) estimée sur des acquisitions répétées.

    Sert d'unité de comparaison à la règle de verdict : un écart n'est interprété que
    s'il dépasse la dispersion propre du banc. Repli documenté (`FALLBACK_DISPERSION_A`)
    si l'échantillon est trop petit ou dégénéré — garde-fou, pas une mesure.
    """
    values = [float(v) for v in runs]
    n = len(values)
    if n < 2:
        return FALLBACK_DISPERSION_A
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = var ** 0.5
    return std if std > 0.0 else FALLBACK_DISPERSION_A


def established_regime(points: Iterable[Any]) -> list[tuple[int, float]]:
    """Sous-ensemble des repos en **régime établi**, la queue d'établissement écartée.

    Méthode : le plateau de référence est la seconde moitié de la session (au moins
    deux points) ; sa dispersion sert de tolérance, et l'on ne garde que les points qui
    y tiennent (à `PLATEAU_SIGMA` près). C'est le tri qui distingue « le repos a
    dérivé » de « le repos est bruité » — sans lui, la dérive gonfle sa propre barre
    d'erreur.

    Returns
    -------
    list of (session_index, i_a)
        Les points retenus, dans l'ordre. Jamais vide : à défaut, le plateau lui-même.
    """
    ranked = _ranked_currents(points)
    n = len(ranked)
    if n < 3:
        return ranked

    plateau = ranked[max(2, n // 2) * -1:] if n >= 4 else ranked[-2:]
    plateau_values = [v for _, v in plateau]
    mean_plateau = sum(plateau_values) / len(plateau_values)
    sigma = bench_dispersion_a(plateau_values)
    tol = PLATEAU_SIGMA * sigma
    kept = [(x, v) for x, v in ranked if abs(v - mean_plateau) <= tol]
    return kept if kept else plateau


def counterbalance_verdict(
    idle_by_rank: Iterable[Any],
    cell_currents: Sequence[float],
    dispersion_a: float | None = None,
) -> tuple[str, str]:
    """Applique la règle de décision S5301 aux courants mesurés.

    La question tranchée n'est pas « le biais de première acquisition existe-t-il ? »
    (il est établi et reproductible, cf. S5301 § éléments mesurés) mais : **l'écart
    repos↔charge de −8 mA en est-il une continuation, ou un effet de charge réel ?**

    Règle codée (dispersion estimée sur le **régime établi**, cf. `established_regime`) :
        - ``artefact_ordre``    si le dernier repos est **au niveau ou en dessous** des
          cellules (``dernier_repos − moyenne_cellules ≤ 3 × dispersion``) : l'écart
          négatif du Sprint 50 ne se reproduit pas une fois la session établie, donc le
          protocole delta est réhabilité ;
        - ``derive_partielle``  si le repos reste au-dessus des cellules mais dérive
          significativement (> 3 × dispersion sur la session) ;
        - ``effet_reel``        sinon : le repos est plat ET reste au-dessus des cellules.

    Le critère est **unilatéral**, et c'est délibéré : un repos qui finit *sous* les
    cellules est une preuve plus forte que l'égalité (le surcoût de charge redevient
    positif, donc physiquement interprétable). Un critère en valeur absolue aurait
    étiqueté « effet réel » une session où le delta est redevenu normal.

    Parameters
    ----------
    idle_by_rank : iterable
        Mesures de repos avec leur rang dans la session (≥ 2 points attendus).
    cell_currents : sequence of float
        Courants moyens des cellules mesurées dans la même session (A).
    dispersion_a : float, optional
        Dispersion du banc (A). Estimée sur le régime établi si absente — jamais sur
        l'ensemble des points, sous peine de mesurer la dérive avec elle-même.

    Returns
    -------
    (verdict, rationale) : tuple of str
        Le verdict et sa justification chiffrée, destinée au champ
        ``verdict_rationale`` du JSON.

    Raises
    ------
    ValueError
        Si aucune mesure de repos ou aucune cellule n'est fournie : sans mesure, il n'y
        a pas de verdict (règle « aucun chiffre inventé »).
    """
    ranked = _ranked_currents(idle_by_rank)
    cells = [float(c) for c in cell_currents]
    if not ranked:
        raise ValueError("aucune mesure de repos : le verdict S5301 exige une session mesurée.")
    if not cells:
        raise ValueError("aucune cellule mesurée : rien à quoi comparer le repos.")

    established = established_regime(ranked)
    dispersion = float(dispersion_a) if dispersion_a else bench_dispersion_a(
        [i for _, i in established]
    )
    if dispersion <= 0.0:
        dispersion = FALLBACK_DISPERSION_A

    first_idle = ranked[0][1]
    last_idle = ranked[-1][1]
    mean_cells = sum(cells) / len(cells)
    gap = last_idle - mean_cells          # < 0 ⇒ la charge coûte plus que le repos
    gap_sigma = gap / dispersion

    slope = idle_drift_slope(ranked)
    span = ranked[-1][0] - ranked[0][0]
    drift_total = slope * span
    drift_sigma = abs(drift_total) / dispersion

    ma = 1e3  # A → mA, affichage seul
    n_ecartes = len(ranked) - len(established)
    commun = (
        f"premier repos {first_idle * ma:.3f} mA, dernier repos {last_idle * ma:.3f} mA, "
        f"moyenne des cellules {mean_cells * ma:.3f} mA, écart final {gap * ma:+.3f} mA "
        f"= {gap_sigma:+.1f}σ (dispersion du régime établi {dispersion * ma:.4f} mA sur "
        f"{len(established)} points, {n_ecartes} écarté(s) comme queue d'établissement) ; "
        f"dérive du repos sur la session {drift_total * ma:+.3f} mA = {drift_sigma:.1f}σ "
        f"(pente {slope * ma:+.4f} mA/acquisition, {len(ranked)} points)"
    )

    if gap_sigma <= CONVERGENCE_SIGMA:
        return VERDICT_ARTEFACT, (
            f"artefact d'ordre : une fois la session établie, le repos revient au niveau "
            f"des cellules ou en dessous (≤ {CONVERGENCE_SIGMA:g}σ) et le surcoût de "
            f"charge redevient positif — l'écart négatif du Sprint 50 est la queue de "
            f"l'établissement de session, pas un effet de charge. {commun}"
        )
    if drift_sigma > DRIFT_SIGMA:
        return VERDICT_DERIVE, (
            f"dérive partielle : le repos dérive significativement "
            f"(> {DRIFT_SIGMA:g}σ) mais reste au-dessus des cellules — préchauffage à "
            f"allonger avant toute campagne (--warmup-repeats). {commun}"
        )
    return VERDICT_EFFET_REEL, (
        f"effet réel : le repos est plat et reste au-dessus des cellules quel que soit "
        f"son rang — l'ordre n'explique pas l'écart, la cause reste non établie et le "
        f"levier suivant est la mise en sommeil de l'attente UART (S5302). {commun}"
    )


def pooled_dispersion_a(groups: Iterable[Sequence[float]]) -> float:
    """Dispersion intra-état (écart-type poolé, A) sur plusieurs groupes de répétitions.

    Chaque état est mesuré plusieurs fois ; c'est la variabilité DANS un état, et non
    entre états, qui donne l'unité de comparaison. Repli documenté
    (`FALLBACK_DISPERSION_A`) si aucun groupe ne fournit deux points.
    """
    num = 0.0
    dof = 0
    for group in groups:
        values = [float(v) for v in group]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        num += sum((v - mean) ** 2 for v in values)
        dof += len(values) - 1
    if dof == 0:
        return FALLBACK_DISPERSION_A
    std = (num / dof) ** 0.5
    return std if std > 0.0 else FALLBACK_DISPERSION_A


def idle_state_verdict(
    states: dict[str, Sequence[float]],
    responsive_after: dict[str, bool] | None = None,
    dispersion_a: float | None = None,
) -> tuple[str, str]:
    """Tranche l'ambiguïté « de quel repos parle-t-on ? » (S5301b).

    POURQUOI CETTE RÈGLE EXISTE. Deux mesures de repos prises à quelques minutes
    d'intervalle sur le même banc ont donné des valeurs très différentes : repos sur une
    carte jamais streamée d'un côté, repos intercalé après un flux interrompu de l'autre.
    Tant que l'écart n'est pas expliqué, on ne sait pas quelle référence le protocole
    contre-balancé (`counterbalance_verdict`) a réellement comparée aux cellules.

    L'hypothèse testée est que l'interruption du flux (`proc.terminate()` ferme le port,
    DTR/RTS retombent) laisse la carte dans un autre état que le repos « frais ». Les
    quatre états mesurés ne diffèrent que par ce que fait l'HÔTE ; le firmware est le même.

    Règle codée, appliquée aux courants moyens de chaque état :
        - ``repos_post_flux_different`` si le repos APRÈS flux interrompu s'écarte du
          repos port fermé de plus de `IDLE_STATE_SIGMA` × dispersion intra-état : les
          deux « repos » ne sont pas la même condition, et le contre-balancement doit
          être rejoué avec une fermeture de flux maîtrisée ;
        - ``etat_port_significatif`` si, à défaut, l'ouverture seule du port (sans
          trafic) déplace le courant au-delà du même seuil : la référence dépend de
          l'état des lignes de contrôle de l'hôte, à fixer explicitement ;
        - ``repos_coherent`` sinon : les repos sont indiscernables, l'écart observé
          entre sessions vient d'ailleurs (à chercher hors du couple port/flux).

    ``responsive_after`` (la carte répond-elle encore à une trame après chaque état ?)
    n'entre pas dans le choix du verdict — il est mesuré, et reporté dans la
    justification : une carte muette après flux fait passer l'hypothèse « maintenue en
    reset » du plausible au démontré.

    Parameters
    ----------
    states : dict of str -> sequence of float
        Courants mesurés (A) par état ; clés attendues `STATE_*`. Les états absents
        sont ignorés (jamais extrapolés).
    responsive_after : dict of str -> bool, optional
        Réponse de la carte à une trame de contrôle après chaque état.
    dispersion_a : float, optional
        Dispersion du banc (A). Poolée sur les répétitions intra-état si absente.

    Returns
    -------
    (verdict, rationale) : tuple of str

    Raises
    ------
    ValueError
        Si l'état de référence (`STATE_PORT_CLOSED`) n'a pas été mesuré : sans lui il
        n'y a rien à comparer, donc pas de verdict.
    """
    means = {k: sum(map(float, v)) / len(v) for k, v in states.items() if len(v) > 0}
    if STATE_PORT_CLOSED not in means:
        raise ValueError(
            f"état de référence « {STATE_PORT_CLOSED} » non mesuré : "
            "sans référence, pas de verdict (règle « aucun chiffre inventé »)."
        )

    dispersion = float(dispersion_a) if dispersion_a else pooled_dispersion_a(states.values())
    if dispersion <= 0.0:
        dispersion = FALLBACK_DISPERSION_A
    # Dispersion non estimée (aucun état répété) : on refuse de conclure plutôt que de
    # comparer des écarts à un repli qui les rendrait tous significatifs.
    estimee = any(len(v) > 1 for v in states.values())
    if not estimee and dispersion_a is None:
        moyennes = " ; ".join(f"{k} {v * 1e3:.3f} mA" for k, v in means.items())
        return VERDICT_INDETERMINE, (
            "aucun état n'est répété : la dispersion du banc n'est pas estimable sur cette "
            "session, donc aucun écart n'est interprétable (relancer avec --repeats ≥ 2, "
            f"ou fournir une dispersion mesurée). États mesurés : {moyennes}"
        )

    ref = means[STATE_PORT_CLOSED]
    ma = 1e3  # A → mA, affichage seul
    deltas = {k: v - ref for k, v in means.items() if k != STATE_PORT_CLOSED}

    commun = (
        f"référence {STATE_PORT_CLOSED} {ref * ma:.3f} mA ; "
        + " ; ".join(
            f"{k} {means[k] * ma:.3f} mA ({d * ma:+.3f} mA = {d / dispersion:+.1f}σ)"
            for k, d in deltas.items()
        )
        + f" ; dispersion intra-état {dispersion * ma:.4f} mA"
    )
    if responsive_after:
        commun += " ; réponse de la carte après état : " + ", ".join(
            f"{k}={'oui' if ok else 'NON'}" for k, ok in responsive_after.items()
        )

    # Effet de référence mesuré dans la même session : le surcoût de charge. Il donne
    # l'échelle à laquelle un biais de référence compte ou non (cf. MARGINAL_FRACTION).
    charge = abs(deltas.get(STATE_STREAM, 0.0))
    marge = MARGINAL_FRACTION * charge
    residus = []

    post = deltas.get(STATE_POST_STREAM)
    if post is not None and abs(post / dispersion) > IDLE_STATE_SIGMA and abs(post) <= marge:
        residus.append(
            f"{STATE_POST_STREAM} s'écarte de {post * ma:+.3f} mA "
            f"({post / dispersion:+.1f}σ) mais reste sous {MARGINAL_FRACTION:g} × le "
            f"surcoût de charge mesuré ({charge * ma:.3f} mA) : significatif, négligeable"
        )
        post = None
    if post is not None and abs(post / dispersion) > IDLE_STATE_SIGMA:
        return VERDICT_POST_FLUX, (
            f"le repos mesuré après interruption d'un flux n'est PAS le même état que le "
            f"repos port fermé (> {IDLE_STATE_SIGMA:g}σ) : les fenêtres « repos » du "
            f"protocole contre-balancé sont prises dans cet état-là, il faut maîtriser la "
            f"fermeture du flux et rejouer S5301. {commun}"
        )

    port = deltas.get(STATE_PORT_OPEN)
    if port is not None and abs(port / dispersion) > IDLE_STATE_SIGMA and abs(port) <= marge:
        residus.append(
            f"{STATE_PORT_OPEN} s'écarte de {port * ma:+.3f} mA "
            f"({port / dispersion:+.1f}σ) mais reste sous {MARGINAL_FRACTION:g} × le "
            f"surcoût de charge mesuré ({charge * ma:.3f} mA) : significatif, négligeable"
        )
        port = None
    if port is not None and abs(port / dispersion) > IDLE_STATE_SIGMA:
        return VERDICT_PORT_STATE, (
            f"l'ouverture seule du port de la carte, sans aucun trafic, déplace le courant "
            f"de plus de {IDLE_STATE_SIGMA:g}σ : l'état des lignes de contrôle de l'hôte "
            f"fait partie de la condition et doit être fixé explicitement. {commun}"
        )

    suffixe = (" Résidus retenus comme négligeables : " + " ; ".join(residus) + ".") if residus else ""
    return VERDICT_REPOS_COHERENT, (
        f"les états de repos sont interchangeables à l'échelle du surcoût de charge : ni "
        f"l'ouverture du port ni l'interruption du flux n'explique l'écart entre sessions, "
        f"la cause est ailleurs. {commun}{suffixe}"
    )
