"""dyn_threshold.py — Nature du plafond du mode dynamique de la sonde (S5303, B6).

LE FAIT QUI MOTIVE CE MODULE

Le X-NUCLEO-LPM01A refuse l'acquisition dynamique « au-delà de 59 mA » — c'est ce
qu'annonce son message d'erreur. Les mesures du Sprint 53 contredisent la lecture naïve de
ce plafond : à 45 MHz l'acquisition PASSE avec un pic relevé à 68,60 mA, à 90 MHz elle
ÉCHOUE avec 69,58 mA. Un écart d'un milliampère sépare les deux, alors que les deux sont
dix milliampères au-dessus du seuil annoncé. Ce n'est donc ni « le pic dépasse 59 mA », ni
rien qui ait été établi : la nature du déclencheur est inconnue, et trois candidats se
distinguent — le PIC instantané, la MOYENNE du courant, ou la DURÉE passée en surintensité.

Ce module porte la règle de décision qui les départage, et le refus de conclure quand
aucun ne sépare les essais. Il ne mesure rien : le pilote `scripts/run_s53_dyn_threshold.py`
produit la matrice d'essais, ce module dit ce qu'elle démontre.

Il porte aussi la règle, commune à toute la campagne, qui distingue une acquisition
dynamique RÉUSSIE d'une acquisition INTERROMPUE : la sonde ne lève rien quand elle
s'arrête en surintensité, elle rend les quelques dizaines de millisecondes décodées avant
l'arrêt. Sans ce contrôle, une acquisition tronquée passe pour un succès — défaut mesuré
et corrigé en séance sur `run_s53_freq_sweep.try_dynamic_mode`, et que
`run_s53_phase_profile.capture_under_load` portait encore.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: Fraction des échantillons attendus en-deçà de laquelle une acquisition dynamique est
#: tenue pour INTERROMPUE. Valeur d'origine : `run_s53_freq_sweep.DYN_COMPLETENESS`.
COMPLETENESS = 0.95

#: Marqueurs d'interruption dans le résumé rendu par la sonde.
INTERRUPTION_MARKERS = ("interrupted", "overcurrent")

#: Plafond de plausibilité d'un courant décodé, en ampères. Symétrique du garde-fou bas de
#: `lpm01a_probe.capture` (A2, 1 mA), et né du même genre de constat : après réouverture de
#: session, des octets résiduels du tampon série se décodent en échantillons et rendent des
#: valeurs sans rapport avec la mesure — 3 338 A relevés le 2026-09-08. La NUCLEO-F439ZI
#: n'a jamais dépassé 76 mA sur ce banc ; un ampère est deux ordres de grandeur au-dessus,
#: donc hors de tout régime réel sans risquer d'écarter une mesure valide.
MAX_PLAUSIBLE_CURRENT_A = 1.0

#: Purge du tampon série après réouverture d'une session : silence exigé, et durée maximale
#: d'attente. La sonde continue d'émettre après un arrêt d'acquisition ; sans cette seconde
#: purge (plus patiente que celle de `take_control`), les octets restants se décodent en
#: échantillons aberrants à l'acquisition suivante.
RECOVERY_QUIET_S = 1.0
RECOVERY_MAX_S = 5.0

#: Durée de l'acquisition de rebut qui suit une réouverture de session. Courte : elle ne
#: sert qu'à remettre la sonde en état, sa valeur n'est jamais retenue.
RECOVERY_WARMUP_S = 1.0

#: Candidats testés comme déclencheur de l'arrêt, et le champ d'essai qui les porte.
CANDIDATES = {
    "pic": "i_max_ma",
    "moyenne": "i_mean_ma",
    "durée": "duration_s",
}

VERDICT_INDETERMINE = "indéterminé"

#: Dispersion relative en deçà de laquelle un temps d'arrêt est tenu pour CONSTANT. 10 % :
#: si la sonde s'arrêtait sur un seuil de courant ou d'énergie accumulée, le temps d'arrêt
#: suivrait la charge — un temps stable à 10 % près sur des charges qui varient d'un facteur
#: dix est le signe contraire.
ABORT_TIME_SPREAD = 0.10


def acquisition_outcome(n_decoded: int, n_expected: int, summary: str) -> dict:
    """Une acquisition dynamique a-t-elle abouti ? — règle unique de la campagne.

    Deux signes, tous deux nécessaires : la sonde n'a pas annoncé d'interruption, ET elle
    a rendu au moins `COMPLETENESS` des échantillons demandés. `lpm01a_probe.capture` ne
    lève que sur un flux VIDE : sans ce contrôle, une acquisition coupée à 20 % passerait
    pour un succès et débloquerait à tort le profil temporel par phase (S5305).
    """
    texte = (summary or "").lower()
    interrompu = any(mot in texte for mot in INTERRUPTION_MARKERS)
    complet = n_expected > 0 and n_decoded >= int(COMPLETENESS * n_expected)
    bloc = {
        "succeeded": bool(complet and not interrompu),
        "n_samples": int(n_decoded),
        "n_samples_expected": int(n_expected),
        "interrupted_by_probe": interrompu,
        # AUCUNE donnée décodée n'est pas le même événement qu'un arrêt en cours
        # d'acquisition. Une acquisition très courte peut ne rien rendre pour une raison
        # qui tient à l'instrument (0/5000 échantillons relevés à 0,05 s le 2026-09-08) :
        # cela ne renseigne pas sur ce qui déclenche l'arrêt en surintensité, et un tel
        # essai ne doit donc pas peser dans le verdict comme s'il en était un.
        "no_data": int(n_decoded) == 0,
    }
    if not bloc["succeeded"]:
        bloc["na_reason"] = (
            f"acquisition dynamique non aboutie : {n_decoded}/{n_expected} échantillons "
            f"décodés"
            + (" ; la sonde annonce une interruption" if interrompu else
               " ; aucune donnée décodée — l'essai ne renseigne pas sur le déclencheur"
               if bloc["no_data"] else
               f" (moins de {COMPLETENESS:.0%} des échantillons demandés)")
        )
    return bloc


def current_trustworthy(i_max_a: float) -> bool:
    """La trace décodée est-elle un courant, ou du tampon série mal interprété ?

    Une acquisition interrompue reste une observation valide — c'est l'arrêt lui-même qu'on
    mesure — mais son COURANT ne l'est plus dès que le décodage a dérivé. Le distinguer
    permet de garder l'essai dans l'axe « durée » tout en l'écartant des axes « pic » et
    « moyenne », au lieu de jeter l'essai entier ou, pire, de classer sur du bruit.
    """
    return 0.0 < float(i_max_a) < MAX_PLAUSIBLE_CURRENT_A


def _sans_donnee(essai: Mapping[str, Any]) -> bool:
    """Aucun échantillon n'est revenu de cet essai.

    Déduit du COMPTE autant que du drapeau : un essai peut avoir été consigné par un
    chemin d'erreur qui ne pose pas `no_data` (c'était le cas des refus levés par la
    sonde), et la règle doit alors s'appliquer quand même — y compris rétroactivement aux
    essais déjà écrits.
    """
    if essai.get("no_data"):
        return True
    n = essai.get("n_samples")
    return isinstance(n, int) and n == 0


def _valeur(essai: Mapping[str, Any], champ: str) -> float | None:
    v = essai.get(champ)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def separability(essais: Sequence[Mapping[str, Any]], champ: str) -> dict:
    """Un seuil sur `champ` sépare-t-il parfaitement les essais réussis des échoués ?

    Séparabilité au sens strict : tous les succès d'un côté, tous les échecs de l'autre.
    Le sens est déduit des données (le déclencheur peut aussi bien être un maximum qu'un
    minimum), la marge est la largeur de l'intervalle où le seuil peut vivre, et un
    recouvrement est CHIFFRÉ au lieu d'être arrondi en « non concluant ».
    """
    ok = [v for e in essais if e.get("succeeded")
          if (v := _valeur(e, champ)) is not None]
    ko = [v for e in essais if not e.get("succeeded")
          if (v := _valeur(e, champ)) is not None]
    bloc = {"field": champ, "n_ok": len(ok), "n_ko": len(ko)}
    if not ok or not ko:
        bloc.update({
            "separable": False,
            "reason": ("aucun essai n'a échoué" if not ko else "aucun essai n'a abouti")
            + " : la séparation n'est pas testable sur ce champ.",
        })
        return bloc

    bloc.update({"max_ok": max(ok), "min_ok": min(ok),
                 "max_ko": max(ko), "min_ko": min(ko)})
    if max(ok) < min(ko):          # échec au-DESSUS d'un seuil
        bloc.update({"separable": True, "direction": "échec au-dessus du seuil",
                     "threshold_between": [max(ok), min(ko)],
                     "margin": min(ko) - max(ok)})
    elif min(ok) > max(ko):        # échec en-DESSOUS d'un seuil
        bloc.update({"separable": True, "direction": "échec en-dessous du seuil",
                     "threshold_between": [max(ko), min(ok)],
                     "margin": min(ok) - max(ko)})
    else:
        recouvrement = min(max(ok), max(ko)) - max(min(ok), min(ko))
        bloc.update({
            "separable": False,
            "overlap": recouvrement,
            "reason": (f"les essais réussis ({min(ok):.3f} … {max(ok):.3f}) et échoués "
                       f"({min(ko):.3f} … {max(ko):.3f}) se recouvrent : aucun seuil sur "
                       f"ce champ ne les sépare."),
        })
    return bloc


def classify(essais: Sequence[Mapping[str, Any]]) -> dict:
    """Lequel des trois candidats déclenche l'arrêt ? — verdict calculé, jamais saisi.

    Trois issues, toutes publiables :

    * un seul candidat sépare les essais → il est nommé, avec son seuil encadré et sa
      marge ;
    * plusieurs séparent → ``indéterminé`` : la matrice d'essais ne les a pas
      décorrélés, et nommer le premier venu serait choisir au hasard. Les candidats
      concurrents sont listés ;
    * aucun ne sépare → ``indéterminé``, avec le recouvrement chiffré de chacun.

    L'issue « indéterminé » n'est pas un échec de mesure : c'est la borne de ce que la
    matrice démontre.
    """
    # Les essais SANS DONNÉE sont écartés du verdict : ils constatent qu'aucun échantillon
    # n'est revenu, pas qu'un seuil a été franchi. Ils restent comptés, pour que le lecteur
    # sache combien d'essais la matrice a réellement mis en jeu.
    retenus = [e for e in essais if not _sans_donnee(e)]
    tests = {nom: separability(retenus, champ) for nom, champ in CANDIDATES.items()}
    gagnants = [nom for nom, t in tests.items() if t.get("separable")]
    bloc = {"candidates": tests, "n_attempts": len(essais),
            "n_no_data": len(essais) - len(retenus),
            "n_considered": len(retenus),
            "n_succeeded": sum(1 for e in retenus if e.get("succeeded"))}
    essais = retenus

    if len(gagnants) == 1:
        nom = gagnants[0]
        t = tests[nom]
        bornes = t["threshold_between"]
        bloc.update({
            "verdict": nom,
            "rationale": (
                f"sur {len(essais)} essais dont {bloc['n_succeeded']} aboutis, seul le "
                f"candidat « {nom} » ({t['field']}) sépare les essais : {t['direction']}, "
                f"seuil entre {bornes[0]:.3f} et {bornes[1]:.3f} (marge {t['margin']:.3f}). "
                f"Les autres candidats se recouvrent."),
        })
        return bloc
    if len(gagnants) > 1:
        bloc.update({
            "verdict": VERDICT_INDETERMINE,
            "rationale": (
                f"{len(gagnants)} candidats séparent les essais ({', '.join(gagnants)}) : "
                f"la matrice ne les a pas décorrélés, les désigner reviendrait à choisir. "
                f"Il faut un essai où ils divergent."),
            "competing": gagnants,
        })
        return bloc
    bloc.update({
        "verdict": VERDICT_INDETERMINE,
        "rationale": (
            f"aucun des {len(CANDIDATES)} candidats ne sépare les {len(essais)} essais "
            f"({bloc['n_succeeded']} aboutis) : "
            + " ".join(f"{nom} — {t.get('reason', '')}" for nom, t in tests.items())),
    })
    return bloc


def abort_time_stability(essais: Sequence[Mapping[str, Any]]) -> dict:
    """Le temps réellement acquis avant l'arrêt dépend-il de la charge ?

    C'est le second observable de la matrice, et il discrimine là où les courants ne le
    font pas : si l'arrêt était provoqué par un seuil de courant ou par une énergie
    accumulée, le temps tenu avant l'arrêt devrait RACCOURCIR quand la charge monte. S'il
    ne bouge pas alors que la charge varie d'un facteur dix, l'arrêt ressemble davantage à
    une limite fixe de l'instrument qu'à une réaction à ce que consomme la carte.

    Ne considère que les essais ARRÊTÉS ayant rendu des données : un essai abouti n'a pas
    de temps d'arrêt, un essai sans donnée n'en a pas non plus.
    """
    temps = [(float(e["time_acquired_s"]), float(e.get("rate_hz", 0.0)))
             for e in essais
             if not e.get("succeeded") and not e.get("no_data")
             and isinstance(e.get("time_acquired_s"), (int, float))]
    if len(temps) < 2:
        return {"n": len(temps), "constant": None,
                "na_reason": "moins de deux arrêts exploitables : la stabilité du temps "
                             "d'arrêt n'est pas testable."}
    valeurs = [t for t, _ in temps]
    moyenne = sum(valeurs) / len(valeurs)
    etendue = max(valeurs) - min(valeurs)
    dispersion = etendue / moyenne if moyenne > 0 else None
    charges = sorted({r for _, r in temps})
    bloc = {
        "n": len(temps),
        "mean_s": moyenne,
        "min_s": min(valeurs),
        "max_s": max(valeurs),
        "spread_ratio": dispersion,
        "rates_hz": charges,
        "constant": bool(dispersion is not None and dispersion <= ABORT_TIME_SPREAD),
    }
    bloc["rationale"] = (
        f"{len(temps)} arrêts exploitables, temps acquis {min(valeurs):.3f}–"
        f"{max(valeurs):.3f} s (moyenne {moyenne:.3f} s, dispersion "
        f"{dispersion:.1%}) sur des cadences {charges[0]:.0f}–{charges[-1]:.0f} Hz : "
        + ("le temps d'arrêt ne suit pas la charge, ce qui oriente vers une limite fixe de "
           "l'instrument plutôt que vers une réaction au courant consommé."
           if bloc["constant"] else
           "le temps d'arrêt varie avec les conditions ; il ne peut pas être tenu pour une "
           "limite fixe.")
    )
    return bloc
