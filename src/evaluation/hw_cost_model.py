"""
hw_cost_model.py — Modèle de coût matériel analytique (proxy temps-HW / FLOPS-W).

Répond au CR du 19 mai 2026 : « une formule pour estimer le nombre de calculs en
fonction du matériel ». Fournit des formules paramétrées reliant le nombre de MACs
(cf. src/evaluation/compute_cost.py) au temps d'inférence estimé, au rendement
énergétique (FLOPS/W) et au débit (inférences/s).

IMPORTANT — c'est un PROXY analytique, PAS une mesure :
- `estimate_inference_time` repose sur un coefficient d'efficacité `efficacite`
  (fraction du pic FLOPs réellement atteinte), incertain par construction
  (intervalle documenté [0.1, 0.6], cf. CR). Il ne remplace pas la latence DWT
  réelle mesurée par firmware/stm32f4_blink/src/profiling.c.
- Toutes les constantes matérielles (FLOPS pic, efficacité, tension/courant)
  proviennent de configs/hw_profile_f439zi.yaml — jamais codées en dur ici
  (règle CLAUDE.md). Ce module ne contient que des formules.

Référence horloge : SYSCLK = 180 MHz (cf. profiling.c, NUCLEO-F439ZI Cortex-M4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Bornes documentaires du coefficient d'efficacité (CR 19 mai 2026).
EFFICACITE_MIN = 0.1
EFFICACITE_MAX = 0.6


def estimate_inference_time(macs: int, flops_peak: float, efficacite: float) -> float:
    """Estime le temps d'inférence T_HW (secondes) — PROXY analytique.

    T_HW ≈ (2 × macs) / (flops_peak × efficacite)

    où `2 × macs` = FLOPs (1 mult + 1 add par MAC, cf. compute_cost.compute_flops).
    Le coefficient `efficacite` modélise la fraction du pic FLOPs réellement
    soutenue (pipeline, accès mémoire, overhead) ; il est incertain par nature.

    Parameters
    ----------
    macs : int
        Nombre de multiply-accumulates de l'inférence (cf. compute_cost).
    flops_peak : float
        FLOPs/s pic théorique du matériel (depuis le profil HW).
    efficacite : float
        Rendement effectif ∈ [0.1, 0.6] (proxy documenté, cf. CR 19 mai).

    Returns
    -------
    float
        Temps d'inférence estimé en secondes. À ne PAS confondre avec la
        latence DWT mesurée (profiling.c).

    Raises
    ------
    ValueError
        Si `flops_peak` ou `efficacite` ≤ 0 (division impossible / non physique).
    """
    if flops_peak <= 0.0:
        raise ValueError(f"flops_peak doit être > 0, reçu {flops_peak!r}.")
    if efficacite <= 0.0:
        raise ValueError(f"efficacite doit être > 0, reçu {efficacite!r}.")
    flops = 2 * macs
    return flops / (flops_peak * efficacite)


def flops_per_watt(flops_peak: float, puissance_watts: float) -> float:
    """Rendement énergétique FLOPS/W (proxy).

    Parameters
    ----------
    flops_peak : float
        FLOPs/s pic théorique du matériel.
    puissance_watts : float
        Puissance consommée en watts (cf. helper power_watts ou mesure LPM01A).

    Returns
    -------
    float
        FLOPs par watt.

    Raises
    ------
    ValueError
        Si `puissance_watts` ≤ 0.
    """
    if puissance_watts <= 0.0:
        raise ValueError(f"puissance_watts doit être > 0, reçu {puissance_watts!r}.")
    return flops_peak / puissance_watts


def throughput(temps_inference_s: float) -> float:
    """Débit en inférences/seconde, inverse du temps d'inférence.

    Parameters
    ----------
    temps_inference_s : float
        Temps d'une inférence en secondes (cf. estimate_inference_time).

    Returns
    -------
    float
        Inférences par seconde.

    Raises
    ------
    ValueError
        Si `temps_inference_s` ≤ 0.
    """
    if temps_inference_s <= 0.0:
        raise ValueError(f"temps_inference_s doit être > 0, reçu {temps_inference_s!r}.")
    return 1.0 / temps_inference_s


def power_watts(courant_ma: float, tension_v: float) -> float:
    """Puissance active (W) = tension × courant, à partir du profil HW.

    Parameters
    ----------
    courant_ma : float
        Courant actif en milliampères (cf. puissance_watts.actif_mA du profil).
    tension_v : float
        Tension d'alimentation en volts (cf. puissance_watts.tension_v).

    Returns
    -------
    float
        Puissance en watts.
    """
    return (courant_ma / 1000.0) * tension_v


def load_hw_profile(path: str | Path) -> dict[str, Any]:
    """Charge le profil matériel YAML (configs/hw_profile_f439zi.yaml).

    Parameters
    ----------
    path : str | Path
        Chemin vers le fichier de profil HW.

    Returns
    -------
    dict
        Contenu du profil (clé `hardware` : sysclk_hz, flops_peak_*, efficacite,
        puissance_watts).
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
