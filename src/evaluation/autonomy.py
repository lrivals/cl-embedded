"""
autonomy.py — Autonomie batterie estimée à partir des µJ/phase mesurés (S3307).

Dernière étape de la chaîne énergie du Sprint 33 : dériver une **autonomie estimée**
(heures) à partir des µJ par phase mesurés par le PowerShield LPM01A (S3306) et d'une
capacité de batterie typique. Répond à la question du CR 9 juin 2026 : « combien
d'accuracy perd-on pour gagner en RAM/autonomie ? ».

Chaîne de calcul :
    I_phase (A) = (uJ_phase / 1e6) / (V × t_phase)         [énergie → courant moyen]
    I_moy (mA)  = Σ(I_phase × t_phase) / T_cycle × 1000     [courant moyen pondéré]
    Autonomie_h = Capacité_mAh / I_moy_mA

Règle CLAUDE.md :
    - Capacités batterie lues depuis `configs/hw_profile_f439zi.yaml` (section
      `batterie.capacites_mah`), jamais en dur.
    - AUCUN CHIFFRE INVENTÉ : si les µJ/phase valent encore ``"à mesurer"``
      (placeholder S3306, LPM01A non exécuté), les fonctions propagent ce statut au
      lieu de fabriquer un courant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Valeur littérale partagée avec scripts/energy_capture.py (placeholder S3306).
A_MESURER = "à mesurer"


def _is_measured(value: Any) -> bool:
    """True si `value` est un nombre exploitable (pas un placeholder ``"à mesurer"``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_current_ma(
    phases_uj: dict, phase_durations_s: dict, tension_v: float = 3.3
) -> float:
    """Courant moyen I_moy (mA) sur un cycle, dérivé des µJ/phase et de leurs durées.

    I_moy = Σ(E_phase / V) / T_cycle, où E_phase = uJ_phase / 1e6 (joules) et
    T_cycle = Σ t_phase. Équivalent à Σ(I_phase × t_phase) / T_cycle.

    Parameters
    ----------
    phases_uj : dict
        Énergie par phase en µJ (cf. `experiments/exp_S33_energy/*.json`).
    phase_durations_s : dict
        Durée de chaque phase (s). Seules les phases présentes dans `phases_uj`
        ET `phase_durations_s` sont prises en compte.
    tension_v : float
        Tension d'alimentation (V), défaut 3.3 V (NUCLEO-F439ZI).

    Returns
    -------
    float
        Courant moyen en milliampères (mA).

    Raises
    ------
    ValueError
        Si la tension est ≤ 0, si le cycle total est nul, ou si une énergie
        exploitée n'est pas mesurée (placeholder ``"à mesurer"``).
    """
    if tension_v <= 0.0:
        raise ValueError(f"tension_v doit être > 0, reçu {tension_v!r}.")

    total_cycle_s = 0.0
    total_charge_as = 0.0  # Σ (E_phase / V) en ampère·seconde (= coulombs)
    for phase, t in phase_durations_s.items():
        if phase not in phases_uj:
            continue
        e_uj = phases_uj[phase]
        if not _is_measured(e_uj):
            raise ValueError(
                f"Énergie phase {phase!r} non mesurée ({e_uj!r}) : impossible de "
                f"calculer I_moy sans mesure LPM01A réelle (règle 'aucun chiffre inventé')."
            )
        energy_j = float(e_uj) / 1e6
        total_charge_as += energy_j / tension_v
        total_cycle_s += float(t)

    if total_cycle_s <= 0.0:
        raise ValueError("Durée de cycle totale nulle : fournir des durées de phase > 0.")

    i_moy_a = total_charge_as / total_cycle_s
    return i_moy_a * 1000.0


def average_current_ma_from_delta(
    i_idle_a: float,
    energy_uj_per_inference: float,
    inference_period_s: float,
    tension_v: float = 3.3,
) -> float:
    """Courant moyen I_moy (mA) à partir du **protocole delta** (S5008).

    Voie alternative à `average_current_ma` lorsque le profil par phase est
    inaccessible : le LPM01A n'a pas de voie de synchronisation et son mode
    dynamique est hors de portée du courant de la F439ZI, si bien que la campagne
    ne produit pas de `phases_uj`. Elle produit en revanche deux grandeurs
    **mesurées** — le courant au repos et l'énergie marginale par inférence — dont
    se déduit le courant moyen d'un cycle applicatif :

        Q_inf (C)  = (uJ_inf / 1e6) / V              [charge marginale d'une inférence]
        I_moy (mA) = (I_repos + Q_inf / T_periode) × 1000

    Aucune extrapolation : `i_idle_a` et `energy_uj_per_inference` viennent tous deux
    de la mesure, `inference_period_s` est l'hypothèse d'usage déclarée du manifeste
    (`duty_cycle.inference_period_s`) — c'est un paramètre de scénario, pas un chiffre
    inventé, et il est reporté tel quel dans le rapport.

    Parameters
    ----------
    i_idle_a : float
        Courant moyen au repos mesuré (A) — `delta_measurement.i_idle_a`.
    energy_uj_per_inference : float
        Énergie marginale par inférence mesurée (µJ).
    inference_period_s : float
        Période entre deux inférences dans le scénario d'usage (s).
    tension_v : float
        Tension d'alimentation (V), défaut 3.3 V (NUCLEO-F439ZI).

    Returns
    -------
    float
        Courant moyen en milliampères (mA).

    Raises
    ------
    ValueError
        Si `tension_v` ≤ 0 ou `inference_period_s` ≤ 0.
    """
    if tension_v <= 0.0:
        raise ValueError(f"tension_v doit être > 0, reçu {tension_v!r}.")
    if inference_period_s <= 0.0:
        raise ValueError(
            f"inference_period_s doit être > 0, reçu {inference_period_s!r}."
        )
    charge_c = (float(energy_uj_per_inference) / 1e6) / float(tension_v)
    return (float(i_idle_a) + charge_c / float(inference_period_s)) * 1e3


def autonomy_hours(capacite_mah: float, i_moy_ma: float) -> float:
    """Autonomie estimée (heures) = Capacité_mAh / I_moy_mA.

    Parameters
    ----------
    capacite_mah : float
        Capacité de la batterie en mAh.
    i_moy_ma : float
        Courant moyen consommé en mA.

    Returns
    -------
    float
        Autonomie en heures.

    Raises
    ------
    ValueError
        Si `i_moy_ma` ≤ 0.
    """
    if i_moy_ma <= 0.0:
        raise ValueError(f"i_moy_ma doit être > 0, reçu {i_moy_ma!r}.")
    return capacite_mah / i_moy_ma


def sweep_capacities(
    i_moy_ma: float, capacites_mah: list[float]
) -> dict[float, float]:
    """Balaye un ensemble de capacités batterie → autonomie (heures).

    Parameters
    ----------
    i_moy_ma : float
        Courant moyen (mA).
    capacites_mah : list[float]
        Capacités à évaluer (typiquement issues de `load_battery_capacities`).

    Returns
    -------
    dict[float, float]
        ``{capacite_mah: autonomie_h}`` pour chaque capacité.
    """
    return {float(c): autonomy_hours(float(c), i_moy_ma) for c in capacites_mah}


def load_battery_capacities(
    path: str | Path = "configs/hw_profile_f439zi.yaml",
) -> list[float]:
    """Charge les capacités batterie typiques depuis la config (jamais en dur).

    Parameters
    ----------
    path : str | Path
        Profil HW contenant la section ``batterie.capacites_mah``.

    Returns
    -------
    list[float]
        Liste des capacités en mAh.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return [float(c) for c in cfg["batterie"]["capacites_mah"]]
