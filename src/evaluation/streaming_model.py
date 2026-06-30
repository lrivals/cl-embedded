"""
streaming_model.py — Modèle analytique de débit/buffer pour streaming embarqué (S3401).

Formalise les formules du CR du 19 mai 2026 (« évaluer la latence pour estimer combien
de données on peut streamer sur la carte par modèle », « impact du stride S », « remplissage
du buffer ») :

    Débit_max       (Hz) = 1 / Latence_inf        — borne soutenable par un modèle donné
    Débit_streaming (Hz) = f_acq × S / W          — fréquence de production de fenêtres
    Marge temps-réel      : Débit_streaming ≤ Débit_max (sinon accumulation de retard)
    Budget buffer (octets) = W × sizeof(sample)   — à comparer à la SRAM disponible

La latence d'inférence (`latence_inf_s`) provient des mesures DWT réelles produites par les
sprints précédents (`firmware/stm32f4_blink/src/profiling.c`, exp_S30/S31/S32), pas d'une
estimation analytique : ce module ne fait que combiner des chiffres mesurés.

Toutes les tailles (W, S, f_acq, sizeof(sample), SRAM) sont lues depuis
`configs/streaming_profile.yaml` (règle CLAUDE.md : jamais de constante de taille en dur).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def debit_max(latence_inf_s: float) -> float:
    """Débit maximal soutenable (Hz) par un modèle, d'après sa latence d'inférence.

    Parameters
    ----------
    latence_inf_s : float
        Latence d'une inférence (+ mise à jour) en secondes, mesurée par DWT sur la board.

    Returns
    -------
    float
        Débit_max = 1 / latence_inf, en Hz — au-delà, le modèle accumule du retard.

    Raises
    ------
    ValueError
        Si `latence_inf_s` est nulle ou négative.
    """
    if latence_inf_s <= 0.0:
        raise ValueError(f"latence_inf_s doit être > 0, reçu {latence_inf_s}")
    return 1.0 / latence_inf_s


def debit_streaming(f_acq_hz: float, stride: int, window: int) -> float:
    """Fréquence effective de production de nouvelles fenêtres d'inférence (Hz).

    Une nouvelle fenêtre est produite tous les `stride` échantillons acquis ; pour une
    fenêtre de taille `window`, la fréquence de fenêtres est `f_acq × stride / window`.

    Parameters
    ----------
    f_acq_hz : float
        Fréquence d'acquisition capteur (Hz).
    stride : int
        Pas entre deux fenêtres consécutives (échantillons).
    window : int
        Taille de la fenêtre d'inférence (échantillons).

    Returns
    -------
    float
        Débit_streaming en Hz.

    Raises
    ------
    ValueError
        Si `window` ≤ 0 ou `stride` ≤ 0.
    """
    if window <= 0:
        raise ValueError(f"window doit être > 0, reçu {window}")
    if stride <= 0:
        raise ValueError(f"stride doit être > 0, reçu {stride}")
    return f_acq_hz * stride / window


def marge_temps_reel(debit_streaming_hz: float, debit_max_hz: float) -> dict[str, Any]:
    """Marge temps-réel entre débit produit et débit soutenable.

    Parameters
    ----------
    debit_streaming_hz : float
        Débit de production de fenêtres (cf. :func:`debit_streaming`).
    debit_max_hz : float
        Débit maximal soutenable par le modèle (cf. :func:`debit_max`).

    Returns
    -------
    dict
        ``{"ok": bool, "marge_pct": float}`` — ``ok`` vrai si le streaming tient
        (``debit_streaming ≤ debit_max``) ; ``marge_pct`` = ``(debit_max - debit_streaming)
        / debit_max`` (positif = marge, négatif = saturation).
    """
    if debit_max_hz <= 0.0:
        raise ValueError(f"debit_max_hz doit être > 0, reçu {debit_max_hz}")
    marge = (debit_max_hz - debit_streaming_hz) / debit_max_hz
    return {"ok": debit_streaming_hz <= debit_max_hz, "marge_pct": marge}


def budget_buffer_bytes(window: int, sizeof_sample: int) -> int:
    """Empreinte mémoire du buffer de fenêtre : ``W × sizeof(sample)``.

    Parameters
    ----------
    window : int
        Taille de fenêtre (échantillons).
    sizeof_sample : int
        Taille d'un échantillon en octets (4 pour float32).

    Returns
    -------
    int
        Nombre d'octets occupés par le buffer.
    """
    if window <= 0 or sizeof_sample <= 0:
        raise ValueError(f"window et sizeof_sample doivent être > 0, reçu {window}, {sizeof_sample}")
    return window * sizeof_sample


def check_sram_budget(buffer_bytes: int, sram_bytes: int) -> bool:
    """Vérifie que le buffer tient dans le budget SRAM.

    Parameters
    ----------
    buffer_bytes : int
        Empreinte du buffer (cf. :func:`budget_buffer_bytes`).
    sram_bytes : int
        Budget SRAM disponible (octets), p. ex. 65536 (64 Ko, Gap 2).

    Returns
    -------
    bool
        Vrai si ``buffer_bytes ≤ sram_bytes``.
    """
    return buffer_bytes <= sram_bytes


def load_profile(path: str | Path) -> dict[str, Any]:
    """Charge un profil de streaming YAML et renvoie la section ``streaming``.

    Parameters
    ----------
    path : str | Path
        Chemin vers ``configs/streaming_profile.yaml``.

    Returns
    -------
    dict
        Contenu de la clé ``streaming`` (W, S, f_acq, sizeof, SRAM, sweep, latences).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("streaming", cfg)
