"""
src/evaluation/drift_metrics.py — Sprint 44 (S4404) : harnais d'évaluation des détecteurs de drift.

Un détecteur de drift ne s'évalue **pas** comme un classifieur : ce qui compte est la **vitesse** de
signalement d'un vrai drift (délai), le **nombre de fausses alarmes** sur les segments stables, et le
**nombre de drifts manqués**. On y ajoute le **coût** (état mémoire, latence) car la finalité est le
portage MCU (Sprint 45).

Fonctions pures (miroir de ``anomaly_metrics.py``) — testables sans détecteur réel à partir de verdicts
synthétiques et de points de drift connus :

- :func:`alarms_from_verdicts` — source unique du mapping verdict → alarme booléenne (enum/str/int).
- :func:`compute_drift_metrics` — métriques de détection vs ``drift_points`` (délai, FAR, MDR, P/R/F1,
  MTFA, MTD), avec gestion honnête ``None`` quand la vérité-terrain ponctuelle est absente.
- :func:`profile_drift_detector` — coût **proxy PC** (``state_bytes`` algorithmique + RAM tracemalloc +
  latence), explicitement étiqueté ``_proxy: True`` (les chiffres board réels viennent de S45).
- :func:`build_comparison_table` — table détecteur × dataset sérialisable JSON.
- :func:`save_drift_metrics` — écriture JSON (miroir ``save_anomaly_metrics``).

Convention d'honnêteté (héritée S33/S38/S43) : ``None`` (jamais 0) pour non calculable / non mesuré.

Références
----------
    Gama et al. 2014 (survey drift) · théorie des change-points (MTFA/MTD) ·
    ``docs/context/drift_detectors.md`` (source de vérité S4401).
"""

from __future__ import annotations

import json
import time
import tracemalloc
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

# Étiquette de verdict considérée comme une **alarme** (déclenchement). WARNING / FAULT ne comptent PAS
# comme alarme de drift confirmé (documenté S4401 : WARNING = zone d'alerte, FAULT = panne instantanée).
_ALARM_LABELS = {"DRIFT"}


def alarms_from_verdicts(verdicts: Iterable) -> list[bool]:
    """Mappe une séquence de verdicts hétérogènes vers des booléens d'alarme (source unique).

    Tolère les trois représentations coexistant dans le projet :

    - ``DriftVerdict`` (enum ``src/models/drift/base.py`` : ``NORMAL/WARNING/DRIFT``) — via ``.name`` ;
    - chaînes du baseline ``SlidingWindowDriftDetector`` (``"NORMAL"/"FAULT"/"DRIFT"``) ;
    - entiers/bool déjà réduits (``2``/``True`` = alarme).

    Parameters
    ----------
    verdicts : Iterable
        Verdicts émis échantillon par échantillon.

    Returns
    -------
    list[bool]
        ``True`` là où le verdict est un DRIFT confirmé.
    """
    alarms: list[bool] = []
    for v in verdicts:
        if isinstance(v, bool):
            alarms.append(v)
        elif isinstance(v, (int, np.integer)):
            alarms.append(int(v) == 2)  # DriftVerdict.DRIFT == 2
        elif hasattr(v, "name"):  # enum DriftVerdict
            alarms.append(v.name in _ALARM_LABELS)
        else:  # chaîne
            alarms.append(str(v).upper() in _ALARM_LABELS)
    return alarms


def _covered_mask(drift_points: Sequence[int], n_samples: int, tolerance: int) -> np.ndarray:
    """Masque booléen des échantillons couverts par une fenêtre de tolérance ``[dp, dp+tolerance]``."""
    mask = np.zeros(n_samples, dtype=bool)
    for dp in drift_points:
        lo = max(int(dp), 0)
        hi = min(int(dp) + tolerance + 1, n_samples)  # +1 : borne inclusive
        mask[lo:hi] = True
    return mask


def compute_drift_metrics(
    alarms: Sequence[bool],
    drift_points: Sequence[int] | None,
    n_samples: int,
    tolerance: int,
) -> dict:
    """Calcule les métriques de détection de drift vs la vérité-terrain ``drift_points``.

    Chaque vrai point de drift est apparié à la **première** alarme tombant dans sa fenêtre de
    tolérance ``[dp, dp+tolerance]`` (appariement glouton, une alarme au plus par point). Le délai est
    la distance à cette alarme ; un point sans alarme dans la fenêtre est **manqué** ; une alarme hors
    de **toute** fenêtre est une **fausse alarme**.

    Parameters
    ----------
    alarms : Sequence[bool]
        Alarmes booléennes (cf. :func:`alarms_from_verdicts`).
    drift_points : Sequence[int] | None
        Indices des vrais points de drift. ``None`` (Electricity/NOAA) → délai/MDR/P/R/F1/MTD non
        calculables (``None`` honnête) ; seuls FAR/MTFA sur flux réputé stable sont renvoyés.
    n_samples : int
        Longueur du flux.
    tolerance : int
        Largeur (en échantillons) de la fenêtre de détection après chaque point de drift.

    Returns
    -------
    dict
        ``mean_detection_delay, missed_detection_rate, false_alarm_rate, precision, recall, f1,
        mtfa, mtd, n_true_drifts, n_alarms, n_detected, n_false_alarms, tolerance``. Les champs non
        calculables valent ``None``.
    """
    alarms = np.asarray(alarms, dtype=bool)
    if alarms.shape[0] != n_samples:
        # Tolérant : on aligne sur la longueur fournie (tronque/complète), sans inventer d'alarme.
        buf = np.zeros(n_samples, dtype=bool)
        m = min(n_samples, alarms.shape[0])
        buf[:m] = alarms[:m]
        alarms = buf

    alarm_idx = np.flatnonzero(alarms)
    n_alarms = int(alarm_idx.size)

    # ── Cas sans vérité-terrain ponctuelle : seuls FAR / MTFA sur flux stable ─────────────────────
    if not drift_points:
        far = float(n_alarms / n_samples) if n_samples > 0 else 0.0
        mtfa = _mean_gap(alarm_idx) if n_alarms >= 2 else None
        return {
            "mean_detection_delay": None,
            "missed_detection_rate": None,
            "false_alarm_rate": far,
            "precision": None,
            "recall": None,
            "f1": None,
            "mtfa": mtfa,
            "mtd": None,
            "n_true_drifts": 0,
            "n_alarms": n_alarms,
            "n_detected": None,
            "n_false_alarms": n_alarms,
            "tolerance": int(tolerance),
        }

    # ── Appariement glouton point → première alarme dans la fenêtre ───────────────────────────────
    drift_points = [int(p) for p in drift_points]
    used = np.zeros(n_alarms, dtype=bool)
    delays: list[int] = []
    n_missed = 0
    matched_alarm_positions: set[int] = set()

    for dp in drift_points:
        found = None
        for k in range(n_alarms):
            if used[k]:
                continue
            idx = int(alarm_idx[k])
            if dp <= idx <= dp + tolerance:
                found = k
                break
        if found is None:
            n_missed += 1
        else:
            used[found] = True
            matched_alarm_positions.add(found)
            delays.append(int(alarm_idx[found]) - dp)

    n_true = len(drift_points)
    n_detected = n_true - n_missed

    # Fausses alarmes = alarmes hors de TOUTE fenêtre de tolérance (les doublons intra-fenêtre sont
    # ignorés, pas pénalisés — convention change-point standard).
    covered = _covered_mask(drift_points, n_samples, tolerance)
    false_positions = [int(i) for i in alarm_idx if not covered[i]]
    n_false = len(false_positions)

    n_stable = int((~covered).sum())
    far = float(n_false / n_stable) if n_stable > 0 else None
    mtfa = _mean_gap(np.asarray(false_positions)) if n_false >= 2 else None

    mdr = float(n_missed / n_true) if n_true > 0 else None
    mean_delay = float(np.mean(delays)) if delays else None
    mtd = mean_delay  # Mean Time to Detection = délai moyen sur les points détectés

    # Précision / rappel / F1 des alarmes vs points de drift.
    tp = n_detected
    precision = float(tp / (tp + n_false)) if (tp + n_false) > 0 else None
    recall = float(tp / n_true) if n_true > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None

    return {
        "mean_detection_delay": mean_delay,
        "missed_detection_rate": mdr,
        "false_alarm_rate": far,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mtfa": mtfa,
        "mtd": mtd,
        "n_true_drifts": int(n_true),
        "n_alarms": n_alarms,
        "n_detected": int(n_detected),
        "n_false_alarms": int(n_false),
        "tolerance": int(tolerance),
    }


def _mean_gap(indices: np.ndarray) -> float | None:
    """Écart moyen entre indices consécutifs (temps moyen entre alarmes) ; ``None`` si < 2."""
    indices = np.asarray(indices)
    if indices.size < 2:
        return None
    return float(np.mean(np.diff(np.sort(indices))))


def profile_drift_detector(detector, stream: Iterable, *, state_bytes: int | None = None) -> dict:
    """Profil de **coût proxy PC** d'un détecteur (prépare la mesure board réelle S45).

    Réutilise ``tracemalloc`` (proxy RAM PC) et ``time.perf_counter`` (latence). L'empreinte
    algorithmique ``state_bytes`` provient de ``detector.get_state_bytes()`` (état majoré, argument
    MCU) — ou de l'argument ``state_bytes`` pour le baseline qui n'expose pas cette API (sa taille est
    documentée par un ``# MEM:``, passée par l'appelant).

    Les chiffres RAM/latence sont des **proxies PC** — clé ``_proxy: True``, à ne pas confondre avec la
    mesure ``.bss``/DWT du Sprint 45.

    Parameters
    ----------
    detector : object
        Détecteur exposant ``update(value)`` (déjà calibré si nécessaire). Idéalement une instance
        dédiée au profilage (le streaming modifie l'état interne).
    stream : Iterable
        Valeurs à passer à ``update`` (erreurs 0/1, scalaires agrégés, ou vecteurs feature).
    state_bytes : int | None
        Empreinte à reporter si le détecteur n'a pas ``get_state_bytes`` (baseline).

    Returns
    -------
    dict
        ``state_bytes, state_bytes_source, ram_peak_bytes, latency_us_per_update,
        latency_us_std, n_updates, requires_label, _proxy``.
    """
    values = list(stream)

    if hasattr(detector, "get_state_bytes"):
        sb = int(detector.get_state_bytes())
        sb_src = "get_state_bytes"
    else:
        sb = int(state_bytes) if state_bytes is not None else None
        sb_src = "mem_annotation" if state_bytes is not None else None

    per_update_us: list[float] = []
    tracemalloc.start()
    for v in values:
        t0 = time.perf_counter()
        detector.update(v)
        per_update_us.append((time.perf_counter() - t0) * 1e6)
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    requires_label = getattr(detector, "requires_label", None)

    return {
        "state_bytes": sb,
        "state_bytes_source": sb_src,
        "ram_peak_bytes": int(ram_peak),
        "latency_us_per_update": float(np.mean(per_update_us)) if per_update_us else None,
        "latency_us_std": float(np.std(per_update_us)) if per_update_us else None,
        "n_updates": len(values),
        "requires_label": bool(requires_label) if requires_label is not None else None,
        "_proxy": True,  # RAM/latence = proxy PC ; mesure board réelle = Sprint 45
    }


# Colonnes canoniques de la table comparative (ordre stable pour slides/notebook).
_TABLE_COLUMNS = [
    "detector",
    "dataset",
    "requires_label",
    "mean_detection_delay",
    "false_alarm_rate",
    "missed_detection_rate",
    "f1",
    "mtfa",
    "mtd",
    "state_bytes",
    "latency_us_per_update",
    "viabilite_mcu",
]


def build_comparison_table(results_by_detector: dict[str, dict[str, dict]]) -> dict:
    """Assemble une table détecteur × dataset sérialisable JSON depuis les résultats de la grille.

    Parameters
    ----------
    results_by_detector : dict[str, dict[str, dict]]
        ``{detector: {dataset: result}}`` où ``result`` a la forme d'un ``results.json`` S4405
        (clés ``drift_metrics``, ``cost``, ``requires_label``, ``viabilite_mcu``).

    Returns
    -------
    dict
        ``{"columns": [...], "rows": [ {col: valeur|None} ]}``.
    """
    rows: list[dict] = []
    for detector in sorted(results_by_detector):
        by_dataset = results_by_detector[detector]
        for dataset in sorted(by_dataset):
            result = by_dataset[dataset] or {}
            dm = result.get("drift_metrics", {}) or {}
            cost = result.get("cost", {}) or {}
            rows.append(
                {
                    "detector": detector,
                    "dataset": dataset,
                    "requires_label": result.get("requires_label"),
                    "mean_detection_delay": dm.get("mean_detection_delay"),
                    "false_alarm_rate": dm.get("false_alarm_rate"),
                    "missed_detection_rate": dm.get("missed_detection_rate"),
                    "f1": dm.get("f1"),
                    "mtfa": dm.get("mtfa"),
                    "mtd": dm.get("mtd"),
                    "state_bytes": cost.get("state_bytes"),
                    "latency_us_per_update": cost.get("latency_us_per_update"),
                    "viabilite_mcu": result.get("viabilite_mcu"),
                }
            )
    return {"columns": list(_TABLE_COLUMNS), "rows": rows}


def save_drift_metrics(metrics: dict, output_path: str | Path, extra_info: dict | None = None) -> None:
    """Sauvegarde les métriques de drift au format JSON (miroir ``save_anomaly_metrics``).

    Parameters
    ----------
    metrics : dict
        Retour de :func:`compute_drift_metrics` (ou table/coût).
    output_path : str | Path
        Chemin de destination (parents créés si besoin).
    extra_info : dict | None
        Métadonnées additionnelles fusionnées à plat (detector, dataset, exp_id…).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if extra_info:
        payload.update(extra_info)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
