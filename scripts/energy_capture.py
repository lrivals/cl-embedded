"""
energy_capture.py — Pilote PowerShield X-NUCLEO-LPM01A / STM32CubeMonitor-Power (S3305).

Acquiert (ou importe) une trace courant/tension du LPM01A, la segmente selon les
marqueurs de phase GPIO du firmware (PA8, S3304 — `EnergyPhase` :
startup/acquisition/inference/idle), intègre l'énergie en **µJ par phase**
(E = Σ I·V·dt) et exporte un JSON normalisé dans `experiments/exp_S33_energy/`.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ :
    Tant que la board + le LPM01A n'ont pas réellement tourné (pas de CSV `--csv`
    fourni), les champs énergie du JSON portent la valeur littérale ``"à mesurer"``
    (constante `A_MESURER`). Ce script NE FABRIQUE jamais de courant/µJ.
    La fréquence d'échantillonnage et la calibration LPM01A restent `TODO(dorra)`
    tant qu'elles ne sont pas confirmées.

Patterns réutilisés :
    - CLI / pilotage série : scripts/sensor_stream.py
    - Structure d'enregistrement d'expérience : scripts/board_experiment_recorder.py

Usage :
    # Un couple modèle × encodage (placeholder tant que pas de CSV LPM01A) :
    python scripts/energy_capture.py --model ewc --encoding fp32 \\
        --duration 10 --output experiments/exp_S33_energy/ewc_fp32.json

    # Avec une trace LPM01A réelle exportée par STM32CubeMonitor-Power :
    python scripts/energy_capture.py --model ewc --encoding fp32 \\
        --csv captures/ewc_fp32.csv --output experiments/exp_S33_energy/ewc_fp32.json

    # Campagne complète (4 modèles × {fp32,int8} + summary.json) :
    python scripts/energy_capture.py --campaign \\
        --output-dir experiments/exp_S33_energy/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Valeur littérale écrite tant qu'aucune mesure réelle n'est disponible (règle
# « aucun chiffre inventé »). Tout consommateur (notebook, autonomy.py) doit la
# détecter avant de calculer.
A_MESURER = "à mesurer"

# Phases énergie — DOIT correspondre à l'enum firmware EnergyPhase (S3304,
# firmware/stm32f4_blink/inc/profiling.h).
PHASES = ("startup", "acquisition", "inference", "idle")

# Couples modèle × encodage de la campagne S3306.
CAMPAIGN_MODELS = ("ewc", "hdc", "tinyol", "maha")
CAMPAIGN_ENCODINGS = ("fp32", "int8")


def capture_session(
    duration_s: float,
    sampling_rate_hz: float,
    output_csv: Path,
    csv_in: Path | None = None,
) -> Path | None:
    """Acquiert une session LPM01A, ou importe un CSV existant.

    Le pilotage headless de STM32CubeMonitor-Power n'étant pas garanti
    disponible sur ce poste, l'acquisition « live » n'est PAS simulée : si aucun
    CSV n'est fourni, la fonction renvoie ``None`` (aucune donnée réelle), ce qui
    déclenche le mode placeholder en aval. Si `csv_in` est fourni, il est validé
    et son chemin renvoyé pour la segmentation.

    Parameters
    ----------
    duration_s : float
        Durée nominale de la capture (s) — métadonnée, reportée dans le JSON.
    sampling_rate_hz : float
        Fréquence d'échantillonnage LPM01A (Hz). `TODO(dorra)` si non calibrée.
    output_csv : Path
        Chemin où une capture live serait écrite (non utilisé en mode import).
    csv_in : Path | None
        Trace LPM01A déjà exportée (colonnes temps, courant, [tension]).

    Returns
    -------
    Path | None
        Le chemin du CSV exploitable, ou ``None`` si aucune donnée réelle.
    """
    if csv_in is not None:
        csv_in = Path(csv_in)
        if not csv_in.is_file():
            raise FileNotFoundError(f"CSV LPM01A introuvable : {csv_in}")
        return csv_in
    # Pas de CSV : aucune fabrication de mesure (règle « aucun chiffre inventé »).
    _ = (duration_s, sampling_rate_hz, output_csv)
    return None


def _load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Charge une trace LPM01A (colonnes : time_s, current_a, [voltage_v]).

    Le format CSV exact de STM32CubeMonitor-Power n'étant pas figé (`TODO(dorra)`),
    on lit de façon tolérante : on cherche des colonnes nommées et, à défaut,
    on retombe sur l'ordre [temps, courant, tension].
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"CSV vide : {csv_path}")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(not _is_float(c) for c in header)
    data_rows = rows[1:] if has_header else rows

    def _col(*names: str, default: int | None = None) -> int | None:
        for n in names:
            if n in header:
                return header.index(n)
        return default

    i_time = _col("time", "time_s", "timestamp", "t", default=0)
    i_cur = _col("current", "current_a", "i", "i_a", default=1)
    i_volt = _col("voltage", "voltage_v", "v", "u", default=None)
    # Colonne numérique de synchronisation = signal GPIO PA8 du firmware (S3304),
    # capté par le LPM01A en parallèle du courant. Ses fronts délimitent les phases
    # (cf. derive_phase_windows). Absente sur un export courant seul → None.
    i_sync = _col("sync", "pa8", "gpio", "digital", "marker", default=None)

    arr = np.array([[float(x) for x in r] for r in data_rows], dtype=np.float64)
    out = {
        "time_s": arr[:, i_time],
        "current_a": arr[:, i_cur],
    }
    out["voltage_v"] = arr[:, i_volt] if i_volt is not None else None
    out["sync"] = arr[:, i_sync] if i_sync is not None else None
    return out


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def integrate_energy_uj(
    courant_a: np.ndarray, tension_v: np.ndarray | float, dt_s: float | np.ndarray
) -> float:
    """Énergie en µJ : E = Σ(I × V × dt) × 1e6.

    Parameters
    ----------
    courant_a : np.ndarray
        Échantillons de courant (A).
    tension_v : np.ndarray | float
        Tension (V), scalaire ou par échantillon.
    dt_s : float | np.ndarray
        Pas de temps (s), scalaire (échantillonnage uniforme) ou par intervalle.

    Returns
    -------
    float
        Énergie en microjoules (µJ).
    """
    i = np.asarray(courant_a, dtype=np.float64)
    v = np.asarray(tension_v, dtype=np.float64)
    power_w = i * v  # W = A × V, échantillon par échantillon
    energy_j = float(np.sum(power_w * dt_s))
    return energy_j * 1e6


def segment_by_phase(
    csv_path: Path, phase_timestamps: list[tuple[str, float, float]]
) -> dict:
    """Découpe la trace I/V selon les fenêtres [t_start, t_end] par `EnergyPhase`.

    Les fenêtres proviennent des marqueurs GPIO PA8 (S3304) horodatés DWT,
    convertis en secondes et alignés sur l'axe temps de la trace LPM01A.

    Parameters
    ----------
    csv_path : Path
        Trace LPM01A (cf. `_load_csv`).
    phase_timestamps : list[tuple[str, float, float]]
        Liste de (nom_phase, t_start_s, t_end_s) — nom_phase ∈ `PHASES`.

    Returns
    -------
    dict
        ``{phase: uj}`` pour chaque phase de `PHASES` (0.0 si fenêtre vide).
    """
    trace = _load_csv(csv_path)
    t = trace["time_s"]
    i = trace["current_a"]
    v = trace["voltage_v"]
    if v is None:
        raise ValueError(
            "Trace sans colonne tension : fournir la tension d'alim (V) dans le CSV "
            "ou utiliser un profil HW. La fabrication d'une tension est interdite."
        )

    phases_uj: dict[str, float] = {p: 0.0 for p in PHASES}
    for name, t0, t1 in phase_timestamps:
        if name not in phases_uj:
            raise ValueError(f"Phase inconnue : {name!r} (attendu {PHASES}).")
        mask = (t >= t0) & (t < t1)
        if not np.any(mask):
            continue
        seg_t = t[mask]
        dt = np.gradient(seg_t) if seg_t.size > 1 else np.array([0.0])
        phases_uj[name] += integrate_energy_uj(i[mask], v[mask], dt)
    return phases_uj


def derive_phase_windows(
    trace: dict[str, np.ndarray], threshold: float = 0.5
) -> list[tuple[str, float, float]]:
    """Déduit les fenêtres de phase des fronts du signal de sync PA8 (S3304).

    Le firmware met PA8 au niveau HAUT pendant les phases actives
    (`startup`/`acquisition`/`inference`, `ENERGY_MARKER_SET`) et au niveau BAS en
    attente (`idle`, `ENERGY_MARKER_CLEAR`). Avec le schéma **1-bit** actuel ces
    trois phases actives partagent le même niveau haut : on ne peut donc pas les
    distinguer depuis la seule trace. On reporte honnêtement chaque plateau HAUT
    comme `"inference"` (phase active mesurée) et chaque plateau BAS comme
    `"idle"` — aucune phase fabriquée. Une granularité 4-phases exigerait un
    encodage multi-bit côté firmware (évolution future, hors périmètre S33).

    Parameters
    ----------
    trace : dict[str, np.ndarray]
        Sortie de `_load_csv` ; doit contenir une colonne `"sync"` non ``None``.
    threshold : float
        Seuil de binarisation du niveau logique (V ou unité brute du LPM01A).

    Returns
    -------
    list[tuple[str, float, float]]
        Fenêtres ``(nom_phase, t_start_s, t_end_s)`` consommables par
        `segment_by_phase` ; ``[]`` si le signal est constant (aucune transition).

    Raises
    ------
    ValueError
        Si la trace ne contient pas de colonne de synchronisation.
    """
    sync = trace.get("sync")
    if sync is None:
        raise ValueError(
            "Trace sans colonne de synchronisation (PA8) : impossible de déduire "
            "les fenêtres de phase. Fournir la colonne sync/pa8/gpio du LPM01A."
        )
    t = trace["time_s"]
    level = np.asarray(sync, dtype=np.float64) >= threshold
    windows: list[tuple[str, float, float]] = []
    # Parcours des plateaux de niveau constant : chaque segment [i, j) homogène
    # devient une fenêtre, bornée par le timestamp d'entrée et celui du front.
    start = 0
    for k in range(1, level.size):
        if level[k] != level[start]:
            name = "inference" if level[start] else "idle"
            windows.append((name, float(t[start]), float(t[k])))
            start = k
    if level.size:
        name = "inference" if level[start] else "idle"
        windows.append((name, float(t[start]), float(t[-1])))
    return windows


def export_energy_json(
    phases_uj: dict,
    model: str,
    encoding: str,
    output_path: Path,
    sampling_rate_hz: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Exporte un JSON énergie normalisé pour un couple modèle × encodage.

    Schéma :
        {"model", "encoding", "phases_uj": {startup, acquisition, inference, idle},
         "total_uj", "sampling_rate_hz", "duration_s", "timestamp", "source"}

    Si `phases_uj` est ``None`` (aucune mesure réelle), tous les champs énergie
    valent la constante littérale `A_MESURER` — aucun chiffre n'est inventé.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    measured = phases_uj is not None
    if measured:
        phases = {p: float(phases_uj.get(p, 0.0)) for p in PHASES}
        total = float(sum(phases.values()))
    else:
        phases = {p: A_MESURER for p in PHASES}
        total = A_MESURER

    payload = {
        "model": model,
        "encoding": encoding,
        "phases_uj": phases,
        "total_uj": total,
        # TODO(dorra) : calibrer la fréquence d'échantillonnage / plage de courant
        # LPM01A (capter la veille en µA comme l'inférence en mA).
        "sampling_rate_hz": sampling_rate_hz if sampling_rate_hz else A_MESURER,
        "duration_s": duration_s if duration_s else A_MESURER,
        "source": "lpm01a_csv" if measured else "placeholder",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    status = "mesuré" if measured else f"placeholder ({A_MESURER})"
    print(f"  ✔ {output_path}  [{model}/{encoding}] — {status}")


def _capture_one(
    model: str,
    encoding: str,
    output_path: Path,
    duration_s: float,
    sampling_rate_hz: float,
    csv_in: Path | None,
) -> None:
    """Pipeline complet pour un couple : capture → segmentation → export JSON."""
    csv_path = capture_session(
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        output_csv=output_path.with_suffix(".csv"),
        csv_in=csv_in,
    )
    if csv_path is None:
        export_energy_json(
            None, model, encoding, output_path,
            sampling_rate_hz=sampling_rate_hz or None, duration_s=duration_s,
        )
        return
    # CSV réel : les fenêtres de phase viennent du signal de sync PA8 (S3304),
    # capté par le LPM01A dans la même trace. Sans cette colonne, on refuse
    # plutôt que de fabriquer une segmentation.
    trace = _load_csv(csv_path)
    if trace.get("sync") is None:
        raise ValueError(
            f"CSV {csv_path} sans colonne de synchronisation (sync/pa8/gpio) : "
            "impossible de segmenter par phase. Exporter le signal PA8 du firmware "
            "(marqueurs S3304) en parallèle du courant depuis STM32CubeMonitor-Power."
        )
    windows = derive_phase_windows(trace)
    phases_uj = segment_by_phase(csv_path, windows)
    export_energy_json(
        phases_uj, model, encoding, output_path,
        sampling_rate_hz=sampling_rate_hz or None, duration_s=duration_s,
    )


def _write_summary(output_dir: Path) -> None:
    """Agrège les JSON modèle×encodage en `summary.json` (table FP32 vs INT8).

    Pour chaque modèle : `delta_uj`, `ratio` (int8/fp32). Tant que les énergies
    sont `A_MESURER`, ces deltas le restent aussi. Le lien Gap 3 est explicité
    en texte (réduction RAM INT8 sans accélération latence FPU, Sprint 29).
    """
    rows: dict[str, dict] = {}
    for model in CAMPAIGN_MODELS:
        entry: dict = {}
        for enc in CAMPAIGN_ENCODINGS:
            p = output_dir / f"{model}_{enc}.json"
            if p.is_file():
                entry[enc] = json.loads(p.read_text(encoding="utf-8")).get("total_uj")
        fp32, int8 = entry.get("fp32"), entry.get("int8")
        numeric = isinstance(fp32, (int, float)) and isinstance(int8, (int, float))
        entry["delta_uj"] = (int8 - fp32) if numeric else A_MESURER
        entry["ratio_int8_fp32"] = (int8 / fp32) if numeric and fp32 else A_MESURER
        rows[model] = entry

    summary = {
        "description": "Campagne énergie S3306 — µJ total par modèle × encodage (LPM01A).",
        "phases": list(PHASES),
        "per_model": rows,
        "gap3_note": (
            "Sprint 29 : l'INT8 réduit la RAM sans accélérer la latence (FPU "
            "Cortex-M4, pas de NPU INT8). Question énergie : l'INT8 réduit-il "
            "néanmoins les µJ (moins d'accès mémoire) ? — réponse via mesures "
            "LPM01A réelles, champs '" + A_MESURER + "' tant que non capturées."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✔ {out}  — agrégat FP32 vs INT8")


def run_campaign(
    output_dir: Path, duration_s: float, sampling_rate_hz: float
) -> None:
    """Campagne S3306 : 4 modèles × {fp32, int8} + summary.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Campagne énergie S3306 → {output_dir}/")
    for model in CAMPAIGN_MODELS:
        for enc in CAMPAIGN_ENCODINGS:
            _capture_one(
                model, enc, output_dir / f"{model}_{enc}.json",
                duration_s, sampling_rate_hz, csv_in=None,
            )
    _write_summary(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture énergie LPM01A → µJ par phase (S3305/S3306)."
    )
    parser.add_argument("--model", choices=CAMPAIGN_MODELS, help="Modèle ciblé.")
    parser.add_argument("--encoding", choices=CAMPAIGN_ENCODINGS, help="Encodage.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Durée nominale de capture (s).")
    parser.add_argument("--sampling-rate-hz", type=float, default=0.0,
                        help="Fréq. échantillonnage LPM01A (0 = TODO(dorra)).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Trace LPM01A exportée (STM32CubeMonitor-Power).")
    parser.add_argument("--output", type=Path,
                        help="Chemin JSON de sortie (mode couple unique).")
    parser.add_argument("--campaign", action="store_true",
                        help="Campagne complète 4 modèles × {fp32,int8} + summary.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/exp_S33_energy"),
                        help="Répertoire de sortie (mode --campaign).")
    args = parser.parse_args()

    if args.campaign:
        run_campaign(args.output_dir, args.duration, args.sampling_rate_hz)
        return

    if not (args.model and args.encoding and args.output):
        print("Erreur : --model, --encoding et --output requis (ou --campaign).",
              file=sys.stderr)
        sys.exit(2)

    _capture_one(
        args.model, args.encoding, args.output,
        args.duration, args.sampling_rate_hz, args.csv,
    )


if __name__ == "__main__":
    main()
