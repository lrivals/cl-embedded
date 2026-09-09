"""run_s53_phase_profile.py — S5305 : profil temporel par phase, µJ par intégration.

CE QUE CE PILOTE DÉBLOQUE :

La chaîne de segmentation par phase existe depuis le Sprint 33 (`derive_phase_windows`,
`segment_by_phase`, `integrate_energy_uj`, marqueurs PA8 `-DENERGY_MARKERS`) et n'a JAMAIS
pu être exercée sur des données réelles, pour une raison unique et mesurée : le LPM01A ne
capture aucune voie numérique (`docs/context/lpm01a_setup.md` §7), donc la colonne `sync`
ne peut pas venir de la sonde.

    Voie A (défaut, aucun câblage) — si S5303 fait passer la carte sous le plafond de
    59 mA du mode dynamique, `acqmode dyn` à 100 kSPS donne une trace où les bursts
    d'inférence sont des créneaux nets : **le signal EST la synchro**. La segmentation se
    fait sur le courant lui-même (`energy_capture.derive_phase_windows_from_current`), avec
    un seuil déduit de la trace (médiane + k·MAD) et jamais saisi à la main.

    Voie B (repli) — `--trigsrc d7` : l'acquisition démarre sur le front de PA8 câblé sur
    le connecteur Arduino D7. Elle exige **une soudure** sur la carte : ce pilote la
    propose, il ne l'engage pas. Une trace qui porte réellement une colonne de
    synchronisation (relevée par un autre appareil) est traitée par la fonction PA8
    EXISTANTE, sans une ligne de code neuve (`--csv trace.csv`).

TROISIÈME ESTIMATEUR — à ne jamais fusionner : les µJ par inférence produits ici portent
`method = "intégration du profil temporel"`, distinct du protocole delta (S5302) et de la
régression de cadence (S5304). La comparaison des trois voies est elle-même un résultat de
méthodologie (S5309) et le meilleur contrôle de validité de la campagne.

LIMITE CONSERVÉE TELLE QUELLE : la voie A sépare l'actif de l'inactif, pas les trois
sous-phases actives — exactement la limite du marqueur 1 bit. `startup` et `acquisition`
sortent « à mesurer » avec leur raison, elles ne valent pas zéro.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : si `acqmode dyn` est refusé, la cellule sort en
N/A avec le courant RELEVÉ à l'échec ; si le nombre de créneaux détectés s'écarte de plus
de 5 % de `cadence × durée`, la segmentation est refusée et aucune énergie n'est publiée.

Usage :
    # Banc (build S5303 `-DSYSCLK_MHZ=45` déjà flashé, sonde posée) :
    python scripts/run_s53_phase_profile.py --model hdc --sysclk-mhz 45 \\
        --board-port /dev/serial/by-id/…STLink…

    # Reprise d'une trace déjà capturée, sans matériel (voie A ou B selon la colonne sync) :
    python scripts/run_s53_phase_profile.py --model hdc --sysclk-mhz 45 \\
        --csv captures/hdc_45mhz.csv --rate-hz 10

    # Recalcule seulement l'agrégat des cellules existantes :
    python scripts/run_s53_phase_profile.py --summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")
lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
#: `stream_command`, `STREAM_MODEL` et `BUILD_SPECIFIC` sont déjà éprouvés (S50) :
#: ils ne sont pas réécrits ici. `BUILD_SPECIFIC` protège du bug du drapeau TinyOL (S52).
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")
fs = _load("run_s53_freq_sweep", ROOT / "scripts" / "run_s53_freq_sweep.py")
from src.evaluation import dyn_threshold as dt   # noqa: E402

A_MESURER = ec.A_MESURER
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_phase_profile"
MA = 1000.0

#: Cadence de validation (spec S5305). Basse et CONNUE : à 10 Hz un burst HDC de 2095 µs
#: couvre ~200 échantillons à 100 kSPS, largement segmentable, et le nombre d'inférences
#: attendu vaut `cadence × durée` par construction.
DEFAULT_RATE_HZ = 10.0

#: Fréquence d'échantillonnage de la sonde en mode dynamique.
DYN_FREQ = "100k"

#: Durée d'acquisition. `acqtime` est plafonné à 10 s par le firmware de la sonde
#: (`ACQTIME_MAX_S`) : à 10 Hz, cela fait 100 créneaux attendus.
DEFAULT_DURATION_S = 10.0

#: Délai laissé au flux pour atteindre son régime établi avant l'ouverture de la fenêtre
#: (même convention que `run_s50_board_current.measure_current`).
DEFAULT_SETTLE_S = 2.0

#: Marge d'échantillons demandés au flux, pour qu'il ne s'arrête pas avant la fin de
#: l'acquisition (il est terminé par le pilote, pas par épuisement).
STREAM_MARGIN_S = 5.0


def start_stream(model_flag: str, dataset: str, port: str, rate_hz: float,
                 duration_s: float, settle_s: float) -> subprocess.Popen:
    """Lance le flux hôte qui produit les inférences à cadence imposée.

    Le flux tourne PENDANT l'acquisition : c'est lui qui grave les créneaux dans la trace
    de courant. `hold_run` ne convient pas ici — il fige `format ascii_dec` / `freq 1`,
    incompatible avec les 100 kSPS du mode dynamique. Le motif retenu est celui, déjà
    éprouvé, de `run_s50_board_current.measure_current` : Popen, temps d'établissement,
    puis acquisition.
    """
    n_samples = int(rate_hz * (duration_s + settle_s + STREAM_MARGIN_S))
    cmd = rc.stream_command(model_flag, dataset, port, n_samples, rate_hz)
    return subprocess.Popen(cmd, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_stream(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def capture_under_load(probe, voltage_mv: int, args, model_flag: str) -> dict:
    """Acquisition dynamique pendant que la carte infère — ou l'échec RELEVÉ.

    Le mode dynamique est ce qui rend la voie A possible ; son plafond de 59 mA est une
    limite matérielle mesurée, pas un réglage à trouver. Un refus est donc consigné avec
    le courant relevé (motif `run_s53_freq_sweep.try_dynamic_mode`), jamais avec une
    appréciation.
    """
    proc = start_stream(model_flag, args.dataset, args.board_port,
                        args.rate_hz, args.duration, args.settle)
    try:
        time.sleep(args.settle)
        samples, voltage_v, summary = lp.capture(
            probe, freq=DYN_FREQ, duration_s=args.duration, voltage_mv=voltage_mv,
            acqmode="dyn", trigsrc=args.trigsrc,
        )
    except Exception as exc:   # LPM01AError ou flux échoué : les deux se consignent
        return {
            "succeeded": False,
            "acqmode": "dyn",
            "na_reason": (
                f"mode dynamique refusé par la sonde pendant le flux « {model_flag} » : "
                f"{exc}"
            ),
        }
    finally:
        stop_stream(proc)
    # `lp.capture` ne lève que sur un flux VIDE : une acquisition INTERROMPUE en
    # surintensité rend les quelques dizaines de millisecondes décodées avant l'arrêt et
    # passerait ici pour un succès. La règle qui distingue les deux est celle de toute la
    # campagne (`dyn_threshold.acquisition_outcome`, issue de
    # `run_s53_freq_sweep.try_dynamic_mode`) : une seule définition de « acquisition
    # aboutie », testée hors banc. Sans elle, une trace tronquée était refusée plus loin
    # par le compte de créneaux — donc avec une raison qui désignait la mauvaise cause.
    n_expected = int(lp.parse_freq_hz(DYN_FREQ) * args.duration)
    bloc = {
        "acqmode": "dyn",
        "samples_a": samples,
        "voltage_v": float(voltage_v),
        "i_mean_ma": float(np.mean(samples)) * MA,
        "i_max_ma": float(np.max(samples)) * MA,
        "summary": summary.strip()[-400:],
    }
    bloc.update(dt.acquisition_outcome(int(samples.size), n_expected, summary))
    return bloc


def trace_from_samples(samples_a: np.ndarray, voltage_v: float,
                       fs_hz: float) -> dict[str, np.ndarray]:
    """Trace au format `energy_capture._load_csv`, sans passer par un fichier.

    La base de temps est reconstruite au pas nominal `1/fs` : la sonde n'horodate pas
    chaque échantillon (`lpm01a_probe.write_csv` fait déjà ce choix, il n'est pas
    réinventé ici).
    """
    return {
        "time_s": np.arange(samples_a.size, dtype=np.float64) / fs_hz,
        "current_a": np.asarray(samples_a, dtype=np.float64),
        "voltage_v": np.full(samples_a.size, float(voltage_v), dtype=np.float64),
        "sync": None,
    }


def profile_trace(trace: dict, rate_hz: float, threshold: float | None,
                  tolerance: float) -> dict:
    """Segmente une trace par la voie qui lui correspond, sans jamais en inventer une.

    Une colonne de synchronisation présente (voie B, ou trace relevée par un autre
    appareil) est traitée par les fonctions PA8 **existantes** — c'est tout l'intérêt de
    la voie B : elle n'exige aucune ligne de code neuve. Sinon, segmentation sur le
    courant (voie A).
    """
    if trace.get("sync") is not None:
        windows = ec.derive_phase_windows(trace)
        t = np.asarray(trace["time_s"], dtype=np.float64)
        i = np.asarray(trace["current_a"], dtype=np.float64)
        v = np.asarray(trace["voltage_v"], dtype=np.float64)
        phases_uj = {p: 0.0 for p in ec.PHASES}
        durees = {p: 0.0 for p in ec.PHASES}
        for name, t0, t1 in windows:
            mask = (t >= t0) & (t < t1)
            if not np.any(mask):
                continue
            seg_t = t[mask]
            dt = np.gradient(seg_t) if seg_t.size > 1 else np.array([0.0])
            phases_uj[name] += ec.integrate_energy_uj(i[mask], v[mask], dt)
            durees[name] += float(t1 - t0)
        n_bursts = ec.count_bursts(windows)
        duree = float(t[-1] - t[0]) if t.size > 1 else 0.0
        ok, ecart, raison = ec.validate_burst_count(n_bursts, rate_hz, duree, tolerance)
        bloc = {
            "segmentation": "pa8_d7 (voie B)",
            "n_bursts_detected": n_bursts,
            "n_bursts_expected": rate_hz * duree,
            "detection_error_pct": ecart,
            "duration_s": duree,
            "method": ec.METHOD_INTEGRATION,
            "phases_na_reason": ec.NA_PHASES_1BIT,
        }
        if not ok or n_bursts == 0:
            return {**bloc,
                    "phases_uj": {p: A_MESURER for p in ec.PHASES},
                    "total_uj": A_MESURER,
                    "energy_uj_per_inference": A_MESURER,
                    "energy_na_reason": raison or "aucun créneau actif détecté"}
        return {**bloc,
                "phases_uj": {"startup": A_MESURER, "acquisition": A_MESURER,
                              "inference": phases_uj["inference"],
                              "idle": phases_uj["idle"]},
                "total_uj": phases_uj["inference"] + phases_uj["idle"],
                "active_duration_s": durees["inference"],
                "energy_uj_per_burst_gross": phases_uj["inference"] / n_bursts,
                # MÊME grandeur que la voie A : un surcoût, repos déduit. Publier ici
                # une consommation brute sous ce nom fausserait la comparaison des trois
                # estimateurs (S5309).
                "energy_uj_per_inference": ec.marginal_uj_per_burst(
                    phases_uj["inference"], phases_uj["idle"],
                    durees["inference"], durees["idle"], n_bursts)}
    return ec.profile_from_current(trace, rate_hz, threshold, tolerance)


def measure_cell(args, parser: argparse.ArgumentParser) -> dict:
    """Une cellule `{model}_{sysclk}` : contrôle carte → acquisition → segmentation."""
    key = (args.model, args.encoding)
    if key not in rc.STREAM_MODEL:
        parser.error(f"cellule inconnue : {args.model}_{args.encoding}")
    if key in rc.BUILD_SPECIFIC:
        print(f"[phase] {args.model}_{args.encoding} exige un build "
              f"{rc.BUILD_SPECIFIC[key]} : la cellule n'est valide QUE s'il est flashé.")
    model_flag = rc.STREAM_MODEL[key]

    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)

    cell: dict = {
        "model": args.model,
        "encoding": args.encoding,
        "dataset": args.dataset,
        "sysclk_mhz": args.sysclk_mhz,
        "acqmode": "dyn",
        "fs_hz": lp.parse_freq_hz(DYN_FREQ),
        "rate_hz": float(args.rate_hz),
        "trigsrc": args.trigsrc,
        "firmware_build": (f"-DSYSCLK_MHZ={args.sysclk_mhz}"
                           if args.sysclk_mhz != 180 else "défaut (180 MHz)"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if args.csv is not None:
        # Reprise hors banc : la trace est déjà là, rien n'est acquis ni supposé.
        trace = ec._load_csv(Path(args.csv))
        cell["source"] = f"trace importée : {args.csv}"
        cell.update(profile_trace(trace, args.rate_hz, args.threshold, args.tolerance))
        return cell

    probe = lp.PowerShield(args.port)
    try:
        probe.take_control()
        # La première acquisition d'une session lit ~8 mA de trop (constat de banc
        # 2026-08-04) : elle est jetée, pas corrigée.
        cell["warmup_discarded_a"] = [float(lp.warmup(probe, voltage_mv))
                                      for _ in range(max(1, args.warmup_repeats))]

        # La carte tourne-t-elle à la fréquence annoncée ? Contrôlé avant toute MESURE :
        # une cellule écrite sous un nom qui ment sur son contenu est pire que pas de
        # mesure (leçon du drapeau TinyOL, Sprint 52).
        #
        # Placé DANS la session de sonde, après le préchauffage : `JP5` étant retiré, la
        # cible n'est alimentée que par la sonde, et `pwrend on` la maintient une fois
        # l'acquisition terminée. Avant `take_control()`, ce contrôle interrogeait une
        # carte hors tension et échouait sur « bannière illisible » (même correctif que
        # `run_s53_freq_sweep.measure_cell`, mesuré 2026-09-01).
        cell["hw_check"] = fs.check_frequency(args)
        acquisition = capture_under_load(probe, voltage_mv, args, model_flag)
    finally:
        probe.release()

    if not acquisition["succeeded"]:
        cell.update({
            "acqmode_dyn_succeeded": False,
            "segmentation": A_MESURER,
            "phases_uj": {p: A_MESURER for p in ec.PHASES},
            "total_uj": A_MESURER,
            "energy_uj_per_inference": A_MESURER,
            "method": ec.METHOD_INTEGRATION,
            "phases_na_reason": ec.NA_PHASES_1BIT,
            "energy_na_reason": acquisition["na_reason"],
        })
        print(f"[phase] {args.model} @ {args.sysclk_mhz} MHz : "
              f"{acquisition['na_reason']}")
        return cell

    samples = acquisition.pop("samples_a")
    fs_hz = lp.parse_freq_hz(DYN_FREQ)
    if args.save_csv:
        lp.write_csv(Path(args.save_csv), samples, acquisition["voltage_v"], fs_hz,
                     metadata={"source": "run_s53_phase_profile.py",
                               "model": model_flag, "rate_hz": f"{args.rate_hz:g}",
                               "sysclk_mhz": str(args.sysclk_mhz)})
    cell["acqmode_dyn_succeeded"] = True
    cell["acquisition"] = acquisition
    trace = trace_from_samples(samples, acquisition["voltage_v"], fs_hz)
    cell.update(profile_trace(trace, args.rate_hz, args.threshold, args.tolerance))
    return cell


def build_summary(out_dir: Path) -> dict:
    """Agrégat lecture seule des cellules — aucune fusion d'estimateurs.

    Le rapprochement des trois estimateurs de µJ par inférence (delta S5302, régression
    S5304, intégration S5305) se fait dans `aggregate_s53_energy.py` (S5309), qui les
    conserve côte à côte. Ici on ne fait que rassembler ce que cette voie a produit, et
    dire pourquoi elle n'a rien produit quand c'est le cas.
    """
    cells = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        cells[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    chiffrees = {k: c["energy_uj_per_inference"] for k, c in cells.items()
                 if isinstance(c.get("energy_uj_per_inference"), (int, float))}
    return {
        "description": "S5305 — profil temporel par phase (intégration), par cellule.",
        "method": ec.METHOD_INTEGRATION,
        "estimator_note": (
            "µJ par inférence obtenus par INTÉGRATION du profil temporel. À conserver "
            "côte à côte avec le protocole delta (S5302) et la régression de cadence "
            "(S5304), jamais fusionnés ni moyennés : leur comparaison est le contrôle de "
            "validité de la campagne (S5309)."
        ),
        "phase_granularity_note": ec.NA_PHASES_1BIT,
        "cells": sorted(cells),
        "energy_uj_per_inference_by_cell": {
            k: c.get("energy_uj_per_inference") for k, c in cells.items()},
        "na_reason_by_cell": {k: c.get("energy_na_reason") for k, c in cells.items()
                              if c.get("energy_na_reason")},
        "detection_error_pct_by_cell": {
            k: c.get("detection_error_pct") for k, c in cells.items()},
        "n_cells_published": len(chiffrees),
        "n_cells_total": len(cells),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="hdc",
                        help="modèle mesuré (HDC par défaut : 2095 µs ⇒ ~200 échantillons "
                             "par créneau à 100 kSPS, largement segmentable)")
    parser.add_argument("--encoding", default="int8", choices=("fp32", "int8"))
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--sysclk-mhz", type=int, default=None,
                        help="fréquence RÉELLEMENT flashée (le mode dynamique exige le "
                             "build S5303 sous 59 mA)")
    parser.add_argument("--board-port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--port", default=None, help="port de la SONDE (auto)")
    parser.add_argument("--skip-hw-check", action="store_true",
                        help="n'exige PAS que la carte confirme sa fréquence")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--trigsrc", default="sw", choices=("sw", "d7"),
                        help="« d7 » = voie B : exige la soudure PA8→D7 (décision "
                             "utilisateur, non engagée par ce pilote)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="seuil actif/inactif (A) ; par défaut DÉDUIT de la trace "
                             "(médiane + k·MAD), ce qui est la règle de la spec")
    parser.add_argument("--tolerance", type=float, default=ec.BURST_TOLERANCE,
                        help="tolérance du contrôle « créneaux détectés vs attendus »")
    parser.add_argument("--csv", type=Path, default=None,
                        help="trace déjà capturée à retraiter (aucun matériel requis)")
    parser.add_argument("--save-csv", type=Path, default=None,
                        help="écrit aussi la trace brute acquise")
    parser.add_argument("--summary", action="store_true",
                        help="recalcule summary.json depuis les cellules existantes")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.sysclk_mhz is not None or args.csv is not None:
        if args.sysclk_mhz is None:
            parser.error("--sysclk-mhz est requis : une cellule doit dire à quelle "
                         "fréquence elle a été acquise.")
        if args.csv is None and not args.board_port:
            parser.error("--board-port est requis pour acquérir une cellule")
        cell = measure_cell(args, parser)
        path = args.out / f"{args.model}_{args.sysclk_mhz}.json"
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
        energie = cell.get("energy_uj_per_inference")
        lisible = (f"{energie:.2f} µJ" if isinstance(energie, (int, float))
                   else str(energie))
        print(f"[phase] {path.name} — {cell.get('segmentation')} · "
              f"{cell.get('n_bursts_detected')}/{cell.get('n_bursts_expected')} créneaux · "
              f"µJ/inférence = {lisible}")

    if args.summary or args.sysclk_mhz is not None:
        summary = build_summary(args.out)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[phase] {summary['n_cells_published']}/{summary['n_cells_total']} "
              f"cellule(s) publiable(s) → {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
