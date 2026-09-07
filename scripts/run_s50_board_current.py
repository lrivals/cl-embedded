"""
run_s50_board_current.py — Campagne énergie S5002 par **courant moyen à cadence imposée**.

POURQUOI CE PILOTE REMPLACE LE PROTOCOLE DELTA (constats de banc mesurés le
2026-08-04, NUCLEO-F439ZI + X-NUCLEO-LPM01A) :

    1. **La première acquisition d'une session est biaisée d'environ +8 mA.**
       Quatre acquisitions identiques au repos : 62,74 / 54,94 / 54,86 / 54,82 mA.
       Le protocole delta plaçait sa fenêtre de référence en première position :
       son biais dépassait l'effet cherché et inversait le signe du résultat.
       → ce pilote commence par `lpm01a_probe.warmup()` (rebut documenté).

    2. **La référence « au repos » est plus haute que TOUS les régimes de flux.**
       Mesuré (3 répétitions/cellule) : carte au repos 54,81 ± 0,11 mA, contre
       46,46 à 51,26 mA sous flux — y compris Mahalanobis, dont l'inférence
       n'occupe que ~0,05 % du temps. Un écart de −8 mA que le taux d'occupation
       ne peut pas expliquer : **la cause n'est pas établie** (le firmware attend
       la trame par scrutation active, `while (!(USART3->SR & RXNE)) {}` dans
       `pipeline.c`, ce qui rend le « repos » tout sauf inactif ; d'autres causes
       de banc ne sont pas exclues). On ne l'explique donc pas, on la constate.
       → Conséquence directe : l'« énergie marginale par inférence » du protocole
       delta ressort NÉGATIVE face à cette référence. Le champ reste `"à mesurer"`
       avec une raison **mesurée** — ce n'est pas un « pas encore fait ». Le levier
       identifié est firmware (mise en sommeil `WFI` de l'attente UART, pour
       disposer d'un vrai repos), hors périmètre de cette campagne.

    3. **Ce qui reste parfaitement exploitable : la comparaison entre cellules.**
       À cadence imposée identique, tout ce qui n'est pas le modèle (trafic UART,
       travail hôte, référence) est commun. Et l'ordre mesuré des courants suit
       exactement l'ordre des latences connues (Maha 5 µs → 46,46 mA … HDC INT8
       1958 µs → 51,26 mA) : l'écart entre deux cellules est bien imputable au
       modèle, avec une dispersion de ±0,05 mA qui le rend significatif.

CE QUI EST DONC MESURÉ, ET QUI EST SOLIDE :
    Le **courant moyen réel** de la carte sous une charge d'inférence à cadence
    imposée, par modèle × encodage, plus la référence au repos, toutes acquises
    dans la même session après préchauffage. À cadence identique, le trafic UART
    et le travail hôte sont communs à toutes les cellules : l'écart entre deux
    cellules est imputable au modèle. La répétabilité mesurée (±0,06 mA) permet
    de séparer les modèles. L'autonomie en découle directement.

PROTOCOLE CONTRE-BALANCÉ (S5301, ajouté le 2026-08-05) :
    Le constat n°2 ci-dessus a été établi avec une séquence où le repos est mesuré
    en TÊTE de session et les cellules ensuite : **l'ordre est confondu avec la
    condition**. `--interleave-idle` intercale une mesure de repos avant et après
    chaque cellule, dans la même session, chaque acquisition portant son rang
    (`session_index`). La règle de verdict vit dans `src/evaluation/counterbalance.py`
    (calculée et testée, jamais saisie).

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : tout champ non mesurable garde la
valeur littérale ``"à mesurer"`` accompagnée de sa raison.

Usage :
    python scripts/run_s50_board_current.py --rate-hz 100 --window 10 \\
        --dataset monitoring --out experiments/exp_S50_energy/

    # S5301 — séquence contre-balancée repos/charge, 2 modèles, 3 répétitions
    python scripts/run_s50_board_current.py --interleave-idle --repeats 3 \\
        --only ewc_fp32,hdc_fp32 --board-port /dev/serial/by-id/…STLink… \\
        --out experiments/exp_S53_counterbalance/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
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
#: Règle de verdict du contre-balancement (S5301) — hors du pilote pour rester
#: testable hors banc, et pour que les seuils ne soient pas des chiffres du pilote.
cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")

A_MESURER = ec.A_MESURER
PHASES = ec.PHASES

#: Drapeau `--model` de sensor_stream.py pour chaque cellule (modèle, encodage).
#: `maha`×`int8` est absent : le Mahalanobis INT8 se choisit à la COMPILATION
#: (`-DMAHA_INT8`, chemin flags=0) et exige donc un firmware dédié — il est
#: mesuré par une exécution séparée avec `--only maha_int8` sur ce build.
STREAM_MODEL = {
    ("ewc", "fp32"): "ewc",
    ("ewc", "int8"): "ewc-int8",
    ("hdc", "fp32"): "hdc",
    ("hdc", "int8"): "hdc-int8",
    ("tinyol", "fp32"): "tinyol",
    ("tinyol", "int8"): "tinyol-int8",
    ("maha", "fp32"): "mahalanobis",
    ("maha", "int8"): "mahalanobis",   # requiert un build -DMAHA_INT8
}

#: Cellules exclues du balayage par défaut car elles exigent un firmware DÉDIÉ.
#: Sans cette exclusion, `maha_int8` serait mesurée sur le build FP32 et écrite
#: sous un nom qui ment sur son contenu — exactement le bug du Sprint 52.
#: Elles se mesurent explicitement, après flash du bon build : `--only maha_int8`.
BUILD_SPECIFIC = {("maha", "int8"): "-DMAHA_INT8"}

#: Conversion A → mA pour l'affichage seul (jamais dans les JSON, en ampères).
MA_PER_A = 1000

#: Raison mesurée du N/A sur l'énergie par inférence (cf. docstring du module).
#: Mise à jour au Sprint 53 : la cause EST désormais établie (artefact d'ordre, S5301).
#: Ce pilote ne produit toujours pas de µJ — ce n'est pas son protocole — mais il ne
#: doit plus publier une incertitude que la mesure a levée.
NA_PER_INFERENCE = (
    "hors protocole de ce pilote (courant moyen, pas de soustraction) — et la référence "
    "au repos exige une session ÉTABLIE : le contre-balancement S5301 a montré que "
    "l'écart « repos plus consommateur que la charge » du Sprint 50 est un ARTEFACT "
    "D'ORDRE (référence prise en tête de session, encore dans la queue d'établissement). "
    "Une fois établi, le repos se stabilise SOUS les cellules et le surcoût de charge "
    "redevient positif et ordonné comme les latences. Les µJ par inférence se prennent "
    "par le protocole delta sur référence établie (S5302) ou par régression de cadence "
    "(S5304). Exploitable ici : le courant moyen par cellule et sa comparaison "
    "inter-cellules à cadence imposée (cf. `current_measurement`)."
)


def stream_command(model_flag: str, dataset: str, port: str,
                   n_samples: int, rate_hz: float) -> list[str]:
    """Commande de flux pour une cellule (cadence imposée → N exact).

    `--rate-hz` fixe la cadence : le nombre d'inférences dans la fenêtre vaut
    ``rate_hz × window`` par construction, sans avoir à estimer un débit.
    `--model` est obligatoire : sans lui le drapeau UART vaut 0, c'est-à-dire le
    chemin Mahalanobis par défaut du firmware (bug historique du Sprint 52).
    """
    return [
        sys.executable, "scripts/sensor_stream.py",
        "--port", port,
        "--dataset", dataset,
        "--model", model_flag,
        "--n-samples", str(n_samples),
        "--protocol-version", "3",
        "--rate-hz", str(rate_hz),
    ]


def stream_in_acquisition(probe, voltage_mv: int, cmd: list[str], label: str) -> dict:
    """Exécute un flux DANS une acquisition maintenue et rend son JSON — ou échoue en clair.

    Hors acquisition, la carte alimentée par la sonde s'effondre et redémarre en boucle
    (constat banc 2026-08-04) : `hold_run` est une nécessité mesurée, pas une commodité.

    Un flux échoué lève TOUJOURS — un contrôle d'intégrité ne se remplace pas par une valeur
    par défaut. Mais les trois modes d'échec ne demandent pas le même geste au banc, et
    étaient jusqu'ici indiscernables (`stderr` partait dans `DEVNULL`) :

      * code ≠ 0 **et** aucun JSON  → le flux n'a pas démarré (port, build, carte muette) ;
      * code ≠ 0 **mais** JSON écrit → le flux a tourné puis échoué en fin de course ;
      * JSON présent mais illisible  → écriture interrompue.

    La fin de la sortie d'erreur du sous-processus est jointe au message : au banc, un échec
    doit dire pourquoi.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        err_path = Path(tmp.name)
    full = list(cmd) + ["--output", str(out_path)]
    shell_cmd = ("cd " + str(ROOT) + " && "
                 + " ".join(f"'{c}'" if " " in c else c for c in full)
                 + f" 2> {err_path}")
    rcode = lp.hold_run(probe, voltage_mv, shell_cmd)
    try:
        journal = err_path.read_text(encoding="utf-8", errors="replace").strip()[-800:]
        detail = f"\n--- sortie d'erreur (fin) ---\n{journal}" if journal else ""
        if rcode != 0 and not out_path.is_file():
            raise RuntimeError(
                f"flux « {label} » : n'a produit AUCUN résultat (code {rcode}). Le flux "
                f"n'a pas démarré — vérifier le port de la carte, le binaire flashé et "
                f"l'alimentation.{detail}"
            )
        if not out_path.is_file():
            raise RuntimeError(
                f"flux « {label} » : code de retour 0 mais aucun fichier de résultat — "
                f"contrat de `sensor_stream.py` rompu.{detail}"
            )
        try:
            resultat = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"flux « {label} » : résultat illisible ({exc}) — écriture interrompue. "
                f"Aucune valeur par défaut n'est substituée.{detail}"
            ) from exc
        if rcode != 0:
            raise RuntimeError(
                f"flux « {label} » : résultat écrit MAIS code de retour {rcode} — le flux "
                f"a tourné puis échoué en fin de course, le contrôle d'intégrité n'est pas "
                f"concluant.{detail}"
            )
        return resultat
    finally:
        out_path.unlink(missing_ok=True)
        err_path.unlink(missing_ok=True)


def measure_current(probe, window_s: float, voltage_mv: int,
                    cmd: list[str] | None, settle_s: float) -> tuple[float, float]:
    """Courant moyen (A) sur une fenêtre, en exécutant `cmd` pendant toute la durée.

    Le flux est lancé `settle_s` avant l'ouverture de la fenêtre pour que la
    cadence soit déjà en régime établi, et n'est arrêté qu'après.
    """
    proc = None
    if cmd is not None:
        proc = subprocess.Popen(cmd, cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(settle_s)
    try:
        samples, voltage_v, _summary = lp.capture(
            probe, freq="1k", duration_s=window_s,
            voltage_mv=voltage_mv, acqmode="stat",
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return float(np.mean(samples)), voltage_v


def build_cell(model: str, encoding: str, i_runs_a: list[float], i_idle_a: float,
               voltage_v: float, window_s: float, rate_hz: float,
               warmup_a: float, idle_runs: list[dict] | None = None,
               session_indices: list[int] | None = None) -> dict:
    """Cellule JSON — schéma compatible `run_s50_energy.py`, champs mesurés en plus.

    Les grandeurs mesurées sont le courant moyen sous charge et la référence au
    repos. Les grandeurs que ce banc ne peut pas produire (profil par phase,
    énergie de mise à jour, énergie par inférence) gardent ``"à mesurer"`` avec
    leur raison propre.

    `idle_runs` et `session_indices` (S5301) sont ADDITIFS : `i_idle_a` garde sa
    sémantique du Sprint 50 (moyenne des repos de la session) et `delta_vs_idle_a`
    avec elle — les 8 cellules déjà écrites en dépendent, ainsi que
    `run_s50_energy.write_autonomy`.
    """
    n_inference = int(round(rate_hz * window_s))
    i_mean_a = float(np.mean(i_runs_a))
    # Écart-type des répétitions : sans lui, rien ne dit si un écart entre deux
    # cellules dépasse la dispersion du banc (mesurée à ±0,06 mA).
    i_std_a = float(np.std(i_runs_a, ddof=1)) if len(i_runs_a) > 1 else 0.0
    # S5301 — rang des acquisitions dans la session : c'est la variable explicative
    # du contre-balancement (l'ordre était jusqu'ici confondu avec la condition).
    measurement_extra: dict = {}
    if idle_runs is not None:
        measurement_extra["i_idle_runs_a"] = [
            {"session_index": int(r["session_index"]), "i_a": float(r["i_a"])}
            for r in idle_runs
        ]
        # Repos du RÉGIME ÉTABLI : `i_idle_a` (moyenne de tous les repos) reste la
        # référence historique S50, mais elle inclut la queue d'établissement. La
        # référence exploitable pour un delta est celle-ci.
        etabli = cb.established_regime(idle_runs)
        measurement_extra["i_idle_established_a"] = float(
            sum(v for _, v in etabli) / len(etabli)
        )
        measurement_extra["n_idle_discarded_as_settling"] = len(idle_runs) - len(etabli)
        measurement_extra["delta_vs_idle_established_a"] = (
            i_mean_a - measurement_extra["i_idle_established_a"]
        )
    if session_indices is not None:
        measurement_extra["session_indices"] = [int(i) for i in session_indices]
    return {
        "model": model,
        "encoding": encoding,
        "tension_v": float(voltage_v),
        "n_inference": n_inference,
        "n_update": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "lpm01a_current",
        "method": (
            "courant moyen mesuré sous charge d'inférence à cadence imposée "
            "(mode statique, préchauffage de session écarté)"
        ),
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "phases_na_reason": (
            "LPM01A sans voie de synchronisation + mode dynamique inaccessible "
            "(carte à 63 mA > plafond 59 mA) — profil temporel par phase impossible"
        ),
        "total_uj": A_MESURER,
        "energy_uj_per_inference": A_MESURER,
        "energy_na_reason": NA_PER_INFERENCE,
        "energy_uj_per_update": A_MESURER,
        "energy_update_na_reason": (
            "campagne d'inférence seule ; isoler la MAJ CL exige la même mesure "
            "avec --update, et se heurte à la même référence de scrutation active"
        ),
        "by_component": {
            "mcu": A_MESURER,
            "periph": A_MESURER,
            "sensor": "na",
            "sensor_na_reason": "capteurs simulés par UART (S5001)",
        },
        # Grandeurs réellement mesurées sur carte, conservées pour audit/recalcul.
        "current_measurement": {
            "i_mean_a": i_mean_a,
            "i_std_a": i_std_a,
            "i_runs_a": [float(x) for x in i_runs_a],
            "n_repeats": len(i_runs_a),
            "i_idle_a": float(i_idle_a),
            "delta_vs_idle_a": i_mean_a - float(i_idle_a),
            "rate_hz": float(rate_hz),
            "window_s": float(window_s),
            "warmup_discarded_a": float(warmup_a),
            **measurement_extra,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--rate-hz", type=float, default=100.0,
                        help="cadence d'inférence imposée (Hz) — fixe N exactement")
    parser.add_argument("--window", type=float, default=10.0, help="durée de fenêtre (s)")
    parser.add_argument("--settle", type=float, default=2.0,
                        help="délai d'établissement du flux avant la fenêtre (s)")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--board-port", required=True,
                        help="port de la CARTE (ST-Link VCP) — de préférence /dev/serial/by-id/…")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "exp_S50_energy")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto par défaut)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="répétitions par cellule → écart-type (dispersion du banc)")
    parser.add_argument("--only", default=None,
                        help="cellules à mesurer, séparées par des virgules, "
                             "ex. « maha_int8 » (build dédié) ou « ewc_fp32,hdc_fp32 »")
    parser.add_argument("--interleave-idle", action="store_true",
                        help="S5301 — intercale une mesure de repos AVANT et APRÈS chaque "
                             "cellule (au lieu du bloc de repos en tête) : sépare l'ordre "
                             "de la condition et produit counterbalance.json")
    parser.add_argument("--warmup-repeats", type=int, default=1,
                        help="nombre d'acquisitions de préchauffage JETÉES en début de "
                             "session (défaut 1 = comportement S50)")
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    args = parser.parse_args(argv)

    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    n_samples = int(args.rate_hz * (args.window + args.settle + 4.0))

    if args.only:
        cells = []
        for token in args.only.split(","):
            model, _, encoding = token.strip().rpartition("_")
            if (model, encoding) not in STREAM_MODEL:
                parser.error(f"cellule inconnue : {token.strip()}")
            cells.append((model, encoding))
    else:
        # Les cellules à firmware dédié ne sont PAS mesurables sur ce build.
        cells = [c for c in STREAM_MODEL if c not in BUILD_SPECIFIC]
        for (m, e), flag in BUILD_SPECIFIC.items():
            print(f"[courant] {m}_{e} ignorée : exige un build {flag} "
                  f"(la mesurer par « --only {m}_{e} » après flash)")

    args.out.mkdir(parents=True, exist_ok=True)
    probe = lp.PowerShield(args.port)
    try:
        probe.take_control()

        # 1. Rebut de préchauffage — la 1re acquisition d'une session est biaisée.
        #    `--warmup-repeats` permet d'allonger l'établissement (piste S5301 si le
        #    verdict sort en « dérive partielle ») ; la dernière valeur est conservée
        #    pour traçabilité, toutes sont consignées dans counterbalance.json.
        warmup_all = [lp.warmup(probe, voltage_mv)
                      for _ in range(max(1, args.warmup_repeats))]
        warmup_a = warmup_all[-1]
        for k, w in enumerate(warmup_all):
            print(f"[courant] préchauffage {k} écarté : {w * MA_PER_A:.3f} mA (non retenu)")

        # `session_index` : rang de CHAQUE acquisition dans la session. C'est la
        # variable explicative du contre-balancement — sans elle, l'ordre reste
        # confondu avec la condition (S5301).
        session = {"index": 0}
        sequence: list[dict] = []

        def acquire(condition: str, cmd, settle_s: float) -> tuple[float, float]:
            i_a, voltage_v = measure_current(probe, args.window, voltage_mv, cmd, settle_s)
            sequence.append({"session_index": session["index"],
                             "condition": condition, "i_a": float(i_a)})
            session["index"] += 1
            return i_a, voltage_v

        idle_points: list[dict] = []

        def acquire_idle() -> float:
            i_a, _ = acquire("idle", None, 0.0)
            idle_points.append({"session_index": sequence[-1]["session_index"], "i_a": i_a})
            print(f"[courant] repos #{sequence[-1]['session_index']} : "
                  f"{i_a * MA_PER_A:.3f} mA")
            return i_a

        cell_runs: dict[tuple[str, str], list[float]] = {c: [] for c in cells}
        cell_indices: dict[tuple[str, str], list[int]] = {c: [] for c in cells}
        voltage_v = float(calib.get("supply_voltage_v", 3.3))

        def acquire_cell(cell: tuple[str, str]) -> None:
            nonlocal voltage_v
            flag = STREAM_MODEL[cell]
            cmd = stream_command(flag, args.dataset, args.board_port,
                                 n_samples, args.rate_hz)
            i_run, voltage_v = acquire(f"{cell[0]}_{cell[1]}", cmd, args.settle)
            cell_runs[cell].append(i_run)
            cell_indices[cell].append(sequence[-1]["session_index"])

        if args.interleave_idle:
            # 2/3. Séquence contre-balancée : repos → cellule → repos → cellule → … → repos.
            for _ in range(args.repeats):
                for cell in cells:
                    acquire_idle()
                    acquire_cell(cell)
            acquire_idle()
        else:
            # 2. Référence au repos, mesurée APRÈS le préchauffage (protocole S50).
            for _ in range(args.repeats):
                acquire_idle()
            # 3. Une cellule par modèle × encodage, à cadence identique.
            for cell in cells:
                for _ in range(args.repeats):
                    acquire_cell(cell)

        idle_values = [p["i_a"] for p in idle_points]
        i_idle = float(np.mean(idle_values))
        print(f"[courant] référence repos : {i_idle * MA_PER_A:.3f} mA "
              f"± {np.std(idle_values, ddof=1) * MA_PER_A if len(idle_values) > 1 else 0.0:.3f} "
              f"(n={len(idle_values)})")

        for cell in cells:
            model, encoding = cell
            runs = cell_runs[cell]
            cell_json = build_cell(model, encoding, runs, i_idle, voltage_v,
                                   args.window, args.rate_hz, warmup_a,
                                   idle_runs=idle_points,
                                   session_indices=cell_indices[cell])
            i_mean = cell_json["current_measurement"]["i_mean_a"]
            i_std = cell_json["current_measurement"]["i_std_a"]
            out_path = args.out / f"{model}_{encoding}.json"
            out_path.write_text(json.dumps(cell_json, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            delta_ma = (i_mean - i_idle) * MA_PER_A
            print(f"[courant] {model:7s} {encoding:5s} "
                  f"({STREAM_MODEL[cell]:12s}) : "
                  f"{i_mean * MA_PER_A:7.3f} ± {i_std * MA_PER_A:.3f} mA  "
                  f"(repos {delta_ma:+.3f} mA) → {out_path.name}")

        if args.interleave_idle:
            # 4. Verdict S5301 — CALCULÉ par la règle codée, jamais saisi.
            all_cell_currents = [float(np.mean(cell_runs[c])) for c in cells if cell_runs[c]]
            etabli = cb.established_regime(idle_points)
            dispersion = cb.bench_dispersion_a([v for _, v in etabli])
            verdict, rationale = cb.counterbalance_verdict(
                idle_points, all_cell_currents, dispersion
            )
            report = {
                "protocol": "repos et cellules alternés, session unique "
                            f"({args.repeats} répétitions, {len(cells)} cellules)",
                "sequence": sequence,
                "idle_by_rank": idle_points,
                "drift_slope_ma_per_acquisition":
                    cb.idle_drift_slope(idle_points) * MA_PER_A,
                "bench_dispersion_a": dispersion,
                "idle_established": [{"session_index": x, "i_a": v} for x, v in etabli],
                "i_idle_established_a": float(sum(v for _, v in etabli) / len(etabli)),
                "i_idle_all_a": i_idle,
                "cells": {f"{m}_{e}": float(np.mean(cell_runs[(m, e)]))
                          for (m, e) in cells if cell_runs[(m, e)]},
                "verdict": verdict,
                "verdict_rationale": rationale,
                "rate_hz": float(args.rate_hz),
                "window_s": float(args.window),
                "n_repeats": int(args.repeats),
                "tension_v": float(voltage_v),
                "warmup_discarded_a": [float(w) for w in warmup_all],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cb_path = args.out / "counterbalance.json"
            cb_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
            print(f"[courant] verdict contre-balancement : {verdict} → {cb_path}")
            print(f"[courant]   {rationale}")
    finally:
        probe.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
