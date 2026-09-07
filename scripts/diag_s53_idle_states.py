"""
diag_s53_idle_states.py — De quel « repos » parle-t-on ? (diagnostic de banc S5301b)

POURQUOI CE DIAGNOSTIC EXISTE. Deux mesures de repos prises à quelques minutes
d'intervalle sur le même banc, même firmware, ont donné des valeurs franchement
différentes : le repos d'une carte jamais streamée d'un côté, le repos intercalé après
un flux interrompu de l'autre. Tant que cet écart n'est pas expliqué, on ne sait pas
quelle référence le protocole contre-balancé (`run_s50_board_current.py
--interleave-idle`) a réellement comparée aux cellules — donc on ne sait pas ce que vaut
le verdict S5301, ni sur quelle référence reposera le protocole delta.

HYPOTHÈSE TESTÉE. `measure_current` interrompt le flux par `proc.terminate()` : le port
série se ferme, DTR/RTS retombent, et la carte pourrait rester tenue dans un autre état
(reset maintenu, périphérique arrêté) — auquel cas les fenêtres « repos » du protocole
mesurent une carte à l'arrêt et non une carte en attente.

CE QUI EST MANIPULÉ. Uniquement ce que fait l'HÔTE, le firmware étant identique dans les
quatre états — port fermé, port ouvert et maintenu sans trafic, flux en cours, repos
juste après interruption d'un flux. Les états sont mesurés en cycles répétés : le rang
de chaque acquisition est consigné, l'ordre ne peut donc pas être confondu avec la
condition (leçon S5301). Une trame de contrôle est envoyée après chaque cycle : une
carte muette ferait passer l'hypothèse « maintenue en reset » du plausible au démontré.

La règle de verdict vit dans `src/evaluation/counterbalance.py::idle_state_verdict`
(calculée et testée hors banc, jamais saisie).

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : ce pilote n'écrit que ce que la sonde a mesuré.

Usage :
    python scripts/diag_s53_idle_states.py \\
        --board-port /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_…-if02
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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


lp = _load("lpm01a_probe", ROOT / "scripts" / "lpm01a_probe.py")
cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")
s50 = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")

MA_PER_A = 1000

#: Modèle utilisé pour l'état « flux » et pour les trames de contrôle. EWC : cellule de
#: référence des campagnes S50/S53, et `--model` explicite (à 0, le drapeau UART
#: sélectionne le chemin Mahalanobis par défaut — bug historique du Sprint 52).
DIAG_MODEL = "ewc"


def capture_idle(probe, window_s: float, voltage_mv: int) -> float:
    """Courant moyen (A) sur une fenêtre, sans rien lancer côté hôte."""
    samples, _voltage, _summary = lp.capture(
        probe, freq="1k", duration_s=window_s, voltage_mv=voltage_mv, acqmode="stat",
    )
    return float(np.mean(samples))


def board_responds(port: str, dataset: str, n_samples: int, rate_hz: float) -> tuple[bool, int]:
    """La carte répond-elle encore à des trames ? (reçus == envoyés).

    Mesuré, pas supposé : c'est ce test qui distingue « la carte attend » de « la carte
    est tenue à l'arrêt ». Renvoie ``(toutes_reçues, n_reçues)``.

    `--n-tasks 1` : le flux répartit sinon `n_samples // n_tasks` trames par tâche et en
    envoie donc moins que demandé — le compte attendu ne serait plus exact.
    """
    cmd = s50.stream_command(DIAG_MODEL, dataset, port, n_samples, rate_hz)
    cmd += ["--n-tasks", "1"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    match = re.search(r"n_samples:\s*(\d+)", proc.stdout)
    received = int(match.group(1)) if match else 0
    return (received == n_samples and proc.returncode == 0), received


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--board-port", required=True,
                        help="port de la CARTE (ST-Link VCP)")
    parser.add_argument("--port", default=None, help="port de la SONDE (auto par défaut)")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--window", type=float, default=10.0,
                        help="durée de chaque fenêtre d'acquisition (s)")
    parser.add_argument("--settle", type=float, default=2.0,
                        help="délai d'établissement du flux avant la fenêtre (s)")
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=3,
                        help="cycles des quatre états (l'ordre est consigné, pas confondu)")
    parser.add_argument("--probe-samples", type=int, default=20,
                        help="trames de la vérification de réponse après chaque cycle")
    parser.add_argument("--with-power-cycle", action="store_true",
                        help="mesure un repos supplémentaire APRÈS coupure/remise de "
                             "l'alimentation cible par la sonde : teste si le niveau "
                             "absolu dépend de l'historique depuis la mise sous tension "
                             "(piste du décalage de niveau ENTRE sessions)")
    parser.add_argument("--settling-points", type=int, default=1,
                        help="repos consécutifs mesurés après la remise sous tension "
                             "(donne le temps d'établissement, pas seulement un écart)")
    parser.add_argument("--power-off-s", type=float, default=10.0,
                        help="durée de coupure de l'alimentation cible (s)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "exp_S53_counterbalance"
                        / "idle_states_diagnostic.json")
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    args = parser.parse_args(argv)

    try:
        import serial  # noqa: F401
    except ImportError:
        parser.error("pyserial requis : pip install pyserial")

    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * MA_PER_A)
    n_stream = int(args.rate_hz * (args.window + args.settle + 4.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    probe = lp.PowerShield(args.port)
    sequence: list[dict] = []
    states: dict[str, list[float]] = {
        cb.STATE_PORT_CLOSED: [], cb.STATE_PORT_OPEN: [],
        cb.STATE_STREAM: [], cb.STATE_POST_STREAM: [],
    }
    responsive: dict[str, list[bool]] = {cb.STATE_POST_STREAM: [], cb.STATE_PORT_CLOSED: []}
    index = {"n": 0}

    def record(state: str, i_a: float) -> None:
        sequence.append({"session_index": index["n"], "state": state, "i_a": float(i_a)})
        states[state].append(float(i_a))
        print(f"[diag] #{index['n']:2d} {state:20s} : {i_a * MA_PER_A:7.3f} mA")
        index["n"] += 1

    try:
        probe.take_control()

        # Rebut de préchauffage : la première acquisition d'une session est biaisée
        # (constat mesuré S50, cf. `lpm01a_probe.warmup`).
        warmup_a = lp.warmup(probe, voltage_mv)
        print(f"[diag] préchauffage écarté : {warmup_a * MA_PER_A:.3f} mA (non retenu)")

        ok_initial, n_initial = board_responds(args.board_port, args.dataset,
                                               args.probe_samples, args.rate_hz)
        print(f"[diag] réponse initiale de la carte : {n_initial}/{args.probe_samples} "
              f"trames → {'oui' if ok_initial else 'NON'}")

        for cycle in range(args.repeats):
            # A — port fermé, aucun trafic (le « repos frais » du protocole S50).
            record(cb.STATE_PORT_CLOSED, capture_idle(probe, args.window, voltage_mv))
            ok_a, _ = board_responds(args.board_port, args.dataset,
                                     args.probe_samples, args.rate_hz)
            responsive[cb.STATE_PORT_CLOSED].append(ok_a)

            # B — port ouvert et MAINTENU, lignes de contrôle relâchées, zéro trame.
            #     Isole l'effet de l'ouverture du port de celui du trafic.
            with serial.Serial(args.board_port, timeout=args.window) as ser:
                ser.dtr = False
                ser.rts = False
                time.sleep(args.settle)
                record(cb.STATE_PORT_OPEN, capture_idle(probe, args.window, voltage_mv))

            # C — flux en cours, puis D — repos juste après son interruption, sans rien
            #     toucher d'autre : c'est exactement la transition que fait
            #     `measure_current` entre une cellule et la fenêtre de repos suivante.
            cmd = s50.stream_command(DIAG_MODEL, args.dataset, args.board_port,
                                     n_stream, args.rate_hz)
            proc = subprocess.Popen(cmd, cwd=ROOT,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                time.sleep(args.settle)
                record(cb.STATE_STREAM, capture_idle(probe, args.window, voltage_mv))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            record(cb.STATE_POST_STREAM, capture_idle(probe, args.window, voltage_mv))

            ok_d, n_d = board_responds(args.board_port, args.dataset,
                                       args.probe_samples, args.rate_hz)
            responsive[cb.STATE_POST_STREAM].append(ok_d)
            print(f"[diag] cycle {cycle} — réponse après flux interrompu : "
                  f"{n_d}/{args.probe_samples} trames → {'oui' if ok_d else 'NON'}")

        # Optionnel — le niveau absolu dépend-il de l'historique depuis la mise sous
        # tension ? Le repos est remesuré après une coupure franche de l'alimentation
        # cible par la sonde. Conservé HORS des états du verdict (question distincte :
        # celui-ci porte sur le couple port/flux, à alimentation continue).
        power_cycle: dict | None = None
        if args.with_power_cycle:
            lp.power_off(probe)
            time.sleep(args.power_off_s)
            # `power_off` a posé `pwrend off` : sans ce rétablissement, la sortie
            # retomberait à la fin du maintien et la carte redémarrerait une fois de plus.
            probe.command("pwrend on")
            lp.power_on(probe, voltage_mv, hold_s=args.settle)
            time.sleep(args.settle)
            # Série de repos consécutifs après la remise sous tension : un point unique
            # dirait « c'est différent » sans dire « pendant combien de temps ». La pente
            # (réutilise `idle_drift_slope`, S5301) donne le temps d'établissement à
            # respecter avant toute prise de référence.
            settling = []
            for k in range(max(1, args.settling_points)):
                i_k = capture_idle(probe, args.window, voltage_mv)
                settling.append({"session_index": k, "i_a": float(i_k)})
                print(f"[diag] repos après coupure d'alimentation #{k} : "
                      f"{i_k * MA_PER_A:7.3f} mA")
            # Discriminant du mécanisme : l'OUVERTURE seule du port suffit-elle à faire
            # basculer le niveau, ou faut-il de vraies trames ? Le premier cas désigne
            # l'état des lignes côté hôte/ST-Link, le second un état du firmware.
            with serial.Serial(args.board_port, timeout=args.window) as ser:
                ser.dtr = False
                ser.rts = False
                time.sleep(args.settle)
                i_open = capture_idle(probe, args.window, voltage_mv)
            print(f"[diag] repos port ouvert, avant toute trame : {i_open * MA_PER_A:7.3f} mA")

            # Deuxième discriminant : `sensor_stream` ouvre le port en pulsant DTR
            # (True → False) avant d'émettre. Cette impulsion seule, sans aucune trame,
            # suffit-elle ? Sépare « impulsion des lignes de contrôle » de « première
            # trame traitée par le firmware ».
            with serial.Serial(args.board_port, timeout=args.window) as ser:
                ser.dtr = True
                time.sleep(args.settle)
                ser.dtr = False
                time.sleep(args.settle)
                i_pulse = capture_idle(probe, args.window, voltage_mv)
            print(f"[diag] repos après impulsion DTR seule : {i_pulse * MA_PER_A:7.3f} mA")

            ok_pc, n_pc = board_responds(args.board_port, args.dataset,
                                         args.probe_samples, args.rate_hz)
            print(f"[diag] réponse après coupure : {n_pc}/{args.probe_samples} trames")

            # Puis le MÊME repos, après le premier flux reçu depuis la mise sous tension.
            # C'est la variable qui distingue les deux niveaux observés : le temps ne les
            # sépare pas (la série ci-dessus est plate), le premier flux peut-être si.
            after_first = []
            for k in range(max(1, args.settling_points)):
                i_k = capture_idle(probe, args.window, voltage_mv)
                after_first.append({"session_index": k, "i_a": float(i_k)})
                print(f"[diag] repos après le 1er flux reçu #{k} : "
                      f"{i_k * MA_PER_A:7.3f} mA")

            power_cycle = {
                "i_idle_after_power_cycle_a": settling[0]["i_a"],
                "settling_series_a": settling,
                "settling_slope_a_per_acquisition": cb.idle_drift_slope(settling),
                "i_idle_port_open_before_any_frame_a": float(i_open),
                "i_idle_after_dtr_pulse_before_any_frame_a": float(i_pulse),
                "post_first_stream_series_a": after_first,
                "step_first_stream_a": (
                    float(np.mean([p["i_a"] for p in after_first]))
                    - float(np.mean([p["i_a"] for p in settling]))
                ),
                "power_off_s": float(args.power_off_s),
                "board_responsive": bool(ok_pc),
                "n_frames_received": int(n_pc),
            }

        dispersion = cb.pooled_dispersion_a(states.values())
        responsive_all = {k: all(v) for k, v in responsive.items() if v}
        verdict, rationale = cb.idle_state_verdict(states, responsive_all, dispersion)

        report = {
            "protocol": (
                f"quatre états de repos/charge en cycles alternés, session unique "
                f"({args.repeats} cycles), rang consigné ; seul l'hôte varie, "
                f"firmware identique"
            ),
            "board_port": args.board_port,
            "dataset": args.dataset,
            "model": DIAG_MODEL,
            "rate_hz": float(args.rate_hz),
            "window_s": float(args.window),
            "settle_s": float(args.settle),
            "n_cycles": int(args.repeats),
            "warmup_discarded_a": float(warmup_a),
            "sequence": sequence,
            "states_a": {k: v for k, v in states.items()},
            "state_mean_a": {k: float(np.mean(v)) for k, v in states.items() if v},
            "state_std_a": {k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                            for k, v in states.items() if v},
            "pooled_dispersion_a": float(dispersion),
            "board_responsive_initial": bool(ok_initial),
            "board_responsive_after": {k: [bool(x) for x in v]
                                       for k, v in responsive.items() if v},
            "probe_frames": int(args.probe_samples),
            "power_cycle_check": power_cycle,
            "verdict": verdict,
            "verdict_rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[diag] verdict : {verdict} → {args.out}")
        print(f"[diag]   {rationale}")
    finally:
        probe.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
