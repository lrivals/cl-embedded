"""
run_s50_energy_delta.py — Campagne énergie S5002 par **protocole delta** (LPM01A).

POURQUOI CE PILOTE (contraintes matérielles mesurées le 2026-08-04, cf.
`docs/context/lpm01a_setup.md`) :
    1. Le flux du LPM01A ne transporte aucune voie de synchronisation → la
       segmentation par phases (`derive_phase_windows`) est **inapplicable**.
    2. La NUCLEO-F439ZI à 180 MHz dépasse le plafond du mode dynamique de la
       sonde (« Overcurrent >59mA ») → **pas de profil temporel** à 100 kSPS.
    3. Hors acquisition, la sortie de la sonde ne tient pas la carte : elle
       redémarre en boucle → toute commande hôte (flash, streaming) doit tourner
       **pendant** une acquisition maintenue.

    Il reste le **courant moyen** (mode statique, jusqu'à 200 mA), et donc
    l'énergie marginale par inférence obtenue par différence de deux fenêtres
    réelles — c'est ce que fait ce pilote. Aucune valeur n'est extrapolée.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ :
    Une cellule dont le delta de courant n'est pas concluant (≤ 0) reste
    ``"à mesurer"`` avec un ``na_reason``, jamais 0.

Réutilisation stricte :
    - primitives d'énergie : `energy_capture.energy_uj_per_inference_delta`
    - pilotage de la sonde  : `lpm01a_probe` (session, capture statique, maintien)
    - schéma JSON de sortie : identique à `run_s50_energy.py` (mêmes clés)

Usage :
    # 1. fenêtre au repos, puis fenêtre active pendant laquelle STREAM_CMD tourne
    python scripts/run_s50_energy_delta.py --model ewc --encoding int8 \\
        --n-inference 500 --window 10 \\
        --stream-cmd "python scripts/sensor_stream.py \\
                      --port /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_...-if02 \\
                      --dataset monitoring --model ewc-int8 --n-samples 900" \\
        --out experiments/exp_S50_energy/

Deux pièges de la commande de flux (vérifiés, cf. S5008) :
    - `sensor_stream.py` n'a **pas** de `--n` : c'est `--n-samples`.
    - `--model` est obligatoire : sans lui le drapeau UART vaut 0, c'est-à-dire le
      chemin par défaut du firmware — Mahalanobis — quel que soit le modèle visé
      (bug historique corrigé au Sprint 52). La cellule mesurerait alors autre chose.
    - le port de la **carte** est celui de l'ST-Link, pas celui de la sonde : les
      désigner par `/dev/serial/by-id/` évite l'interversion des `ttyACMn`.
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

A_MESURER = ec.A_MESURER
PHASES = ec.PHASES


def measure_mean_current(
    probe, window_s: float, voltage_mv: int, stream_cmd: str | None = None
) -> tuple[float, float]:
    """Mesure le courant moyen sur une fenêtre, en exécutant `stream_cmd` pendant.

    L'acquisition **statique** du LPM01A renvoie une unique valeur moyennée sur
    la fenêtre — c'est exactement la grandeur voulue par le protocole delta.
    Si `stream_cmd` est fourni, il est lancé juste avant la fenêtre et arrêté
    juste après : la fenêtre couvre donc l'activité mesurée.

    Parameters
    ----------
    probe : lpm01a_probe.PowerShield
        Session série ouverte, contrôle pris.
    window_s : float
        Durée de la fenêtre (s).
    voltage_mv : int
        Tension d'alimentation de la cible (mV).
    stream_cmd : str | None
        Commande hôte à exécuter pendant la fenêtre (``None`` = fenêtre au repos).

    Returns
    -------
    tuple[float, float]
        (courant moyen en A, tension relue en V).
    """
    proc = None
    if stream_cmd:
        proc = subprocess.Popen(stream_cmd, shell=True)
        time.sleep(1.0)  # laisse le flux s'établir avant d'ouvrir la fenêtre
    try:
        samples, voltage_v, _summary = lp.capture(
            probe,
            freq="1k",
            duration_s=window_s,
            voltage_mv=voltage_mv,
            acqmode="stat",
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return float(np.mean(samples)), voltage_v


def build_cell(
    model: str,
    encoding: str,
    i_idle_a: float,
    i_active_a: float,
    voltage_v: float,
    window_s: float,
    n_inference: int,
) -> dict:
    """Construit la cellule JSON (même schéma que `run_s50_energy.py`).

    Les champs que le protocole delta ne peut **pas** produire (profil par phase,
    énergie de mise à jour) restent ``"à mesurer"`` avec leur raison — ils ne
    sont pas remplis par une valeur de substitution.
    """
    per_inf = ec.energy_uj_per_inference_delta(
        i_idle_a, i_active_a, voltage_v, window_s, n_inference
    )
    measured = per_inf is not None

    cell = {
        "model": model,
        "encoding": encoding,
        "tension_v": float(voltage_v),
        "n_inference": int(n_inference),
        "n_update": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "lpm01a_delta" if measured else "placeholder",
        "method": "protocole delta (courant moyen actif − repos, mode statique)",
        # Le profil par phase exige une voie de synchronisation que la sonde n'a
        # pas, et un mode dynamique que le courant de la carte interdit.
        "phases_uj": {p: A_MESURER for p in PHASES},
        "phase_durations_s": {p: A_MESURER for p in PHASES},
        "phases_na_reason": (
            "LPM01A sans voie de synchronisation + mode dynamique inaccessible "
            "(Overcurrent >59mA) — profil temporel par phase impossible"
        ),
        "total_uj": A_MESURER,
        "energy_uj_per_inference": per_inf if measured else A_MESURER,
        "energy_uj_per_update": A_MESURER,
        "energy_update_na_reason": (
            "protocole delta inférence seule ; une campagne delta dédiée "
            "inférence+MAJ est nécessaire pour isoler la MAJ CL"
        ),
        "by_component": {
            "mcu": A_MESURER,
            "periph": A_MESURER,
            "sensor": "na",
            "sensor_na_reason": "capteurs simulés par UART (S5001)",
        },
        # Grandeurs brutes mesurées, conservées pour audit et recalcul.
        "delta_measurement": {
            "i_idle_a": float(i_idle_a),
            "i_active_a": float(i_active_a),
            "delta_a": float(i_active_a) - float(i_idle_a),
            "window_s": float(window_s),
        },
    }
    if not measured:
        cell["na_reason"] = (
            "delta de courant nul ou négatif : activité non distinguable du repos "
            "sur cette fenêtre (augmenter N ou la durée)"
        )
    return cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True, choices=("ewc", "hdc", "tinyol", "maha"))
    parser.add_argument("--encoding", required=True, choices=("fp32", "int8"))
    parser.add_argument(
        "--n-inference",
        type=int,
        required=True,
        help="nombre d'inférences réellement exécutées dans la fenêtre active",
    )
    parser.add_argument("--window", type=float, default=10.0, help="durée de fenêtre (s)")
    parser.add_argument(
        "--stream-cmd",
        required=True,
        help="commande générant les N inférences pendant la fenêtre active",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "exp_S50_energy")
    parser.add_argument("--port", default=None)
    parser.add_argument(
        "--hw-profile", type=Path, default=ROOT / "configs" / "hw_profile_f439zi.yaml"
    )
    args = parser.parse_args(argv)

    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)

    probe = lp.PowerShield(args.port)
    try:
        probe.take_control()
        # La 1re acquisition d'une session lit ~8 mA de trop (mesuré, cf.
        # `lpm01a_probe.warmup`). Sans ce rebut, la fenêtre de référence — qui
        # vient en premier — est surestimée et le delta ressort négatif.
        warmup_a = lp.warmup(probe, voltage_mv)
        print(f"[delta] préchauffage écarté : {warmup_a * 1e3:.3f} mA (non retenu)")
        print("[delta] fenêtre au repos…")
        i_idle, _ = measure_mean_current(probe, args.window, voltage_mv, None)
        print(f"[delta]   I_repos = {i_idle * 1e3:.3f} mA")

        print("[delta] fenêtre active…")
        i_active, voltage_v = measure_mean_current(probe, args.window, voltage_mv, args.stream_cmd)
        print(f"[delta]   I_actif = {i_active * 1e3:.3f} mA")
    finally:
        probe.release()

    cell = build_cell(
        args.model, args.encoding, i_idle, i_active, voltage_v, args.window, args.n_inference
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}_{args.encoding}.json"
    out_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")

    per_inf = cell["energy_uj_per_inference"]
    verdict = f"{per_inf:.3f} µJ/inférence" if isinstance(per_inf, float) else str(per_inf)
    print(f"[delta] {args.model}/{args.encoding} → {verdict} → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
