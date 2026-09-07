"""
run_s53_wfi.py — S5302 : mesure du repos réel (`__WFI`) et reprise du protocole delta.

CE QUE CE PILOTE MESURE, ET POURQUOI :

Jusqu'au Sprint 52, le firmware attendait chaque trame en scrutation active
(`while (!(USART3->SR & RXNE)) {}`) : le cœur tournait à 180 MHz 100 % du temps, y
compris « au repos ». Il n'existait donc **aucune référence de repos exploitable**, et
l'énergie marginale par inférence du protocole delta ressortait négative.

Le build `-DUART_WFI_IDLE` endort le cœur entre deux trames. Ce pilote mesure :

    1. `idle_reference.json` — le repos AVEC sommeil, contre-balancé (protocole S5301,
       le repos est mesuré à plusieurs rangs de la session) et comparé au repos en
       scrutation active mesuré dans la même campagne. Le gain du WFI est un résultat
       de déploiement en soi. Avec `--write-hw-profile`, ce repos remplit
       `puissance_watts.veille_uA` du profil matériel, `null` depuis le Sprint 33.
    2. `{model}_{encoding}.json` — les 8 cellules reprises sur ce build, schéma
       identique à `exp_S50_energy/` plus `firmware_build` et `infer_batch_n`.
    3. `delta_recovery.json` — µJ par inférence par différence, sur la référence WFI.
       `energy_capture.energy_uj_per_inference_delta` renvoie `None` si le delta reste
       négatif : dans ce cas la valeur reste `"à mesurer"` avec une raison ACTUALISÉE,
       jamais un chiffre.
    4. `batch_sweep.json` — régression `I_moy(N)` à cadence fixe, avec `N` fixé à la
       compilation (`-DINFER_BATCH_N`). C'est la seule voie mesurée vers un µJ par
       inférence pour les modèles rapides, dont le taux d'occupation est autrement noyé
       sous le plafond de cadence de l'UART (~209 Hz mesuré).

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ. Le pilote ne devine jamais le build flashé :
`--firmware-build` est OBLIGATOIRE et recopié tel quel dans les JSON.

Usage (le build correspondant doit être flashé AVANT) :
    python scripts/run_s53_wfi.py --mode idle --firmware-build UART_WFI_IDLE \\
        --board-port /dev/serial/by-id/…STLink… --write-hw-profile
    python scripts/run_s53_wfi.py --mode cells --firmware-build UART_WFI_IDLE \\
        --board-port … --idle-from experiments/exp_S53_wfi/idle_reference.json
    python scripts/run_s53_wfi.py --mode batch --firmware-build UART_WFI_IDLE \\
        --batch-n 50 --model ewc --board-port …
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")
rc = _load("run_s50_board_current", ROOT / "scripts" / "run_s50_board_current.py")

A_MESURER = ec.A_MESURER
DEFAULT_OUT = ROOT / "experiments" / "exp_S53_wfi"
MA = 1000.0   # A → mA, affichage seul

#: Préfixes des fichiers du répertoire de sortie qui ne sont PAS des cellules de
#: mesure (rapports, traces, références). Source unique : le profil matériel, la
#: reprise du delta et les tests lisent tous les cellules par `iter_cell_files`,
#: sinon un nouveau rapport déposé ici serait un jour relu comme une mesure.
NON_CELL_PREFIXES = ("idle_reference", "delta_recovery", "batch_sweep",
                     "batch_ref_preds", "control_stream", "wake_validation",
                     "autonomy_delta")


def iter_cell_files(out_dir: Path):
    """Fichiers de cellule `{modèle}_{encodage}.json` du répertoire de sortie."""
    return [p for p in sorted(out_dir.glob("*_*.json"))
            if not p.name.startswith(NON_CELL_PREFIXES)]


def measure_idle_reference(probe, args, voltage_mv: int) -> dict:
    """Repos contre-balancé sur le build courant (protocole S5301).

    Le repos est mesuré plusieurs fois, chaque acquisition portant son rang : sans
    cela, la queue d'établissement de session se confondrait avec l'effet du WFI —
    c'est exactement l'erreur que S5301 a mise au jour sur la campagne S50.
    """
    warmups = [lp.warmup(probe, voltage_mv) for _ in range(max(1, args.warmup_repeats))]
    points = []
    for k in range(args.repeats):
        i_a, voltage_v = rc.measure_current(probe, args.window, voltage_mv, None, 0.0)
        points.append({"session_index": k, "i_a": float(i_a)})
        print(f"[wfi] repos #{k} : {i_a * MA:.3f} mA")
    etabli = cb.established_regime(points)
    i_etabli = float(sum(v for _, v in etabli) / len(etabli))
    return {
        "firmware_build": args.firmware_build,
        "method": "courant moyen au repos, port hôte fermé, session contre-balancée "
                  "(rangs conservés, régime établi isolé)",
        "i_idle_runs_a": points,
        "i_idle_all_a": float(np.mean([p["i_a"] for p in points])),
        "i_idle_established_a": i_etabli,
        "i_idle_established_ma": i_etabli * MA,
        "bench_dispersion_a": cb.bench_dispersion_a([v for _, v in etabli]),
        "drift_slope_ma_per_acquisition": cb.idle_drift_slope(points) * MA,
        "n_idle_discarded_as_settling": len(points) - len(etabli),
        "warmup_discarded_a": [float(w) for w in warmups],
        "tension_v": float(voltage_v),
        "window_s": float(args.window),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_hw_profile_currents(path: Path, veille_a: float | None,
                              actif_a: float | None, source: str) -> list[str]:
    """Écrit les courants MESURÉS dans `puissance_watts` du profil matériel.

    `actif_mA` et `veille_uA` sont `null` depuis le Sprint 33 : la sonde n'avait
    jamais tourné. Les remplir est un critère d'acceptation de S5302 — mais seulement
    depuis une mesure, jamais depuis une estimation : un `None` en entrée laisse la
    clé telle quelle plutôt que d'écrire un chiffre inventé.

    La substitution est faite LIGNE À LIGNE, pas par aller-retour YAML : le fichier est
    intégralement commenté (le pourquoi de chaque coefficient) et `yaml.safe_dump`
    détruirait cette documentation. Une clé sœur `mesure_source` porte la provenance.

    Parameters
    ----------
    path : Path
        Profil matériel à modifier (``configs/hw_profile_f439zi.yaml``).
    veille_a, actif_a : float | None
        Courants mesurés en ampères. `None` ⇒ la clé correspondante n'est pas touchée.
    source : str
        Provenance recopiée telle quelle (fichier de mesure + build firmware).

    Returns
    -------
    list[str]
        Les clés effectivement écrites.

    Raises
    ------
    ValueError
        Si aucune mesure n'est fournie — écrire un profil sans mesure serait
        exactement le chiffre inventé que la règle projet interdit.
    """
    if veille_a is None and actif_a is None:
        raise ValueError(
            "aucun courant mesuré à écrire : le profil matériel ne se remplit pas "
            "par estimation (lancer --mode idle, puis --mode cells pour l'actif)."
        )

    written: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    indent = "    "
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("mesure_source:"):
            continue          # régénérée ci-dessous, jamais dupliquée
        if stripped.startswith("veille_uA:") and veille_a is not None:
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}veille_uA: {veille_a * 1e6:.1f}"
                       f"      # mesuré — courant au repos (µA)")
            written.append("veille_uA")
            continue
        if stripped.startswith("actif_mA:") and actif_a is not None:
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}actif_mA: {actif_a * 1e3:.3f}"
                       f"   # mesuré — courant actif moyen (mA)")
            written.append("actif_mA")
            continue
        out.append(line)
        if stripped.startswith("tension_v:"):
            ind = line[: len(line) - len(stripped)]
            out.append(f"{ind}mesure_source: \"{source}\"")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return written


def build_delta_recovery(cells: dict, i_idle_a: float, voltage_v: float,
                         window_s: float, firmware_build: str) -> dict:
    """µJ par inférence par différence, cellule par cellule, sur la référence fournie.

    Aucune cellule n'est forcée : si le delta reste ≤ 0, le champ garde ``"à mesurer"``
    avec la raison mesurée. C'est le comportement de
    `energy_capture.energy_uj_per_inference_delta`, réutilisé tel quel.
    """
    out = {}
    for key, cell in cells.items():
        mes = cell["current_measurement"]
        per_inf = ec.energy_uj_per_inference_delta(
            i_idle_a, mes["i_mean_a"], voltage_v, window_s, cell["n_inference"]
        )
        entry = {
            "i_mean_a": mes["i_mean_a"],
            "i_idle_a": float(i_idle_a),
            "delta_a": mes["i_mean_a"] - float(i_idle_a),
            "n_inference": cell["n_inference"],
        }
        if per_inf is None:
            entry["energy_uj_per_inference"] = A_MESURER
            entry["na_reason"] = (
                "delta de courant nul ou négatif face à la référence de repos "
                f"« {firmware_build} » : l'activité de cette cellule n'est pas "
                "séparable du repos sur cette fenêtre. Le contre-balancement S5301 "
                "ayant écarté l'artefact d'ordre, la piste restante est le taux "
                "d'occupation (augmenter N par -DINFER_BATCH_N, cf. batch_sweep.json)."
            )
        else:
            entry["energy_uj_per_inference"] = float(per_inf)
            entry["method"] = f"delta vs repos {firmware_build}"
        out[key] = entry
    return out


def propagate_delta_into_cells(recovery: dict, out_dir: Path) -> list[str]:
    """Reporte le résultat du delta DANS chaque fichier de cellule.

    Les cellules sont écrites par `run_s50_board_current.build_cell`, dont le
    protocole (courant moyen, sans soustraction) laisse `energy_uj_per_inference`
    à ``"à mesurer"`` avec la raison du Sprint 50. Une fois le delta calculé sur la
    référence de repos WFI, laisser cette raison en place ferait dire à deux
    fichiers du MÊME répertoire deux choses contraires sur la même cellule — et
    c'est la raison, plus lisible qu'un JSON annexe, qui serait recopiée.

    Ne touche que les cellules du build du rapport ; ne fabrique jamais de valeur
    (une cellule restée N/A reçoit la raison ACTUALISÉE du delta, pas un chiffre).
    """
    touched: list[str] = []
    for path in iter_cell_files(out_dir):
        cell = json.loads(path.read_text(encoding="utf-8"))
        entry = recovery["cells"].get(path.stem)
        if entry is None or cell.get("firmware_build") != recovery["firmware_build"]:
            continue
        cell["energy_uj_per_inference"] = entry["energy_uj_per_inference"]
        if entry["energy_uj_per_inference"] == A_MESURER:
            cell["energy_na_reason"] = entry["na_reason"]
            cell.pop("energy_method", None)
        else:
            cell["energy_method"] = entry["method"]
            cell["energy_delta_a"] = entry["delta_a"]
            cell["energy_source"] = "delta_recovery.json"
            cell.pop("energy_na_reason", None)
        path.write_text(json.dumps(cell, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        touched.append(path.name)
    return touched


#: Un point de lot n'est retenu dans la régression que si la cadence RÉELLEMENT
#: atteinte reste à moins de 5 % de la cadence demandée. Au-delà, la carte ne tient
#: plus la trame — `sensor_stream` sature EN SILENCE (aucune trame perdue, aucun CRC,
#: le flux tourne simplement moins vite) : le courant moyen correspond alors à MOINS
#: d'inférences par seconde que `rate × N`, ce qui aplatit la pente et sous-estime
#: l'énergie. La saturation est donc CONSTATÉE sur la cadence mesurée, jamais déduite
#: d'une latence supposée.
SATURATION_RATE_TOLERANCE = 0.95


def control_stream(probe, lp_mod, voltage_mv: int, cmd: list[str],
                   out_json: Path) -> dict:
    """Flux de contrôle d'un point de lot, sous alimentation maintenue.

    Mesure ce que la mesure de courant ne peut pas voir : la cadence réellement
    atteinte (détection de saturation), les erreurs CRC, et les prédictions
    échantillon par échantillon (parité du lot vis-à-vis de `N = 1`).

    Hors acquisition, la sortie de la sonde ne tient pas la carte (elle redémarre en
    boucle) : le flux passe donc par `lpm01a_probe.hold_run`.
    """
    shell = " ".join(str(c) for c in cmd) + f" --dump-samples --output {out_json}"
    lp_mod.hold_run(probe, voltage_mv, shell)
    return json.loads(out_json.read_text(encoding="utf-8"))


def prediction_parity(reference: dict[str, int], samples: list[dict]) -> tuple[int, int]:
    """Compare les prédictions à celles de `N = 1`, appariées par vecteur de features.

    L'appariement se fait sur les features et non sur l'indice : `sensor_stream`
    tire les échantillons du dataset à chaque exécution, deux flux successifs ne
    parcourent donc pas la même séquence. Seuls les échantillons communs sont
    comparés — leur nombre est reporté, pour que la force de la preuve soit lisible.
    """
    common = 0
    mismatches = 0
    for s in samples:
        fkey = ",".join(f"{f:.7g}" for f in s["features"])
        if fkey in reference:
            common += 1
            if reference[fkey] != s["pred"]:
                mismatches += 1
    return common, mismatches


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Régression `y = a·x + b` → (pente, ordonnée, r²). r²=0 si indéterminé."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def apply_fit(entry: dict, voltage_v: float, rate_hz: float) -> dict:
    """Régression `I(N)` d'une cellule de lot, points saturés écartés.

    Fonction PURE (aucune mesure, aucun accès banc) : elle est appelée par le mode
    `batch` après chaque point, et rejouable telle quelle sur un fichier existant.

    Les clés du verdict opposé sont RETIRÉES à chaque passage : sans cela, une
    cellule d'abord non concluante puis chiffrée par un point supplémentaire
    conserverait sa `na_reason` — un fichier portant à la fois une valeur et la
    raison de son absence.
    """
    fit_points = [p for p in entry["points"] if not p.get("saturated")]
    entry["n_points_excluded_saturation"] = len(entry["points"]) - len(fit_points)
    xs = [float(p["n"]) for p in fit_points]
    ys = [p["i_mean_a"] for p in fit_points]
    for stale in ("energy_uj_per_inference", "method", "na_reason",
                  "slope_a_per_inference_per_frame", "intercept_a", "r2"):
        entry.pop(stale, None)

    if len(xs) >= 3:
        slope, intercept, r2 = linear_fit(xs, ys)
        entry["slope_a_per_inference_per_frame"] = slope
        entry["intercept_a"] = intercept
        entry["r2"] = r2
        # E_par_inférence = pente × V / cadence  (la pente est un courant par
        # inférence-et-par-trame ; à `rate` trames/s cela fait `rate` inférences
        # supplémentaires par seconde et par unité de N).
        per_inf_uj = slope * voltage_v / rate_hz * 1e6
        if r2 >= 0.9 and slope > 0:
            entry["energy_uj_per_inference"] = per_inf_uj
            entry["method"] = ("régression I(N) à cadence fixe, N imposé par "
                               "-DINFER_BATCH_N")
        else:
            entry["energy_uj_per_inference"] = A_MESURER
            entry["na_reason"] = (
                f"régression non concluante (r²={r2:.3f}, pente="
                f"{slope * MA:+.5f} mA/N) : la pente n'est pas séparable du "
                f"bruit du banc sur ces points."
            )
    else:
        entry["energy_uj_per_inference"] = A_MESURER
        entry["na_reason"] = (
            f"{len(xs)} point(s) de N exploitable(s) sur "
            f"{len(entry['points'])} mesuré(s) : une régression exige au "
            f"moins 3 valeurs de N non saturées (chacune est un reflash)."
        )
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=["idle", "cells", "batch"], required=True)
    parser.add_argument("--firmware-build", required=True,
                        help="build RÉELLEMENT flashé, ex. « UART_WFI_IDLE » ou "
                             "« default » — recopié tel quel, jamais deviné")
    parser.add_argument("--batch-n", type=int, default=1,
                        help="valeur de -DINFER_BATCH_N du build flashé (mode batch)")
    parser.add_argument("--model", default="ewc", help="cellule du mode batch")
    parser.add_argument("--encoding", default="fp32")
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=2,
                        help="acquisitions de préchauffage jetées — 2 par défaut : "
                             "S5301 a mesuré que l'établissement dépasse une acquisition")
    parser.add_argument("--dataset", default="monitoring")
    parser.add_argument("--board-port", required=True)
    parser.add_argument("--port", default=None, help="port de la SONDE (auto)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--idle-from", type=Path, default=None,
                        help="idle_reference.json à utiliser comme référence (mode cells)")
    parser.add_argument("--only", default=None,
                        help="mode cells — restreint la campagne à ces cellules "
                             "(ex. « maha_int8 »). INDISPENSABLE pour les cellules à "
                             "firmware dédié (rc.BUILD_SPECIFIC), qui sont exclues du "
                             "balayage par défaut : sans flash du bon build, elles "
                             "seraient écrites sous un nom qui ment sur leur contenu "
                             "(bug du Sprint 52).")
    parser.add_argument("--hw-profile", type=Path,
                        default=ROOT / "configs" / "hw_profile_f439zi.yaml")
    parser.add_argument("--write-hw-profile", action="store_true",
                        help="écrit le repos mesuré dans puissance_watts.veille_uA du "
                             "profil matériel (et actif_mA si des cellules existent). "
                             "N'écrit QUE des valeurs mesurées.")
    args = parser.parse_args(argv)

    calib = lp._load_calibration(args.hw_profile)
    voltage_mv = int(float(calib.get("supply_voltage_v", 3.3)) * 1000)
    n_samples = int(args.rate_hz * (args.window + args.settle + 4.0))
    args.out.mkdir(parents=True, exist_ok=True)

    probe = lp.PowerShield(args.port)
    try:
        probe.take_control()

        if args.mode == "idle":
            ref = measure_idle_reference(probe, args, voltage_mv)
            path = args.out / "idle_reference.json"
            # Un fichier par build : comparer WFI et scrutation exige de garder les deux.
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            existing[args.firmware_build] = ref
            # Gain du WFI : calculé seulement si les DEUX références existent.
            builds = {k: v for k, v in existing.items() if isinstance(v, dict)
                      and "i_idle_established_a" in v}
            if len(builds) > 1 and "UART_WFI_IDLE" in builds:
                base = min((v["i_idle_established_a"] for k, v in builds.items()
                            if k != "UART_WFI_IDLE"), default=None)
                wfi = builds["UART_WFI_IDLE"]["i_idle_established_a"]
                existing["wfi_gain"] = {
                    "i_idle_scrutation_a": base,
                    "i_idle_wfi_a": wfi,
                    "gain_ma": (base - wfi) * MA,
                    "gain_pct": (base - wfi) / base * 100.0 if base else None,
                }
            path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            print(f"[wfi] repos établi ({args.firmware_build}) : "
                  f"{ref['i_idle_established_ma']:.3f} mA → {path.name}")

            if args.write_hw_profile:
                # `actif_mA` n'est écrit que si des cellules ont DÉJÀ été mesurées sur
                # ce build : il ne se déduit pas du repos. Sinon la clé reste `null`.
                actif = None
                cells_a = [json.loads(p.read_text(encoding="utf-8"))
                           for p in iter_cell_files(args.out)]
                cells_a = [c["current_measurement"]["i_mean_a"] for c in cells_a
                           if c.get("firmware_build") == args.firmware_build
                           and "current_measurement" in c]
                if cells_a:
                    actif = float(np.mean(cells_a))
                written = write_hw_profile_currents(
                    args.hw_profile,
                    veille_a=float(ref["i_idle_established_a"]),
                    actif_a=actif,
                    source=(f"experiments/exp_S53_wfi/idle_reference.json "
                            f"({args.firmware_build}, {ref['timestamp'][:10]})"),
                )
                print(f"[wfi] profil matériel mis à jour : {', '.join(written)} "
                      f"→ {args.hw_profile.name}")

        elif args.mode == "cells":
            if not args.idle_from or not args.idle_from.is_file():
                parser.error("--idle-from est requis : la référence de repos ne "
                             "s'invente pas (lancer --mode idle d'abord).")
            refs = json.loads(args.idle_from.read_text(encoding="utf-8"))
            if args.firmware_build not in refs:
                parser.error(f"aucune référence de repos pour le build "
                             f"« {args.firmware_build} » dans {args.idle_from}")
            ref = refs[args.firmware_build]
            i_idle = float(ref["i_idle_established_a"])

            warmups = [lp.warmup(probe, voltage_mv)
                       for _ in range(max(1, args.warmup_repeats))]
            cells_out: dict[str, dict] = {}
            if args.only:
                targets = []
                for token in args.only.split(","):
                    model, _, encoding = token.strip().rpartition("_")
                    if (model, encoding) not in rc.STREAM_MODEL:
                        parser.error(f"cellule inconnue : {token.strip()}")
                    targets.append((model, encoding))
            else:
                # Les cellules à firmware dédié sont exclues du balayage : elles se
                # mesurent après flash du bon build, via « --only ».
                targets = [c for c in rc.STREAM_MODEL if c not in rc.BUILD_SPECIFIC]
                for (m, e), flag in rc.BUILD_SPECIFIC.items():
                    print(f"[wfi] {m}_{e} ignorée : exige un build {flag} "
                          f"(la mesurer par « --only {m}_{e} » après flash)")
            voltage_v = float(ref["tension_v"])
            for model, encoding in targets:
                flag = rc.STREAM_MODEL[(model, encoding)]
                cmd = rc.stream_command(flag, args.dataset, args.board_port,
                                        n_samples, args.rate_hz)
                runs = []
                for _ in range(args.repeats):
                    i_run, voltage_v = rc.measure_current(
                        probe, args.window, voltage_mv, cmd, args.settle)
                    runs.append(i_run)
                cell = rc.build_cell(model, encoding, runs, i_idle, voltage_v,
                                     args.window, args.rate_hz, warmups[-1])
                cell["firmware_build"] = args.firmware_build
                cell["infer_batch_n"] = int(args.batch_n)
                cell["current_measurement"]["i_idle_source"] = str(args.idle_from.name)
                key = f"{model}_{encoding}"
                (args.out / f"{key}.json").write_text(
                    json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
                cells_out[key] = cell
                mes = cell["current_measurement"]
                print(f"[wfi] {key:14s} : {mes['i_mean_a'] * MA:7.3f} "
                      f"± {mes['i_std_a'] * MA:.3f} mA "
                      f"(repos {mes['delta_vs_idle_a'] * MA:+.3f} mA)")

            # Le delta porte sur TOUTES les cellules déjà mesurées sur ce build, pas
            # seulement celles de cette exécution : sans cela un passage « --only »
            # (cellule à firmware dédié) écraserait le delta des sept autres.
            for path in iter_cell_files(args.out):
                key = path.stem
                if key in cells_out:
                    continue
                prev = json.loads(path.read_text(encoding="utf-8"))
                if (prev.get("firmware_build") == args.firmware_build
                        and "current_measurement" in prev):
                    cells_out[key] = prev

            recovery = {
                "firmware_build": args.firmware_build,
                "method": f"delta vs repos {args.firmware_build}",
                "i_idle_a": i_idle,
                "tension_v": voltage_v,
                "window_s": float(args.window),
                "rate_hz": float(args.rate_hz),
                "cells": build_delta_recovery(cells_out, i_idle, voltage_v,
                                              args.window, args.firmware_build),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (args.out / "delta_recovery.json").write_text(
                json.dumps(recovery, indent=2, ensure_ascii=False), encoding="utf-8")
            propagate_delta_into_cells(recovery, args.out)
            n_ok = sum(1 for c in recovery["cells"].values()
                       if c["energy_uj_per_inference"] != A_MESURER)
            print(f"[wfi] delta : {n_ok}/{len(recovery['cells'])} cellules chiffrées "
                  f"→ delta_recovery.json")

        else:  # batch
            path = args.out / "batch_sweep.json"
            report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            key = f"{args.model}_{args.encoding}"
            entry = report.get(key, {"points": [], "model": args.model,
                                     "encoding": args.encoding,
                                     "firmware_build": args.firmware_build,
                                     "rate_hz": float(args.rate_hz),
                                     "window_s": float(args.window)})
            warmups = [lp.warmup(probe, voltage_mv)
                       for _ in range(max(1, args.warmup_repeats))]
            flag = rc.STREAM_MODEL[(args.model, args.encoding)]
            cmd = rc.stream_command(flag, args.dataset, args.board_port,
                                    n_samples, args.rate_hz)
            runs = []
            for _ in range(args.repeats):
                i_run, voltage_v = rc.measure_current(
                    probe, args.window, voltage_mv, cmd, args.settle)
                runs.append(i_run)
            # Flux de contrôle : cadence atteinte, CRC, et prédictions du lot.
            ctrl_dump = args.out / f"control_stream_{key}_N{args.batch_n}.json"
            ctrl_cmd = rc.stream_command(flag, args.dataset, args.board_port,
                                         int(args.rate_hz * 20), args.rate_hz)
            ctrl = control_stream(probe, lp, voltage_mv, ctrl_cmd, ctrl_dump)
            ref_path = args.out / f"batch_ref_preds_{key}.json"
            if int(args.batch_n) == 1:
                # Référence de parité : les prédictions du build N=1, appariées par
                # features (l'indice ne l'est pas, cf. `prediction_parity`).
                ref_path.write_text(json.dumps(
                    {",".join(f"{f:.7g}" for f in s["features"]): s["pred"]
                     for s in ctrl["samples"]}, indent=2), encoding="utf-8")
            reference = (json.loads(ref_path.read_text(encoding="utf-8"))
                         if ref_path.is_file() else {})
            n_common, n_mismatch = prediction_parity(reference, ctrl["samples"])
            achieved = float(ctrl["achieved_rate_hz"])
            saturated = achieved < SATURATION_RATE_TOLERANCE * float(args.rate_hz)

            entry["tension_v"] = float(voltage_v)
            entry["points"] = [p for p in entry["points"] if p["n"] != args.batch_n]
            entry["points"].append({
                "n": int(args.batch_n),
                "i_mean_a": float(np.mean(runs)),
                "i_std_a": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
                "i_runs_a": [float(x) for x in runs],
                "warmup_discarded_a": [float(w) for w in warmups],
                "achieved_rate_hz": achieved,
                "crc_errors": int(ctrl["crc_errors"]),
                "latency_p50_us": ctrl.get("latency_p50_us"),
                "saturated": bool(saturated),
                "pred_parity_vs_n1": (None if int(args.batch_n) == 1 or not n_common
                                      else 1.0 - n_mismatch / n_common),
                "pred_parity_n_common": int(n_common),
            })
            entry["points"].sort(key=lambda p: p["n"])
            entry["saturation_rule"] = (
                f"point écarté de la régression si la cadence atteinte tombe sous "
                f"{SATURATION_RATE_TOLERANCE:.0%} de la cadence demandée "
                f"({args.rate_hz:g} Hz) : la carte ne tient plus la trame et le "
                f"courant moyen porte alors moins d'inférences que « cadence × N »."
            )

            apply_fit(entry, float(voltage_v), float(args.rate_hz))
            report[key] = entry
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            print(f"[wfi] batch N={args.batch_n} {key} : "
                  f"{np.mean(runs) * MA:.3f} mA "
                  f"({len(entry['points'])} point(s)) → {path.name}")
    finally:
        probe.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
