"""
refresh_s50_na_reason.py — Recalcul du verdict S5301 et mise à jour des raisons N/A S50.

POURQUOI CE SCRIPT EXISTE :

Les 8 cellules `experiments/exp_S50_energy/*.json` portent une `energy_na_reason` qui
affirme que la cause de l'écart repos↔charge « n'est pas établie ». Le protocole
contre-balancé (S5301) a tranché : cette phrase doit refléter le verdict **mesuré**,
sinon le dépôt continue de publier une incertitude qui n'existe plus.

Deux opérations, toutes deux en lecture/écriture ciblée :

    1. **Recalcul du verdict** à partir des courants bruts de
       `counterbalance.json` — la règle vit dans `src/evaluation/counterbalance.py`,
       donc le JSON n'est jamais la source de vérité du verdict : il est recalculable.
       Cela permet de faire évoluer la règle sans refaire une session de banc.
    2. **Mise à jour de `energy_na_reason`** dans les cellules S50, en ne touchant QUE
       ce champ. Aucune mesure n'est modifiée : ni les courants, ni `i_idle_a`, ni
       `delta_vs_idle_a`, dont d'autres outils dépendent.

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ : le script refuse d'écrire si le rapport de
contre-balancement n'existe pas. Sans session mesurée, pas de mise à jour.

Usage :
    python scripts/refresh_s50_na_reason.py --dry-run
    python scripts/refresh_s50_na_reason.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load("counterbalance", ROOT / "src" / "evaluation" / "counterbalance.py")
ec = _load("energy_capture", ROOT / "scripts" / "energy_capture.py")
A_MESURER = ec.A_MESURER

DEFAULT_REPORT = ROOT / "experiments" / "exp_S53_counterbalance" / "counterbalance.json"
DEFAULT_CELLS = ROOT / "experiments" / "exp_S50_energy"

#: Raison N/A par verdict. Chacune cite le CONSTAT de la session contre-balancée, pas
#: une absence de banc — c'est la règle d'honnêteté du sprint.
REASON_BY_VERDICT = {
    cb.VERDICT_ARTEFACT: (
        "mesuré, cause établie au Sprint 53 : l'écart « repos plus consommateur que la "
        "charge » relevé au Sprint 50 est un ARTEFACT D'ORDRE, pas un effet de charge. "
        "Le protocole contre-balancé (repos intercalé entre chaque cellule, session "
        "unique, S5301) montre que le repos décroît sur les premières acquisitions puis "
        "se stabilise nettement SOUS les cellules : le surcoût de charge redevient "
        "positif et ordonné comme les latences. La référence au repos du Sprint 50 était "
        "prise en tête de session, donc encore dans la queue d'établissement — l'ordre y "
        "était confondu avec la condition. Le MÉCANISME est identifié depuis (S5301b, "
        "`experiments/exp_S53_counterbalance/idle_states_power_cycle.json`) : la carte "
        "présente deux niveaux de repos stables, et elle bascule du haut vers le bas au "
        "PREMIER flux reçu depuis la mise sous tension — ni le temps, ni l'ouverture du "
        "port, ni l'impulsion DTR ne l'y font passer, seules de vraies trames. La "
        "référence du Sprint 50, prise avant tout flux, était donc mesurée sur le niveau "
        "haut tandis que les cellules l'étaient sur le niveau bas. Le courant moyen de "
        "cette cellule reste "
        "valide (toutes les cellules partagent la même référence et la même cadence) ; "
        "seule la soustraction repos↔charge de cette campagne ne l'est pas. Les µJ par "
        "inférence sont repris par le protocole delta sur référence établie (S5302) et "
        "par régression de cadence (S5304)."
    ),
    cb.VERDICT_EFFET_REEL: (
        "mesuré, non séparable : le protocole contre-balancé (S5301) écarte l'artefact "
        "d'ordre — le repos reste plat et au-dessus des cellules quel que soit son rang "
        "dans la session. L'écart n'est donc pas une dérive d'établissement, et sa cause "
        "reste non établie (le firmware attend la trame UART par scrutation active : le "
        "« repos » n'est pas inactif). L'énergie marginale par inférence ressort négative "
        "et n'a pas de sens face à cette référence. Reste exploitable : le courant moyen "
        "par cellule et sa comparaison inter-cellules à cadence imposée."
    ),
    cb.VERDICT_DERIVE: (
        "mesuré, référence encore instable : le protocole contre-balancé (S5301) montre "
        "que le repos dérive significativement sur la session sans rejoindre les "
        "cellules. La référence du Sprint 50, prise en tête de session, était donc "
        "partiellement établie. Toute campagne ultérieure doit allonger le préchauffage "
        "(--warmup-repeats) avant de produire un delta. Reste exploitable : le courant "
        "moyen par cellule à cadence imposée."
    ),
}


def recompute_verdict(report: dict) -> tuple[str, str]:
    """Rejoue la règle de décision sur les courants bruts du rapport."""
    idle = report.get("idle_by_rank") or []
    cells = list((report.get("cells") or {}).values())
    if not idle or not cells:
        raise ValueError(
            "rapport de contre-balancement sans courants exploitables : "
            "relancer une session mesurée (--interleave-idle)."
        )
    etabli = cb.established_regime(idle)
    dispersion = cb.bench_dispersion_a([v for _, v in etabli])
    return cb.counterbalance_verdict(idle, cells, dispersion)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="counterbalance.json produit par --interleave-idle")
    parser.add_argument("--cells-dir", type=Path, default=DEFAULT_CELLS,
                        help="répertoire des cellules dont la raison N/A est à corriger")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche ce qui serait écrit, sans rien modifier")
    args = parser.parse_args(argv)

    if not args.report.is_file():
        parser.error(
            f"{args.report} absent : le verdict ne peut pas être inventé, "
            f"lancer d'abord une session contre-balancée (--interleave-idle)."
        )

    report = json.loads(args.report.read_text(encoding="utf-8"))
    verdict, rationale = recompute_verdict(report)
    print(f"[na_reason] verdict recalculé : {verdict}")
    print(f"[na_reason]   {rationale}")

    # Le rapport porte le verdict de la règle COURANTE, pas celui figé à la mesure.
    etabli = cb.established_regime(report["idle_by_rank"])
    report["verdict"] = verdict
    report["verdict_rationale"] = rationale
    report["idle_established"] = [{"session_index": x, "i_a": v} for x, v in etabli]
    report["i_idle_established_a"] = float(sum(v for _, v in etabli) / len(etabli))
    report["bench_dispersion_a"] = cb.bench_dispersion_a([v for _, v in etabli])
    if not args.dry_run:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[na_reason] {args.report.name} resynchronisé sur la règle courante")

    reason = REASON_BY_VERDICT[verdict]
    touched = 0
    for path in sorted(args.cells_dir.glob("*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        if ("energy_na_reason" not in cell
                or cell.get("energy_uj_per_inference") != A_MESURER):
            continue
        if cell["energy_na_reason"] == reason:
            continue
        cell["energy_na_reason"] = reason
        # Traçabilité : d'où vient la correction, pour qu'elle ne soit pas orpheline.
        cell["energy_na_reason_source"] = "S5301 — " + args.report.name
        if not args.dry_run:
            path.write_text(json.dumps(cell, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        touched += 1
        print(f"[na_reason] {'(dry-run) ' if args.dry_run else ''}{path.name} mis à jour")

    print(f"[na_reason] {touched} cellule(s) concernée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
