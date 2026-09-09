"""
run_s50_energy.py — Campagne énergie S50 (LPM01A réel) + autonomie (S5002/S5003).

Couche mince au-dessus de `scripts/energy_capture.py` (S33) : elle **réutilise
strictement** la segmentation (`derive_phase_windows`) et l'intégration
(`segment_by_phase` / `integrate_energy_uj`) — aucune réécriture — et ajoute ce dont
la chaîne S33 manquait pour aller jusqu'à l'autonomie :

    1. `phase_durations_s` : durées par phase déduites des mêmes fenêtres PA8 réelles
       (indispensable à `autonomy.average_current_ma`, qui lisait une clé que rien
       n'écrivait — cf. profile_memory.py). Dérivées de la trace, jamais fabriquées.
    2. `energy_uj_per_inference` : µJ de la phase active / N inférences (N ← manifeste).
    3. `by_component` : delta MCU / périphériques si des CSV séparés sont fournis,
       sinon `"na"` honnête (le capteur n'est pas isolable sur ce banc — S5001 §2).

Règle CLAUDE.md — AUCUN CHIFFRE INVENTÉ :
    Tant qu'une cellule n'a pas de CSV LPM01A réel (`csv: null` dans le manifeste),
    tous ses champs énergie valent la constante littérale ``"à mesurer"`` et
    l'autonomie correspondante aussi. Ce script NE FABRIQUE jamais de courant/µJ.

État au 2026-07-24 : STM32CubeMonitor-Power / X-NUCLEO-LPM01A non encore disponibles
→ le manifeste par défaut (`configs/energy_campaign_s50.yaml`) a tous les `csv: null`
→ la campagne produit des placeholders honnêtes, prêts à re-remplir depuis les CSV.

Usage :
    # Placeholders (aucun CSV) ou remplissage réel (CSV renseignés dans le manifeste) :
    python scripts/run_s50_energy.py --manifest configs/energy_campaign_s50.yaml \\
        --out experiments/exp_S50_energy/

    # Une cellule unique avec un CSV réel :
    python scripts/run_s50_energy.py --model ewc --encoding int8 \\
        --csv captures/ewc_int8.csv --n-inference 1000 \\
        --out experiments/exp_S50_energy/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Réutilisation stricte de la chaîne S33 (0 réécriture segmentation/intégration).
from scripts.energy_capture import (
    A_MESURER,
    CAMPAIGN_ENCODINGS,
    CAMPAIGN_MODELS,
    PHASES,
    _load_csv,
    derive_phase_windows,
    integrate_energy_uj,
    segment_by_phase,
)
from src.evaluation import autonomy as autonomy_mod

DEFAULT_MANIFEST = Path("configs/energy_campaign_s50.yaml")
DEFAULT_OUT = Path("experiments/exp_S50_energy")
HW_PROFILE = Path("configs/hw_profile_f439zi.yaml")

# Sources qui attestent d'une mesure réelle sur carte (par opposition à "placeholder") :
# `lpm01a_csv` = trace segmentée par phases, `lpm01a_delta` = protocole delta (S5008).
MEASURED_SOURCES = ("lpm01a_csv", "lpm01a_delta", "lpm01a_current")


def _supply_voltage_v(hw_profile: Path = HW_PROFILE) -> float:
    """Tension d'alim (V) depuis le profil HW (jamais en dur). Défaut 3.3 V."""
    try:
        cfg = yaml.safe_load(hw_profile.read_text(encoding="utf-8"))
        cal = (cfg or {}).get("energy_calibration") or {}
        v = cal.get("supply_voltage_v")
        return float(v) if v is not None else 3.3
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return 3.3


def phase_durations_from_windows(
    windows: list[tuple[str, float, float]],
) -> dict[str, float]:
    """Durées (s) par phase, sommées depuis les fenêtres PA8 réelles.

    Chaque fenêtre ``(nom, t0, t1)`` contribue ``t1 - t0`` à sa phase. Toutes les
    phases de ``PHASES`` sont présentes (0.0 si aucune fenêtre) pour un schéma stable.
    Les durées viennent de la trace mesurée — aucune valeur fabriquée.
    """
    durations = {p: 0.0 for p in PHASES}
    for name, t0, t1 in windows:
        if name in durations:
            durations[name] += max(0.0, float(t1) - float(t0))
    return durations


def _total_uj_from_csv(csv_path: Path) -> float:
    """µJ total intégré sur toute la trace (pour les deltas par composant)."""
    trace = _load_csv(Path(csv_path))
    if trace.get("sync") is None:
        raise ValueError(
            f"CSV {csv_path} sans colonne de synchronisation (sync/pa8/gpio) : "
            "impossible de segmenter/intégrer par phase."
        )
    windows = derive_phase_windows(trace)
    phases_uj = segment_by_phase(Path(csv_path), windows)
    return float(sum(phases_uj.values()))


def _by_component(components: dict, base_dir: Path) -> dict:
    """Décomposition par composant : delta MCU / périphériques, capteur ``"na"``.

    - ``mcu`` : µJ total de la capture « MCU seul » (si CSV fourni).
    - ``periph`` : delta µJ (capture périphériques actifs) − (MCU seul), si les deux CSV.
    - ``sensor`` : ``"na"`` (capteurs simulés par UART, non isolables — S5001 §2).

    Toute cible sans CSV → ``"na"`` + ``na_reason`` (jamais un delta fabriqué).
    """
    comp = components or {}
    out: dict = {"mcu": A_MESURER, "periph": A_MESURER, "sensor": "na"}
    out["sensor_na_reason"] = "capteurs simulés par UART, non isolables sur ce banc (S5001)"

    mcu_csv = comp.get("mcu")
    periph_csv = comp.get("periph")
    mcu_uj = None
    if mcu_csv and mcu_csv != "na":
        mcu_uj = _total_uj_from_csv(base_dir / mcu_csv if not Path(mcu_csv).is_absolute() else Path(mcu_csv))
        out["mcu"] = mcu_uj
    if periph_csv and periph_csv != "na":
        periph_total = _total_uj_from_csv(
            base_dir / periph_csv if not Path(periph_csv).is_absolute() else Path(periph_csv)
        )
        out["periph"] = (periph_total - mcu_uj) if mcu_uj is not None else A_MESURER
    return out


def compute_cell(
    model: str,
    encoding: str,
    csv: str | None,
    n_inference: int | None,
    n_update: int | None,
    tension_v: float,
    components: dict | None = None,
    manifest_dir: Path = Path("."),
) -> dict:
    """Calcule le JSON énergie d'une cellule (modèle × encodage).

    Sans CSV réel → tous les champs énergie valent ``"à mesurer"`` (placeholder honnête).
    Avec CSV → segmentation PA8 réelle → µJ/phase, durées, total, µJ/inférence.
    """
    payload: dict = {
        "model": model,
        "encoding": encoding,
        "tension_v": tension_v,
        "n_inference": n_inference,
        "n_update": n_update,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not csv:
        payload.update(
            source="placeholder",
            phases_uj={p: A_MESURER for p in PHASES},
            phase_durations_s={p: A_MESURER for p in PHASES},
            total_uj=A_MESURER,
            energy_uj_per_inference=A_MESURER,
            # Le marquage PA8 1-bit ne sépare pas inférence et MAJ (S33) : la MAJ
            # exige un protocole delta dédié (inférence-seule vs inférence+MAJ, cf.
            # latences séparées S26). Reste « à mesurer » d'ici là.
            energy_uj_per_update=A_MESURER,
            by_component={"mcu": A_MESURER, "periph": A_MESURER, "sensor": "na",
                          "sensor_na_reason": "capteurs simulés par UART (S5001)"},
        )
        return payload

    csv_path = Path(csv)
    if not csv_path.is_absolute():
        csv_path = manifest_dir / csv_path
    trace = _load_csv(csv_path)
    if trace.get("sync") is None:
        raise ValueError(
            f"CSV {csv_path} sans colonne de synchronisation (sync/pa8/gpio) : "
            "exporter le signal PA8 en parallèle du courant (STM32CubeMonitor-Power)."
        )
    windows = derive_phase_windows(trace)
    phases_uj = segment_by_phase(csv_path, windows)
    durations = phase_durations_from_windows(windows)
    total_uj = float(sum(phases_uj.values()))

    if n_inference and n_inference > 0:
        per_inf: float | str = float(phases_uj.get("inference", 0.0)) / n_inference
    else:
        per_inf = A_MESURER

    payload.update(
        source="lpm01a_csv",
        phases_uj={p: float(phases_uj.get(p, 0.0)) for p in PHASES},
        phase_durations_s=durations,
        total_uj=total_uj,
        energy_uj_per_inference=per_inf,
        energy_uj_per_update=A_MESURER,  # 1-bit PA8 : MAJ non séparée (S33)
        by_component=_by_component(components or {}, manifest_dir),
    )
    return payload


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def write_summary(out_dir: Path, cells: dict[str, dict]) -> None:
    """Agrège les cellules en `summary.json` (delta / ratio INT8 vs FP32)."""
    per_model: dict = {}
    for model in CAMPAIGN_MODELS:
        entry: dict = {}
        for enc in CAMPAIGN_ENCODINGS:
            cell = cells.get(f"{model}_{enc}")
            if cell:
                mes = cell.get("current_measurement") or {}
                entry[enc] = {
                    "total_uj": cell.get("total_uj"),
                    "energy_uj_per_inference": cell.get("energy_uj_per_inference"),
                    # Grandeur réellement mesurée par la campagne S5008.
                    "i_mean_ma": (mes["i_mean_a"] * 1e3
                                  if _is_num(mes.get("i_mean_a")) else A_MESURER),
                    "i_std_ma": (mes["i_std_a"] * 1e3
                                 if _is_num(mes.get("i_std_a")) else A_MESURER),
                }
        fp32 = entry.get("fp32", {}).get("energy_uj_per_inference")
        int8 = entry.get("int8", {}).get("energy_uj_per_inference")
        num = _is_num(fp32) and _is_num(int8)
        entry["delta_int8_vs_fp32_uj"] = (int8 - fp32) if num else A_MESURER
        entry["ratio_int8_fp32"] = (int8 / fp32) if num and fp32 else A_MESURER

        # Comparaison INT8 vs FP32 sur le COURANT mesuré : c'est elle qui répond
        # à la question Gap 3 tant que les µJ/inférence restent inaccessibles.
        i_fp32 = entry.get("fp32", {}).get("i_mean_ma")
        i_int8 = entry.get("int8", {}).get("i_mean_ma")
        num_i = _is_num(i_fp32) and _is_num(i_int8)
        entry["delta_int8_vs_fp32_ma"] = (i_int8 - i_fp32) if num_i else A_MESURER
        entry["ratio_i_int8_fp32"] = (i_int8 / i_fp32) if num_i and i_fp32 else A_MESURER
        per_model[model] = entry

    summary = {
        "description": "Campagne énergie S50 — µJ/inférence par modèle × encodage (LPM01A réel).",
        "phases": list(PHASES),
        "per_model": per_model,
        "gap3_note": (
            "Gap 3 : l'INT8 divise la RAM par ~4 (S28/S49) sans accélérer la latence "
            "(FPU Cortex-M4, pas de NPU — paradoxe S29). Question : réduit-il malgré "
            "tout la consommation ? Réponse mesurée (S5008) par le COURANT MOYEN à "
            "cadence imposée — `delta_int8_vs_fp32_ma` — et non par les µJ/inférence, "
            "qui restent inaccessibles sur ce firmware (référence de scrutation "
            "active, cf. `energy_na_reason` des cellules)."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _i_moy_from_current(cell: dict):
    """I_moy (mA) depuis une cellule à courant moyen mesuré (S5008), ou placeholder.

    C'est la voie la plus directe : `run_s50_board_current.py` mesure déjà le
    courant moyen de la carte sous charge d'inférence. Aucun modèle de cycle
    n'est nécessaire — la valeur EST le courant moyen du régime mesuré.
    """
    mes = cell.get("current_measurement") or {}
    i_mean = mes.get("i_mean_a")
    return float(i_mean) * 1e3 if _is_num(i_mean) and i_mean > 0 else A_MESURER


def _i_moy_from_delta(cell: dict, periode_s, tension_v: float):
    """I_moy (mA) depuis une cellule du protocole delta, ou ``"à mesurer"``.

    Exige les trois grandeurs mesurées/déclarées : courant au repos, énergie
    marginale par inférence, période d'inférence du scénario. Toute absence
    (cellule placeholder, delta non concluant, manifeste sans `duty_cycle`)
    propage le placeholder — jamais de valeur de substitution.
    """
    delta = cell.get("delta_measurement") or {}
    i_idle = delta.get("i_idle_a")
    per_inf = cell.get("energy_uj_per_inference")
    if not (_is_num(i_idle) and _is_num(per_inf) and _is_num(periode_s)):
        return A_MESURER
    try:
        return autonomy_mod.average_current_ma_from_delta(
            i_idle, per_inf, periode_s, tension_v
        )
    except ValueError:
        return A_MESURER


def write_autonomy(out_dir: Path, cells: dict[str, dict], duty_cycle: dict) -> None:
    """Autonomie recalculée (S5003) via `autonomy.py` depuis les µJ/durées mesurés.

    Deux voies, dans cet ordre de préférence :
      1. profil par phase → `average_current_ma` (I_moy = Σ(E/V)/T_cycle) ;
      2. protocole delta (S5008) → `average_current_ma_from_delta`, seule voie
         disponible quand la sonde ne peut pas segmenter les phases.
    Placeholder « à mesurer » si aucune des deux n'est chiffrable.
    """
    capacites = autonomy_mod.load_battery_capacities(HW_PROFILE)
    periode_s = duty_cycle.get("inference_period_s")
    per_model: dict = {}
    for name, cell in cells.items():
        phases_uj = cell.get("phases_uj", {})
        durations = cell.get("phase_durations_s", {})
        tension_v = float(cell.get("tension_v", 3.3))
        measured = all(_is_num(phases_uj.get(p)) for p in phases_uj) and all(
            _is_num(durations.get(p)) for p in durations
        ) and bool(durations)
        methode = A_MESURER
        if measured:
            i_moy = autonomy_mod.average_current_ma(phases_uj, durations, tension_v)
            methode = "profil par phase"
        else:
            i_moy = _i_moy_from_current(cell)
            if _is_num(i_moy):
                methode = (
                    f"courant moyen mesuré sous charge "
                    f"({cell['current_measurement'].get('rate_hz')} Hz)"
                )
            else:
                i_moy = _i_moy_from_delta(cell, periode_s, tension_v)
                if _is_num(i_moy):
                    methode = "protocole delta (I_repos + charge marginale / période)"
        if _is_num(i_moy):
            sweep = {str(c): h for c, h in
                     autonomy_mod.sweep_capacities(i_moy, capacites).items()}
        else:
            i_moy = A_MESURER
            sweep = {str(c): A_MESURER for c in capacites}
        entry = {
            "i_moy_ma": i_moy,
            "methode": methode,
            "autonomy_h_by_mah": sweep,
            "duty_cycle": duty_cycle,
        }
        # Le courant mesuré l'est SOUS un régime précis. Afficher à côté de lui le
        # `duty_cycle` du manifeste (une hypothèse d'usage qui n'a pas produit cette
        # valeur) ferait lire l'autonomie comme celle d'un scénario duty-cyclé.
        mes = cell.get("current_measurement") or {}
        if _is_num(mes.get("i_mean_a")):
            entry["duty_cycle"] = None
            entry["regime_mesure"] = {
                "rate_hz": mes.get("rate_hz"),
                "note": (
                    "autonomie du régime RÉELLEMENT mesuré (flux continu à cette "
                    "cadence, carte jamais endormie) — ce n'est PAS l'autonomie d'un "
                    "déploiement duty-cyclé, qui exige une mise en sommeil du firmware"
                ),
            }
        per_model[name] = entry

    report = {
        "description": "Autonomie estimée S50 — I_moy + balayage capacités (depuis µJ réels).",
        "capacites_mah": capacites,
        "duty_cycle": duty_cycle,
        "per_model": per_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "autonomy.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_measured_cell(out_dir: Path, key: str) -> dict | None:
    """Relit une cellule déjà **mesurée** sur carte, si elle existe.

    Le protocole delta (`run_s50_energy_delta.py`) écrit ses cellules dans le même
    répertoire et sous les mêmes noms que cette campagne. Sans cette relecture,
    rejouer le manifeste écraserait des mesures réelles par des placeholders —
    une perte de données silencieuse.

    Parameters
    ----------
    out_dir : Path
        Répertoire de campagne.
    key : str
        Clé de cellule (``{model}_{encoding}``).

    Returns
    -------
    dict | None
        La cellule mesurée, ou ``None`` si absente/non mesurée/illisible.
    """
    path = Path(out_dir) / f"{key}.json"
    if not path.is_file():
        return None
    try:
        cell = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return cell if cell.get("source") in MEASURED_SOURCES else None


def run_from_manifest(manifest_path: Path, out_dir: Path) -> None:
    """Campagne complète depuis le manifeste (placeholders ou mesures réelles)."""
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    manifest_dir = Path(manifest_path).parent
    tension_v = _supply_voltage_v()
    duty_cycle = manifest.get("duty_cycle", {})
    components = manifest.get("components", {})
    cells_cfg = manifest.get("cells", {})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Campagne énergie S50 → {out_dir}/ (tension {tension_v} V)")

    cells: dict[str, dict] = {}
    for model in CAMPAIGN_MODELS:
        for enc in CAMPAIGN_ENCODINGS:
            key = f"{model}_{enc}"
            # Une cellule déjà mesurée sur carte fait autorité : on la conserve.
            existing = load_measured_cell(out_dir, key)
            if existing is not None:
                cells[key] = existing
                print(f"  = {key}.json — conservé (mesuré, source={existing['source']})")
                continue

            cfg = cells_cfg.get(key, {}) or {}
            cell = compute_cell(
                model=model,
                encoding=enc,
                csv=cfg.get("csv"),
                n_inference=cfg.get("n_inference"),
                n_update=cfg.get("n_update"),
                tension_v=tension_v,
                components=components,
                manifest_dir=manifest_dir,
            )
            cells[key] = cell
            (out_dir / f"{key}.json").write_text(
                json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            status = "mesuré" if cell["source"] in MEASURED_SOURCES else f"placeholder ({A_MESURER})"
            print(f"  ✔ {key}.json — {status}")

    write_summary(out_dir, cells)
    write_autonomy(out_dir, cells, duty_cycle)
    print(f"  ✔ summary.json + autonomy.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Campagne énergie S50 (LPM01A) → µJ/inférence + autonomie."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Manifeste de campagne (défaut : configs/energy_campaign_s50.yaml).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Répertoire de sortie (défaut : experiments/exp_S50_energy).")
    # Mode cellule unique (contourne le manifeste).
    parser.add_argument("--model", choices=CAMPAIGN_MODELS, help="Modèle (cellule unique).")
    parser.add_argument("--encoding", choices=CAMPAIGN_ENCODINGS, help="Encodage (cellule unique).")
    parser.add_argument("--csv", type=Path, default=None, help="CSV LPM01A (cellule unique).")
    parser.add_argument("--n-inference", type=int, default=None, help="N inférences (µJ/inférence).")
    parser.add_argument("--n-update", type=int, default=None, help="N MAJ CL.")
    args = parser.parse_args()

    if args.model and args.encoding:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cell = compute_cell(
            model=args.model, encoding=args.encoding,
            csv=str(args.csv) if args.csv else None,
            n_inference=args.n_inference, n_update=args.n_update,
            tension_v=_supply_voltage_v(),
        )
        (out_dir / f"{args.model}_{args.encoding}.json").write_text(
            json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  ✔ {args.model}_{args.encoding}.json — source {cell['source']}")
        return

    run_from_manifest(args.manifest, args.out)


if __name__ == "__main__":
    main()
