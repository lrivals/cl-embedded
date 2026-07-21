#!/usr/bin/env python3
"""aggregate_sprint45.py — S4504 : agrégat unique du portage board des détecteurs de drift.

Fusionne les sorties dispersées du Sprint 45 (board réelle S4503/S4504, parité S4503) et le
**proxy PC S44** dans un seul ``experiments/exp_S45_summary.json`` indexé
``[dataset][detector][platform]`` avec ``platform ∈ {board, pc_proxy}``.

**Lecture seule** : aucune métrique recalculée ; on reprend les valeurs déjà stockées (à l'image
des ``exp_S3{2,5,6,8}_*_summary.json``). Les champs absents → ``null``.

Message : *combien coûte réellement la détection de drift sur MCU, par méthode* — latence DWT
mesurée (board) vs proxy Python (S44), ``.bss`` firmware total vs état algorithmique pur, parité
board↔PC, et vérification des gaps (Gap 2 latence < 100 ms, Gap 3 RAM dans le budget 256 Ko).

**Honnêteté** : distinction stricte *mesuré-board* (ici) vs *proxy-PC* (S44) ; ``null`` +
``na_reason`` partout où non flashé / sans vérité-terrain ponctuelle. Aucun chiffre inventé.

Usage :
    python scripts/aggregate_sprint45.py            # → experiments/exp_S45_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS = Path("experiments")

# Grille des détecteurs *portés* (S4501) × datasets de drift (S43).
DETECTORS = ("page_hinkley", "ddm", "psi")
DATASETS = ("gas_sensor_drift", "hydraulic", "synthetic", "electricity")

GAP2_LATENCY_US = 100_000
RAM_BUDGET_BYTES = 256 * 1024  # NUCLEO-F439ZI : 192 Ko SRAM + 64 Ko CCM

# ``.bss`` du build par défaut (sans ``-DDRIFT_DETECT``) — invariant, condition de recevabilité
# (S4502 : « `.bss` défaut invariant 105 036 B »). Les deltas firmware par méthode sont mesurés
# à k fixe dans S4502 ; on les expose comme constantes documentées (NE PAS les recalculer depuis
# ``bss_bytes`` mesuré, qui inclut la tête EWC dont la taille varie avec k).
BSS_DEFAULT = 105_036
BSS_DELTA_BY_METHOD = {  # source : docs/sprints/sprint_45/S4502_firmware_detecteurs.md
    "page_hinkley": 36,
    "ddm": 40,
    "psi": 132,
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _first(d: dict | None, keys: list[str]):
    """Premier champ présent (non-None) parmi `keys`, sinon None (tolère la dérive de schéma)."""
    d = d or {}
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _delta(a, b):
    return (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None


def _board_cell(board: dict | None, parity: dict | None) -> dict:
    """Cellule *mesurée-board* (S4503/S4504) fusionnée avec sa parité (S4503)."""
    b = board or {}
    p = parity or {}
    dm = b.get("drift_metrics") or {}
    lat_p99 = b.get("latency_us_p99")
    bss = b.get("bss_bytes")
    return {
        "measured": bool(b) and b.get("platform") == "nucleo_f439zi"
        and b.get("latency_us_p50") is not None,
        "n_samples": b.get("n_samples"),
        "n_features": b.get("n_features"),
        "family": b.get("family"),
        "requires_label": b.get("requires_label"),
        "latency_us_p50": b.get("latency_us_p50"),
        "latency_us_p99": lat_p99,
        "mean_latency_us": b.get("mean_latency_us"),
        "bss_bytes": bss,
        "f1": dm.get("f1") if isinstance(dm, dict) else None,
        "metric_value": b.get("metric_value"),
        "na_reason": b.get("na_reason"),
        "verdict_counts": b.get("verdict_counts_board"),
        "verdict_parity": p.get("verdict_parity"),
        "pred_parity": p.get("pred_parity"),
        "verdict_mismatch_count": p.get("verdict_mismatch_count"),
        "gap2_ok": (lat_p99 < GAP2_LATENCY_US) if isinstance(lat_p99, (int, float)) else None,
        "gap3_ram_ok": (bss < RAM_BUDGET_BYTES) if isinstance(bss, (int, float)) else None,
    }


def _pc_proxy_cell(pc: dict | None) -> dict:
    """Cellule *proxy-PC* (S44) : latence Python + état algorithmique (jamais board)."""
    pc = pc or {}
    cost = pc.get("cost") or {}
    dm = pc.get("drift_metrics") or {}
    return {
        "measured": bool(pc),
        "is_proxy": True,
        "latency_us_per_update": cost.get("latency_us_per_update"),
        "state_bytes_algo": cost.get("state_bytes"),
        "ram_peak_bytes": cost.get("ram_peak_bytes"),
        "f1": dm.get("f1") if isinstance(dm, dict) else None,
        "viabilite_mcu": pc.get("viabilite_mcu"),
        "na_reason": pc.get("na_reason"),
    }


def build_cell(detector: str, dataset: str) -> dict:
    """Fusionne board (+ parité) et proxy PC pour une cellule (détecteur, dataset)."""
    board = _load(EXPERIMENTS / f"exp_S45_board_{detector}_{dataset}" / "results.json")
    parity = _load(EXPERIMENTS / f"exp_S45_parity_{detector}_{dataset}.json")
    pc = _load(EXPERIMENTS / f"exp_S44_PC_{detector}_{dataset}" / "results.json")

    cell = {
        "board": _board_cell(board, parity),
        "pc_proxy": _pc_proxy_cell(pc),
        "bss_delta_vs_default": BSS_DELTA_BY_METHOD.get(detector),
    }

    b, q = cell["board"], cell["pc_proxy"]
    # Écart latence mesuré-board ↔ proxy-PC (paradoxe latence FPU Cortex-M4, cf. Sprint 29).
    cell["latency_board_vs_proxy_us"] = _delta(b["latency_us_p50"], q["latency_us_per_update"])

    if b["measured"]:
        cell["na_reason"] = None
    elif b["na_reason"]:                          # raison propre au board (ex. overflow SRAM mesuré)
        cell["na_reason"] = b["na_reason"]
    elif dataset == "electricity":
        cell["na_reason"] = "pas de vérité-terrain ponctuelle"
    else:
        cell["na_reason"] = "non flashé (runbook — board 1 colonne mesurée, cf. S4503)"
    return cell


def main() -> None:
    results: dict = {}
    for ds in DATASETS:
        results[ds] = {det: build_cell(det, ds) for det in DETECTORS}

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sprint": 45,
        "detectors": list(DETECTORS),
        "datasets": list(DATASETS),
        "bss_default": BSS_DEFAULT,
        "bss_delta_by_method": BSS_DELTA_BY_METHOD,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "gap2_latency_us": GAP2_LATENCY_US,
        "results": results,
    }
    out = EXPERIMENTS / "exp_S45_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"→ {out}")
    for ds in DATASETS:
        print(f"\n  ── {ds} ──")
        for det in DETECTORS:
            c = results[ds][det]
            b, q = c["board"], c["pc_proxy"]
            if b["measured"]:
                print(f"    {det:14s} board: lat_p50={b['latency_us_p50']}µs "
                      f"p99={b['latency_us_p99']}µs bss={b['bss_bytes']}B "
                      f"vparity={b['verdict_parity']} f1={b['f1']} "
                      f"| gap2={b['gap2_ok']} gap3={b['gap3_ram_ok']} "
                      f"| proxyPC lat={q['latency_us_per_update']}µs "
                      f"state={q['state_bytes_algo']}B (Δlat={c['latency_board_vs_proxy_us']})")
            else:
                print(f"    {det:14s} board: — ({c['na_reason']}) "
                      f"| proxyPC lat={q['latency_us_per_update']}µs "
                      f"state={q['state_bytes_algo']}B")


if __name__ == "__main__":
    main()
