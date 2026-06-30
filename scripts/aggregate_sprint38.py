#!/usr/bin/env python3
"""aggregate_sprint38.py — S3807 : agrégat unique + table d'économie (livrable central).

Fusionne les sorties dispersées (PC S3802, board S3804/S3805, parité S3806) en **un seul**
``experiments/exp_S38_summary.json`` indexé ``[dataset][init_mode][policy][platform]`` avec
``platform ∈ {pc, board}``.

**Lecture seule** : aucune métrique recalculée ; on reprend les valeurs déjà stockées (à l'image
des ``exp_S3{2,5,6}_*_summary.json``). Les champs absents → ``null``.

Le bloc **``economy_table``** (par ``dataset × init_mode``) donne les deltas **vs ``always`` (P1)**
côté board → lecture directe « P3 économise X µs et Y % de MAJ vs P1, au coût de Z B de RAM et ΔF1 ».

Usage :
    python scripts/aggregate_sprint38.py            # → experiments/exp_S38_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS = Path("experiments")
DATASETS = ("monitoring", "pronostia")
INIT_MODES = ("pretrained", "scratch")
POLICIES = ("frozen", "always", "gated_truelabel", "gated_pseudolabel")
GAP2_LATENCY_US = 100_000


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _first(d: dict | None, keys: list[str]):
    """Premier champ présent (non-None) parmi `keys`, sinon None."""
    d = d or {}
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _delta(a, b):
    return (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None


def _pc_cell(pc: dict | None) -> dict:
    pc = pc or {}
    return {
        "acc_final": pc.get("acc_final"),
        "f1_faulty": pc.get("f1_faulty"),
        "af": pc.get("af"),
        "n_updates": pc.get("n_updates"),
        "update_rate": pc.get("update_rate"),
        "mean_latency_us": None,   # PC : pas de latence par-échantillon streamée
        "inference_latency_us": (round(pc["inference_latency_ms"] * 1e3, 3)
                                 if pc.get("inference_latency_ms") is not None else None),
        "gate_overhead_us": None,
        "bss_bytes": None, "bss_delta_vs_default": None,
        "prediction_parity_rate": None, "verdict_parity_rate": None,
        "gap2_ok": True,   # PC non contraint matériellement
    }


def _board_cell(board: dict | None, parity: dict | None) -> dict:
    board = board or {}
    parity = parity or {}
    lat_p50 = board.get("latency_us_p50")
    # update_rate : présent pour gated ; pour frozen/always on le dérive de n_updates/n_streamed.
    update_rate = board.get("update_rate")
    if update_rate is None and board.get("n_updates") is not None and board.get("n_streamed"):
        update_rate = board["n_updates"] / board["n_streamed"]
    return {
        "acc_final": board.get("online_accuracy"),
        "f1_faulty": board.get("f1_faulty"),
        "af": board.get("af"),
        "n_updates": board.get("n_updates"),
        "update_rate": update_rate,
        "mean_latency_us": _first(board, ["mean_latency_us", "latency_us_mean"]),
        "inference_latency_us": _first(board, ["inference_latency_us", "latency_us_p50"]),
        "gate_overhead_us": board.get("gate_overhead_us"),
        "bss_bytes": board.get("bss_bytes"),
        "bss_delta_vs_default": board.get("bss_delta_vs_default"),
        "prediction_parity_rate": _first(parity, ["prediction_parity_rate"])
                                  if parity else board.get("parity_rate"),
        "verdict_parity_rate": parity.get("verdict_parity_rate"),
        "gap2_ok": board.get("gap2_latency_compliant",
                             (lat_p50 is not None and lat_p50 < GAP2_LATENCY_US)),
    }


def build_cell(dataset: str, init: str, policy: str) -> dict:
    pc = _load(EXPERIMENTS / f"exp_S38_PC_{policy}_{dataset}_{init}" / "results.json")
    board = _load(EXPERIMENTS / f"exp_S38_board_{policy}_{dataset}_{init}" / "results.json")
    parity = _load(EXPERIMENTS / f"exp_S38_parity_{policy}_{dataset}_{init}.json")
    return {"pc": _pc_cell(pc), "board": _board_cell(board, parity)}


def _economy_table(cells: dict) -> dict:
    """Deltas board vs `always` (P1) pour chaque politique (le livrable d'arbitrage)."""
    ref = cells.get("always", {}).get("board", {})
    ref_lat = ref.get("mean_latency_us")
    ref_f1 = ref.get("f1_faulty")
    ref_rate = ref.get("update_rate")
    table = {}
    for pol in POLICIES:
        b = cells.get(pol, {}).get("board", {})
        lat = b.get("mean_latency_us")
        f1 = b.get("f1_faulty")
        rate = b.get("update_rate")
        table[pol] = {
            "latency_saved_us": _delta(ref_lat, lat),                 # gain de latence vs P1
            "ram_added_bytes": b.get("bss_delta_vs_default"),         # coût RAM du gate
            "f1_lost": _delta(ref_f1, f1),                            # précision perdue vs P1
            "updates_saved_pct": (1.0 - rate / ref_rate
                                  if (isinstance(rate, (int, float))
                                      and isinstance(ref_rate, (int, float)) and ref_rate)
                                  else None),
        }
    return table


def main() -> None:
    results: dict = {}
    for ds in DATASETS:
        results[ds] = {}
        for init in INIT_MODES:
            cells = {pol: build_cell(ds, init, pol) for pol in POLICIES}
            cells["economy_table"] = _economy_table(cells)
            results[ds][init] = cells

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "ewc",
        "sprint": 38,
        "datasets": list(DATASETS),
        "init_modes": list(INIT_MODES),
        "policies": list(POLICIES),
        "results": results,
    }
    out = EXPERIMENTS / "exp_S38_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"→ {out}")
    for ds in DATASETS:
        for init in INIT_MODES:
            print(f"\n  ── {ds} / {init} ──")
            for pol in POLICIES:
                b = results[ds][init][pol]["board"]
                e = results[ds][init]["economy_table"][pol]
                print(f"    {pol:18s} board F1={b['f1_faulty']} rate={b['update_rate']} "
                      f"lat={b['mean_latency_us']}µs vparity={b['verdict_parity_rate']} "
                      f"| eco: lat_saved={e['latency_saved_us']} f1_lost={e['f1_lost']} "
                      f"upd_saved={e['updates_saved_pct']} ram+={e['ram_added_bytes']}")


if __name__ == "__main__":
    main()
