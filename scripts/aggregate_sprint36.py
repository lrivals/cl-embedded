#!/usr/bin/env python3
"""aggregate_sprint36.py — S3606 : agrégat unique de tous les métriques Sprint 36.

Fusionne les sorties dispersées (PC S3602, board gelé S3603, board online S3604, parité
S3605, et — rework S3612 — board INT8 frozen/online S3611) en
``experiments/exp_S36_summary.json`` indexé ``[dataset][condition][platform]`` avec
``platform ∈ {pc, board_frozen, board_online, board_frozen_int8, board_online_int8}``.

**Lecture seule** : aucune métrique n'est recalculée ; on reprend les valeurs déjà stockées
(à l'image des ``exp_S3{2,5}_*_sweep_summary.json``). Les champs absents → ``null``.

Usage :
    python scripts/aggregate_sprint36.py            # → experiments/exp_S36_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS = Path("experiments")
DATASETS = ["pronostia", "monitoring"]
CONDITIONS = ["5feat", "all"]


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _pick(d: dict | None, keys: list[str]) -> dict:
    """Sous-dict {k: d[k] or None} robuste à l'absence du fichier/clé."""
    d = d or {}
    return {k: d.get(k) for k in keys}


def _abs_delta(a, b):
    return abs(a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None


def _int8_cell(int8_d: dict | None, fp32_d: dict | None) -> dict:
    """Métriques d'une passe board INT8 + comparatifs vs la passe FP32 de même condition."""
    m = _pick(int8_d, ["online_accuracy", "f1_faulty", "metric_value",
                       "latency_us_p50", "latency_us_p99", "bss_bytes",
                       "ram_weights_fp32_bytes", "ram_weights_int8_bytes",
                       "ram_ratio_fp32_over_int8", "agreement_int8_vs_fp32",
                       "latency_inference_only_us_p50", "latency_update_overhead_us_p50"])
    fp32_d = fp32_d or {}
    lat_i, lat_f = m["latency_us_p50"], fp32_d.get("latency_us_p50")
    m["latency_ratio_int8_over_fp32"] = (
        round(lat_i / lat_f, 3) if (isinstance(lat_i, (int, float))
                                    and isinstance(lat_f, (int, float)) and lat_f) else None)
    m["delta_metric_int8_vs_fp32"] = _abs_delta(m["f1_faulty"], fp32_d.get("f1_faulty"))
    # Gap 3 : objectif RAM des poids divisée par ~4 (INT8 vs FP32).
    r = m["ram_ratio_fp32_over_int8"]
    m["gap3_ram_ok"] = bool(r is not None and r >= 3.5)
    return m


def build_cell(dataset: str, condition: str) -> dict:
    pc = _load(EXPERIMENTS / f"exp_S36_PC_{condition}_ewc_{dataset}" / "results.json")
    frozen = _load(EXPERIMENTS / f"exp_S36_board_frozen_{condition}_ewc_{dataset}" / "results.json")
    online = _load(EXPERIMENTS / f"exp_S36_board_online_{condition}_ewc_{dataset}" / "results.json")
    frozen_i8 = _load(EXPERIMENTS / f"exp_S36_board_frozen_int8_{condition}_ewc_{dataset}" / "results.json")
    online_i8 = _load(EXPERIMENTS / f"exp_S36_board_online_int8_{condition}_ewc_{dataset}" / "results.json")

    pc_m = _pick(pc, ["acc_final", "aa", "af", "bwt", "f1_faulty", "f1_macro",
                      "roc_auc", "ram_peak_bytes", "inference_latency_ms", "per_task_acc"])
    frozen_m = _pick(frozen, ["online_accuracy", "f1_faulty", "roc_auc",
                              "latency_us_p50", "latency_us_p99", "bss_bytes", "parity_rate"])
    online_m = _pick(online, ["online_accuracy", "online_forgetting",
                              "latency_us_p50", "latency_us_p99", "parity_rate",
                              "latency_inference_only_us_p50", "latency_update_overhead_us_p50"])

    delta = {
        "acc_final": _abs_delta(pc_m["acc_final"], frozen_m["online_accuracy"]),
        "f1_faulty": _abs_delta(pc_m["f1_faulty"], frozen_m["f1_faulty"]),
    }

    return {
        "pc": pc_m,
        "board_frozen": frozen_m,
        "board_online": online_m,
        # Sprint 36 rework (S3612) : axe INT8 vs FP32 board (frozen + online).
        "board_frozen_int8": _int8_cell(frozen_i8, frozen),
        "board_online_int8": _int8_cell(online_i8, online),
        "delta_pc_board": delta,
    }


def main() -> None:
    results: dict = {}
    for ds in DATASETS:
        results[ds] = {}
        for cond in CONDITIONS:
            results[ds][cond] = build_cell(ds, cond)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "ewc",
        "results": results,
    }
    out = EXPERIMENTS / "exp_S36_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # Récap console.
    print(f"→ {out}")
    for ds in DATASETS:
        for cond in CONDITIONS:
            c = results[ds][cond]
            print(f"  {ds:10s} {cond:5s} | PC acc_final={c['pc']['acc_final']} "
                  f"F1={c['pc']['f1_faulty']} | board_frozen acc={c['board_frozen']['online_accuracy']} "
                  f"lat={c['board_frozen']['latency_us_p50']}µs parity={c['board_frozen']['parity_rate']} "
                  f"| online lat={c['board_online']['latency_us_p50']}µs "
                  f"parity~{c['board_online']['parity_rate']} | Δacc={c['delta_pc_board']['acc_final']}")
            i8 = c["board_frozen_int8"]
            print(f"             INT8 | frozen lat={i8['latency_us_p50']}µs "
                  f"F1={i8['f1_faulty']} agree_fp32={i8['agreement_int8_vs_fp32']} "
                  f"ramx={i8['ram_ratio_fp32_over_int8']} gap3_ram={i8['gap3_ram_ok']} "
                  f"| online lat={c['board_online_int8']['latency_us_p50']}µs")


if __name__ == "__main__":
    main()
