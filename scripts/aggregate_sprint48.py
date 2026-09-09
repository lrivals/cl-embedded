#!/usr/bin/env python3
"""aggregate_sprint48.py — Agrégat unique du portage board sub-INT8 EWC (S4805).

Fusionne les cellules **mesurées-board** (S4804 : `.bss` packé/non-packé, latence DWT, parité,
CRC) et la **RAM théorique bit-packée** du Sprint 47 (émulateur) dans un seul
``experiments/exp_S48_summary.json`` indexé ``[dataset][weight_bits][granularity][platform]``
avec ``platform ∈ {board, pc}``.

**Lecture seule** : aucune métrique recalculée. Le nœud d'honnêteté du sprint est mis en
regard explicite (jamais conflaté) :
  - board *non-packé* : `.bss` ≈ INT8 (un sub-INT8 dans un conteneur int8 n'économise rien) ;
  - board *packé*     : `.bss` réduit → matérialise le ÷2/÷4/÷8 (bytes réellement économisés) ;
  - pc (S47)          : RAM **théorique** (÷8 à INT4) — ce que l'émulateur ne peut que prédire.

Le **résultat clé** = l'écart théorie↔matériel : la RAM théorique (S47) vs le `.bss` réel (S48).

Usage :
    python scripts/aggregate_sprint48.py       # → experiments/exp_S48_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS = Path("experiments")
BOARD_DIR = EXPERIMENTS / "exp_S48_board"
S47_DIR = EXPERIMENTS / "exp_S47_depth"
OUT = EXPERIMENTS / "exp_S48_summary.json"

GAP2_LATENCY_US = 100_000
RAM_BUDGET_BYTES = 256 * 1024

DATASETS = ("monitoring", "pronostia")
GRANULARITY = "per_channel"
# mode → (weight_bits, tag S47).
MODES = {"int4": (4, "int4"), "ternary": (2, "ternaire"), "binary": (1, "binaire")}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _delta(a, b):
    return round(a - b, 6) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None


def _num(v):
    """Valeur numérique ou None (les sentinelles ``"à mesurer"`` deviennent None)."""
    return v if isinstance(v, (int, float)) else None


def _board_platform(dataset: str, mode: str) -> dict:
    """Sous-bloc board : non-packé + packé + gain de packing mesuré."""
    out: dict = {"nonpacked": None, "packed": None,
                 "bss_saved_by_packing": None, "na_reason": None}
    reasons = []
    for packed in (False, True):
        key = "packed" if packed else "nonpacked"
        suffix = "_packed" if packed else ""
        cell = _load(BOARD_DIR / f"exp_S48_{dataset}_{mode}_{GRANULARITY}{suffix}" / "results.json")
        parity = _load(BOARD_DIR / f"exp_S48_parity_{dataset}_{mode}_{GRANULARITY}{suffix}.json")
        if cell is None:
            reasons.append(f"{key} non flashé")
            continue
        measured = _num(cell.get("latency_dwt_p50_us")) is not None
        out[key] = {
            "measured": bool(measured),
            "auroc_board": _num(cell.get("auroc_board")),
            "parity_pred": (parity or cell).get("parity_pred"),
            "latency_dwt_p50_us": _num(cell.get("latency_dwt_p50_us")),
            "latency_dwt_p99_us": _num(cell.get("latency_dwt_p99_us")),
            "bss_bytes": _num(cell.get("bss_bytes")),
            "crc_errors": cell.get("crc_errors"),
            "f1_faulty": _num(cell.get("f1_faulty")),
            "gap2_ok": cell.get("gap2_ok"),
            "gap3_ram_ok": cell.get("gap3_ram_ok"),
            "na_reason": cell.get("na_reason"),
        }
        if cell.get("na_reason"):
            reasons.append(f"{key}: {cell['na_reason']}")

    np_cell, p_cell = out["nonpacked"], out["packed"]
    if np_cell and p_cell and np_cell["bss_bytes"] and p_cell["bss_bytes"]:
        # Octets réellement économisés par le packing (les matrices de poids ; le reste du
        # .bss est un overhead fixe partagé) — jamais conflaté avec bss_bytes brut.
        out["bss_saved_by_packing"] = np_cell["bss_bytes"] - p_cell["bss_bytes"]
    if reasons:
        out["na_reason"] = " ; ".join(reasons)
    return out


def _pc_platform(dataset: str, tag: str) -> dict:
    """Sous-bloc PC (S47) : AUROC quantifiée + RAM théorique bit-packée."""
    s47 = _load(S47_DIR / f"exp_S47_ewc_{dataset}_{tag}_{GRANULARITY}.json")
    if s47 is None:
        return {"na_reason": f"exp_S47 {dataset}/{tag} absent"}
    return {
        "auroc_quant": _num(s47.get("auroc_quant")),
        "auroc_fp32": _num(s47.get("auroc_fp32")),
        "delta_auroc_vs_fp32": _num(s47.get("delta_auroc")),
        "ram_ratio_theoretical_vs_fp32": _num(s47.get("ram_ratio_vs_fp32")),
        "ram_weight_bytes_theoretical": _num(s47.get("ram_weight_bytes_theoretical")),
        "ram_note": s47.get("ram_note"),
        "na_reason": None,
    }


def build_cell(dataset: str, mode: str) -> dict:
    bits, tag = MODES[mode]
    board = _board_platform(dataset, mode)
    pc = _pc_platform(dataset, tag)

    # Deltas explicites (théorie↔matériel = résultat clé). Le PC est FP32-relatif
    # (théorique) ; le board est un .bss absolu — on ne conflate pas les unités, on
    # expose les deux et le delta AUROC board(packé)↔PC.
    board_packed = board.get("packed") or {}
    deltas = {
        "auroc_board_packed_minus_pc_quant":
            _delta(board_packed.get("auroc_board"), pc.get("auroc_quant")),
        "ram_ratio_theoretical_vs_fp32": pc.get("ram_ratio_theoretical_vs_fp32"),
        "bss_saved_by_packing_bytes": board.get("bss_saved_by_packing"),
    }
    return {"mode": mode, "board": board, "pc": pc, "deltas": deltas}


def main() -> None:
    summary: dict = {}
    for ds in DATASETS:
        summary[ds] = {}
        for mode, (bits, _tag) in MODES.items():
            summary[ds].setdefault(str(bits), {})[GRANULARITY] = build_cell(ds, mode)

    doc = {
        "sprint": 48, "task": "S4805",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gap2_latency_us_threshold": GAP2_LATENCY_US,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "bss_default_invariant": 105_036,   # doc : build par défaut sans flag sub-INT8
        "note": ("board = mesuré NUCLEO-F439ZI (S4804) ; pc = théorique émulateur S47. "
                 "bss_saved_by_packing = octets réels gagnés par le bit-packing (matrices de poids)."),
        "results_by_condition": summary,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"[S4805] écrit {OUT}")
    # Table compacte.
    for ds in DATASETS:
        for bits in summary[ds]:
            c = summary[ds][bits][GRANULARITY]
            b = c["board"]
            np_c, p_c = b.get("nonpacked") or {}, b.get("packed") or {}
            print(f"  {ds:11s} bits={bits} np.bss={np_c.get('bss_bytes')} "
                  f"pk.bss={p_c.get('bss_bytes')} saved={b.get('bss_saved_by_packing')} "
                  f"parity_pk={p_c.get('parity_pred')} lat_p50={p_c.get('latency_dwt_p50_us')}")


if __name__ == "__main__":
    main()
