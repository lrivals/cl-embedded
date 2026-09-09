#!/usr/bin/env python3
"""aggregate_article_ewc.py — Agrégat unique des métriques EWC de l'article (S4008).

Rassemble en **lecture seule** les mesures dispersées sur dix répertoires d'expériences
(S36, S39, S40, S46, S47, S48, S49, S50, S53, S54) dans un seul
``experiments/exp_S40_article_metrics/summary.json`` indexé ``[dataset][axe][cellule]``
pour ``dataset ∈ {monitoring, pronostia}`` — les deux jeux de l'article.

Conventions d'honnêteté du dépôt respectées :
  - **aucun chiffre inventé** : chaque cellule porte ``value`` + ``source_json`` + ``platform``
    ∈ {``mesuré board``, ``émulé PC``, ``mesuré PC``, ``théorique``} et ``na_reason`` si absente ;
    une valeur manquante vaut ``None`` ou le sentinel littéral ``"à mesurer"`` — **jamais 0** ;
  - **les trois estimateurs d'énergie ne sont JAMAIS fusionnés** (consigne portée par les JSON
    S53 eux-mêmes) : ``energy.estimators`` les reporte côte à côte avec leur ``method_note`` ;
  - l'oubli catastrophique (S54) est mesuré sur **CWRU**, hors des deux jeux de l'article : il
    va dans un bloc ``context`` étiqueté, jamais mélangé aux lignes Pronostia/Monitoring.

Aucune métrique n'est recalculée : seuls les *dérivés arithmétiques* (BOPs, ratios, deltas)
sont produits, à partir de valeurs lues.

Usage :
    python scripts/aggregate_article_ewc.py   # → experiments/exp_S40_article_metrics/summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.compute_cost import (  # noqa: E402
    compute_bops_for_model,
    compute_flops_for_model,
    compute_macs,
    compute_params_for_model,
)

EXPERIMENTS = ROOT / "experiments"
OUT_DIR = EXPERIMENTS / "exp_S40_article_metrics"
OUT = OUT_DIR / "summary.json"

#: Les deux jeux de l'article, et la dimension d'entrée native de chacun (condition `5feat`).
DATASETS: dict[str, int] = {"monitoring": 4, "pronostia": 5}
HIDDEN_DIMS = (32, 16)
N_CLASSES = 2
CONDITION = "5feat"

BOARD, EMU_PC, PC, THEO = "mesuré board", "émulé PC", "mesuré PC", "théorique"

#: Sentinel littéral des mesures non faites (convention Sprint 33 — LPM01A).
A_MESURER = "à mesurer"


# ── Utilitaires de lecture (aucune écriture, aucune invention) ────────────────

def _load(path: Path) -> dict | None:
    """JSON ou ``None`` si le fichier n'existe pas (mesure jamais produite)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    """Chemin relatif au dépôt, pour la traçabilité de chaque cellule."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def cell(value, source: Path | str | None, platform: str, na_reason: str | None = None) -> dict:
    """Cellule traçable ``{value, source_json, platform, na_reason}``.

    Une valeur absente reste ``None`` (ou le sentinel ``"à mesurer"``) accompagnée d'une
    ``na_reason`` — la convention du dépôt interdit de la remplacer par 0.
    """
    src = _rel(source) if isinstance(source, Path) else source
    if value is None and na_reason is None:
        na_reason = f"valeur absente de {src}"
    return {"value": value, "source_json": src, "platform": platform, "na_reason": na_reason}


def _dig(obj, *keys):
    """Descente tolérante dans des dicts imbriqués (``None`` dès qu'une clé manque)."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def _delta(a, b):
    """``a - b`` si les deux sont numériques, sinon ``None`` (jamais 0 par défaut)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(a - b, 6)
    return None


# ── Axe 1 — Performance CL appariée PC ↔ board (S36) ──────────────────────────

def axis_performance(ds: str) -> dict:
    """Performance CL PC/board, frozen et online, en FP32 et INT8 legacy (Sprint 36)."""
    path = EXPERIMENTS / "exp_S36_summary.json"
    blk = _dig(_load(path), "results", ds, CONDITION) or {}
    pc, frz, onl = blk.get("pc") or {}, blk.get("board_frozen") or {}, blk.get("board_online") or {}
    frz8, onl8 = blk.get("board_frozen_int8") or {}, blk.get("board_online_int8") or {}
    out = {}
    for key in ("acc_final", "aa", "af", "bwt", "f1_faulty", "f1_macro", "roc_auc"):
        out[f"pc_{key}"] = cell(pc.get(key), path, PC)
    out["pc_inference_latency_ms"] = cell(pc.get("inference_latency_ms"), path, PC)
    out["board_frozen_f1_faulty"] = cell(frz.get("f1_faulty"), path, BOARD)
    out["board_frozen_accuracy"] = cell(frz.get("online_accuracy"), path, BOARD)
    out["board_frozen_latency_us_p50"] = cell(frz.get("latency_us_p50"), path, BOARD)
    out["board_frozen_bss_bytes"] = cell(frz.get("bss_bytes"), path, BOARD)
    out["board_frozen_parity_rate"] = cell(frz.get("parity_rate"), path, BOARD)
    out["board_online_parity_rate"] = cell(onl.get("parity_rate"), path, BOARD)
    out["board_online_latency_us_p50"] = cell(onl.get("latency_us_p50"), path, BOARD)
    out["board_online_update_overhead_us_p50"] = cell(
        onl.get("latency_update_overhead_us_p50"), path, BOARD)
    # INT8 « legacy » embarqué : l'effondrement mesuré carte, point de départ de l'article.
    out["board_frozen_int8_legacy_f1_faulty"] = cell(frz8.get("f1_faulty"), path, BOARD)
    out["board_frozen_int8_legacy_agreement"] = cell(frz8.get("agreement_int8_vs_fp32"), path, BOARD)
    out["board_online_int8_legacy_f1_faulty"] = cell(onl8.get("f1_faulty"), path, BOARD)
    out["board_frozen_int8_ram_ratio"] = cell(frz8.get("ram_ratio_fp32_over_int8"), path, BOARD)
    return out


# ── Axe 2 — Ablation INT8 émulée bit-exact (S39) ──────────────────────────────

def axis_ablation(ds: str) -> dict:
    """Échelle d'ablation `legacy_c → … → q15` et RAM des poids par schéma (émulateur PC)."""
    abl_path = EXPERIMENTS / "exp_S39_ablation" / f"{ds}.json"
    sweep_path = EXPERIMENTS / "exp_S39_quant_sweep" / f"ewc_{ds}.json"
    abl, sweep = _load(abl_path), _load(sweep_path)
    out: dict = {}
    if abl:
        out["f1_fp32_reference"] = cell(abl.get("f1_fp32"), abl_path, EMU_PC)
        out["dominant_factor"] = cell(abl.get("dominant_factor"), abl_path, EMU_PC)
        for step in abl.get("ladder") or []:
            out[f"ladder_{step['scheme']}_f1"] = cell(step.get("f1"), abl_path, EMU_PC)
    else:
        out["f1_fp32_reference"] = cell(None, abl_path, EMU_PC, "exp_S39_ablation absent")
    for scheme, blk in (_dig(sweep, "schemes") or {}).items():
        out[f"scheme_{scheme}_metric"] = cell(blk.get("metric"), sweep_path, EMU_PC)
        out[f"scheme_{scheme}_ram_weights_bytes"] = cell(
            blk.get("ram_weights_bytes"), sweep_path, EMU_PC)
        out[f"scheme_{scheme}_bops_proxy"] = cell(blk.get("bops_proxy"), sweep_path, THEO)
    return out


# ── Axe 3 — Récupération INT8 mesurée carte (S40 board v2) ────────────────────

def axis_recovery(ds: str) -> dict:
    """Kernel v2 calibré flashé : ce que l'émulateur prédisait, mesuré sur la carte."""
    out: dict = {}
    for proto in ("frozen", "online"):
        path = EXPERIMENTS / "exp_S40_board_v2" / f"results_per_channel_{ds}_{proto}.json"
        d = _load(path)
        if d is None:
            reason = "cellule board v2 non streamée (voir S4010_mesures_manquantes.md)"
            for k in ("f1_faulty", "latency_us_p50", "bss_bytes", "parity_rate",
                      "agreement_int8_vs_fp32", "crc_errors"):
                out[f"per_channel_{proto}_{k}"] = cell(None, path, BOARD, reason)
            continue
        for k in ("f1_faulty", "f1_macro", "latency_us_p50", "bss_bytes", "parity_rate",
                  "agreement_int8_vs_fp32", "crc_errors", "ram_ratio_fp32_over_quant",
                  "ram_weights_fp32_bytes", "ram_weights_quant_bytes"):
            out[f"per_channel_{proto}_{k}"] = cell(d.get(k), path, BOARD)
    # Schémas de la grille S4002 restés non mesurés : tracés, jamais inventés.
    for scheme in ("q15", "int8_legacy"):
        for proto in ("frozen", "online"):
            path = EXPERIMENTS / "exp_S40_board_v2" / f"results_{scheme}_{ds}_{proto}.json"
            d = _load(path)
            out[f"{scheme}_{proto}_f1_faulty"] = cell(
                d.get("f1_faulty") if d else A_MESURER, path, BOARD,
                None if d else "cellule de la grille S4002 non streamée (carte requise)")
    return out


# ── Axe 4 — Moment de quantification (S46) ────────────────────────────────────

def axis_moment(ds: str) -> dict:
    """QAT (avant) / PTQ (après) / les deux — émulé PC (S46 A–D) puis mesuré carte (S4608)."""
    pc_path = EXPERIMENTS / "exp_S46_ewc" / f"{ds}_all.json"
    pc = _load(pc_path)
    out: dict = {}
    for moment, blk in (_dig(pc, "moments") or {}).items():
        out[f"{moment}_auroc"] = cell(blk.get("metric"), pc_path, EMU_PC)
        out[f"{moment}_ram_weights_bytes"] = cell(blk.get("ram_weights_bytes"), pc_path, EMU_PC)
    for k in ("delta_before_vs_fp32", "delta_after_vs_fp32", "delta_both_vs_fp32"):
        out[k] = cell(_dig(pc, k), pc_path, EMU_PC)
    board_path = EXPERIMENTS / "exp_S46_board" / f"{ds}_both.json"
    b = _load(board_path)
    if b is None:
        out["board_both_f1_faulty"] = cell(
            None, board_path, BOARD, "cellule QAT board non streamée")
        return out
    for k in ("f1_faulty", "f1_macro", "latency_dwt_us_p50", "bss_bytes", "parity_board_pc",
              "ab_vs_after", "f1_after_board", "crc_errors", "agreement_int8_vs_fp32"):
        out[f"board_both_{k}"] = cell(b.get(k), board_path, BOARD)
    return out


# ── Axe 5 — Profondeur / granularité / symétrie, émulé PC (S47) ───────────────

#: Profondeurs balayées au Sprint 47 (le tag de fichier n'est pas le nombre de bits).
DEPTH_TAGS = {"int8": 8, "int6": 6, "int4": 4, "int3": 3, "int2": 2, "ternaire": 2, "binaire": 1}


def axis_depth_pc(ds: str) -> dict:
    """AUROC vs nombre de bits × granularité, et RAM **théorique** bit-packée (S47)."""
    out: dict = {}
    depth_dir = EXPERIMENTS / "exp_S47_depth"
    for tag in DEPTH_TAGS:
        for gran in ("per_tensor", "per_channel"):
            path = depth_dir / f"exp_S47_ewc_{ds}_{tag}_{gran}.json"
            d = _load(path)
            if d is None:
                continue
            pfx = f"{tag}_{gran}"
            out[f"{pfx}_auroc_quant"] = cell(d.get("auroc_quant"), path, EMU_PC)
            out[f"{pfx}_delta_auroc"] = cell(d.get("delta_auroc"), path, EMU_PC)
            out[f"{pfx}_weight_bits"] = cell(d.get("weight_bits"), path, EMU_PC)
            # La RAM sub-INT8 n'est un gain que bit-packée : théorique tant que S48 ne la mesure pas.
            out[f"{pfx}_ram_ratio_vs_fp32_theoretical"] = cell(
                d.get("ram_ratio_vs_fp32"), path, THEO)
    sym_dir = EXPERIMENTS / "exp_S47_symmetry"
    for path in sorted(sym_dir.glob(f"exp_S47_ewc_{ds}_*.json")) if sym_dir.exists() else []:
        d = _load(path) or {}
        tag = path.stem.replace(f"exp_S47_ewc_{ds}_", "")
        out[f"symmetry_{tag}_auroc_quant"] = cell(d.get("auroc_quant"), path, EMU_PC)
        out[f"symmetry_{tag}_delta_auroc"] = cell(d.get("delta_auroc"), path, EMU_PC)
    return out


# ── Axe 6 — Profondeur mesurée carte, packé vs non-packé (S48) ────────────────

def axis_depth_board(ds: str) -> dict:
    """`.bss` réelle packé/non-packé, latence de dépacking et parité vs émulateur (S48)."""
    path = EXPERIMENTS / "exp_S48_summary.json"
    by_cond = _dig(_load(path), "results_by_condition", ds) or {}
    out: dict = {}
    for bits, grans in by_cond.items():
        for gran, cellblk in grans.items():
            board = cellblk.get("board") or {}
            for packing in ("nonpacked", "packed"):
                blk = board.get(packing) or {}
                pfx = f"bits{bits}_{gran}_{packing}"
                for k in ("auroc_board", "f1_faulty", "latency_dwt_p50_us", "bss_bytes",
                          "parity_pred", "crc_errors"):
                    out[f"{pfx}_{k}"] = cell(blk.get(k), path, BOARD,
                                             blk.get("na_reason") if blk.get(k) is None else None)
            out[f"bits{bits}_{gran}_bss_saved_by_packing"] = cell(
                board.get("bss_saved_by_packing"), path, BOARD)
    return out


# ── Axe 7 — RAM totale = .data + .bss + pic de pile (S49) ─────────────────────

def axis_ram(ds: str) -> dict:
    """RAM système complète (la RAM des poids ÷4 ne dit rien du total — nuance S49)."""
    path = EXPERIMENTS / "exp_S49_ram" / "summary.json"
    d = _load(path)
    budget = _dig(d, "_meta", "ram_budget_bytes")
    blk = _dig(d, "ewc", ds, CONDITION) or {}
    out: dict = {}
    for enc in ("fp32", "int8"):
        for plat_key, plat_label in (("board", BOARD), ("pc", PC)):
            b = _dig(blk, enc, plat_key) or {}
            reason = b.get("na_reason")
            for k in ("data", "bss", "stack_peak_inference", "stack_peak_update", "total"):
                out[f"{enc}_{plat_key}_{k}"] = cell(
                    b.get(k), path, plat_label, reason if b.get(k) is None else None)
            if plat_key == "board" and isinstance(b.get("total"), (int, float)) and budget:
                out[f"{enc}_board_total_pct_budget"] = cell(
                    round(100.0 * b["total"] / budget, 2), path, BOARD)
        ratio = _dig(blk, enc, "board", "ratio_int8_vs_fp32")
        if ratio is not None:
            out[f"{enc}_board_ratio_int8_vs_fp32"] = cell(ratio, path, BOARD)
    out["ram_budget_bytes"] = cell(budget, path, THEO)
    return out


# ── Axe 8 — Décomposition de la latence INT8 (S50) ────────────────────────────

def axis_latency(ds: str) -> dict:
    """Poste par poste du kernel INT8 v2 : déquant / MAC / requant (cycles DWT bruts)."""
    path = EXPERIMENTS / "exp_S50_int8_latency" / "ewc.json"
    d = _load(path)
    out: dict = {}
    for seg, blk in (_dig(d, "segments") or {}).items():
        out[f"segment_{seg}_cycles_p50"] = cell(
            _dig(blk, "by_dataset_cycles_p50", ds), path, BOARD)
        out[f"segment_{seg}_cycles_p50_mean"] = cell(
            blk.get("cycles_p50_mean_over_datasets"), path, BOARD)
    int8_us = _dig(d, "total_int8_us_p50_by_dataset", ds)
    fp32_us = _dig(d, "total_fp32_us_p50_by_dataset", ds)
    out["total_int8_us_p50"] = cell(int8_us, path, BOARD)
    out["total_fp32_us_p50"] = cell(fp32_us, path, BOARD)
    out["int8_minus_fp32_us"] = cell(_delta(int8_us, fp32_us), path, BOARD)
    out["cpu_hz"] = cell(_dig(d, "cpu_hz"), path, BOARD)
    return out


# ── Axe 9 — Énergie : trois estimateurs, jamais fusionnés (S53) ───────────────

def energy_estimators() -> dict:
    """Chaque estimateur d'énergie garde sa méthode, sa note et ses cellules propres.

    Le dépôt interdit explicitement de les moyenner ou de les fondre en une valeur unique :
    « leur comparaison est elle-même un résultat » (``method_note`` des JSON S53).
    """
    est: dict = {}

    # (a) Régression du courant en fonction de la cadence imposée (S5304).
    rate_path = EXPERIMENTS / "exp_S53_rate_sweep" / "summary.json"
    rate = _load(rate_path)
    cells: dict = {}
    for name in ("ewc_fp32", "ewc_int8"):
        blk = _dig(rate, "cells", name) or {}
        for k in ("energy_uj_per_inference", "energy_uncertainty_uj", "r2",
                  "slope_ua_per_hz", "intercept_ma", "dwt_latency_us_p50"):
            cells[f"{name}_{k}"] = cell(blk.get(k), rate_path, BOARD,
                                        blk.get("energy_na_reason") if blk.get(k) is None else None)
    est["regression"] = {
        "method": _dig(rate, "method"),
        "method_note": _dig(rate, "method_note"),
        "gap3_energy_verdict": _dig(rate, "gap3_energy_verdict"),
        "cells": cells,
    }

    # (b) Delta charge − repos WFI (S5302).
    delta_cells: dict = {}
    method_note = None
    for name in ("ewc_fp32", "ewc_int8"):
        path = EXPERIMENTS / "exp_S53_wfi" / f"{name}.json"
        d = _load(path) or {}
        method_note = method_note or d.get("method")
        delta_cells[f"{name}_energy_uj_per_inference"] = cell(
            d.get("energy_uj_per_inference"), path, BOARD)
        delta_cells[f"{name}_energy_uj_per_update"] = cell(
            d.get("energy_uj_per_update"), path, BOARD, d.get("energy_update_na_reason"))
    auto_path = EXPERIMENTS / "exp_S53_wfi" / "autonomy_delta.json"
    auto = _load(auto_path) or {}
    delta_cells["i_idle_ma_wfi"] = cell(auto.get("i_idle_ma"), auto_path, BOARD)
    delta_cells["autonomy_h_2000mah_period_1s"] = cell(
        _dig(auto, "per_model", "ewc_fp32", "autonomy_h_by_period", "1.0",
             "autonomy_h_by_mah", "2000.0"), auto_path, BOARD)
    est["delta_wfi"] = {
        "method": "delta charge − repos WFI",
        "method_note": method_note,
        "scope_mesure": auto.get("scope_mesure"),
        "cells": delta_cells,
    }

    # (c) Lot d'inférences `-DINFER_BATCH_N` : isole le calcul de la trame UART (S5302).
    batch_path = EXPERIMENTS / "exp_S53_wfi" / "batch_sweep.json"
    batch = _load(batch_path) or {}
    batch_cells: dict = {}
    for name in ("ewc_fp32",):
        blk = batch.get(name) or {}
        for k in ("energy_uj_per_inference", "r2", "slope_a_per_inference_per_frame",
                  "intercept_a", "n_points_excluded_saturation"):
            batch_cells[f"{name}_{k}"] = cell(blk.get(k), batch_path, BOARD)
    est["batch"] = {
        "method": "lot d'inférences par trame (INFER_BATCH_N)",
        "method_note": _dig(batch, "ewc_fp32", "method"),
        "cells": batch_cells,
    }

    # (d) Balayage de fréquence : la marge de latence est convertible en autonomie (S5303).
    freq_path = EXPERIMENTS / "exp_S53_freq_sweep" / "summary.json"
    freq = _load(freq_path) or {}
    freq_cells: dict = {}
    for mhz in (freq.get("frequencies_mhz") or []):
        key = str(mhz)
        freq_cells[f"{key}MHz_energy_uj_per_inference"] = cell(
            _dig(freq, "energy_uj_per_inference_by_mhz", key), freq_path, BOARD)
        freq_cells[f"{key}MHz_i_idle_ma"] = cell(
            _dig(freq, "i_idle_ma_by_mhz", key), freq_path, BOARD)
        freq_cells[f"{key}MHz_ewc_latency_us_p50"] = cell(
            _dig(freq, "dwt_latency_us_p50_by_mhz", key, "ewc"), freq_path, BOARD)
        freq_cells[f"{key}MHz_gap2_ok"] = cell(
            _dig(freq, "gap2_ok_by_mhz", key), freq_path, BOARD)
    est["freq_sweep"] = {
        "method": "régression I(rate) répétée par fréquence SYSCLK",
        "method_note": freq.get("tendance_rationale"),
        "tendance": freq.get("tendance_energie"),
        "cells": freq_cells,
    }

    # (e) Énergie par mise à jour CL : différence de pentes, non publiable en l'état.
    pol_path = EXPERIMENTS / "exp_S53_policy_energy" / "economy_energy.json"
    pol = _load(pol_path) or {}
    pol_cells: dict = {}
    for ds in DATASETS:
        blk = _dig(pol, "by_dataset", ds, "energy_uj_per_update") or {}
        pol_cells[f"{ds}_energy_uj_per_update"] = cell(
            blk.get("value_uj", A_MESURER), pol_path, BOARD, blk.get("na_reason"))
    est["policy_update"] = {
        "method": pol.get("method"),
        "method_note": pol.get("method_note"),
        "cells": pol_cells,
    }
    return est


# ── Axe 10 — Coût de calcul analytique (étape S4008-1) ────────────────────────

def axis_compute_cost(ds: str, n_in: int) -> dict:
    """MACs / FLOPs / BOPs / paramètres de la tête ``k→32→16→2``.

    Les MACs mesurés proviennent de ``scripts/measure_macs.py --out`` (cross-check torchinfo) ;
    FLOPs et BOPs en dérivent analytiquement. Le ratio BOPs FP32/INT8 = (32/8)² = 16 est
    **théorique** : la latence réellement mesurée, elle, *augmente* (paradoxe FPU, S50).
    """
    macs_path = OUT_DIR / f"compute_cost_ewc_{ds}.json"
    measured = _load(macs_path)
    kw = dict(n_features=n_in, hidden_dims=list(HIDDEN_DIMS), n_classes=N_CLASSES)
    macs = compute_macs("EWC", **kw)
    flops = compute_flops_for_model("EWC", **kw)
    bops32 = compute_bops_for_model("EWC", n_bits=32, **kw)
    bops8 = compute_bops_for_model("EWC", n_bits=8, **kw)
    n_params = compute_params_for_model("EWC", **kw)
    out = {
        "macs": cell(macs, macs_path if measured else "src/evaluation/compute_cost.py", THEO),
        "flops": cell(flops, "src/evaluation/compute_cost.py", THEO),
        "bops_fp32": cell(bops32, "src/evaluation/compute_cost.py", THEO),
        "bops_int8": cell(bops8, "src/evaluation/compute_cost.py", THEO),
        "bops_ratio_fp32_over_int8": cell(
            bops32 // bops8 if bops8 else None, "src/evaluation/compute_cost.py", THEO),
        "n_params": cell(n_params, "src/evaluation/compute_cost.py", THEO),
        "n_features": cell(n_in, "src/evaluation/compute_cost.py", THEO),
    }
    # Grounding : le compte de paramètres doit coïncider avec celui mesuré au Sprint 39
    # (sinon l'architecture supposée ici n'est pas celle réellement quantifiée/portée).
    sweep_path = EXPERIMENTS / "exp_S39_quant_sweep" / f"ewc_{ds}.json"
    n_params_s39 = _dig(_load(sweep_path), "n_params")
    out["n_params_s39_reference"] = cell(n_params_s39, sweep_path, EMU_PC)
    out["n_params_matches_s39"] = cell(
        (n_params == n_params_s39) if n_params_s39 is not None else None, sweep_path, THEO,
        None if n_params_s39 is not None else "exp_S39_quant_sweep absent")

    if measured:
        out["macs_torchinfo"] = cell(measured.get("macs_torchinfo"), macs_path, THEO)
        out["macs_analytical_vs_torchinfo_delta_pct"] = cell(
            measured.get("delta_pct"), macs_path, THEO)
    else:
        out["macs_torchinfo"] = cell(
            None, macs_path, THEO,
            "lancer scripts/measure_macs.py --model ewc --n-in "
            f"{n_in} --out {_rel(macs_path)}")
    return out


# ── Contexte hors-périmètre : oubli catastrophique mesuré sur CWRU (S54) ──────

def context_forgetting() -> dict:
    """AF mesuré au Sprint 54 — sur **CWRU**, donc jamais fondu dans les lignes de l'article."""
    out: dict = {}
    for variant in ("ewc", "naive"):
        path = EXPERIMENTS / f"exp_S54_forgetting_{variant}" / "results.json"
        d = _load(path) or {}
        out[f"{variant}_avg_forgetting_f1"] = cell(d.get("avg_forgetting_f1"), path, PC)
        out[f"{variant}_dataset"] = cell(d.get("dataset"), path, PC)
    out["_note"] = (
        "Mesuré sur CWRU (class-incremental, 3 tâches), hors des deux jeux de l'article : "
        "reporté comme contexte, jamais mélangé aux lignes Pronostia/Monitoring. "
        "Sur Pronostia/Monitoring, exp_S36 donne AF ≤ 0.01 sur un scénario CL court."
    )
    return out


# ── Assemblage ───────────────────────────────────────────────────────────────

AXES = ("performance", "ablation_int8", "recovery_board", "moment", "depth_pc",
        "depth_board", "ram", "latency", "compute_cost")


def collect_missing(doc: dict) -> list[str]:
    """Chemins ``dataset.axe.cellule`` (ou ``energy.…``) dont la mesure n'existe pas."""
    missing: list[str] = []

    def scan(node, prefix: str) -> None:
        if isinstance(node, dict):
            if "value" in node and "platform" in node:
                if node["value"] is None or node["value"] == A_MESURER:
                    missing.append(prefix)
                return
            for k, v in node.items():
                if k.startswith("_"):
                    continue
                scan(v, f"{prefix}.{k}" if prefix else k)

    for ds in DATASETS:
        scan(doc[ds], ds)
    scan(doc["energy"]["estimators"], "energy.estimators")
    scan(doc["context"]["forgetting"], "context.forgetting")
    return sorted(missing)


def main() -> None:
    doc: dict = {
        "_meta": {
            "generated_by": "scripts/aggregate_article_ewc.py",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sprint": 40, "task": "S4008", "model": "ewc", "condition": CONDITION,
            "platforms": [BOARD, EMU_PC, PC, THEO],
            "note": ("Lecture seule : aucune mesure recalculée, aucun JSON source réécrit. "
                     "Chaque cellule porte value/source_json/platform/na_reason ; une mesure "
                     "absente vaut null ou « à mesurer », jamais 0."),
            "energy_note": ("Les trois estimateurs d'énergie (régression de cadence, delta/WFI, "
                            "lot INFER_BATCH_N) ne sont JAMAIS fusionnés : leur comparaison est "
                            "elle-même un résultat."),
        },
    }
    for ds, n_in in DATASETS.items():
        doc[ds] = {
            "performance": axis_performance(ds),
            "ablation_int8": axis_ablation(ds),
            "recovery_board": axis_recovery(ds),
            "moment": axis_moment(ds),
            "depth_pc": axis_depth_pc(ds),
            "depth_board": axis_depth_board(ds),
            "ram": axis_ram(ds),
            "latency": axis_latency(ds),
            "compute_cost": axis_compute_cost(ds, n_in),
        }
    doc["energy"] = {
        "_note": doc["_meta"]["energy_note"],
        "estimators": energy_estimators(),
    }
    doc["context"] = {"forgetting": context_forgetting()}
    doc["missing"] = collect_missing(doc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[S4008] écrit {_rel(OUT)}")
    for ds in DATASETS:
        n = sum(len(doc[ds][a]) for a in AXES)
        print(f"  {ds:11s} {n} cellules sur {len(AXES)} axes")
    print(f"  énergie : {len(doc['energy']['estimators'])} estimateurs séparés "
          f"({', '.join(doc['energy']['estimators'])})")
    print(f"  manquantes : {len(doc['missing'])} cellule(s) → S4010_mesures_manquantes.md")


if __name__ == "__main__":
    main()
