"""
generate_comparison_sprint23.py — Tableau comparatif final 5 datasets × 5 modèles.

Extension de compare_experiments.py (Sprint 21) avec :
  - 2 nouveaux datasets : CMAPSS, Paderborn
  - 5 modèles : mahalanobis, ewc, ewc_int8, tinyol, hdc
  - 2 plateformes : pc, nucleo_f439zi (board)
  - Chargement multi-format : Sprint 21 JSON, Sprint 22 nested results/, Sprint 23 pending

Sources par dataset :
  - cwru/monitoring/pronostia  : comparison_sprint21.json (board) + exp_S18/S19/S21 (board)
  - cmapss                    : exp_S22_01..04 (pc) + exp_S23_01..04 (board pending)
  - paderborn                 : exp_S22_05..08 (pc) + exp_S23_05..06 (board pending)

Usage :
    python scripts/generate_comparison_sprint23.py
    python scripts/generate_comparison_sprint23.py --output experiments/comparison_sprint23.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DATASETS  = ["cwru", "monitoring", "pronostia", "cmapss", "paderborn"]
_MODELS    = ["mahalanobis", "ewc", "ewc_int8", "tinyol", "hdc"]
_PLATFORMS = ["pc", "nucleo_f439zi"]

_GAP2_RAM_BUDGET  = 65_536   # 64 Ko — budget embarqué Gap 2
_GAP2_LATENCY_MS  = 100.0    # 100 ms — critère Gap 2

_SPRINT21_JSON = Path("experiments/comparison_sprint21.json")

_DEFAULT_EXP_DIRS: list[str] = [
    # Sprint 18-19 board
    "experiments/exp_S18_01",
    "experiments/exp_S18_01_board",
    "experiments/exp_S18_02",
    "experiments/exp_S19_01",
    "experiments/exp_S19_02",
    # Sprint 21 board (monitoring, pronostia)
    "experiments/exp_S21_01",
    "experiments/exp_S21_02",
    "experiments/exp_S21_03",
    "experiments/exp_S21_04",
    # Sprint 22 PC — CMAPSS
    "experiments/exp_S22_01",   # ewc cmapss
    "experiments/exp_S22_02",   # hdc cmapss
    "experiments/exp_S22_03",   # tinyol cmapss
    "experiments/exp_S22_04",   # mahalanobis cmapss
    # Sprint 22 PC — Paderborn
    "experiments/exp_S22_05",   # ewc paderborn
    "experiments/exp_S22_06",   # (absent ou incomplet)
    "experiments/exp_S22_07",   # tinyol paderborn
    "experiments/exp_S22_08",   # hdc paderborn
    # Sprint 22 INT8
    "experiments/exp_S22_INT8_01",
    "experiments/exp_S22_INT8_02",
    # Sprint 23 board CMAPSS
    "experiments/exp_S23_01",   # ewc cmapss board
    "experiments/exp_S23_02",   # tinyol cmapss board
    "experiments/exp_S23_03",   # mahalanobis cmapss board
    "experiments/exp_S23_04",   # hdc cmapss board
    # Sprint 23 board Paderborn
    "experiments/exp_S23_05",   # ewc paderborn board
    "experiments/exp_S23_06",   # mahalanobis paderborn board
    # Sprint 23 INT8 + benchmark
    "experiments/exp_S23_INT8",
    "experiments/exp_S23_benchmark",
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_model(model: str | None) -> str | None:
    if model is None:
        return None
    m = model.lower()
    if m in ("mahalanobis",):
        return "mahalanobis"
    if m in ("ewc", "ewc_head", "ewc_online"):
        return "ewc"
    if m in ("ewc_int8", "ewc_head_int8"):
        return "ewc_int8"
    if m in ("tinyol", "tiny_ol", "tinyol_oto"):
        return "tinyol"
    if m in ("hdc", "hdc_classifier"):
        return "hdc"
    return model


def _normalize_dataset(dataset: str | None) -> str | None:
    if dataset is None:
        return None
    d = dataset.lower()
    if d.startswith("cwru"):
        return "cwru"
    if d in ("monitoring", "equipment_monitoring"):
        return "monitoring"
    if d in ("pronostia", "femto"):
        return "pronostia"
    if d in ("cmapss",):
        return "cmapss"
    if d in ("paderborn",):
        return "paderborn"
    return dataset


def _normalize_platform(platform: str | None) -> str:
    if platform is None:
        return "pc"
    p = platform.lower()
    if p in ("nucleo_f439zi", "nucleo", "board", "stm32"):
        return "nucleo_f439zi"
    return "pc"


def _safe_float(val: Any) -> float | None:
    """Convertit une valeur en float, retourne None pour les placeholders."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("à mesurer", "a mesurer", "tbd", "pending", "—", "-", ""):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Chargement config_snapshot.yaml
# ---------------------------------------------------------------------------

def _load_config_snapshot(exp_dir: Path) -> dict:
    cfg_file = exp_dir / "config_snapshot.yaml"
    if not cfg_file.exists() or not _YAML_AVAILABLE:
        return {}
    try:
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception:
        return {}


def _dataset_from_config(cfg: dict) -> str | None:
    dataset = cfg.get("dataset") or cfg.get("data", {}).get("dataset")
    return _normalize_dataset(dataset)


def _model_from_config(cfg: dict) -> str | None:
    model = cfg.get("model") or cfg.get("model_type")
    return _normalize_model(model)


def _platform_from_config(cfg: dict) -> str:
    platform = cfg.get("platform")
    return _normalize_platform(platform)


# ---------------------------------------------------------------------------
# Loaders par format
# ---------------------------------------------------------------------------

def _cell_from_flat(r: dict, exp_dir_name: str) -> dict:
    """Construit une cellule depuis un dict plat (Sprint 21 / Sprint 23 style)."""
    cell: dict = {"exp": exp_dir_name}
    for src, dst in [
        ("acc_final",            "acc_final"),
        ("avg_forgetting",       "avg_forgetting"),
        ("backward_transfer",    "backward_transfer"),
        ("inference_latency_ms", "latency_ms"),
        ("latency_ms",           "latency_ms"),
        ("ram_peak_bytes",       "ram_bytes"),
        ("ram_bytes",            "ram_bytes"),
        ("lambda_ewc",           "lambda_ewc"),
        ("n_repetitions",        "n_repetitions"),
        ("gap2_ram_compliant",   "gap2_ram_compliant"),
        ("gap2_latency_compliant", "gap2_latency_compliant"),
        ("note",                 "note"),
    ]:
        val = r.get(src)
        if val is not None:
            safe = _safe_float(val) if dst in (
                "acc_final", "avg_forgetting", "backward_transfer",
                "latency_ms", "ram_bytes", "lambda_ewc") else val
            if safe is not None or dst not in (
                "acc_final", "avg_forgetting", "backward_transfer",
                "latency_ms", "ram_bytes", "lambda_ewc"):
                cell[dst] = safe if dst in (
                    "acc_final", "avg_forgetting", "backward_transfer",
                    "latency_ms", "ram_bytes", "lambda_ewc") else val
    return cell


def _load_s22_nested(exp_dir: Path) -> list[dict]:
    """Charge les résultats Sprint 22 depuis results/ (structure imbriquée)."""
    results_dir = exp_dir / "results"
    if not results_dir.is_dir():
        return []

    cfg = _load_config_snapshot(exp_dir)
    dataset = _dataset_from_config(cfg)
    platform = "pc"  # Sprint 22 = PC uniquement

    records: list[dict] = []

    # Chercher metrics.json ou metrics_cl.json
    for fname in ("metrics.json", "metrics_cl.json"):
        f = results_dir / fname
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        exp_id = data.get("exp_id", exp_dir.name)

        # Format C/D : dict plat avec model/dataset/acc_final
        if "acc_final" in data or "model" in data:
            model = _normalize_model(data.get("model")) or _model_from_config(cfg)
            ds = _normalize_dataset(data.get("dataset")) or dataset
            if model and ds:
                cell = _cell_from_flat(data, exp_id)
                records.append({
                    "model": model, "dataset": ds, "platform": platform,
                    "_exp_dir": exp_id, **cell,
                })
            break

        cl = data.get("cl_metrics", {})
        if not cl:
            break

        # Format B : cl_metrics plat (HDC) — clés = métrique (aa, af, ...)
        if "aa" in cl:
            model = _infer_model_from_results_dir(results_dir)
            ds = dataset
            if model and ds:
                cell = {
                    "exp": exp_id,
                    "acc_final": _safe_float(cl.get("aa")),
                    "avg_forgetting": _safe_float(cl.get("af")),
                    "backward_transfer": _safe_float(cl.get("bwt")),
                }
                # RAM depuis memory_report
                mem = _load_memory_report(results_dir)
                if mem:
                    cell.update(mem)
                records.append({
                    "model": model, "dataset": ds, "platform": platform,
                    "_exp_dir": exp_id, **cell,
                })
            break

        # Format A : cl_metrics.{model_name}.{aa, af}
        for model_key, metrics in cl.items():
            if model_key in ("memory", "naive", "joint"):
                continue
            normalized = _normalize_model(model_key)
            if normalized is None:
                continue
            ds = dataset
            if not ds:
                continue
            cell = {
                "exp": exp_id,
                "acc_final": _safe_float(metrics.get("aa")),
                "avg_forgetting": _safe_float(metrics.get("af")),
                "backward_transfer": _safe_float(metrics.get("bwt")),
            }
            # RAM + latence depuis memory_report
            mem = _load_memory_report(results_dir)
            if mem:
                cell.update(mem)
            records.append({
                "model": normalized, "dataset": ds, "platform": platform,
                "_exp_dir": exp_id, **cell,
            })
        break

    # Format E : results.json avec fp32/int8 (INT8 comparison)
    f = results_dir / "results.json"
    if f.exists() and not records:
        try:
            data = json.loads(f.read_text())
            ds = _normalize_dataset(data.get("dataset")) or dataset
            if "fp32" in data and ds:
                # EWC FP32
                fp32 = data["fp32"]
                records.append({
                    "model": "ewc", "dataset": ds, "platform": platform,
                    "_exp_dir": data.get("exp_id", exp_dir.name),
                    "exp": data.get("exp_id", exp_dir.name),
                    "acc_final": _safe_float(fp32.get("acc_final")),
                    "avg_forgetting": _safe_float(fp32.get("avg_forgetting")),
                    "ram_bytes": _safe_float(fp32.get("ram_peak_bytes")),
                    "auroc": _safe_float(fp32.get("auroc_final")),
                })
                # EWC INT8
                int8 = data.get("int8", {})
                records.append({
                    "model": "ewc_int8", "dataset": ds, "platform": platform,
                    "_exp_dir": data.get("exp_id", exp_dir.name),
                    "exp": data.get("exp_id", exp_dir.name),
                    "acc_final": _safe_float(int8.get("acc_final")),
                    "avg_forgetting": _safe_float(int8.get("avg_forgetting")),
                    "ram_bytes": _safe_float(int8.get("ram_peak_bytes")),
                    "auroc": _safe_float(int8.get("auroc_final")),
                    "gap3_criterion_met": data.get("gap3_criterion_met"),
                })
        except Exception:
            pass

    return records


def _infer_model_from_results_dir(results_dir: Path) -> str | None:
    """Infère le modèle depuis les fichiers présents dans results/."""
    files = {f.name for f in results_dir.iterdir()}
    if "acc_matrix_hdc.npy" in files:
        return "hdc"
    if "acc_matrix_ewc.npy" in files:
        return "ewc"
    if "acc_matrix_tinyol.npy" in files or "acc_matrix_naive.npy" in files:
        return "tinyol"
    return None


def _load_memory_report(results_dir: Path) -> dict:
    """Extrait RAM + latence depuis memory_report.json."""
    f = results_dir / "memory_report.json"
    if not f.exists():
        return {}
    try:
        mem = json.loads(f.read_text())
        forward = mem.get("forward", {})
        update = mem.get("update", {})
        result = {}
        ram = forward.get("ram_peak_bytes") or update.get("ram_peak_bytes_update")
        lat = forward.get("inference_latency_ms")
        if ram:
            result["ram_bytes"] = _safe_float(ram)
        if lat:
            result["latency_ms"] = _safe_float(lat)
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Chargement Sprint 21 JSON
# ---------------------------------------------------------------------------

def _load_sprint21_json(path: Path) -> list[dict]:
    """Charge comparison_sprint21.json et injecte les cellules dans des records."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []

    records: list[dict] = []
    for dataset, model_cells in data.get("results", {}).items():
        for model, cell in model_cells.items():
            if not isinstance(cell, dict):
                continue
            exp_id = cell.get("exp")
            if exp_id is None:
                continue
            # Inférer platform : les exp_S19/S21 sont toutes board
            platform = "nucleo_f439zi"
            r: dict = {
                "model":    _normalize_model(model),
                "dataset":  _normalize_dataset(dataset),
                "platform": platform,
                "_exp_dir": exp_id,
                **cell,
            }
            # Renommer ram_bytes → ram_bytes (déjà OK), latency_ms → latency_ms
            if "latency_ms" not in r and "inference_latency_ms" in cell:
                r["latency_ms"] = cell["inference_latency_ms"]
            if "ram_bytes" not in r and "ram_peak_bytes" in cell:
                r["ram_bytes"] = cell["ram_peak_bytes"]
            records.append(r)
    return records


# ---------------------------------------------------------------------------
# Chargement Sprint 32 (board threshold sweep — online_accuracy mesurée)
# ---------------------------------------------------------------------------

# Seuil de référence par dataset (24 % du RUL_CAP — voir
# docs/sprints/sprint_32/S3200_sprint_32.md). C'est le « native config
# threshold » retenu pour représenter chaque cellule du heatmap.
_S32_REFERENCE_THRESHOLD = {
    "cmapss":    30,
    "pronostia": 72,
}

_S32_SWEEP_SUMMARY = Path("experiments/exp_S32_board_sweep_summary.json")


def _load_s32_board_sweep(path: Path) -> list[dict]:
    """Charge le balayage de seuil board Sprint 32 (acc_final = online_accuracy).

    Ne conserve que le seuil de référence par dataset (``_S32_REFERENCE_THRESHOLD``)
    et expose ``online_accuracy`` comme ``acc_final`` board. Les datasets hors
    heatmap (ex. battery) ou sans seuil de référence sont ignorés.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []

    records: list[dict] = []
    for r in data:
        ds = _normalize_dataset(r.get("dataset"))
        ref_thr = _S32_REFERENCE_THRESHOLD.get(ds)
        if ref_thr is None or r.get("threshold") != ref_thr:
            continue
        acc = _safe_float(r.get("online_accuracy"))
        if acc is None:
            continue
        lat_us = r.get("latency_us_p50")
        records.append({
            "model":      _normalize_model(r.get("model")),
            "dataset":    ds,
            "platform":   "nucleo_f439zi",
            "_exp_dir":   r.get("exp_id"),
            "exp":        r.get("exp_id"),
            "acc_final":  acc,
            "latency_ms": (_safe_float(lat_us) / 1000.0) if lat_us is not None else None,
            "ram_bytes":  _safe_float(r.get("ram_response_bytes") or r.get("bss_bytes")),
            "note":       f"online_accuracy board, seuil réf. {ref_thr} (S32)",
        })
    return records


# ---------------------------------------------------------------------------
# Chargement Sprint 33 (PC CL acc_final + board gap1 HW-only)
# ---------------------------------------------------------------------------

_S33_PC_GLOB    = "exp_S33_PC_*"
_S33_BOARD_GLOB = "exp_S33_board_gap1"


def _load_s33_pc(experiments_dir: Path) -> list[dict]:
    """Charge les runs PC CL Sprint 33 (``exp_S33_PC_{model}_{dataset}``).

    Comble les cellules PC vides du heatmap (cwru/monitoring/pronostia × 4 modèles).
    ``acc_final`` lu depuis ``results.json`` ; modèle/dataset déduits du nom de dossier.
    """
    records: list[dict] = []
    for exp_dir in sorted(experiments_dir.glob(_S33_PC_GLOB)):
        name = exp_dir.name[len("exp_S33_PC_"):]
        # split model_dataset : le dataset est le dernier token connu
        ds = next((d for d in _DATASETS if name.endswith(d)), None)
        if ds is None:
            continue
        model = _normalize_model(name[: -(len(ds) + 1)])
        rj = exp_dir / "results.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        acc = _safe_float(r.get("acc_final"))
        if acc is None:
            continue
        records.append({
            "model": model, "dataset": ds, "platform": "pc",
            "_exp_dir": exp_dir.name, "exp": exp_dir.name,
            "acc_final": acc,
            "avg_forgetting": _safe_float(r.get("avg_forgetting")),
            "note": "PC CL acc_final (S33)",
        })
    return records


def _load_s33_board_gap1(experiments_dir: Path) -> list[dict]:
    """Charge la campagne board Sprint 33 (``exp_S33_board_gap1/results_*.json``).

    Comble les cellules board HW-only (TinyOL/HDC × cwru/paderborn/monitoring).
    ``acc_final = online_accuracy``. Absent tant que la board n'a pas été streamée.
    """
    gap1 = experiments_dir / _S33_BOARD_GLOB
    if not gap1.is_dir():
        return []
    records: list[dict] = []
    for rj in sorted(gap1.glob("results_*.json")):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        ds = _normalize_dataset(r.get("dataset"))
        model = _normalize_model(r.get("model"))
        acc = _safe_float(r.get("online_accuracy"))
        if ds is None or model is None or acc is None:
            continue
        lat_us = r.get("latency_us_p50")
        records.append({
            "model": model, "dataset": ds, "platform": "nucleo_f439zi",
            "_exp_dir": r.get("exp_id", gap1.name), "exp": r.get("exp_id", gap1.name),
            "acc_final": acc,
            "latency_ms": (_safe_float(lat_us) / 1000.0) if lat_us is not None else None,
            "ram_bytes": _safe_float(r.get("ram_response_bytes") or r.get("bss_bytes")),
            "note": r.get("parity_note") or "online_accuracy board HW-only (S33)",
        })
    return records


# ---------------------------------------------------------------------------
# Chargement Sprint 23 (board, pending)
# ---------------------------------------------------------------------------

def _load_s23_exp(exp_dir: Path) -> list[dict]:
    """Charge un répertoire Sprint 23 — avec ou sans results.json."""
    cfg = _load_config_snapshot(exp_dir)
    dataset  = _dataset_from_config(cfg)
    model    = _model_from_config(cfg)
    platform = _platform_from_config(cfg) or "nucleo_f439zi"

    # Essayer de charger results.json
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            r = json.loads(results_file.read_text())
            # Cas INT8 board
            if r.get("model_fp32") and r.get("model_int8"):
                ds = _normalize_dataset(r.get("dataset")) or dataset
                records = []
                if ds:
                    records.append({
                        "model": "ewc", "dataset": ds, "platform": platform,
                        "_exp_dir": exp_dir.name,
                        "exp": exp_dir.name,
                        "ram_bytes": _safe_float(r.get("ram_fp32_bytes")),
                        "latency_ms": _safe_float(r.get("latency_fp32_ms")),
                        "note": "exp_S23_INT8 — FP32 branch",
                    })
                    records.append({
                        "model": "ewc_int8", "dataset": ds, "platform": platform,
                        "_exp_dir": exp_dir.name,
                        "exp": exp_dir.name,
                        "ram_bytes": _safe_float(r.get("ram_int8_bytes")),
                        "latency_ms": _safe_float(r.get("latency_int8_ms")),
                        "note": "exp_S23_INT8 — INT8 branch (Gap 3)",
                        "gap3_note": r.get("gap3_note"),
                    })
                return records

            # Cas benchmark (cwru_proxy)
            ds = _normalize_dataset(r.get("dataset")) or dataset
            m  = _normalize_model(r.get("model")) or model
            p  = _normalize_platform(r.get("platform")) or platform
            if ds and m:
                cell = _cell_from_flat(r, exp_dir.name)
                note = r.get("note_scenario") or r.get("note")
                if note:
                    cell["note"] = note
                return [{
                    "model": m, "dataset": ds, "platform": p,
                    "_exp_dir": exp_dir.name, **cell,
                }]
        except Exception:
            pass

    # Pas de résultats — créer une entrée "pending" depuis config_snapshot
    if dataset and model:
        return [{
            "model":    model,
            "dataset":  dataset,
            "platform": platform,
            "_exp_dir": exp_dir.name,
            "exp":      exp_dir.name,
            "note":     "pending — board experiment not yet executed",
        }]
    return []


# ---------------------------------------------------------------------------
# Loader principal
# ---------------------------------------------------------------------------

def _load_all_records(exp_dirs: list[Path]) -> list[dict]:
    """Charge tous les enregistrements depuis les répertoires + Sprint 21 JSON."""
    records: list[dict] = []

    # Source 0 : Sprint 21 comparison JSON (cwru/monitoring/pronostia board)
    s21_records = _load_sprint21_json(_SPRINT21_JSON)
    records.extend(s21_records)
    if s21_records:
        print(f"  Sprint 21 JSON : {len(s21_records)} cellule(s) chargée(s)")

    # Source 0bis : Sprint 32 board threshold sweep (cmapss/pronostia board réel)
    s32_records = _load_s32_board_sweep(_S32_SWEEP_SUMMARY)
    records.extend(s32_records)
    if s32_records:
        print(f"  Sprint 32 board sweep : {len(s32_records)} cellule(s) chargée(s)")

    # Source 0ter : Sprint 33 PC CL (cwru/monitoring/pronostia × 4 modèles)
    s33_pc = _load_s33_pc(Path("experiments"))
    records.extend(s33_pc)
    if s33_pc:
        print(f"  Sprint 33 PC CL : {len(s33_pc)} cellule(s) chargée(s)")

    # Source 0quater : Sprint 33 board gap1 HW-only (TinyOL/HDC × cwru/paderborn/monitoring)
    s33_board = _load_s33_board_gap1(Path("experiments"))
    records.extend(s33_board)
    if s33_board:
        print(f"  Sprint 33 board gap1 : {len(s33_board)} cellule(s) chargée(s)")

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            print(f"  [!] Absent : {exp_dir}")
            continue

        # Sprint 22 : présence du sous-répertoire results/
        if (exp_dir / "results").is_dir():
            new = _load_s22_nested(exp_dir)
            if new:
                records.extend(new)
                print(f"  {exp_dir.name} (S22 nested) : {len(new)} enreg.")
            else:
                print(f"  {exp_dir.name} (S22 nested) : aucun résultat parsé")
            continue

        # Sprint 21 / 18-19 style : results*.json à la racine
        root_files = sorted(exp_dir.glob("results*.json"))
        if root_files:
            for f in root_files:
                try:
                    r = json.loads(f.read_text())
                    r["_exp_dir"] = exp_dir.name
                    r["_file"]    = str(f)
                    # Normaliser dataset/model/platform
                    r["dataset"]  = _normalize_dataset(r.get("dataset"))
                    r["model"]    = _normalize_model(r.get("model"))
                    r["platform"] = _normalize_platform(r.get("platform"))
                    if r.get("dataset") and r.get("model") and r.get("acc_final") is not None:
                        records.append(r)
                except Exception:
                    pass
            if root_files:
                print(f"  {exp_dir.name} (root results) : {len(root_files)} fichier(s)")
            continue

        # Sprint 23 style : config_snapshot + éventuel results.json
        new = _load_s23_exp(exp_dir)
        if new:
            records.extend(new)
            print(f"  {exp_dir.name} (S23) : {len(new)} enreg. ({new[0].get('note','metrics')})")
        else:
            print(f"  {exp_dir.name} : vide")

    return records


# ---------------------------------------------------------------------------
# Construction de la comparaison
# ---------------------------------------------------------------------------

def _cell_from_record(r: dict, exp_dir_name: str) -> dict:
    """Construit la cellule finale à insérer dans le JSON de comparaison."""
    cell: dict = {"exp": r.get("exp") or exp_dir_name}
    for src, dst in [
        ("acc_final",           "acc_final"),
        ("avg_forgetting",      "avg_forgetting"),
        ("backward_transfer",   "backward_transfer"),
        ("latency_ms",          "latency_ms"),
        ("inference_latency_ms","latency_ms"),
        ("ram_bytes",           "ram_bytes"),
        ("ram_peak_bytes",      "ram_bytes"),
        ("lambda_ewc",          "lambda_ewc"),
        ("n_repetitions",       "n_repetitions"),
        ("gap2_ram_compliant",  "gap2_ram_compliant"),
        ("gap2_latency_compliant", "gap2_latency_compliant"),
        ("auroc",               "auroc"),
        ("gap3_note",           "gap3_note"),
        ("note",                "note"),
    ]:
        val = r.get(src)
        if val is not None and dst not in cell:
            cell[dst] = val

    # Forcer les valeurs numériques (certains champs peuvent être des strings)
    for numeric_key in ("acc_final", "avg_forgetting", "backward_transfer",
                        "latency_ms", "ram_bytes", "lambda_ewc"):
        if numeric_key in cell:
            cell[numeric_key] = _safe_float(cell[numeric_key])

    # Calcul conformité Gap 2 si non déjà présent
    ram = cell.get("ram_bytes")
    lat = cell.get("latency_ms")
    if isinstance(ram, (int, float)) and "gap2_ram_compliant" not in cell:
        cell["gap2_ram_compliant"] = bool(ram < _GAP2_RAM_BUDGET)
    if isinstance(lat, (int, float)) and "gap2_latency_compliant" not in cell:
        cell["gap2_latency_compliant"] = bool(lat < _GAP2_LATENCY_MS)

    return cell


def _better_cell(existing: dict, new: dict) -> bool:
    """Retourne True si new est meilleure qu'existing."""
    new_acc = new.get("acc_final")
    old_acc = existing.get("acc_final")
    if new_acc is not None and (old_acc is None or new_acc > old_acc):
        return True
    # Préférer la cellule avec note "pending" seulement si l'autre est aussi vide
    if new_acc is None and old_acc is None:
        return False
    return False


def _gap_summary(index: dict) -> dict:
    """Calcule le résumé des gaps depuis l'index 3D."""
    board_cells = []
    for dataset in _DATASETS:
        for model in _MODELS:
            cell = index.get((dataset, model, "nucleo_f439zi"), {})
            if cell and cell.get("acc_final") is not None:
                board_cells.append(cell)

    ram_vals = [c["ram_bytes"] for c in board_cells if c.get("ram_bytes")]
    lat_vals = [c["latency_ms"] for c in board_cells if c.get("latency_ms")]

    gap2_compliant = [c for c in board_cells
                      if c.get("gap2_ram_compliant") or c.get("gap2_latency_compliant")]

    return {
        "gap1_datasets": _DATASETS,
        "gap1_datasets_count": len(_DATASETS),
        "gap2_ram_budget_bytes": _GAP2_RAM_BUDGET,
        "gap2_latency_budget_ms": _GAP2_LATENCY_MS,
        "gap2_board_results_count": len(board_cells),
        "gap2_compliant_count": len(gap2_compliant),
        "gap2_ram_max_observed_bytes": max(ram_vals) if ram_vals else None,
        "gap2_latency_max_observed_ms": max(lat_vals) if lat_vals else None,
        "gap3_int8_exp": "exp_S23_INT8",
        "gap3_int8_ram_reduction": "2.7x (9728 → 3600 bytes, exp_S22_INT8_01)",
    }


def build_comparison(records: list[dict]) -> dict:
    """Construit le JSON 3D datasets × models × platforms."""
    # Index 3D : (dataset, model, platform) → meilleure cellule
    index: dict[tuple[str, str, str], dict] = {}

    for r in records:
        dataset  = r.get("dataset")
        model    = r.get("model")
        platform = r.get("platform") or "pc"

        if dataset not in _DATASETS or model not in _MODELS or platform not in _PLATFORMS:
            continue

        key  = (dataset, model, platform)
        cell = _cell_from_record(r, r.get("_exp_dir", "?"))

        existing = index.get(key)
        if existing is None or _better_cell(existing, cell):
            index[key] = cell

    # Structure hiérarchique
    results: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset in _DATASETS:
        results[dataset] = {}
        for model in _MODELS:
            results[dataset][model] = {}
            for platform in _PLATFORMS:
                key  = (dataset, model, platform)
                cell = index.get(key, {"exp": None, "note": "pending"})
                results[dataset][model][platform] = cell

    return {
        "metadata": {
            "sprint":           23,
            "date":             str(date.today()),
            "datasets":         _DATASETS,
            "models":           _MODELS,
            "platforms":        _PLATFORMS,
            "gap2_budget_bytes": _GAP2_RAM_BUDGET,
            "source":           "generate_comparison_sprint23.py",
        },
        "results":     results,
        "gap_summary": _gap_summary(index),
    }


# ---------------------------------------------------------------------------
# Sprint 35 — résultats par condition de features (F1 + acc_final)
# ---------------------------------------------------------------------------

_S35_MODELS = ["mahalanobis", "ewc", "tinyol", "hdc"]
_S35_CONDITIONS = ["5feat", "all", "best"]


def _num_or_none(v: Any) -> float | None:
    """Convertit une valeur en float, ou None si « à mesurer »/absente/non numérique."""
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _load_s35_conditions(root: Path) -> dict:
    """Ingère exp_S35_PC_* et exp_S35_board_* → results_by_condition[cond][ds][model][platform].

    Chaque cellule : ``{acc_final, f1_faulty, exp, note}`` (None si « à mesurer »/absent).
    PC ← ``acc_final`` ; board ← ``online_accuracy`` (S3508). Aucune valeur inventée :
    les champs board non mesurés restent None (masqués « pending » dans la heatmap).
    """
    by_cond: dict[str, dict] = {
        c: {d: {m: {p: {"exp": None, "acc_final": None, "f1_faulty": None, "note": "pending"}
                     for p in _PLATFORMS}
                for m in _S35_MODELS}
            for d in _DATASETS}
        for c in _S35_CONDITIONS
    }

    def _ingest(path: Path, platform: str, acc_key: str) -> None:
        try:
            r = json.loads(path.read_text())
        except Exception:
            return
        cond, ds, model = r.get("condition"), r.get("dataset"), r.get("model")
        if cond not in _S35_CONDITIONS or ds not in _DATASETS or model not in _S35_MODELS:
            return
        by_cond[cond][ds][model][platform] = {
            "exp": r.get("exp_id") or path.parent.name,
            "acc_final": _num_or_none(r.get(acc_key)),
            "f1_faulty": _num_or_none(r.get("f1_faulty")),
            "n_features": r.get("n_features"),
            "note": r.get("note") or r.get("parity_note") or "",
        }

    for d in sorted(root.glob("exp_S35_PC_*")):
        _ingest(d / "results.json", "pc", "acc_final")
    for d in sorted(root.glob("exp_S35_board_*")):
        _ingest(d / "results.json", "nucleo_f439zi", "online_accuracy")

    n = sum(1 for c in _S35_CONDITIONS for ds in _DATASETS for m in _S35_MODELS
            for p in _PLATFORMS if by_cond[c][ds][m][p]["acc_final"] is not None)
    print(f"  Sprint 35 conditions : {n} cellule(s) mesurée(s) ingérée(s)")
    return by_cond


def _apply_s3509_override(results: dict, by_cond: dict) -> None:
    """S3509 : remplace l'artefact HDC×monitoring board (0.1133) par la valeur mesurée.

    Source = re-run board corrigé (condition `all`, monitoring natif 4-feat, sans padding).
    """
    for cond in ("all", "5feat"):
        cell = by_cond.get(cond, {}).get("monitoring", {}).get("hdc", {}).get("nucleo_f439zi", {})
        acc = cell.get("acc_final")
        if acc is not None:
            legacy = results.get("monitoring", {}).get("hdc", {}).get("nucleo_f439zi")
            if isinstance(legacy, dict):
                legacy["acc_final"] = acc
                legacy["exp"] = cell.get("exp")
                legacy["note"] = (
                    "S3509 : valeur mesurée board (monitoring natif 4-feat, sans zéro-padding) "
                    "— remplace l'artefact 0.1133")
            print(f"  S3509 : monitoring/hdc/board acc_final → {acc:.4f} (condition {cond})")
            return
    print("  S3509 : pas de re-run board HDC×monitoring trouvé → artefact inchangé")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère comparison_sprint23.json (5 datasets × 5 modèles × 2 plateformes)")
    parser.add_argument(
        "--exps", nargs="*", type=Path, default=None, metavar="DIR",
        help="Répertoires d'expériences (défaut : liste interne)")
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/comparison_sprint23.json"),
        help="Fichier JSON de sortie")
    parser.add_argument(
        "--no-sprint21", action="store_true",
        help="Ne pas charger comparison_sprint21.json comme source")
    args = parser.parse_args()

    exp_dirs = [Path(d) for d in _DEFAULT_EXP_DIRS] if args.exps is None else args.exps

    print(f"Chargement depuis {len(exp_dirs)} répertoire(s)…")
    if args.no_sprint21:
        global _SPRINT21_JSON
        _SPRINT21_JSON = Path("__disabled__")
    records = _load_all_records(exp_dirs)
    print(f"  Total : {len(records)} enregistrement(s) chargé(s)\n")

    comparison = build_comparison(records)

    # Sprint 35 — conditions de features (F1 + acc_final par condition) + fix S3509.
    by_cond = _load_s35_conditions(Path("experiments"))
    comparison["results_by_condition"] = by_cond
    comparison["metadata"]["conditions"] = _S35_CONDITIONS
    _apply_s3509_override(comparison["results"], by_cond)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON sauvé : {args.output}")

    # Résumé terminal
    print("\n--- Résumé datasets × models × platforms ---")
    for dataset in _DATASETS:
        for model in _MODELS:
            for platform in _PLATFORMS:
                cell = comparison["results"][dataset][model][platform]
                acc  = cell.get("acc_final", "—")
                exp  = cell.get("exp") or "null"
                note = cell.get("note", "")
                if acc != "—" or exp != "null":
                    tag = f"  {dataset:12s} / {model:10s} / {platform:14s}"
                    print(f"{tag} exp={exp:30s} acc={acc}"
                          + (f"  [{note}]" if note else ""))

    gap = comparison["gap_summary"]
    print(f"\nGap 1 : {gap['gap1_datasets_count']} datasets — {gap['gap1_datasets']}")
    print(f"Gap 2 : {gap['gap2_compliant_count']}/{gap['gap2_board_results_count']}"
          f" board résultats conformes")
    if gap.get("gap2_ram_max_observed_bytes"):
        print(f"        RAM max : {gap['gap2_ram_max_observed_bytes']} B"
              f" / budget {gap['gap2_ram_budget_bytes']} B")
    print(f"Gap 3 : {gap['gap3_int8_ram_reduction']}")


if __name__ == "__main__":
    main()
