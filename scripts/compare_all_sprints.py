"""
compare_all_sprints.py — Agrégation historique cross-Sprint 1–24.

Parcourt tous les dossiers experiments/exp_*/ et produit un tableau unifié
JSON + CSV compatible avec le notebook 24_comprehensive_comparison.ipynb.

Gère tous les formats de results.json rencontrés depuis Sprint 1 :
  - Sprint 1–17 : results/metrics.json avec cl_metrics[model] imbriqué
  - Sprint 18+  : results.json plat ou results/results.json
  - Sprint 22 INT8 : results/results.json avec fp32/int8 sub-dicts
  - Sprint 23   : placeholders "à mesurer" → None
  - Sprint 24   : results.json plat à la racine de exp_dir

Usage :
    python scripts/compare_all_sprints.py \\
      --exp_dir experiments/ \\
      --output_json experiments/comparison_sprint24.json \\
      --output_csv experiments/comparison_sprint24.csv

    python scripts/compare_all_sprints.py \\
      --exp_dir experiments/ \\
      --sprint_filter S24 \\
      --output_json experiments/comparison_sprint24_only.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import csv
    _CSV_AVAILABLE = True
except ImportError:
    _CSV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colonnes CSV de sortie (ordre canonique)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "exp_id", "sprint", "model", "dataset", "scenario", "uint8_activations",
    "acc_final", "avg_forgetting", "bwt", "auroc",
    "ram_peak_bytes", "ram_peak_kb", "gap2_compliant",
    "inference_latency_ms", "n_params",
    "compression_ratio", "delta_acc_vs_fp32",
    "reference_exp", "notes",
]


# ---------------------------------------------------------------------------
# Normalisation des valeurs
# ---------------------------------------------------------------------------

def _safe_value(v: Any) -> float | None:
    """Convertit une valeur en float, retourne None pour les placeholders."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("à mesurer", "a mesurer", "tbd", "pending", "—", "-", "", "n/a"):
            return None
        # Gérer les formats "< 1.0", "<= 65536", ">= 0.9"
        m = re.match(r"^[<>]=?\s*([\d.]+)", s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _normalize_model(model: str | None) -> str | None:
    if model is None:
        return None
    m = model.lower().strip()
    if m in ("mahalanobis", "mahalanobis_detector"):
        return "mahalanobis"
    if m in ("ewc", "ewc_head", "ewc_online", "ewc_mlp"):
        return "ewc"
    if m in ("ewc_int8", "ewc_head_int8", "ewc_mlp_int8"):
        return "ewc_int8"
    if m in ("tinyol", "tinyol_autoencoder", "tinyol_anomaly"):
        return "tinyol"
    if m in ("hdc", "hdc_classifier", "hdc_online"):
        return "hdc"
    if m in ("kmeans", "kmeans_detector"):
        return "kmeans"
    if m in ("dbscan", "dbscan_detector"):
        return "dbscan"
    if m in ("knn", "knn_detector"):
        return "knn"
    return m


def _normalize_dataset(dataset: str | None) -> str | None:
    if dataset is None:
        return None
    d = dataset.lower().strip()
    if d in ("cwru", "cwru_bearing", "cwru_proxy"):
        return "cwru"
    if d in ("monitoring", "equipment_monitoring", "industrial_equipment_monitoring"):
        return "monitoring"
    if d in ("pronostia", "femto", "pronostia_femto"):
        return "pronostia"
    if d in ("pump", "pump_maintenance"):
        return "pump"
    if d in ("cmapss", "nasa_cmapss"):
        return "cmapss"
    if d in ("paderborn", "paderborn_bearing"):
        return "paderborn"
    if d in ("battery",):
        return "battery"
    return d


# ---------------------------------------------------------------------------
# Extraction du numéro de sprint depuis exp_id
# ---------------------------------------------------------------------------

def _sprint_from_exp_id(exp_id: str) -> int | str | None:
    """
    Exemples :
      "exp_001_ewc_…"      → 1
      "exp_S22_01"         → 22
      "exp_S24_04"         → 24
      "tinyol_monitoring"  → None
    """
    # Format exp_SXX_YY
    m = re.match(r"exp_[Ss](\d+)_", exp_id)
    if m:
        return int(m.group(1))
    # Format exp_NNN_… (sprint inféré depuis numéro d'expérience)
    m = re.match(r"exp_(\d+)_", exp_id)
    if m:
        n = int(m.group(1))
        # Mapping approximatif numéro d'expérience → sprint
        if n <= 10:
            return 1
        if n <= 30:
            return 6
        if n <= 50:
            return 8
        if n <= 80:
            return 10
        if n <= 120:
            return 12
        if n <= 160:
            return 15
        return None
    return None


# ---------------------------------------------------------------------------
# Chargement multi-format des résultats
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_from_cl_metrics(data: dict, exp_id: str) -> dict | None:
    """Extrait les métriques du format Sprint 1–17 (cl_metrics imbriqué)."""
    cl = data.get("cl_metrics")
    if not isinstance(cl, dict):
        return None
    # Trouver la clé modèle (ewc, hdc, tinyol, mahalanobis, …)
    model_key = None
    for k in ("ewc", "hdc", "tinyol", "mahalanobis", "naive", "joint"):
        if k in cl:
            model_key = k
            break
    if model_key is None:
        # Prendre la première clé non-"memory"
        for k, v in cl.items():
            if k != "memory" and isinstance(v, dict):
                model_key = k
                break
    if model_key is None:
        return None
    m = cl[model_key]
    mem = cl.get("memory", {}).get("forward", {})
    result: dict = {
        "exp_id": data.get("exp_id", exp_id),
        "model": _normalize_model(model_key),
        "acc_final": _safe_value(m.get("aa")),
        "avg_forgetting": _safe_value(m.get("af")),
        "bwt": _safe_value(m.get("bwt")),
        "ram_peak_bytes": _safe_value(mem.get("ram_peak_bytes")),
        "inference_latency_ms": _safe_value(mem.get("inference_latency_ms")),
        "n_params": _safe_value(mem.get("n_params")),
    }
    return result


def _extract_from_flat(data: dict, exp_id: str) -> dict:
    """Extrait depuis un dict plat (format Sprint 18+)."""
    # Gérer les sous-dicts fp32/int8 (Sprint 22 INT8)
    if "fp32" in data and isinstance(data["fp32"], dict):
        fp32 = data["fp32"]
        int8 = data.get("int8", {})
        result: dict = {
            "exp_id": data.get("exp_id", exp_id),
            "model": _normalize_model(data.get("model")),
            "dataset": _normalize_dataset(data.get("dataset")),
            "acc_final": _safe_value(fp32.get("acc_final")),
            "avg_forgetting": _safe_value(fp32.get("avg_forgetting")),
            "bwt": _safe_value(fp32.get("backward_transfer") or fp32.get("bwt")),
            "auroc": _safe_value(fp32.get("auroc_final") or fp32.get("auroc")),
            "ram_peak_bytes": _safe_value(fp32.get("ram_peak_bytes")),
            "inference_latency_ms": _safe_value(fp32.get("inference_latency_ms")),
            "n_params": _safe_value(data.get("n_params")),
            "int8_acc_final": _safe_value(int8.get("acc_final")),
            "int8_ram_peak_bytes": _safe_value(int8.get("ram_peak_bytes")),
            "delta_auroc": _safe_value(data.get("delta_auroc")),
            "gap3_criterion_met": data.get("gap3_criterion_met"),
        }
        return result

    # Format plat standard
    result = {
        "exp_id": data.get("exp_id", exp_id),
        "model": _normalize_model(data.get("model")),
        "dataset": _normalize_dataset(data.get("dataset")),
        "scenario": data.get("scenario") or data.get("task_split"),
        "sprint": data.get("sprint"),
        "uint8_activations": data.get("uint8_activations"),
        "acc_final": _safe_value(data.get("acc_final") or data.get("aa")),
        "avg_forgetting": _safe_value(
            data.get("avg_forgetting") or data.get("af") or data.get("avg_forget")
        ),
        "bwt": _safe_value(
            data.get("backward_transfer") or data.get("bwt")
        ),
        "auroc": _safe_value(data.get("auroc_final") or data.get("auroc")),
        "ram_peak_bytes": _safe_value(
            data.get("ram_peak_bytes")
            or data.get("ram_bytes")
            or (_safe_value(data.get("ram_kb")) * 1024 if data.get("ram_kb") is not None else None)
        ),
        "inference_latency_ms": _safe_value(
            data.get("inference_latency_ms")
            or data.get("latency_ms")
            or data.get("lat_ms")
        ),
        "n_params": _safe_value(data.get("n_params")),
        "compression_ratio": _safe_value(data.get("compression_ratio")),
        "delta_acc_vs_fp32": _safe_value(
            data.get("delta_acc_vs_fp32") or data.get("delta_aa_vs_fp32")
        ),
        "reference_exp": data.get("reference_exp") or data.get("reference_exp_fp32"),
        "notes": data.get("notes") or data.get("note"),
        "gap2_compliant": (
            data.get("gap2_compliant")
            or data.get("gap2_ram_compliant")
            or data.get("gap2")
        ),
        "ram_training_peak_bytes": _safe_value(data.get("ram_training_peak_bytes")),
    }
    return result


def load_experiment(exp_dir: Path) -> dict | None:
    """
    Charge les résultats d'un dossier expérience.

    Stratégie multi-format (ordre de priorité) :
    1. exp_dir/results.json (plat, Sprint 18+ et Sprint 24)
    2. exp_dir/results/results.json (nested, Sprint 22+)
    3. exp_dir/results/metrics_cl.json (format unifié S12-05)
    4. exp_dir/results/metrics.json → cl_metrics (Sprint 1–17)
    5. exp_dir/results/*.json (glob, premier fichier)

    Retourne None si aucun résultat valide trouvé.
    """
    exp_id = exp_dir.name
    candidates = [
        exp_dir / "results.json",
        exp_dir / "results" / "results.json",
        exp_dir / "results" / "metrics_cl.json",
        exp_dir / "results" / "metrics.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = _load_json(path)
        if data is None:
            continue
        # Cas : redirect vers un autre fichier
        if isinstance(data, dict) and "report_path" in data and len(data) == 1:
            redirect = Path(data["report_path"])
            if redirect.exists():
                data = _load_json(redirect)
                if data is None:
                    continue
        if not isinstance(data, dict):
            continue
        # Cas Sprint 1–17 : cl_metrics imbriqué
        if "cl_metrics" in data:
            result = _extract_from_cl_metrics(data, exp_id)
            if result is None:
                continue
        else:
            result = _extract_from_flat(data, exp_id)
        # Toujours forcer exp_id depuis le nom du dossier si absent
        if not result.get("exp_id"):
            result["exp_id"] = exp_id
        return result

    # Fallback : glob sur tous les *.json dans results/
    results_subdir = exp_dir / "results"
    if results_subdir.exists():
        for path in sorted(results_subdir.glob("*.json")):
            data = _load_json(path)
            if isinstance(data, dict) and (
                "acc_final" in data or "aa" in data or "cl_metrics" in data
            ):
                if "cl_metrics" in data:
                    result = _extract_from_cl_metrics(data, exp_id)
                    if result:
                        result.setdefault("exp_id", exp_id)
                        return result
                else:
                    result = _extract_from_flat(data, exp_id)
                    result.setdefault("exp_id", exp_id)
                    return result
    return None


# ---------------------------------------------------------------------------
# Dérivation des champs manquants
# ---------------------------------------------------------------------------

def _enrich(record: dict) -> dict:
    """Dérive les champs calculés manquants."""
    # sprint depuis exp_id si absent
    if record.get("sprint") is None:
        record["sprint"] = _sprint_from_exp_id(record.get("exp_id", ""))

    # ram_peak_kb
    ram = record.get("ram_peak_bytes")
    if ram is not None:
        record["ram_peak_kb"] = round(ram / 1024, 2)
    else:
        record["ram_peak_kb"] = None

    # gap2_compliant : ram ≤ 65 536 B et latence ≤ 100 ms
    if record.get("gap2_compliant") is None:
        ram_ok = ram is not None and ram <= 65_536
        lat = record.get("inference_latency_ms")
        lat_ok = lat is not None and lat <= 100.0
        if ram is not None or lat is not None:
            record["gap2_compliant"] = bool(ram_ok and lat_ok)

    # Normaliser les booléens "✅" / "❌"
    for k in ("gap2_compliant", "uint8_activations"):
        v = record.get(k)
        if isinstance(v, str):
            record[k] = v.strip() in ("true", "✅", "1", "yes", "oui")

    return record


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------

def aggregate(exp_dir: Path, sprint_filter: str | None = None) -> list[dict]:
    """
    Parcourt exp_dir/exp_*/ et charge chaque expérience.

    Parameters
    ----------
    exp_dir : Path
        Répertoire contenant les dossiers exp_*.
    sprint_filter : str | None
        Si fourni, ne conserve que les expériences dont exp_id contient ce filtre
        (ex. "S24" → garde exp_S24_*, "S22" → garde exp_S22_*).

    Returns
    -------
    list[dict]
        Liste de records normalisés, triés par exp_id.
    """
    records: list[dict] = []
    exp_dirs = sorted(exp_dir.glob("exp_*"))

    for d in exp_dirs:
        if not d.is_dir():
            continue
        if sprint_filter and sprint_filter.lower() not in d.name.lower():
            continue
        record = load_experiment(d)
        if record is None:
            continue
        record = _enrich(record)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def _to_csv_row(record: dict) -> dict:
    """Produit un dict avec exactement les colonnes CSV (None → "")."""
    row: dict = {}
    for col in _CSV_COLUMNS:
        v = record.get(col)
        row[col] = "" if v is None else v
    return row


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(_to_csv_row(record))
    print(f"CSV → {path} ({len(records)} lignes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrégation historique cross-Sprint 1–24 de tous les experiments/"
    )
    parser.add_argument(
        "--exp_dir",
        default="experiments/",
        help="Répertoire racine des expériences (défaut : experiments/)",
    )
    parser.add_argument(
        "--output_json",
        default="experiments/comparison_sprint24.json",
        help="Chemin de sortie JSON",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Chemin de sortie CSV (optionnel)",
    )
    parser.add_argument(
        "--sprint_filter",
        default=None,
        help="Filtre par sprint dans le nom de exp_id (ex. S24 → seulement exp_S24_*)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    if not exp_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {exp_dir}")

    print(f"Chargement des expériences depuis {exp_dir} …")
    records = aggregate(exp_dir, sprint_filter=args.sprint_filter)

    print(f"Expériences chargées : {len(records)}")
    models = sorted({r.get("model") for r in records if r.get("model")})
    datasets = sorted({r.get("dataset") for r in records if r.get("dataset")})
    sprints = sorted({r.get("sprint") for r in records if r.get("sprint") is not None})
    print(f"Modèles  : {models}")
    print(f"Datasets : {datasets}")
    print(f"Sprints  : {sprints}")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": str(date.today()),
        "sprint": 24,
        "sprint_filter": args.sprint_filter,
        "n_experiments": len(records),
        "models": models,
        "datasets": datasets,
        "sprints": sprints,
        "experiments": records,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"JSON → {output_json}")

    if args.output_csv:
        write_csv(records, Path(args.output_csv))


if __name__ == "__main__":
    main()
