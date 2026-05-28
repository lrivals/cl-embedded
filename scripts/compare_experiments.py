"""
compare_experiments.py — Tableau comparatif 3 datasets × 3 modèles pour le manuscrit.

Lit les fichiers results*.json dans les répertoires spécifiés et génère
experiments/comparison_sprint21.json au format hiérarchique datasets × models.

Usage :
    python scripts/compare_experiments.py \\
        --exps experiments/exp_S21_01 experiments/exp_S21_02 \\
               experiments/exp_S21_03 experiments/exp_S21_04 \\
               experiments/exp_S19_01 experiments/exp_S19_02 \\
               experiments/exp_S18_01 \\
        --output experiments/comparison_sprint21.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


_DATASETS = ["cwru", "monitoring", "pronostia"]
_MODELS   = ["mahalanobis", "ewc", "tinyol"]

_GAP2_RAM_BUDGET = 65_536   # 64 Ko

# Cellules connues à ne pas remplacer par des entrées du fichier de résultats
# (dataset, model): {"exp": ..., "note": ...}
_STATIC_NOT_TESTED: dict[tuple[str, str], dict] = {
    ("cwru", "ewc"):       {"exp": None, "note": "non testé CWRU board"},
    ("pronostia", "tinyol"): {"exp": None, "note": "non prévu Sprint 21"},
}


def _load_results(exp_dirs: list[Path]) -> list[dict]:
    """Charge tous les results*.json depuis les répertoires donnés."""
    records: list[dict] = []
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            print(f"  [!] Répertoire absent : {exp_dir}")
            continue
        for json_file in sorted(exp_dir.glob("results*.json")):
            with open(json_file) as f:
                r = json.load(f)
            r["_file"]   = str(json_file)
            r["_exp_dir"] = exp_dir.name
            records.append(r)
    return records


def _normalize_model(model: str | None) -> str | None:
    """Normalise les noms de modèle vers les 3 catégories du projet."""
    if model is None:
        return None
    m = model.lower()
    if m in ("mahalanobis",):
        return "mahalanobis"
    if m in ("ewc", "ewc_head", "ewc_online"):
        return "ewc"
    if m in ("tinyol", "tiny_ol", "tinyol_oto"):
        return "tinyol"
    return model  # conserve tel quel (streaming_pipeline, etc.)


def _cell_from_record(r: dict, exp_dir_name: str) -> dict:
    """Construit une cellule résultat pour le JSON de comparaison."""
    cell: dict = {"exp": exp_dir_name}
    for src, dst in [
        ("acc_final",         "acc_final"),
        ("avg_forgetting",    "avg_forgetting"),
        ("backward_transfer", "backward_transfer"),
        ("inference_latency_ms", "latency_ms"),
        ("ram_peak_bytes",    "ram_bytes"),
        ("lambda_ewc",        "lambda_ewc"),
    ]:
        val = r.get(src)
        if val is not None:
            cell[dst] = val
    return cell


def _gap2_summary(cells: list[dict]) -> dict:
    """Calcule le résumé Gap 2 à partir des cellules collectées."""
    ram_vals = [c["ram_bytes"] for c in cells if c.get("ram_bytes")]
    all_compliant = all(v < _GAP2_RAM_BUDGET for v in ram_vals) if ram_vals else True
    return {
        "ram_budget_bytes":      _GAP2_RAM_BUDGET,
        "all_compliant":         all_compliant,
        "ram_max_observed_bytes": max(ram_vals) if ram_vals else 0,
    }


def build_comparison(records: list[dict]) -> dict:
    """Construit le JSON de comparaison datasets × models."""
    # Index (dataset, model) → meilleure cellule (la plus récente / avec acc_final non-null)
    index: dict[tuple[str, str], dict] = {}
    for r in records:
        dataset = r.get("dataset")
        model   = _normalize_model(r.get("model"))
        if dataset not in _DATASETS or model not in _MODELS:
            continue
        key = (dataset, model)
        cell = _cell_from_record(r, r["_exp_dir"])
        # Priorité : résultat avec acc_final renseigné > résultat sans
        existing = index.get(key)
        if existing is None:
            index[key] = cell
        else:
            # Préférer le résultat avec le meilleur acc_final
            new_acc = cell.get("acc_final")
            old_acc = existing.get("acc_final")
            if new_acc is not None and (old_acc is None or new_acc > old_acc):
                index[key] = cell

    # Construction de la structure hiérarchique
    results: dict[str, dict[str, dict]] = {}
    all_cells: list[dict] = []
    for dataset in _DATASETS:
        results[dataset] = {}
        for model in _MODELS:
            key = (dataset, model)
            if key in _STATIC_NOT_TESTED:
                results[dataset][model] = _STATIC_NOT_TESTED[key]
            elif key in index:
                results[dataset][model] = index[key]
                all_cells.append(index[key])
            else:
                results[dataset][model] = {"exp": None, "note": "Sprint 21 pending"}

    return {
        "generated": str(date.today()),
        "sprint":    21,
        "datasets":  _DATASETS,
        "models":    _MODELS,
        "results":   results,
        "gap2_summary": _gap2_summary(all_cells),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère comparison_sprint21.json (3 datasets × 3 modèles)")
    parser.add_argument("--exps", nargs="+", required=True, type=Path,
                        metavar="DIR", help="Répertoires experiments/exp_S2X_XX")
    parser.add_argument("--output", required=True, type=Path,
                        help="Fichier JSON de sortie")
    args = parser.parse_args()

    print(f"Chargement depuis {len(args.exps)} répertoire(s)…")
    records = _load_results(args.exps)
    print(f"  {len(records)} fichier(s) résultat chargé(s)")
    for r in records:
        model = _normalize_model(r.get("model"))
        print(f"  {r['_exp_dir']} — model={model} dataset={r.get('dataset')}"
              f" acc={r.get('acc_final')}")

    comparison = build_comparison(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nJSON sauvé : {args.output}")

    # Résumé terminal
    print("\n--- Résumé datasets × models ---")
    for dataset in _DATASETS:
        for model in _MODELS:
            cell = comparison["results"][dataset][model]
            acc  = cell.get("acc_final", "—")
            exp  = cell.get("exp", "null")
            note = cell.get("note", "")
            print(f"  {dataset:12s} / {model:12s} : exp={exp}  acc={acc}"
                  + (f"  [{note}]" if note else ""))

    gap2 = comparison["gap2_summary"]
    icon = "✅" if gap2["all_compliant"] else "❌"
    print(f"\nGap 2 {icon} — RAM max observée : {gap2['ram_max_observed_bytes']} B"
          f" / budget {gap2['ram_budget_bytes']} B")


if __name__ == "__main__":
    main()
