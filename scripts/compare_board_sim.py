"""
compare_board_sim.py — Tableau comparatif board vs simulation (dry-run) Sprint 19 & 20.

Lit tous les fichiers results*.json dans les répertoires d'expériences spécifiés,
regroupe par (modèle, lambda, platform) et produit un tableau Markdown + JSON.

Usage :
    python scripts/compare_board_sim.py \\
        --exp-dirs experiments/exp_S19_01 experiments/exp_S19_02 \\
        --output experiments/comparison_sprint19_20.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_METRICS = [
    "acc_final",
    "avg_forgetting",
    "backward_transfer",
    "ram_peak_bytes",
    "inference_latency_ms",
]

_GAP2_RAM_LIMIT_BYTES  = 64 * 1024   # 64 Ko — critère Gap 2
_GAP2_LAT_LIMIT_MS     = 100.0       # 100 ms — critère Gap 2


def _exp_label(r: dict) -> str:
    """Identifiant lisible : ex. 'E19-01', 'E19-02'."""
    exp_id = r.get("exp_id", r.get("exp_dir", "?"))
    # Normalise S19_01 → E19-01
    return exp_id.replace("exp_S", "E").replace("_", "-")


def _lambda_str(r: dict) -> str:
    lam = r.get("lambda_ewc")
    if lam is None:
        return "—"
    return str(int(lam)) if float(lam) == int(float(lam)) else str(lam)


def _gap2_ok(r: dict) -> str:
    ram = r.get("ram_peak_bytes", 0) or 0
    lat = r.get("inference_latency_ms", 0) or 0
    if ram == 0 and lat == 0:
        return "—"
    ram_ok = (ram < _GAP2_RAM_LIMIT_BYTES) if ram > 0 else True
    lat_ok = (lat < _GAP2_LAT_LIMIT_MS) if lat > 0 else True
    return "✅" if (ram_ok and lat_ok) else "❌"


def _fmt(val, fmt: str = ".4f") -> str:
    if val is None:
        return "—"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _fmt_ram(val) -> str:
    if val is None or val == 0:
        return "—"
    kb = float(val) / 1024.0
    return f"{kb:.1f}"


def load_results(exp_dirs: list[Path]) -> list[dict]:
    """Charge tous les fichiers results*.json dans les répertoires donnés."""
    records: list[dict] = []
    for exp_dir in exp_dirs:
        for json_file in sorted(exp_dir.glob("results*.json")):
            with open(json_file) as f:
                r = json.load(f)
            r["_file"]    = str(json_file)
            r["exp_dir"]  = exp_dir.name
            records.append(r)
    return records


def build_table(records: list[dict]) -> tuple[str, list[dict]]:
    """Retourne (markdown_table, rows_json)."""
    rows: list[dict] = []
    for r in records:
        row = {
            "exp":        _exp_label(r),
            "model":      r.get("model", "?"),
            "lambda":     _lambda_str(r),
            "platform":   r.get("platform", "?"),
            "acc_final":  r.get("acc_final"),
            "avg_forget": r.get("avg_forgetting"),
            "bwt":        r.get("backward_transfer"),
            "ram_kb":     r.get("ram_peak_bytes"),
            "lat_ms":     r.get("inference_latency_ms"),
            "gap2":       _gap2_ok(r),
            "_file":      r.get("_file", ""),
        }
        rows.append(row)

    # Tri : exp → model → lambda → platform
    rows.sort(key=lambda x: (x["exp"], x["model"], x["lambda"], x["platform"]))

    header = (
        "| Expérience | Modèle | λ | Platform | acc_final | avg_forgetting"
        " | BWT | RAM (Ko) | Latence (ms) | Gap2 ✓ |\n"
        "|-----------|--------|---|----------|:---------:|:--------------:"
        "|:---:|:--------:|:------------:|:------:|"
    )
    lines = [header]
    for row in rows:
        line = (
            f"| {row['exp']} | {row['model']} | {row['lambda']} | {row['platform']}"
            f" | {_fmt(row['acc_final'])}"
            f" | {_fmt(row['avg_forget'])}"
            f" | {_fmt(row['bwt'])}"
            f" | {_fmt_ram(row['ram_kb'])}"
            f" | {_fmt(row['lat_ms'])}"
            f" | {row['gap2']} |"
        )
        lines.append(line)

    md = "# Comparaison board vs simulation — Sprint 19 & 20\n\n" + "\n".join(lines)
    return md, rows


def check_ewc_property(rows: list[dict]) -> list[str]:
    """Vérifie que avg_forgetting(EWC λ>0) < avg_forgetting(λ=0) par platform."""
    findings: list[str] = []
    from itertools import groupby

    for platform in set(r["platform"] for r in rows if r["model"] == "ewc"):
        ewc_rows = [r for r in rows if r["model"] == "ewc" and r["platform"] == platform]
        baseline = next((r for r in ewc_rows if r["lambda"] == "0"), None)
        ewc_runs = [r for r in ewc_rows if r["lambda"] not in ("0", "—")]

        if baseline is None or not ewc_runs:
            continue

        af_base = baseline.get("avg_forget")
        for ewc in ewc_runs:
            af_ewc = ewc.get("avg_forget")
            if af_base is not None and af_ewc is not None:
                ok = af_ewc <= af_base
                status = "✅" if ok else "❌"
                findings.append(
                    f"  {status} [{platform}] EWC λ={ewc['lambda']}: "
                    f"avg_forgetting={af_ewc:.4f} vs λ=0: {af_base:.4f}"
                )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère tableau comparatif board vs simulation Sprint 19&20")
    parser.add_argument("--exp-dirs", nargs="+", required=True, type=Path,
                        metavar="DIR", help="Répertoires experiments/exp_S19_0X")
    parser.add_argument("--output", required=True, type=Path,
                        help="Fichier Markdown de sortie")
    args = parser.parse_args()

    records = load_results(args.exp_dirs)
    if not records:
        print(f"Aucun fichier results*.json trouvé dans {args.exp_dirs}")
        return

    print(f"{len(records)} fichier(s) résultat chargé(s)")
    for r in records:
        print(f"  {r['_file']} — model={r.get('model')} platform={r.get('platform')}"
              f" acc={r.get('acc_final')} λ={r.get('lambda_ewc', '—')}")

    md, rows = build_table(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")

    # JSON parallèle
    json_out = args.output.with_suffix(".json")
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{md}\n")

    # Vérification propriété EWC
    findings = check_ewc_property(rows)
    if findings:
        print("Propriété EWC (avg_forgetting(λ>0) ≤ avg_forgetting(λ=0)) :")
        for f in findings:
            print(f)

    print(f"\nTableau sauvé : {args.output}")
    print(f"JSON sauvé    : {json_out}")


if __name__ == "__main__":
    main()
