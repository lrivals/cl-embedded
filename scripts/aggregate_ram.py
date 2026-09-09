#!/usr/bin/env python3
"""aggregate_ram.py — Agrégat unique des consommations RAM complètes (Sprint 49, S4904).

Fusionne les 32 cellules mesurées de ``experiments/exp_S49_ram/`` (4 modèles × 2 datasets ×
{fp32,int8} × {board,pc}, S4901–S4903) en deux artefacts, **jamais édités à la main** :

1. ``experiments/exp_S49_ram/summary.json`` indexé
   ``[model][dataset][condition][encoding][platform]`` → bloc RAM complet
   (`data`, `bss`, `stack_peak_inference`, `stack_peak_update`, `total`, `stack_history`,
   `status`, `na_reason`, `ratio_int8_vs_fp32`). Le ratio int8/fp32 est **calculé** (total int8
   ÷ total fp32, même model/dataset/platform ; ``null`` si l'un manque ou est N/A).

2. ``docs/context/ram_report.md`` — doc structurée pour export rapport (méthode, tableau par
   cellule, historique du pic de pile, constats sans chiffre en dur, limites/N/A).

**Lecture seule** sur ``exp_S49_ram/`` : aucune mesure recalculée. Formule officielle du CR du
16 juillet 2026 : ``RAM totale = .data + .bss + pic de pile``.

Usage :
    python scripts/aggregate_ram.py     # → summary.json + ram_report.md (idempotent)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAM_DIR = ROOT / "experiments" / "exp_S49_ram"
SUMMARY_OUT = RAM_DIR / "summary.json"
REPORT_OUT = ROOT / "docs" / "context" / "ram_report.md"

RAM_BUDGET_BYTES = 256 * 1024

MODELS = ("ewc", "hdc", "tinyol", "mahalanobis")
DATASETS = ("monitoring", "pronostia")
ENCODINGS = ("fp32", "int8")
PLATFORMS = ("board", "pc")
CONDITION = "5feat"

MODEL_LABEL = {"ewc": "EWC", "hdc": "HDC", "tinyol": "TinyOL", "mahalanobis": "Mahalanobis"}
PLATFORM_LABEL = {"board": "board (NUCLEO-F439ZI)", "pc": "PC (tracemalloc)"}
# Étiquettes de phase neutres — jamais d'étapes « réseau de neurones » (fix plots CR/S4905).
PHASE_LABEL = {"idle": "idle", "inference": "inférence", "update": "mise à jour CL"}


def _cell_path(model: str, dataset: str, encoding: str, platform: str) -> Path:
    return RAM_DIR / f"{model}_{dataset}_{CONDITION}_{encoding}_{platform}.json"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _num(v) -> float | int | None:
    """Valeur numérique ou None (sentinelles / null → None)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _block(cell: dict | None) -> dict:
    """Bloc RAM normalisé d'une cellule (ratio ajouté plus tard)."""
    if cell is None:
        return {"status": "missing", "na_reason": "cellule absente", "total": None}
    return {
        "data": _num(cell.get("data_bytes")),
        "bss": _num(cell.get("bss_bytes")),
        "stack_peak_inference": _num(cell.get("stack_peak_inference_bytes")),
        "stack_peak_update": _num(cell.get("stack_peak_update_bytes")),
        "total": _num(cell.get("total_ram_bytes")),
        "stack_history": cell.get("stack_history", []),
        "status": cell.get("status", "unknown"),
        "na_reason": cell.get("na_reason"),
        "ratio_int8_vs_fp32": None,
    }


def build_summary() -> dict:
    """Construit l'index ``[model][dataset][condition][encoding][platform]`` + ratios."""
    summary: dict = {}
    for model in MODELS:
        summary[model] = {}
        for dataset in DATASETS:
            summary[model][dataset] = {CONDITION: {}}
            node = summary[model][dataset][CONDITION]
            for encoding in ENCODINGS:
                node[encoding] = {}
                for platform in PLATFORMS:
                    cell = _load(_cell_path(model, dataset, encoding, platform))
                    node[encoding][platform] = _block(cell)
            # Ratio int8/fp32 de la RAM totale, par plateforme (calculé, jamais saisi).
            for platform in PLATFORMS:
                fp32_total = node["fp32"][platform]["total"]
                int8_total = node["int8"][platform]["total"]
                if isinstance(fp32_total, (int, float)) and fp32_total and \
                        isinstance(int8_total, (int, float)):
                    node["int8"][platform]["ratio_int8_vs_fp32"] = round(
                        int8_total / fp32_total, 4
                    )
    return {
        "_meta": {
            "generated_by": "scripts/aggregate_ram.py",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "formula": "RAM totale = .data + .bss + pic de pile",
            "source_dir": "experiments/exp_S49_ram/",
            "condition": CONDITION,
            "ram_budget_bytes": RAM_BUDGET_BYTES,
            "note": "Lecture seule ; deltas/ratios calculés, aucune valeur saisie à la main.",
        },
        **summary,
    }


# ── Rendu Markdown (valeurs issues du summary, aucune valeur figée à la main) ─────────────

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:g}"
    return f"{v:,}".replace(",", " ")


def _table_rows(summary: dict) -> list[str]:
    rows: list[str] = []
    for model in MODELS:
        for dataset in DATASETS:
            node = summary[model][dataset][CONDITION]
            for encoding in ENCODINGS:
                for platform in PLATFORMS:
                    b = node[encoding][platform]
                    ratio = b.get("ratio_int8_vs_fp32")
                    ratio_str = _fmt(ratio) if encoding == "int8" else "—"
                    if b["status"] in ("na", "missing"):
                        rows.append(
                            f"| {MODEL_LABEL[model]} | {dataset} | {encoding} | "
                            f"{PLATFORM_LABEL[platform]} | N/A | N/A | N/A | N/A | N/A | {ratio_str} |"
                        )
                        continue
                    rows.append(
                        f"| {MODEL_LABEL[model]} | {dataset} | {encoding} | "
                        f"{PLATFORM_LABEL[platform]} | {_fmt(b['data'])} | {_fmt(b['bss'])} | "
                        f"{_fmt(b['stack_peak_inference'])} | {_fmt(b['stack_peak_update'])} | "
                        f"{_fmt(b['total'])} | {ratio_str} |"
                    )
    return rows


def _history_blocks(summary: dict) -> list[str]:
    out: list[str] = []
    for model in MODELS:
        out.append(f"\n### {MODEL_LABEL[model]}\n")
        any_hist = False
        for dataset in DATASETS:
            node = summary[model][dataset][CONDITION]
            b = node["fp32"]["board"]  # historique = phases board (pile réelle)
            hist = b.get("stack_history") or []
            if not hist:
                continue
            any_hist = True
            steps = " → ".join(
                f"{PHASE_LABEL.get(h['phase'], h['phase'])} {_fmt(h['stack_peak_bytes'])} B"
                for h in hist
            )
            out.append(f"- **{dataset}** (fp32, board) : {steps}")
        if not any_hist:
            out.append("- (aucun historique board disponible pour ce modèle)")
    return out


def _na_blocks(summary: dict) -> list[str]:
    out: list[str] = []
    for model in MODELS:
        for dataset in DATASETS:
            node = summary[model][dataset][CONDITION]
            for encoding in ENCODINGS:
                for platform in PLATFORMS:
                    b = node[encoding][platform]
                    if b["status"] == "na" and b.get("na_reason"):
                        out.append(
                            f"- **{MODEL_LABEL[model]} · {dataset} · {encoding} · "
                            f"{PLATFORM_LABEL[platform]}** : {b['na_reason']}"
                        )
    return out or ["- (aucune cellule N/A)"]


def build_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Consommations RAM par expérience — CL-Embedded")
    lines.append("")
    lines.append(
        "> Document **généré** par `scripts/aggregate_ram.py` depuis "
        "`experiments/exp_S49_ram/summary.json` — **aucune valeur saisie à la main**. "
        "Régénérer : `python scripts/aggregate_ram.py`."
    )
    lines.append("")
    # §1
    lines.append("## 1. Méthode")
    lines.append("")
    lines.append(
        "Formule officielle (CR du 16 juillet 2026) : **`RAM totale = .data + .bss + "
        "pic de pile`**. La mesure précédente ne remontait que `.bss` → **sous-estimée** "
        "(la pile vit hors `.bss`). Détails et mécanisme (stack painting) : "
        "[`ram_measurement.md`](ram_measurement.md) et "
        "[`../presentation_ram_measurement.md`](../presentation_ram_measurement.md)."
    )
    lines.append("")
    lines.append(
        "- **`.data`** : globales initialisées. **`.bss`** : globales à zéro / mémoire non "
        "constante (état CL). **Pic de pile** : high-water mark mesuré **après chaque phase** "
        "(`idle` → `inférence` → `mise à jour CL`)."
    )
    lines.append(
        "- Board : `.bss`+pile réels (NUCLEO-F439ZI, `profiling_total_ram_bytes`). "
        "PC : pic `tracemalloc` d'un forward (`.data`/`.bss` non applicables → N/A)."
    )
    lines.append("")
    # §2
    lines.append("## 2. Tableau par modèle × dataset × encodage × plateforme")
    lines.append("")
    lines.append(
        "| Modèle | Dataset | Encodage | Plateforme | `.data` | `.bss` | pic inférence | "
        "pic MAJ | total | ratio int8/fp32 |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    lines.extend(_table_rows(summary))
    lines.append("")
    lines.append("_Board et PC en lignes distinctes (jamais fusionnés). Toutes en octets._")
    lines.append("")
    # §3
    lines.append("## 3. Historique d'évolution du pic de pile")
    lines.append("")
    lines.append(
        "Pic mesuré **après chaque phase**, chronologiquement. La **mise à jour CL** (SGD "
        "embarqué) creuse plus la pile que l'inférence seule — pile transitoire, `.bss` inchangé."
    )
    lines.extend(_history_blocks(summary))
    lines.append("")
    # §4
    lines.append("## 4. Constats")
    lines.append("")
    lines.append(
        "- **`.bss` seul ≠ RAM totale** : rapporter `.bss` oublie la pile ; le total corrige "
        "la sous-estimation (cf. tableau §2)."
    )
    lines.append(
        "- **Le pic dépend de la phase** : `pic MAJ ≥ pic inférence` sur les cellules board "
        "applicables (cf. historique §3)."
    )
    lines.append(
        "- **INT8 réduit la mémoire non constante sans changer la pile** : le ratio int8/fp32 "
        "de la RAM totale (§2) reflète l'économie de poids, pas de la pile."
    )
    lines.append(
        "- **Pile ~partagée entre modèles** : trame unique `pipeline_run()` (max des branches) "
        "→ pic de pile board ~homogène (cf. §2)."
    )
    lines.append("")
    # §5
    lines.append("## 5. Limites & N/A honnêtes")
    lines.append("")
    lines.extend(_na_blocks(summary))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not RAM_DIR.exists():
        raise SystemExit(f"Répertoire introuvable : {RAM_DIR}")
    summary = build_summary()
    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    REPORT_OUT.write_text(build_report(summary), encoding="utf-8")
    print(f"✅ {SUMMARY_OUT.relative_to(ROOT)}")
    print(f"✅ {REPORT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
