#!/usr/bin/env python3
"""generate_article_tables.py — Tables d'annexe de l'article EWC INT8 (S4011).

Produit les tables LaTeX des annexes de ``docs/article/ewc_int8_mcu/`` **depuis
l'agrégat** ``experiments/exp_S40_article_metrics/summary.json``, en FR et EN.

Motivation : l'article déclare qu'aucune valeur n'est saisie à la main. Une annexe
qui détaille des dizaines de cellules ne peut donc pas être tapée ; elle est générée,
et se régénère quand l'agrégat change. Aucun calcul ici : uniquement de la mise en
forme de valeurs lues, plus le formatage des cellules absentes.

Convention d'honnêteté conservée : une cellule sans mesure sort en ``--`` (jamais 0),
et le registre des mesures manquantes est lui-même une table.

Usage :
    python scripts/generate_article_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "experiments" / "exp_S40_article_metrics" / "summary.json"
OUT = ROOT / "docs" / "article" / "ewc_int8_mcu" / "tables"

DATASETS = ("monitoring", "pronostia")
DEPTHS = ("int8", "int6", "int4", "int3", "int2", "ternaire", "binaire")
#: Profondeurs réellement flashées (S4804). Les clés de ``depth_board`` sont indexées par
#: nombre de bits ; le ternaire y porte ``bits2`` (mode ``ternary`` du kernel packé).
BOARD_DEPTHS = (("bits4", "int4"), ("bits2", "ternary"), ("bits1", "binary"))
NA = "--"

L = {
    "fr": {
        "depth_cap": "Balayage de profondeur complet : $\\Delta$AUROC contre la référence FP32, par "
                     "profondeur et granularité (\\emph{émulé PC bit-exact}). Un $\\Delta$ négatif est une "
                     "dégradation.",
        "depth_board_cap": "Schémas sub-INT8 \\emph{mesurés sur carte} : AUROC et F1 de la classe "
                           "\\texttt{faulty} sur les mêmes cellules. L'AUROC reste haute là où le F1 "
                           "décroche, la profondeur ne déplaçant pas les deux métriques ensemble.",
        "sym_cap": "Symétrie de la quantification : $\\Delta$AUROC symétrique contre affine à "
                   "zéro-point (\\emph{émulé PC bit-exact}).",
        "ram_cap": "Composantes de la RAM, en octets (\\emph{mesuré sur carte}). Le total est la somme "
                   "des trois premières lignes, le pic de pile étant le maximum des deux phases.",
        "energy_cap": "Les trois estimateurs d'énergie, côte à côte et jamais fusionnés "
                      "(\\emph{mesuré sur carte}). En \\si{\\micro\\joule} par inférence.",
        "missing_cap": "Registre des cellules non mesurées. Elles portent « à mesurer » dans le texte "
                       "et ne sont jamais remplacées par une estimation.",
        "depth": "Profondeur", "pt": "Par tenseur", "pc": "Par canal",
        "ternary": "ternaire", "binary": "binaire",
        "scheme": "Schéma", "sym": "Symétrique", "aff": "Affine",
        "comp": "Composante", "ratio": "Rapport",
        "data": "\\texttt{.data}", "bss": "\\texttt{.bss}",
        "stack_i": "Pic de pile (inférence)", "stack_u": "Pic de pile (mise à jour)",
        "total": "Total", "est": "Estimateur", "cell": "Cellule", "reason": "Raison",
        "reg": "Régression en cadence", "delta": "Différence contre repos \\texttt{WFI}",
        "batch": "Lot d'inférences par trame",
    },
    "en": {
        "depth_cap": "Full depth sweep: $\\Delta$AUROC against the FP32 reference, by depth and "
                     "granularity (\\emph{bit-exact PC emulation}). A negative $\\Delta$ is a degradation.",
        "depth_board_cap": "Sub-INT8 schemes \\emph{measured on-board}: AUROC and \\texttt{faulty}-class "
                           "F1 on the same cells. AUROC stays high where F1 breaks down, depth not moving "
                           "the two metrics together.",
        "sym_cap": "Quantization symmetry: $\\Delta$AUROC, symmetric against affine with a zero point "
                   "(\\emph{bit-exact PC emulation}).",
        "ram_cap": "RAM components, in bytes (\\emph{measured on-board}). The total is the sum of the "
                   "first three rows, the stack peak being the maximum of the two phases.",
        "energy_cap": "The three energy estimators, side by side and never merged "
                      "(\\emph{measured on-board}). In \\si{\\micro\\joule} per inference.",
        "missing_cap": "Registry of unmeasured cells. They carry \"to be measured\" in the text and are "
                       "never replaced by an estimate.",
        "depth": "Depth", "pt": "Per tensor", "pc": "Per channel",
        "scheme": "Scheme", "sym": "Symmetric", "aff": "Affine",
        "comp": "Component", "ratio": "Ratio",
        "data": "\\texttt{.data}", "bss": "\\texttt{.bss}",
        "stack_i": "Stack peak (inference)", "stack_u": "Stack peak (update)",
        "total": "Total", "est": "Estimator", "cell": "Cell", "reason": "Reason",
        "reg": "Rate regression", "delta": "Difference against \\texttt{WFI} idle",
        "batch": "Batch of inferences per frame",
    },
}


def cell(block: dict, key: str):
    """Valeur d'une cellule de l'agrégat, ou None si absente/non mesurée."""
    entry = block.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    return value if isinstance(value, (int, float)) else None


def num(value, digits: int = 4) -> str:
    if value is None:
        return NA
    return f"\\num{{{value:.{digits}f}}}"


def integer(value) -> str:
    return NA if value is None else f"\\num{{{int(value)}}}"


def table(caption: str, label: str, spec: str, header: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def build_depth(agg: dict, t: dict) -> str:
    rows = []
    for depth in DEPTHS:
        cells = [f"\\texttt{{{depth}}}"]
        for ds in DATASETS:
            blk = agg[ds]["depth_pc"]
            cells.append(num(cell(blk, f"{depth}_per_tensor_delta_auroc")))
            cells.append(num(cell(blk, f"{depth}_per_channel_delta_auroc")))
        rows.append(" & ".join(cells) + " \\\\")
    header = (f"{t['depth']} & \\multicolumn{{2}}{{c}}{{Monitoring}} & "
              f"\\multicolumn{{2}}{{c}}{{Pronostia}} \\\\\n"
              f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\n"
              f" & {t['pt']} & {t['pc']} & {t['pt']} & {t['pc']}")
    return table(t["depth_cap"], "tab:annex-depth", "lcccc", header, rows)


def build_depth_board(agg: dict, t: dict) -> str:
    """AUROC et F1 **mesurés sur carte** des schémas sub-INT8 réellement flashés.

    Cette table est le pendant mesuré de ``tab:annex-depth`` : le balayage émulé ne rapporte
    qu'un $\\Delta$AUROC, alors que la carte a mesuré les deux métriques sur les mêmes schémas.
    Les lire côte à côte est le seul moyen de voir que l'ordre des scores et le seuil de
    décision ne décrochent pas à la même profondeur.
    """
    rows = []
    for key, label in BOARD_DEPTHS:
        cells = [f"\\texttt{{{t.get(label, label)}}}"]
        for ds in DATASETS:
            blk = agg[ds]["depth_board"]
            cells.append(num(cell(blk, f"{key}_per_channel_nonpacked_auroc_board"), 5))
            cells.append(num(cell(blk, f"{key}_per_channel_nonpacked_f1_faulty")))
        rows.append(" & ".join(cells) + " \\\\")
    header = (f"{t['depth']} & \\multicolumn{{2}}{{c}}{{Monitoring}} & "
              f"\\multicolumn{{2}}{{c}}{{Pronostia}} \\\\\n"
              f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\n"
              " & AUROC & F1 & AUROC & F1")
    return table(t["depth_board_cap"], "tab:depth-board", "lcccc", header, rows)


def build_symmetry(agg: dict, t: dict) -> str:
    rows = []
    for depth in ("int4", "int3", "int2"):
        cells = [f"\\texttt{{{depth}}}"]
        for ds in DATASETS:
            blk = agg[ds]["depth_pc"]
            cells.append(num(cell(blk, f"symmetry_{depth}_symmetric_delta_auroc")))
            cells.append(num(cell(blk, f"symmetry_{depth}_affine_delta_auroc")))
        rows.append(" & ".join(cells) + " \\\\")
    header = (f"{t['scheme']} & \\multicolumn{{2}}{{c}}{{Monitoring}} & "
              f"\\multicolumn{{2}}{{c}}{{Pronostia}} \\\\\n"
              f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\n"
              f" & {t['sym']} & {t['aff']} & {t['sym']} & {t['aff']}")
    return table(t["sym_cap"], "tab:annex-symmetry", "lcccc", header, rows)


def build_ram(agg: dict, t: dict) -> str:
    lines = [
        (t["data"], "data"),
        (t["bss"], "bss"),
        (t["stack_i"], "stack_peak_inference"),
        (t["stack_u"], "stack_peak_update"),
        (t["total"], "total"),
    ]
    rows = []
    for label, suffix in lines:
        cells = [label]
        for ds in DATASETS:
            blk = agg[ds]["ram"]
            cells.append(integer(cell(blk, f"fp32_board_{suffix}")))
            cells.append(integer(cell(blk, f"int8_board_{suffix}")))
        rows.append(" & ".join(cells) + " \\\\")
    ratio = [t["ratio"]]
    for ds in DATASETS:
        ratio.append("\\multicolumn{2}{c}{" + num(cell(agg[ds]["ram"], "int8_board_ratio_int8_vs_fp32")) + "}")
    rows.append("\\midrule")
    rows.append(" & ".join(ratio) + " \\\\")
    header = (f"{t['comp']} & \\multicolumn{{2}}{{c}}{{Monitoring}} & "
              f"\\multicolumn{{2}}{{c}}{{Pronostia}} \\\\\n"
              f"\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\n"
              " & FP32 & INT8 & FP32 & INT8")
    return table(t["ram_cap"], "tab:annex-ram", "lcccc", header, rows)


def build_energy(agg: dict, t: dict) -> str:
    est = agg["energy"]["estimators"]
    rows = []
    for key, label in (("regression", t["reg"]), ("delta_wfi", t["delta"]), ("batch", t["batch"])):
        cells = est.get(key, {}).get("cells", {})
        rows.append(" & ".join([
            label,
            num(cell(cells, "ewc_fp32_energy_uj_per_inference"), 2),
            num(cell(cells, "ewc_int8_energy_uj_per_inference"), 2),
        ]) + " \\\\")
    return table(t["energy_cap"], "tab:annex-energy", "lcc", f"{t['est']} & FP32 & INT8", rows)


def build_missing(agg: dict, t: dict) -> str:
    rows = []
    for path in agg.get("missing", []):
        node = agg
        for part in path.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        reason = node.get("na_reason") if isinstance(node, dict) else None
        reason = (reason or NA).replace("_", "\\_")
        if len(reason) > 78:
            reason = reason[:75].rsplit(" ", 1)[0] + "…"
        rows.append(f"\\texttt{{{path.replace('_', chr(92) + '_')}}} & {reason} \\\\")
    return table(t["missing_cap"], "tab:annex-missing", "p{0.42\\linewidth}p{0.5\\linewidth}",
                 f"{t['cell']} & {t['reason']}", rows)


def main() -> None:
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    builders = {
        "annex_depth": build_depth,
        "depth_board": build_depth_board,
        "annex_symmetry": build_symmetry,
        "annex_ram": build_ram,
        "annex_energy": build_energy,
        "annex_missing": build_missing,
    }
    written = 0
    for lang, t in L.items():
        for name, build in builders.items():
            path = OUT / f"{name}_{lang}.tex"
            header = ("% Généré par scripts/generate_article_tables.py — NE PAS ÉDITER À LA MAIN.\n"
                      "% Source : experiments/exp_S40_article_metrics/summary.json\n")
            path.write_text(header + build(agg, t), encoding="utf-8")
            written += 1
    print(f"{written} tables écrites dans {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
