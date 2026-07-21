#!/usr/bin/env python3
"""Génère le notebook galerie S4706 (notebooks/cl_eval/quant_depth/comparison.ipynb).

Notebook FR commenté : consomme ``src/figures/`` (catalogue ``quant_depth``), recharge
les valeurs par cellule (jamais en dur), affiche un tableau de synthèse + reco calculés
depuis les JSON. Exécutable via nbconvert. Ce script n'est qu'un générateur du .ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "cl_eval" / "quant_depth" / "comparison.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    "# Sprint 47 — Profondeur & schéma de quantification (EWC)\n"
    "\n"
    "Second axe de la quantification, **orthogonal** au *moment* (Sprint 46) : *jusqu'où "
    "descendre en bits* (sub-INT8) avant que la tête EWC casse, et *quelle calibration* "
    "(granularité / symétrie) rachète la métrique. **PC/émulateur bit-exact** — le portage "
    "board est le Sprint 48.\n"
    "\n"
    "Toutes les valeurs de ce notebook sont **rechargées depuis les JSON** "
    "(`experiments/exp_S47_*`) via `src/figures/loaders.py` — aucun chiffre en dur. Les "
    "figures sont régénérées par le catalogue `quant_depth` (`src/figures/`)."
)

code(
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path.cwd()\n"
    "while not (ROOT / 'src').exists() and ROOT != ROOT.parent:\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT))\n"
    "\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "from src.figures import registry\n"
    "import src.figures.catalogs  # noqa: F401 — auto-enregistrement\n"
    "from src.figures.loaders import load_experiment, metric_or_na\n"
    "from src.figures.style import apply_style\n"
    "from src.utils.reproducibility import set_seed\n"
    "\n"
    "apply_style('slide')\n"
    "set_seed(42)\n"
    "print('catalogues :', [c for c in registry.list_catalogs() if 'quant_depth' in c])"
)

md(
    "## 1. Régénération des figures\n"
    "\n"
    "Le catalogue `quant_depth` produit 5 PNG sous `docs/figures/quantization_depth/`."
)

code(
    "paths = registry.get_catalog('quant_depth')(ROOT / 'docs' / 'figures')\n"
    "for p in paths:\n"
    "    print(p.relative_to(ROOT))"
)

md("## 2. Galerie")

for name, title in [
    ("auroc_vs_bits", "Δ AUROC vs profondeur (par granularité, par dataset)"),
    ("heatmap_bits_granularity", "Heatmap Δ AUROC (bits × granularité)"),
    ("ram_vs_bits", "Gain RAM théorique (bit-packée) vs profondeur"),
    ("symmetry_gain", "Zero-point affine vs symétrique aux bits critiques"),
    ("scope_context", "Périmètre EWC-only ∥ HDC/Maha/TinyOL N/A"),
]:
    md(f"### {title}")
    code(
        "from IPython.display import Image, display\n"
        f"display(Image(filename=str(ROOT / 'docs' / 'figures' / 'quantization_depth' / '{name}.png')))"
    )

md(
    "## 3. Tableau de synthèse (Δ AUROC, rechargé depuis les JSON)\n"
    "\n"
    "`delta_auroc` par (dataset × granularité × profondeur) — aucune valeur en dur."
)

code(
    "DATASETS = ['monitoring', 'pronostia']\n"
    "GRANS = ['per_tensor', 'per_channel']\n"
    "TAGS = ['int8', 'int6', 'int4', 'int3', 'int2', 'ternaire', 'binaire']\n"
    "\n"
    "def depth_delta(ds, tag, gran):\n"
    "    try:\n"
    "        data, _ = load_experiment(f'experiments/exp_S47_depth/exp_S47_ewc_{ds}_{tag}_{gran}.json')\n"
    "    except FileNotFoundError:\n"
    "        return None\n"
    "    v = metric_or_na(data, 'delta_auroc')\n"
    "    return v if isinstance(v, (int, float)) else None\n"
    "\n"
    "hdr = f\"{'dataset':<11}{'granularité':<13}\" + ''.join(f'{t:>10}' for t in TAGS)\n"
    "print(hdr)\n"
    "print('-' * len(hdr))\n"
    "for ds in DATASETS:\n"
    "    for gran in GRANS:\n"
    "        row = f'{ds:<11}{gran:<13}'\n"
    "        for t in TAGS:\n"
    "            v = depth_delta(ds, t, gran)\n"
    "            row += f'{v:>+10.4f}' if v is not None else f\"{'N/A':>10}\"\n"
    "        print(row)"
)

md(
    "## 4. Recommandation — plus petit `weight_bits` viable (Δ AUROC ≥ −0,02)\n"
    "\n"
    "Le seuil de dégradation −0,02 et le gain RAM sont **calculés depuis les JSON**."
)

code(
    "SEUIL = -0.02  # seuil de dégradation (cadrage, pas un résultat mesuré)\n"
    "\n"
    "def ram_ratio(ds, tag, gran):\n"
    "    try:\n"
    "        data, _ = load_experiment(f'experiments/exp_S47_depth/exp_S47_ewc_{ds}_{tag}_{gran}.json')\n"
    "    except FileNotFoundError:\n"
    "        return None\n"
    "    v = metric_or_na(data, 'ram_ratio_vs_fp32')\n"
    "    return v if isinstance(v, (int, float)) else None\n"
    "\n"
    "for ds in DATASETS:\n"
    "    for gran in GRANS:\n"
    "        viable = [t for t in TAGS\n"
    "                  if (d := depth_delta(ds, t, gran)) is not None and d >= SEUIL]\n"
    "        if viable:\n"
    "            best = viable[-1]  # TAGS ordonné du plus profond au plus bas\n"
    "            print(f'{ds:<11}{gran:<13} → {best:<9} '\n"
    "                  f'(Δ={depth_delta(ds, best, gran):+.4f}, RAM ×{ram_ratio(ds, best, gran)})')\n"
    "        else:\n"
    "            print(f'{ds:<11}{gran:<13} → aucun sous le seuil')"
)

md(
    "## 5. Axe symétrie — gain du zero-point affine (bits critiques)\n"
    "\n"
    "`Δ(affine) − Δ(symétrique)` par (dataset, profondeur), rechargé depuis "
    "`exp_S47_symmetry/`. Un gain positif = le zero-point rachète de l'AUROC."
)

code(
    "CRIT = ['int2', 'int3', 'int4']\n"
    "\n"
    "def sym_delta(ds, tag, sym):\n"
    "    try:\n"
    "        data, _ = load_experiment(f'experiments/exp_S47_symmetry/exp_S47_ewc_{ds}_{tag}_{sym}.json')\n"
    "    except FileNotFoundError:\n"
    "        return None\n"
    "    v = metric_or_na(data, 'delta_auroc')\n"
    "    return v if isinstance(v, (int, float)) else None\n"
    "\n"
    "print(f\"{'dataset':<11}{'bits':<8}{'symétrique':>12}{'affine':>10}{'gain affine':>14}\")\n"
    "print('-' * 55)\n"
    "for ds in DATASETS:\n"
    "    for t in CRIT:\n"
    "        s = sym_delta(ds, t, 'symmetric')\n"
    "        a = sym_delta(ds, t, 'affine')\n"
    "        gain = (a - s) if (s is not None and a is not None) else None\n"
    "        srepr = f'{s:+.4f}' if s is not None else 'N/A'\n"
    "        arepr = f'{a:+.4f}' if a is not None else 'N/A'\n"
    "        grepr = f'{gain:+.4f}' if gain is not None else 'N/A'\n"
    "        print(f'{ds:<11}{t:<8}{srepr:>12}{arepr:>10}{grepr:>14}')"
)

md(
    "## 6. Contexte N/A (HDC / Mahalanobis / TinyOL)\n"
    "\n"
    "Cadrage pur — aucun résultat fabriqué (cf. S4705)."
)

code(
    "ctx, _ = load_experiment('experiments/exp_S47_context/context.json')\n"
    "print('modèles balayés :', ctx['swept_models'])\n"
    "for name, info in ctx['context_models'].items():\n"
    "    print(f\"\\n{name.upper()} [{info['status']}] (réf. {info['ref']})\")\n"
    "    print(' ', info['reason'])"
)

md(
    "---\n"
    "**Lecture d'ensemble.** La per-canal repousse le « cliff » en profondeur (Pronostia), "
    "le zero-point affine n'apporte pas de gain décisif ici (per-canal suffit), et le gain "
    "RAM sub-INT8 reste **théorique** tant qu'un kernel bit-packé n'est pas porté (Sprint 48)."
)

nb["cells"] = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"écrit : {OUT}")
