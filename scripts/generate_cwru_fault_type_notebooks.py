"""
generate_cwru_fault_type_notebooks.py

Génère les 7 notebooks d'analyse CL pour le scénario cwru_by_fault_type.
    notebooks/cl_eval/cwru_by_fault_type/{ewc,hdc,tinyol,kmeans,mahalanobis,dbscan,comparison}.ipynb

Usage :
    python scripts/generate_cwru_fault_type_notebooks.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "notebooks/cl_eval/cwru_by_fault_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Ball", "Inner Race", "Outer Race"]
RAM_BUDGET = 65_536  # bytes

MODEL_CONFIGS = [
    {
        "model_name": "EWC",
        "model_key": "ewc",
        "exp_id": "exp_074_ewc_cwru_by_fault_type",
        "exp_num": "074",
        "baseline_exp": "exp_068_ewc_cwru_single_task",
        "baseline_num": "068",
        "description": "EWC Online + MLP (865 paramètres, input_dim=9, hidden=[32, 16])",
        "cl_strategy": "Régularisation EWC Online (λ=1000, γ=0.9) — aucun buffer de replay",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": False,
    },
    {
        "model_name": "HDC",
        "model_key": "hdc",
        "exp_id": "exp_075_hdc_cwru_by_fault_type",
        "exp_num": "075",
        "baseline_exp": "exp_069_hdc_cwru_single_task",
        "baseline_num": "069",
        "description": "HDC Hyperdimensional (1024 dimensions, input_dim=9)",
        "cl_strategy": "Architecture-based HDC — mise à jour des hypervecteurs de classe en ligne",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": False,
    },
    {
        "model_name": "TinyOL",
        "model_key": "tinyol",
        "exp_id": "exp_076_tinyol_cwru_by_fault_type",
        "exp_num": "076",
        "baseline_exp": "exp_070_tinyol_cwru_single_task",
        "baseline_num": "070",
        "description": "TinyOL + tête OtO (397 paramètres, input_dim=9, encoder=[16,8])",
        "cl_strategy": "Architecture-based TinyOL — encodeur gelé + tête OtO réentraînable",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": True,
    },
    {
        "model_name": "KMeans",
        "model_key": "kmeans",
        "exp_id": "exp_077_kmeans_cwru_by_fault_type",
        "exp_num": "077",
        "baseline_exp": "exp_071_kmeans_cwru_single_task",
        "baseline_num": "071",
        "description": "KMeans clustering (non-supervisé, n_clusters=2 par tâche)",
        "cl_strategy": "Refit KMeans complet à chaque tâche (sans mémoire explicite)",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": False,
    },
    {
        "model_name": "Mahalanobis",
        "model_key": "mahalanobis",
        "exp_id": "exp_078_mahalanobis_cwru_by_fault_type",
        "exp_num": "078",
        "baseline_exp": "exp_072_mahalanobis_cwru_single_task",
        "baseline_num": "072",
        "description": "Mahalanobis distance (non-supervisé, μ + Σ par tâche)",
        "cl_strategy": "Mise à jour online de μ et Σ par tâche (sans replay)",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": False,
    },
    {
        "model_name": "DBSCAN",
        "model_key": "dbscan",
        "exp_id": "exp_079_dbscan_cwru_by_fault_type",
        "exp_num": "079",
        "baseline_exp": "exp_073_dbscan_cwru_single_task",
        "baseline_num": "073",
        "description": "DBSCAN detector (non-supervisé, core points cumulés par tâche)",
        "cl_strategy": "Accumulation des core points DBSCAN — mémoire croissante",
        "sprint_ref": "12 — S12-07",
        "ragged_acc_matrix": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": source,
    }


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def make_nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Model notebook builder
# ---------------------------------------------------------------------------

def build_model_notebook(cfg: dict) -> dict:
    mn = cfg["model_name"]
    mk = cfg["model_key"]
    eid = cfg["exp_id"]
    enum = cfg["exp_num"]
    bdesc = cfg["description"]
    bline_exp = cfg["baseline_exp"]
    bline_num = cfg["baseline_num"]
    cl_strat = cfg["cl_strategy"]
    sprint = cfg["sprint_ref"]
    ragged = cfg["ragged_acc_matrix"]

    # --- Cell 0: header markdown ---
    header_md = f"""# Évaluation CL — {mn} — Dataset CWRU Bearing — by_fault_type

| Champ | Valeur |
|-------|--------|
| **Modèle** | {bdesc} |
| **Dataset** | CWRU Bearing Dataset (Case Western Reserve University) |
| **Scénario** | by_fault_type : Ball → Inner Race → Outer Race (3 tâches) |
| **Expérience** | {eid} — voir experiments/{eid}/config_snapshot.yaml |
| **Sprint** | {sprint} |

> **Stratégie CL** : {cl_strat}
> **Gap 1** : Validation CL sur données réelles de roulements (CWRU — benchmark académique reconnu).
> **Gap 2** : RAM mesurée vs budget STM32N6 ≤ 64 Ko.

```bash
jupyter nbconvert --to notebook --execute \\
    notebooks/cl_eval/cwru_by_fault_type/{mk}.ipynb \\
    --output /tmp/{mk}_cwru_fault_type_executed.ipynb --ExecutePreprocessor.timeout=300
```"""

    # --- Cell 1: setup ---
    acc_matrix_load_comment = (
        "# TinyOL : acc_matrix est une liste ragged (longueurs 1, 2, 3) — reconstruction 3×3"
        if ragged
        else f"# acc_matrix embarquée dans metrics_cl.json (liste 3×3)"
    )

    setup_code = f"""\
# Section 1 — Setup & imports
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

# --- CWD navigation : notebook 3 niveaux de profondeur ---
_cwd = Path(".").resolve()
if _cwd.name == "cwru_by_fault_type":
    os.chdir(_cwd.parent.parent.parent)
elif _cwd.name == "cl_eval":
    os.chdir(_cwd.parent.parent)
elif _cwd.name == "notebooks":
    os.chdir(_cwd.parent)
REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.plots import (
    plot_accuracy_matrix,
    plot_forgetting_curve,
    save_figure,
)

# --- Chemins ---
EXP_DIR     = REPO_ROOT / "experiments/{eid}/results"
FIGURES_DIR = REPO_ROOT / "notebooks/figures/cl_evaluation/{mk}/cwru/by_fault_type"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = REPO_ROOT / "experiments/{bline_exp}/results"
CSV_PATH     = REPO_ROOT / "data/raw/CWRU Bearing Dataset/feature_time_48k_2048_load_1.csv"

# --- Constantes ---
TASK_NAMES    = ["Ball", "Inner Race", "Outer Race"]
MODEL_NAME    = "{mn}"
DATA_AVAILABLE = CSV_PATH.exists()

print(f"REPO_ROOT    : {{REPO_ROOT}}")
print(f"EXP_DIR      : {{EXP_DIR}}")
print(f"FIGURES_DIR  : {{FIGURES_DIR}}")
print(f"Data CSV     : {{DATA_AVAILABLE}}")
print(f"Date         : {{datetime.now():%Y-%m-%d %H:%M}}")"""

    # --- Cell 2: load metrics ---
    if ragged:
        acc_matrix_reconstruction = """\

# TinyOL acc_matrix : liste ragged (longueurs variables) → 3×3 avec NaN
raw_mat = metrics["acc_matrix"]
acc_matrix_np = np.full((3, 3), np.nan, dtype=float)
for i, row in enumerate(raw_mat):
    for j, v in enumerate(row):
        if v is not None:
            acc_matrix_np[i, j] = v"""
    else:
        acc_matrix_reconstruction = """\

# acc_matrix : liste 3×3 avec null → NaN
raw_mat = metrics["acc_matrix"]
acc_matrix_np = np.array(
    [[v if v is not None else np.nan for v in row] for row in raw_mat],
    dtype=float,
)"""

    load_code = f"""\
# Section 2 — Chargement des résultats {eid}

metrics_path = EXP_DIR / "metrics_cl.json"
metrics = json.loads(metrics_path.read_text())
{acc_matrix_comment if (acc_matrix_comment := acc_matrix_load_comment) else ""}
{acc_matrix_reconstruction}

aa    = metrics["acc_final"]
af    = metrics["avg_forgetting"]
bwt   = metrics["backward_transfer"]
per_task_acc = metrics["per_task_acc"]
ram_b = metrics["ram_peak_bytes"]
lat   = metrics["inference_latency_ms"]
n_par = metrics["n_params"]

print("=" * 58)
print(f"  Modèle         : {mn}")
print(f"  AA             = {{aa:.4f}}")
print(f"  AF             = {{af:.4f}}")
print(f"  BWT            = {{bwt:+.4f}}")
print(f"  per_task_acc   = {{[round(v, 4) for v in per_task_acc]}}")
print(f"  RAM peak       = {{ram_b}} B ({{ram_b/1024:.2f}} Ko)")
print(f"  Latence        = {{lat:.5f}} ms")
print(f"  n_params       = {{n_par}}")
print(f"  Budget 64 Ko   : {{'OK' if ram_b <= 65536 else 'DEPASSE'}}")
print("=" * 58)
print("\\nMatrice acc (3×3) :")
print(acc_matrix_np)"""

    # --- Cell 3: accuracy matrix heatmap ---
    acc_matrix_code = f"""\
# Section 3 — Matrice d'accuracy (heatmap)
# acc_matrix_np[i, j] = accuracy sur tâche j après entraînement sur tâche i
# Triangle supérieur = NaN (tâche pas encore vue)

fig = plot_accuracy_matrix(
    acc_matrix_np,
    task_names=TASK_NAMES,
    title=f"{{MODEL_NAME}} — cwru/by_fault_type",
)
save_figure(fig, FIGURES_DIR / "accuracy_matrix.png")
display(Image(str(FIGURES_DIR / "accuracy_matrix.png")))"""

    # --- Cell 4: per-task barplot ---
    barplot_code = f"""\
# Section 4 — Barplot accuracy finale par tâche (Ball / Inner Race / Outer Race)

TASK_COLORS = ["#2196F3", "#FF9800", "#9C27B0"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(TASK_NAMES, per_task_acc, color=TASK_COLORS, edgecolor="black", linewidth=0.6)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy finale", fontsize=11)
ax.set_title(f"{{MODEL_NAME}} — Accuracy par type de défaut (cwru/by_fault_type)", fontsize=12, fontweight="bold")
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Seuil 50%")

for bar, val in zip(bars, per_task_acc):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        min(val + 0.02, 1.0),
        f"{{val:.3f}}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_figure(fig, FIGURES_DIR / "per_task_accuracy_bar.png")
display(Image(str(FIGURES_DIR / "per_task_accuracy_bar.png")))"""

    # --- Cell 5: forgetting curve ---
    forgetting_code = f"""\
# Section 5 — Courbe d'oubli par tâche
# Montre l'évolution de l'accuracy sur les tâches passées au fil de l'entraînement

fig = plot_forgetting_curve(
    acc_matrix_np,
    task_names=TASK_NAMES,
    title=f"{{MODEL_NAME}} — Évolution accuracy par tâche (cwru/by_fault_type)",
)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
display(Image(str(FIGURES_DIR / "forgetting_curve.png")))"""

    # --- Cell 6: RAM profile ---
    ram_code = f"""\
# Section 6 — Profil mémoire : RAM peak vs budget STM32N6 (64 Ko)

RAM_BUDGET = 65_536  # bytes = 64 Ko

fig, ax = plt.subplots(figsize=(6, 3.5))

labels  = [f"{{MODEL_NAME}}\\n(RAM peak)", "Budget STM32N6\\n(64 Ko)"]
values  = [ram_b, RAM_BUDGET]
colors  = ["#1f77b4" if ram_b <= RAM_BUDGET else "#d62728", "#2ca02c"]

bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.6, height=0.4)
ax.axvline(x=RAM_BUDGET, color="red", linestyle="--", linewidth=1.5, label="Limite 64 Ko")

for bar, val in zip(bars, values):
    ax.text(
        val + RAM_BUDGET * 0.01, bar.get_y() + bar.get_height() / 2,
        f"{{val:,}} B ({{val/1024:.1f}} Ko)",
        va="center", ha="left", fontsize=10,
    )

pct = ram_b / RAM_BUDGET * 100
status = "✅ Dans budget" if ram_b <= RAM_BUDGET else "❌ Dépasse budget"
ax.set_title(
    f"{{MODEL_NAME}} — RAM STM32N6 : {{ram_b:,}} B = {{pct:.1f}}% du budget\\n{{status}}",
    fontsize=11, fontweight="bold",
)
ax.set_xlabel("Octets (B)", fontsize=10)
ax.set_xlim(0, max(RAM_BUDGET, ram_b) * 1.25)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_figure(fig, FIGURES_DIR / "ram_vs_budget.png")
display(Image(str(FIGURES_DIR / "ram_vs_budget.png")))"""

    # --- Cell 7: summary table + acceptance check ---
    summary_code = f"""\
# Section 7 — Tableau récapitulatif + comparaison baseline single-task ({bline_exp})

baseline_path = BASELINE_DIR / "metrics_single_task.json"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())
    acc_st  = baseline.get("accuracy") or baseline.get("acc_final", float("nan"))
    ram_st  = baseline.get("ram_peak_bytes", 0)
    lat_st  = baseline.get("inference_latency_ms", 0.0)
    n_par_st = baseline.get("n_params", 0)
else:
    display(Markdown("> ⚠️ Baseline {bline_exp} non disponible."))
    acc_st = float("nan")
    ram_st = ram_b
    lat_st = lat
    n_par_st = n_par

display(Markdown(f"### Résultats finaux — {{MODEL_NAME}} — cwru/by_fault_type ({eid})"))

table_md = f\"\"\"
| Métrique | Valeur CL | Baseline single-task |
|----------|-----------|---------------------|
| **AA (avg accuracy)** | {{aa:.4f}} | {{acc_st:.4f}} |
| **AF (avg forgetting)** | {{af:.4f}} | — |
| **BWT** | {{bwt:+.4f}} | — |
| **RAM peak** | {{ram_b:,}} B ({{ram_b/1024:.2f}} Ko) | {{ram_st:,}} B ({{ram_st/1024:.2f}} Ko) |
| **Latence** | {{lat:.5f}} ms | {{lat_st:.5f}} ms |
| **n_params** | {{n_par}} | {{n_par_st}} |
| **Budget 64 Ko** | {{'✅' if ram_b <= 65536 else '❌'}} | {{'✅' if ram_st <= 65536 else '❌'}} |
\"\"\"
display(Markdown(table_md))

# Vérification critères d'acceptation (S12-07)
print("=" * 58)
print("  Critères d'acceptation (S12-07)")
print("=" * 58)
expected_figs = [
    "accuracy_matrix.png",
    "per_task_accuracy_bar.png",
    "forgetting_curve.png",
    "ram_vs_budget.png",
    "summary_table.png",
]
for fig_name in expected_figs:
    p = FIGURES_DIR / fig_name
    status = "OK" if p.exists() else "MANQUANTE"
    print(f"  [{{status}}] {{fig_name}}")

print()
print(f"  [{{'OK' if ram_b <= 65536 else 'FAIL'}}] RAM = {{ram_b}} B (budget ≤ 65 536 B)")
print(f"  [{{'OK' if lat < 100.0 else 'WARN'}}] Latence = {{lat:.5f}} ms (contrainte ≤ 100 ms)")

# --- Sauvegarde du tableau récapitulatif ---
fig_table, ax_t = plt.subplots(figsize=(8, 2.5))
ax_t.axis("off")
col_labels = ["Métrique", "CL ({mn})", "Baseline single-task"]
row_data = [
    ["AA", f"{{aa:.4f}}", f"{{acc_st:.4f}}"],
    ["AF", f"{{af:.4f}}", "—"],
    ["BWT", f"{{bwt:+.4f}}", "—"],
    ["RAM (Ko)", f"{{ram_b/1024:.2f}}", f"{{ram_st/1024:.2f}}"],
    ["Latence (ms)", f"{{lat:.5f}}", f"{{lat_st:.5f}}"],
    ["n_params", str(n_par), str(n_par_st)],
]
tbl = ax_t.table(
    cellText=row_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.5)
fig_table.tight_layout()
save_figure(fig_table, FIGURES_DIR / "summary_table.png")
display(Image(str(FIGURES_DIR / "summary_table.png")))"""

    cells = [
        cell_md(header_md),
        cell_code(setup_code),
        cell_code(load_code),
        cell_code(acc_matrix_code),
        cell_code(barplot_code),
        cell_code(forgetting_code),
        cell_code(ram_code),
        cell_code(summary_code),
    ]

    return make_nb(cells)


# ---------------------------------------------------------------------------
# Comparison notebook builder
# ---------------------------------------------------------------------------

def build_comparison_notebook() -> dict:
    header_md = """\
# Comparaison 6 modèles — CWRU Bearing Dataset — by_fault_type

| Champ | Valeur |
|-------|--------|
| **Scénario** | by_fault_type : Ball → Inner Race → Outer Race (3 tâches) |
| **Modèles** | EWC · HDC · TinyOL · KMeans · Mahalanobis · DBSCAN |
| **Dataset** | CWRU Bearing Dataset (CWRU) — 9 features statistiques |
| **Sprint** | 12 — S12-07 |

Ce notebook agrège les résultats des expériences **exp_074, exp_075, exp_076, exp_077, exp_078, exp_079**.

**Figures générées** :
1. `comparison_aa_af_bwt.png` — Barplot groupé AA/AF/BWT (6 modèles)
2. `scatter_af_vs_ram.png` — Scatter AF vs RAM (trade-off oubli/contrainte embarquée)
3. `ranking_models.png` — Ranking par score composite (AA − AF)"""

    setup_code = """\
# Section 1 — Setup + chargement normalisé des 6 modèles
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

# --- CWD navigation ---
_cwd = Path(".").resolve()
if _cwd.name == "cwru_by_fault_type":
    os.chdir(_cwd.parent.parent.parent)
elif _cwd.name == "cl_eval":
    os.chdir(_cwd.parent.parent)
elif _cwd.name == "notebooks":
    os.chdir(_cwd.parent)
REPO_ROOT = Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.plots import plot_metrics_comparison, save_figure

FIGURES_DIR = REPO_ROOT / "notebooks/figures/cl_evaluation/comparison/cwru/by_fault_type"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES  = ["Ball", "Inner Race", "Outer Race"]
MODEL_ORDER = ["EWC", "HDC", "TinyOL", "KMeans", "Mahalanobis", "DBSCAN"]
RAM_BUDGET  = 65_536

MODEL_EXP_MAP = {
    "EWC":         "exp_074_ewc_cwru_by_fault_type",
    "HDC":         "exp_075_hdc_cwru_by_fault_type",
    "TinyOL":      "exp_076_tinyol_cwru_by_fault_type",
    "KMeans":      "exp_077_kmeans_cwru_by_fault_type",
    "Mahalanobis": "exp_078_mahalanobis_cwru_by_fault_type",
    "DBSCAN":      "exp_079_dbscan_cwru_by_fault_type",
}

BASELINE_EXP_MAP = {
    "EWC":         "exp_068_ewc_cwru_single_task",
    "HDC":         "exp_069_hdc_cwru_single_task",
    "TinyOL":      "exp_070_tinyol_cwru_single_task",
    "KMeans":      "exp_071_kmeans_cwru_single_task",
    "Mahalanobis": "exp_072_mahalanobis_cwru_single_task",
    "DBSCAN":      "exp_073_dbscan_cwru_single_task",
}


def load_acc_matrix(raw_mat: list, ragged: bool = False) -> np.ndarray:
    mat = np.full((3, 3), np.nan, dtype=float)
    for i, row in enumerate(raw_mat):
        for j, v in enumerate(row):
            if v is not None:
                mat[i, j] = v
    return mat


BASE = REPO_ROOT / "experiments"
results     = {}
acc_matrices = {}
baselines   = {}

for model in MODEL_ORDER:
    exp_dir = BASE / MODEL_EXP_MAP[model] / "results"
    metrics_path = exp_dir / "metrics_cl.json"
    if not metrics_path.exists():
        print(f"⚠️  {MODEL_EXP_MAP[model]} non disponible — mock activé")
        results[model] = {"aa": 0.0, "af": 0.0, "bwt": 0.0,
                          "ram_peak_bytes": 0, "inference_latency_ms": 0.0,
                          "n_params": 0, "per_task_acc": [0.0, 0.0, 0.0]}
        acc_matrices[model] = np.zeros((3, 3))
        continue
    raw = json.loads(metrics_path.read_text())
    # Schéma plat uniforme pour CWRU
    results[model] = {
        "aa":  raw["acc_final"],
        "af":  raw["avg_forgetting"],
        "bwt": raw["backward_transfer"],
        "ram_peak_bytes":      raw["ram_peak_bytes"],
        "inference_latency_ms": raw["inference_latency_ms"],
        "n_params": raw["n_params"],
        "per_task_acc": raw["per_task_acc"],
    }
    acc_matrices[model] = load_acc_matrix(raw["acc_matrix"])

    # Baseline single-task
    bline_path = BASE / BASELINE_EXP_MAP[model] / "results" / "metrics_single_task.json"
    if bline_path.exists():
        braw = json.loads(bline_path.read_text())
        baselines[model] = braw.get("accuracy") or braw.get("acc_final", float("nan"))
    else:
        baselines[model] = float("nan")

    r = results[model]
    print(f"{model:12s} → AA={r['aa']:.4f} AF={r['af']:.4f} BWT={r['bwt']:+.4f} "
          f"RAM={r['ram_peak_bytes']/1024:5.1f}Ko lat={r['inference_latency_ms']:.5f}ms n={r['n_params']}")

print(f"\\n6 modèles chargés | {datetime.now():%Y-%m-%d %H:%M}")"""

    comparison_table_code = """\
# Section 2 — Tableau comparatif AA / AF / BWT / RAM / latence

RAM_LIMIT = 65_536
header = "| Modèle | AA ↑ | AF ↓ | BWT | RAM | Latence | n_params | Baseline ST |"
sep    = "|--------|:----:|:----:|:---:|:---:|:-------:|:--------:|:-----------:|"
rows   = [header, sep]

for model in MODEL_ORDER:
    r = results[model]
    ram_b = r["ram_peak_bytes"]
    ram_s = f"{ram_b/1024:.1f} Ko{'  ⚠️' if ram_b > RAM_LIMIT else ''}"
    bst   = baselines.get(model, float("nan"))
    bst_s = f"{bst:.4f}" if not (isinstance(bst, float) and bst != bst) else "—"
    line = (
        f"| {model} | {r['aa']:.4f} | {r['af']:.4f} | {r['bwt']:+.4f} | "
        f"{ram_s} | {r['inference_latency_ms']:.5f} ms | {r['n_params']} | {bst_s} |"
    )
    rows.append(line)

display(Markdown("### Tableau comparatif — 6 modèles CL (cwru/by_fault_type)\\n\\n" + "\\n".join(rows)))"""

    barplot_code = """\
# Section 3 — Barplot groupé AA / AF / BWT (6 modèles) → comparison_aa_af_bwt.png

fig = plot_metrics_comparison(
    results,
    metrics=["aa", "af", "bwt"],
    title="AA / AF / BWT — CWRU/by_fault_type (6 modèles)",
)
save_figure(fig, FIGURES_DIR / "comparison_aa_af_bwt.png")
display(Image(str(FIGURES_DIR / "comparison_aa_af_bwt.png")))"""

    scatter_code = """\
# Section 4 — Scatter AF vs RAM (trade-off oubli / contrainte embarquée) → scatter_af_vs_ram.png

SCATTER_MARKERS = {
    "EWC":         ("o", "#1f77b4"),
    "HDC":         ("s", "#ff7f0e"),
    "TinyOL":      ("^", "#2ca02c"),
    "KMeans":      ("D", "#d62728"),
    "Mahalanobis": ("P", "#9467bd"),
    "DBSCAN":      ("*", "#8c564b"),
}

fig, ax = plt.subplots(figsize=(8, 5))

ax.axvspan(0, RAM_BUDGET / 1024, alpha=0.07, color="green", label=f"Zone STM32 ≤ 64 Ko")
ax.axvline(RAM_BUDGET / 1024, color="red", linestyle="--", linewidth=1.5, label="Budget 64 Ko")

for name in MODEL_ORDER:
    r = results[name]
    ram_kb = r["ram_peak_bytes"] / 1024
    af_val = r["af"]
    marker, color = SCATTER_MARKERS[name]
    ax.scatter(ram_kb, af_val, marker=marker, color=color, s=130, zorder=5, label=name,
               edgecolor="black", linewidth=0.5)
    ax.annotate(name, xy=(ram_kb, af_val), xytext=(ram_kb * 1.04, af_val + 0.001), fontsize=9)

ax.set_xlabel("RAM peak (Ko)", fontsize=11)
ax.set_ylabel("AF (Average Forgetting) ↓", fontsize=11)
ax.set_title(
    "Trade-off embarqué : RAM vs. Oubli\\n(CWRU/by_fault_type — Gap 2 STM32 ≤ 64 Ko)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, FIGURES_DIR / "scatter_af_vs_ram.png")
display(Image(str(FIGURES_DIR / "scatter_af_vs_ram.png")))"""

    ranking_code = """\
# Section 5 — Ranking des modèles par score composite (AA − AF) → ranking_models.png

scores = {model: results[model]["aa"] - results[model]["af"] for model in MODEL_ORDER}
sorted_models = sorted(scores, key=scores.get, reverse=True)
sorted_scores = [scores[m] for m in sorted_models]

RANK_COLORS = {
    "EWC":         "#1f77b4",
    "HDC":         "#ff7f0e",
    "TinyOL":      "#2ca02c",
    "KMeans":      "#d62728",
    "Mahalanobis": "#9467bd",
    "DBSCAN":      "#8c564b",
}

fig, ax = plt.subplots(figsize=(8, 4))
colors_bar = [RANK_COLORS[m] for m in sorted_models]
bars = ax.barh(sorted_models, sorted_scores, color=colors_bar, edgecolor="black", linewidth=0.5)

for bar, score in zip(bars, sorted_scores):
    ax.text(
        score + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{score:.4f}", va="center", ha="left", fontsize=9,
    )

ax.set_xlabel("Score composite (AA − AF)", fontsize=11)
ax.set_title(
    "Ranking des modèles CL — cwru/by_fault_type\\n(Score = AA − AF : performance nette de l'oubli)",
    fontsize=12, fontweight="bold",
)
ax.set_xlim(0, max(sorted_scores) * 1.15)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
fig.tight_layout()
save_figure(fig, FIGURES_DIR / "ranking_models.png")
display(Image(str(FIGURES_DIR / "ranking_models.png")))"""

    baseline_delta_code = """\
# Section 6 — Comparaison avec baseline single-task (exp_068–073)
# Delta AA = AA_CL − acc_single_task

display(Markdown("### Comparaison CL vs Baseline single-task (cwru/by_fault_type)"))

header = "| Modèle | Baseline ST (acc) | AA CL | Delta AA | AF CL |"
sep    = "|--------|:-----------------:|:-----:|:--------:|:-----:|"
rows   = [header, sep]

for model in MODEL_ORDER:
    r     = results[model]
    bst   = baselines.get(model, float("nan"))
    delta = r["aa"] - bst if not (isinstance(bst, float) and bst != bst) else float("nan")
    bst_s   = f"{bst:.4f}"   if not (isinstance(bst, float) and bst != bst) else "—"
    delta_s = f"{delta:+.4f}" if not (isinstance(delta, float) and delta != delta) else "—"
    sign = "↑" if isinstance(delta, float) and delta == delta and delta >= 0 else "↓"
    rows.append(
        f"| {model} | {bst_s} | {r['aa']:.4f} | {delta_s} {sign} | {r['af']:.4f} |"
    )

display(Markdown("\\n".join(rows)))

print("\\nNote : Delta AA > 0 indique que le modèle CL surpasse la baseline single-task.")
print("       Delta AA < 0 indique une perte due au scénario multi-tâche.")
print("\\nCritères d'acceptation (S12-07) :")
for fig_name in ["comparison_aa_af_bwt.png", "scatter_af_vs_ram.png", "ranking_models.png"]:
    p = FIGURES_DIR / fig_name
    print(f"  [{'OK' if p.exists() else 'MANQUANTE'}] {fig_name}")"""

    cells = [
        cell_md(header_md),
        cell_code(setup_code),
        cell_code(comparison_table_code),
        cell_code(barplot_code),
        cell_code(scatter_code),
        cell_code(ranking_code),
        cell_code(baseline_delta_code),
    ]

    return make_nb(cells)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Generate 6 model notebooks
    for cfg in MODEL_CONFIGS:
        nb = build_model_notebook(cfg)
        path = OUT_DIR / f"{cfg['model_key']}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"[OK] {path}")

    # Generate comparison notebook
    nb_comp = build_comparison_notebook()
    path_comp = OUT_DIR / "comparison.ipynb"
    path_comp.write_text(json.dumps(nb_comp, indent=1, ensure_ascii=False))
    print(f"[OK] {path_comp}")

    print(f"\n7 notebooks générés dans {OUT_DIR}")


if __name__ == "__main__":
    main()
