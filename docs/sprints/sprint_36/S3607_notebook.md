# S3607 — Notebook de comparaison PC ↔ board (tous les plots)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 36 |
| **Priorité** | 🔴 Critique — livrable visuel central : « faire un notebook avec tous les plots ». |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 5h |
| **Dépendances** | S3606 ✅ (`exp_S36_summary.json`) + S3602–S3605 (acc_matrix, samples, parité) · `src/evaluation/plots.py` ✅ (`save_figure`, `plot_accuracy_matrix`, `plot_forgetting_curve`, `plot_metrics_comparison`, `plot_performance_by_task_bar`) · gabarit `notebooks/cl_eval/threshold_impact/comparison.ipynb` ✅ |
| **Fichiers cibles** | `notebooks/cl_eval/pc_board_ewc/comparison.ipynb`, `docs/figures/sprint36_pc_board_ewc/*.png` |
| **Références** | `notebooks/cl_eval/cwru_by_severity/comparison.ipynb` (variété de plots) · `notebooks/sprint30_pairs_disagreement.ipynb` (parité/désaccords) |

---

## Contexte

Tous les chiffres produits doivent être visualisables d'un coup d'œil et comparables PC↔board.
On suit les conventions des notebooks existants (chargement JSON, `save_figure` dpi=150,
figures → `docs/figures/`, structure markdown+code alternée).

## Spec

Notebook `notebooks/cl_eval/pc_board_ewc/comparison.ipynb`, figures → `docs/figures/sprint36_pc_board_ewc/`.

**Chargement** (pattern projet) :

```python
import json; from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.evaluation.plots import save_figure, plot_accuracy_matrix, plot_forgetting_curve

ROOT = Path.cwd()
while not (ROOT / "experiments").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
summary = json.load(open(ROOT / "experiments/exp_S36_summary.json"))
FIGS = ROOT / "docs/figures/sprint36_pc_board_ewc"; FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["pronostia", "monitoring"]; CONDITIONS = ["5feat", "all"]
LATENCY_BUDGET_MS = 100.0; RAM_BUDGET_BYTES = 65_536
```

**Plots à produire** (un par cellule, figure sauvegardée) :

1. `accuracy_per_task.png` — acc par tâche PC vs board (barres groupées, par dataset). `plot_performance_by_task_bar`.
2. `cl_accuracy_matrix.png` — matrices CL PC et board côte à côte. `plot_accuracy_matrix`.
3. `accfinal_vs_forgetting.png` — scatter acc finale vs oubli (AF), PC vs board.
4. `forgetting_curves.png` — courbes d'oubli par tâche. `plot_forgetting_curve`.
5. `latency_inference_vs_update.png` — board : inférence (S3603) vs inférence+MAJ (S3604), échelle log, ligne Gap 2 (100 ms).
6. `latency_pc_vs_board.png` — comparaison latence PC vs board.
7. `accuracy_vs_ram.png` — scatter acc vs RAM, zone budget STM32 (`axvspan`), PC (`ram_peak_bytes`) vs board (`.bss`).
8. `f1_rocauc_pc_vs_board.png` — F1 + ROC-AUC PC vs board (barres groupées). `plot_metrics_comparison`.
9. `prediction_parity.png` — table concordance + diagonale `pred_pc` vs `pred_board` + matrice de confusion des désaccords (depuis `exp_S36_parity_*`).
10. `condition_5feat_vs_all.png` — effet condition sur perf / parité / latence.
11. **Tableau récapitulatif final** (markdown/DataFrame) — tous les métriques.

**Règles** :
- `save_figure(fig, FIGS / "...png")` (dpi=150 projet).
- Gérer gracieusement les champs `null`/« à mesurer » (ne pas planter le notebook).
- Exécution finale via `nbconvert` (reproductibilité).

## Vérification

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/cl_eval/pc_board_ewc/comparison.ipynb
ls docs/figures/sprint36_pc_board_ewc/   # 10 PNG attendus
```

## Implémentation (✅)

- [x] `notebooks/cl_eval/pc_board_ewc/comparison.ipynb` créé (chargement `exp_S36_summary.json` +
      PC `results.json` + parité ; 10 figures + tableau récap DataFrame).
- [x] Helpers `src/evaluation/plots.py` réutilisés (`plot_accuracy_matrix`, `plot_forgetting_curve`,
      `plot_performance_by_task_bar`, `save_figure` dpi=150). Champs `null` gérés sans planter.
- [x] **Exécuté via nbconvert sans erreur** → **10 PNG** dans `docs/figures/sprint36_pc_board_ewc/`.

### Résultats (27 juin 2026)

10 figures : `accuracy_per_task`, `cl_accuracy_matrix`, `accfinal_vs_forgetting`, `forgetting_curves`,
`latency_inference_vs_update` (ligne Gap 2 100 ms), `latency_pc_vs_board`, `accuracy_vs_ram`
(`axvspan` 64 Ko), `f1_rocauc_pc_vs_board`, `prediction_parity` (concordance + confusion désaccords
online), `condition_5feat_vs_all`. Note matrices CL : la board ne produit pas de matrice `T×T`
(accuracy online par tâche, plot 1) → les heatmaps `plot_accuracy_matrix` affichent les matrices **PC**
des deux datasets côte à côte (honnête, documenté en cellule).
