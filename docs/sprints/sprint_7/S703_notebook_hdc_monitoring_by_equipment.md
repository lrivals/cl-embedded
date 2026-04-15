# S7-03 — Notebook HDC — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-03 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_002_hdc_dataset2/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/hdc.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **HDC (Hyperdimensional Computing, D=1024)** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_002_hdc_dataset2`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/hdc/monitoring/by_equipment/`.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── hdc.ipynb                           ← notebook créé

notebooks/figures/cl_evaluation/hdc/monitoring/by_equipment/
├── acc_matrix.png
├── forgetting_curve.png
├── confusion_matrix_grid.png
├── roc_curves.png
└── feature_space_pca.png
```

---

## Contenu du notebook (8 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Évaluation CL — HDC (D=1024) — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | HDC Hyperdimensional Computing (D=1024, 2 048 paramètres) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_002 — voir experiments/exp_002_hdc_dataset2/config_snapshot.yaml |
| **Date** | {date d'exécution} |
```

### Section 1 — Setup & imports

```python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path("../..").resolve()))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.plots import (
    plot_accuracy_matrix, plot_forgetting_curve,
    plot_confusion_matrix_grid, plot_roc_curves_per_task, save_figure
)
from src.evaluation.metrics import compute_cl_metrics

EXP_DIR     = Path("../../experiments/exp_002_hdc_dataset2/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/hdc/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump", "Turbine", "Compressor"]
MODEL_NAME = "HDC"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / "metrics.json"
acc_matrix_path = EXP_DIR / "acc_matrix.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

cl = metrics.get("cl_metrics", metrics)
print(f"AA  = {cl['aa']:.4f}")
print(f"AF  = {cl['af']:.4f}")
print(f"BWT = {cl['bwt']:.4f}")
print(f"RAM = {metrics['ram_peak_bytes']} B")
print(f"Latence = {metrics['inference_latency_ms']:.5f} ms")
```

### Section 3 — Matrice d'accuracy (heatmap)

```python
# Diagonale = perf immédiate post-training, sous-diagonale = oubli
# Note HDC : AF=0.0 — pas d'oubli catastophique par construction (prototypes par classe)
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# HDC : AF=0.0 par design (les hypervecteurs de classe ne sont pas écrasés)
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche

```python
preds_dict = ...  # à adapter selon format exp_002
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES,
                                 model_name=MODEL_NAME)
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
plt.show()
```

### Section 6 — Courbes ROC par tâche

```python
fig = plot_roc_curves_per_task(preds_dict, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "roc_curves.png")
plt.show()
```

### Section 7 — Espace des features (PCA 2D)

```python
from src.evaluation.feature_space_plots import plot_clustering_with_correctness
# Projection PCA 2D colorée par tâche et par correctness
# Note : visualiser les hypervecteurs de prototype par classe si disponibles
fig = plot_clustering_with_correctness(...)
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
plt.show()
```

### Section 8 — Tableau récapitulatif (Markdown)

```python
cl  = metrics.get("cl_metrics", metrics)
aa  = cl["aa"]
af  = cl["af"]
bwt = cl["bwt"]
ram = metrics["ram_peak_bytes"] / 1024
lat = metrics["inference_latency_ms"]
n   = metrics.get("n_params", 2048)

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {ram:.2f} Ko | {lat:.5f} ms | {n} |")
```

---

## Métriques attendues (exp_002)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | 0.8698 |
| `af` (AF) | 0.0000 ← zéro oubli par construction |
| `bwt` (BWT) | +0.0019 ← transfert positif |
| `ram_peak_bytes` | 14 504 B (14.2 Ko) |
| `inference_latency_ms` | 0.04758 ms |
| `n_params` | 2 048 (hypervecteurs de prototype) |

> **Note scientifique** : HDC présente AF=0.0 par design (les prototypes de classe sont des hypervecteurs additifs, non destructifs). Cependant, l'AA (0.8698) est plus faible que EWC (0.9824), illustrant le compromis stabilité/plasticité. À discuter dans S7-13.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/hdc.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF=0.0 (courbes plates)
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `roc_curves.png` sauvegardé
- [ ] `feature_space_pca.png` sauvegardé
- [ ] Tableau récapitulatif final : AA=0.8698, AF=0.0000, BWT=+0.0019, RAM=14.2 Ko, latence=0.04758 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/hdc.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/hdc/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : HDC AF=0.0 mais AA=0.8698 (plus faible que EWC et TinyOL) — documenter ce compromis dans le manuscrit comme illustration du dilemme stabilité/plasticité.
- `TODO(dorra)` : La dimension D=1024 est-elle optimale pour ce dataset ? Tester D=512 et D=2048 pour analyse de sensibilité (hors périmètre sprint 7, à planifier sprint 10).
