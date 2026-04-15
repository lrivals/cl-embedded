# S7-07 — Notebook TinyOL — monitoring_by_location

| Champ | Valeur |
|-------|--------|
| **ID** | S7-07 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_018_tinyol_monitoring_by_location/` (Sprint 6) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_location/tinyol.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **TinyOL** sur le scénario **monitoring_by_location** (Atlanta → Chicago → Houston → New York → San Francisco, 5 tâches) en chargeant les résultats de `exp_018_tinyol_monitoring_by_location`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/tinyol/monitoring/by_location/`.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_location/
└── tinyol.ipynb                        ← notebook créé

notebooks/figures/cl_evaluation/tinyol/monitoring/by_location/
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
# Évaluation CL — TinyOL — Dataset 2 Equipment Monitoring — by_location

| Champ | Valeur |
|-------|--------|
| **Modèle** | TinyOL (encodeur 184 params + tête OtO 10 params) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_location : Atlanta → Chicago → Houston → New York → San Francisco (5 tâches) |
| **Expérience** | exp_018 — voir experiments/exp_018_tinyol_monitoring_by_location/config_snapshot.yaml |
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

EXP_DIR     = Path("../../experiments/exp_018_tinyol_monitoring_by_location/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/tinyol/monitoring/by_location")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]
MODEL_NAME = "TinyOL"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / "metrics.json"
acc_matrix_path = EXP_DIR / "acc_matrix.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

print(f"AA  = {metrics['acc_final']:.4f}")
print(f"AF  = {metrics['avg_forgetting']:.4f}")
print(f"BWT = {metrics['backward_transfer']:.4f}")
print(f"RAM = {metrics['ram_peak_bytes']} B")
print(f"Latence = {metrics['inference_latency_ms']:.5f} ms")
```

### Section 3 — Matrice d'accuracy (heatmap)

```python
# Diagonale = perf immédiate post-training, sous-diagonale = oubli
# Matrice 5×5 pour les 5 locations
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_location")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# Courbe plate = pas d'oubli, courbe descendante = oubli catastrophique
# 5 courbes correspondant aux 5 locations
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche

```python
# preds_dict chargé depuis les checkpoints ou résultats de prédiction
preds_dict = ...  # à adapter selon format exp_018
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
# Projection PCA 2D colorée par tâche et par correctness — 5 couleurs distinctes
fig = plot_clustering_with_correctness(...)
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
plt.show()
```

### Section 8 — Tableau récapitulatif (Markdown)

```python
aa  = metrics["acc_final"]
af  = metrics["avg_forgetting"]
bwt = metrics["backward_transfer"]
ram = metrics["ram_peak_bytes"] / 1024
lat = metrics["inference_latency_ms"]

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {ram:.2f} Ko | {lat:.5f} ms | 194 |")
```

---

## Métriques attendues (exp_018)

| Métrique | Valeur |
|----------|--------|
| `acc_final` (AA) | À déterminer — expérience non encore exécutée |
| `avg_forgetting` (AF) | À déterminer |
| `backward_transfer` (BWT) | À déterminer |
| `ram_peak_bytes` | À déterminer (référence by_equipment : 4 379 B) |
| `inference_latency_ms` | À déterminer (référence by_equipment : 0.00975 ms) |
| `n_params` | 194 (184 encodeur + 10 OtO) — identique à by_equipment |

> **Note** : TinyOL utilise la même architecture (184+10 params) pour by_location. La tête OtO s'adapte à 5 tâches géographiques. Comparer AA par rapport au scénario by_equipment pour évaluer la sensibilité au type de drift.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_location/tinyol.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 5×5 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — 5 courbes distinctes
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `roc_curves.png` sauvegardé
- [ ] `feature_space_pca.png` sauvegardé
- [ ] Tableau récapitulatif final avec AA/AF/BWT/RAM/latence

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_location/tinyol.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/tinyol/monitoring/by_location').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : La tête OtO de TinyOL s'adapte-t-elle de la même manière pour 5 tâches géographiques que pour 3 types d'équipements ? La cardinalité plus élevée (5 vs 3) affecte-t-elle la convergence ?
- `FIXME(gap1)` : Comparer AA by_location vs AA by_equipment pour TinyOL — le drift géographique est-il plus difficile à apprendre que le drift par type d'équipement ?
