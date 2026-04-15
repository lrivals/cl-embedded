# S7-02 — Notebook EWC — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-02 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_001_ewc_dataset2/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/ewc.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **EWC Online + MLP** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_001_ewc_dataset2`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/ewc/monitoring/by_equipment/`.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── ewc.ipynb                           ← notebook créé

notebooks/figures/cl_evaluation/ewc/monitoring/by_equipment/
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
# Évaluation CL — EWC Online + MLP — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | EWC Online + MLP (705 paramètres) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_001 — voir experiments/exp_001_ewc_dataset2/config_snapshot.yaml |
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

EXP_DIR     = Path("../../experiments/exp_001_ewc_dataset2/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/ewc/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump", "Turbine", "Compressor"]
MODEL_NAME = "EWC"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / "metrics.json"
acc_matrix_path = EXP_DIR / "acc_matrix.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

# Extraire les métriques EWC (clé "ewc" dans le JSON)
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
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# Courbe plate = pas d'oubli, courbe descendante = oubli catastrophique
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche

```python
preds_dict = ...  # à adapter selon format exp_001
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
n   = metrics.get("n_params", 705)

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {ram:.2f} Ko | {lat:.5f} ms | {n} |")
```

---

## Métriques attendues (exp_001)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | 0.9824 |
| `af` (AF) | 0.0010 |
| `bwt` (BWT) | 0.0000 |
| `ram_peak_bytes` | 1 171 B (1.1 Ko) ← meilleure RAM supervisée |
| `inference_latency_ms` | 0.03580 ms |
| `n_params` | 705 |

> **Note** : EWC atteint le meilleur AA (0.9824) parmi tous les modèles supervisés sur ce scénario, avec la RAM la plus faible (1.1 Ko). À valoriser dans le notebook de comparaison S7-13.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/ewc.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF ≈ 0.0010 (quasi-nul)
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `roc_curves.png` sauvegardé
- [ ] `feature_space_pca.png` sauvegardé
- [ ] Tableau récapitulatif final : AA=0.9824, AF=0.0010, BWT=0.0000, RAM=1.1 Ko, latence=0.03580 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/ewc.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/ewc/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `FIXME(gap2)` : EWC RAM = 1 171 B (1.1 Ko) — bien sous la contrainte 64 Ko. Documenter la marge disponible pour la pénalité Fisher dans les notebooks manuscrit.
- `TODO(arnaud)` : Les matrices Fisher sont-elles sauvegardées dans les checkpoints ? Utile pour visualiser quels paramètres sont protégés par EWC.
