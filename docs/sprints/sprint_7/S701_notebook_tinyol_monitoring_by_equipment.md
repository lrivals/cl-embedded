# S7-01 — Notebook TinyOL — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-01 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_011_tinyol_dataset2/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/tinyol.ipynb` |
| **Statut** | ✅ Notebook exécuté bout-en-bout (2026-04-19), outputs embarqués, 5 figures régénérées |

---

## Objectif

Créer le notebook d'évaluation CL pour **TinyOL** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_011_tinyol_dataset2`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/tinyol/monitoring/by_equipment/`.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── tinyol.ipynb                        ← notebook créé

notebooks/figures/cl_evaluation/tinyol/monitoring/by_equipment/
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
# Évaluation CL — TinyOL — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | TinyOL (encodeur 184 params + tête OtO 10 params) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_011 — voir experiments/exp_011_tinyol_dataset2/config_snapshot.yaml |
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

EXP_DIR    = Path("../../experiments/exp_011_tinyol_dataset2/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/tinyol/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump", "Turbine", "Compressor"]
MODEL_NAME = "TinyOL"
```

### Section 2 — Chargement des résultats

```python
metrics_path = EXP_DIR / "metrics.json"
acc_matrix_path = EXP_DIR / "acc_matrix.npy"

metrics = json.loads(metrics_path.read_text())
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
# preds_dict chargé depuis les checkpoints ou résultats de prédiction
preds_dict = ...  # à adapter selon format exp_011
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
aa  = metrics["acc_final"]
af  = metrics["avg_forgetting"]
bwt = metrics["backward_transfer"]
ram = metrics["ram_peak_bytes"] / 1024
lat = metrics["inference_latency_ms"]

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {ram:.2f} Ko | {lat:.5f} ms | 194 |")
```

---

## Métriques attendues (exp_011)

| Métrique | Valeur |
|----------|--------|
| `acc_final` (AA) | 0.9123 |
| `avg_forgetting` (AF) | 0.0079 |
| `backward_transfer` (BWT) | −0.0029 |
| `ram_peak_bytes` | 4 379 B (4.3 Ko) |
| `inference_latency_ms` | 0.00975 ms |
| `n_params` | 194 (184 encodeur + 10 OtO) |

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/tinyol.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF ≈ 0.0079
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `roc_curves.png` sauvegardé
- [ ] `feature_space_pca.png` sauvegardé
- [ ] Tableau récapitulatif final : AA=0.9123, AF=0.0079, BWT=−0.0029, RAM=4.3 Ko, latence=0.00975 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/tinyol.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/tinyol/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : La tête OtO de TinyOL nécessite-t-elle un affichage spécifique dans la section confusion matrix (séparation encodeur/classificateur) ?
- `FIXME(gap1)` : TinyOL AA=0.9123 vs EWC AA=0.9824 sur ce scénario — noter cet écart dans la section discussion du notebook de comparaison (S7-13).
