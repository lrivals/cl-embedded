# S7-04 — Notebook KMeans — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-04 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_005_unsupervised_dataset2/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/kmeans.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **K-Means (non-supervisé)** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_005_unsupervised_dataset2`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/kmeans/monitoring/by_equipment/`.

> **Note** : K-Means est un modèle de **détection d'anomalie non-supervisée** — il ne reçoit pas les labels à l'entraînement. La section ROC est remplacée par une section AUROC (aire sous la courbe ROC pour la détection d'anomalie). Il n'y a pas de `n_params` (modèle basé sur des centroïdes).

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── kmeans.ipynb                        ← notebook créé

notebooks/figures/cl_evaluation/kmeans/monitoring/by_equipment/
├── acc_matrix.png
├── forgetting_curve.png
├── confusion_matrix_grid.png
├── auroc_curve.png                     ← AUROC (remplace roc_curves.png)
└── feature_space_pca.png
```

---

## Contenu du notebook (8 sections)

### Section 0 — En-tête (cellule Markdown)

```markdown
# Évaluation CL — K-Means (non-supervisé) — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | K-Means (détection d'anomalie, non-supervisé) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_005 — voir experiments/exp_005_unsupervised_dataset2/config_snapshot.yaml |
| **Date** | {date d'exécution} |

> **Modèle non-supervisé** : K-Means ne reçoit pas les labels à l'entraînement.
> L'AUROC est la métrique principale pour évaluer la détection d'anomalie.
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
    plot_confusion_matrix_grid, save_figure
)
from src.evaluation.metrics import compute_cl_metrics

EXP_DIR     = Path("../../experiments/exp_005_unsupervised_dataset2/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/kmeans/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Pump", "Turbine", "Compressor"]
MODEL_NAME = "KMeans"
METRICS_FILE = "metrics_kmeans.json"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / METRICS_FILE
acc_matrix_path = EXP_DIR / "acc_matrix_kmeans.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

print(f"AA       = {metrics['aa']:.4f}")
print(f"AF       = {metrics['af']:.4f}")
print(f"BWT      = {metrics['bwt']:.4f}")
print(f"AUROC    = {metrics['auroc_avg']:.4f}")
print(f"RAM      = {metrics['ram_peak_bytes']} B")
print(f"Latence  = {metrics['inference_latency_ms']:.5f} ms")
```

### Section 3 — Matrice d'accuracy (heatmap)

```python
# Note : accuracy calculée par seuillage du score d'anomalie (distance au centroïde le plus proche)
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche (détection anomalie)

```python
# Seuil optimal pour chaque tâche (maximise F1 ou Youden J)
preds_dict = ...  # à adapter selon format exp_005
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES,
                                 model_name=MODEL_NAME)
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
plt.show()
```

### Section 6 — Courbe AUROC (détection d'anomalie)

```python
from sklearn.metrics import roc_curve, auc
# AUROC par tâche — la distance au centroïde est le score d'anomalie
fig, axes = plt.subplots(1, len(TASK_NAMES), figsize=(15, 4))
auroc_per_task = metrics.get("auroc_per_task", [])
for i, (task, ax) in enumerate(zip(TASK_NAMES, axes)):
    # Tracer la courbe ROC pour chaque tâche
    ax.set_title(f"AUROC — {task}")
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "auroc_curve.png")
plt.show()
```

### Section 7 — Espace des features (PCA 2D)

```python
from src.evaluation.feature_space_plots import plot_clustering_with_correctness
# Centroïdes K-Means visualisés dans l'espace PCA 2D
fig = plot_clustering_with_correctness(...)
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
plt.show()
```

### Section 8 — Tableau récapitulatif (Markdown)

```python
aa     = metrics["aa"]
af     = metrics["af"]
bwt    = metrics["bwt"]
auroc  = metrics["auroc_avg"]
ram    = metrics["ram_peak_bytes"] / 1024
lat    = metrics["inference_latency_ms"]

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {auroc:.4f} | {ram:.2f} Ko | {lat:.5f} ms | — |")
```

---

## Métriques attendues (exp_005 — K-Means)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | 0.9433 |
| `af` (AF) | 0.0049 |
| `bwt` (BWT) | −0.0040 |
| `auroc_avg` | 0.9621 |
| `ram_peak_bytes` | 5 358 B (5.2 Ko) |
| `inference_latency_ms` | 0.39870 ms ← le plus lent des non-supervisés |
| `n_params` | — (centroïdes, non paramétrique) |

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/kmeans.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF ≈ 0.0049
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `auroc_curve.png` sauvegardé — AUROC ≈ 0.9621 par tâche
- [ ] `feature_space_pca.png` sauvegardé avec centroïdes visibles
- [ ] Tableau récapitulatif final : AA=0.9433, AF=0.0049, BWT=−0.0040, AUROC=0.9621, RAM=5.2 Ko, latence=0.39870 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/kmeans.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/kmeans/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Inclure aussi les résultats KNN et PCA (également dans exp_005) dans ce notebook, ou créer des notebooks séparés ? Actuellement seule la section K-Means est traitée.
- `FIXME(gap3)` : K-Means ne nécessite pas de labels à l'entraînement — documenter explicitement cet avantage dans la section discussion (économie de coût d'annotation en production industrielle).
