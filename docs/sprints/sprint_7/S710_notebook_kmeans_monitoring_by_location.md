# S7-10 — Notebook KMeans — monitoring_by_location

| Champ | Valeur |
|-------|--------|
| **ID** | S7-10 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_022_kmeans_monitoring_by_location/` (Sprint 6) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_location/kmeans.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **K-Means (non-supervisé)** sur le scénario **monitoring_by_location** (Atlanta → Chicago → Houston → New York → San Francisco, 5 tâches) en chargeant les résultats de `exp_022_kmeans_monitoring_by_location`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/kmeans/monitoring/by_location/`.

> **Note** : K-Means est un modèle de **détection d'anomalie non-supervisée** — il ne reçoit pas les labels à l'entraînement. La section ROC est remplacée par une section AUROC. Il n'y a pas de `n_params` (modèle basé sur des centroïdes).

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_location/
└── kmeans.ipynb                        ← notebook créé

notebooks/figures/cl_evaluation/kmeans/monitoring/by_location/
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
# Évaluation CL — K-Means (non-supervisé) — Dataset 2 Equipment Monitoring — by_location

| Champ | Valeur |
|-------|--------|
| **Modèle** | K-Means (détection d'anomalie, non-supervisé) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_location : Atlanta → Chicago → Houston → New York → San Francisco (5 tâches) |
| **Expérience** | exp_022 — voir experiments/exp_022_kmeans_monitoring_by_location/config_snapshot.yaml |
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

EXP_DIR     = Path("../../experiments/exp_022_kmeans_monitoring_by_location/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/kmeans/monitoring/by_location")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES   = ["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]
MODEL_NAME   = "KMeans"
METRICS_FILE = "metrics_kmeans.json"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / METRICS_FILE
acc_matrix_path = EXP_DIR / "acc_matrix_kmeans.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

print(f"AA      = {metrics['aa']:.4f}")
print(f"AF      = {metrics['af']:.4f}")
print(f"BWT     = {metrics['bwt']:.4f}")
print(f"AUROC   = {metrics['auroc_avg']:.4f}")
print(f"RAM     = {metrics['ram_peak_bytes']} B")
print(f"Latence = {metrics['inference_latency_ms']:.5f} ms")
```

### Section 3 — Matrice d'accuracy (heatmap)

```python
# Note : accuracy calculée par seuillage du score d'anomalie (distance au centroïde le plus proche)
# Matrice 5×5 pour les 5 locations
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_location")
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
preds_dict = ...  # à adapter selon format exp_022
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES,
                                 model_name=MODEL_NAME)
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
plt.show()
```

### Section 6 — Courbe AUROC (détection d'anomalie)

```python
from sklearn.metrics import roc_curve, auc
# AUROC par tâche — la distance au centroïde est le score d'anomalie
fig, axes = plt.subplots(1, len(TASK_NAMES), figsize=(20, 4))
auroc_per_task = metrics.get("auroc_per_task", [])
for i, (task, ax) in enumerate(zip(TASK_NAMES, axes)):
    ax.set_title(f"AUROC — {task}")
plt.tight_layout()
save_figure(fig, FIGURES_DIR / "auroc_curve.png")
plt.show()
```

### Section 7 — Espace des features (PCA 2D)

```python
from src.evaluation.feature_space_plots import plot_clustering_with_correctness
# Centroïdes K-Means des 5 locations visualisés dans l'espace PCA 2D
fig = plot_clustering_with_correctness(...)
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
plt.show()
```

### Section 8 — Tableau récapitulatif (Markdown)

```python
aa    = metrics["aa"]
af    = metrics["af"]
bwt   = metrics["bwt"]
auroc = metrics["auroc_avg"]
ram   = metrics["ram_peak_bytes"] / 1024
lat   = metrics["inference_latency_ms"]

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {auroc:.4f} | {ram:.2f} Ko | {lat:.5f} ms | — |")
```

---

## Métriques attendues (exp_022)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | À déterminer — expérience non encore exécutée |
| `af` (AF) | À déterminer (référence by_equipment : 0.0049) |
| `bwt` (BWT) | À déterminer (référence by_equipment : −0.0040) |
| `auroc_avg` | À déterminer (référence by_equipment : 0.9621) |
| `ram_peak_bytes` | À déterminer (référence by_equipment : 5 358 B — 5.2 Ko) |
| `inference_latency_ms` | À déterminer (référence by_equipment : 0.39870 ms) |
| `n_params` | — (centroïdes, non paramétrique) |

> **Note** : K-Means stocke 5 centroïdes (un par location) au lieu de 3 (by_equipment). La RAM peut augmenter légèrement mais reste dans la contrainte 64 Ko. L'AUROC by_location peut différer selon la séparabilité des données par location.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_location/kmeans.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 5×5 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — 5 courbes distinctes
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `auroc_curve.png` sauvegardé — AUROC par tâche pour les 5 locations
- [ ] `feature_space_pca.png` sauvegardé avec centroïdes visibles
- [ ] Tableau récapitulatif final avec AA/AF/BWT/AUROC/RAM/latence

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_location/kmeans.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/kmeans/monitoring/by_location').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `TODO(arnaud)` : Inclure aussi les résultats KNN et PCA (potentiellement également dans exp_022) dans ce notebook, ou créer des notebooks séparés ?
- `FIXME(gap3)` : K-Means ne nécessite pas de labels à l'entraînement — documenter explicitement cet avantage dans la section discussion (économie de coût d'annotation en production industrielle, pertinent pour les 5 sites géographiques Edge Spectrum).
- `TODO(fred)` : Dans le contexte industriel, les locations géographiques (Atlanta, Chicago…) présentent-elles des distributions d'anomalie distinctes ? Cela renforce l'intérêt du scénario by_location pour le contexte Edge Spectrum.
