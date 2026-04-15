# S7-05 — Notebook Mahalanobis — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-05 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_007_mahalanobis/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/mahalanobis.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **Mahalanobis Distance (non-supervisé)** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_007_mahalanobis`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/mahalanobis/monitoring/by_equipment/`.

> **Note** : Mahalanobis est un modèle de **détection d'anomalie non-supervisée** — il ne reçoit pas les labels à l'entraînement. La section ROC est remplacée par une section AUROC. Il n'y a pas de `n_params` (le modèle stocke une matrice de covariance inverse par tâche). C'est le modèle avec la **meilleure RAM** parmi les non-supervisés (1 504 B).

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── mahalanobis.ipynb                   ← notebook créé

notebooks/figures/cl_evaluation/mahalanobis/monitoring/by_equipment/
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
# Évaluation CL — Mahalanobis Distance — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | Mahalanobis Distance (détection d'anomalie, non-supervisé) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_007 — voir experiments/exp_007_mahalanobis/config_snapshot.yaml |
| **Date** | {date d'exécution} |

> **Modèle non-supervisé** : Mahalanobis ne reçoit pas les labels à l'entraînement.
> Score d'anomalie = distance de Mahalanobis à la distribution normale de la tâche courante.
> RAM = 1 504 B — meilleure empreinte mémoire parmi les modèles non-supervisés.
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

EXP_DIR     = Path("../../experiments/exp_007_mahalanobis/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/mahalanobis/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES   = ["Pump", "Turbine", "Compressor"]
MODEL_NAME   = "Mahalanobis"
METRICS_FILE = "metrics_mahalanobis_dataset2.json"
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / METRICS_FILE
acc_matrix_path = EXP_DIR / "acc_matrix_mahalanobis.npy"

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
# Note : accuracy calculée par seuillage de la distance de Mahalanobis
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# AF ≈ 0.001 — très faible oubli (la matrice de covariance par tâche est préservée)
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche (détection anomalie)

```python
# Seuil optimal pour chaque tâche (maximise F1 ou Youden J)
preds_dict = ...  # à adapter selon format exp_007
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES,
                                 model_name=MODEL_NAME)
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
plt.show()
```

### Section 6 — Courbe AUROC (détection d'anomalie)

```python
from sklearn.metrics import roc_curve, auc
# AUROC par tâche — la distance de Mahalanobis est le score d'anomalie
fig, axes = plt.subplots(1, len(TASK_NAMES), figsize=(15, 4))
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
# Ellipses de Mahalanobis dans l'espace PCA 2D (1-sigma et 2-sigma)
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

## Métriques attendues (exp_007)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | 0.9524 |
| `af` (AF) | 0.0010 |
| `bwt` (BWT) | −0.0010 |
| `auroc_avg` | 0.9718 ← meilleure AUROC des non-supervisés après DBSCAN |
| `ram_peak_bytes` | 1 504 B (1.5 Ko) ← **meilleure RAM parmi les non-supervisés** |
| `inference_latency_ms` | 0.01801 ms ← **plus rapide parmi les non-supervisés** |
| `n_params` | — (matrice de covariance inverse, non paramétrique) |

> **Note scientifique** : Mahalanobis offre le meilleur ratio performance/RAM parmi les non-supervisés : AA=0.9524, AUROC=0.9718, RAM=1.5 Ko, latence=0.018 ms. Candidat fort pour embarquement STM32N6. À valoriser dans S7-13 et le manuscrit (Gap 2).

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/mahalanobis.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF ≈ 0.0010
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `auroc_curve.png` sauvegardé — AUROC ≈ 0.9718 par tâche
- [ ] `feature_space_pca.png` sauvegardé avec ellipses Mahalanobis visibles
- [ ] Tableau récapitulatif final : AA=0.9524, AF=0.0010, BWT=−0.0010, AUROC=0.9718, RAM=1.5 Ko, latence=0.01801 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/mahalanobis.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/mahalanobis/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `FIXME(gap2)` : Mahalanobis RAM=1 504 B (1.5 Ko) — bien sous la contrainte 64 Ko. Documenter la marge disponible pour stocker les matrices de covariance inverses des 3 tâches. Chiffre à citer dans le manuscrit Gap 2.
- `TODO(dorra)` : La matrice de covariance inverse en FP32 (6×6 pour 6 features) = 144 octets par tâche. Estimer la RAM totale pour 5 tâches (by_location) = 720 octets. Confirmer le chiffre.
