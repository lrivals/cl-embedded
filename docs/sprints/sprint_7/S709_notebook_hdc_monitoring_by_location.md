# S7-09 — Notebook HDC — monitoring_by_location

| Champ | Valeur |
|-------|--------|
| **ID** | S7-09 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_017_hdc_monitoring_by_location/` (Sprint 6) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_location/hdc.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **HDC (Hyperdimensional Computing, D=1024)** sur le scénario **monitoring_by_location** (Atlanta → Chicago → Houston → New York → San Francisco, 5 tâches) en chargeant les résultats de `exp_017_hdc_monitoring_by_location`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/hdc/monitoring/by_location/`.

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_location/
└── hdc.ipynb                           ← notebook créé

notebooks/figures/cl_evaluation/hdc/monitoring/by_location/
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
# Évaluation CL — HDC (D=1024) — Dataset 2 Equipment Monitoring — by_location

| Champ | Valeur |
|-------|--------|
| **Modèle** | HDC Hyperdimensional Computing (D=1024, 2 048 paramètres) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_location : Atlanta → Chicago → Houston → New York → San Francisco (5 tâches) |
| **Expérience** | exp_017 — voir experiments/exp_017_hdc_monitoring_by_location/config_snapshot.yaml |
| **Date** | {date d'exécution} |

> **Propriété HDC** : AF=0.0 par construction — les hypervecteurs de prototype par classe ne sont pas écrasés lors de l'apprentissage de nouvelles tâches.
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

EXP_DIR     = Path("../../experiments/exp_017_hdc_monitoring_by_location/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/hdc/monitoring/by_location")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES = ["Atlanta", "Chicago", "Houston", "New York", "San Francisco"]
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
# HDC : AF=0.0 — les hypervecteurs de prototype sont préservés par construction
# Matrice 5×5 pour les 5 locations
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_location")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# HDC : AF=0.0 par design (les hypervecteurs de classe ne sont pas écrasés)
# Les 5 courbes devraient être plates
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche

```python
preds_dict = ...  # à adapter selon format exp_017
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
# Note : visualiser les hypervecteurs de prototype des 5 locations si disponibles
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

## Métriques attendues (exp_017)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | À déterminer — expérience non encore exécutée |
| `af` (AF) | 0.0000 (attendu — zéro oubli par construction HDC) |
| `bwt` (BWT) | À déterminer (référence by_equipment : +0.0019 — transfert positif) |
| `ram_peak_bytes` | À déterminer (référence by_equipment : 14 504 B — 14.2 Ko) |
| `inference_latency_ms` | À déterminer (référence by_equipment : 0.04758 ms) |
| `n_params` | 2 048 (hypervecteurs de prototype) — identique à by_equipment |

> **Note scientifique** : HDC présente AF=0.0 par design sur tous les scénarios. Avec 5 tâches by_location (vs 3 by_equipment), la RAM HDC pourrait augmenter légèrement si le nombre de prototypes est proportionnel au nombre de tâches. À vérifier.

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_location/hdc.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 5×5 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF=0.0 (5 courbes plates)
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
nb_path = Path('notebooks/cl_eval/monitoring_by_location/hdc.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/hdc/monitoring/by_location').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
"
```

---

## Questions ouvertes

- `FIXME(gap1)` : HDC AF=0.0 mais AA plus faible que EWC sur by_equipment (0.8698 vs 0.9824) — ce compromis stabilité/plasticité est-il maintenu sur 5 tâches by_location ? Documenter dans le notebook de comparaison S7-14.
- `TODO(dorra)` : Avec 5 tâches by_location, la RAM HDC augmente-t-elle ? Chaque prototype = D=1024 × 4 octets = 4 Ko par classe. Estimer la RAM totale pour 5 tâches.
- `FIXME(gap1)` : Le transfert positif BWT=+0.0019 observé sur by_equipment se reproduit-il sur by_location ? Les locations géographiques présentent-elles un drift similaire bénéfique pour les hypervecteurs ?
