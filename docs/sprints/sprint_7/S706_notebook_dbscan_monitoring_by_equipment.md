# S7-06 — Notebook DBSCAN — monitoring_by_equipment

| Champ | Valeur |
|-------|--------|
| **ID** | S7-06 |
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | `experiments/exp_008_dbscan/` (complété) |
| **Fichier cible** | `notebooks/cl_eval/monitoring_by_equipment/dbscan.ipynb` |
| **Statut** | ⬜ À faire |

---

## Objectif

Créer le notebook d'évaluation CL pour **DBSCAN (non-supervisé)** sur le scénario **monitoring_by_equipment** (Pump → Turbine → Compressor, 3 tâches) en chargeant les résultats de `exp_008_dbscan`.

**Critère de succès** : notebook exécutable du début à la fin sans erreur, avec toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/dbscan/monitoring/by_equipment/`.

> **⚠️ Avertissement RAM** : DBSCAN RAM = 73 638 B (71.9 Ko), ce qui **dépasse la contrainte STM32N6 de 64 Ko**. Ce point doit être signalé explicitement dans le notebook avec un bloc d'avertissement et un `FIXME(gap2)`.

> **Note** : DBSCAN est un modèle de **détection d'anomalie non-supervisée** — il ne reçoit pas les labels à l'entraînement. La section ROC est remplacée par une section AUROC. Il n'y a pas de `n_params` (modèle basé sur la densité).

---

## Structure de sortie attendue

```
notebooks/cl_eval/monitoring_by_equipment/
└── dbscan.ipynb                        ← notebook créé

notebooks/figures/cl_evaluation/dbscan/monitoring/by_equipment/
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
# Évaluation CL — DBSCAN — Dataset 2 Equipment Monitoring — by_equipment

| Champ | Valeur |
|-------|--------|
| **Modèle** | DBSCAN (détection d'anomalie par densité, non-supervisé) |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | by_equipment : Pump → Turbine → Compressor (3 tâches) |
| **Expérience** | exp_008 — voir experiments/exp_008_dbscan/config_snapshot.yaml |
| **Date** | {date d'exécution} |

> ⚠️ **Contrainte embarquée NON respectée** : RAM = 73 638 B (71.9 Ko) > 64 Ko (limite STM32N6).
> DBSCAN est inadapté à un déploiement direct sur STM32N6 sans réduction d'empreinte.
> `FIXME(gap2)` — voir section discussion.
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

EXP_DIR     = Path("../../experiments/exp_008_dbscan/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/dbscan/monitoring/by_equipment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAMES   = ["Pump", "Turbine", "Compressor"]
MODEL_NAME   = "DBSCAN"
METRICS_FILE = "metrics_dbscan_dataset2.json"

# FIXME(gap2) : RAM = 73 638 B > contrainte 64 Ko STM32N6
RAM_LIMIT_BYTES = 64 * 1024
```

### Section 2 — Chargement des résultats

```python
metrics_path    = EXP_DIR / METRICS_FILE
acc_matrix_path = EXP_DIR / "acc_matrix_dbscan.npy"

metrics    = json.loads(metrics_path.read_text())
acc_matrix = np.load(acc_matrix_path, allow_pickle=True)

ram = metrics["ram_peak_bytes"]
print(f"AA       = {metrics['aa']:.4f}")
print(f"AF       = {metrics['af']:.4f}")
print(f"BWT      = {metrics['bwt']:.4f}")
print(f"AUROC    = {metrics['auroc_avg']:.4f}")
print(f"RAM      = {ram} B ({ram/1024:.1f} Ko)")
print(f"Latence  = {metrics['inference_latency_ms']:.5f} ms")

# Vérification contrainte RAM
if ram > RAM_LIMIT_BYTES:
    print(f"\n⚠️  RAM ({ram/1024:.1f} Ko) > contrainte STM32N6 (64 Ko) — FIXME(gap2)")
```

### Section 3 — Matrice d'accuracy (heatmap)

```python
# Note : accuracy calculée par seuillage du score d'anomalie DBSCAN
# AF=0.0 — les clusters DBSCAN sont réappris par tâche (pas d'accumulation destructive)
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES,
                           title=f"{MODEL_NAME} — monitoring/by_equipment")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
plt.show()
```

### Section 4 — Courbe d'oubli par tâche

```python
# DBSCAN : AF=0.0 — modèle réinitialisé par tâche dans ce scénario
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
plt.show()
```

### Section 5 — Matrices de confusion par tâche (détection anomalie)

```python
preds_dict = ...  # à adapter selon format exp_008
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES,
                                 model_name=MODEL_NAME)
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
plt.show()
```

### Section 6 — Courbe AUROC (détection d'anomalie)

```python
from sklearn.metrics import roc_curve, auc
# DBSCAN AUROC = 0.9786 — meilleure AUROC parmi tous les modèles non-supervisés
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
# Clusters DBSCAN dans l'espace PCA 2D — points bruit en gris
fig = plot_clustering_with_correctness(...)
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
plt.show()
```

### Section 8 — Tableau récapitulatif + Discussion RAM

```python
aa     = metrics["aa"]
af     = metrics["af"]
bwt    = metrics["bwt"]
auroc  = metrics["auroc_avg"]
ram    = metrics["ram_peak_bytes"] / 1024
lat    = metrics["inference_latency_ms"]

print(f"| {MODEL_NAME} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {auroc:.4f} | {ram:.1f} Ko ⚠️ | {lat:.5f} ms | — |")
```

```markdown
### Discussion — Contrainte embarquée (FIXME gap2)

DBSCAN atteint la meilleure AUROC (0.9786) et un bon AA (0.9557), mais sa RAM peak
de **71.9 Ko dépasse la contrainte de 64 Ko** du STM32N6.

Pistes d'optimisation à explorer :
- Réduire le nombre de points noyaux conservés par tâche (buffer borné)
- Utiliser une version mini-batch ou streaming de DBSCAN
- Quantification des points noyaux en INT8 (Gap 3)
```

---

## Métriques attendues (exp_008)

| Métrique | Valeur |
|----------|--------|
| `aa` (AA) | 0.9557 |
| `af` (AF) | 0.0000 ← zéro oubli (clusters réinitialisés par tâche) |
| `bwt` (BWT) | +0.0010 |
| `auroc_avg` | 0.9786 ← **meilleure AUROC** parmi tous les modèles |
| `ram_peak_bytes` | 73 638 B (71.9 Ko) ← ⚠️ **dépasse 64 Ko** |
| `inference_latency_ms` | 0.42320 ms ← le plus lent |
| `n_params` | — (points noyaux, non paramétrique) |

---

## Critères d'acceptation

- [ ] `notebooks/cl_eval/monitoring_by_equipment/dbscan.ipynb` créé
- [ ] Notebook exécutable sans erreur (Restart Kernel + Run All)
- [ ] `acc_matrix.png` sauvegardé — matrice 3×3 affichée correctement
- [ ] `forgetting_curve.png` sauvegardé — AF=0.0 (courbes plates)
- [ ] `confusion_matrix_grid.png` sauvegardé
- [ ] `auroc_curve.png` sauvegardé — AUROC ≈ 0.9786 par tâche
- [ ] `feature_space_pca.png` sauvegardé avec clusters visibles
- [ ] Avertissement RAM visible dans le notebook (`⚠️ RAM > 64 Ko`)
- [ ] Tableau récapitulatif final : AA=0.9557, AF=0.0, BWT=+0.0010, AUROC=0.9786, RAM=71.9 Ko ⚠️, latence=0.42320 ms

---

## Vérification post-création

```bash
python -c "
import nbformat
from pathlib import Path
nb_path = Path('notebooks/cl_eval/monitoring_by_equipment/dbscan.ipynb')
nb = nbformat.read(nb_path, as_version=4)
print(f'Nombre de cellules : {len(nb.cells)}')
figs = list(Path('notebooks/figures/cl_evaluation/dbscan/monitoring/by_equipment').glob('*.png'))
print(f'Figures générées : {[f.name for f in figs]}')
# Vérifier que l'avertissement RAM est présent
has_warning = any('71.9 Ko' in c.source or 'gap2' in c.source.lower() for c in nb.cells)
print(f'Avertissement RAM présent : {has_warning}')
"
```

---

## Questions ouvertes

- `FIXME(gap2)` : DBSCAN RAM = 71.9 Ko > 64 Ko — bloquer sur STM32N6. Explorer une implémentation streaming avec buffer borné de points noyaux (ex. max 500 points × 6 features × 4 octets = 12 Ko). À planifier en sprint 10 (MCU).
- `TODO(fred)` : En contexte industriel Edge Spectrum, le DBSCAN est-il utilisé en pratique pour la détection d'anomalie sur équipement ? Contrainte RAM acceptable si l'appareil a plus de 64 Ko ?
