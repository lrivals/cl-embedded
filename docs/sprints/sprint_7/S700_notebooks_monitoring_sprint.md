# Sprint 7 (Phase 1) — Notebooks individuels Dataset 2 (Equipment Monitoring)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 7 — Phase 1 Extension |
| **Priorité globale** | 🔴 Critique — notebooks support présentation encadrants |
| **Durée estimée totale** | ~28h |
| **Dépendances** | Sprint 6 terminé (exp_016–023 exécutées, `save_figure()` corrigé) |

---

## Objectif

Créer 14 notebooks d'évaluation CL pour le Dataset 2 (Equipment Monitoring — `equipment_anomaly_data.csv`) :
- 6 notebooks individuels pour le scénario **by_equipment** (Pump → Turbine → Compressor, 3 tâches)
- 6 notebooks individuels pour le scénario **by_location** (Atlanta → Chicago → Houston → New York → San Francisco, 5 tâches)
- 2 notebooks de comparaison (un par scénario)

Ces notebooks sont **le support de présentation principal** pour les réunions d'encadrement.

**Critère de succès** : 14 notebooks exécutables du début à la fin sans erreur, avec toutes les figures sauvegardées dans la structure `notebooks/figures/cl_evaluation/{model}/{dataset}/{task}/`.

---

## Structure cible

```
notebooks/cl_eval/monitoring_by_equipment/
├── tinyol.ipynb       ← exp_011
├── ewc.ipynb          ← exp_001
├── hdc.ipynb          ← exp_002
├── kmeans.ipynb       ← exp_005 (section K-Means)
├── mahalanobis.ipynb  ← exp_007
├── dbscan.ipynb       ← exp_008
└── comparison.ipynb   ← tous les modèles

notebooks/cl_eval/monitoring_by_location/
├── tinyol.ipynb       ← exp_018
├── ewc.ipynb          ← exp_016
├── hdc.ipynb          ← exp_017
├── kmeans.ipynb       ← exp_022
├── mahalanobis.ipynb  ← exp_019
├── dbscan.ipynb       ← exp_023
└── comparison.ipynb   ← tous les modèles
```

---

## Structure des figures

```
notebooks/figures/cl_evaluation/
├── tinyol/monitoring/by_equipment/
│   ├── acc_matrix.png
│   ├── forgetting_curve.png
│   ├── confusion_matrix_grid.png
│   ├── roc_curves.png
│   └── feature_space_pca.png
├── ewc/monitoring/by_equipment/
│   └── (même 5 figures)
├── hdc/monitoring/by_equipment/
│   └── ...
├── kmeans/monitoring/by_equipment/
│   └── ...
├── mahalanobis/monitoring/by_equipment/
│   └── ...
├── dbscan/monitoring/by_equipment/
│   └── ...
├── comparison/monitoring/by_equipment/
│   ├── radar_comparison.png
│   ├── barplot_aa_comparison.png
│   └── acc_matrix_grid.png
... (même structure pour /by_location/)
```

---

## Tâches

### Scénario monitoring_by_equipment

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S7-01 | Notebook TinyOL — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/tinyol.ipynb` | 2h | exp_011 |
| S7-02 | Notebook EWC — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/ewc.ipynb` | 2h | exp_001 |
| S7-03 | Notebook HDC — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/hdc.ipynb` | 2h | exp_002 |
| S7-04 | Notebook KMeans — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/kmeans.ipynb` | 2h | exp_005 |
| S7-05 | Notebook Mahalanobis — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/mahalanobis.ipynb` | 2h | exp_007 |
| S7-06 | Notebook DBSCAN — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/dbscan.ipynb` | 2h | exp_008 |
| S7-13 | Notebook Comparaison — monitoring_by_equipment | 🔴 | `notebooks/cl_eval/monitoring_by_equipment/comparison.ipynb` | 3h | S7-01 à S7-06 |

### Scénario monitoring_by_location

| ID | Tâche | Priorité | Fichier cible | Durée est. | Dépendances |
|----|-------|:---:|---------------|:---:|-------------|
| S7-07 | Notebook TinyOL — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/tinyol.ipynb` | 2h | exp_018 (Sprint 6) |
| S7-08 | Notebook EWC — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/ewc.ipynb` | 2h | exp_016 (Sprint 6) |
| S7-09 | Notebook HDC — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/hdc.ipynb` | 2h | exp_017 (Sprint 6) |
| S7-10 | Notebook KMeans — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/kmeans.ipynb` | 2h | exp_022 (Sprint 6) |
| S7-11 | Notebook Mahalanobis — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/mahalanobis.ipynb` | 2h | exp_019 (Sprint 6) |
| S7-12 | Notebook DBSCAN — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/dbscan.ipynb` | 2h | exp_023 (Sprint 6) |
| S7-14 | Notebook Comparaison — monitoring_by_location | 🔴 | `notebooks/cl_eval/monitoring_by_location/comparison.ipynb` | 3h | S7-07 à S7-12 |

---

## Contenu type d'un notebook individuel

Chaque notebook individuel suit cette structure (8 sections) :

### Section 0 — En-tête (cellule Markdown)
```markdown
# Évaluation CL — {MODÈLE} — Dataset 2 Equipment Monitoring — {SCÉNARIO}

| Champ | Valeur |
|-------|--------|
| **Modèle** | {nom complet + nb params} |
| **Dataset** | equipment_anomaly_data.csv — 7 672 échantillons |
| **Scénario** | {by_equipment: Pump→Turbine→Compressor (3 tâches)} ou {by_location: ATL→CHI→HOU→NYC→SFO (5 tâches)} |
| **Expérience** | exp_{XXX} — voir experiments/exp_{XXX}/config_snapshot.yaml |
| **Date** | {date d'exécution} |
```

### Section 1 — Setup & imports
```python
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path("../..")))  # racine du projet

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.plots import (
    plot_accuracy_matrix, plot_forgetting_curve,
    plot_confusion_matrix_grid, plot_roc_curves_per_task, save_figure
)
from src.evaluation.metrics import compute_cl_metrics

# Chemins
EXP_DIR = Path("../../experiments/exp_{XXX}/results")
FIGURES_DIR = Path("../../notebooks/figures/cl_evaluation/{model}/monitoring/{scenario}")
TASK_NAMES = [...]  # selon scénario
```

### Section 2 — Chargement des résultats
```python
# Charger metrics.json + acc_matrix.npy
# Fallback mock si exp non exécutée
metrics = json.loads((EXP_DIR / "metrics.json").read_text())
acc_matrix = np.load(EXP_DIR / "acc_matrix.npy", allow_pickle=True)
```

### Section 3 — Matrice d'accuracy (heatmap)
```python
fig = plot_accuracy_matrix(acc_matrix, task_names=TASK_NAMES, title="...")
save_figure(fig, FIGURES_DIR / "acc_matrix.png")
```
Commentaire : diagonale = performance immédiate post-training, sous-diagonale = oubli.

### Section 4 — Courbe d'oubli par tâche
```python
fig = plot_forgetting_curve(acc_matrix, task_names=TASK_NAMES)
save_figure(fig, FIGURES_DIR / "forgetting_curve.png")
```
Commentaire : une courbe plate = pas d'oubli, courbe descendante = oubli catastrophique.

### Section 5 — Matrices de confusion par tâche
```python
fig = plot_confusion_matrix_grid(preds_dict, task_names=TASK_NAMES, model_name="...")
save_figure(fig, FIGURES_DIR / "confusion_matrix_grid.png")
```

### Section 6 — Courbes ROC par tâche
```python
fig = plot_roc_curves_per_task(preds_dict)
save_figure(fig, FIGURES_DIR / "roc_curves.png")
```

### Section 7 — Espace des features (PCA 2D)
```python
from src.evaluation.feature_space_plots import plot_clustering_with_correctness
# Projection PCA 2D colorée par tâche/correctness
save_figure(fig, FIGURES_DIR / "feature_space_pca.png")
```

### Section 8 — Tableau récapitulatif (pour le rapport)
```python
# AA, AF, BWT, RAM peak, latence, n_params
# Cellule Markdown générée automatiquement
print(f"| {MODEL} | {aa:.4f} | {af:.4f} | {bwt:.4f} | {ram_ko:.1f} Ko | {latency_ms:.3f} ms |")
```

---

## Contenu type d'un notebook de comparaison

### Section 0 — En-tête
```markdown
# Comparaison 6 modèles — Dataset 2 Equipment Monitoring — {SCÉNARIO}
Scénario : {by_equipment (3 tâches) ou by_location (5 tâches)}
Modèles : TinyOL · EWC · HDC · KMeans · Mahalanobis · DBSCAN
```

### Section 1 — Chargement de tous les résultats
```python
MODEL_EXP_MAP = {
    "TinyOL": "exp_{XXX}", "EWC": "exp_{XXX}", "HDC": "exp_{XXX}",
    "KMeans": "exp_{XXX}", "Mahalanobis": "exp_{XXX}", "DBSCAN": "exp_{XXX}"
}
# Charger metrics.json + acc_matrix.npy pour chaque modèle
```

### Section 2 — Radar multi-critères (6 modèles)
Axes normalisés : AA, Stabilité (1−AF), BWT neutre (1−|BWT|), RAM (1−RAM/64Ko), Vitesse (1−latence/100ms)
```python
from src.evaluation.plots import plot_model_radar
save_figure(fig, FIGURES_DIR / "radar_comparison.png")
```

### Section 3 — Barplot comparaison AA
```python
fig = plot_metrics_comparison(results, metrics=["aa", "af", "bwt"])
save_figure(fig, FIGURES_DIR / "barplot_aa_comparison.png")
```

### Section 4 — Grille de matrices d'accuracy (6 modèles)
```python
# 2×3 subplots, une acc_matrix par modèle
save_figure(fig, FIGURES_DIR / "acc_matrix_grid.png")
```

### Section 5 — Tableau comparatif complet
```markdown
| Modèle | AA | AF | BWT | RAM peak | Latence | n_params |
|--------|:--:|:--:|:---:|:--------:|:-------:|:--------:|
| TinyOL | ... | ... | ... | ... Ko | ... ms | ... |
| EWC    | ... | ... | ... | ... Ko | ... ms | ... |
...
```

### Section 6 — Discussion
Cellule Markdown structurée avec :
- Quel modèle a la meilleure accuracy sur ce scénario ?
- Quel modèle a le meilleur ratio performance/RAM ?
- Y a-t-il de l'oubli catastrophique ? Quel modèle y résiste le mieux ?
- Questions scientifiques ouvertes (FIXME gap 1/2/3)

---

## Critères d'acceptation

- [ ] 14 notebooks créés dans `notebooks/cl_eval/monitoring_*/`
- [ ] Chaque notebook s'exécute sans erreur (kernel restart + run all)
- [ ] Toutes les figures sauvegardées dans `notebooks/figures/cl_evaluation/{model}/monitoring/{scenario}/`
- [ ] Les notebooks de comparaison chargent bien les 6 modèles
- [ ] Chaque notebook se termine par un tableau récapitulatif AA/AF/BWT/RAM
- [ ] Le fallback mock est fonctionnel pour les expériences non encore exécutées

---

## Livrable sprint 7

14 notebooks pour Dataset 2 (7 by_equipment + 7 by_location) prêts pour présentation aux encadrants. Support de présentation avancement avec tableaux récapitulatifs AA/AF/BWT/RAM.

---

## Questions ouvertes

- `TODO(arnaud)` : Pour la présentation, faut-il inclure les modèles KNN et PCA (exp_005) dans les comparaisons, ou se limiter aux 4 modèles principaux (TinyOL, EWC, HDC, Mahalanobis) + DBSCAN/KMeans ?
- `TODO(arnaud)` : Les notebooks de comparaison doivent-ils inclure une section de discussion qualitative ou uniquement des métriques quantitatives ?
- `FIXME(gap3)` : Les modèles non-supervisés ne nécessitent pas de labels à l'entraînement — noter explicitement cet avantage dans la section discussion des notebooks de comparaison.

---

> **⚠️ Après l'implémentation de ce sprint** : exécuter tous les notebooks via "Restart Kernel & Run All Cells" et vérifier l'absence d'erreurs. Contrôler que chaque sous-dossier de figures est bien créé. Mettre à jour `docs/roadmap_phase1.md` en marquant S7-01 à S7-14 comme ✅.
