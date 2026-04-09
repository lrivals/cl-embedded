# S3-10 — Réorganisation de notebooks/figures/ en sous-dossiers

| Champ | Valeur |
|-------|--------|
| **ID** | S3-10 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 1h |
| **Dépendances** | S3-09 (structure eda/ créée), S5-13 (feature_space_*.png existants) |
| **Fichiers cibles** | `notebooks/figures/**`, `scripts/visualize_feature_space.py`, `scripts/_run_kpca_snippet.py`, `notebooks/01_data_exploration.ipynb`, `notebooks/02_baseline_comparison.ipynb` |
| **Complété le** | 2026-04-09 |

---

## Objectif

Organiser le dossier `notebooks/figures/` (anciennement plat avec 11 fichiers mélangés) en sous-dossiers thématiques pour faciliter la navigation, le versionning et la référence dans le manuscrit.

---

## Nouvelle structure

```
notebooks/figures/
├── eda/
│   ├── equipment_monitoring/    ← EDA Dataset 2 (faulty)
│   │   ├── boxplots_by_faulty.png
│   │   ├── histograms_by_faulty.png
│   │   ├── violin_by_faulty.png
│   │   ├── kde_by_faulty.png
│   │   └── pairplot_by_faulty.png
│   └── pump_maintenance/        ← EDA Dataset 1 (maintenance)
│       ├── boxplots_by_maintenance.png
│       ├── histograms_by_maintenance.png
│       ├── violin_by_maintenance.png
│       ├── kde_by_maintenance.png
│       ├── pairplot_by_maintenance.png
│       ├── temporal_by_maintenance.png
│       ├── temporal_drift.png           ← anciennement pump_maintenance_drift.png
│       └── correlations.png             ← anciennement pump_correlations.png
├── cl_evaluation/               ← Métriques CL (AA, AF, comparaisons)
│   ├── acc_matrix_comparison.png
│   └── memory_comparison.png
├── model_viz/                   ← Visualisations internes des modèles
│   ├── hdc_prototypes.png
│   └── kpca_rbf_snippet.png
└── feature_space/               ← Espace des features (S5-13)
    ├── 05_feature_space_scatter.png
    ├── 05_feature_space_kmeans.png
    ├── 05_feature_space_mahalanobis.png
    ├── 05_feature_space_pca_recon.png
    └── 05_feature_space_cl_evolution.png
```

---

## Fichiers déplacés (git mv)

| Source (racine figures/) | Destination |
|--------------------------|-------------|
| `pump_maintenance_drift.png` | `eda/pump_maintenance/temporal_drift.png` |
| `pump_correlations.png` | `eda/pump_maintenance/correlations.png` |
| `acc_matrix_comparison.png` | `cl_evaluation/acc_matrix_comparison.png` |
| `memory_comparison.png` | `cl_evaluation/memory_comparison.png` |
| `hdc_prototypes.png` | `model_viz/hdc_prototypes.png` |
| `kpca_rbf_snippet.png` | `model_viz/kpca_rbf_snippet.png` |
| `05_feature_space_scatter.png` | `feature_space/05_feature_space_scatter.png` |
| `05_feature_space_kmeans.png` | `feature_space/05_feature_space_kmeans.png` |
| `05_feature_space_mahalanobis.png` | `feature_space/05_feature_space_mahalanobis.png` |
| `05_feature_space_pca_recon.png` | `feature_space/05_feature_space_pca_recon.png` |
| `05_feature_space_cl_evolution.png` | `feature_space/05_feature_space_cl_evolution.png` |

---

## Mises à jour du code

### `scripts/visualize_feature_space.py` (ligne 47)

```python
# Avant
OUTPUT_DIR = Path("notebooks/figures")
# Après
OUTPUT_DIR = Path("notebooks/figures/feature_space")
```

### `scripts/_run_kpca_snippet.py` (ligne 30)

```python
# Avant
out = Path("notebooks/figures/kpca_rbf_snippet.png")
# Après
out = Path("notebooks/figures/model_viz/kpca_rbf_snippet.png")
```

### `notebooks/01_data_exploration.ipynb`

- `FIGURES_DIR = pathlib.Path("figures")` → `pathlib.Path("figures/eda/pump_maintenance")`
- `plt.savefig(FIGURES_DIR / "pump_maintenance_drift.png", ...)` → `FIGURES_DIR / "temporal_drift.png"`
- `plt.savefig(FIGURES_DIR / "pump_correlations.png", ...)` → `FIGURES_DIR / "correlations.png"`
- Ajout de `MONITORING_FIGURES_DIR = pathlib.Path("figures/eda/equipment_monitoring")` en setup

### `notebooks/02_baseline_comparison.ipynb`

- `FIGURES = Path("notebooks/figures")` → `FIGURES_CL_EVAL` + `FIGURES_MODEL_VIZ`
- `FIGURES / "acc_matrix_comparison.png"` → `FIGURES_CL_EVAL / "acc_matrix_comparison.png"`
- `FIGURES / "hdc_prototypes.png"` → `FIGURES_MODEL_VIZ / "hdc_prototypes.png"`
- `FIGURES / "memory_comparison.png"` → `FIGURES_CL_EVAL / "memory_comparison.png"`

---

## Critères d'acceptation

- [x] `ls notebooks/figures/` liste uniquement des sous-dossiers (plus de PNG à la racine)
- [x] `scripts/visualize_feature_space.py` génère dans `notebooks/figures/feature_space/`
- [x] `notebooks/02_baseline_comparison.ipynb` sauvegarde dans `cl_evaluation/` et `model_viz/`
- [x] `notebooks/01_data_exploration.ipynb` sauvegarde dans `eda/pump_maintenance/`
- [x] `git status` montre les 11 `git mv` correctement tracés (renommages)

---

## Notes

- Les sous-dossiers `eda/`, `cl_evaluation/`, `model_viz/`, `feature_space/` sont créés automatiquement par `mkdir(parents=True, exist_ok=True)` dans les scripts/notebooks → pas de `.gitkeep` nécessaire
- Les fichiers `.gitignore` ne couvrent pas `notebooks/figures/` (les PNG sont versionnés pour reproductibilité des figures manuscrit)
