# S3-09 — EDA plots enrichis : visualisation normale/anomalie par dataset

| Champ | Valeur |
|-------|--------|
| **ID** | S3-09 |
| **Sprint** | Sprint 3 — Semaine 3 (29 avril – 6 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S3-01 (`notebooks/01_data_exploration.ipynb` section Dataset 1 existante), S1-02 (section Dataset 2) |
| **Fichiers cibles** | `src/evaluation/eda_plots.py`, `scripts/explore_eda.py`, `notebooks/01_data_exploration.ipynb` |
| **Complété le** | 2026-04-09 |

---

## Objectif

Ajouter un ensemble complet de visualisations EDA ciblées sur la **séparation normal/anomalie** pour les deux datasets, accessibles à la fois via le notebook d'exploration et via un script CLI autonome.

Ces figures servent à :
1. Diagnostiquer visuellement si les features discriminent le label cible
2. Identifier d'éventuels déséquilibres ou chevauchements de distributions
3. Préparer les figures pour le manuscrit (Section Données)

---

## Visualisations implémentées

### Dataset 2 — Equipment Monitoring (`faulty` = 0/1, taux ~10%)

| Type | Fichier | Description |
|------|---------|-------------|
| Boxplots | `boxplots_by_faulty.png` | Médiane, Q1/Q3, outliers par classe |
| Histogrammes + KDE | `histograms_by_faulty.png` | Distribution densité par classe |
| Violin plots | `violin_by_faulty.png` | Forme distribution + quartiles |
| Densités KDE | `kde_by_faulty.png` | Courbes KDE superposées — séparation |
| Pairplot | `pairplot_by_faulty.png` | Scatter matrix 4×4, hue = faulty |
| Boxplots (domaine × faulty) | `boxplots_by_equipment_faulty.png` | Grille 3×4 (équipement × feature), hue = faulty |
| Violin (domaine × faulty) | `violin_by_equipment_faulty.png` | Idem, violin plots |
| KDE (domaine × faulty) | `kde_by_equipment_faulty.png` | Idem, courbes KDE par faulty |

### Dataset 1 — Pump Maintenance (`maintenance_required` = 0/1, taux ~50%)

| Type | Fichier | Description |
|------|---------|-------------|
| Boxplots | `boxplots_by_maintenance.png` | Idem, 5 features |
| Histogrammes + KDE | `histograms_by_maintenance.png` | |
| Violin plots | `violin_by_maintenance.png` | |
| Densités KDE | `kde_by_maintenance.png` | |
| Pairplot | `pairplot_by_maintenance.png` | Scatter matrix 5×5 |
| Temporal | `temporal_by_maintenance.png` | Features vs operational_hours, scatter + rolling mean |

---

## Module `src/evaluation/eda_plots.py`

Fonctions publiques :

```python
plot_boxplots_by_label(df, feature_cols, label_col, label_name, title) -> plt.Figure
plot_histograms_by_label(df, feature_cols, label_col, label_name, bins, title) -> plt.Figure
plot_violin_by_label(df, feature_cols, label_col, label_name, title) -> plt.Figure
plot_kde_by_label(df, feature_cols, label_col, label_name, title) -> plt.Figure
plot_pairplot_by_label(df, feature_cols, label_col, label_name, sample_n, title) -> plt.Figure
plot_label_distribution(df, label_col, label_name, group_col, title) -> plt.Figure
plot_temporal_by_label(df, feature_cols, label_col, time_col, label_name, rolling_window, title) -> plt.Figure
# Fonctions croisées domaine × label (Equipment Monitoring)
plot_boxplots_by_group_and_label(df, feature_cols, label_col, group_col, label_name, title) -> plt.Figure
plot_violin_by_group_and_label(df, feature_cols, label_col, group_col, label_name, title) -> plt.Figure
plot_kde_by_group_and_label(df, feature_cols, label_col, group_col, label_name, title) -> plt.Figure
```

Palette fixe : `COLORS_LABEL = {0: "#4CAF50", 1: "#F44336"}` (vert/rouge, cohérent avec `feature_space_plots.py`).

---

## Script `scripts/explore_eda.py`

```bash
python scripts/explore_eda.py --dataset all
python scripts/explore_eda.py --dataset equipment --config-equipment configs/ewc_config.yaml
python scripts/explore_eda.py --dataset pump --config-pump configs/tinyol_config.yaml
```

Lit les CSV via `configs/` (clé `data.csv_path`) et sauvegarde dans `notebooks/figures/eda/`.

---

## Ajouts dans `notebooks/01_data_exploration.ipynb`

- **Section 3-bis** (Dataset 2) : 5 cellules code (boxplots, histogrammes, violin, KDE, pairplot) après la section 3 (distribution du label `faulty`)
- **Section 2.4-bis** (Dataset 1) : 6 cellules code (boxplots, histogrammes, violin, KDE, pairplot, temporal) après la section 2.4 (corrélations)
- Variable `MONITORING_FIGURES_DIR = pathlib.Path("figures/eda/equipment_monitoring")` ajoutée en setup (cell-1)

---

## Contraintes techniques

- `matplotlib.use("Agg")` déjà appliqué dans `eda_plots.py` (cohérent avec `plots.py`)
- Dans le notebook, les cellules EDA sont inline (pas d'import de `eda_plots`) pour éviter le conflit de backend Jupyter
- `save_figure()` de `plots.py` utilisé dans le script CLI
- Pas d'annotations `# MEM:` — code PC-only évaluation

---

## Critères d'acceptation

- [x] `python scripts/explore_eda.py --dataset all` génère 15 PNG sans erreur
- [x] `notebooks/figures/eda/equipment_monitoring/` contient 8 PNG (5 existants + 3 domaine × faulty)
- [x] `notebooks/figures/eda/pump_maintenance/` contient 6 PNG (+ temporal)
- [x] `from src.evaluation.eda_plots import plot_boxplots_by_label` s'importe sans erreur
- [x] Les cellules notebook Section 3-bis et 2.4-bis sont exécutables avec `jupyter nbconvert`

---

## Questions ouvertes

- `TODO(arnaud)` : le pairplot (5×5 pour Dataset 1) est lisible en manuscrit ? Envisager un corner plot.
- `FIXME(gap1)` : si les distributions normal/maintenance sont quasi-identiques (Dataset 1, taux ~50%), documenter cette limitation dans le notebook.
