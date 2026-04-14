# S5-15 — EDA affinée : Pump_ID × Operational_Hours + Equipment × Location

| Champ | Valeur |
|-------|--------|
| **ID** | S5-15 |
| **Sprint** | Sprint 5 — Semaine 5 (13–20 mai 2026) |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S3-01 (Dataset 1 chargé), S1-02 (Dataset 2 chargé), S3-09 (eda_plots.py existant) |
| **Fichiers cibles** | `src/evaluation/eda_plots.py`, `notebooks/01_data_exploration.ipynb` |
| **Complété le** | 2026-04-10 |
| **Statut** | ✅ Terminé |

---

## Objectif

Enrichir l'EDA existante (groupement uniquement par label) avec des analyses affinées par
identifiant de pompe (`Pump_ID`), par fenêtres temporelles (`Operational_Hours`) et par
croisement `equipment × location`. Utiliser seaborn (déjà dépendance du projet) pour des
visualisations plus expressives : violin plots split, boxplots multi-hue, heatmaps de taux de
panne, pairplots multi-domaine, matrices de corrélation par domaine.

**Critère de succès** : 9 nouvelles fonctions dans `eda_plots.py`, cellules correspondantes
dans `notebooks/01_data_exploration.ipynb` (sections 7-bis et 2.4-ter), figures sauvegardées
dans `notebooks/figures/eda/`.

---

## Sous-tâches

### 1. Helper `_save_fig()` dans `eda_plots.py`

Factorisation de la logique de sauvegarde (création dossier parent + `fig.savefig`),
utilisée par toutes les nouvelles fonctions.

```python
def _save_fig(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
```

### 2. Dataset 1 — Pump : fonctions par Pump_ID

**`plot_boxplots_by_pump_id`** — Boxplots seaborn avec `x=pump_col`, `hue=label_col`.
Fallback matplotlib avec double positionnement (offsets).

**`plot_violin_by_pump_id`** — Violin plots `x=pump_col`, `hue=label_col`, `inner="quartile"`.
Fallback avec violinplot décalé à ±0.15.

### 3. Dataset 1 — Pump : fonctions par fenêtres Operational_Hours

**`plot_operational_hour_windows`** — `pd.cut` en `n_windows` intervalles équidistants,
puis `sns.boxplot + sns.stripplot` superposés pour montrer densité et outliers.

**`plot_fault_rate_heatmap_pump`** — `groupby([pump_col, hour_window]).mean()` →
pivot → `sns.heatmap(cmap="RdYlGn_r", annot=True, fmt=".2f")`.
Rows = Pump_ID, Cols = fenêtres temporelles.

### 4. Dataset 2 — Equipment : fonctions par equipment × location

**`plot_boxplots_by_equipment_location`** — Grille `n_equipment × n_features`,
boxplot par location dans chaque cellule, hue = label.

**`plot_violin_by_location`** — Violin plots `x=location_col`, hue = label.
Annotation `n=` pour chaque location.

**`plot_fault_rate_heatmap_equipment`** — Heatmap `equipment × location` du taux
de panne. Annotation en pourcentage (`fmt=""`  avec applymap). `cmap="RdYlGn_r"`.

**`plot_correlation_by_equipment`** — Une heatmap de corrélation `sns.heatmap(cmap="coolwarm")`
par equipment type. Colorbar uniquement sur la dernière colonne.

**`plot_pairplot_by_equipment`** — `sns.pairplot(hue=equipment_col)` avec
couleurs domaine `["#2196F3", "#FF9800", "#9C27B0"]` et marqueurs distincts `["o","s","D"]`.
Subsample à 1 500 points.

### 5. Notebook `01_data_exploration.ipynb`

Deux nouvelles sections insérées :

- **Section 7-bis** (après Section 7 — Scénario CL Dataset 2) : 6 cellules
  `equip_imports → plot_boxplots_by_equipment_location → plot_violin_by_location →
  plot_fault_rate_heatmap_equipment → plot_correlation_by_equipment → plot_pairplot_by_equipment`

- **Section 2.4-ter** (après Section 2.4-bis — EDA Pump) : 5 cellules
  `pump_imports → plot_boxplots_by_pump_id → plot_violin_by_pump_id →
  plot_operational_hour_windows → plot_fault_rate_heatmap_pump`

---

## Critères d'acceptation

- [x] `_save_fig()` présent dans `eda_plots.py`, utilisé par toutes les nouvelles fonctions
- [x] 9 fonctions ajoutées avec docstrings NumPy et fallback matplotlib
- [x] Pattern `try/except` seaborn conservé (identique aux fonctions existantes)
- [x] Signatures uniformes : `(df, ..., save_path: Path | None = None) -> plt.Figure`
- [x] Cellules ajoutées dans `notebooks/01_data_exploration.ipynb` (sections 7-bis et 2.4-ter)
- [x] Dossier `notebooks/figures/eda/` créé automatiquement au premier appel
- [x] Ligne S5-15 ajoutée dans `docs/roadmap_phase1.md` sprint 5

---

## Artefacts produits

| Fichier | Chemin | Commitable | Utilisé par |
|---------|--------|:----------:|-------------|
| `eda_plots.py` (9 fonctions) | `src/evaluation/eda_plots.py` | ✅ | Notebook, scripts EDA |
| Notebook mis à jour | `notebooks/01_data_exploration.ipynb` | ✅ | Exploration manuelle |
| Figures EDA (PNG) | `notebooks/figures/eda/*.png` | ❌ (gitignore figures) | Rapport |
| Sprint doc | `docs/sprints/sprint_5/S515_eda_refined_plots.md` | ✅ | Ce fichier |

---

## Commandes de vérification

```bash
# Import propre
python -c "
from src.evaluation.eda_plots import (
    plot_boxplots_by_pump_id, plot_violin_by_pump_id,
    plot_operational_hour_windows, plot_fault_rate_heatmap_pump,
    plot_boxplots_by_equipment_location, plot_violin_by_location,
    plot_fault_rate_heatmap_equipment, plot_correlation_by_equipment,
    plot_pairplot_by_equipment, _save_fig
)
print('OK')
"

# Tests existants toujours verts
pytest tests/ -v

# Lancer les cellules du notebook (optionnel, nécessite jupyter)
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb \
    --output notebooks/01_data_exploration_executed.ipynb
```

---

## Questions ouvertes

- `TODO(arnaud)` : Les fenêtres `Operational_Hours` sont actuellement équidistantes — envisager
  un découpage par quantile (Q1/Q2/Q3/Q4) si la distribution temporelle est très asymétrique ?
- `TODO(fred)` : Les Pump_ID ont-ils une signification industrielle particulière (lots, sites,
  âge) permettant d'interpréter les différences inter-pompes observées dans les boxplots ?
