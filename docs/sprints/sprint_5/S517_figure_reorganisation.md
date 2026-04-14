# S517 — Réorganisation figures EDA + correction légendes

**Sprint** : 5  
**Date** : 2026-04-12  
**Auteur** : Léonard Rivals  
**Statut** : ✅ Complété  

---

## Problème

Les figures EDA générées dans `notebooks/01_data_exploration.ipynb` n'étaient pas toutes rangées dans les sous-dossiers dédiés :

- **4 plots pump_*** (`pump_boxplots_by_pump_id`, `pump_violin_by_pump_id`, `pump_boxplots_hour_windows`, `pump_fault_rate_heatmap`) sauvegardés dans `figures/eda/` (racine) au lieu de `figures/eda/pump_maintenance/`
- **5 plots equip_*** (`equip_boxplots_by_location`, `equip_violin_by_location`, `equip_fault_rate_heatmap`, `equip_correlation_by_equipment`, `equip_pairplot_by_equipment`) sauvegardés dans `figures/eda/` (racine) au lieu de `figures/eda/equipment_monitoring/`
- **Dossier `pump_boxplots_by_pump_id_by_hour_windows/`** (5 fichiers feature PNG) existait comme sous-dossier ad hoc au lieu d'être dans `pump_maintenance/`
- **5 fonctions `eda_plots.py`** avaient des légendes fragiles (couleur vert/rouge pouvait être inversée selon la version de seaborn)

---

## Changements effectués

### 1. Déplacement des fichiers existants

| Fichier | Avant | Après |
|---------|-------|-------|
| `pump_boxplots_by_pump_id.png` | `figures/eda/` | `figures/eda/pump_maintenance/` |
| `pump_violin_by_pump_id.png` | `figures/eda/` | `figures/eda/pump_maintenance/` |
| `pump_boxplots_hour_windows.png` | `figures/eda/` | `figures/eda/pump_maintenance/` |
| `pump_fault_rate_heatmap.png` | `figures/eda/` | `figures/eda/pump_maintenance/` |
| `pump_boxplots_by_pump_id_by_hour_windows/{feat}.png` (×5) | `figures/eda/pump_boxplots_by_pump_id_by_hour_windows/` | `figures/eda/pump_maintenance/` |
| `equip_boxplots_by_location.png` | `figures/eda/` | `figures/eda/equipment_monitoring/` |
| `equip_violin_by_location.png` | `figures/eda/` | `figures/eda/equipment_monitoring/` |
| `equip_fault_rate_heatmap.png` | `figures/eda/` | `figures/eda/equipment_monitoring/` |
| `equip_correlation_by_equipment.png` | `figures/eda/` | `figures/eda/equipment_monitoring/` |
| `equip_pairplot_by_equipment.png` | `figures/eda/` | `figures/eda/equipment_monitoring/` |

Le dossier vide `pump_boxplots_by_pump_id_by_hour_windows/` a été supprimé.

### 2. Chemins corrigés dans `notebooks/01_data_exploration.ipynb`

- Cellules 30–34 : `FIGURES_EDA / "equip_*.png"` → `MONITORING_FIGURES_DIR / "equip_*.png"`
- Cellules 54–58 : `FIGURES_EDA / "pump_*.png"` → `FIGURES_DIR / "pump_*.png"`
- Cellule 57 : `PUMP_ID_HOUR_DIR = FIGURES_EDA / "pump_boxplots_by_pump_id_by_hour_windows"` → `PUMP_ID_HOUR_DIR = FIGURES_DIR`

### 3. Correction légendes dans `src/evaluation/eda_plots.py`

**Problème** : `ax.legend(labels=["Normal (0)", "Faulty (1)"])` remplace les textes positionnellement mais ne garantit pas la correspondance couleur↔label selon la version de seaborn.

**Solution** : Utilisation de handles `matplotlib.patches.Patch` explicites (même pattern que `plot_boxplots_by_pump_id_hour_windows()` déjà existant).

Fonctions corrigées (les 5 utilisant seaborn avec hue) :

| Fonction | Lignes avant correction |
|----------|------------------------|
| `plot_boxplots_by_pump_id()` | ~897–902 |
| `plot_violin_by_pump_id()` | ~1000–1005 |
| `plot_operational_hour_windows()` | ~1125–1130 |
| `plot_boxplots_by_equipment_location()` | ~1457–1462 |
| `plot_violin_by_location()` | ~1576–1581 |

**Pattern appliqué** :
```python
from matplotlib.patches import Patch
_legend_handles = [
    Patch(color=COLORS_LABEL[0], label="Normal (0)"),
    Patch(color=COLORS_LABEL[1], label=f"{label_name} (1)"),
]
ax.legend(handles=_legend_handles, title=label_name, ...)
```

---

## Structure finale de `notebooks/figures/eda/`

```
figures/eda/
├── pump_maintenance/       ← tous les plots Dataset 1 (pump)
│   ├── boxplots_by_maintenance.png
│   ├── correlations.png
│   ├── flow_rate.png
│   ├── histograms_by_maintenance.png
│   ├── kde_by_maintenance.png
│   ├── label_distribution.png
│   ├── pairplot_by_maintenance.png
│   ├── pressure.png
│   ├── pump_boxplots_by_pump_id.png
│   ├── pump_boxplots_hour_windows.png
│   ├── pump_fault_rate_heatmap.png
│   ├── pump_violin_by_pump_id.png
│   ├── rpm.png
│   ├── temperature.png
│   ├── temporal_by_maintenance.png
│   ├── temporal_drift.png
│   ├── vibration.png
│   └── violin_by_maintenance.png
└── equipment_monitoring/   ← tous les plots Dataset 2 (equipment)
    ├── boxplots_by_equipment_faulty.png
    ├── boxplots_by_faulty.png
    ├── equip_boxplots_by_location.png
    ├── equip_correlation_by_equipment.png
    ├── equip_fault_rate_heatmap.png
    ├── equip_pairplot_by_equipment.png
    ├── equip_violin_by_location.png
    ├── histograms_by_faulty.png
    ├── kde_by_equipment_faulty.png
    ├── kde_by_faulty.png
    ├── label_distribution.png
    ├── pairplot_by_faulty.png
    ├── violin_by_equipment_faulty.png
    └── violin_by_faulty.png
```
