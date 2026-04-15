# S6-05 — `plot_performance_heatmap_equipment_location()` dans `plots.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S6-05 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | — |
| **Fichiers cibles** | `src/evaluation/plots.py` |
| **Complété le** | 14 avril 2026 |
| **Statut** | ✅ Déjà implémenté |

---

## Objectif

Implémenter `plot_performance_heatmap_equipment_location()` dans `src/evaluation/plots.py` :
heatmap 2D (equipment type × location) montrant l'accuracy moyenne par cellule sur le Dataset 2.
Utile pour détecter des patterns géographiques ou par type d'équipement après entraînement CL.

---

## Résultat de vérification

La fonction existe déjà à partir de la ligne 688 de `src/evaluation/plots.py` avec la signature :

```python
def plot_performance_heatmap_equipment_location(
    results_by_cell: dict[str, dict[tuple[str, str], float]],
    equipment_types: list[str],
    locations: list[str],
    title: str = "Accuracy par équipement × location",
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
```

**→ Aucune modification nécessaire.**

Fonctionnalités implémentées :
- Grille 2×2 pour 4 modèles (1×N sinon)
- Heatmap `n_equipment × n_locations` par modèle
- Cellules NaN affichées en grisé (équipements non vus dans le dataset)
- Annotations de valeur dans chaque cellule
- Colormap `YlOrRd_r` cohérente avec `plot_accuracy_matrix()`

---

## Critères d'acceptation

- [x] `from src.evaluation.plots import plot_performance_heatmap_equipment_location` — aucune erreur
- [x] Retourne `plt.Figure` prêt pour `save_figure()`
- [x] Gestion des cellules manquantes (NaN → grisé)
- [x] Docstring NumPy complète

---

## Commandes de vérification

```bash
python -c "
import matplotlib; matplotlib.use('Agg')
from src.evaluation.plots import plot_performance_heatmap_equipment_location
results = {
    'EWC':  {('Pump', 'Atlanta'): 0.95, ('Turbine', 'Chicago'): 0.88},
    'HDC':  {('Pump', 'Atlanta'): 0.82, ('Turbine', 'Chicago'): 0.79},
}
fig = plot_performance_heatmap_equipment_location(
    results,
    equipment_types=['Pump', 'Turbine', 'Compressor'],
    locations=['Atlanta', 'Chicago', 'Dallas'],
)
print(type(fig))
print('plot_performance_heatmap_equipment_location OK')
"
```

---

## Questions ouvertes

Aucune — tâche close.
