# S6-04 — `plot_performance_by_pump_id_bar()` dans `plots.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S6-04 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | — |
| **Fichiers cibles** | `src/evaluation/plots.py` |
| **Complété le** | 14 avril 2026 |
| **Statut** | ✅ Déjà implémenté |

---

## Objectif

Implémenter `plot_performance_by_pump_id_bar()` dans `src/evaluation/plots.py` :
barplot groupé montrant l'accuracy finale par Pump_ID (axe X = Pump_ID 1–5, barres = modèles).
Révèle quels Pump_ID sont les plus difficiles à retenir après entraînement CL complet.

---

## Résultat de vérification

La fonction existe déjà aux lignes 618–685 de `src/evaluation/plots.py` avec la signature :

```python
def plot_performance_by_pump_id_bar(
    results: dict[str, dict],
    pump_ids: list[int],
    title: str = "Accuracy finale par Pump_ID",
    ax: plt.Axes | None = None,
) -> plt.Figure:
```

**→ Aucune modification nécessaire.**

Fonctionnalités implémentées :
- Barres groupées par modèle (`tab10` colormap, `n_models` barres par Pump_ID)
- Largeur de barre adaptative : `0.8 / max(n_models, 1)`
- Légende positionnée en bas à droite, grille Y semi-transparente
- Retourne `plt.Figure` prêt pour `save_figure()`
- Supporte un axes existant (`ax` parameter) pour intégration dans une figure composite

---

## Critères d'acceptation

- [x] `from src.evaluation.plots import plot_performance_by_pump_id_bar` — aucune erreur
- [x] `fig = plot_performance_by_pump_id_bar({"EWC": {1: 0.7, 2: 0.6}}, pump_ids=[1,2])` retourne `plt.Figure`
- [x] Barres visibles pour chaque modèle × Pump_ID
- [x] Docstring NumPy complète

---

## Commandes de vérification

```bash
python -c "
import matplotlib; matplotlib.use('Agg')
from src.evaluation.plots import plot_performance_by_pump_id_bar
results = {
    'EWC': {1: 0.72, 2: 0.65, 3: 0.78, 4: 0.51, 5: 0.70},
    'HDC': {1: 0.61, 2: 0.55, 3: 0.68, 4: 0.44, 5: 0.62},
}
fig = plot_performance_by_pump_id_bar(results, pump_ids=[1, 2, 3, 4, 5])
print(type(fig))
print('plot_performance_by_pump_id_bar OK')
"
```

---

## Questions ouvertes

Aucune — tâche close.
