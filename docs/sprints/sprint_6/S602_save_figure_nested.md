# S6-01 — Vérifier `save_figure()` pour création auto de sous-dossiers

| Champ | Valeur |
|-------|--------|
| **ID** | S6-01 |
| **Sprint** | Sprint 6 — Phase 1 Extension (≥ 15 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1h |
| **Dépendances** | — |
| **Fichiers cibles** | `src/evaluation/plots.py` |
| **Complété le** | 14 avril 2026 |
| **Statut** | ✅ Déjà implémenté |

---

## Objectif

Vérifier que `save_figure()` dans `src/evaluation/plots.py` appelle
`Path(output_path).parent.mkdir(parents=True, exist_ok=True)` avant `plt.savefig()`,
de façon à créer automatiquement les sous-dossiers imbriqués requis par les notebooks
Sprint 7 et Sprint 8.

**Pattern cible pour les notebooks :**
```python
FIGURES_DIR = Path("../notebooks/figures/cl_evaluation")
fig_path = FIGURES_DIR / "ewc" / "monitoring" / "by_equipment" / "acc_matrix.png"
save_figure(fig, fig_path)  # doit créer les sous-dossiers automatiquement
```

**Critère de succès** : `save_figure(fig, "a/b/c/d.png")` ne lève pas de `FileNotFoundError`
même si les dossiers `a/b/c/` n'existent pas encore.

---

## Résultat de vérification

La fonction `save_figure()` (lignes 808–830 de `plots.py`) appelle déjà :

```python
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(path, dpi=dpi, bbox_inches="tight")
print(f"[plots] Figure saved → {path}")
plt.close(fig)
```

**→ Aucune modification nécessaire.** `parents=True, exist_ok=True` est déjà présent.

---

## Critères d'acceptation

- [x] `save_figure(fig, "nested/path/figure.png")` crée les sous-dossiers automatiquement
- [x] Format inféré depuis l'extension (`.png`, `.pdf`, `.svg`)
- [x] `plt.close(fig)` appelé après sauvegarde (évite les fuites mémoire)
- [x] DPI configurable (défaut : `FIGURE_DPI = 150`)

---

## Commandes de vérification

```bash
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.plots import save_figure
import tempfile, pathlib

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / 'a' / 'b' / 'c' / 'test.png'
    save_figure(fig, path)
    assert path.exists(), 'Fichier non créé'
print('save_figure nested paths OK')
"
```

---

## Questions ouvertes

Aucune — tâche close.
