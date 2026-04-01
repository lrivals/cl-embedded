# S1-07 — Implémenter `metrics.py` (AA, AF, BWT)

| Champ | Valeur |
|-------|--------|
| **ID** | S1-07 |
| **Sprint** | Sprint 1 — Semaine 1 (15–22 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | — |
| **Fichier cible** | `src/evaluation/metrics.py` |

---

## Objectif

Implémenter les métriques standard de l'évaluation Continual Learning :

- **AA** (Average Accuracy) — accuracy moyenne sur toutes les tâches après entraînement complet
- **AF** (Average Forgetting) — chute moyenne d'accuracy entre le pic et la fin par tâche
- **BWT** (Backward Transfer) — impact de l'apprentissage futur sur les tâches passées
- **FWT** (Forward Transfer, optionnel) — facilitation de l'apprentissage d'une nouvelle tâche

L'interface principale est la **matrice d'accuracy** `acc_matrix [T, T]` où `acc_matrix[i, j]` = accuracy sur la tâche j après entraînement jusqu'à la tâche i.

**Critère de succès** : `compute_cl_metrics(acc_matrix)` retourne un dict avec les quatre métriques scalaires, et les tests `tests/test_metrics.py` passent.

> **Statut** : ✅ **Implémenté** — `src/evaluation/metrics.py` présent et fonctionnel.

---

## Interface implémentée

### `compute_cl_metrics(acc_matrix, random_baseline=None)`

```python
def compute_cl_metrics(
    acc_matrix: np.ndarray,              # [T, T], NaN pour j > i
    random_baseline: np.ndarray | None,  # [T], défaut 0.5 (binaire)
) -> dict:
    """
    Retourne :
        aa   : float — Average Accuracy (dernière ligne de acc_matrix)
        af   : float — Average Forgetting (≥ 0 = oubli)
        bwt  : float — Backward Transfer (< 0 = oubli)
        fwt  : float — Forward Transfer
        forgetting_per_task : list[float]
        bwt_per_task        : list[float]
        acc_matrix          : list[list] — JSON-sérialisable
        n_tasks             : int
    """
```

**Formules** (référence : DeLange2021Survey) :

| Métrique | Formule |
|----------|---------|
| AA | `mean(acc_matrix[T-1, :])` |
| AF | `mean(max_j(acc_matrix[:, j]) - acc_matrix[T-1, j])` pour j < T |
| BWT | `mean(acc_matrix[T-1, j] - acc_matrix[j, j])` pour j < T |
| FWT | `mean(acc_matrix[j-1, j] - random_baseline[j])` pour j > 0 |

### `format_metrics_report(metrics, model_name, baseline_finetune, baseline_joint)`

Formate un rapport texte multi-ligne pour affichage terminal et archivage dans `experiments/`.

### `save_metrics(metrics, output_path, extra_info)`

Sauvegarde en JSON dans `experiments/exp_XXX/results/metrics.json`.

### `accuracy_binary(y_true, y_pred, threshold=0.5)`

Accuracy binaire avec seuil, utilisée dans les boucles d'évaluation.

---

## Critères d'acceptation

- [x] `from src.evaluation.metrics import compute_cl_metrics` — aucune erreur d'import
- [x] `compute_cl_metrics(M)` retourne les clés `aa`, `af`, `bwt`, `fwt` pour une matrice 3×3
- [x] `af ≥ 0` pour une matrice présentant de l'oubli (acc décroissante hors diagonale)
- [x] `bwt < 0` pour une matrice présentant de l'oubli catastrophique
- [x] `save_metrics()` crée le répertoire parent si absent et produit un JSON valide
- [x] `format_metrics_report()` inclut les comparaisons baseline fine-tuning / joint
- [x] `ruff check src/evaluation/metrics.py` et `black --check` passent

---

## Sorties attendues à reporter ailleurs

| Élément | Où reporter |
|---------|-------------|
| `acc_matrix` JSON | `experiments/exp_001_ewc_dataset2/results/metrics.json` |
| Rapport texte | Affiché par `scripts/train_ewc.py` + sauvegardé dans `experiments/exp_001/` |
| AA, AF, BWT | Table comparative du manuscrit (Section 4.2) |

---

## Questions ouvertes

- `TODO(arnaud)` : inclure la métrique **Intransigence** (résistance à l'apprentissage de nouvelles tâches) pour la comparaison M2 vs M3 ?
- `FIXME(gap1)` : normaliser AF par le nombre de tâches (T-1) ou reporter les valeurs par tâche séparément dans le manuscrit ?
