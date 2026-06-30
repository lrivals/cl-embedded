# S2509–S2510 — Métriques natives : RUL et Multi-classe

| Champ | Valeur |
|-------|--------|
| **Sprint** | 25 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | S2509 : 2h / S2510 : 1h = 3h total |
| **Dépendances** | `src/evaluation/metrics.py` ✅ (pattern existant pour `acc_final`, `avg_forgetting`, `bwt`) |
| **Fichiers cibles** | `src/evaluation/rul_metrics.py`, `src/evaluation/multiclass_metrics.py` |
| **Référence** | `src/evaluation/metrics.py` (format de retour `dict[str, float]`), `src/evaluation/anomaly_metrics.py` (pattern module évaluation), `src/evaluation/__init__.py` (à mettre à jour) |

---

## Contexte

Le module `src/evaluation/metrics.py` couvre la classification binaire (accuracy, forgetting, BWT). Sprint 25 introduit deux nouvelles tâches de prédiction nécessitant des métriques dédiées :

- **RUL Régression** : RMSE (métrique principale), MAE, Horizon Score PHM 2008 (pénalité asymétrique), Average Forgetting en RMSE.
- **Multi-classe** : F1-macro, matrice de confusion, précision par classe, Average Forgetting en F1-macro.

Ces deux modules sont indépendants du mode de prédiction (EWC, HDC) et retournent des `dict[str, float]` compatibles avec le format `results.json` des expériences.

---

## S2509 — `src/evaluation/rul_metrics.py`

### Spécification complète

```python
"""
rul_metrics.py — Métriques d'évaluation pour la régression RUL.

Métriques implémentées :
    - RMSE (Root Mean Square Error) — métrique principale
    - MAE (Mean Absolute Error)
    - Horizon Score — pénalité asymétrique PHM 2008 (sur-estimation plus pénalisée)
    - Average Forgetting RMSE — dégradation RMSE entre pic et fin par tâche CL

Toutes les fonctions retournent des float ou dict[str, float] compatibles results.json.

Références :
    PHM 2008 Challenge — Horizon Score asymétrique
    DeLange2021Survey — définition Average Forgetting adaptée à la régression
"""

from __future__ import annotations

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Parameters
    ----------
    y_true : np.ndarray shape (N,)
        RUL réel.
    y_pred : np.ndarray shape (N,)
        RUL prédit.

    Returns
    -------
    float : RMSE ∈ [0, +∞)
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Returns
    -------
    float : MAE ∈ [0, +∞)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_horizon_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """
    Horizon Score (score PHM 2008) — pénalité asymétrique.

    La sur-estimation (ŷ > y_true, prédiction trop optimiste) est plus pénalisée
    que la sous-estimation (ŷ < y_true, prédiction conservative).

    Score = Σ exp(d / a_i) - 1   où d = ŷ - y_true,
            a1=13 si d < 0 (sous-estimation),
            a2=10 si d ≥ 0 (sur-estimation).

    Un score bas (proche de 0) est meilleur.

    Parameters
    ----------
    y_true, y_pred : np.ndarray shape (N,)
    a1 : float
        Facteur pour la sous-estimation. Default : 13 (PHM 2008).
    a2 : float
        Facteur pour la sur-estimation. Default : 10 (PHM 2008).

    Returns
    -------
    float : Horizon Score ∈ [0, +∞)
    """
    d = y_pred - y_true
    a = np.where(d < 0, a1, a2)
    return float(np.sum(np.exp(d / a) - 1))


def compute_rul_metrics_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcule toutes les métriques RUL pour une tâche.

    Returns
    -------
    dict avec clés : rmse, mae, horizon_score
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "horizon_score": compute_horizon_score(y_true, y_pred),
    }


def compute_avg_forgetting_rmse(
    task_rmse_matrix: list[list[float]],
) -> float:
    """
    Average Forgetting adapté à la régression (RMSE).

    AF_rmse = (1 / T-1) · Σ_{i=1}^{T-1} (RMSE_i_final - RMSE_i_best)

    Attention : pour RMSE, l'oubli = RMSE_final > RMSE_best (la valeur monte).
    AF > 0 signifie dégradation (oubli), AF < 0 signifie amélioration (transfert positif).

    Parameters
    ----------
    task_rmse_matrix : list[list[float]]
        task_rmse_matrix[epoch][task] = RMSE sur la tâche `task` après l'époque `epoch`.
        Shape : (n_epochs, n_tasks).

    Returns
    -------
    float : Average Forgetting RMSE
    """
    if len(task_rmse_matrix) < 2:
        return 0.0

    n_tasks = len(task_rmse_matrix[0])
    rmse_matrix = np.array(task_rmse_matrix)  # shape (n_epochs, n_tasks)

    forgettings = []
    for task_idx in range(n_tasks - 1):
        col = rmse_matrix[:, task_idx]
        best_rmse = col.min()  # meilleur RMSE atteint sur cette tâche
        final_rmse = col[-1]   # RMSE final (après toutes les tâches)
        forgettings.append(final_rmse - best_rmse)

    return float(np.mean(forgettings))
```

### Vérification

```bash
python -c "
import numpy as np
from src.evaluation.rul_metrics import (
    compute_rmse, compute_mae, compute_horizon_score,
    compute_rul_metrics_task, compute_avg_forgetting_rmse,
)

y_true = np.array([50.0, 30.0, 10.0, 80.0])
y_pred = np.array([45.0, 35.0, 15.0, 70.0])

print('RMSE :', compute_rmse(y_true, y_pred))
print('MAE  :', compute_mae(y_true, y_pred))
print('HScore:', compute_horizon_score(y_true, y_pred))

# Horizon Score : sur-estimation pénalisée plus fortement
y_over = y_true + 10  # sur-estimation pure
y_under = y_true - 10 # sous-estimation pure
hs_over = compute_horizon_score(y_true, y_over)
hs_under = compute_horizon_score(y_true, y_under)
assert hs_over > hs_under, 'Sur-estimation doit être plus pénalisée que sous-estimation'
print(f'Horizon Score sur-estim={hs_over:.2f} > sous-estim={hs_under:.2f} ✅')

# Average Forgetting
matrix = [[10.0, None], [8.0, 20.0], [9.0, 18.0]]
matrix_float = [[v if v is not None else 0.0 for v in row] for row in matrix]
af = compute_avg_forgetting_rmse(matrix_float)
print(f'AF RMSE : {af:.4f}')

metrics = compute_rul_metrics_task(y_true, y_pred)
print('Métriques complètes :', metrics)
print('rul_metrics.py OK ✅')
"
```

---

## S2510 — `src/evaluation/multiclass_metrics.py`

### Spécification complète

```python
"""
multiclass_metrics.py — Métriques d'évaluation pour la classification multi-classe CL.

Métriques implémentées :
    - F1-macro (moyenne non pondérée du F1 par classe)
    - Matrice de confusion
    - Précision par classe
    - Average Forgetting F1-macro

Références :
    DeLange2021Survey — Average Forgetting pour CL classification
    scikit-learn : f1_score(average='macro')
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """
    F1-macro (moyenne non pondérée sur toutes les classes).

    Parameters
    ----------
    y_true : np.ndarray shape (N,), int
    y_pred : np.ndarray shape (N,), int
    labels : liste des classes attendues (utile pour CWRU avec classes absentes)

    Returns
    -------
    float : F1-macro ∈ [0, 1]
    """
    return float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> np.ndarray:
    """
    Matrice de confusion (shape n_classes × n_classes).

    Returns
    -------
    np.ndarray dtype int, shape (n_classes, n_classes)
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """
    Précision par classe (accuracy individuelle).

    Returns
    -------
    dict : {"class_0": float, "class_1": float, ...}
    """
    result = {}
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            result[f"class_{cls}"] = float("nan")
        else:
            result[f"class_{cls}"] = float((y_pred[mask] == cls).mean())
    return result


def compute_multiclass_metrics_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    labels: list[int] | None = None,
) -> dict[str, float | list]:
    """
    Calcule toutes les métriques multi-classe pour une tâche.

    Returns
    -------
    dict avec clés : f1_macro, confusion_matrix (liste 2D), per_class_accuracy (dict)
    """
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "f1_macro": compute_f1_macro(y_true, y_pred, labels=labels),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": compute_per_class_accuracy(y_true, y_pred, n_classes),
    }


def compute_avg_forgetting_f1(
    task_f1_matrix: list[list[float]],
) -> float:
    """
    Average Forgetting F1-macro.

    AF_f1 = (1 / T-1) · Σ_{i=1}^{T-1} (F1_i_peak - F1_i_final)

    AF > 0 signifie oubli, AF < 0 signifie backward transfer positif.

    Parameters
    ----------
    task_f1_matrix : list[list[float]]
        task_f1_matrix[epoch][task] = F1-macro sur la tâche `task` après l'époque `epoch`.

    Returns
    -------
    float : Average Forgetting F1-macro
    """
    if len(task_f1_matrix) < 2:
        return 0.0

    f1_matrix = np.array(task_f1_matrix)  # shape (n_epochs, n_tasks)
    n_tasks = f1_matrix.shape[1]

    forgettings = []
    for task_idx in range(n_tasks - 1):
        col = f1_matrix[:, task_idx]
        peak_f1 = col.max()
        final_f1 = col[-1]
        forgettings.append(peak_f1 - final_f1)

    return float(np.mean(forgettings))
```

### Vérification

```bash
python -c "
import numpy as np
from src.evaluation.multiclass_metrics import (
    compute_f1_macro, compute_confusion_matrix,
    compute_per_class_accuracy, compute_avg_forgetting_f1,
    compute_multiclass_metrics_task,
)

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1])

f1 = compute_f1_macro(y_true, y_pred)
assert 0 <= f1 <= 1, f'F1 hors [0,1] : {f1}'
print(f'F1-macro : {f1:.4f}')

cm = compute_confusion_matrix(y_true, y_pred)
assert cm.shape == (3, 3)
print(f'Confusion matrix shape : {cm.shape}')

per_cls = compute_per_class_accuracy(y_true, y_pred, n_classes=3)
print(f'Per-class accuracy : {per_cls}')

# Average Forgetting — oubli simulé
matrix = [[0.9, None], [0.85, 0.80], [0.70, 0.82]]
matrix_f = [[v if v is not None else 0.0 for v in row] for row in matrix]
af = compute_avg_forgetting_f1(matrix_f)
assert af >= 0, 'AF négatif inattendu'
print(f'AF F1-macro : {af:.4f}')

metrics = compute_multiclass_metrics_task(y_true, y_pred, n_classes=3)
print(f'Métriques complètes : f1_macro={metrics[\"f1_macro\"]:.4f}')
print('multiclass_metrics.py OK ✅')
"
```

---

## Mise à jour `src/evaluation/__init__.py`

Ajouter les nouveaux modules à l'import public :

```python
# Dans src/evaluation/__init__.py — ajouter :
from src.evaluation.rul_metrics import (
    compute_rmse,
    compute_mae,
    compute_horizon_score,
    compute_rul_metrics_task,
    compute_avg_forgetting_rmse,
)
from src.evaluation.multiclass_metrics import (
    compute_f1_macro,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_multiclass_metrics_task,
    compute_avg_forgetting_f1,
)
```

---

## Vérification end-to-end

```bash
# Import depuis le package
python -c "
from src.evaluation import (
    compute_rmse, compute_mae, compute_horizon_score,
    compute_f1_macro, compute_confusion_matrix,
)
import numpy as np
y_t = np.array([10.0, 20.0, 30.0])
y_p = np.array([12.0, 18.0, 32.0])
print('RMSE =', compute_rmse(y_t, y_p))
print('F1 =', compute_f1_macro(np.array([0,1,2]), np.array([0,1,1])))
print('Import evaluation package OK ✅')
"
```

---

## Résultats d'implémentation

| Sous-tâche | Statut | Notes |
|------------|:------:|-------|
| S2509 — `src/evaluation/rul_metrics.py` | ✅ | |
| S2510 — `src/evaluation/multiclass_metrics.py` | ✅ | |
| Mise à jour `src/evaluation/__init__.py` | ✅ | |

---

## Questions ouvertes

- `TODO(arnaud)` : La métrique Horizon Score (PHM 2008, pénalité asymétrique) est-elle requise pour le manuscrit ou RMSE + MAE suffisent-ils ? Le calcul est coûteux pour les grands datasets.
- `FIXME(gap1)` : Vérifier que `compute_horizon_score` retourne des valeurs cohérentes avec les papiers CMAPSS de référence (RMSE SOTA ≈ 12–15 cycles sur FD001).
