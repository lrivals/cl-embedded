# S11-11 à S11-17 — Extension Feature Importance : CWRU et Pronostia

| Champ | Valeur |
|-------|--------|
| **IDs** | S11-11, S11-12, S11-13, S11-14, S11-15, S11-16, S11-17 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🔴 (S11-11–13, S11-16), 🟡 (S11-14, S11-15, S11-17) |
| **Durée estimée** | 8h total |
| **Dépendances** | S11-01 (`feature_importance.py` ✅), exp_071–085 (CWRU), exp_050–065 (Pronostia) |

---

## S11-11 — Extension `feature_importance.py`

### Constantes à ajouter

Importer depuis les modules dataset pour éviter la duplication :

```python
from src.data.cwru_dataset import FEATURE_COLS as FEATURE_NAMES_CWRU
from src.data.pronostia_dataset import FEATURE_NAMES as FEATURE_NAMES_PRONOSTIA

# Groupement par canal — pour visualisation Pronostia
CHANNEL_GROUPS_PRONOSTIA: dict[str, list[str]] = {
    "acc_horiz": [
        "mean_acc_horiz", "std_acc_horiz", "rms_acc_horiz",
        "kurtosis_acc_horiz", "peak_acc_horiz", "crest_factor_acc_horiz",
    ],
    "acc_vert": [
        "mean_acc_vert", "std_acc_vert", "rms_acc_vert",
        "kurtosis_acc_vert", "peak_acc_vert", "crest_factor_acc_vert",
    ],
    "temporal": ["temporal_position"],
}
```

### Nouvelle fonction `permutation_importance_per_task()`

```python
def permutation_importance_per_task(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    tasks: list[dict],          # [{"task_name": str, "X": np.ndarray, "y": np.ndarray}, ...]
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Lance permutation_importance sur chaque tâche séparément.

    Returns
    -------
    dict[str, dict[str, float]]
        {task_name: {feature_name: score}}
    """
```

La fonction appelle `permutation_importance` pour chaque élément de `tasks`, retourne
un dict `{task_name: importances}`. Le calcul global (toutes tâches concaténées) est
obtenu en appelant `permutation_importance` normalement avec `X = np.concatenate(...)`.

---

## S11-12 et S11-13 — Extension des scripts d'entraînement

### Où insérer dans `train_kmeans.py` et `train_mahalanobis.py`

Après la boucle CL d'entraînement, avant la sauvegarde des métriques :

```python
from src.evaluation.feature_importance import (
    permutation_importance,
    permutation_importance_per_task,
    FEATURE_NAMES_CWRU,
    FEATURE_NAMES_PRONOSTIA,
)

# Sélectionner les noms de features selon le dataset
feature_names = _resolve_feature_names(cfg)  # voir ci-dessous

# Construire la liste de tâches pour l'analyse per-task
task_arrays = []
for task in tasks:
    X_task, y_task = _extract_test_arrays(task)  # depuis test_loader ou val_loader
    task_arrays.append({"task_name": task["domain"], "X": X_task, "y": y_task})

# Analyse globale
X_all = np.concatenate([t["X"] for t in task_arrays])
y_all = np.concatenate([t["y"] for t in task_arrays])
global_imp = permutation_importance(predict_fn, X_all, y_all, feature_names)

# Analyse per-task
per_task_imp = permutation_importance_per_task(predict_fn, task_arrays, feature_names)

# Sauvegarder dans results/feature_importance.json
importance_results = {
    "model": cfg["model"]["name"],
    "dataset": cfg["dataset"]["name"],
    "scenario": cfg["scenario"],
    "global": {"permutation_importance": global_imp},
    "per_task": {
        name: {"permutation_importance": imp}
        for name, imp in per_task_imp.items()
    },
}
```

### `_resolve_feature_names(cfg)` — logique de sélection

```python
def _resolve_feature_names(cfg: dict) -> list[str]:
    dataset = cfg["dataset"]["name"]
    if dataset == "cwru":
        return FEATURE_NAMES_CWRU          # 9 features
    if dataset == "pronostia":
        return FEATURE_NAMES_PRONOSTIA     # 13 features
    return FEATURE_NAMES_MONITORING        # 4 features (défaut)
```

### Cas particulier Pronostia — pas de `test_loader` en mode CL

`get_pronostia_dataloaders` n'expose que `train_loader` et `val_loader`. Utiliser
`val_loader` pour l'analyse d'importance (le split est stratifié → représentatif) :

```python
def _extract_test_arrays(task: dict) -> tuple[np.ndarray, np.ndarray]:
    loader = task.get("test_loader") or task["val_loader"]
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.squeeze().numpy())
    return np.concatenate(X_list), np.concatenate(y_list)
```

---

## S11-14 et S11-15 — Extension EWC et HDC

### EWC — gradient saliency per-task

En plus de `permutation_importance_per_task`, calculer `gradient_saliency` sur
`X_all` (une seule passe, rapide) et l'inclure dans le JSON :

```json
"global": {
    "permutation_importance": {"rms": 0.041, ...},
    "gradient_saliency":      {"kurtosis": 0.038, ...}
}
```

### HDC — masking per-task

En plus de `permutation_importance_per_task`, calculer `feature_masking_importance`
sur `X_all`. Les trois méthodes sont stockées :

```json
"global": {
    "permutation_importance":    {"rms": 0.041, ...},
    "feature_masking_importance": {"kurtosis": 0.035, ...}
}
```

---

## S11-16 et S11-17 — Expériences à lancer

| Exp | Modèle | Dataset | Scénario | Script |
|-----|--------|---------|----------|--------|
| exp_100 | KMeans | CWRU | by_fault_type | `train_kmeans.py --config configs/cwru_by_fault_config.yaml --exp_id exp_100` |
| exp_101 | KMeans | CWRU | by_severity | `train_kmeans.py --config configs/cwru_by_severity_config.yaml --exp_id exp_101` |
| exp_102 | KMeans | Pronostia | by_condition | `train_kmeans.py --config configs/kmeans_pronostia_by_condition_config.yaml --exp_id exp_102` |
| exp_103 | Mahalanobis | CWRU | by_fault_type | `train_mahalanobis.py --config configs/cwru_by_fault_config.yaml --exp_id exp_103` |
| exp_104 | Mahalanobis | CWRU | by_severity | `train_mahalanobis.py --config configs/cwru_by_severity_config.yaml --exp_id exp_104` |
| exp_105 | Mahalanobis | Pronostia | by_condition | `train_mahalanobis.py --config configs/mahalanobis_pronostia_by_condition_config.yaml --exp_id exp_105` |
| exp_106 | EWC | CWRU | by_fault_type | `train_ewc.py --config configs/cwru_by_fault_config.yaml --exp_id exp_106` |
| exp_107 | EWC | CWRU | by_severity | `train_ewc.py --config configs/cwru_by_severity_config.yaml --exp_id exp_107` |
| exp_108 | EWC | Pronostia | by_condition | `train_ewc.py --config configs/ewc_pronostia_by_condition_config.yaml --exp_id exp_108` |
| exp_109 | HDC | CWRU | by_fault_type | `train_hdc.py --config configs/cwru_by_fault_config.yaml --exp_id exp_109` |
| exp_110 | HDC | CWRU | by_severity | `train_hdc.py --config configs/cwru_by_severity_config.yaml --exp_id exp_110` |
| exp_111 | HDC | Pronostia | by_condition | `train_hdc.py --config configs/hdc_pronostia_by_condition_config.yaml --exp_id exp_111` |

### Structure JSON cible (commune à tous)

```json
{
  "model": "KMeans",
  "dataset": "cwru",
  "scenario": "by_fault_type",
  "global": {
    "permutation_importance": {
      "kurtosis": 0.041, "rms": 0.038, "crest": 0.031,
      "sd": 0.025, "form": 0.018, "skewness": 0.012,
      "max": 0.009, "mean": 0.005, "min": 0.002
    }
  },
  "per_task": {
    "ball":       {"permutation_importance": {"kurtosis": 0.052, ...}},
    "inner_race": {"permutation_importance": {"rms": 0.049, ...}},
    "outer_race": {"permutation_importance": {"crest": 0.044, ...}}
  }
}
```

## Statut

✅ S11-11 — Extension `feature_importance.py` (constantes CWRU/Pronostia + `permutation_importance_per_task()`)
✅ S11-12 — Extension `train_kmeans.py` (export `feature_importance.json` per-task)
✅ S11-13 — Extension `train_mahalanobis.py` (export `feature_importance.json` per-task)
✅ S11-14 — Extension `train_ewc.py` (permutation + gradient saliency per-task)
✅ S11-15 — Extension `train_hdc.py` (permutation + masking per-task)
✅ S11-16 — exp_100–105 lancées et `feature_importance.json` produits
✅ S11-17 — exp_106–111 lancées et `feature_importance.json` produits
