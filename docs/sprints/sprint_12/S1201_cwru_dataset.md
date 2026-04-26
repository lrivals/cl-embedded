# S12-01 — Loader `cwru_dataset.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S12-01 |
| **Sprint** | Sprint 12 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 5h |
| **Dépendances** | — |
| **Fichier cible** | `src/data/cwru_dataset.py` |

---

## Objectif

Implémenter le loader du **CWRU Bearing Dataset** exposant trois classes : `CWRUDataset` (chargement brut CSV/MAT), `CWRUFaultTypeStream` (stream domain-incremental par type de défaut) et `CWRUSeverityStream` (stream domain-incremental par sévérité). Ce module est le prérequis bloquant pour toutes les tâches CWRU du sprint 12.

---

## Features extraites (9)

| Feature | Formule | Unité |
|---------|---------|-------|
| `max` | `max(x)` | g |
| `min` | `min(x)` | g |
| `mean` | `mean(x)` | g |
| `sd` | `std(x)` | g |
| `rms` | `sqrt(mean(x²))` | g |
| `skewness` | moment normalisé d'ordre 3 | — |
| `kurtosis` | moment normalisé d'ordre 4 | — |
| `crest` | `max(|x|) / rms(x)` | — |
| `form` | `rms(x) / mean(|x|)` | — |

Ces features sont pré-calculées dans `feature_time_48k_2048_load_1.csv` (2 299 fenêtres, fenêtre = 2 048 points @ 48 kHz).

---

## Interface

```python
from src.data.cwru_dataset import CWRUDataset, CWRUFaultTypeStream, CWRUSeverityStream

# Chargement depuis CSV pré-traité
dataset = CWRUDataset(
    csv_path: str | Path,          # feature_time_48k_2048_load_1.csv
    mat_dir: str | Path | None,    # data/raw/CWRU Bearing Dataset/ (optionnel)
    random_state: int = 42,
) -> CWRUDataset

# Accès aux données complètes
X: np.ndarray  # shape (2299, 9), dtype float32
y: np.ndarray  # shape (2299,),  dtype int8, valeurs {0, 1}

# Stream by_fault_type — 3 tâches : Ball → Inner Race → Outer Race
stream = CWRUFaultTypeStream(dataset)
for task_id, task_name, X_task, y_task in stream.iter_tasks():
    ...  # task_id ∈ {0, 1, 2}, task_name ∈ {"ball", "inner_race", "outer_race"}

# Stream by_severity — 3 tâches : 0.007" → 0.014" → 0.021"
stream = CWRUSeverityStream(dataset)
for task_id, task_name, X_task, y_task in stream.iter_tasks():
    ...  # task_id ∈ {0, 1, 2}, task_name ∈ {"007", "014", "021"}
```

---

## Mapping fichiers MAT → tâches

### by_fault_type

| Tâche | Nom | Fichiers MAT | Fenêtres approx. |
|-------|-----|-------------|-----------------|
| 0 | `ball` | B007 + B014 + B021 + Normal (⅓) | ~920 |
| 1 | `inner_race` | IR007 + IR014 + IR021 + Normal (⅓) | ~920 |
| 2 | `outer_race` | OR007 + OR014 + OR021 + Normal (⅓) | ~460 |

### by_severity

| Tâche | Nom | Fichiers MAT | Fenêtres approx. |
|-------|-----|-------------|-----------------|
| 0 | `007` | B007 + IR007 + OR007 + Normal (⅓) | ~920 |
| 1 | `014` | B014 + IR014 + OR014 + Normal (⅓) | ~920 |
| 2 | `021` | B021 + IR021 + OR021 + Normal (⅓) | ~460 |

> Le sous-ensemble Normal est réparti équitablement entre les 3 tâches pour maintenir l'équilibre de classe.

---

## Critères d'acceptation

- [x] `CWRUDataset`, `CWRUFaultTypeStream`, `CWRUSeverityStream` importables depuis `src.data.cwru_dataset`
- [x] `dataset.X.shape == (2300, 9)` et `dataset.y.shape == (2300,)` avec valeurs `{0, 1}` (la spec mentionnait 2299 — valeur réelle : 2300, 10 classes × 230 fenêtres)
- [x] `CWRUFaultTypeStream.iter_tasks()` produit exactement 3 tâches dans l'ordre Ball → IR → OR
- [x] `CWRUSeverityStream.iter_tasks()` produit exactement 3 tâches dans l'ordre 0.007" → 0.014" → 0.021"
- [x] Le loader CSV fonctionne sans `scipy.io` (le loader MAT est optionnel)
- [x] Aucune dépendance non portable MCU dans les imports du module

## Statut

✅ Terminé — 24 avril 2026
