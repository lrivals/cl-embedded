# Skill : Évaluation CL et Profiling Mémoire

> **Usage** : Demander à Claude d'implémenter l'évaluation, de calculer les métriques, ou d'interpréter des résultats.  
> **Déclencheur** : "calcule les métriques CL" / "interprète ces résultats" / "profile la mémoire de [modèle]"

---

## Métriques CL obligatoires

### Définitions formelles

Soit `acc(i, j)` = accuracy du modèle après entraînement sur la tâche `i`, évaluée sur la tâche `j`.

```python
# Matrice d'accuracy (T × T, triangulaire inférieure)
# acc_matrix[i][j] = acc(après tâche i, évalué sur tâche j)  pour j ≤ i
# acc_matrix[i][j] = NaN  pour j > i (pas encore vu)
```

| Métrique | Formule | Interprétation |
|---------|---------|----------------|
| **AA** (Average Accuracy) | `(1/T) Σᵢ acc(T, i)` | Précision globale après tout l'entraînement |
| **AF** (Average Forgetting) | `(1/(T-1)) Σⱼ₌₁ᵀ⁻¹ [max_{i≤T} acc(i,j) - acc(T,j)]` | Chute moyenne de précision sur les tâches passées |
| **BWT** (Backward Transfer) | `(1/(T-1)) Σⱼ₌₁ᵀ⁻¹ [acc(T,j) - acc(j,j)]` | Impact net de l'apprentissage futur sur le passé (négatif = oubli) |
| **FWT** (Forward Transfer) | `(1/(T-1)) Σⱼ₌₂ᵀ [acc(j-1,j) - acc_random(j)]` | Facilitation de l'apprentissage futur grâce au passé |

### Implémentation de référence

```python
# src/evaluation/metrics.py

import numpy as np
from typing import List, Optional


def compute_cl_metrics(acc_matrix: np.ndarray) -> dict:
    """
    Calcule les métriques CL standard depuis la matrice d'accuracy.
    
    Parameters
    ----------
    acc_matrix : np.ndarray [T, T]
        acc_matrix[i, j] = accuracy après tâche i sur tâche j.
        Valeurs NaN pour j > i (tâches non encore vues).
    
    Returns
    -------
    dict avec clés : aa, af, bwt, fwt
    
    References
    ----------
    Lopez-Paz & Ranzato (2017). Gradient Episodic Memory for CL.
    De Lange et al. (2021). A CL Survey.
    """
    T = acc_matrix.shape[0]
    
    # Average Accuracy : dernière ligne de la matrice
    aa = np.nanmean(acc_matrix[T-1, :T])
    
    # Average Forgetting
    max_acc = np.nanmax(acc_matrix[:, :T], axis=0)   # [T] max atteint par tâche
    final_acc = acc_matrix[T-1, :T]                   # [T] acc finale par tâche
    af = np.mean(max_acc[:T-1] - final_acc[:T-1])     # exclure dernière tâche
    
    # Backward Transfer
    diag_acc = np.diag(acc_matrix)[:T-1]               # acc juste après entraînement
    bwt = np.mean(final_acc[:T-1] - diag_acc)
    
    return {
        "aa": float(aa),
        "af": float(af),
        "bwt": float(bwt),
        "acc_matrix": acc_matrix.tolist(),
    }
```

---

## Profiling mémoire

### Méthode sur PC (proxy pour MCU)

```python
# src/evaluation/memory_profiler.py

import tracemalloc
import torch
from typing import Callable


def profile_forward_pass(model, input_tensor: torch.Tensor) -> dict:
    """
    Profile la mémoire lors d'un forward pass.
    
    Returns
    -------
    dict :
        ram_peak_bytes : pic mémoire Python (tracemalloc)
        n_params : nombre de paramètres
        params_fp32_bytes : taille des poids en FP32
        params_int8_bytes : taille estimée en INT8
        activation_bytes : activations intermédiaires estimées
    """
    tracemalloc.start()
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    n_params = sum(p.numel() for p in model.parameters())
    
    return {
        "ram_peak_bytes": peak,
        "ram_current_bytes": current,
        "n_params": n_params,
        "params_fp32_bytes": n_params * 4,
        "params_int8_bytes": n_params * 1,
    }


def profile_cl_update(model, update_fn: Callable, input_tensor, label) -> dict:
    """Profile la mémoire lors d'une mise à jour CL (forward + backward + step)."""
    tracemalloc.start()
    update_fn(input_tensor, label)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"ram_peak_bytes_update": peak}
```

### Interprétation des résultats

| Seuil | Signification |
|-------|--------------|
| `ram_peak_bytes` < 65 536 | ✅ Compatible STM32N6 (Gap 2 fermé) |
| `ram_peak_bytes` < 100 000 | ⚠️ Compatible MCU plus grand (STM32H7) |
| `ram_peak_bytes` > 100 000 | ❌ Nécessite optimisation avant portage |

> ⚠️ Note : `tracemalloc` mesure l'allocateur Python, pas la RAM C réelle. Les chiffres MCU réels seront mesurés en Phase 2 (portage). Les chiffres PC sont des **proxies** à reporter comme "estimations PC" dans le manuscrit.

---

## Visualisations obligatoires pour chaque expérience

### 1. Accuracy Matrix (heatmap)
```python
# Heatmap triangulaire inférieure de acc_matrix
# X-axis : tâche évaluée ; Y-axis : après entraînement tâche i
# Diagonale = acc juste après entraînement de chaque tâche
```

### 2. Forgetting Curve
```python
# Pour chaque tâche j < T : courbe de acc(i, j) pour i = j..T
# Montre comment la précision sur la tâche j évolue au fil du temps
```

### 3. RAM Timeline
```python
# Profil mémoire au fil du stream CL
# X-axis : numéro de sample ; Y-axis : RAM courante
# Marqueurs : transitions entre tâches
```

---

## Format de sortie des expériences

Chaque expérience doit produire un fichier `results/metrics.json` :

```json
{
  "exp_id": "exp_001_ewc_dataset2",
  "model": "ewc_mlp",
  "dataset": "equipment_monitoring",
  "date": "2026-04-20T14:30:00",
  "config": "configs/ewc_config.yaml",
  "cl_metrics": {
    "aa": 0.87,
    "af": 0.05,
    "bwt": -0.04,
    "acc_matrix": [[0.91, null, null], [0.88, 0.85, null], [0.86, 0.83, 0.89]]
  },
  "memory": {
    "n_params": 769,
    "params_fp32_bytes": 3076,
    "ram_peak_bytes_forward": 9200,
    "ram_peak_bytes_update": 11400
  },
  "baselines": {
    "fine_tuning_naive": {"aa": 0.71, "af": 0.21},
    "joint_training": {"aa": 0.92, "af": 0.00}
  }
}
```

---

## Interprétation type des résultats (script de rapport)

Quand Claude interprète des résultats, répondre en ce format :

```
Résultats exp_XXX — [Modèle] sur [Dataset]

PRÉCISION :
  AA = X.XX  (baseline FT naïf : Y.YY | upper bound joint : Z.ZZ)
  → [Excellent / Acceptable / Insuffisant] — [commentaire]

OUBLI :
  AF = X.XX  (0 = aucun oubli | 1 = oubli total)
  → [commentaire sur la gestion de l'oubli]

MÉMOIRE :
  RAM peak = X Ko (cible : 64 Ko)
  → [Compatible MCU / Limite / Incompatible]

CONCLUSION TRIPLE GAP :
  Gap 1 [données] : [contribue / ne contribue pas] parce que [...]
  Gap 2 [mémoire] : [fermé / partiellement / non] — X Ko < 64 Ko ✅/❌
  Gap 3 [INT8]    : [contribue / ne contribue pas] parce que [...]
```
