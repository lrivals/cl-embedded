# S10-01 — Loader `pronostia_dataset.py` (FEMTO PRONOSTIA)

| Champ | Valeur |
|-------|--------|
| **ID** | S10-01 |
| **Sprint** | Sprint 10 — Phase 1 Extension |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 4h |
| **Dépendances** | — (premier fichier du sprint) |
| **Fichiers cibles** | `src/data/pronostia_dataset.py`, `tests/test_pronostia_dataset.py` |

---

## Objectif

Implémenter le loader `PronostiaDataset` pour le dataset IEEE PHM 2012 FEMTO PRONOSTIA. Ce module est le prérequis à toutes les autres tâches du sprint : configs, EDA, expériences.

Le loader doit :
1. Lire les fichiers `.npy` pré-convertis depuis `data/raw/Pronostia dataset/binaries/`
2. Appliquer le fenêtrage (WINDOW_SIZE=2560, sans overlap)
3. Calculer 13 features statistiques par fenêtre
4. Générer les labels TTF binaires (derniers 10% = pré-défaillance)
5. Exposer un itérateur séquentiel par condition opératoire (pour le scénario CL)

---

## Structure des fichiers source

```
data/raw/Pronostia dataset/binaries/
├── Bearing1_1.npy    # shape (N_epochs, 2, 2560) — Condition 1, roulement 1
├── Bearing1_2.npy    # shape (N_epochs, 2, 2560) — Condition 1, roulement 2
├── Bearing2_1.npy    # shape (N_epochs, 2, 2560) — Condition 2, roulement 1
├── Bearing2_2.npy    # shape (N_epochs, 2, 2560) — Condition 2, roulement 2
├── Bearing3_1.npy    # shape (N_epochs, 2, 2560) — Condition 3, roulement 1
└── Bearing3_2.npy    # shape (N_epochs, 2, 2560) — Condition 3, roulement 2
```

Axe 1 du shape `(N, 2, 2560)` : `[0]` = accélération horizontale, `[1]` = accélération verticale.

---

## Sous-tâches

### 1. Feature extraction (6 stats × 2 canaux)

```python
import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    window : shape (2, 2560) — (n_channels, window_size)
    returns : shape (13,) — 6 stats × 2 canaux + temporal_position
    """
    features = []
    for ch in range(2):  # horizontal, vertical
        x = window[ch]
        features += [
            float(np.mean(x)),          # MEM: 4 B @ FP32
            float(np.std(x)),           # MEM: 4 B @ FP32
            float(np.sqrt(np.mean(x**2))),  # RMS  # MEM: 4 B @ FP32
            float(kurtosis(x)),         # MEM: 4 B @ FP32
            float(skew(x)),             # MEM: 4 B @ FP32
            float(np.max(np.abs(x))),   # peak # MEM: 4 B @ FP32
        ]
    # temporal_position ajouté par PronostiaDataset.__getitem__
    return np.array(features, dtype=np.float32)
```

### 2. Label TTF binaire

```python
def make_ttf_labels(n_epochs: int, failure_ratio: float = 0.10) -> np.ndarray:
    """
    Derniers failure_ratio * n_epochs epochs = pré-défaillance (label=1).
    Cohérent avec le protocole PHM 2012 : la dégradation s'accélère en fin de vie.
    """
    labels = np.zeros(n_epochs, dtype=np.int64)
    failure_start = int(n_epochs * (1.0 - failure_ratio))
    labels[failure_start:] = 1
    return labels
```

### 3. Classe principale `PronostiaDataset`

```python
class PronostiaDataset:
    """Dataset FEMTO PRONOSTIA pour un roulement ou une condition entière.

    Parameters
    ----------
    npy_dir : str
        Chemin vers le répertoire contenant les fichiers .npy pré-convertis.
    bearing_ids : list[str]
        Liste des identifiants de roulements à charger, ex. ['Bearing1_1', 'Bearing1_2'].
    failure_ratio : float
        Fraction de la fin du signal labelisée comme pré-défaillance. Défaut : 0.10.
    normalize : bool
        Appliquer la normalisation (mean=0, std=1) calculée sur ce sous-ensemble.
    """

    def __init__(
        self,
        npy_dir: str,
        bearing_ids: list[str],
        failure_ratio: float = 0.10,
        normalize: bool = True,
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Retourne (features_13d, label) pour l'epoch idx."""
        # MEM features: 13 × 4 B = 52 B @ FP32
        ...
```

### 4. Classe `PronostiaConditionStream`

```python
class PronostiaConditionStream:
    """Itérateur CL domain-incremental : yield les conditions une à une.

    Utilisé par les scripts d'entraînement CL pour simuler l'arrivée séquentielle
    de nouvelles conditions opératoires (3 tâches).

    Parameters
    ----------
    npy_dir : str
    condition_map : dict[int, list[str]]
        Mapping condition_id → liste de bearing_ids.
        Ex. : {1: ['Bearing1_1', 'Bearing1_2'], 2: [...], 3: [...]}
    failure_ratio : float
    """

    def __iter__(self) -> Iterator[tuple[int, PronostiaDataset]]:
        """Yield (condition_id, dataset) pour chaque condition."""
        ...
```

### 5. Normalisation

- Calculer mean/std sur le jeu d'entraînement de chaque condition séparément
- Sauvegarder les paramètres dans `configs/pronostia_normalizer.yaml` pour reproductibilité
- **Ne pas** normaliser au niveau du roulement individuel (introduirait un data leakage inter-conditions)

---

## Contrainte embarquée

```python
# Budget RAM estimé pour PronostiaDataset en mode streaming (1 fenêtre en mémoire)
# MEM: features (13 × 4 B) = 52 B @ FP32 / 13 B @ INT8
# MEM: window_buffer (2 × 2560 × 4 B) = 20 480 B @ FP32 — temporaire pendant extract_features
# MEM: total streaming = 52 B actif + 20 480 B buffer → libérer window après extraction
```

---

## Critères d'acceptation

- [x] `PronostiaDataset(['Bearing1_1', 'Bearing1_2'])` charge sans erreur sur les données réelles
- [x] `len(dataset)` retourne le nombre total d'epochs des roulements fournis
- [x] `dataset[0]` retourne un tuple `(np.ndarray shape (13,), int)` avec label ∈ {0, 1}
- [x] La proportion de labels=1 est ≈ 10% pour chaque roulement individuellement
- [x] `PronostiaConditionStream` yield exactement 3 tuples `(condition_id, PronostiaDataset)`
- [x] `pytest tests/test_pronostia_dataset.py -v` → 9 tests implémentés sur fixtures synthétiques
- [x] Annotations `# MEM:` présentes sur toutes les structures de données persistantes

---

## Tests à implémenter (`tests/test_pronostia_dataset.py`)

```python
def test_dataset_shape()           # features.shape == (13,)
def test_label_ratio()             # proportion labels=1 ≈ 0.10 ± 0.02
def test_temporal_position()       # temporal_position ∈ [0, 1]
def test_condition_stream_len()    # 3 conditions yielded
def test_normalize_no_leakage()    # mean/std calculés sans voir les données de test
def test_npy_missing_raises()      # FileNotFoundError si npy_dir absent
```

---

## Questions ouvertes

- `TODO(arnaud)` : Faut-il utiliser le fichier `Bearing_*_RUL.txt` (fourni avec PRONOSTIA) pour calibrer le seuil de pré-défaillance roulement par roulement, ou conserver le seuil global failure_ratio=0.10 pour la simplicité d'implémentation embarquée ?
- `TODO(dorra)` : La normalisation par condition (calculée sur Bearing1_1 + Bearing1_2 ensemble) est-elle cohérente avec le scénario online où les paramètres sont mis à jour en ligne ? Ou faut-il un normalizer incrémental (running mean/std) ?

---

**Complété le** : 2026-04-23
