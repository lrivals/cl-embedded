# S2-01 — Implémenter `base_vectors.py` (génération + save HDC)

| Champ | Valeur |
|-------|--------|
| **ID** | S2-01 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 2h |
| **Dépendances** | S1-03 (monitoring_dataset.py — pour connaître N_FEATURES) |
| **Fichiers cibles** | `src/models/hdc/base_vectors.py`, `src/models/hdc/__init__.py` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Générer, valider et sauvegarder les hypervecteurs de base HDC :
- `H_level` : matrice `[N_LEVELS, D]` → associe chaque niveau de quantification à un hypervecteur binaire
- `H_pos` : matrice `[N_FEATURES, D]` → associe chaque position de feature à un hypervecteur binaire

Ces vecteurs sont le "modèle" HDC. Ils sont générés **une seule fois** avec `seed=42` et rechargés identiquement à chaque run et sur MCU. Toute modification du seed invalide le modèle entraîné.

**Critère de succès** : `python -c "from src.models.hdc.base_vectors import generate_base_hvectors, load_base_vectors"` passe, et `pytest tests/test_base_vectors.py -v` passe intégralement.

---

## Sous-tâches

### 1. Constantes du module

```python
# src/models/hdc/base_vectors.py

# Constantes globales — valeurs par défaut conformes à hdc_config.yaml
# Ne jamais modifier ici : passer par configs/hdc_config.yaml
D: int = 1024          # Dimension des hypervecteurs (puissance de 2 pour SIMD)
N_LEVELS: int = 10     # Niveaux de quantification par feature
N_FEATURES: int = 4    # Features numériques : temperature, pressure, vibration, humidity
                       # (équipement exclut de X — utilisé uniquement pour le split domaine)
                       # Référence : src/data/monitoring_dataset.py::NUMERIC_FEATURES
```

> ⚠️ `hdc_spec.md §2.1` indique 6 features (4 numériques + 2 one-hot équipement). Mais
> `monitoring_dataset.py` exclut l'encodage équipement des tenseurs X retournés par le
> DataLoader. N_FEATURES = 4 est la valeur correcte pour l'implémentation Python.
> `TODO(arnaud)` : confirmer si l'encodage one-hot doit être réintégré en feature pour HDC.

### 2. Implémenter `generate_base_hvectors`

```python
import numpy as np
from pathlib import Path


def generate_base_hvectors(
    D: int = D,
    n_levels: int = N_LEVELS,
    n_features: int = N_FEATURES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère les hypervecteurs de base pseudo-aléatoires pour HDC.

    Ces vecteurs sont FIXES (non appris). Ils doivent être sauvegardés
    et rechargés identiquement sur PC et sur MCU.

    Parameters
    ----------
    D : int
        Dimension des hypervecteurs. Doit être une puissance de 2 (SIMD).
    n_levels : int
        Nombre de niveaux de quantification par feature.
    n_features : int
        Nombre de features numériques en entrée.
    seed : int
        Graine aléatoire. CRITIQUE : tout changement invalide le modèle.

    Returns
    -------
    H_level : np.ndarray [n_levels, D], dtype=int8, valeurs ∈ {-1, +1}
        MEM: 10 × 1024 × 1 B = 10 Ko @ INT8 (Flash MCU)
    H_pos : np.ndarray [n_features, D], dtype=int8, valeurs ∈ {-1, +1}
        MEM: 4 × 1024 × 1 B = 4 Ko @ INT8 (Flash MCU)

    Notes
    -----
    Référence : Benatti2019HDC, docs/models/hdc_spec.md §3.1
    """
    rng = np.random.default_rng(seed)
    H_level = rng.choice([-1, 1], size=(n_levels, D)).astype(np.int8)
    H_pos = rng.choice([-1, 1], size=(n_features, D)).astype(np.int8)
    return H_level, H_pos
```

### 3. Implémenter `save_base_vectors` et `load_base_vectors`

```python
def save_base_vectors(
    H_level: np.ndarray,
    H_pos: np.ndarray,
    path: str | Path,
) -> None:
    """
    Sauvegarde les hypervecteurs de base au format .npz.

    Le fichier .npz contient les clés 'H_level' et 'H_pos'.
    La sauvegarde est idempotente : même seed → même fichier.

    Parameters
    ----------
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8
    path : str | Path
        Chemin de destination (ex. configs/hdc_base_vectors.npz).
        Le répertoire parent est créé si nécessaire.

    Notes
    -----
    MEM (fichier disque) :
        H_level : 10 × 1024 × 1 B = 10 240 B  # MEM: 10 Ko @ INT8
        H_pos   :  4 × 1024 × 1 B =  4 096 B  # MEM:  4 Ko @ INT8
        Total   : ~14 Ko (avec overhead .npz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, H_level=H_level, H_pos=H_pos)
    print(
        f"[HDC] Base vectors saved → {path} "
        f"(H_level={H_level.shape}, H_pos={H_pos.shape}, dtype={H_level.dtype})"
    )


def load_base_vectors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les hypervecteurs de base depuis un fichier .npz.

    Parameters
    ----------
    path : str | Path
        Chemin vers le fichier .npz (ex. configs/hdc_base_vectors.npz).

    Returns
    -------
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas. Lancer `generate_base_hvectors` + `save_base_vectors`.
    KeyError
        Si les clés 'H_level' ou 'H_pos' sont absentes du fichier.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base vectors not found at {path}. "
            "Run generate_base_hvectors() + save_base_vectors() first."
        )
    data = np.load(path)
    return data["H_level"], data["H_pos"]
```

### 4. Mettre à jour `src/models/hdc/__init__.py`

```python
# src/models/hdc/__init__.py
from .base_vectors import generate_base_hvectors, save_base_vectors, load_base_vectors

__all__ = ["generate_base_hvectors", "save_base_vectors", "load_base_vectors"]
```

### 5. Script de génération one-shot (exécution unique)

Ajouter en fin de fichier `base_vectors.py` pour permettre la génération en ligne de commande :

```python
if __name__ == "__main__":
    import yaml

    config_path = Path("configs/hdc_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hdc_cfg = config["hdc"]
    out_path = Path(hdc_cfg["base_vectors_path"])

    H_level, H_pos = generate_base_hvectors(
        D=hdc_cfg["D"],
        n_levels=hdc_cfg["n_levels"],
        n_features=config["data"]["n_features"],
        seed=hdc_cfg["seed"],
    )
    save_base_vectors(H_level, H_pos, out_path)
    print(f"H_level orthogonality (mean |dot|/D): {_mean_dot(H_level):.4f} (expected ~0)")
```

### 6. Écrire `tests/test_base_vectors.py`

```python
import numpy as np
import pytest
from pathlib import Path
from src.models.hdc.base_vectors import generate_base_hvectors, save_base_vectors, load_base_vectors

D, N_LEVELS, N_FEATURES = 1024, 10, 4

def test_shapes():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert H_level.shape == (N_LEVELS, D)
    assert H_pos.shape == (N_FEATURES, D)

def test_dtype():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert H_level.dtype == np.int8
    assert H_pos.dtype == np.int8

def test_binary_values():
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    assert set(np.unique(H_level)) == {-1, 1}
    assert set(np.unique(H_pos)) == {-1, 1}

def test_reproducibility():
    H1, P1 = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    H2, P2 = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    np.testing.assert_array_equal(H1, H2)
    np.testing.assert_array_equal(P1, P2)

def test_different_seeds_differ():
    H1, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    H2, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=99)
    assert not np.array_equal(H1, H2)

def test_approximate_orthogonality():
    """Les hypervecteurs de niveaux différents doivent être approximativement orthogonaux."""
    H_level, _ = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    # Dot product normalisé entre deux vecteurs aléatoires → N(0, 1/sqrt(D))
    dots = []
    for i in range(N_LEVELS):
        for j in range(i + 1, N_LEVELS):
            dot = np.dot(H_level[i].astype(np.float32), H_level[j].astype(np.float32)) / D
            dots.append(abs(dot))
    assert np.mean(dots) < 0.1, f"Mean |dot|/D = {np.mean(dots):.4f} > 0.1"

def test_save_load_roundtrip(tmp_path):
    H_level, H_pos = generate_base_hvectors(D, N_LEVELS, N_FEATURES, seed=42)
    path = tmp_path / "test_vectors.npz"
    save_base_vectors(H_level, H_pos, path)
    H_loaded, P_loaded = load_base_vectors(path)
    np.testing.assert_array_equal(H_level, H_loaded)
    np.testing.assert_array_equal(H_pos, P_loaded)

def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_base_vectors(Path("/tmp/nonexistent_vectors.npz"))
```

---

## Contraintes embarquées (STM32N6)

| Contrainte | Valeur | Vérification |
|------------|--------|--------------|
| `H_level` en Flash | 10 × 1024 × 1 B = 10 Ko | `H_level.nbytes == 10240` |
| `H_pos` en Flash | 4 × 1024 × 1 B = 4 Ko | `H_pos.nbytes == 4096` |
| dtype `int8` obligatoire | valeurs ∈ {-1, +1} | `test_dtype` + `test_binary_values` |
| seed 42 figé | non modifiable en source | constant dans le YAML uniquement |
| Export C : tableaux statiques | `const int8_t H_level[10][1024]` | hors scope Sprint 2 (Phase 2) |

---

## Critères d'acceptation

- [ ] `from src.models.hdc.base_vectors import generate_base_hvectors` — aucune erreur d'import
- [ ] `H_level.shape == (10, 1024)` et `H_pos.shape == (4, 1024)`
- [ ] `dtype == int8`, valeurs ∈ {-1, +1} exclusivement
- [ ] Reproductibilité : deux appels avec `seed=42` donnent des résultats identiques
- [ ] Orthogonalité approximative : `mean |dot(H_i, H_j)| / D < 0.1`
- [ ] `save_base_vectors` + `load_base_vectors` : round-trip sans perte
- [ ] `pytest tests/test_base_vectors.py -v` — tous les tests passent (8/8)
- [ ] `ruff check src/models/hdc/base_vectors.py` + `black --check` passent
- [ ] Fichier `.npz` généré dans `configs/hdc_base_vectors.npz` (path depuis hdc_config.yaml)

---

## Interface attendue par S2-02 (`hdc_classifier.py`)

```python
# Usage dans HDCClassifier.__init__()
from src.models.hdc.base_vectors import load_base_vectors

H_level, H_pos = load_base_vectors(config["hdc"]["base_vectors_path"])
# H_level : [N_LEVELS, D] int8
# H_pos   : [N_FEATURES, D] int8
```

---

## Questions ouvertes

- `TODO(arnaud)` : N_FEATURES = 4 (sans one-hot équipement) ou 6 (avec) ? La spec §2.1 dit 6 mais le loader retourne 4. Décision architecturale à fixer avant S2-02.
- `TODO(dorra)` : Export des tableaux statiques `H_level`/`H_pos` en C pour Flash STM32N6 — format `const int8_t` ou bitpack `uint8_t` (1 bit/dim) ?
- `FIXME(gap2)` : Mesurer l'empreinte Flash réelle après export C (estimée : ~14 Ko, budget Flash >> 14 Ko).
