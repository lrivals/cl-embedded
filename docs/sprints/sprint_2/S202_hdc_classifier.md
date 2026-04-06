# S2-02 — Implémenter `hdc_classifier.py` (encodage + prototypes + inférence)

| Champ | Valeur |
|-------|--------|
| **ID** | S2-02 |
| **Sprint** | Sprint 2 — Semaine 2 (22–29 avril 2026) |
| **Priorité** | ✅ Terminé |
| **Durée estimée** | 4h |
| **Dépendances** | S2-01 (base_vectors.py), S1-03 (monitoring_dataset.py), S1-07 (metrics.py) |
| **Fichiers cibles** | `src/models/hdc/hdc_classifier.py`, `src/models/hdc/__init__.py` |
| **Complété le** | 6 avril 2026 |

---

## Objectif

Implémenter la classe `HDCClassifier` héritant de `BaseCLModel` (`src/models/base_cl_model.py`). Cette classe encapsule :
1. L'encodage des observations en hypervecteurs binaires (via `H_level` et `H_pos`)
2. L'accumulation des prototypes de classe (apprentissage sans gradient)
3. L'inférence par similarité cosinus

**Avantage CL fondamental** : pas d'oubli catastrophique par construction — les prototypes sont une mémoire additive (`prototypes_acc[y] += H_obs`).

**Critère de succès** : `pytest tests/test_hdc_classifier.py -v` passe intégralement, et `model.check_ram_budget()["within_budget"] == True` (< 12 Ko RAM).

---

## Sous-tâches

### 1. Constantes du module

```python
# src/models/hdc/hdc_classifier.py

# Constantes — valeurs par défaut conformes à hdc_config.yaml
# Ne jamais modifier ici : passer par configs/hdc_config.yaml
D: int = 1024          # Dimension des hypervecteurs
N_LEVELS: int = 10     # Niveaux de quantification par feature
N_FEATURES: int = 4    # Features : temperature, pressure, vibration, humidity
N_CLASSES: int = 2     # Classes : 0=normal, 1=faulty
```

### 2. Fonctions d'encodage (stateless, réutilisables sur MCU)

```python
import numpy as np


def quantize_feature(
    value: float,
    feature_min: float,
    feature_max: float,
    n_levels: int,
) -> int:
    """
    Mappe une feature continue dans [feature_min, feature_max] vers un indice ∈ [0, n_levels-1].

    Quantification linéaire uniforme. Clip aux bornes pour les valeurs hors plage.

    Parameters
    ----------
    value : float
        Valeur de la feature (normalisée Z-score).
    feature_min, feature_max : float
        Bornes observées sur Task 1 (chargées depuis hdc_config.yaml → feature_bounds).
    n_levels : int
        Nombre de niveaux de quantification.

    Returns
    -------
    int
        Indice de niveau ∈ [0, n_levels - 1].

    Notes
    -----
    Sur MCU : calcul en FP32 une fois par feature. Résultat stocké en uint8.
    Référence : docs/models/hdc_spec.md §3.2
    """
    normalized = (value - feature_min) / (feature_max - feature_min + 1e-8)
    level_idx = int(normalized * (n_levels - 1))
    return int(np.clip(level_idx, 0, n_levels - 1))


def encode_observation(
    x: np.ndarray,
    H_level: np.ndarray,
    H_pos: np.ndarray,
    feature_bounds: list[tuple[float, float]],
    n_levels: int = N_LEVELS,
    D: int = D,
) -> np.ndarray:
    """
    Encode un vecteur de features en hypervecteur d'observation binarisé.

    Pipeline :
    1. Pour chaque feature i : quantifier → indice de niveau l_i
    2. H_feature_i = H_level[l_i] ⊗ H_pos[i]  (produit Hadamard, XOR sur MCU)
    3. H_sum = Σ H_feature_i  (sommation entière)
    4. H_obs_bin = sign(H_sum)  (binarisation)

    Parameters
    ----------
    x : np.ndarray [n_features]
        Vecteur de features normalisé (Z-score, float32).
    H_level : np.ndarray [n_levels, D], dtype=int8
    H_pos : np.ndarray [n_features, D], dtype=int8
    feature_bounds : list[tuple[float, float]]
        [(min_0, max_0), ..., (min_{n-1}, max_{n-1})] — calculées sur Task 1.
    n_levels : int
    D : int

    Returns
    -------
    np.ndarray [D], dtype=int8, valeurs ∈ {-1, +1}
        MEM: D × 4 B = 4 Ko @ INT32 (buffer temporaire H_sum)
             D × 1 B = 1 Ko @ INT8  (H_obs_bin, sortie)

    Notes
    -----
    Référence : docs/models/hdc_spec.md §3.3
    """
    H_sum = np.zeros(D, dtype=np.int32)  # MEM: 1024 × 4 B = 4 Ko @ INT32

    for i, (feat_val, (f_min, f_max)) in enumerate(zip(x, feature_bounds)):
        level_idx = quantize_feature(feat_val, f_min, f_max, n_levels)
        H_feature = H_level[level_idx] * H_pos[i]  # Hadamard (XOR sur MCU)
        H_sum += H_feature.astype(np.int32)

    H_obs_bin = np.sign(H_sum).astype(np.int8)
    H_obs_bin[H_obs_bin == 0] = 1  # cas dégénéré (parité exacte)
    return H_obs_bin
```

### 3. Implémenter `HDCClassifier`

```python
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models.base_cl_model import BaseCLModel
from src.models.hdc.base_vectors import load_base_vectors, generate_base_hvectors, save_base_vectors


class HDCClassifier(BaseCLModel):
    """
    Classifieur HDC (Hyperdimensional Computing) pour la maintenance prédictive.

    Apprentissage incrémental sans gradient : accumulation additive de prototypes.
    Pas d'oubli catastrophique par construction (mémoire additive).

    Architecture :
        - Encodage : quantize → Hadamard → sommation → binarisation
        - Prototypes : C_c = Σ H_obs pour chaque obs de classe c (INT32 accumulateurs)
        - Inférence : ŷ = argmax_c cosine_similarity(H_obs, C_c)

    Budget mémoire :
        - prototypes_acc [2, 1024] INT32 : 2 × 1024 × 4 B = 8 Ko   # MEM: 8 Ko @ INT32
        - prototypes_bin [2, 1024] INT8  : 2 × 1024 × 1 B = 2 Ko   # MEM: 2 Ko @ INT8
        - buffer encodage [1024]   INT32 : 1 × 1024 × 4 B = 4 Ko   # MEM: 4 Ko @ INT32 (temporaire)
        - TOTAL RAM : < 12 Ko (cible STM32N6 : ≤ 64 Ko)

    Parameters
    ----------
    config : dict
        Configuration chargée depuis configs/hdc_config.yaml.

    References
    ----------
    Benatti2019HDC, docs/models/hdc_spec.md
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        hdc_cfg = config["hdc"]
        self.D: int = hdc_cfg["D"]                   # 1024
        self.n_levels: int = hdc_cfg["n_levels"]     # 10
        self.n_classes: int = config["data"]["n_classes"]  # 2
        self.n_features: int = config["data"]["n_features"]  # 4

        # Feature bounds pour quantification — calculées sur Task 1
        # Format : [(min_0, max_0), ..., (min_3, max_3)]
        self.feature_bounds: list[tuple[float, float]] = self._load_feature_bounds(config)

        # Hypervecteurs de base (depuis .npz ou génération on-the-fly)
        bv_path = Path(hdc_cfg["base_vectors_path"])
        if bv_path.exists():
            self.H_level, self.H_pos = load_base_vectors(bv_path)
        else:
            self.H_level, self.H_pos = generate_base_hvectors(
                D=self.D, n_levels=self.n_levels,
                n_features=self.n_features, seed=hdc_cfg["seed"]
            )
            save_base_vectors(self.H_level, self.H_pos, bv_path)

        # Prototypes de classe (accumulateurs INT32 + version binarisée pour l'inférence)
        self.prototypes_acc = np.zeros((self.n_classes, self.D), dtype=np.int32)
        # MEM: 2 × 1024 × 4 B = 8 Ko @ INT32
        self.prototypes_bin = np.zeros((self.n_classes, self.D), dtype=np.int8)
        # MEM: 2 × 1024 × 1 B = 2 Ko @ INT8
        self.class_counts = np.zeros(self.n_classes, dtype=np.int32)

        self._fitted: bool = False  # True après au moins 1 appel à update()

    # ------------------------------------------------------------------
    # Interface BaseCLModel — méthodes abstraites
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Inférence par similarité cosinus entre H_obs et les prototypes binarisés.

        Sur MCU : calcul via POPCOUNT(XOR(H_obs, C_c)) pour chaque classe c.

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features], dtype=float32
            Observations normalisées.

        Returns
        -------
        np.ndarray [batch_size], dtype=int64
            Classe prédite ∈ {0, 1} pour chaque observation.
        """
        if not self._fitted:
            raise RuntimeError("HDCClassifier not fitted. Call update() first.")

        preds = []
        for sample in x:
            H_obs = encode_observation(
                sample, self.H_level, self.H_pos, self.feature_bounds,
                self.n_levels, self.D
            )
            # Similarité cosinus avec chaque prototype binarisé
            # Sur MCU : dot product INT8 (équivalent à count_agreements - count_disagreements)
            similarities = self.prototypes_bin.astype(np.float32) @ H_obs.astype(np.float32)
            preds.append(int(np.argmax(similarities)))
        return np.array(preds, dtype=np.int64)

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Mise à jour incrémentale : accumulation des hypervecteurs dans les prototypes.

        Complexité : O(batch_size × N_FEATURES × D) en temps, O(1) en mémoire additionnelle.
        Pas de gradient, pas de catastrophic forgetting par construction.

        Parameters
        ----------
        x : np.ndarray [batch_size, n_features], dtype=float32
        y : np.ndarray [batch_size], dtype=int

        Returns
        -------
        float
            Taux d'erreur sur ce batch (proxy de loss pour compatibilité BaseCLModel).
        """
        errors = 0
        for sample, label in zip(x, y):
            H_obs = encode_observation(
                sample, self.H_level, self.H_pos, self.feature_bounds,
                self.n_levels, self.D
            )
            self.prototypes_acc[int(label)] += H_obs.astype(np.int32)
            self.class_counts[int(label)] += 1

        # Re-binariser les prototypes après accumulation
        self._rebinarize_prototypes()
        self._fitted = True

        # Évaluer sur le batch courant (proxy d'erreur)
        if self._fitted:
            preds = self.predict(x)
            errors = int(np.sum(preds != y.astype(np.int64)))
        return errors / len(y)

    def on_task_end(self, task_id: int, dataloader: Any) -> None:
        """
        Callback fin de tâche. HDC n'a pas de post-processing obligatoire.

        Re-binarise les prototypes pour s'assurer de la cohérence après la tâche.
        Optionnel : renormalisation (hors scope Sprint 2).

        Parameters
        ----------
        task_id : int
        dataloader : iterable (non utilisé par HDC)
        """
        self._rebinarize_prototypes()

    def count_parameters(self) -> int:
        """
        Retourne le nombre d'éléments dans les prototypes accumulateurs.

        HDC n'a pas de paramètres au sens neuronal. On compte les éléments
        des prototypes INT32 (état entraînable) pour comparaison inter-modèles.

        Returns
        -------
        int
            n_classes × D = 2 × 1024 = 2048 éléments.
        """
        return self.n_classes * self.D  # MEM: 2 × 1024 = 2048 éléments

    def estimate_ram_bytes(self, dtype: str = "fp32") -> int:
        """
        Estime l'empreinte RAM du modèle HDC.

        Parameters
        ----------
        dtype : str
            "fp32" → utilise les prototypes INT32 (4 B/élément).
            "int8" → utilise les prototypes binarisés INT8 (1 B/élément).

        Returns
        -------
        int
            Estimation en octets.

        Notes
        -----
        Budget détaillé (docs/models/hdc_spec.md §2.3) :
            prototypes_acc [2, 1024] INT32 : 8 192 B  # MEM: 8 Ko @ INT32
            prototypes_bin [2, 1024] INT8  : 2 048 B  # MEM: 2 Ko @ INT8
            buffer encodage [1024]   INT32 : 4 096 B  # MEM: 4 Ko @ INT32 (temporaire)
            class_counts    [2]      INT32 :     8 B
            TOTAL (worst case)             : ~14 Ko @ FP32, ~6 Ko @ INT8
        """
        if dtype == "int8":
            return (
                self.n_classes * self.D * 1    # prototypes_bin INT8
                + 4 * self.D                   # buffer encodage INT32 (temporaire)
                + self.n_classes * 4            # class_counts INT32
            )
        else:  # fp32 / worst case
            return (
                self.n_classes * self.D * 4    # prototypes_acc INT32
                + self.n_classes * self.D * 1  # prototypes_bin INT8
                + 4 * self.D                   # buffer encodage INT32 (temporaire)
                + self.n_classes * 4            # class_counts INT32
            )

    def save(self, path: str) -> None:
        """Sauvegarde l'état complet (prototypes + counts) en .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            prototypes_acc=self.prototypes_acc,
            prototypes_bin=self.prototypes_bin,
            class_counts=self.class_counts,
        )

    def load(self, path: str) -> None:
        """Charge un état sauvegardé depuis un .npz."""
        data = np.load(path)
        self.prototypes_acc = data["prototypes_acc"]
        self.prototypes_bin = data["prototypes_bin"]
        self.class_counts = data["class_counts"]
        self._fitted = True

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _rebinarize_prototypes(self) -> None:
        """Re-binarise les prototypes INT32 → INT8 après accumulation."""
        proto_bin = np.sign(self.prototypes_acc).astype(np.int8)
        proto_bin[proto_bin == 0] = 1
        self.prototypes_bin = proto_bin

    @staticmethod
    def _load_feature_bounds(config: dict) -> list[tuple[float, float]]:
        """
        Charge les bornes de features depuis hdc_config.yaml → feature_bounds.

        Returns
        -------
        list[tuple[float, float]]
            [(min_0, max_0), ..., (min_3, max_3)]

        Raises
        ------
        ValueError
            Si les bornes contiennent des None (non encore calculées sur Task 1).
        """
        bounds_cfg = config.get("feature_bounds", {})
        bounds = []
        for feat_name, (f_min, f_max) in bounds_cfg.items():
            if f_min is None or f_max is None:
                raise ValueError(
                    f"Feature bound '{feat_name}' contains None. "
                    "Run S2-03 (train_hdc.py Task 1 fit) to compute bounds from data."
                )
            bounds.append((float(f_min), float(f_max)))
        if not bounds:
            raise ValueError("feature_bounds is empty in hdc_config.yaml.")
        return bounds
```

### 4. Mettre à jour `src/models/hdc/__init__.py`

```python
# src/models/hdc/__init__.py
from .base_vectors import generate_base_hvectors, save_base_vectors, load_base_vectors
from .hdc_classifier import HDCClassifier, encode_observation, quantize_feature

__all__ = [
    "HDCClassifier",
    "encode_observation",
    "quantize_feature",
    "generate_base_hvectors",
    "save_base_vectors",
    "load_base_vectors",
]
```

### 5. Écrire `tests/test_hdc_classifier.py`

```python
import numpy as np
import pytest
from src.models.hdc.hdc_classifier import HDCClassifier, quantize_feature, encode_observation

D, N_LEVELS, N_FEATURES, N_CLASSES = 1024, 10, 4, 2

MOCK_CONFIG = {
    "hdc": {"D": D, "n_levels": N_LEVELS, "seed": 42, "base_vectors_path": "/tmp/test_hdc_bv.npz"},
    "data": {"n_features": N_FEATURES, "n_classes": N_CLASSES},
    "feature_bounds": {
        "temperature": (20.0, 80.0),
        "pressure": (1.0, 10.0),
        "vibration": (0.0, 5.0),
        "humidity": (10.0, 90.0),
    },
    "memory": {"target_ram_bytes": 65536, "warn_if_above_bytes": 52000},
}

def make_model() -> HDCClassifier:
    return HDCClassifier(MOCK_CONFIG)

def test_quantize_feature_clipping():
    assert quantize_feature(0.0, 0.0, 1.0, 10) == 0
    assert quantize_feature(1.0, 0.0, 1.0, 10) == 9
    assert quantize_feature(-99.0, 0.0, 1.0, 10) == 0
    assert quantize_feature(99.0, 0.0, 1.0, 10) == 9

def test_encode_observation_shape_dtype():
    model = make_model()
    x = np.array([50.0, 5.0, 2.5, 50.0], dtype=np.float32)
    bounds = list(MOCK_CONFIG["feature_bounds"].values())
    H_obs = encode_observation(x, model.H_level, model.H_pos, bounds, N_LEVELS, D)
    assert H_obs.shape == (D,)
    assert H_obs.dtype == np.int8
    assert set(np.unique(H_obs)).issubset({-1, 1})

def test_update_increments_prototypes():
    model = make_model()
    x = np.random.randn(10, N_FEATURES).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    model.update(x, y)
    assert model.class_counts[0] == 5
    assert model.class_counts[1] == 5

def test_predict_after_update():
    model = make_model()
    x_train = np.random.randn(50, N_FEATURES).astype(np.float32)
    y_train = np.array([i % 2 for i in range(50)], dtype=np.int64)
    model.update(x_train, y_train)
    preds = model.predict(x_train[:5])
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_predict_before_fit_raises():
    model = make_model()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((1, N_FEATURES), dtype=np.float32))

def test_ram_budget_within_64ko():
    model = make_model()
    budget = model.check_ram_budget()
    assert budget["within_budget"] is True
    assert budget["estimated_bytes"] < 65536

def test_ram_estimate_fp32():
    model = make_model()
    # prototypes_acc: 2*1024*4 + prototypes_bin: 2*1024*1 + buffer: 4*1024 + counts: 2*4
    expected = 2 * 1024 * 4 + 2 * 1024 * 1 + 4 * 1024 + 2 * 4
    assert model.estimate_ram_bytes("fp32") == expected

def test_count_parameters():
    model = make_model()
    assert model.count_parameters() == N_CLASSES * D  # 2048

def test_save_load_roundtrip(tmp_path):
    model = make_model()
    x = np.random.randn(20, N_FEATURES).astype(np.float32)
    y = np.array([i % 2 for i in range(20)], dtype=np.int64)
    model.update(x, y)
    path = str(tmp_path / "hdc_state.npz")
    model.save(path)
    model2 = make_model()
    model2.load(path)
    np.testing.assert_array_equal(model.prototypes_acc, model2.prototypes_acc)
    np.testing.assert_array_equal(model.prototypes_bin, model2.prototypes_bin)

def test_on_task_end_rebinarizes():
    model = make_model()
    model.prototypes_acc[0, :5] = 3   # valeurs positives → bin doit être +1
    model.prototypes_acc[0, 5:10] = -2  # valeurs négatives → bin doit être -1
    model.on_task_end(task_id=0, dataloader=None)
    assert (model.prototypes_bin[0, :5] == 1).all()
    assert (model.prototypes_bin[0, 5:10] == -1).all()

def test_incremental_no_catastrophic_forgetting():
    """
    Vérifie que l'accumulation de nouvelles données n'écrase pas les précédentes.
    Les prototypes doivent être monotoniquement plus riches (|acc| ≥ |acc_before|).
    """
    model = make_model()
    x1 = np.ones((10, N_FEATURES), dtype=np.float32)
    y1 = np.zeros(10, dtype=np.int64)
    model.update(x1, y1)
    acc_before = model.prototypes_acc[0].copy()

    x2 = np.ones((5, N_FEATURES), dtype=np.float32) * 2
    y2 = np.zeros(5, dtype=np.int64)
    model.update(x2, y2)
    acc_after = model.prototypes_acc[0]

    # Les counts de la classe 0 doivent avoir augmenté
    assert model.class_counts[0] == 15
```

---

## Contraintes embarquées (STM32N6)

| Composant | RAM | Annotation requise |
|-----------|-----|-------------------|
| `prototypes_acc [2, 1024]` INT32 | 8 192 B | `# MEM: 8 Ko @ INT32` |
| `prototypes_bin [2, 1024]` INT8 | 2 048 B | `# MEM: 2 Ko @ INT8` |
| Buffer `H_sum [1024]` INT32 (temporaire) | 4 096 B | `# MEM: 4 Ko @ INT32 (temporaire)` |
| `class_counts [2]` INT32 | 8 B | inclus dans `estimate_ram_bytes` |
| **TOTAL worst case** | **~14 Ko** | `check_ram_budget()` → < 64 Ko |

> Pas de `torch` dans `hdc_classifier.py` — NumPy pur pour portabilité MCU.
> L'inférence doit être implémentable en C sans bibliothèque.

---

## Critères d'acceptation

- [ ] `from src.models.hdc import HDCClassifier` — aucune erreur d'import
- [ ] `HDCClassifier` hérite correctement de `BaseCLModel` — toutes les méthodes abstraites implémentées
- [ ] `predict()` avant `update()` lève `RuntimeError`
- [ ] `update()` incrémente `prototypes_acc` et `class_counts` correctement
- [ ] `on_task_end()` re-binarise les prototypes sans erreur
- [ ] `estimate_ram_bytes("fp32")` < 65 536 (64 Ko)
- [ ] `check_ram_budget()["within_budget"] == True`
- [ ] `save()` + `load()` : round-trip sans perte
- [ ] Annotations `# MEM:` présentes sur `prototypes_acc`, `prototypes_bin`, `H_sum`
- [ ] Pas d'import `torch` dans `hdc_classifier.py`
- [ ] `pytest tests/test_hdc_classifier.py -v` — tous les tests passent (11/11)
- [ ] `ruff check src/models/hdc/hdc_classifier.py` + `black --check` passent

---

## Interface attendue par S2-03 (`train_hdc.py`)

```python
from src.models.hdc import HDCClassifier
import yaml

with open("configs/hdc_config.yaml") as f:
    config = yaml.safe_load(f)

model = HDCClassifier(config)

# Boucle tâche i
for x_batch, y_batch in train_loader:
    loss = model.update(x_batch.numpy(), y_batch.numpy())

# Fin tâche i
model.on_task_end(task_id=i, dataloader=train_loader)

# Inférence
preds = model.predict(x_val.numpy())
```

---

## Questions ouvertes

- `TODO(arnaud)` : Le scénario Domain-Incremental avec prototypes partagés (faulty/normal accumulés sur tous les domaines) est-il pertinent, ou faut-il séparer les prototypes par (classe, domaine) → Class-Incremental avec 6 classes ?  Voir `hdc_spec.md §5.1`.
- `TODO(dorra)` : L'accumulation INT32 peut déborder après ~2 milliards d'observations (int32 max). Faut-il prévoir une renormalisation périodique des `prototypes_acc` → `int16` ? Hors scope Sprint 2 mais à anticiper.
- `FIXME(gap2)` : Valider `estimate_ram_bytes()` par mesure réelle via `memory_profiler.py` (S2-03).
