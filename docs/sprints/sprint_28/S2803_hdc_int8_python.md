# S2803 — `src/models/hdc/hdc_int8.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 28 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (12 juin 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | `src/models/hdc/hdc_classifier.py` ✅ (pattern existant) · `src/utils/quantization.py` ✅ · `docs/sprints/sprint_24/S2402_uint8_ewc_hdc.md` ✅ (méthode `get_memory_footprint()` déjà esquissée) |
| **Fichier cible** | `src/models/hdc/hdc_int8.py` |

---

## Contexte

HDC (Hyperdimensional Computing) est architecturalement entier — les hypervecteurs de base binarisés ont des valeurs ±1 naturellement stockables en `int8`. Sprint 24 a esquissé la méthode `get_memory_footprint()` et calculé une compression ×2.67 (18 Ko INT vs 49 Ko FP32 hypothétique). `hdc_int8.py` formalise cette représentation et expose la même interface que `hdc_classifier.py`.

**Point clé** : pour HDC, "INT8" signifie principalement une réduction mémoire (stockage des hypervecteurs en `int8` au lieu de `float32`). La précision est identique car les valeurs binarisées sont exactement représentées en `int8`.

---

## Spec de l'interface

```python
class HDCClassifierInt8:
    """HDC classifier with int8 hypervector storage.

    Base vectors stored as int8 (values ±1), associative memory as int16
    to avoid overflow on bundle accumulation.
    """

    def __init__(self, config: dict):
        self.D = config.get("hdc_dim", 2048)
        self.n_features = config.get("n_features", 9)
        self.n_classes = config.get("n_classes", 4)

        # Base vectors: int8 (values in {-1, +1})
        # MEM: D × n_features × 1B = 2048 × 9 × 1 = 18 432 B @ INT8
        self.base_vecs: np.ndarray  # shape (n_features, D), dtype=int8

        # Associative memory: int16 (accumulates bundles without overflow)
        # MEM: n_classes × D × 2B = 4 × 2048 × 2 = 16 384 B @ INT16
        self.am: np.ndarray  # shape (n_classes, D), dtype=int16

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize base vectors (binarized random) and encode training samples."""

    def encode_int8(self, x: np.ndarray) -> np.ndarray:
        """Encode sample x into int8 hypervector. Returns int8 array of shape (D,)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Query AM with int8-encoded hypervectors. Returns class indices."""

    def update_int8(self, x: np.ndarray, y: int) -> None:
        """Online update: add encoded x to AM[y] (int16 accumulation)."""

    def get_memory_footprint_int8(self) -> dict:
        """Returns {'base_vecs_bytes': ..., 'am_bytes': ..., 'total_bytes': ...}"""
        base_bytes = self.D * self.n_features * 1   # int8
        am_bytes = self.n_classes * self.D * 2       # int16
        return {
            "base_vecs_bytes": base_bytes,
            "am_bytes": am_bytes,
            "total_bytes": base_bytes + am_bytes
        }
```

---

## Notes d'implémentation

- Les hypervecteurs de base sont générés avec `np.random.choice([-1, 1], size=(n_features, D)).astype(np.int8)`
- L'encodage utilise la multiplication entière : `hv = np.prod(base_vecs[features > threshold], axis=0)` → valeurs ±1 en `int8`
- Le bundle AM s'accumule en `int16` pour éviter l'overflow (N bundles × ±1 → range [-N, N])
- La distance cosinus pour la query se calcule en `int32` (produit scalaire) puis ramenée à `float32` pour le softmax

## Comparaison FP32 vs INT8

| Composant | FP32 | INT8/INT16 |
|-----------|:----:|:----------:|
| Base vectors (2048×9) | 73 728 B | **18 432 B** (×4) |
| AM (4×2048) | 32 768 B | **16 384 B** (×2, int16) |
| **Total** | **106 496 B** | **34 816 B** (×3.06) |

> **Note manuscrit** : La compression HDC est une propriété architecturale, pas une quantification au sens strict. HDC est nativement entier — `TODO(dorra)` pour arbitrage terminologique dans le manuscrit.
