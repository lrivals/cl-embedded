# S2804 — `src/models/tinyol/tinyol_int8.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 28 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (12 juin 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | `src/models/tinyol/oto_head.py` ✅ · `src/models/tinyol/autoencoder.py` ✅ · `src/utils/quantization.py` ✅ (`quantize_uint8`, `dequantize_uint8`) · Sprint 4 (UINT8 buffer OtOHead, pattern déjà existant dans `oto_head.py`) |
| **Fichier cible** | `src/models/tinyol/tinyol_int8.py` |
| **Références** | `src/models/ewc/ewc_mlp_int8.py` (pattern complet fake-quant), `src/utils/quantization.py` |

---

## Contexte

Sprint 4 a introduit `uint8_activations` dans `OtOHead` pour stocker les activations intermédiaires en UINT8 (réduction buffer 4×). `tinyol_int8.py` étend ce pattern à l'autoencoder complet et expose une interface `forward_int8()` avec poids INT8. Le pattern suit `ewc_mlp_int8.py` : fake-quantization (poids stockés en INT8, calcul dequantifié en FP32).

---

## Spec de l'interface

```python
class TinyOLAutoencoderInt8:
    """TinyOL autoencoder with INT8 weight storage and UINT8 activation buffers.

    Fake-quantization: weights stored as int8 (via calibration), forward
    dequantizes to float32 for computation. Same approach as ewc_mlp_int8.py.
    """

    def calibrate_int8(self, X_calib: np.ndarray) -> None:
        """Calibrate scale/zero_point for each weight tensor using X_calib.
        Called once after initial training, before INT8 inference."""

    def forward_int8(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encoder + decoder pass with INT8 weights (dequantized to fp32).
        Returns (encoded, reconstructed). Intermediate activations as uint8."""

    def reconstruction_error_int8(self, x: np.ndarray) -> float:
        """MSE reconstruction error using INT8 forward."""

    def get_memory_footprint_int8(self) -> dict:
        """Returns {'weights_bytes': ..., 'activation_buffer_bytes': ..., 'total_bytes': ...}"""


class OtOHeadInt8:
    """Online-to-Offline head with INT8 weight storage.

    SGD update uses fake-quantization: dequant → gradient fp32 → requant.
    """

    def predict_int8(self, encoded: np.ndarray) -> int:
        """Forward with INT8 weights (dequantized to fp32)."""

    def update_int8(self, encoded: np.ndarray, y: int) -> None:
        """SGD 1 step with INT8 fake-quant weights."""

    def get_memory_footprint_int8(self) -> dict:
        """Returns weight footprint in INT8."""
```

---

## Architecture mémoire (architecture 9→32→16→9 encoder/decoder, OtOHead 16→2)

| Composant | RAM FP32 | RAM INT8 | Ratio |
|-----------|:--------:|:--------:|:-----:|
| Encoder weights (9→32 + 32→16) | ~3 456 B | ~864 B | ×4.0 |
| Decoder weights (16→32 + 32→9) | ~3 456 B | ~864 B | ×4.0 |
| OtOHead weights (16→2) | 136 B | ~34 B | ×4.0 |
| Activation buffers (uint8, run-time) | 128+64=192 B | 48+24=72 B | ×2.67 |
| **Total** | **~7 240 B** | **~1 834 B** | **×3.9** |

> **MEM annotation** : `hidden = relu(self.fc1_enc(x))  # MEM: 128 B @ FP32 / 32 B @ UINT8`

---

## Notes d'implémentation

- Reprendre le pattern `calibrate_uint8()` déjà dans `ewc_mlp.py` (Sprint 24)
- Les poids INT8 sont stockés comme attributs `int8` + scale/zero_point FP32
- L'update SGD INT8 (OtOHead) : dequant poids → gradient FP32 → clip → requant (pattern identique à `ewc_mlp_int8.py::sgd_step_int8()`)
- Les activations intermédiaires de l'encoder sont stockées en UINT8 (asymétrique, range [0,255] après ReLU)
