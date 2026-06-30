# S2805 — `src/models/unsupervised/mahalanobis_int8.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 28 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté (12 juin 2026) |
| **Durée estimée** | 2h |
| **Dépendances** | `src/models/unsupervised/mahalanobis_detector.py` ✅ · `src/utils/quantization.py` ✅ |
| **Fichier cible** | `src/models/unsupervised/mahalanobis_int8.py` |
| **Références** | `src/models/unsupervised/mahalanobis_detector.py` (pattern fit/score) · commentaire ligne ~31 : `# MEM: 80 B @ FP32 / 20 B @ INT8 (d=4)` |

---

## Contexte

Mahalanobis est le détecteur non-neuronal le plus compact du projet (~200 B RAM pour d=5, mesuré Sprint 20). La quantification porte sur `mu_` (vecteur moyen, `int8` affine) et `sigma_inv_` (matrice de covariance inverse, `int8` par-matrice). Pas de mise à jour de poids pendant l'inférence — le fit est offline.

**Différence structurelle vs modèles neuronaux** : pas de forward pass complexe, juste une distance. La quantification en INT8 donne une compression ×4 mais la distance de Mahalanobis est recalculée en FP32 après dequantification.

---

## Spec de l'interface

```python
class MahalanobisDetectorInt8:
    """Mahalanobis anomaly detector with int8 parameter storage.

    mu_ and sigma_inv_ stored as int8 affine quantized.
    score() dequantizes to float32 before distance computation.
    """

    def fit(self, X: np.ndarray) -> "MahalanobisDetectorInt8":
        """Fit, then calibrate int8 quantization. Returns self."""

    def calibrate_int8(self) -> None:
        """Quantize mu_ and sigma_inv_ to int8 affine.
        Uses compute_scale_zero_point() from quantization.py."""

    def score_int8(self, x: np.ndarray) -> float:
        """Mahalanobis distance with int8 parameters (dequantized to fp32).
        Returns anomaly score (higher = more anomalous)."""

    def get_memory_footprint_int8(self) -> dict:
        """Returns {'mu_bytes': d, 'sigma_inv_bytes': d*d, 'total_bytes': d+d*d}"""
        d = len(self.mu_)
        return {
            "mu_bytes": d * 1,            # int8 (1B per element)
            "sigma_inv_bytes": d * d * 1,  # int8 (1B per element)
            "scales_bytes": (d + d * d) * 4,  # float32 scales (overhead)
            "total_bytes": d + d * d
        }
```

---

## Budget mémoire

| Composant | FP32 | INT8 | Notes |
|-----------|:----:|:----:|-------|
| `mu_` (d=5) | 20 B | **5 B** | vecteur moyen |
| `sigma_inv_` (5×5) | 100 B | **25 B** | matrice inverse |
| Scales FP32 (overhead) | — | 40 B | scale + zero_point par tensor |
| **Total paramètres** | **120 B** | **70 B** | compression ×1.7 (avec scales) |
| **Total poids purs** | **120 B** | **30 B** | compression ×4.0 (sans scales) |

> **Note** : Mahalanobis est déjà très compact (~200 B en FP32 mesuré Sprint 20). L'impact réel de l'INT8 est minimal en valeur absolue mais valide la généralité de l'approche.

---

## Notes d'implémentation

- `sigma_inv_` a des valeurs potentiellement grandes → quantification par-matrice avec scale global (pas par-ligne)
- Vérifier `TODO(arnaud)` : quantification INT8 pour `sigma_inv_` peut dégrader la précision si les valeurs ont une grande dynamique — prévoir test `delta_auroc < 0.02`
- Si INT8 `sigma_inv_` ne tient pas le critère, fallback sur INT16 (Q15) et documenter
