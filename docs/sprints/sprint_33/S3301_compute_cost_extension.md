# S3301 — Étendre `compute_cost.py` (FLOPs/BOPs/Params)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 33 |
| **Priorité** | 🔴 Critique — bloquant pour S3302 (hw_cost_model.py), S3303 (cross-check), S3308 (notebook) |
| **Statut** | ✅ Implémenté (23 juin 2026) |
| **Durée estimée** | 3h |
| **Dépendances** | `src/evaluation/compute_cost.py` ✅ (MACs only) |
| **Fichiers cibles** | `src/evaluation/compute_cost.py` |
| **Références** | `compute_cost.py:22` (`macs_ewc_mlp`), `:47` (`macs_tinyol`), `:72` (`macs_hdc`), `:90` (`macs_kmeans`), `:106` (`macs_mahalanobis`), `:121` (`macs_dbscan`), `:137` (`macs_tinyol_ae`), `:379` (`compute_macs` dispatcher), `:406` (`compute_training_macs`) |

---

## Contexte

`compute_cost.py` (432 lignes) ne calcule aujourd'hui **que les MACs**, par couche et par
modèle, via une somme de produits de dimensions (`sum(dims[i] * dims[i+1] ...)`). Le CR du
19 mai 2026 (Dorra, Frédéric) demande explicitement les **FLOPs**, les **BOPs** (pour rendre
la comparaison FP32/INT8 honnête en pondérant par le nombre de bits) et le **comptage de
paramètres**. Ce sprint étend le module existant sans toucher au comportement actuel.

---

## Spec

```python
# Pattern à dupliquer pour chaque macs_*() existant : macs_ewc_mlp, macs_tinyol, macs_hdc,
# macs_kmeans, macs_mahalanobis, macs_dbscan, macs_tinyol_ae (+ équivalents training_macs_*)

def compute_flops(macs: int) -> int:
    """FLOPs = 2 x MACs (1 multiplication + 1 addition par MAC)."""
    return 2 * macs

def compute_bops(macs: int, n_bits: int) -> int:
    """BOPs = FLOPs x n_bits^2 — rend la comparaison FP32 (n_bits=32) vs INT8 (n_bits=8)
    honnête : BOPs_INT8 / BOPs_FP32 = (8/32)^2 = 1/16 pour les mêmes FLOPs.
    """
    return compute_flops(macs) * (n_bits ** 2)

def count_params(model_name: str, trainable: bool = True, **kwargs) -> int:
    """Compte les paramètres (inférence + entraînables selon `trainable`), par couche et
    par modèle. Réutilise les mêmes kwargs de dimensions que macs_*() pour cohérence.
    """
    ...

# Dispatchers à étendre (même signature que compute_macs / compute_training_macs)
def compute_flops_for_model(model_name: str, **kwargs) -> int: ...
def compute_bops_for_model(model_name: str, n_bits: int, **kwargs) -> int: ...
def compute_params_for_model(model_name: str, trainable: bool = True, **kwargs) -> int: ...
```

**Règles** :
- **Non-régression stricte** : aucune fonction `macs_*()` / `compute_macs()` /
  `compute_training_macs()` existante ne change de comportement ni de valeur retournée.
  FLOPs/BOPs/Params sont des fonctions **additionnelles** qui consomment la valeur MACs déjà
  calculée (`compute_flops(macs_ewc_mlp(...))`), pas une réimplémentation parallèle.
  `params_*()` doit lire les mêmes paramètres de dimensions (`dims`, `n_features`,
  `n_classes`, etc.) que les `macs_*()` correspondants pour rester cohérent par couche.
- BOPs avec `n_bits=32` (FP32) vs `n_bits=8` (INT8) : le ratio attendu est ×1024 (32²) vs
  ×64 (8²) — documenter ce facteur dans le docstring, c'est l'argument central du CR 19 mai.
- Annoter chaque tenseur de paramètres avec `# MEM:` (règle CLAUDE.md), cohérent avec les
  annotations déjà présentes dans les modèles (`src/models/*/*.py`).

---

## Vérification

```bash
# Non-régression : les MACs identiques avant/après
python -c "from src.evaluation.compute_cost import compute_macs; print(compute_macs('ewc_mlp', dims=[5,16,2]))"

# Nouvelles fonctions
python -c "from src.evaluation.compute_cost import compute_flops, compute_bops, count_params; \
m = 80; print(compute_flops(m), compute_bops(m, 32), compute_bops(m, 8))"

pytest tests/test_compute_cost.py -v   # S3309 : FLOPs=2xMACs, BOPs FP32/INT8, non-régression MACs
```
