# S11-01 — Module `feature_importance.py`

| Champ | Valeur |
|-------|--------|
| **ID** | S11-01 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 3h |
| **Dépendances** | — |
| **Fichiers cibles** | `src/evaluation/feature_importance.py`, `src/evaluation/__init__.py` |

---

## Objectif

Implémenter un module d'analyse de contribution des variables d'entrée pour les modèles CL embarqués,
sans dépendances lourdes (pas de SHAP, pas de LIME — incompatibles avec le workflow MCU).

## Méthodes implémentées

| Fonction | Modèles | Principe |
|----------|---------|---------|
| `permutation_importance()` | EWC, HDC, TinyOL | Permuter colonne j → mesurer chute d'accuracy |
| `gradient_saliency()` | EWC uniquement (PyTorch) | `mean_i |∂ŷ_i/∂x_ij|` sur le test set |
| `feature_masking_importance()` | HDC (alternative déterministe) | Remplacer colonne j par 0 → mesurer chute |
| `plot_feature_importance()` | — | Barplot horizontal |
| `plot_feature_importance_comparison()` | — | Barplot groupé multi-méthodes |

## Interface

```python
# Agnostique au modèle — EWC, HDC, TinyOL
perm_imp = permutation_importance(
    predict_fn=lambda X: model_output(X),  # [N, F] → [N]
    X=X_test,     # [N, 4] normalisé Z-score
    y=y_test,     # [N] labels 0/1
    feature_names=["temperature", "pressure", "vibration", "humidity"],
    n_repeats=10,
    random_state=42,
)
# → {"temperature": 0.023, "vibration": 0.015, "pressure": 0.008, "humidity": 0.001}

# EWC uniquement
grad_sal = gradient_saliency(model, X_test, feature_names)
```

## Statut

✅ Implémenté — `src/evaluation/feature_importance.py`
✅ Exporté depuis `src/evaluation/__init__.py`
