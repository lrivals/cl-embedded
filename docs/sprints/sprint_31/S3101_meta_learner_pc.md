# S3101 — Méta-learner `src/ensemble/meta_learner.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 31 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Dépendances** | Sprint 30 ✅ (`ModelPair`, `disagreement_metrics`) |
| **Fichiers cibles** | `src/ensemble/meta_learner.py` ✅, `tests/test_meta_learner.py` ✅ (12/12 PASS) |
| **Références** | `src/ensemble/model_pair.py` (S3001), `src/evaluation/disagreement_metrics.py` (S3003) |

---

## Contexte

Remplace la fusion statique du Sprint 30 par un arbitrage **appris**. Stacking léger, vecteur d'entrée compact → portable MCU (S3105).

---

## Spec

```python
class MetaLearner:
    """Stacking : arbitre les sorties de 2 modèles de base.

    kind : {"logreg", "mlp"}   # mlp = 1 couche cachée
    input_features : liste, p. ex. [score_maha, prob_sup, disagreement, conf_sup]
    """

    def fit(self, meta_X, y) -> None:
        """meta_X = features dérivées des 2 modèles de base (out-of-fold)."""

    def predict(self, meta_X) -> np.ndarray: ...
    def predict_proba(self, meta_X) -> np.ndarray: ...
    def export_weights(self) -> dict:
        """Poids prêts pour scripts/export_weights_c.py (S3105)."""
```

- **Anti-fuite** : `meta_X` collecté sur split out-of-fold dédié (le méta ne voit jamais les prédictions des bases sur leurs propres données d'entraînement).
- Vecteur d'entrée volontairement compact (≤ ~8 features) pour rester portable.
- Annoter `# MEM:` (cible MCU).

---

## Vérification

```bash
python -c "from src.ensemble.meta_learner import MetaLearner; print('OK')"
pytest tests/test_meta_learner.py -v   # S3112 : pas de fuite, méta ≥ baseline synthétique
```

---

## Notes d'implémentation (✅)

- **`build_meta_features(pair, X)`** (fonction module) construit le vecteur compact en réutilisant
  les internes calibrés de `ModelPair` : `_maha_proba`, `_supervised_proba`, `predict_individual`.
  Features disponibles `AVAILABLE_FEATURES` (`p_maha`, `p_sup`, `pred_maha`, `pred_sup`,
  `disagreement`, `conf_sup`, `conf_maha`), défaut `DEFAULT_FEATURES = [p_maha, p_sup,
  disagreement, conf_sup]`. **Toutes bornées [0, 1]** → portable MCU sans scaler.
- **`MetaLearner`** : `logreg` → `sklearn.LogisticRegression` ; `mlp` → `MLPClassifier` 1 couche.
  Paramètre **`class_weight="balanced"`** (config-driven) sur la logreg : évite l'effondrement vers
  la classe majoritaire sur les datasets à panne minoritaire (sans lui, F1=0 observé sur
  cmapss/pronostia).
- **`export_weights()`** : `{kind, w, b, feature_names}` (logreg) ou `{kind, w1, b1, w2, b2,
  feature_names}` (mlp), poids FP32 → consommables par `scripts/export_weights_c.py` (S3105).
- Tests `tests/test_meta_learner.py` : **12/12 PASS** (construction, shapes, bornes des features,
  anti-fuite split disjoint, méta ≥ max(base) sur cas synthétique arbitrable, export logreg+mlp).
