# S3003 — Métriques de désaccord `src/evaluation/disagreement_metrics.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🔴 Critique |
| **Statut** | ⬜ À démarrer |
| **Durée estimée** | 3h |
| **Dépendances** | S3001 (`ModelPair`) · `src/training/scenarios.py` (`run_cl_scenario_full`) · `src/evaluation/metrics.py` |
| **Fichiers cibles** | `src/evaluation/disagreement_metrics.py` |
| **Références** | `src/evaluation/metrics.py`, `src/evaluation/anomaly_metrics.py` |

---

## Contexte

Net-new : aucune mesure de désaccord inter-modèles n'existe. Cette tâche quantifie **où** les 2 modèles d'une paire divergent, **qui a raison** quand ils divergent, et **pourquoi** (origine dans l'espace des features). C'est ce qui distingue le benchmark « paire » d'un simple empilement de deux résultats individuels.

---

## Spec

```python
def disagreement_rate(y_a, y_b) -> float:
    """Fraction d'échantillons où pred_a != pred_b."""

def cohen_kappa(y_a, y_b) -> float:
    """Accord inter-modèles corrigé du hasard."""

def disagreement_confusion(y_a, y_b, y_true) -> dict:
    """Sur le sous-ensemble en désaccord : combien de fois a a raison,
    b a raison, ni l'un ni l'autre. Renvoie {a_correct, b_correct, both_wrong}."""

def per_sample_disagreement_mask(y_a, y_b) -> np.ndarray:
    """Masque booléen des échantillons en désaccord (pour analyse origine)."""

def analyze_disagreement_origin(X, mask, y_true, maha_scores=None) -> dict:
    """Corrèle le masque de désaccord aux features, au score Mahalanobis
    et à la proximité de frontière de décision. Renvoie les features les
    plus discriminantes du désaccord + stats (mean score in/out mask)."""
```

Récupérer `(y_true, y_pred_a)` et `(y_true, y_pred_b)` via `run_cl_scenario_full()` exécuté pour chaque modèle, aligner par index d'échantillon, puis appliquer ces fonctions.

---

## Vérification

```bash
python -c "from src.evaluation.disagreement_metrics import disagreement_rate, cohen_kappa; print('OK')"
pytest tests/test_disagreement.py -v   # ajouté en S3012
```
