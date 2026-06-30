# S3001 — Infrastructure paires `src/ensemble/model_pair.py`

| Champ | Valeur |
|-------|--------|
| **Sprint** | 30 |
| **Priorité** | 🔴 Critique |
| **Statut** | ✅ Implémenté |
| **Durée estimée** | 4h |
| **Dépendances** | `src/models/unsupervised/mahalanobis_detector.py` ✅ · `src/models/base_cl_model.py` ✅ · cadre binarisé Sprint 28 ✅ |
| **Fichiers cibles** | `src/ensemble/model_pair.py`, `src/ensemble/__init__.py` |
| **Références** | `src/models/base_cl_model.py` (`BaseCLModel`) · `src/training/scenarios.py` (`run_cl_scenario_full`) |

---

## Contexte

Aucune brique d'ensemble n'existe (confirmé par exploration). `ModelPair` est le wrapper net-new qui co-exécute un détecteur non-supervisé **Mahalanobis** et un modèle **supervisé** (`BaseCLModel` : HDC, EWC, TinyOL), expose leurs prédictions individuelles et une prédiction d'ensemble combinée. C'est le socle des Parties A/B du Sprint 30 et du méta-modèle Sprint 31.

---

## Spec

```python
class ModelPair:
    """Co-exécution Mahalanobis (non-supervisé) + modèle supervisé.

    Parameters
    ----------
    detector : MahalanobisDetector       # baseline anomaly, sortie 0/1
    classifier : BaseCLModel             # HDC / EWC / TinyOL
    mode : {"binary", "native"}          # A: normal-vs-fault ; B: tâche native
    fusion_rule : {"or", "and", "soft_vote", "weighted"}
    """

    def predict_individual(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (pred_maha, pred_sup) — sorties brutes alignées par échantillon."""

    def predict_ensemble(self, x, rule: str | None = None) -> np.ndarray:
        """Combine les 2 sorties selon fusion_rule (override possible)."""

    def predict_proba(self, x) -> np.ndarray:
        """Scores continus pour AUROC (moyenne pondérée des 2 modèles)."""
```

- **Mode `"binary"` (Partie A)** : binarise la sortie supervisée en normal-vs-fault via le mapping exact du Sprint 28 → les 2 sorties sont directement comparables (ensemble/désaccord propres).
- **Mode `"native"` (Partie B, S3007)** : conserve la sortie native du modèle supervisé (RUL / multi-classe) ; la fusion et le désaccord sont redéfinis par dataset.
- Annoter `# MEM:` sur les buffers (cible portable MCU à terme).

> Mahalanobis n'a pas l'interface `BaseCLModel` complète — l'adapter par une fine couche d'adaptation plutôt que forcer l'héritage.

---

## Vérification

```bash
python -c "from src.ensemble.model_pair import ModelPair; print('import OK')"
pytest tests/test_model_pair.py -v   # ajouté en S3012
```
