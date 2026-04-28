# S11-23 — Ablation Study : retrait progressif de features

| Champ | Valeur |
|-------|--------|
| **ID** | S11-23 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟡 Important |
| **Durée estimée** | 3h |
| **Dépendances** | S11-16 (exp_100–105), S11-17 (exp_106–111) — JSONs `feature_importance.json` requis |
| **Fichiers cibles** | `src/evaluation/feature_importance.py` (nouvelle fonction), `notebooks/cl_eval/ablation_feature_removal/` |

---

## Objectif

Mesurer la **dégradation progressive** des performances quand on retire les features les moins
importantes une par une, dans l'ordre dicté par `permutation_importance`. Produit une courbe
*nb_features → AUC* qui répond directement à : "jusqu'à combien de features peut-on descendre
sans perte significative ?"

Argument scientifique ciblé : **Gap 2** — justifier chiffres de RAM précis sur MCU (chaque feature
retirée = réduction proportionnelle de la couche d'entrée et des buffers intermédiaires).

---

## Nouvelle fonction dans `feature_importance.py`

```python
def ablation_by_removal(
    train_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn_factory: Callable[[Any], Callable[[np.ndarray], np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    importance_ranking: list[str],
    score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> list[dict]:
    """
    Retire les features de la moins importante à la plus importante et évalue à chaque étape.

    À chaque itération k (k = 0, …, n_features - 1) :
        - features actives = importance_ranking[: n_features - k]  (les k moins importantes retirées)
        - réentraîne le modèle sur X_train[:, active_cols]
        - évalue score_fn(y_test, predict(X_test[:, active_cols]))

    Parameters
    ----------
    train_fn : Callable[[X, y], model]
        Entraîne un modèle et le retourne. Appelé à chaque étape.
    predict_fn_factory : Callable[[model], predict_fn]
        Construit la fonction de prédiction depuis le modèle retourné par train_fn.
    X_train, y_train : np.ndarray
        Données d'entraînement normalisées.
    X_test, y_test : np.ndarray
        Données de test (fixes à travers toutes les itérations).
    feature_names : list[str]
        Noms des features dans l'ordre des colonnes.
    importance_ranking : list[str]
        Features ordonnées de la plus importante à la moins importante
        (sortie de permutation_importance — clés dans l'ordre du dict).
    score_fn : Callable[[y_true, y_pred], float] | None
        Métrique d'évaluation. Par défaut : AUC-ROC (sklearn.metrics.roc_auc_score).

    Returns
    -------
    list[dict]
        [
          {"n_features": 9, "features": [...], "score": 0.972},
          {"n_features": 8, "features": [...], "score": 0.968},
          ...
          {"n_features": 1, "features": ["kurtosis"], "score": 0.831},
        ]
    """
```

### Logique interne

```
importance_ranking = ["kurtosis", "rms", "crest", "sd", "form", "skewness", "max", "mean", "min"]
                       ↑ plus importante                                              ↑ moins importante

étape 0 : actives = toutes (9) → score_9
étape 1 : retire "min"        → actives (8) → score_8
étape 2 : retire "mean"       → actives (7) → score_7
...
étape 8 : ne garde que "kurtosis" → actives (1) → score_1
```

---

## Notebook

### Fichier

```
notebooks/cl_eval/ablation_feature_removal/ablation_cwru_pronostia.ipynb
```

### Structure (6 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, chargement des JSONs d'importance (exp_100–111) |
| 2 | Courbe ablation CWRU — AUC vs nb_features (4 modèles × 2 scénarios) sur un même plot |
| 3 | Courbe ablation Pronostia — AUC vs nb_features (4 modèles × 1 scénario) |
| 4 | Annotation RAM économisée — axe secondaire : Ko RAM gagnés par feature retirée (calcul FP32) |
| 5 | Tableau de synthèse — "feature budget" optimal par modèle (coude de la courbe) |
| 6 | Markdown — Recommandation : jeu de features minimal pour déploiement STM32N6 |

### Calcul RAM par feature (section 4)

```python
# CWRU — couche d'entrée MLP EWC (exemple)
n_features = 9
hidden_size = 32
ram_input_layer_fp32 = n_features * hidden_size * 4  # poids + biais ≈ négligeables
# Pour KMeans : n_clusters × n_features × 4 B @ FP32
# Pour Mahalanobis : n_features × n_features × 4 B (matrice Σ⁻¹)
```

Les chiffres exacts dépendent du modèle — annoter avec `# MEM:` selon la convention du projet.

### Format JSON de sortie

```json
{
  "model": "KMeans",
  "dataset": "cwru",
  "scenario": "by_fault_type",
  "ablation_removal": [
    {"n_features": 9, "features_active": ["kurtosis", "rms", "crest", "sd", "form", "skewness", "max", "mean", "min"], "auc_roc": 0.972},
    {"n_features": 8, "features_active": ["kurtosis", "rms", "crest", "sd", "form", "skewness", "max", "mean"],         "auc_roc": 0.968},
    {"n_features": 5, "features_active": ["kurtosis", "rms", "crest", "sd", "form"],                                    "auc_roc": 0.961},
    {"n_features": 3, "features_active": ["kurtosis", "rms", "crest"],                                                  "auc_roc": 0.943},
    {"n_features": 1, "features_active": ["kurtosis"],                                                                  "auc_roc": 0.831}
  ]
}
```

Sauvegarder dans `experiments/exp_1XX/results/ablation_feature_removal.json`.

---

## Questions ouvertes

- `TODO(arnaud)` : Le coude de la courbe (où la pente s'accélère) est-il le bon critère
  de sélection, ou préférer un seuil absolu de dégradation acceptable (ex. AUC < 0.95) ?
- `FIXME(gap2)` : Annoter dans le notebook le gain RAM exact par feature retirée pour chaque
  modèle — chiffres à reporter dans le manuscrit pour justifier la réduction de l'empreinte MCU.

## Statut

⬜ Fonction `ablation_by_removal()` dans `feature_importance.py`
⬜ Notebook `ablation_cwru_pronostia.ipynb`
