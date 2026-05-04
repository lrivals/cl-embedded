# S14-03 — Intégration `DBSCANDetector` dans `run_anomaly_detection_scenario()`

| Champ | Valeur |
|-------|--------|
| **ID** | S14-03 |
| **Sprint** | Sprint 14 |
| **Priorité** | 🔴 Critique |
| **Durée estimée** | 1.5h |
| **Dépendances** | — (`DBSCANDetector` déjà implémenté) |
| **Fichier cible** | `src/models/unsupervised/dbscan_detector.py` |

---

## Objectif

Vérifier que `DBSCANDetector` expose correctement l'API attendue par `run_anomaly_detection_scenario()` et corriger les écarts éventuels. Le scénario générique attend : `fit_task(X_normal)`, `predict_score(X) → np.ndarray`, `on_task_end()`.

---

## Checklist de vérification

### API à valider

```python
detector = DBSCANDetector(eps=0.5, min_samples=5, strategy="refit")

# Tâche 1
detector.fit_task(X_train_normal_t1)       # entraîne DBSCAN sur données normales
scores = detector.predict_score(X_test)    # retourne distance au cluster le plus proche
detector.on_task_end()                     # réinitialise ou accumule selon strategy

# Tâche 2
detector.fit_task(X_train_normal_t2)
scores = detector.predict_score(X_test)
```

### Points critiques

| Point | Attendu | À vérifier |
|-------|---------|------------|
| `fit_task` | Entraîne uniquement sur X_normal (label=0) | Pas de filtrage interne requis si le loader fournit déjà X_normal |
| `predict_score` | Retourne distance au centroïde/cluster le plus proche | Score plus élevé = plus anormal (convention AUROC) |
| `on_task_end` | `refit` : réinitialise `self.X_seen_` ; `accumulate` : accumule `self.X_seen_` | Vérifier que le mode est bien contrôlé par `strategy` |
| Retour dtype | `np.ndarray`, shape `(N,)`, dtype `float32` | Vérifier compatibilité avec `sklearn.metrics.roc_auc_score` |

### Problème connu potentiel

Pour la stratégie **accumulate**, DBSCAN est refitté sur `X_seen_` cumulé à chaque `fit_task()`. Avec 5 tâches (by_location), le dataset cumulé peut devenir très grand → risque de lenteur. Documenter la complexité dans `dbscan_detector.py` avec un commentaire.

---

## Modification minimale attendue

Si l'API est déjà conforme, seul un commentaire de confirmation suffit. Si des ajustements sont nécessaires :

```python
def predict_score(self, X: np.ndarray) -> np.ndarray:
    # Calcule la distance min de chaque point au cluster DBSCAN le plus proche.
    # Points hors cluster (label=-1 lors du fit) → distance maximale.
    ...
```

---

## Critères d'acceptation

- [ ] `DBSCANDetector.fit_task(X_normal)` accepte `np.ndarray` de shape `(N, d)`
- [ ] `DBSCANDetector.predict_score(X)` retourne `np.ndarray` de shape `(N,)`, valeurs ≥ 0
- [ ] `DBSCANDetector.on_task_end()` fonctionne pour les deux stratégies `refit` et `accumulate`
- [ ] Pas de modification de `src/training/scenarios.py` (interface générique respectée)
- [ ] Complexité accumulate documentée en commentaire

## Statut

⬜ À faire
