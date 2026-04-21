# S9-07 — Use Case : Déploiement Embarqué (Détection Anomalie + Dérive)

| Champ | Valeur |
|-------|--------|
| **ID** | S9-07 à S9-10 |
| **Sprint** | Sprint 9 — Phase 1 Extension (use case déploiement) |
| **Priorité globale** | 🔴 Critique — positionnement Gap 1 + Gap 2 |
| **Durée estimée totale** | ~14h |
| **Dépendances** | `MahalanobisDetector` (S5-10), `train_unsupervised.py` (S5-05), `metrics.py` (S1-07) |

---

## Objectif

Modéliser et étudier le **scénario de déploiement réaliste** d'un modèle non supervisé sur une machine industrielle embarquée :

1. **Phase d'enrôlement** : collecter N échantillons depuis une machine "neuve" (état normal connu), entraîner le modèle uniquement sur ces données normales.
2. **Phase d'inférence** : le modèle tourne en continu sur le microcontrôleur, classifie chaque nouvelle mesure.
3. **Logique d'alerte** : distinguer deux types de déviations du score d'anomalie :
   - **FAULT** : dépassement ponctuel et intense → signal panne à l'opérateur
   - **DRIFT** : augmentation soutenue et progressive → signal besoin de MAJ du modèle
4. **MAJ sans stockage** : le modèle peut-il s'adapter au flux sans conserver les données brutes (uniquement les statistiques internes : centroïdes, μ, Σ⁻¹) ?

---

## Questions de recherche

### Q1 — Courbe d'apprentissage : combien de données d'enrôlement ?

> *Plot AUROC vs N — pour chaque modèle, quel est le N minimal pour atteindre AUROC ≥ 0.85 ?*

- Faire varier N ∈ {10, 25, 50, 100, 250, 500, 1000, 2500}
- Répéter 5 fois avec sous-échantillonnages différents → mean ± std
- Modèles candidats : Mahalanobis (priorité MCU), KMeans, PCA, KNN (borne supérieure)
- Guards : Mahalanobis skip N < 5, KMeans `k_max = min(config, N//2)`

**Résultat attendu** : Mahalanobis converge rapidement (~50–100 samples pour d=4), car il n'y a que d + d² = 20 paramètres à estimer.

### Q2 — Stratégie de MAJ : batch, mini-batch ou continu ?

> *Comparaison de 5 stratégies sur un stream de 1000 échantillons après enrôlement sur 500 samples*

| Stratégie | Description | MCU-compatible ? |
|-----------|-------------|:---:|
| `batch_refit` | Refit complet sur toutes les données vues | ❌ (stocke tout) |
| `online_welford` | MAJ sample-by-sample via Welford | ✅ |
| `minibatch_10` | MAJ toutes les 10 mesures | ✅ |
| `minibatch_50` | MAJ toutes les 50 mesures | ✅ |
| `minibatch_100` | MAJ toutes les 100 mesures | ✅ |

**Métrique** : AUROC recalculé tous les 10 pas sur un jeu de test fixe.

### Q3 — Discrimination FAULT vs DRIFT

> *À partir du seul score d'anomalie, comment distinguer une panne soudaine d'une dérive progressive ?*

Approche : fenêtre glissante de taille W sur les scores bruts.
- **FAULT** : score courant > `fault_threshold` (seuil = P95_enrôlement × 2.5)
- **DRIFT** : >60% des W derniers scores > `drift_threshold` (seuil = P95_enrôlement × 1.3)
- **NORMAL** : sinon

Simulation : injecter un spike à t=200 (FAULT) et une dérive linéaire progressive à t=400 (DRIFT), évaluer précision/rappel de la détection.

---

## Modèles prioritaires

| Modèle | Welford/Online ? | RAM MCU | Priorité |
|--------|:---:|---------|:---:|
| **MahalanobisDetector** | ✅ à implémenter | 80 B @ FP32 | 🔴 P1 |
| **PCABaseline** | ✅ déjà via `IncrementalPCA.partial_fit()` | 48 B (2 comp, d=4) | 🟡 P2 |
| **KMeansDetector** | 🟡 centroïdes EMA | 48 B (K=3, d=4) | 🟡 P2 |
| KNNDetector | ❌ stocke X_ref | 110 Ko (accumulate) | 📊 borne sup. courbes |
| DBSCANDetector | ❌ requiert graphe voisins | — | ❌ exclu online |

---

## Tâches planifiées

### S9-07 — `MahalanobisDetector.partial_fit()` — Algorithme de Welford

**Fichier** : `src/models/unsupervised/mahalanobis_detector.py`

Ajouter sans casser `fit_task()` existant :

```python
# Attributs internes à ajouter dans __init__()
self._n_seen_: int = 0
self._M2_: np.ndarray | None = None   # [d, d] somme des produits extérieurs

def _init_welford_from_fit(self) -> None:
    """Initialise l'état Welford depuis les statistiques du batch fit."""
    # _n_seen_ = n_train utilisés dans fit_task()
    # _M2_ = cov_batch * (n_train - 1)

def partial_fit(self, x: np.ndarray) -> "MahalanobisDetector":
    """
    MAJ online de mu_ et sigma_inv_ via l'algorithme de Welford.
    x : [d] ou [N, d]. Aucune donnée brute stockée.
    Ne met pas à jour threshold_ (seuil fixé depuis l'enrôlement).
    MAJ de sigma_inv_ conditionnée à _n_seen_ >= welford_min_samples.
    """

def reset_welford_state(self) -> None:
    """Réinitialise l'état online (nouvelle machine / nouveau contexte)."""
```

**Budget RAM MCU** :
- `mu_` : d × 4 B = 16 B
- `sigma_inv_` : d² × 4 B = 64 B
- `_M2_` : d² × 4 B = 64 B (uniquement pendant la MAJ)
- **Inférence** : 80 B / **MAJ** : 144 B → loin du budget 64 Ko ✅

**Nouveaux paramètres** dans `configs/unsupervised_config.yaml` :
```yaml
mahalanobis:
  welford_min_samples: 10   # min. samples avant MAJ Σ⁻¹
  update_sigma_every: 1     # 1=continu, 10/50/100=mini-batch
```

**Tests** : `tests/test_unsupervised.py` — vérifier cohérence `partial_fit()` vs `fit_task()` sur les mêmes données.

| Impl. | Doc | Exec |
|:-----:|:---:|:----:|
| ⬜ | ✅ | ⬜ |

---

### S9-08 — Expérience Learning Curve — exp_042

**Fichier** : `scripts/learning_curve_study.py`

```
python scripts/learning_curve_study.py \
  --config configs/unsupervised_config.yaml \
  --models mahalanobis,kmeans,pca,knn \
  --n_values 10,25,50,100,250,500,1000,2500 \
  --n_repeats 5 \
  --enrollment_domain Pump \
  --exp_id exp_042_learning_curve
```

**Output** :
```
experiments/exp_042_learning_curve/
├── config_snapshot.yaml
└── results/
    ├── learning_curve_mahalanobis.json   # {N: [auroc_run1, ..., auroc_run5]}
    ├── learning_curve_kmeans.json
    ├── learning_curve_pca.json
    ├── learning_curve_knn.json
    └── learning_curve_all.json
```

| Impl. | Doc | Exec |
|:-----:|:---:|:----:|
| ⬜ | ✅ | ⬜ |

---

### S9-09 — Expérience Online Update — exp_043

**Fichier** : `scripts/online_update_study.py`

Comparer les 5 stratégies sur `MahalanobisDetector` (+ `PCABaseline` comme référence) :
- Enrôlement : 500 samples normaux (Pump)
- Stream : 1 000 samples (mélange normal + anomalies)
- Évaluation : AUROC toutes les 10 MAJ

```
experiments/exp_043_online_update/
├── config_snapshot.yaml
└── results/
    ├── online_update_mahalanobis.json   # {strategy: [(step, auroc), ...]}
    └── online_update_pca.json
```

| Impl. | Doc | Exec |
|:-----:|:---:|:----:|
| ⬜ | ✅ | ⬜ |

---

### S9-10 — Module `SlidingWindowDriftDetector` + Notebook déploiement

**Fichier 1** : `src/evaluation/drift_detector.py`

```python
class SlidingWindowDriftDetector:
    """
    Détection FAULT vs DRIFT par fenêtre glissante sur scores d'anomalie.

    FAULT : score_t > fault_threshold  (dépassement instantané)
    DRIFT : fraction(scores[-W:] > drift_threshold) > drift_ratio
    NORMAL : sinon

    Compatible MCU : état = deque(maxlen=W), O(W) RAM.
    """

    def set_thresholds_from_normal(self, normal_scores: np.ndarray) -> None: ...
    def update(self, score: float) -> Literal["NORMAL", "FAULT", "DRIFT"]: ...
    def update_batch(self, scores: np.ndarray) -> list[str]: ...
    def get_window_stats(self) -> dict: ...
    def reset(self) -> None: ...
```

**Fichier 2** : `notebooks/cl_eval/deployment_scenario.ipynb`

Trois sections :
1. **Courbe d'apprentissage** — AUROC vs N, mean ± std, annotation "N minimal viable"
2. **Comparaison MAJ** — AUROC vs position stream par stratégie
3. **Discrimination FAULT/DRIFT** — stream simulé avec spike (t=200) + dérive (t=400), alertes du détecteur avec W={30, 50, 100}

| Impl. | Doc | Exec |
|:-----:|:---:|:----:|
| ⬜ | ✅ | ⬜ |

---

## Liens avec les Gaps

| Gap | Impact |
|-----|--------|
| **Gap 1** (validation industrielle) | La courbe d'apprentissage répond à : "combien de données réelles faut-il sur une machine neuve avant de pouvoir détecter les pannes ?" — valeur directe pour Edge Spectrum. |
| **Gap 2** (< 100 Ko RAM mesurée) | `partial_fit()` de Mahalanobis : **MAJ = 144 B**, **inférence = 80 B** — à mesurer via `memory_profiler.py` et reporter dans le manuscrit. |
| **Gap 3** (quantification INT8 pendant MAJ) | `FIXME(gap3)` : Welford's en FP32 uniquement pour l'instant. La MAJ INT8 est à explorer après validation FP32. |

---

## Résultats attendus

| Expérience | Résultat attendu | Utilisé dans le manuscrit |
|------------|-----------------|:---:|
| exp_042 | AUROC Mahalanobis ≥ 0.85 avec N ≥ 50 samples d'enrôlement | ✅ Section "Résultats" Phase 1 |
| exp_043 | `online_welford` ≈ `batch_refit` (< 2% d'écart AUROC) | ✅ Argument pour Gap 2 (MCU-faisable) |
| drift_detector | Précision/Rappel FAULT > 0.90, DRIFT > 0.80 avec W=50 | ✅ Figure "alerting pipeline" |
