# S11-07/08/09 — Notebooks Single-Task Online Non Supervisés (KMeans, Mahalanobis, DBSCAN)

| Champ | Valeur |
|-------|--------|
| **IDs** | S11-07, S11-08, S11-09 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🟡 (KMeans, Mahalanobis), 🟢 (DBSCAN) |
| **Durée estimée** | 6h total |
| **Dépendances** | S11-01 (`feature_importance.py`), exp_033–035 existent |
| **Fichiers cibles** | `notebooks/cl_eval/monitoring_single_task/{kmeans,mahalanobis,dbscan}.ipynb` |

---

## Objectif

Évaluer les trois modèles non supervisés en mode **online strict** (un échantillon à la fois,
évaluation prequential predict-then-update) sur le dataset Equipment Monitoring fusionné
(tous équipements, single-task, pas de scénario CL).

Stratégie CL : **refit en priorité** — réentraîner le modèle sur le buffer de normaux croissant
à chaque nouveau point classifié "normal" avec confiance suffisante.

## Structure commune (9 sections)

| Section | Contenu |
|---------|---------|
| 1 | Setup, imports, constantes (`NORMAL_BUFFER_SIZE`, `REFIT_FREQ`) |
| 2 | Métriques batch pré-calculées (`exp_033/034/035`) |
| 3 | Bar chart AUC-ROC / F1 (seuil optimal) / accuracy |
| 4 | Chargement données |
| 5 | **Boucle prequential** + courbe rolling AUC (w=100, score d'anomalie) |
| 6 | Confusion matrix (test set, seuil fixé post-hoc) |
| 7 | Courbe ROC (test set) |
| 8 | PCA 2D (normal vs anomalie détectée + erreurs) |
| 9 | **Contribution des variables** (permutation sur score d'anomalie) + JSON |

> Pas de section 0 de préentraînement — les modèles non supervisés s'initialisent sur les
> premiers `N_INIT` échantillons supposés normaux (N_INIT défini dans `configs/`).

## Spécificités par modèle

### KMeans (`kmeans.ipynb`) — exp_033

- Score d'anomalie : distance au centroïde le plus proche (normalisée par rayon moyen du cluster)
- Prequential : si `score < θ` → point classifié normal → ajouté au buffer → refit KMeans
  toutes les `REFIT_FREQ` observations (ex. 50)
- `REFIT_FREQ` et `NORMAL_BUFFER_SIZE` définis dans `configs/kmeans_config.yaml`
- Contribution : permutation sur `anomaly_score = min_dist(x, centroids)` (AUC-ROC comme référence)

### Mahalanobis (`mahalanobis.ipynb`) — exp_034

- Score d'anomalie : distance de Mahalanobis au centroïde de la classe normale
  `d(x) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))`
- Prequential : si `d(x) < θ` → point classifié normal → mise à jour en ligne de μ et Σ
  (moyenne/covariance glissantes avec facteur d'oubli `α`) **ou** refit complet sur buffer
- Stratégie refit par défaut (`cl_strategy: "refit"` dans config)
- Contribution : permutation sur `d(x)` — interprétable directement (chaque feature contribue
  quadratiquement via Σ⁻¹)

### DBSCAN (`dbscan.ipynb`) — exp_035

- Score d'anomalie : point hors de tout cluster (`label == -1`) → binaire, pas de score continu
  → score proxy : distance au cluster le plus proche (norme L2 au voisin core-point le plus proche)
- Prequential : DBSCAN n'a pas de mise à jour en ligne native → refit sur fenêtre glissante
  de taille `WINDOW_SIZE` (ex. 500 derniers normaux)
- Refit déclenché toutes les `REFIT_FREQ` observations (coût O(n²) → limiter `WINDOW_SIZE ≤ 500`)
- Contribution : permutation sur le score proxy de distance — résultats à interpréter avec prudence
  (baseline, protocole heuristique)
- `TODO(arnaud)` : marquer ce protocole comme "baseline exploratoire" dans le manuscrit

## Protocole de boucle prequential (commun)

```python
normal_buffer = []          # échantillons classifiés normaux
model = init_model()        # initialisé sur N_INIT premiers points

for t, (x, y_true) in enumerate(stream):
    score = model.anomaly_score(x)       # → float ∈ [0, +∞)
    y_pred = int(score > threshold)      # seuil calibré sur split init

    rolling_scores.append(score)
    rolling_labels.append(y_true)

    if y_pred == 0:                      # classifié normal
        normal_buffer.append(x)
        if t % REFIT_FREQ == 0 and len(normal_buffer) >= N_INIT:
            model.fit(normal_buffer[-NORMAL_BUFFER_SIZE:])   # refit
```

## Résultats de sortie

```
experiments/exp_033_kmeans_monitoring_single_task/results/feature_importance.json
experiments/exp_034_mahalanobis_monitoring_single_task/results/feature_importance.json
experiments/exp_035_dbscan_monitoring_single_task/results/feature_importance.json

notebooks/figures/cl_evaluation/kmeans/monitoring/single_task/
  ├── batch_metrics.png
  ├── prequential_rolling_auc.png
  ├── confusion_matrix.png
  ├── roc_curve.png
  ├── pca_feature_space.png
  └── feature_importance_permutation.png
[idem pour mahalanobis/ et dbscan/]
```

## Statut

⬜ `kmeans.ipynb` — à implémenter (S11-07)
⬜ `mahalanobis.ipynb` — à implémenter (S11-08)
⬜ `dbscan.ipynb` — à implémenter (S11-09)
