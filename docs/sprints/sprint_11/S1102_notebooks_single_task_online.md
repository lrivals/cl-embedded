# S11-02/03/04 — Notebooks Single-Task Online (EWC, HDC, TinyOL)

| Champ | Valeur |
|-------|--------|
| **IDs** | S11-03, S11-04, S11-05 |
| **Sprint** | Sprint 11 |
| **Priorité** | 🔴 (EWC), 🟡 (HDC, TinyOL) |
| **Durée estimée** | 7h total |
| **Dépendances** | S11-01 (`feature_importance.py`), exp_030–032 existent |
| **Fichiers cibles** | `notebooks/cl_eval/monitoring_single_task/{ewc,hdc,tinyol}.ipynb` |

---

## Objectif

Évaluer les trois modèles CL en mode **online strict** (un échantillon à la fois,
évaluation prequential predict-then-update) sur le dataset Equipment Monitoring fusionné
(tous équipements, single-task, pas de scénario CL).

## Structure commune (10 sections)

| Section | Contenu |
|---------|---------|
| 0 (TinyOL only) | Pretraining backbone `MonitoringAutoencoder 4→8→4` sur données normales |
| 1 | Setup, imports, constantes |
| 2 | Métriques batch pré-calculées (`exp_030/031/032`) |
| 3 | Bar chart accuracy/F1/AUC-ROC |
| 4 | Chargement données |
| 5 | **Boucle prequential** + courbe rolling accuracy (w=100) |
| 6 | Confusion matrix (test set) |
| 7 | Courbe ROC (test set) |
| 8 | PCA 2D (vérité terrain + erreurs) |
| 9 | Tableau récapitulatif (batch vs prequential vs test) |
| 10 | **Contribution des variables** (plots + JSON) |

## Spécificités par modèle

### EWC (`ewc.ipynb`)
- Prequential : SGD(`lr=0.001, momentum=0.9`) sur 1 échantillon
- Contribution : permutation + gradient saliency (`|∂ŷ/∂x|`)
- Figure comparative : permutation vs. gradient saliency (normalisé)

### HDC (`hdc.ipynb`)
- Prequential : warm-start nécessaire (`_fitted=False` → prior=0 pour t=0)
- Score ROC : similarité cosinus au prototype faulty (`prototypes_bin @ H_obs`)
- Contribution : permutation + feature masking (mask=0.0 = moyenne Z-score)

### TinyOL (`tinyol.ipynb`)
- Section 0 : `MonitoringAutoencoder(4→8→4)` — duck-typed pour `TinyOLOnlineTrainer`
  - `forward(x: [batch,4]) → (z: [batch,8], x_hat: [batch,4])`
  - Préentraîné sur 30% des normaux du train split, 50 epochs, MSE, Adam
- Backbone gelé, seule `OtOHead(input_dim=9)` est mise à jour (SGD)
- Contribution : permutation (espace 4D d'entrée) + normes colonnes de l'encodeur

## Résultats de sortie

```
experiments/exp_030_ewc_monitoring_single_task/results/feature_importance.json
experiments/exp_031_hdc_monitoring_single_task/results/feature_importance.json
experiments/exp_032_tinyol_monitoring_single_task/results/feature_importance.json

notebooks/figures/cl_evaluation/ewc/monitoring/single_task/
  ├── batch_metrics.png
  ├── prequential_rolling_acc.png
  ├── confusion_matrix.png
  ├── roc_curve.png
  ├── pca_feature_space.png
  ├── feature_importance_side_by_side.png
  └── feature_importance_comparison.png
[idem pour hdc/ et tinyol/]
```

## Statut

✅ `ewc.ipynb` — implémenté
✅ `hdc.ipynb` — implémenté
✅ `tinyol.ipynb` — implémenté (avec MonitoringAutoencoder inline)
⬜ Exécution à valider (données CSV requises)
