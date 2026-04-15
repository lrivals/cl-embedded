# S7-16 — Notebook Baseline Single-Task — Dataset 2 Monitoring

> **Sprint 7**  
> Durée estimée : 2h  
> Statut : 🟢 Terminé  
> Fichier cible : `notebooks/cl_eval/baselines/monitoring_single_task.ipynb`

---

## Objectif

Créer le notebook de référence **hors-CL** pour le Dataset 2 (Equipment Monitoring).  
Ce notebook agrège les résultats des 6 modèles (exp_030–035) entraînés sans découpage en tâches et les compare sur les métriques de classification standard.

Il sert de **point d'ancrage** pour tous les notebooks CL de Sprint 7 : la discussion finale de chaque notebook CL compare ses résultats à cette baseline.

---

## Dépendances

- **Pré-requis obligatoire** : S7-15 (exp_030–035 exécutées)
- Fallback mock activé si une expérience est manquante (métriques fictives pour développement)

---

## Chemin et organisation des figures

```
notebooks/cl_eval/baselines/monitoring_single_task.ipynb
notebooks/figures/cl_evaluation/baseline/monitoring/single_task/
├── comparison_bar_accuracy.png
├── roc_curves_all_models.png
├── scatter_ram_vs_accuracy.png
└── confusion_matrices_grid.png
```

---

## Structure du notebook (8 sections)

### Section 0 — Header Markdown

```markdown
# Baseline Single-Task — Dataset 2 : Equipment Monitoring

| Champ | Valeur |
|-------|--------|
| Dataset | Equipment Monitoring (Kaggle) |
| Scénario | `no_split` — toutes les données, une seule tâche |
| Modèles | EWC, HDC, TinyOL, KMeans, Mahalanobis, DBSCAN |
| Expériences | exp_030 – exp_035 |
| Date | {date} |

**Objectif** : Établir la performance maximale de chaque modèle en l'absence de toute contrainte CL.
Cet notebook est la référence absolue pour mesurer le coût du continual learning dans les notebooks suivants.
```

### Section 1 — Setup & imports

```python
EXP_DIRS = {
    "EWC":         "experiments/exp_030_ewc_monitoring_single_task",
    "HDC":         "experiments/exp_031_hdc_monitoring_single_task",
    "TinyOL":      "experiments/exp_032_tinyol_monitoring_single_task",
    "KMeans":      "experiments/exp_033_kmeans_monitoring_single_task",
    "Mahalanobis": "experiments/exp_034_mahalanobis_monitoring_single_task",
    "DBSCAN":      "experiments/exp_035_dbscan_monitoring_single_task",
}
FIGURES_DIR = Path("figures/cl_evaluation/baseline/monitoring/single_task")
SUPERVISED_MODELS = ["EWC", "HDC", "TinyOL"]
UNSUPERVISED_MODELS = ["KMeans", "Mahalanobis", "DBSCAN"]
```

### Section 2 — Chargement des résultats

- Lire `metrics_single_task.json` pour chaque modèle
- Fallback mock si le fichier est absent (valeurs neutres pour développement du notebook)
- Construire un `DataFrame` unique avec colonnes :  
  `model | accuracy | f1 | auc_roc | ram_peak_bytes | inference_latency_ms | n_params`

### Section 3 — Tableau comparatif global

- Afficher le DataFrame trié par `accuracy` décroissante
- Mettre en gras le meilleur score par colonne
- Ajouter une ligne "best CL scenario" pour pont futur vers les notebooks CL (valeur NaN au départ)

### Section 4 — Bar plot accuracy comparée

- 6 barres verticales, couleurs distinctes par famille :
  - Supervisé (EWC, HDC, TinyOL) : palette bleue
  - Non-supervisé (KMeans, Mahalanobis, DBSCAN) : palette verte
- Ligne horizontale rouge pointillée à `accuracy = 0.5` (random baseline)
- Annotation de la valeur au-dessus de chaque barre
- Sauvegarde : `comparison_bar_accuracy.png`

### Section 5 — Courbes ROC superposées

- 6 courbes ROC sur un seul axes (une couleur par modèle)
- AUC annotée dans la légende : `EWC (AUC=0.XX)`
- Ligne diagonale grise (random classifier)
- Sauvegarde : `roc_curves_all_models.png`

### Section 6 — Scatter RAM peak vs. accuracy (Gap 2)

- Axe X : `ram_peak_bytes` (en Ko)
- Axe Y : `accuracy`
- Chaque modèle = un point annoté
- Zone de contrainte STM32 : rectangle rouge `x ≤ 64 Ko`
- Titre : *"Trade-off embarqué : RAM vs. performance (baseline hors-CL)"*
- Sauvegarde : `scatter_ram_vs_accuracy.png`

### Section 7 — Confusion matrices (grille 2×3)

- Un subplot par modèle, ordre : EWC / HDC / TinyOL / KMeans / Mahalanobis / DBSCAN
- Classes : `[0 = normal, 1 = faulty]`
- Sauvegarde : `confusion_matrices_grid.png`

### Section 8 — Discussion Markdown

Points à couvrir :
1. **Performance sans CL** : quel modèle performe le mieux sur l'ensemble des données ?
2. **Écart supervisé/non-supervisé** : le label `faulty` apporte-t-il un gain significatif ?
3. **Contrainte embarquée** : quels modèles rentrent dans ≤ 64 Ko RAM ?
4. **Pont vers les notebooks CL** : *"Ces scores sont la référence. Les notebooks S7-01 à S7-12 montrent ce qu'il se passe quand on introduit des tâches séquentielles."*
5. **Hypothèse à vérifier** : l'accuracy CL devrait être inférieure à cette baseline — si ce n'est pas le cas, le découpage en tâches est peut-être trop facile.

---

## Critères de succès

- [x] Notebook exécuté entièrement sans erreur (avec fallback mock ou résultats réels)
- [x] 4 figures sauvegardées dans `notebooks/figures/cl_evaluation/baseline/monitoring/single_task/`
- [x] Tableau comparatif lisible et trié
- [x] Scatter RAM vs. accuracy affiche la zone STM32 ≤ 64 Ko
- [x] Section discussion rédigée avec hypothèses CL formulées

---

## Notes

- Ce notebook n'utilise **pas** `acc_matrix.npy` ni les métriques AF/BWT — ce sont des métriques CL inapplicables ici
- La section 3 réserve une ligne "best CL scenario" vide qui sera complétée manuellement après les notebooks S7-01 à S7-14
