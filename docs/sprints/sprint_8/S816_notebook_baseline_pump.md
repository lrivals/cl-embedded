# S8-15 — Notebook Baseline Single-Task — Dataset 1 Pump Maintenance

> **Sprint 8 — tâche S8-16**  
> Durée estimée : 2h  
> Statut : 🔴 À faire  
> Fichier cible : `notebooks/cl_eval/baselines/pump_single_task.ipynb`

---

## Objectif

Créer le notebook de référence **hors-CL** pour le Dataset 1 (Large Industrial Pump Maintenance).  
Ce notebook agrège les résultats des 6 modèles (exp_036–041) entraînés sans découpage en tâches et les compare sur les métriques de classification standard.

Il répond notamment à une question critique : *"Si les scénarios CL sur pump donnent AA ≈ 0.50, est-ce dû à la contrainte CL ou aux données elles-mêmes ?"*

---

## Dépendances

- **Pré-requis obligatoire** : S8-14 (exp_036–041 exécutées)
- Fallback mock activé si une expérience est manquante
- Ce notebook est **parallèle** à `monitoring_single_task.ipynb` (Sprint 7 — S7-16)

---

## Chemin et organisation des figures

```
notebooks/cl_eval/baselines/pump_single_task.ipynb
notebooks/figures/cl_evaluation/baseline/pump/single_task/
├── comparison_bar_accuracy.png
├── roc_curves_all_models.png
├── scatter_ram_vs_accuracy.png
└── confusion_matrices_grid.png
```

---

## Structure du notebook (8 sections)

### Section 0 — Header Markdown

```markdown
# Baseline Single-Task — Dataset 1 : Large Industrial Pump Maintenance

| Champ | Valeur |
|-------|--------|
| Dataset | Large Industrial Pump Maintenance (Kaggle) |
| Scénario | `no_split` — toutes les données, une seule tâche |
| Features | 25 statistiques (mean, std, min, max, skew, kurt × 4 canaux + temporal_position) |
| Label | `maintenance_required` (binaire) |
| Modèles | EWC, HDC, TinyOL, KMeans, Mahalanobis, DBSCAN |
| Expériences | exp_036 – exp_041 |
| Date | {date} |

**Hypothèse** : Si AA ≈ 0.50 également ici (sans contrainte CL), le problème est structurel.
Voir FIXME(gap1) dans le code pour alternatives.
```

### Section 1 — Setup & imports

```python
EXP_DIRS = {
    "EWC":         "experiments/exp_036_ewc_pump_single_task",
    "HDC":         "experiments/exp_037_hdc_pump_single_task",
    "TinyOL":      "experiments/exp_038_tinyol_pump_single_task",
    "KMeans":      "experiments/exp_039_kmeans_pump_single_task",
    "Mahalanobis": "experiments/exp_040_mahalanobis_pump_single_task",
    "DBSCAN":      "experiments/exp_041_dbscan_pump_single_task",
}
FIGURES_DIR = Path("figures/cl_evaluation/baseline/pump/single_task")
SUPERVISED_MODELS = ["EWC", "HDC", "TinyOL"]
UNSUPERVISED_MODELS = ["KMeans", "Mahalanobis", "DBSCAN"]
```

### Section 2 — Chargement des résultats

- Lire `metrics_single_task.json` pour chaque modèle
- Fallback mock si le fichier est absent
- Construire un `DataFrame` avec colonnes :  
  `model | accuracy | f1 | auc_roc | ram_peak_bytes | inference_latency_ms | n_params`

### Section 3 — Tableau comparatif global

- DataFrame trié par `accuracy` décroissante
- Mettre en gras le meilleur score par colonne
- Ligne "best CL scenario" réservée (NaN) pour comparaison future avec S8-01 à S8-13

### Section 4 — Bar plot accuracy comparée

- 6 barres verticales, couleurs par famille (supervisé bleu / non-supervisé vert)
- Ligne horizontale rouge pointillée à `accuracy = 0.5` (random baseline)
- Annotation de la valeur au-dessus de chaque barre
- **Annotation spéciale** : si toutes les accuracies ≈ 0.50, ajouter une note "*Dataset difficilement séparable avec ces features*"
- Sauvegarde : `comparison_bar_accuracy.png`

### Section 5 — Courbes ROC superposées

- 6 courbes sur un seul axes
- AUC dans la légende : `EWC (AUC=0.XX)`
- Ligne diagonale grise
- Sauvegarde : `roc_curves_all_models.png`

### Section 6 — Scatter RAM peak vs. accuracy (Gap 2)

- Axe X : `ram_peak_bytes` (en Ko), Axe Y : `accuracy`
- Zone de contrainte STM32 : rectangle rouge `x ≤ 64 Ko`
- Chaque modèle annoté
- Sauvegarde : `scatter_ram_vs_accuracy.png`

### Section 7 — Confusion matrices (grille 2×3)

- Un subplot par modèle
- Classes : `[0 = normal, 1 = maintenance_required]`
- Sauvegarde : `confusion_matrices_grid.png`

### Section 8 — Discussion Markdown

Points à couvrir :
1. **Diagnostic structurel** : AA single-task ≈ 0.50 ou > 0.50 ?
   - Si > 0.50 → la contrainte CL dégrade la performance → normal
   - Si ≈ 0.50 → problème structurel dans les features ou le dataset → FIXME(gap1)
2. **Comparaison Dataset 1 vs. Dataset 2** : le pump est-il plus difficile que le monitoring ?
3. **Contrainte embarquée** : quels modèles rentrent dans ≤ 64 Ko RAM ?
4. **Recommandation** : si Dataset 1 est structurellement difficile, explorer FEMTO-Bearing ou revoir les features
5. **Pont vers les notebooks CL** : *"Ces scores fixent le plafond. Les notebooks S8-01 à S8-13 montrent l'impact des splits temporels et par pump_id."*

---

## Critères de succès

- [ ] Notebook exécuté entièrement sans erreur (avec fallback mock ou résultats réels)
- [ ] 4 figures sauvegardées dans `notebooks/figures/cl_evaluation/baseline/pump/single_task/`
- [ ] Tableau comparatif lisible et trié
- [ ] Section discussion tranche sur le diagnostic structurel (dataset vs. CL)
- [ ] Scatter RAM vs. accuracy affiche la zone STM32 ≤ 64 Ko

---

## Notes

- Ce notebook est le pendant de `monitoring_single_task.ipynb` (S7-16) pour Dataset 1
- La comparaison croisée Dataset 1 / Dataset 2 peut être esquissée ici mais sera formalisée dans le manuscrit
- Ne pas oublier `set_seed(42)` en début de notebook pour reproductibilité
