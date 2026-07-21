# S4306 — EDA exhaustive : Gas Sensor Array Drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🟠 Haute — dataset ⭐ du corpus (drift capteur réel, dual-usage 6 gaz) ; l'EDA feature-level justifie son choix et prépare les détecteurs S44. |
| **Statut** | ✅ Implémenté — `eda_gas_sensor.ipynb` généré et exécuté (nbconvert OK, 11 figures inline, 0 erreur). |
| **Durée estimée** | 4h |
| **Dépendances** | S4302 ✅ (`src/data/gas_sensor_drift_dataset.py`, `configs/gas_sensor_drift_config.yaml`) · S4303 ✅ (`experiments/exp_S43_drift_char/gas_sensor_drift/characterization.json`) · `src/evaluation/eda_plots.py` ✅ · `src/evaluation/feature_space_plots.py` ✅ · `src/evaluation/plots.py` ✅ |
| **Fichiers cibles** | `notebooks/cl_eval/drift_datasets/eda_gas_sensor.ipynb` |
| **Références** | `notebooks/eda_paderborn.ipynb` (style feature-EDA capteur) · `notebooks/cl_eval/drift_datasets/analysis.ipynb` (galerie drift S4305) · `docs/context/drift_datasets.md` (fiche dataset) |

---

## Contexte

Le notebook de synthèse `analysis.ipynb` (S4305) est une **galerie orientée drift** (timeline / shift /
PCA / heatmap chargés des JSON S4303) — il ne descend **pas** au niveau des features. Pour le dataset
**Gas Sensor Array Drift** (16 capteurs × 8 features = 128 feat., 6 gaz, **10 batches temporels** sur
36 mois, drift incrémental réel), on veut une **EDA exhaustive feature-level** qui rende lisibles à la
fois la **structure de classes** (6 gaz) et la **dérive temporelle** (batches), en miroir du style
`eda_paderborn.ipynb`. C'est le dataset ⭐ du corpus : son EDA sert de vitrine et prépare le choix des
détecteurs S44.

## Spec

Notebook `notebooks/cl_eval/drift_datasets/eda_gas_sensor.ipynb` (FR, backend `Agg`, `set_seed(42)`),
chargé via `src/data/gas_sensor_drift_dataset.py::load("configs/gas_sensor_drift_config.yaml")` →
`DriftDataset` (`X`, `y`, `segments` = batches, `feature_names`, `drift_type="incremental"`, `metadata`).
Résumé numérique lu depuis `experiments/exp_S43_drift_char/gas_sensor_drift/characterization.json`.

Sections (miroir `eda_paderborn.ipynb`) :

1. **Chargement & vue d'ensemble** : formes, nombre de batches/gaz, mapping capteur→features (16×8),
   résumé du `characterization.json` (drift confirmé, `features_most_drifted`, pic Mahalanobis).
2. **Stats descriptives** par batch et par gaz (moyennes/écarts, valeurs manquantes).
3. **Distribution du label** : effectifs par gaz (`plot_label_distribution`) et effectifs par batch.
4. **Distributions de features** : `plot_histograms_by_label` / `plot_boxplots_by_label` /
   `plot_violin_by_label` / `plot_kde_by_label` colorées **par gaz** (structure de classe) **et** par
   batch (dérive), sur un sous-ensemble représentatif de capteurs (les `features_most_drifted` en tête).
5. **Trajectoires de réponse capteur** : évolution des réponses des 16 capteurs **sur les 10 batches**
   (`plot_temporal_by_label` / séries par segment) → visualise la dérive incrémentale.
6. **Corrélations** : heatmap de corrélation inter-capteurs (redondance / familles de capteurs).
7. **Projection 2D drift** : `fit_tsne2d` / `fit_pca2d` + `plot_feature_space_2d`, coloré **par batch**
   (déplacement temporel du nuage) puis **par gaz** (séparabilité des classes).
8. **Magnitude de dérive par batch** : barplot de distance de distribution batch→batch (relie au pic
   Mahalanobis batch1→2 documenté en S4303).
9. **Résumé pour le Sprint 44** : implications pour un détecteur de drift (fenêtre, features informatives,
   dimension d'entrée → note Gap 2/3).

## Contraintes

- **Aucun chiffre en dur** : tout sort d'une exécution (loader + `characterization.json`) ; aucune valeur
  recopiée dans le texte.
- Réutiliser **exclusivement** les helpers existants `src/evaluation/eda_plots.py`,
  `feature_space_plots.py`, `plots.py` — ne pas réimplémenter de logique de plot.
- **Sous-échantillonnage documenté** (≈13910 échantillons, 128 features) pour rendre t-SNE/pairplot
  tractables ; graine fixée, sélection reproductible.
- Labels/titres en **français** ; distinguer explicitement l'axe **classe (gaz)** de l'axe **drift (batch)**.
- **Skip gracieux** si `data/raw/Gas Sensor Array Drift Dataset` absent (données en `.gitignore`) :
  cellule d'avertissement, pas d'échec dur.
- Notebook rangé sous `notebooks/cl_eval/drift_datasets/` (règle CLAUDE.md : notebooks dans `notebooks/`).

## Vérification

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/cl_eval/drift_datasets/eda_gas_sensor.ipynb
```
- Exécution nbconvert sans erreur (ou skip gracieux propre si `data/raw` absent).
- Chaque figure se régénère de façon idempotente depuis le loader + le JSON S4303 (aucune donnée inline).
- Le notebook distingue visuellement la dérive (batch) de la structure de classe (gaz).
