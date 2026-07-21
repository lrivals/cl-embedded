# S4308 — EDA exhaustive : Electricity (ELEC2)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🟡 Moyenne — benchmark concept-drift classique (drift graduel, sans vérité-terrain ponctuelle) ; EDA de référence pour un drift *non annoté*. |
| **Statut** | ✅ Implémenté — `eda_electricity.ipynb` généré et exécuté (nbconvert OK, 9 figures inline, 0 erreur ; aucune verticale vérité-terrain). |
| **Durée estimée** | 4h |
| **Dépendances** | S4302 ✅ (`src/data/electricity_dataset.py`, `configs/electricity_drift_config.yaml`) · S4303 ✅ (`experiments/exp_S43_drift_char/electricity/characterization.json`, `alignment_score=null`) · `src/evaluation/eda_plots.py` ✅ · `src/evaluation/feature_space_plots.py` ✅ · `src/evaluation/plots.py` ✅ |
| **Fichiers cibles** | `notebooks/cl_eval/drift_datasets/eda_electricity.ipynb` |
| **Références** | `notebooks/01_data_exploration.ipynb` (EDA temporelle par fenêtres) · `notebooks/cl_eval/drift_datasets/analysis.ipynb` · `docs/context/drift_datasets.md` |

---

## Contexte

Le dataset **Electricity / ELEC2** (marché NSW, ~7 features, label binaire prix ↑/↓) est le benchmark
**concept-drift** classique du corpus. Particularité : **pas de point de drift ponctuel** →
`drift_points=None` et `alignment_score=null` (honnête, S4303). L'EDA doit donc illustrer un drift
**graduel/continu** (pas de verticale « vérité-terrain »), en miroir de l'EDA temporelle par fenêtres de
`01_data_exploration.ipynb`.

## Spec

Notebook `notebooks/cl_eval/drift_datasets/eda_electricity.ipynb` (FR, backend `Agg`, `set_seed(42)`),
chargé via `src/data/electricity_dataset.py::load("configs/electricity_drift_config.yaml")` →
`DriftDataset` (`X`, `y` = prix ↑/↓, `drift_points=None`, `feature_names`, `metadata`). Résumé lu depuis
`experiments/exp_S43_drift_char/electricity/characterization.json`.

Sections :

1. **Chargement & vue d'ensemble** : formes, période couverte, liste des 7 features, résumé du
   `characterization.json` (**note honnête** : `drift_points=None`, `alignment_score=null`).
2. **Stats descriptives** globales et par fenêtre temporelle.
3. **Distribution du label** : équilibre prix ↑/↓ global, puis **évolution du taux de label dans le
   temps** (dérive de la cible = concept drift).
4. **Évolution temporelle des features** : séries + **stats glissantes** (moyenne/écart glissants par
   fenêtre) → visualise le drift graduel ; **aucune verticale de vérité-terrain** (pas de point ponctuel).
5. **Distributions de features** : `plot_histograms_by_label` / `plot_boxplots_by_label` /
   `plot_kde_by_label` par **classe** (prix ↑/↓).
6. **Corrélations** : heatmap des 7 features.
7. **Projection 2D par fenêtres temporelles** : `fit_pca2d` / `fit_tsne2d` + `plot_feature_space_2d`,
   coloré **par fenêtre temporelle** (déplacement graduel du nuage) puis **par classe**.
8. **Résumé pour le Sprint 44** : nature du drift (graduel, non annoté) → conséquences pour l'évaluation
   d'un détecteur (délai/fausses alarmes non mesurables sans GT ponctuelle ; le synthétique S4303 sert de
   calibration de la chaîne de mesure) ; dimension d'entrée (note Gap 2/3).

## Contraintes

- **Aucun chiffre en dur** : tout sort d'une exécution (loader + `characterization.json`).
- **Honnêteté ground-truth** : `drift_points=None` — ne jamais tracer de « point de drift vérité-terrain » ;
  le drift n'est montré que comme tendance graduelle. Rappeler que l'alignement quantitatif est reporté au
  synthétique (S4303).
- Réutiliser **exclusivement** les helpers `src/evaluation/eda_plots.py`, `feature_space_plots.py`,
  `plots.py` — ne pas réimplémenter de logique de plot.
- **Normalisation figée** (segment/fenêtre initial, cohérence `freeze_zscore` S4302).
- Labels/titres en **français** ; distinguer l'axe **classe (prix ↑/↓)** de l'axe **temps (drift graduel)**.
- **Skip gracieux** si `data/raw/The Elec2 Dataset` absent (`.gitignore`).
- Notebook rangé sous `notebooks/cl_eval/drift_datasets/`.

## Vérification

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/cl_eval/drift_datasets/eda_electricity.ipynb
```
- Exécution nbconvert sans erreur (ou skip gracieux propre si `data/raw` absent).
- Aucune verticale « vérité-terrain » (cohérent `drift_points=None`) ; le drift graduel est visible via
  stats glissantes + projection par fenêtres.
- Aucune valeur recopiée dans le texte : tout provient du loader / du JSON S4303.
