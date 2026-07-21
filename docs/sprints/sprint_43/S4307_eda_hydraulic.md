# S4307 — EDA exhaustive : Condition Monitoring Hydraulique

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🟠 Haute — dataset dual-usage « fault-primary, drift-secondary » ; l'EDA par condition cooler prépare l'évaluation tandem drift+faute (S44). |
| **Statut** | ✅ Implémenté — `eda_hydraulic.ipynb` généré et exécuté (nbconvert OK, 9 figures inline, 0 erreur). |
| **Durée estimée** | 4h |
| **Dépendances** | S4302 ✅ (`src/data/hydraulic_dataset.py`, `configs/hydraulic_drift_config.yaml`) · S4303 ✅ (`experiments/exp_S43_drift_char/hydraulic/characterization.json`) · `src/evaluation/eda_plots.py` ✅ · `src/evaluation/feature_space_plots.py` ✅ · `src/evaluation/plots.py` ✅ |
| **Fichiers cibles** | `notebooks/cl_eval/drift_datasets/eda_hydraulic.ipynb` |
| **Références** | `notebooks/eda_paderborn.ipynb` · `notebooks/01_data_exploration.ipynb` (EDA par groupe/condition) · `docs/context/drift_datasets.md` |

---

## Contexte

Le dataset **Condition Monitoring of Hydraulic Systems** (ZeMA, ~17 features de cycle, cycles **segmentés
par condition cooler**, labels de faute) a été retenu en substitution d'INSECTS : c'est un dataset
**fault-primary, drift-secondary**. Le notebook de synthèse S4305 ne l'explore qu'au niveau agrégé.
On veut une **EDA exhaustive feature-level** qui rende lisibles (1) la **structure de faute** (labels) et
(2) le **drift de régime segmenté par condition** (`segments`), en miroir des EDA par groupe existantes
(`01_data_exploration.ipynb`).

## Spec

Notebook `notebooks/cl_eval/drift_datasets/eda_hydraulic.ipynb` (FR, backend `Agg`, `set_seed(42)`),
chargé via `src/data/hydraulic_dataset.py::load("configs/hydraulic_drift_config.yaml")` → `DriftDataset`
(`X`, `y` = label de faute, `segments` = conditions cooler, `feature_names`, `metadata`). Résumé lu depuis
`experiments/exp_S43_drift_char/hydraulic/characterization.json`.

Sections :

1. **Chargement & vue d'ensemble** : formes, nombre de conditions/segments, liste des 17 capteurs de
   cycle, résumé du `characterization.json` (`features_most_drifted`, drift confirmé).
2. **Stats descriptives** par condition et par label de faute.
3. **Distribution du label** : effectifs de faute (`plot_label_distribution`) + **taux de faute par
   segment/condition** (heatmap façon `plot_fault_rate_heatmap_*`).
4. **Distributions de features** : `plot_boxplots_by_label` / `plot_violin_by_label` /
   `plot_kde_by_label` par **label de faute**, puis variantes `_by_group_and_label` par **condition
   cooler** → contraste faute vs régime.
5. **Corrélations** : heatmap des 17 capteurs (familles capteurs pression/température/débit/vibration).
6. **Projection 2D** : `fit_pca2d` / `fit_tsne2d` + `plot_feature_space_2d`, coloré **par condition**
   (drift de régime segmenté) puis **par label de faute** (séparabilité).
7. **Ranking d'importance** : mutual-information / variance des features vis-à-vis du label (barplot top-k).
8. **Résumé pour le Sprint 44** : quelles features portent la faute vs le drift de régime ; implications
   pour le tandem détecteur-drift + détecteur-faute ; dimension d'entrée (note Gap 2/3).

## Contraintes

- **Aucun chiffre en dur** : tout sort d'une exécution (loader + `characterization.json`).
- Réutiliser **exclusivement** les helpers `src/evaluation/eda_plots.py` (dont variantes
  `_by_group_and_label` et `plot_fault_rate_heatmap_*`), `feature_space_plots.py`, `plots.py`.
- **Normalisation figée sur le segment 0** (cohérence avec `freeze_zscore` S4302) — ne pas re-fitter par
  segment.
- Labels/titres en **français** ; distinguer l'axe **faute (label)** de l'axe **drift (condition cooler)**.
- **Skip gracieux** si `data/raw/Condition Monitoring of Hydraulic Systems` absent (`.gitignore`).
- Notebook rangé sous `notebooks/cl_eval/drift_datasets/`.

## Vérification

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/cl_eval/drift_datasets/eda_hydraulic.ipynb
```
- Exécution nbconvert sans erreur (ou skip gracieux propre si `data/raw` absent).
- Le taux de faute par condition et la projection 2D montrent conjointement faute + drift de régime.
- Aucune valeur recopiée dans le texte : tout provient du loader / du JSON S4303.
