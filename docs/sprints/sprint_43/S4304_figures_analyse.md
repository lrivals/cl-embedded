# S4304 — Figures d'analyse du drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🟠 Haute — support visuel pour présentations/manuscrit ; rend le drift « lisible » et justifie le choix des datasets. |
| **Statut** | ✅ Implémenté — catalogue `src/figures/catalogs/drift_datasets.py` (registre S4201) → **17 PNG** `docs/figures/drift_datasets/` : timelines (pic mesuré ↔ vérité-terrain), shift avant/après (réutilise `plot_anomaly_score_distributions`), trajectoire PCA temporelle, heatmap JS fenêtre×fenêtre, comparatif inter-datasets. Labels FR, synthétique étiqueté « validation », 0 chiffre en dur (garde AST S4305). Figures JSON-only (timeline/comparatif) + raw via `DRIFT_LOADERS` (shift/PCA/heatmap, skip gracieux si `data/raw` absent). |
| **Durée estimée** | 4h |
| **Dépendances** | S4303 ✅ (`characterization.json`) · `src/evaluation/plots.py` ✅ (`plot_anomaly_score_distributions`) · `src/evaluation/eda_plots.py` ✅ · `src/figures/` 🟡 (registre S4201, si présent) |
| **Fichiers cibles** | `docs/figures/drift_datasets/*.png` (+ éventuel `src/figures/catalogs/drift_datasets.py` si registre S42 disponible) |
| **Références** | Sprint 42 S4201 (infrastructure figures régénérable) · style commun `src/figures/style.py` |

---

## Contexte

La caractérisation S4303 produit des séries numériques (JSON). Cette tâche les **rend visuelles** :
timelines de drift, décalages de distributions, trajectoire temporelle en espace réduit — figures FR
réutilisables en slide/manuscrit qui montrent *où* et *comment* chaque dataset dérive.

## Spec

Générer, **par dataset** (données chargées depuis `experiments/exp_S43_drift_char/<dataset>/`) :

1. **Timeline de drift** : statistiques glissantes (KS / PSI / MMD / Mahalanobis) en fonction du temps,
   avec les `drift_points` ground-truth marqués en verticales → montre la coïncidence pic ↔ point.
2. **Shift de distributions** : histogrammes/densités des features les plus dérivées (`features_most_
   drifted`) **avant vs après** un point de drift (réutilise `plot_anomaly_score_distributions`).
3. **Trajectoire PCA temporelle** : projection 2D des fenêtres successives colorées par le temps →
   visualise le déplacement du nuage (incremental = glissement continu ; sudden = saut ; recurring =
   retour).
4. **Heatmap distance-distribution × temps** : matrice [fenêtre_i × fenêtre_j] de distance de
   distribution (MMD/JS) → bloc-diagonale = régimes stables ; ruptures = drift.
5. **Figure comparative inter-datasets** : intensité/type de drift côte-à-côte (une ligne par dataset)
   → justifie la diversité du corpus retenu.

## Contraintes

- **Aucun chiffre en dur** : toute valeur chargée depuis les JSON S4303 ; cellules non mesurées →
  `« à mesurer »` ou masquées, jamais extrapolées.
- Style commun et **régénérable** : si le registre `src/figures/` (S4201) est disponible, enregistrer un
  catalogue `drift_datasets` ; sinon utiliser `eda_plots.py`/`plots.py` (backend Agg) sans dupliquer le
  style.
- Labels/titres en **français** ; `drift_points` explicitement légendés « vérité-terrain ».
- Pas de gaussienne synthétique présentée comme donnée réelle (règle héritée S42) — le synthétique n'est
  montré que comme **outil de validation** de la chaîne de mesure, étiqueté comme tel.

## Vérification

```bash
python scripts/generate_figures.py --catalog drift_datasets   # si registre S42 dispo
# sinon : exécution via le notebook S4305 (nbconvert)
ls docs/figures/drift_datasets/                               # timelines, shifts, pca, heatmaps, comparatif
```
- Chaque PNG se régénère de façon idempotente depuis les JSON S4303 (aucune donnée inline).
- Sur le synthétique, la timeline montre le pic aligné sur le point imposé (contrôle visuel de S4303).
