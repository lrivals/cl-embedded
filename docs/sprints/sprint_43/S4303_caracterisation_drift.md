# S4303 — Caractérisation & quantification du drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🔴 Critique — établit la nature/l'intensité du drift de chaque dataset et **valide la ground-truth** qui servira de référence aux détecteurs (S44). |
| **Statut** | 📝 Doc — spec ; implémentation à venir. |
| **Durée estimée** | 7h |
| **Dépendances** | S4302 ✅ (loaders + `drift_points`) · `src/models/unsupervised/mahalanobis_detector.py` ✅ · `src/evaluation/plots.py` ✅ · `scipy.stats` (KS) |
| **Fichiers cibles** | `scripts/characterize_drift.py`, `experiments/exp_S43_drift_char/<dataset>/characterization.json` |
| **Références** | `src/evaluation/drift_detector.py` (baseline score-based, ne pas dupliquer) · Gama et al. 2014 (taxonomie drift) |

---

## Contexte

Avant d'évaluer des **détecteurs** de drift (S44), il faut **caractériser** le drift présent dans chaque
dataset : de quel type est-il, à quelle intensité, et les `drift_points` annoncés par le loader
correspondent-ils à une dérive **mesurable** de la distribution ? Cette tâche produit une description
quantitative, indépendante de tout détecteur, qui sert de **vérité-terrain de référence** et de
justification (« ce dataset exhibe bien un drift de tel type à tel endroit »).

## Spec

`scripts/characterize_drift.py --dataset <nom>` charge le `DriftDataset` (S4302) et calcule, **sans
détecteur** (analyse offline exhaustive) :

1. **Typage du drift** (par segment) : confirmer `drift_type` du loader par la forme de la dérive —
   sudden (saut ponctuel), gradual (mélange progressif), incremental (glissement continu), recurring
   (retour d'une distribution antérieure).
2. **Quantification de la dérive glissante** (fenêtre de référence vs fenêtre courante) :
   - **KS deux-échantillons** par feature (`scipy.stats.ks_2samp`) → statistique + p-value dans le temps.
   - **PSI** (Population Stability Index) et **Jensen-Shannon** sur histogrammes par feature.
   - **MMD** (Maximum Mean Discrepancy, noyau RBF) multivarié entre fenêtres.
   - Dérive de **moyenne/variance** par feature (drift de premier/second ordre).
   - **Distance de Mahalanobis** au segment initial (réutilise `MahalanobisDetector` calibré sur le
     segment 0) + **résidu de reconstruction PCA** (composantes du segment 0).
3. **Validation vs ground-truth** : superposer les courbes de statistique glissante aux `drift_points`
   annoncés → mesurer si les pics/franchissements coïncident (alignement pic ↔ point). Rapporter un
   **score d'alignement** (ex. distance médiane pic-mesuré ↔ point-annoncé) — diagnostic de qualité de la
   vérité-terrain.
4. **Sortie** `experiments/exp_S43_drift_char/<dataset>/characterization.json` :
   - `drift_type_confirmed`, `n_features`, `n_samples`, `drift_points` (rappel loader),
   - séries temporelles de chaque statistique (KS/PSI/JS/MMD/Maha/PCA),
   - `alignment_score`, `features_most_drifted` (top-k par intensité de dérive),
   - `metadata` (source, licence, config_snapshot).

## Contraintes

- **Aucun chiffre en dur** : toutes les valeurs sortent du loader/calcul, écrites en JSON, jamais dans la
  doc. Tant que non exécuté → pas de JSON (règle projet).
- **Ne pas réimplémenter** la détection en ligne : `SlidingWindowDriftDetector` reste la baseline
  *détecteur* (S44) ; ici on fait une **analyse offline** descriptive (accès à tout le flux).
- Réutiliser `MahalanobisDetector` (ne pas réécrire de distance) et `plots.py` pour les visuels (délégué
  à S4304).
- Distinguer explicitement les datasets à **ground-truth ponctuelle** (alignement calculable) de ceux à
  ground-truth **structurelle/absente** (Electricity/NOAA → `alignment_score=null`, honnête).

## Vérification

```bash
python scripts/characterize_drift.py --dataset gas_sensor_drift   # → exp_S43_drift_char/gas_sensor_drift/
python scripts/characterize_drift.py --dataset synthetic          # vérité-terrain exacte → alignment_score ~ 0
```
- Sur le **synthétique** (points de drift exacts, S4302), le pic de statistique KS/MMD doit coïncider
  avec le point imposé → `alignment_score` proche de 0 (validation de la chaîne de mesure).
- Sur Gas Sensor Drift, la distance de Mahalanobis au batch 0 doit **croître** de batch en batch
  (dérive incrémentale de capteur documentée).
