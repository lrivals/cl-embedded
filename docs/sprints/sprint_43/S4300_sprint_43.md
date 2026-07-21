# Sprint 43 — Recherche & analyse de datasets pour la détection de drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 43 |
| **Semaine** | 20 – 26 juillet 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🔴 Critique — socle des Sprints 44 (modèles PC) et 45 (portage board) ; sans dataset à drift labellisé, aucune évaluation de détecteur de drift n'est mesurable. |
| **Durée estimée totale** | ~38h (recherche/sélection ~5h · acquisition/loaders ~7h · caractérisation ~7h · figures ~4h · notebook+tests+docs ~3h · EDA exhaustive par dataset ~12h) |
| **Dépendances** | `src/evaluation/plots.py` ✅ (`plot_anomaly_score_distributions`) · `src/evaluation/eda_plots.py` ✅ · `src/evaluation/drift_detector.py` ✅ (baseline de référence) · `src/data/pump_dataset.py` ✅ (style de loader, normalisation figée) · Sprint 38 ✅ (nœud drift ≠ faute) |

## Contexte et motivation

Le projet distingue déjà **FAULT vs DRIFT** à bord (Sprint 38) via `SlidingWindowDriftDetector` (double
seuil sur score Mahalanobis + persistance temporelle). Mais cette brique est **la seule** infrastructure
de drift du dépôt, et surtout elle a été évaluée sur des datasets projet (**Monitoring, Pronostia**)
**sans points de drift labellisés** — le « drift » y est déduit du scénario CL (changement d'équipement,
arrivée temporelle de la faute), jamais annoté échantillon par échantillon.

Pour **rechercher, construire et évaluer** de vrais modèles de détection de drift (Sprints 44/45), il
faut d'abord une **vérité-terrain de drift** : des datasets où l'on sait *quand* la distribution change,
afin de mesurer précisément le **délai de détection**, le **taux de fausses alarmes** et le **taux de
manqués**. Aucun dataset du projet ne l'offre aujourd'hui.

Ce sprint **recherche des datasets externes** (sur internet), en **propose plusieurs**, les acquiert, les
analyse et **caractérise/quantifie le drift**. Priorité au drift ; on privilégie les datasets **dual-usage
drift+faute** afin que le même corpus puisse ensuite servir à évaluer, en tandem, un détecteur de drift
*et* un détecteur de faute (sprint futur, `docs/context/drift_fault_tandem.md`).

## Décisions de cadrage (utilisateur, 7 juillet 2026)

- **Datasets externes trouvés sur internet** — proposer **plusieurs** candidats, priorité à la détection
  de drift **pour l'instant**.
- **Dual-usage privilégié** : de préférence des datasets utilisables **aussi** pour la détection de faute,
  pour évaluer les deux familles de modèles (préparation du tandem).
- **Ne pas réutiliser les datasets projet actuels** (Pump/Battery/CMAPSS/Pronostia/Monitoring/CWRU/
  Paderborn) **sauf s'ils portent des éléments labellisés pour la détection de drift** — ce qui n'est pas
  le cas → nouveaux datasets requis.
- **Langue** : français (textes et labels), cible présentations + manuscrit.

## Liens triple gap

- **Gap 1 (données industrielles réelles)** : les datasets retenus doivent, idéalement, être des séries
  temporelles industrielles/capteurs réels avec drift documenté — renforce directement Gap 1 sur l'axe
  *drift* (jusqu'ici couvert par CWRU/Pronostia/CMAPSS/Paderborn côté *faute* uniquement).
- **Gaps 2/3 (latence/RAM)** : les datasets conditionnent le format d'entrée (dimension, fenêtre) des
  détecteurs → impact direct sur l'état mémoire porté board en S45. Documenter la dimension de features.

## Tâches

### Bloc A — Recherche & acquisition

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4301 | **Recherche & sélection** : short-list de datasets externes (type de drift, points de drift ground-truth, licence, dual-usage faute) → doc de référence unique | 🔴 | `docs/context/drift_datasets.md` | 📝 Doc |
| S4302 | **Acquisition & loaders** : scripts de téléchargement + loaders alignés sur le style projet + configs, ground-truth `drift_points`/`drift_type` exposée | 🔴 | `scripts/download_drift_datasets.py`, `src/data/*_dataset.py`, `configs/*_drift_config.yaml` | 📝 Doc |

### Bloc B — Analyse & caractérisation

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4303 | **Caractérisation & quantification du drift** : typer (sudden/gradual/incremental/recurring), quantifier la dérive glissante (KS/MMD/PSI, moyenne/variance, PCA/Maha), valider vs ground-truth | 🔴 | `scripts/characterize_drift.py` → `experiments/exp_S43_drift_char/` | 📝 Doc |
| S4304 | **Figures d'analyse** : timelines de drift, shift de distributions, trajectoire PCA temporelle, heatmap distance × temps | 🟠 | `docs/figures/drift_datasets/` | 📝 Doc |

### Bloc C — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4305 | Notebook EDA + tests (loaders, ground-truth, normalisation figée, 0 chiffre en dur) + roadmap + `CLAUDE.md` + `graphify_sprint_update` | 🟡 | `notebooks/cl_eval/drift_datasets/analysis.ipynb`, `tests/test_drift_datasets.py` | 📝 Doc |

### Bloc D — EDA exhaustive par dataset

> Une EDA **feature-level exhaustive** par dataset réel (structure de classe + dérive), complémentaire de
> la galerie de synthèse `analysis.ipynb` (S4305). Réutilise les loaders S4302, les JSON S4303 et les
> helpers `src/evaluation/eda_plots.py` / `feature_space_plots.py` / `plots.py`. Le synthétique est exclu
> (outil de validation, pas un dataset à explorer).

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4306 | **EDA Gas Sensor Array Drift** : distributions par gaz **et** par batch, trajectoires capteurs sur 10 batches, corrélations inter-capteurs, t-SNE/PCA (batch/gaz), magnitude de dérive par batch | 🟠 | `notebooks/cl_eval/drift_datasets/eda_gas_sensor.ipynb` | 📝 Doc |
| S4307 | **EDA Condition Monitoring Hydraulique** : distributions par faute et par condition cooler, taux de faute par segment, corrélations 17 capteurs, PCA/t-SNE (condition/faute), ranking d'importance | 🟠 | `notebooks/cl_eval/drift_datasets/eda_hydraulic.ipynb` | 📝 Doc |
| S4308 | **EDA Electricity (ELEC2)** : évolution temporelle + stats glissantes (drift graduel, `drift_points=None`), taux de label dans le temps, distributions par classe, PCA/t-SNE par fenêtres | 🟡 | `notebooks/cl_eval/drift_datasets/eda_electricity.ipynb` | 📝 Doc |

## Ordre d'exécution recommandé

```
S4301 (recherche/sélection) ──► S4302 (acquisition/loaders) ──┬─► S4303 (caractérisation) ─► S4304 (figures) ─┐
                                                              └────────────────────────────────────────────────► S4305 (notebook+tests+clôture)

S4302 (loaders) + S4303 (JSON) ──► S4306 / S4307 / S4308 (Bloc D — EDA exhaustive par dataset, parallélisables)
```

S4301 est purement documentaire et peut démarrer immédiatement. S4302 dépend du choix final S4301.
S4303/S4304 consomment les loaders S4302. S4305 clôt le sprint. Le **Bloc D** (S4306–S4308) est
indépendant entre tâches et ne requiert que les loaders S4302 et les JSON S4303 (parallélisable).

## Sources de données (candidats externes, à valider en S4301)

| Dataset | Type de drift | Ground-truth drift | Dual-usage faute | Source |
|---------|---------------|:------------------:|:----------------:|--------|
| **UCI Gas Sensor Array Drift** | incrémental (36 mois) | ✅ batches temporels | ✅ classification 6 gaz | UCI ML Repository |
| **USP INSECTS** | abrupt / gradual / incremental / recurring | ✅ points documentés | ✅ classification d'espèces | USP data stream repo |
| **Electricity / ELEC2** | concept drift | ⚠️ non ponctuel (supervisé) | ➖ | benchmark concept-drift classique |
| **NOAA Weather** | concept drift saisonnier | ⚠️ non ponctuel | ➖ | benchmark concept-drift |
| **Générateurs synthétiques** (SEA, Hyperplane, Agrawal) | contrôlé | ✅ points exacts | ➖ | `river` / scikit-multiflow |
| **UCI Condition Monitoring hydraulique / SECOM** | secondaire | ⚠️ | ✅ faute primaire | UCI ML Repository |

## Livrables

1. `docs/sprints/sprint_43/` (ce dossier) — specs S4300–S4305.
2. `docs/context/drift_datasets.md` — référence textuelle unique (fiche par dataset).
3. `scripts/download_drift_datasets.py` + `src/data/*_dataset.py` + `configs/*_drift_config.yaml`.
4. `experiments/exp_S43_drift_char/` — caractérisation quantifiée (aucun chiffre en dur, tout depuis loader).
5. `docs/figures/drift_datasets/*.png` — figures d'analyse régénérables.
6. `notebooks/cl_eval/drift_datasets/analysis.ipynb` + `tests/test_drift_datasets.py`.
7. `notebooks/cl_eval/drift_datasets/eda_gas_sensor.ipynb` — EDA exhaustive Gas Sensor Array Drift (S4306).
8. `notebooks/cl_eval/drift_datasets/eda_hydraulic.ipynb` — EDA exhaustive Condition Monitoring Hydraulique (S4307).
9. `notebooks/cl_eval/drift_datasets/eda_electricity.ipynb` — EDA exhaustive Electricity/ELEC2 (S4308).

## Questions ouvertes

- `TODO(arnaud)` : validation de la short-list — priorité aux datasets **industriels capteurs réels**
  (Gas Sensor Drift, INSECTS) vs benchmarks concept-drift classiques (Electricity/NOAA) pour la
  cohérence Gap 1 ?
- `TODO(fred)` : Edge Spectrum dispose-t-il de données terrain avec drift documenté (dérive de capteur,
  changement de régime machine) mobilisables ici ?
- `TODO(dorra)` : contrainte de licence/redistribution pour les datasets retenus (données brutes en
  `.gitignore`, mais scripts de téléchargement publics GitLab).

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4300 | 📝 Doc | — | Overview + cadrage |
| S4301 | ✅ | — | `docs/context/drift_datasets.md` — Gas Sensor ⭐, Electricity, Hydraulic (remplace INSECTS), synthétique numpy (remplace `river`) |
| S4302 | ✅ | — | 4 loaders + `src/data/drift_dataset.py` + 4 configs + `download_drift_datasets.py` + registre `DRIFT_LOADERS` |
| S4303 | ✅ | — | `characterize_drift.py` → `exp_S43_drift_char/` ; synthétique `alignment_score=75` (±1 fenêtre), Electricity `null` |
| S4304 | ✅ | — | Catalogue `src/figures/catalogs/drift_datasets.py` → **17 PNG** `docs/figures/drift_datasets/` (timeline/shift/PCA/heatmap/comparatif) ; FR, 0 chiffre en dur |
| S4305 | ✅ | — | Notebook `analysis.ipynb` (nbconvert OK) + `tests/test_drift_datasets.py` **16 PASS** + roadmap/CLAUDE.md/graphify |
| S4306 | 📝 Doc | — | EDA exhaustive Gas Sensor Array Drift (gaz + batch) — notebook à générer |
| S4307 | 📝 Doc | — | EDA exhaustive Condition Monitoring Hydraulique (faute + condition cooler) — notebook à générer |
| S4308 | 📝 Doc | — | EDA exhaustive Electricity/ELEC2 (drift graduel, `drift_points=None`) — notebook à générer |

> **Note de cadrage réalisé** : substitutions validées avec l'utilisateur — **INSECTS → Hydraulic** (dataset de faute segmenté par condition cooler) et **`river` → générateurs numpy** (dépendance non installée ; cohérent « pas de dépendances lourdes non justifiées »). **Carte non utilisée** (aucune des tâches S4301–S4303 ne la requiert ; board = S45).
