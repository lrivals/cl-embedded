# Datasets de drift — Recherche, sélection et fiches de référence (Sprint 43, S4301)

> Source de vérité unique pour le corpus **drift** du projet (Sprints 44/45). Miroir du gabarit de
> `docs/context/datasets.md`. Aucune donnée brute n'est committée (`data/raw/` est en `.gitignore`) —
> seuls les scripts de téléchargement et les loaders sont versionnés.

## Contexte

Aucun dataset actuel du projet ne porte de **points de drift labellisés** : le drift y est déduit du
scénario CL (Monitoring = changement d'équipement ; Pronostia = arrivée temporelle de la faute), jamais
annoté échantillon par échantillon. Pour mesurer un **délai de détection** et un **taux de fausses
alarmes** (métriques S44) et porter un détecteur board (S45), il faut une **vérité-terrain de drift** :
*à quel indice la distribution change*. Cette doc recherche/propose plusieurs candidats et fige le choix.

**Distinction décisive** — ground-truth de drift :

- **Ponctuelle** (indices exacts) → délai de détection **calculable** (ex. synthétique).
- **Structurelle** (frontières de batches/segments) → alignement approximatif ; FAR/stabilité mesurables
  (Gas Sensor, Hydraulic).
- **Absente** (drift établi par la littérature, pas de points) → seules FAR/stabilité (Electricity, NOAA).

**Taxonomie du drift** (Gama et al. 2014) : `sudden` · `gradual` · `incremental` · `recurring`.

---

## Short-list — datasets **retenus**

### D-Drift-1 — UCI Gas Sensor Array Drift Dataset ⭐ (retenu prioritaire)

| Propriété | Valeur |
|-----------|--------|
| **Source** | UCI ML Repository — Vergara (2012), DOI 10.24432/C5RP6W |
| **Chemin local** | `data/raw/Gas Sensor Array Drift Dataset/Dataset/batch{1..10}.dat` |
| **Type de données** | Séries capteurs chimiques, format libsvm (`label idx:val …`) |
| **Dimension features** | **128** (16 capteurs × 8 descripteurs) |
| **Volume** | 13 910 mesures, 10 batches temporels (janv. 2007 → févr. 2011, 36 mois) |
| **Type de drift** | `incremental` — dérive **réelle** de capteur chimique |
| **Ground-truth de drift** | **Structurelle** — frontières des 10 batches (cible d'étude originale) |
| **Dual-usage faute** | ✅ `y` = identité du gaz ∈ {1..6} (Ammoniac, Acétaldéhyde, Acétone, Éthylène, Éthanol, Toluène) — tâche de classification d'état supervisée |
| **Scénario CL** | Domain-incremental par batch temporel (drift de capteur) |
| **Licence & redistribution** | **Recherche uniquement — usage commercial exclu**. Brut en `.gitignore`, script versionné. |
| **Loader / config** | `src/data/gas_sensor_drift_dataset.py` · `configs/gas_sensor_drift_config.yaml` |

**Justification (retenu)** : capteur industriel **réel** + drift documenté + dual-usage → cœur Gap 1 sur
l'axe drift. Caractérisation S4303 : distance de Mahalanobis au batch 1 saute de ~7 à ~760 dès le batch 2
(dérive marquée), fluctue ensuite — drift massif confirmé. 128 features → **le plus lourd** pour le
portage board S45 (impact état mémoire à documenter).

### D-Drift-2 — Electricity / ELEC2 (retenu secondaire)

| Propriété | Valeur |
|-----------|--------|
| **Source** | Harries (1999) ; Gama et al. (2004) ; version normalisée par Bifet |
| **Chemin local** | `data/raw/The Elec2 Dataset/electricity-normalized.csv` |
| **Type de données** | Tabulaire temporel (marché électrique NSW, mai 1996 → déc. 1998) |
| **Dimension features** | **7** (day, period, nswprice, nswdemand, vicprice, vicdemand, transfer) |
| **Volume** | 45 312 instances (pas de 30 min) |
| **Type de drift** | `gradual` (concept drift classique) |
| **Ground-truth de drift** | **Absente** (non ponctuelle) → `drift_points = None`, `alignment_score = null` |
| **Dual-usage faute** | ➖ label `class` = hausse/baisse du prix (UP/DOWN → 1/0), supervisé mais pas « faute » |
| **Scénario CL** | Flux temporel unique (détecteurs supervisés, flux d'erreur) |
| **Licence & redistribution** | Public / usage recherche. Brut en `.gitignore`. |
| **Loader / config** | `src/data/electricity_dataset.py` · `configs/electricity_drift_config.yaml` |

**Justification (retenu secondaire)** : benchmark concept-drift **de référence** de la littérature.
Ground-truth non ponctuelle → sert à mesurer **FAR/stabilité** des détecteurs, pas le délai de détection.
Déjà normalisé [0,1] (Bifet) → pas de re-fit destructeur.

### D-Drift-3 — Condition Monitoring of Hydraulic Systems (retenu — dual-usage faute)

| Propriété | Valeur |
|-----------|--------|
| **Source** | ZeMA gGmbH / UCI (2018) — Helwig, Schütze et al. |
| **Chemin local** | `data/raw/Condition Monitoring of Hydraulic Systems/*.txt` |
| **Type de données** | Séries multi-capteurs (banc hydraulique, cycles de 60 s) |
| **Dimension features** | **17** (moyenne par cycle des capteurs PS1-6, EPS1, FS1-2, TS1-4, VS1, CE, CP, SE) |
| **Volume** | 2 205 cycles |
| **Type de drift** | `incremental` (segmenté par condition du refroidisseur) |
| **Ground-truth de drift** | **Structurelle** — pas de points ponctuels natifs ; segmentation par la condition cooler (profile col. 1 ∈ {3, 20, 100} %) → frontières = `drift_points` |
| **Dual-usage faute** | ✅ `y` = `stable_flag` binaire (profile col. 5) ; conditions de faute disponibles (valve, pompe, accumulateur, cooler) pour le tandem futur |
| **Scénario CL** | Domain/condition-incremental par degré de dégradation du refroidisseur |
| **Licence & redistribution** | Usage recherche (ZeMA gGmbH). Brut en `.gitignore`. |
| **Loader / config** | `src/data/hydraulic_dataset.py` · `configs/hydraulic_drift_config.yaml` |

**Justification (retenu, décision utilisateur)** : dataset de **faute** réel réutilisé comme réservoir de
**drift secondaire** ; remplace INSECTS dans la sélection. Le `drift_points` est **structurel** (segments
cooler), documenté honnêtement — le délai de détection n'est pas ponctuellement calculable, mais le
dual-usage faute prépare le tandem drift+faute (sprint futur).

### D-Drift-4 — Synthétique (numpy) (retenu — calibration, points EXACTS)

| Propriété | Valeur |
|-----------|--------|
| **Source** | Générateurs numpy internes (**pas de dépendance `river`**) : `sea`, `rotating_hyperplane`, `gradual_mixture` |
| **Chemin local** | Aucun — **généré à la volée** (seed reproductible) |
| **Dimension features** | Paramétrable (défaut 6 pour `gradual_mixture`, 3 pour `sea`) |
| **Volume** | Paramétrable (défaut 6 000) |
| **Type de drift** | Contrôlé : `sudden` (sea), `incremental` (hyperplane), `gradual` (mixture) |
| **Ground-truth de drift** | **PONCTUELLE EXACTE** — `drift_points` imposés en config == retournés par le loader |
| **Dual-usage faute** | ➖ label du générateur |
| **Scénario CL** | Flux à concepts imposés (calibration des métriques S44) |
| **Licence & redistribution** | N/A (généré). |
| **Loader / config** | `src/data/synthetic_drift_dataset.py` · `configs/synthetic_drift_config.yaml` |

**Justification (retenu, contrôle)** : **vérité-terrain parfaite** pour valider la chaîne de mesure du
délai de détection **avant** de l'appliquer aux datasets réels. Défaut = `gradual_mixture` (drift
**covariate** = décalage de moyenne), détectable par la chaîne KS/MMD/Mahalanobis de S4303 ; la
caractérisation aligne les pics à ±1 fenêtre du point imposé (`alignment_score ≈ 75` sur n=6000). Le
générateur `sea` (concept drift pur, `P(x)` constant) reste disponible mais **non recommandé** pour
valider une chaîne covariate. **PC-only** — non porté MCU.

---

## Candidats **documentés mais rejetés / réservés**

| Dataset | Type de drift | Ground-truth | Dual-usage | Statut & justification |
|---------|---------------|--------------|-----------|------------------------|
| **USP INSECTS** | abrupt / gradual / incremental / recurring | ✅ points documentés | ✅ espèces | **Remplacé par Hydraulic** (décision utilisateur). Reste un excellent candidat (tous types de drift + points ponctuels) → réserve S44+ si besoin de calibration réelle multi-types. |
| **NOAA Weather** | concept drift saisonnier (recurring) | ⚠️ non ponctuel | ➖ | **Rejeté** ici : redondant avec Electricity (concept drift supervisé, GT non ponctuelle). Réserve. |
| **SECOM** | secondaire | ⚠️ | ✅ faute primaire | **Réservé** au tandem drift+faute (sprint futur), non prioritaire. |

---

## Récapitulatif de sélection

| Dataset | Retenu | Prio | GT drift | Dual-usage faute | d (features) | Loader |
|---------|:------:|:----:|----------|:----------------:|:------------:|--------|
| Gas Sensor Array Drift | ✅ | 🔴 | structurelle (10 batches) | ✅ 6 gaz | 128 | `gas_sensor_drift_dataset.py` |
| Electricity / ELEC2 | ✅ | 🟠 | absente | ➖ | 7 | `electricity_dataset.py` |
| Hydraulic (Condition Monitoring) | ✅ | 🟠 | structurelle (cooler) | ✅ conditions | 17 | `hydraulic_dataset.py` |
| Synthétique (numpy) | ✅ | 🔴 (contrôle) | **ponctuelle exacte** | ➖ | 3–6 | `synthetic_drift_dataset.py` |
| USP INSECTS | ➖ réserve | — | ponctuelle | ✅ | — | — |
| NOAA / SECOM | ➖ réserve | — | — | (SECOM ✅) | — | — |

**Critères d'acceptation S4301 satisfaits** : (1) ≥ 5 candidats documentés ; (2) ≥ 2 datasets dual-usage
à ground-truth de drift retenus prioritaires (**Gas Sensor** + **Hydraulic** dual-usage) ; (3) ≥ 1 source à
points de drift exacts (**synthétique numpy**) ; (4) ligne licence/redistribution par dataset ;
(5) cohérent avec la table « Sources de données » de `S4300_sprint_43.md` (INSECTS remplacé par Hydraulic,
`river` remplacé par générateurs numpy — cf. décisions utilisateur).

## Questions ouvertes

- `TODO(arnaud)` : valider la substitution INSECTS → Hydraulic et le générateur synthétique numpy (vs `river`).
- `TODO(dorra)` : les 128 features de Gas Sensor impactent l'état mémoire board S45 — confirmer la
  stratégie (sélection de features / projection) avant portage.
- `TODO(fred)` : données terrain Edge Spectrum avec drift documenté mobilisables ?
