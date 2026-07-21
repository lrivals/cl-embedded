# S4301 — Recherche & sélection de datasets de drift

| Champ | Valeur |
|-------|--------|
| **Sprint** | 43 |
| **Priorité** | 🔴 Critique — définit le corpus sur lequel toute l'évaluation drift (S44/S45) reposera. |
| **Statut** | ✅ Implémenté — `docs/context/drift_datasets.md` produit ; sélection : Gas Sensor ⭐, Electricity, Hydraulic (remplace INSECTS), synthétique numpy (remplace `river`). `TODO(arnaud)` = valider substitutions. |
| **Durée estimée** | 5h |
| **Dépendances** | Aucune (tâche documentaire) · s'appuie sur le nœud « drift ≠ faute » de Sprint 38 (`docs/sprints/sprint_38/S3800_sprint_38.md`) |
| **Fichiers cibles** | `docs/context/drift_datasets.md` (source de vérité) |
| **Références** | `docs/context/datasets.md` (gabarit de fiche dataset projet) · CLAUDE.md § Datasets |

---

## Contexte

Aucun dataset actuel du projet ne porte de **points de drift labellisés** : le drift y est déduit du
scénario CL (Monitoring = changement d'équipement ; Pronostia = arrivée temporelle de la faute), jamais
annoté. Pour mesurer un **délai de détection** et un **taux de fausses alarmes** (métriques S44), il faut
une vérité-terrain : *à quel indice la distribution change*. Cette tâche recherche et **propose plusieurs
datasets externes** et fige le choix dans une doc de référence unique.

## Spec

Produire `docs/context/drift_datasets.md` : pour **chaque candidat**, une fiche normalisée (miroir de
`docs/context/datasets.md`) contenant :

- **Nom & source** (URL, dépôt, année, référence bibliographique éventuelle).
- **Type de données** : séries temporelles capteurs / tabulaire / flux ; dimension de features ; volume.
- **Type de drift** : `sudden` / `gradual` / `incremental` / `recurring` (taxonomie standard concept
  drift — Gama et al. 2014).
- **Ground-truth de drift** : points de drift connus (indices/timestamps) ? exacts, approximatifs, ou
  déduits du protocole (batches temporels) ? — champ **décisif** pour la sélection.
- **Dual-usage faute** : le dataset porte-t-il aussi un label de faute/classe exploitable par un
  détecteur de faute (EWC/Maha) ? Décrire le label.
- **Scénario CL proposé** : découpage en tâches/segments (comme les autres loaders projet).
- **Licence & redistribution** : compatibilité GitLab public (données brutes en `.gitignore`, seul le
  script de téléchargement est versionné).
- **Justification de sélection / rejet**.

### Candidats à documenter

**Retenus prioritaires (dual-usage, ground-truth exploitable) :**

1. **UCI Gas Sensor Array Drift Dataset** ⭐
   - Dérive **réelle de capteurs** chimiques sur **36 mois**, 6 gaz, 16 capteurs (128 features).
   - Ground-truth de drift = **10 batches temporels** (le drift entre batches est la cible d'étude
     originale du dataset — c'est *le* benchmark de drift de capteur industriel).
   - Dual-usage : classification 6 gaz = tâche de « faute »/état exploitable par un modèle supervisé.
   - Justification : capteur industriel réel + drift documenté + dual-usage → cœur Gap 1 sur l'axe drift.

2. **USP INSECTS (Data Stream)** ⭐
   - Flux capteur (comptage optique d'insectes), variantes **abrupt / gradual / incremental /
     incremental-recurring / incremental-abrupt-recurring** — **points de drift documentés** par variante.
   - Justification : couvre **tous les types de drift** avec ground-truth ponctuelle → calibration fine
     des métriques de délai de détection (S44). Dual-usage : classification d'espèces.

**Retenus secondaires (concept drift classique, supervisé) :**

3. **Electricity / ELEC2** — prix électricité NSW, drift conceptuel bien connu ; ground-truth non
   ponctuelle mais drift établi dans la littérature → utile pour détecteurs **supervisés** (flux d'erreur).
4. **NOAA Weather** — prédiction de pluie, drift saisonnier récurrent → détecteurs supervisés.

**Contrôle (drift à points exacts) :**

5. **Générateurs synthétiques** (SEA, Hyperplane rotatif, Agrawal) via `river` — points de drift **exacts
   par construction** → **vérité-terrain parfaite** pour valider la mesure de délai de détection avant de
   l'appliquer aux datasets réels. Non porté MCU (usage PC-only, calibration métriques).

**Dual-usage orienté faute (drift secondaire) :**

6. **UCI Condition Monitoring of Hydraulic Systems** ou **SECOM** — label de faute primaire, drift
   secondaire → réservoir pour le tandem drift+faute (sprint futur), documenté mais non prioritaire ici.


Dataset selectionnes :
/home/leonard/Documents/ENAC/cl-embedded/data/raw/Condition Monitoring of Hydraulic Systems

/home/leonard/Documents/ENAC/cl-embedded/data/raw/Gas Sensor Array Drift Dataset

/home/leonard/Documents/ENAC/cl-embedded/data/raw/The Elec2 Dataset
## Contraintes

- **Aucune donnée brute committée** (`.gitignore`) — seul le script de téléchargement est versionné.
- Chaque fiche doit expliciter **si le ground-truth de drift est ponctuel** (indices exacts) ou seulement
  **structurel** (batches/segments) — cela conditionne quelles métriques S44 sont calculables (délai de
  détection exige des points ponctuels ; FAR se mesure sur segments stables).
- Privilégier les datasets **capteurs réels** pour la cohérence Gap 1 ; les synthétiques restent des
  **outils de calibration**, pas des résultats scientifiques.
- Vérifier la **taille de features** de chaque dataset (impact direct sur l'état mémoire porté board S45).

## Critères d'acceptation

1. `docs/context/drift_datasets.md` existe, ≥ 5 candidats documentés, chacun avec les champs ci-dessus.
2. Au moins **2 datasets dual-usage à ground-truth de drift** sont marqués « retenus prioritaires ».
3. Au moins **1 source à points de drift exacts** (synthétique) est identifiée pour la calibration des
   métriques S44.
4. Chaque dataset a une ligne de licence/redistribution explicite (compatibilité GitLab public).
5. La short-list est cohérente avec la table « Sources de données » de `S4300_sprint_43.md`.
