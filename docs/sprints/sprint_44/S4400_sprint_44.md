# Sprint 44 — Modèles de détection de drift sur PC (supervisés ∥ non-supervisés)

| Champ | Valeur |
|-------|--------|
| **Sprint** | Sprint 44 |
| **Semaine** | 27 juillet – 2 août 2026 |
| **Statut** | 📝 Doc — spec complète ; implémentation à venir |
| **Priorité globale** | 🔴 Critique — cœur scientifique de l'étude drift ; produit la reco des détecteurs à porter board (S45). |
| **Durée estimée totale** | ~34h (inventaire ~4h · détecteurs supervisés ~6h · non-supervisés ~8h · harnais éval ~6h · exécution/figures ~6h · notebook+tests+docs ~4h) |
| **Dépendances** | Sprint 43 ✅ (datasets à ground-truth de drift) · `src/evaluation/drift_detector.py` ✅ (baseline) · `src/evaluation/memory_profiler.py` ✅ (RAM/latence proxies) · `src/evaluation/metrics.py` ✅ · `src/models/ewc/ewc_mlp.py` ✅ + `src/models/unsupervised/mahalanobis_detector.py` ✅ (flux d'erreur/score) |

## Contexte et motivation

Le projet ne possède **qu'un seul détecteur de drift** : `SlidingWindowDriftDetector` (double seuil sur
score Mahalanobis). C'est une baseline utile mais isolée — aucune **famille statistique** de référence
(ADWIN, DDM/EDDM, Page-Hinkley), aucun **test deux-échantillons** (KS glissant, KSWIN, MMD, PSI/JS),
aucun **harnais de métriques dédiées** au drift (délai de détection, fausses alarmes, manqués).

Ce sprint **implémente et évalue** ces détecteurs sur PC, sur les datasets à ground-truth du Sprint 43,
avec deux exigences :

1. **Évaluation précise et justifiable** : métriques de détection *plus* **RAM et latence** mesurées
   (proxies PC honnêtes via `memory_profiler.py`), pour pouvoir arbitrer objectivement quels détecteurs
   sont **portables MCU** (préparation S45).
2. **Supervisé ∥ non-supervisé à parité** : comparer explicitement les détecteurs pilotés par le **flux
   d'erreur** d'un modèle (DDM/EDDM/Page-Hinkley — nécessitent des labels) et ceux pilotés par la
   **distribution des features/scores** (ADWIN/KS/MMD/PSI — sans labels). C'est un axe d'étude en soi :
   sur une carte déployée seule (scénario S38), les non-supervisés sont plus réalistes ; les supervisés
   sont plus précis mais exigent un retour de vérité.

## Décisions de cadrage (utilisateur, 7 juillet 2026)

- **Trois familles** : statistiques streaming (état borné O(1)/O(W)) + tests deux-échantillons +
  **baseline projet** (`SlidingWindowDriftDetector`).
- **Signal supervisé ET non-supervisé à parité** — comparaison explicite, pas un choix.
- **Priorité aux méthodes portables MCU** : état mémoire borné, pas d'allocation dynamique côté algo →
  chaque détecteur est annoté d'une **viabilité MCU** dès le PC (guide S45).
- **Évaluation = détection + coût** : métriques de drift *et* RAM/latence dans le même tableau.
- **Langue** : français.

## Nœud honnête : que signifie « détecter le drift » ici

Un détecteur de drift signale un **changement de distribution**, pas une faute. Sur les datasets S43 à
ground-truth ponctuelle, on mesure sa capacité à **signaler près du vrai point de drift** (délai) **sans
crier au loup** sur les segments stables (fausses alarmes). Deux détecteurs peuvent avoir la même
« accuracy » de détection mais des **délais** et **coûts mémoire** opposés — d'où la nécessité des
métriques spécialisées et du profilage conjoint.

## Les familles couvertes

| Famille | Détecteurs | Signal | État mémoire | Viabilité MCU |
|---------|-----------|--------|--------------|:-------------:|
| **Statistique supervisé** | DDM, EDDM, Page-Hinkley | flux d'erreur (labels requis) | O(1) | ✅ excellente |
| **Statistique non-supervisé** | ADWIN | valeur scalaire (feature/score) | O(log W) histogrammes exp. | 🟡 modérée |
| **Test deux-échantillons** | KSWIN, KS glissant, MMD, PSI / Jensen-Shannon | fenêtre de features | O(W) (+ tri pour KS) | 🟡 à ⚠️ |
| **Baseline projet** | `SlidingWindowDriftDetector` | score anomaly (Maha) | O(W) | ✅ (déjà porté C) |

## Tâches

### Bloc A — Inventaire & config

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4401 | **Inventaire de référence** des détecteurs (taxonomie, formule, état mémoire, labels requis, viabilité MCU) + config sprint | 🔴 | `docs/context/drift_detectors.md`, `configs/sprint44_drift_detection.yaml` | 📝 Doc |

### Bloc B — Implémentation des détecteurs

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4402 | **Détecteurs supervisés** (DDM, EDDM, Page-Hinkley) sur flux d'erreur, interface commune, annotations `# MEM:` | 🔴 | `src/models/drift/{ddm,eddm,page_hinkley}.py` | 📝 Doc |
| S4403 | **Détecteurs non-supervisés** (ADWIN, KSWIN, KS glissant, MMD, PSI/JS) sur features, interface commune bornée | 🔴 | `src/models/drift/{adwin,kswin,ks_test,mmd,psi}.py`, `src/models/drift/base.py` | 📝 Doc |

### Bloc C — Évaluation & exécution

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4404 | **Harnais de métriques drift** (délai, FAR, MDR, MTFA/MTD, précision/rappel points) + RAM/latence proxies | 🔴 | `src/evaluation/drift_metrics.py` | 📝 Doc |
| S4405 | **Exécution grille** (détecteurs × datasets × {supervisé,non-supervisé}) + figures d'impact | 🟠 | `scripts/run_sprint44_pc.py` → `experiments/exp_S44_PC_*`, `docs/figures/drift_detection_pc/` | 📝 Doc |

### Bloc D — Assemblage & clôture

| ID | Tâche | Prio | Fichier cible | Statut |
|----|-------|:---:|---------------|:------:|
| S4406 | Notebook comparatif + tests + **recommandation MCU** (détecteurs à porter S45) + roadmap + `CLAUDE.md` + graphify | 🟡 | `notebooks/cl_eval/drift_detection/comparison.ipynb`, `tests/test_drift_detectors.py`, `tests/test_drift_metrics.py` | 📝 Doc |

## Ordre d'exécution recommandé

```
S4401 (inventaire+config) ──┬─► S4402 (supervisés) ──┐
                            └─► S4403 (non-supervisés)┤─► S4405 (exécution+figures) ─► S4406 (notebook+tests+reco MCU)
S4404 (harnais métriques) ──────────────────────────┘
```

S4404 (harnais) peut être développé en parallèle de S4402/S4403 (interface figée en S4401). S4405 exige
les trois. S4406 clôt et **produit la reco de portage** consommée par S45.

## Sources de données (Sprint 43, lecture seule)

| Dataset | Rôle en S44 | Métriques calculables |
|---------|-------------|----------------------|
| **Synthétique** (points exacts) | calibration/validation des métriques (délai vérité-terrain) | toutes (délai, FAR, MDR) |
| **Gas Sensor Drift** | drift capteur réel incrémental | délai (batches), FAR |
| **INSECTS** (variantes) | tous types de drift, ground-truth ponctuelle | toutes |
| **Electricity / NOAA** | concept drift supervisé | FAR/stabilité (pas de délai ponctuel) |

## Livrables

1. `docs/sprints/sprint_44/` (ce dossier) — specs S4400–S4406.
2. `docs/context/drift_detectors.md` — référence textuelle unique (taxonomie + viabilité MCU).
3. `src/models/drift/` — détecteurs supervisés + non-supervisés + interface commune.
4. `src/evaluation/drift_metrics.py` — harnais de métriques de détection.
5. `scripts/run_sprint44_pc.py` + `experiments/exp_S44_PC_*` (grille) + `docs/figures/drift_detection_pc/`.
6. `notebooks/cl_eval/drift_detection/comparison.ipynb` + tests + **reco MCU pour S45**.

## Questions ouvertes

- `TODO(arnaud)` : pour l'axe supervisé, quel modèle de faute fournit le flux d'erreur — EWC (tête
  binaire) ou un classifieur dédié par dataset ? Le choix influence DDM/EDDM.
- `TODO(dorra)` : ADWIN (histogrammes exponentiels) est-il portable board dans le budget, ou faut-il une
  variante à état fixe (fenêtre bornée) pour S45 ?
- `TODO(arnaud)` : seuils de détection — calibrés par dataset (P95 enrôlement, cohérent S38) ou fixés par
  les valeurs de référence littérature (DDM 2σ/3σ, Page-Hinkley δ/λ) ?

## Bilan (à compléter)

| Tâche | Statut | Temps réel | Notes |
|-------|:------:|:----------:|-------|
| S4400 | 📝 Doc | — | Overview + cadrage |
| S4401–S4406 | 📝 Doc | — | Documentés ; implémentation à venir |
