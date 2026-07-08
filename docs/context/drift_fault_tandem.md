# Détection de drift & détection de faute en tandem, automatisées sur carte

> **Statut** : 📝 Document de cadrage pour un **sprint futur** (post-Sprint 45). Matière de construction,
> pas une spec de tâche exécutable. Rédigé le 7 juillet 2026.
>
> **Objet** : décrire comment **allier** un modèle de détection de drift (Sprints 44/45) à un modèle de
> détection de faute (EWC/Mahalanobis, Sprints 26–38) pour les faire fonctionner **en tandem** et
> **automatiser** la décision d'adaptation **à bord**, sans PC.

---

## 1. Problème & motivation

Le Sprint 38 a introduit une **mise à jour EWC autonome** pilotée par un **gate de nouveauté** : un score
Mahalanobis + `SlidingWindowDriftDetector` (double seuil + persistance) décide à bord *quand* et *avec
quel label* updater. Ce gate résout déjà le nœud honnête **drift ≠ faute** — une distance seule ne
distingue pas un *sain dérivé* (→ réabsorber comme normal) d'une *première faute* (→ apprendre la classe
faulty).

Mais ce gate repose sur **un seul détecteur** (fenêtre sur score Maha). Les Sprints 44/45 produisent une
**famille de détecteurs de drift** évalués et portés board (Page-Hinkley, DDM/EDDM, PSI, ADWIN, …), avec
des **métriques de détection** (délai, fausses alarmes) et un **coût board mesuré**. Le sprint tandem
consiste à **remplacer/compléter** le gate S38 par un **vrai détecteur de drift** choisi sur preuves, en
tandem avec le détecteur de faute, et à quantifier le gain.

**Question centrale** : un détecteur de drift dédié (mesuré en S44/S45) améliore-t-il l'arbitrage
*adapter vs apprendre* de S38 — moins de fausses adaptations, meilleur délai, coût board acceptable ?

## 2. Architecture proposée

Deux détecteurs s'exécutent en parallèle sur chaque échantillon, à bord :

```
                  ┌─────────────────────────┐
   échantillon ──►│  Détecteur de FAUTE      │──► pred (sain/faulty) + score/confiance
        x_t       │  (EWC tête binaire /     │
                  │   Mahalanobis)           │
                  ├─────────────────────────┤
                  │  Détecteur de DRIFT      │──► verdict (NORMAL / WARNING / DRIFT)
                  │  (S44/S45 : Page-Hinkley │
                  │   / PSI / … sur feature  │
                  │   ou score Maha)         │
                  └───────────┬─────────────┘
                              ▼
                   ┌────────────────────┐
                   │  TABLE DE DÉCISION  │  (arbitrage tandem)
                   └────────┬───────────┘
                            ▼
        adapter le normal │ apprendre la faute │ ne rien faire
        (maha_update)     │ (ewc_sgd_step,     │
                          │  label vrai/pseudo)│
```

### Table de décision (arbitrage drift × faute)

| Verdict drift | Prédiction faute | Action | Rationale |
|---------------|------------------|--------|-----------|
| NORMAL | sain | rien | régime stable |
| NORMAL | faulty (transitoire) | rien / compteur | bruit ponctuel, pas de drift collectif |
| DRIFT | sain | **adapter le normal** (`maha_update`) | dérive du sain → réabsorber |
| DRIFT | faulty | **arbitrage** (§3) | ambiguïté : dérive saine *ou* faute émergente |
| WARNING | — | pré-alerte, fenêtre d'observation | réduit les fausses adaptations |

C'est une **généralisation** des 4 politiques S38 (frozen/always/gated_truelabel/gated_pseudolabel) : le
gate binaire NORMAL/DRIFT/FAULT de S38 devient une **table 2D** (verdict de drift × prédiction de faute),
alimentée par un détecteur de drift **choisi et mesuré**.

## 3. Le nœud à trancher : drift ET faute simultanés

Le cas `DRIFT × faulty` est l'ambiguïté résiduelle : la distribution change **et** le modèle prédit une
faute. Deux hypothèses opposées :
- dérive saine que le modèle de faute confond avec une panne (→ **adapter**, ne pas apprendre) ;
- vraie faute émergente qui déplace aussi la distribution (→ **apprendre**).

Pistes d'arbitrage (à évaluer) : persistance temporelle (une faute vraie persiste, une dérive saine se
stabilise après réabsorption) ; désaccord entre détecteur supervisé (erreur) et non-supervisé (features) ;
consultation d'un label vrai **uniquement** sur ce cas (active learning ciblé, coût minimal).

## 4. Réutilisation directe (déjà en place)

- **Gate `pipeline.c`** sous `-DEWC_AUTO_UPDATE` (S3803) — point d'insertion de la table de décision.
- **4 politiques** S38 (`run_sprint38_pc.py`, `run_sprint38_board.py`) — cadre expérimental à généraliser.
- **Détecteurs de drift** portés S45 (`firmware/.../src/drift/*.c`, `-DDRIFT_DETECT`) — remplacent la
  brique unique du gate.
- **`maha_update` / `ewc_sgd_step`** — actions d'adaptation/apprentissage inchangées.
- **Métriques d'économie** `aggregate_sprint38.py::economy_table` (updates économisés, latence, RAM) —
  directement réutilisables.
- **Pattern apparié** PC→board→parité→agrégat→notebook→tests.

## 5. Axes d'évaluation proposés

- **Économie vs précision** : updates économisés (vs `always`) et F1 de faute préservé — métrique clé S38,
  à recomparer avec chaque détecteur de drift S44.
- **Taux de fausses adaptations** : combien de fois le tandem réabsorbe comme « sain » un point en fait
  faulty (ou l'inverse) — nouvelle métrique propre au tandem.
- **Délai d'adaptation** : entre le vrai début du drift (ground-truth S43) et l'action d'adaptation.
- **Coût board combiné** : latence (faute + drift) < 100 ms (Gap 2), `.bss` des deux détecteurs (Gap 3).
- **Parité board↔PC** du verdict *et* de l'action (déterministe → 1.000 attendu).
- **Supervisé ∥ non-supervisé** : le tandem est-il meilleur avec un détecteur de drift supervisé (erreur)
  ou non-supervisé (features), compte tenu de l'autonomie visée ?

## 6. Datasets

Prioriser les datasets **dual-usage drift+faute** identifiés en S43 (`docs/context/drift_datasets.md`) :
- **UCI Gas Sensor Array Drift** — dérive capteur réelle + classification (faute-like) : cas d'école du
  tandem (drift de capteur qu'il ne faut **pas** confondre avec une faute).
- **INSECTS** — drift ground-truth + classes : mesure du délai d'adaptation.
- Comparaison de contrôle avec Monitoring/Pronostia (S38) pour la continuité.

## 7. Questions ouvertes

- Arbitrage `DRIFT × faulty` : règle fixe (persistance) vs méta-modèle (précédent stacking S31) ?
- Label du SGD sur adaptation : actif (vrai label ciblé, S38 P2) vs pseudo (autonome, S38 P3) — le tandem
  change-t-il l'équilibre ?
- Intégration protocole UART : les **deux** verdicts (faute + drift) tiennent-ils dans la réponse V3 par
  réinterprétation de champs (précédent S3805), ou faut-il une réponse étendue ?
- Un détecteur de drift **par feature** vs sur le **score Maha agrégé** : granularité vs coût board.
- Recouvrement avec le méta-modèle de stacking (S31) : la table de décision peut-elle être *apprise*
  plutôt que fixée ?

## 8. Pointeurs

- Gate de nouveauté & 4 politiques : `docs/sprints/sprint_38/S3800_sprint_38.md`.
- Datasets de drift dual-usage : `docs/context/drift_datasets.md` (Sprint 43).
- Détecteurs de drift PC + reco MCU : `docs/context/drift_detectors.md` (Sprint 44).
- Détecteurs de drift board (coût mesuré, parité) : `docs/sprints/sprint_45/` + `exp_S45_summary.json`.
- Triple gap : `docs/triple_gap.md` (§ Gap 2 latence combinée, § Gap 3 RAM combinée).
