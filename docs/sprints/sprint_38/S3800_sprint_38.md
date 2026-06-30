# Sprint 38 — Mise à jour EWC autonome déclenchée par détection de drift/nouveauté

## Contexte et motivation

Jusqu'au Sprint 37, la **mise à jour en ligne d'EWC sur board est pilotée par l'hôte** : c'est le bit
`PROTO_FLAG_UPDATE = 0x01` de la trame UART (`pipeline.c`, chemin EWC 0x10) qui décide, échantillon par
échantillon, si le modèle fait un pas de SGD. **Un humain au PC décide donc quand le modèle apprend.**

Le scénario cible de ce sprint est une **carte déployée seule, sans PC** sur une machine industrielle
neuve :

1. Au début, la machine est saine → la carte ne voit que des **données saines** (enrôlement one-class).
2. À un moment, elle est confrontée soit à un **distribution drift** (l'équipement change de régime mais
   reste sain), soit à l'**arrivée des premières données faulty**.
3. Le **nombre de classes ne change pas** (binaire sain/faulty) — pas de class-incremental, juste à
   peupler la classe faulty quand elle apparaît.

**Objectif** : remplacer le déclencheur humain par un **gate de nouveauté embarqué** et quantifier
l'arbitrage **économie (RAM + latence) vs précision** entre « EWC mis à jour en permanence » et
« EWC mis à jour seulement quand le gate l'estime nécessaire ».

## Le nœud honnête : drift ≠ faute

Une distance (Mahalanobis) seule ne distingue pas un **sain dérivé** d'une **vraie faute** : les deux
sont simplement « loin du normal ». Or les deux exigent l'action *opposée* :

| Cas | Action correcte |
|-----|-----------------|
| Drift du sain | **adapter** la notion de normal (réabsorber le point comme sain) |
| Première faute | **ne pas** absorber comme normal, mais **apprendre** la classe faulty |

La brique `SlidingWindowDriftDetector` (`src/evaluation/drift_detector.py`, Sprint 9) résout exactement
cette ambiguïté via **double seuil + persistance temporelle** :

- **FAULT** : dépassement instantané `score > fault_threshold` (P95 × 2.5) → panne soudaine.
- **DRIFT** : fraction de la fenêtre glissante au-dessus de `drift_threshold` (P95 × 1.3) dépasse
  `drift_ratio` (0.6) → dérive progressive et collective.
- **NORMAL** : sinon.

Second nœud : `ewc_sgd_step` est **supervisé** (il faut un label). Détecter qu'un point est « loin » ne
dit pas *quelle classe* c'est → on teste **deux politiques de label** (vrai label sur flag = active
learning ; pseudo-label par verdict = 100 % autonome).

## Les 4 politiques comparées

Toutes exécutent le **même EWC** ; seul change *qui décide d'updater* et *avec quel label* :

| ID | Politique | Déclencheur | Label SGD | Autonomie |
|----|-----------|-------------|-----------|-----------|
| P0 | `frozen` | aucun | — | référence plancher (pas d'adaptation) |
| P1 | `always` | chaque échantillon (flag UART actuel) | vrai label hôte | référence plafond (coût max) |
| P2 | `gated_truelabel` | gate maha + fenêtre embarqué | vrai label **uniquement sur flag** | semi (active learning) |
| P3 | `gated_pseudolabel` | gate maha + fenêtre embarqué | **pseudo** (FAULT→faulty 1 ; DRIFT→`maha_update` ; NORMAL→rien) | **100 %** |

Le vrai label hôte (`g_recv_label`) reste **transmis** dans tous les cas, mais sert au scoring/parité
hors-ligne (F1, accuracy). En P3 il n'alimente jamais le SGD (autonomie réelle) ; en P2 seulement sur
les échantillons flaggés.

## Datasets

- **Monitoring (D2)** — domain-incremental par équipement (Pump → Turbine → Compressor) : teste le
  **drift inter-équipements** (changement de régime sain).
- **Pronostia (D4)** — class-incremental temporel (dégradation jusqu'à la panne) : teste l'**arrivée des
  premières données faulty** dans le temps.

## Liens triple gap

- **Gap 2 (latence)** : le gate ajoute un coût constant par échantillon (maha_score + fenêtre) mais
  économise les pas de SGD sur les échantillons NORMAL → latence moyenne mesurée vs `always`.
- **Gap 3 (RAM)** : coût RAM du gate (drift detector ~200 B + ring buffer) vs EWC seul.

## Découpage des tâches

| Tâche | Objet |
|-------|-------|
| S3800 | Overview & cadrage scénario (ce fichier) |
| S3801 | Config `configs/sprint38_autonomous_update.yaml` |
| S3802 | Référence PC `run_sprint38_pc.py` (4 politiques × 2 datasets) |
| S3803 | Firmware : `drift_detector.c/.h`, gate `pipeline.c` sous `-DEWC_AUTO_UPDATE`, export seuils, test Unity |
| S3804 | Board P0/P1 (`run_sprint38_board.py --policy frozen\|always`) |
| S3805 | Board P2/P3 autonomes + mesure d'économie |
| S3806 | Parité `board_pc_parity38.py` |
| S3807 | Agrégat `aggregate_sprint38.py` → `exp_S38_summary.json` (table d'économie) |
| S3808 | Notebook `notebooks/cl_eval/autonomous_ewc/comparison.ipynb` |
| S3809 | Tests Python + Unity + docs + roadmap + graphify |

## Précédents architecturaux réutilisés

- `SlidingWindowDriftDetector` (`src/evaluation/drift_detector.py`) — verdict NORMAL/DRIFT/FAULT.
- `RingBuffer` (`firmware/.../inc/ring_buffer.h`, Sprint 34) — fenêtre glissante zéro-malloc.
- `maha_score` / `maha_update` (`firmware/.../src/mahalanobis.c`) — distance + adaptation sans label.
- Sélection à la compilation `-DEWC_AUTO_UPDATE` (précédent `-DMAHA_INT8`, Sprint 29) car nibble UART saturé.
- Pattern de sprint apparié PC→board→parité→agrégat→notebook→tests (Sprint 36).
