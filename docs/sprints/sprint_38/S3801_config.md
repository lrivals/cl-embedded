# S3801 — Config appariée (source de vérité Sprint 38)

| Champ | Valeur |
|-------|--------|
| **Sprint** | 38 |
| **Priorité** | 🔴 Critique — source de vérité unique consommée par tous les scripts du sprint. |
| **Statut** | ✅ Implémentée — `configs/sprint38_autonomous_update.yaml` (4 politiques × 2 datasets × **2 init_modes**). |
| **Durée estimée** | 1h |
| **Dépendances** | `configs/board_ewc.yaml` ✅ (hyperparamètres EWC) · `src/evaluation/feature_conditions.py` ✅ (`resolve_feature_indices`, `load_condition_arrays`) · `src/evaluation/drift_detector.py` ✅ (`SlidingWindowDriftDetector`) |
| **Fichiers cibles** | `configs/sprint38_autonomous_update.yaml` |
| **Références** | S3601 (config appariée Sprint 36 comme modèle) · règle CLAUDE.md « aucun hyperparamètre en dur » |

---

## Contexte

Comme au Sprint 36, **une seule** config pilote toute l'étude : 4 politiques de mise à jour EWC ×
2 datasets × 2 plateformes (PC, NUCLEO-F439ZI). Elle est consommée à l'identique par
`run_sprint38_pc.py` (S3802), `run_sprint38_board.py` (S3804/S3805) et `aggregate_sprint38.py` (S3807).
Les indices de features ne sont **pas** listés : ils sont résolus par
`feature_conditions.resolve_feature_indices` (parité par construction, Sprint 35).

## Spec — sections de `configs/sprint38_autonomous_update.yaml`

1. **`policies`** — les 4 politiques (cœur du sprint), toutes sur le **même EWC** :
   - `frozen` (P0) : aucune MAJ — référence plancher.
   - `always` (P1) : MAJ chaque échantillon, vrai label — référence plafond (flag UART actuel).
   - `gated_truelabel` (P2) : MAJ déclenchée par le gate, vrai label **uniquement sur flag** (active learning).
   - `gated_pseudolabel` (P3) : MAJ déclenchée par le gate, **pseudo-label** par verdict (100 % autonome) —
     FAULT→faulty(1)+SGD ; DRIFT→`maha_update` (adapte le normal, pas de SGD faute) ; NORMAL→rien.

2. **`drift_detector`** — paramètres du gate (= défauts `SlidingWindowDriftDetector`) :
   `window_size=50`, `fault_multiplier=2.5`, `drift_multiplier=1.3`, `drift_ratio=0.6`.
   Seuils calibrés à l'exécution sur l'enrôlement healthy : `fault_threshold = P95 × 2.5`,
   `drift_threshold = P95 × 1.3`.

3. **`enrollment`** — scénario one-class : `healthy_only: true`, `n_samples: 500` (les N premiers
   échantillons sains calibrent **maha** ET les seuils du gate). Reflète la machine neuve = saine.

4. **`datasets`** — `[monitoring, pronostia]` : Monitoring teste le **drift inter-équipements**,
   Pronostia le **temporel / première faute**.

5. **`init_modes`** (décision utilisateur — étude des **deux** stratégies d'initialisation) :
   `pretrained` (base CL offline partagée, section `training`, miroir Sprint 36 → checkpoint flashé
   ⇒ parité frozen exacte ; `frozen` = plancher d'un **modèle déployé**, F1 déjà élevé) et `scratch`
   (pas d'entraînement offline ; le streaming **est** l'apprentissage ; `frozen` reste ~aléatoire =
   plancher absolu, `always` apprend en ligne = plafond). Chaque cellule porte le suffixe
   `_{init_mode}` → `exp_S38_*_{policy}_{ds}_{init_mode}`.

6. **`condition: 5feat`** — condition d'entrée fixe (board câblé 5-feat) ; résolue par `feature_conditions`.

7. **`uart`** — `rate_hz: 50`, `proto: 3`, `dump_samples: true` (prédictions + verdict par échantillon → parité S3806).

8. **`ewc_base_config: configs/board_ewc.yaml`** — hyperparamètres EWC hérités (jamais dupliqués).

9. **`training`** — `n_tasks: 3`, `epochs_per_task: 15`, `batch_size: 32`, `test_ratio: 0.2` —
   appariés board pour que le checkpoint PC == checkpoint flashé (parité exacte en frozen).

## Vérification

- La config se charge sans erreur (`yaml.safe_load`).
- Les 4 noms de politiques correspondent exactement à ceux attendus par `run_sprint38_pc.py` et
  `run_sprint38_board.py --policy`.
- `drift_detector.*` correspond aux noms d'arguments de `SlidingWindowDriftDetector.__init__`.
