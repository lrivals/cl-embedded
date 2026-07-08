# Fiche de cadrage — Ch. 5 Gap 1 : validation sur données industrielles (~4 p., cible md `05_gap1_validation.md`)

## Messages clés

1. Les 4 modèles tournent en CL sur les datasets focus, **PC et board réelle**, avec métriques CL
   complètes (acc_final, F1, AF/BWT).
2. **« L'accuracy est trompeuse → F1 »** : classes déséquilibrées en PdM (ex. Maha×cmapss
   acc 0.745 / F1 0.269, source S3512).
3. **L'oubli catastrophique est observé sur données réelles** (pas seulement dans la littérature) :
   cas EWC multiclasse Sprint 26 — F1 modèle final tous-tâches 0.240, avg_forgetting_f1 0.847,
   alors que la moyenne des F1 post-tâche (0.981) masquait le phénomène. Illustre la nécessité
   des métriques CL.
4. Comportements par scénario : domain-incremental (Monitoring, CMAPSS) vs class-incremental
   (Pronostia) ; EWC le plus robuste sur les scénarios difficiles.
5. RUL CMAPSS sur board : RMSE_RUL 21.23 (ratio 0.94 vs PC) — la régression fonctionne embarquée.

## Sources de chiffres (chemins vérifiés)

| Donnée | Source |
|---|---|
| RMSE_RUL board 21.23, latences 130/403 µs | `experiments/exp_S26_01/board_rul_results.json` (+ statut S26 CLAUDE.md) |
| Oubli catastrophique EWC MC (0.243/0.507, forgetting 0.847) | `scripts/diagnose_multiclass_parity.py` + exps S26 (`exp_S26_02`, `exp_S26_03`) |
| Grille 4 modèles × 5 datasets board | `experiments/exp_S35_board_5feat_*` (16 cellules) — corps : colonnes cmapss/pronostia/monitoring, grille complète en annexe |
| Benchmark PC de référence | `experiments/exp_S23_benchmark/results.json` |
| Chiffres d'analyse (F1 0.38→0.62 EWC×cmapss 5feat→all, etc.) | `docs/sprints/sprint_35/S3512_analysis_update.md` — n'utiliser au corps que les cas focus |

## Figures prévues (S4109)

- 1 heatmap F1 board (sous-ensemble focus ou 4×5 avec renvoi annexe) — régénérer depuis les JSON
  (helpers de `notebooks/.../generate_comparison_sprint23.py` / heatmap builders S3510,
  PNG existants `docs/figures/gap1_heatmap_*_5feat_board.png`).
- 1 figure oubli catastrophique (courbe F1 par tâche, S26).

## Refs bib

`Saxena2008`, `Nectoux2012`, `Hurtado2023`, `Kirkpatrick2017`. Métriques CL : `LopezPaz2017` (BWT).

## Glossaire touché

AF/FM, BWT, F1 (existant), AUROC (à créer), RUL (existant).

## Points ouverts

- Choisir : heatmap 4×5 complète au corps (force du Gap 1 : ampleur) vs focus 3 datasets
  (décision utilisateur = focus, grille complète en annexe) → au corps, tableau 4 modèles × 3 datasets.
- Cohérence des versions de chiffres : si S39/S40 régénèrent des cellules, S4110 revalide.
