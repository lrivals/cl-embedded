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

## Points ouverts — **résolus en S4110 (30 juillet 2026)**

- ~~Choisir : heatmap 4×5 complète au corps vs focus 3 datasets~~ → **arbitré** : tableau
  4 modèles × 3 datasets au corps (`\label{tab:gap1-grille}`), grille complète 4×5 en annexe.
- ~~Cohérence des versions de chiffres~~ → **revalidé** : S39/S40 ne touchent pas la grille S35
  (ils portent sur le noyau INT8, ch. 7). Chiffres relus dans les JSON.

## Chiffres consolidés au corps (S4110)

**Tableau 4×3 — F1 de la classe « fautif », condition `5feat`, PC | carte**
(source : `experiments/exp_S35_{PC,board}_5feat_{model}_{ds}/results.json`,
champs `f1_faulty`) :

| Modèle | Monitoring (D2) | CMAPSS (D5) | Pronostia (D4) |
|---|---|---|---|
| EWC | 0,893 \| 0,947 | 0,456 \| 0,381 | 0,930 \| 0,968 |
| HDC | 0,565 \| 0,000 | 0,000 \| 0,000 | 0,425 \| 0,000 |
| TinyOL | 0,754 \| 0,710 | 0,197 \| 0,214 | 0,285 \| 0,273 |
| Mahalanobis | 0,698 \| 0,710 | 0,269 \| 0,214 | 0,305 \| 0,273 |

- **Message ajouté** : HDC F1 = 0 sur carte pour une accuracy de 0,867–0,900 → il prédit la classe
  majoritaire. C'est une **seconde illustration**, indépendante de Maha×CMAPSS, du message
  « accuracy trompeuse → F1 ».
- **Oubli catastrophique complété** (`exp_S26_02/results.json`) : `f1_macro_pc_per_task_mean`
  0,981 (trompeur) vs `f1_macro_pc_final_all_tasks` **0,2402** et `avg_forgetting_f1_pc` **0,8475** ;
  carte `f1_macro_board_inference` **0,2431**, `f1_macro_board_online` **0,5072** → parité exacte,
  donc oubli et non bug de portage (`FIXME(gap1)` clos).
- **TODO Paderborn résolu** : EWC F1 = 0,800 PC **et** carte ; HDC 0,565 → 0,000 ;
  TinyOL 0,703 → 0,113 ; Mahalanobis 0,071 → 0,113.
- **Vérifié inchangé** : Maha×CMAPSS acc 0,7446 / F1 0,2694 (arrondis 0,745 / 0,269 corrects).
